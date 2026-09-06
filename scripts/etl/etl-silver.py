"""ETL da camada Silver — Avaliação de Alfabetização (INEP) — Databricks.

Consome as entidades da Bronze (tabelas Delta em <catalog>.bronze.*), aplica
padronização, tipagem, decodificação de domínios, deduplicação e
enriquecimento dimensional, e grava como tabela Delta gerenciada em
<catalog>.silver.<entidade>, particionada por `ano`. As dimensões de apoio
(UF e município) também são lidas da Bronze — não há bypass de camada. A
Silver não calcula indicadores nem agregações de negócio — isso pertence à
Gold.
"""

import logging
import unicodedata
from itertools import chain
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, DoubleType, StringType

# ---------------------------------------------------------------------------
# Parâmetros (widgets do Databricks) e inicialização do Spark
# ---------------------------------------------------------------------------
dbutils.widgets.text("catalog", "workspace", "Catálogo (Unity Catalog)")
dbutils.widgets.text("bronze_schema", "bronze", "Schema de origem (Bronze)")
dbutils.widgets.text("silver_schema", "silver", "Schema de destino (Silver)")
dbutils.widgets.text("dq_max_null_pct", "50", "DQ: % máx. de nulo em coluna crítica (bloqueia)")
dbutils.widgets.text("dq_max_orphan_pct", "20", "DQ: % máx. de órfão na FK de dimensão (bloqueia)")

CATALOG = dbutils.widgets.get("catalog")
BRONZE_SCHEMA = dbutils.widgets.get("bronze_schema")
SILVER_SCHEMA = dbutils.widgets.get("silver_schema")
DQ_MAX_NULL_PCT = float(dbutils.widgets.get("dq_max_null_pct"))
DQ_MAX_ORPHAN_PCT = float(dbutils.widgets.get("dq_max_orphan_pct"))

BRONZE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.%s"
SILVER_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.%s"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()
# partitionOverwriteMode não é configurável em compute serverless/shared; não é
# necessário aqui, pois cada entidade sempre sobrescreve o path inteiro (todos
# os anos são relidos da Bronze a cada execução — sem carga incremental).
try:
    # spark.sparkContext não existe em compute serverless (Spark Connect) — cosmético, ignora se falhar.
    spark.sparkContext.setLogLevel("WARN")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Contrato de transformação por entidade: tipos, chave de deduplicação (grão),
# dimensão de enriquecimento, colunas monitoradas e nº de arquivos na escrita.
# Colunas `proporcao_aluno_nivel_*` e `meta_alfabetizacao_*` viram double em
# qualquer entidade.
# ---------------------------------------------------------------------------
ENTITIES = {
    "avaliacao_alfabetizacao_uf": {
        "integer_cols": ["ano"],
        "double_cols": ["taxa_alfabetizacao", "media_portugues"],
        "string_cols": ["sigla_uf", "serie", "rede"],
        "dedup_keys": ["ano", "sigla_uf", "serie", "rede"],
        "enrich": "uf",
        "critical_cols": ["ano", "sigla_uf", "taxa_alfabetizacao", "media_portugues"],
        "coalesce": 1,
    },
    "avaliacao_alfabetizacao_municipio": {
        "integer_cols": ["ano"],
        "double_cols": ["taxa_alfabetizacao", "media_portugues"],
        "string_cols": ["id_municipio", "serie", "rede"],
        "dedup_keys": ["ano", "id_municipio", "serie", "rede"],
        "enrich": "municipio",
        "critical_cols": ["ano", "id_municipio", "taxa_alfabetizacao", "media_portugues"],
        "coalesce": 1,
    },
    "avaliacao_alfabetizacao_meta_alfabetizacao_brasil": {
        "integer_cols": ["ano"],
        "double_cols": ["taxa_alfabetizacao", "percentual_participacao"],
        "string_cols": ["rede"],
        "dedup_keys": ["ano", "rede"],
        "enrich": None,
        "critical_cols": ["ano", "taxa_alfabetizacao"],
        "coalesce": 1,
    },
    "avaliacao_alfabetizacao_meta_alfabetizacao_uf": {
        "integer_cols": ["ano"],
        "double_cols": ["taxa_alfabetizacao", "percentual_participacao"],
        "string_cols": ["sigla_uf", "rede"],
        "dedup_keys": ["ano", "sigla_uf", "rede"],
        "enrich": "uf",
        "critical_cols": ["ano", "sigla_uf", "taxa_alfabetizacao"],
        "coalesce": 1,
    },
    "avaliacao_alfabetizacao_meta_alfabetizacao_municipio": {
        "integer_cols": ["ano", "nivel_alfabetizacao"],
        "double_cols": ["taxa_alfabetizacao", "percentual_participacao"],
        "string_cols": ["id_municipio", "rede"],
        "dedup_keys": ["ano", "id_municipio", "rede"],
        "enrich": "municipio",
        "critical_cols": ["ano", "id_municipio", "taxa_alfabetizacao"],
        "coalesce": 1,
    },
    "avaliacao_alfabetizacao_aluno": {
        # Domínios dos campos: `serie` = '2'; `rede` ∈ {2,3,4}; `presenca` e
        # `alfabetizado` são indicadores 0/1 (inteiros, sem nulos); `proficiencia`
        # e `peso_aluno` são nulos apenas para alunos ausentes (`presenca` = 0).
        "integer_cols": ["ano", "alfabetizado", "presenca"],
        "double_cols": ["preenchimento_caderno", "proficiencia", "peso_aluno"],
        "string_cols": ["id_municipio", "id_escola", "id_aluno", "caderno", "serie", "rede"],
        "dedup_keys": ["ano", "id_aluno", "id_escola"],
        "enrich": "municipio",
        "critical_cols": ["ano", "id_aluno", "alfabetizado"],
        "coalesce": None,  # Tabela de alto volume: mantém o paralelismo de escrita do Spark.
    },
}


# ---------------------------------------------------------------------------
# Domínios do INEP: código de `rede`/`serie` -> rótulo. Nas tabelas de meta,
# `rede` já é texto ('Pública'/'Municipal') e é preservada como está —
# 'Pública' não mapeia de forma única para os códigos 5 ou 6.
# ---------------------------------------------------------------------------
REDE_MAP = {
    "0": "Total (Federal, Estadual, Municipal e Privada)",
    "1": "Federal",
    "2": "Estadual",
    "3": "Municipal",
    "4": "Privada",
    "5": "Pública (Estadual e Municipal)",
    "6": "Pública (Federal, Estadual e Municipal)",
}

SERIE_MAP = {
    "2": "2º ano do Ensino Fundamental",
}


# ---------------------------------------------------------------------------
# Funções de transformação (reutilizáveis entre entidades)
# ---------------------------------------------------------------------------
def bronze_path_exists(entity: str) -> bool:
    """Verifica se a entidade existe na Bronze (entidades não ingeridas são puladas)."""
    return spark.catalog.tableExists(BRONZE_TABLE % entity)


def read_bronze(entity: str) -> DataFrame:
    table = BRONZE_TABLE % entity
    log.info(f"Lendo Bronze: {table}")
    df = spark.table(table)
    log.info(f"  -> {df.count()} registros | {len(df.columns)} colunas")
    return df


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def standardize_column_names(df: DataFrame) -> DataFrame:
    """Padroniza nomes de colunas para snake_case ASCII, preservando o prefixo `_` técnico."""
    renamed = {}
    for col in df.columns:
        is_technical = col.startswith("_")
        clean = _strip_accents(col).strip().lower()
        clean = "".join(ch if ch.isalnum() else "_" for ch in clean)
        while "__" in clean:
            clean = clean.replace("__", "_")
        clean = clean.strip("_")
        if is_technical:
            clean = "_" + clean
        renamed[col] = clean
    for old, new in renamed.items():
        if old != new:
            df = df.withColumnRenamed(old, new)
    return df


def empty_strings_to_null(df: DataFrame) -> DataFrame:
    """Aplica trim nas colunas de texto e converte string vazia em NULL."""
    for field in df.schema.fields:
        if isinstance(field.dataType, StringType) and not field.name.startswith("_"):
            trimmed = F.trim(F.col(field.name))
            df = df.withColumn(
                field.name,
                F.when(trimmed == "", None).otherwise(trimmed),
            )
    return df


def cast_columns(df: DataFrame, config: dict) -> DataFrame:
    """Aplica o contrato de tipos; ids ficam como texto para preservar zeros à esquerda."""
    cols = set(df.columns)

    for col in config.get("integer_cols", []):
        if col in cols:
            df = df.withColumn(col, F.col(col).cast(IntegerType()))

    for col in config.get("double_cols", []):
        if col in cols:
            df = df.withColumn(col, F.col(col).cast(DoubleType()))

    for col in df.columns:
        if col.startswith("proporcao_aluno_nivel_") or col.startswith("meta_alfabetizacao_"):
            df = df.withColumn(col, F.col(col).cast(DoubleType()))

    for col in config.get("string_cols", []):
        if col in cols:
            df = df.withColumn(col, F.col(col).cast(StringType()))

    return df


def add_domain_labels(df: DataFrame) -> DataFrame:
    """Cria `rede_nome`/`serie_nome`; valores já textuais (metas) são mantidos pelo coalesce."""
    if "rede" in df.columns:
        rede_map = F.create_map([F.lit(v) for v in chain(*REDE_MAP.items())])
        df = df.withColumn("rede_nome", F.coalesce(rede_map[F.col("rede")], F.col("rede")))

    if "serie" in df.columns:
        serie_map = F.create_map([F.lit(v) for v in chain(*SERIE_MAP.items())])
        df = df.withColumn("serie_nome", F.coalesce(serie_map[F.col("serie")], F.col("serie")))

    return df


def deduplicate(df: DataFrame, keys: list, entity: str) -> DataFrame:
    """Remove duplicidades pela chave de negócio, logando o volume descartado."""
    keys = [k for k in keys if k in df.columns]
    if not keys:
        log.warning(f"  [{entity}] Sem chave de deduplicação válida — nenhuma remoção aplicada.")
        return df
    before = df.count()
    df = df.dropDuplicates(keys)
    after = df.count()
    removed = before - after
    if removed > 0:
        log.warning(f"  [{entity}] Deduplicação por {keys}: {removed} registro(s) removido(s) "
                    f"({before} -> {after}).")
    else:
        log.info(f"  [{entity}] Sem duplicidades na chave {keys}.")
    return df


def add_technical_columns(df: DataFrame, entity: str) -> DataFrame:
    """Adiciona colunas de auditoria da Silver (as herdadas da Bronze são preservadas)."""
    now = datetime.now(timezone.utc)
    return df \
        .withColumn("_silver_processing_date", F.lit(now.strftime("%Y-%m-%d"))) \
        .withColumn("_silver_processing_timestamp", F.lit(now.strftime("%Y%m%d_%H%M%S"))) \
        .withColumn("_silver_source_entity", F.lit(entity)) \
        .withColumn("_source_layer", F.lit("bronze")) \
        .withColumn("_pipeline_stage", F.lit("silver"))


def data_quality_report(df: DataFrame, entity: str, critical_cols: list) -> dict:
    """Loga volume, anos presentes e % de nulos nas colunas críticas; retorna os % por coluna."""
    total = df.count()
    log.info(f"  [DQ:{entity}] registros={total}")
    if "ano" in df.columns:
        anos = sorted(r["ano"] for r in df.select("ano").distinct().collect() if r["ano"] is not None)
        log.info(f"  [DQ:{entity}] anos={anos}")
    cols = [c for c in critical_cols if c in df.columns]
    null_pcts = {}
    if cols and total:
        nulls = df.select([F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in cols]).collect()[0]
        for c in cols:
            pct = 100 * nulls[c] / total
            null_pcts[c] = pct
            level = log.warning if pct > 0 else log.info
            level(f"  [DQ:{entity}] nulos em '{c}': {nulls[c]} ({pct:.1f}%)")
    return null_pcts


def enforce_critical_nulls(null_pcts: dict, entity: str, max_pct: float) -> None:
    """Interrompe o job se algum % de nulo em coluna crítica passar do limite configurado."""
    breaches = {c: round(p, 1) for c, p in null_pcts.items() if p >= max_pct}
    if breaches:
        raise RuntimeError(
            f"[{entity}] Coluna(s) crítica(s) com % de nulo acima do limite ({max_pct}%): {breaches}. "
            "Interrompendo antes da gravação da Silver."
        )


def check_referential_integrity(df: DataFrame, entity: str, kind: str, max_orphan_pct: float) -> None:
    """Após o enrich, verifica se a FK para a dimensão (UF/município) realmente casou.

    Um registro com chave preenchida (`id_municipio`/`sigla_uf`) mas sem a
    coluna trazida pela dimensão (`nome_municipio`/`nome_uf`) é órfão: a chave
    não existe no diretório oficial.
    """
    key_col, dim_col = {
        "municipio": ("id_municipio", "nome_municipio"),
        "uf": ("sigla_uf", "nome_uf"),
    }.get(kind, (None, None))
    if key_col is None or key_col not in df.columns or dim_col not in df.columns:
        return

    total = df.filter(F.col(key_col).isNotNull()).count()
    if total == 0:
        return
    orphans = df.filter(F.col(key_col).isNotNull() & F.col(dim_col).isNull()).count()
    pct = 100 * orphans / total
    level = log.warning if orphans else log.info
    level(f"  [DQ:{entity}] integridade referencial ({kind}): {orphans}/{total} órfão(s) ({pct:.1f}%)")

    if pct >= max_orphan_pct:
        raise RuntimeError(
            f"[{entity}] Integridade referencial quebrada: {pct:.1f}% dos registros com "
            f"'{key_col}' preenchido não encontraram correspondência na dimensão de {kind} "
            f"(limite={max_orphan_pct}%). Interrompendo antes da gravação da Silver."
        )


def write_silver(df: DataFrame, entity: str, coalesce_n) -> None:
    """Grava como tabela Delta gerenciada <catalog>.silver.<entidade>, particionada
    por `ano`; coalesce_n limita arquivos por partição."""
    table = SILVER_TABLE % entity
    partition = ["ano"] if "ano" in df.columns else []
    if coalesce_n:
        df = df.coalesce(coalesce_n)
    log.info(f"Gravando Silver: {table} (partition={partition or 'nenhuma'}, coalesce={coalesce_n})")
    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    if partition:
        writer = writer.partitionBy(*partition)
    writer.saveAsTable(table)
    log.info(f"  -> gravação concluída em {table}")


# ---------------------------------------------------------------------------
# Dimensões de apoio e enriquecimento
# ---------------------------------------------------------------------------
def load_dim_uf() -> DataFrame:
    """Dimensão de UF (uma linha por sigla), lida da Bronze (dim_uf)."""
    df = spark.table(BRONZE_TABLE % "dim_uf")
    return df.select(
        F.trim(F.col("sigla")).alias("sigla_uf"),
        F.trim(F.col("nome")).alias("nome_uf"),
        F.trim(F.col("regiao")).alias("nome_regiao"),
    ).dropDuplicates(["sigla_uf"])


def load_dim_municipio() -> DataFrame:
    """Dimensão de município (uma linha por código IBGE), lida da Bronze (dim_municipio)."""
    df = spark.table(BRONZE_TABLE % "dim_municipio")
    return df.select(
        F.trim(F.col("id_municipio")).cast(StringType()).alias("id_municipio"),
        F.trim(F.col("nome")).alias("nome_municipio"),
        F.trim(F.col("sigla_uf")).alias("sigla_uf"),
        F.trim(F.col("nome_uf")).alias("nome_uf"),
        F.trim(F.col("nome_regiao")).alias("nome_regiao"),
    ).dropDuplicates(["id_municipio"])


def enrich(df: DataFrame, kind: str, dims: dict) -> DataFrame:
    """Left join com a dimensão de UF ou município; apenas acrescenta colunas descritivas."""
    if kind == "uf" and "sigla_uf" in df.columns:
        return df.join(F.broadcast(dims["uf"]), on="sigla_uf", how="left")
    if kind == "municipio" and "id_municipio" in df.columns:
        return df.join(F.broadcast(dims["municipio"]), on="id_municipio", how="left")
    return df


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def process_entity(entity: str, config: dict, dims: dict) -> None:
    """Executa o fluxo Bronze -> Silver de uma entidade."""
    log.info("=" * 70)
    log.info(f"Processando entidade: {entity}")

    df = read_bronze(entity)
    df = standardize_column_names(df)
    df = empty_strings_to_null(df)
    df = cast_columns(df, config)
    df = add_domain_labels(df)
    df = deduplicate(df, config["dedup_keys"], entity)

    if config.get("enrich"):
        df = enrich(df, config["enrich"], dims)

    df = add_technical_columns(df, entity)
    # Sem .cache(): PERSIST não é suportado em compute serverless. df é
    # recalculado a cada ação (report, checks, escrita) — custo aceitável
    # dado o volume das entidades.

    null_pcts = data_quality_report(df, entity, config.get("critical_cols", []))
    enforce_critical_nulls(null_pcts, entity, DQ_MAX_NULL_PCT)
    if config.get("enrich"):
        check_referential_integrity(df, entity, config["enrich"], DQ_MAX_ORPHAN_PCT)

    write_silver(df, entity, config.get("coalesce"))


def main() -> None:
    """Carrega as dimensões e processa cada entidade; interrompe na primeira falha."""
    log.info("Iniciando ETL da camada Silver — Alfabetização INEP")
    log.info(f"Origem: {CATALOG}.{BRONZE_SCHEMA} | Destino: {CATALOG}.{SILVER_SCHEMA}")

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SILVER_SCHEMA}")

    dims = {
        "uf": load_dim_uf(),
        "municipio": load_dim_municipio(),
    }
    log.info(f"Dimensões carregadas: UF={dims['uf'].count()} | "
             f"Município={dims['municipio'].count()}")

    for entity, config in ENTITIES.items():
        if not bronze_path_exists(entity):
            log.warning(f"Entidade ausente na Bronze — pulando: {entity}")
            continue
        try:
            process_entity(entity, config, dims)
        except Exception as e:
            log.error(f"Falha ao processar {entity}: {e}")
            raise

    log.info("=" * 70)
    log.info("ETL Silver concluído com sucesso.")


main()
