"""ETL da camada Gold — consolida a Silver em uma tabela analítica única.

Lê as entidades da Silver (tabelas Delta em <catalog>.silver.*) e grava como
tabela Delta gerenciada em <catalog>.gold.<tabela>. Grão: ano x município x
série x rede. A meta comparada é sempre a do ano da linha (coluna
`meta_alfabetizacao_<ano>`); anos sem meta definida (ex.: 2023) recebem o
status 'Sem meta'.
"""

import logging
from itertools import chain
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import DoubleType

# ---------------------------------------------------------------------------
# Parâmetros (widgets do Databricks) e inicialização do Spark
# ---------------------------------------------------------------------------
dbutils.widgets.text("catalog", "workspace", "Catálogo (Unity Catalog)")
dbutils.widgets.text("silver_schema", "silver", "Schema de origem (Silver)")
dbutils.widgets.text("gold_schema", "gold", "Schema de destino (Gold)")

CATALOG = dbutils.widgets.get("catalog")
SILVER_SCHEMA = dbutils.widgets.get("silver_schema")
GOLD_SCHEMA = dbutils.widgets.get("gold_schema")

SILVER_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.%s"
GOLD_TABLE_TEMPLATE = f"{CATALOG}.{GOLD_SCHEMA}.%s"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()
# partitionOverwriteMode não é configurável em compute serverless/shared; não é
# necessário aqui, pois a Gold sempre sobrescreve o path inteiro.
try:
    # spark.sparkContext não existe em compute serverless (Spark Connect) — cosmético, ignora se falhar.
    spark.sparkContext.setLogLevel("WARN")
except Exception:
    pass


GOLD_TABLE = "alfabetizacao_analise_output"
GOLD_ML_TABLE = "alfabetizacao_aluno_features"
SILVER_ENTITIES = [
    "avaliacao_alfabetizacao_uf",
    "avaliacao_alfabetizacao_municipio",
    "avaliacao_alfabetizacao_meta_alfabetizacao_brasil",
    "avaliacao_alfabetizacao_meta_alfabetizacao_uf",
    "avaliacao_alfabetizacao_meta_alfabetizacao_municipio",
    "avaliacao_alfabetizacao_aluno",
]
REQUIRED_ENTITIES = [
    "avaliacao_alfabetizacao_municipio",
    "avaliacao_alfabetizacao_uf",
]

# Tolerância (em pontos percentuais) para o check de consistência entre a taxa
# recalculada a partir do microdado de aluno e a taxa publicada nas tabelas de
# avaliação; divergências acima disso são sinalizadas, não bloqueiam o job —
# as fontes usam critérios de apuração diferentes (ex.: peso amostral).
CONSISTENCY_TOLERANCE_PP = 5.0

# As tabelas de meta usam `rede` como texto curto ('Pública', 'Municipal'),
# diferente do código usado nas tabelas de avaliação/aluno ('0'..'6'). Sem essa
# tradução, o join de meta não pode ser feito por rede e a meta de uma rede
# vaza silenciosamente para as demais (bug corrigido nesta versão).
META_REDE_TO_CODIGO = {
    "Federal": "1",
    "Estadual": "2",
    "Municipal": "3",
    "Privada": "4",
    "Pública": "5",
    "Total": "0",
}


def read_silver(entity: str) -> DataFrame:
    table = SILVER_TABLE % entity

    log.info("=" * 70)
    log.info(f"Tabela Silver carregada: {entity}")
    log.info(f"Tabela: {table}")
    log.info("=" * 70)

    df = spark.table(table)

    log.info(f"Total lido -> {df.count()} registros | {len(df.columns)} colunas")
    log.info("=" * 70)

    return df


def load_silver_tables() -> dict:
    tables = {}
    for entity in SILVER_ENTITIES:
        table = SILVER_TABLE % entity
        if spark.catalog.tableExists(table):
            tables[entity] = read_silver(entity)
        else:
            log.warning(f"Tabela Silver ausente: {table} — essa tabela será ignorada.")

    log.info("=" * 70)
    log.info(f"Total de tabelas Silver carregadas: {len(tables)} / {len(SILVER_ENTITIES)}")
    log.info("=" * 70)

    return tables


def aggregate_student_metrics(aluno_df: DataFrame) -> DataFrame:
    """Agrega os indicadores de aluno por (ano, município, série, rede).

    Além dos códigos presentes no dado de aluno (2, 3, 4), deriva as redes
    compostas usadas pelas tabelas de avaliação: 5 = Pública (2+3) e
    0 = Total. Taxa observada e proficiência consideram apenas os presentes.
    """
    metrics = [
        F.count(F.lit(1)).alias("alunos_total"),
        F.sum("presenca").alias("alunos_presentes"),
        F.sum("alfabetizado").alias("alunos_alfabetizados"),
        F.avg("proficiencia").alias("_proficiencia_media"),
        F.sum(F.col("proficiencia") * F.col("peso_aluno")).alias("_proficiencia_x_peso"),
        F.sum(F.when(F.col("proficiencia").isNotNull(), F.col("peso_aluno"))).alias("_peso_presentes"),
    ]

    por_rede = aluno_df.groupBy("ano", "id_municipio", "serie", "rede").agg(*metrics)
    rede_publica = (
        aluno_df.filter(F.col("rede").isin("2", "3"))
        .groupBy("ano", "id_municipio", "serie").agg(*metrics)
        .withColumn("rede", F.lit("5"))
    )
    rede_total = (
        aluno_df.groupBy("ano", "id_municipio", "serie").agg(*metrics)
        .withColumn("rede", F.lit("0"))
    )
    aggregates = por_rede.unionByName(rede_publica).unionByName(rede_total)

    return (
        aggregates
        .withColumn(
            "proporcao_presenca",
            F.round(F.col("alunos_presentes") / F.col("alunos_total") * 100, 2),
        )
        .withColumn(
            "taxa_alfabetizacao_observada",
            F.when(
                F.col("alunos_presentes") > 0,
                F.round(F.col("alunos_alfabetizados") / F.col("alunos_presentes") * 100, 2),
            ),
        )
        .withColumn("proficiencia_media", F.round(F.col("_proficiencia_media"), 2))
        .withColumn(
            "proficiencia_media_ponderada",
            F.when(
                F.col("_peso_presentes") > 0,
                F.round(F.col("_proficiencia_x_peso") / F.col("_peso_presentes"), 2),
            ),
        )
        .drop("_proficiencia_media", "_proficiencia_x_peso", "_peso_presentes")
    )


def meta_do_ano(df: DataFrame) -> Column:
    """Retorna a meta anual do ano da linha (`meta_alfabetizacao_<ano>`); nulo se não houver."""
    meta_cols = sorted(
        c for c in df.columns
        if c.startswith("meta_alfabetizacao_") and c.rsplit("_", 1)[-1].isdigit()
    )
    expr = F.lit(None).cast(DoubleType())
    for col in meta_cols:
        year = int(col.rsplit("_", 1)[-1])
        expr = F.when(F.col("ano") == year, F.col(col)).otherwise(expr)
    return expr


def map_meta_rede_to_codigo(df: DataFrame, entity: str) -> DataFrame:
    """Traduz `rede` textual da meta ('Pública'/'Municipal') para o código usado
    na avaliação ('0'..'6'), para permitir o join por rede. Valores fora do
    mapa viram nulo (a meta não é aplicada) e são logados para investigação."""
    rede_map = F.create_map([F.lit(v) for v in chain(*META_REDE_TO_CODIGO.items())])
    mapped = rede_map[F.col("rede")]

    unmapped = (
        df.filter(F.col("rede").isNotNull() & mapped.isNull())
        .select("rede").distinct().collect()
    )
    if unmapped:
        valores = [r["rede"] for r in unmapped]
        log.warning(f"  [{entity}] Valor(es) de rede não mapeado(s) para código: {valores} "
                    f"— meta não será aplicada a essas linhas.")

    return df.withColumn("rede", mapped)


def add_goal_flag(df: DataFrame, flag_col: str, value_col: str, target_col: str) -> DataFrame:
    """Flag 'Atingida' | 'Abaixo' | 'Sem meta' | 'Sem dado'; não é criada se as colunas não existem."""
    if value_col not in df.columns or target_col not in df.columns:
        return df
    return df.withColumn(
        flag_col,
        F.when(F.col(target_col).isNull(), F.lit("Sem meta"))
         .when(F.col(value_col).isNull(), F.lit("Sem dado"))
         .when(F.col(value_col) >= F.col(target_col), F.lit("Atingida"))
         .otherwise(F.lit("Abaixo")),
    )


def unify_gold_table(tables: dict) -> DataFrame:
    avaliacao_municipio_df = tables["avaliacao_alfabetizacao_municipio"]
    avaliacao_uf_df        = tables["avaliacao_alfabetizacao_uf"]
    aluno_df               = tables.get("avaliacao_alfabetizacao_aluno")
    meta_municipio_df      = tables.get("avaliacao_alfabetizacao_meta_alfabetizacao_municipio")
    meta_uf_df             = tables.get("avaliacao_alfabetizacao_meta_alfabetizacao_uf")
    meta_brasil_df         = tables.get("avaliacao_alfabetizacao_meta_alfabetizacao_brasil")

    base = avaliacao_municipio_df.select(
        "ano",
        "id_municipio",
        "nome_municipio",
        "sigla_uf",
        "nome_uf",
        "nome_regiao",
        "serie",
        "serie_nome",
        "rede",
        "rede_nome",
        "taxa_alfabetizacao",
        "media_portugues",
        ).withColumnRenamed("taxa_alfabetizacao", "taxa_alfabetizacao_municipio"
        ).withColumnRenamed("media_portugues"   , "media_portugues_municipio"
    )

    if aluno_df is not None:
        base = base.join(
            aggregate_student_metrics(aluno_df),
            on=["ano", "id_municipio", "serie", "rede"],
            how="left",
        )
    else:
        log.warning("Silver de aluno ausente — métricas de aluno não serão calculadas.")

    uf_avaliacao = avaliacao_uf_df.select(
        "ano",
        "sigla_uf",
        "serie",
        "rede",
        "taxa_alfabetizacao",
        "media_portugues",
        ).withColumnRenamed("taxa_alfabetizacao", "taxa_alfabetizacao_uf"
        ).withColumnRenamed("media_portugues"   , "media_portugues_uf"
    )
    base = base.join(
        uf_avaliacao,
        on=["ano", "sigla_uf", "serie", "rede"],
        how="left",
    )

    # Visão Brasil: média simples das UFs (fontes não têm pesos por UF);
    # a taxa nacional oficial vem da tabela de meta do Brasil, mais abaixo.
    brasil_avaliacao = avaliacao_uf_df.groupBy("ano", "serie", "rede").agg(
        F.round(F.avg("taxa_alfabetizacao"), 2).alias("taxa_alfabetizacao_brasil"),
        F.round(F.avg("media_portugues"), 2).alias("media_portugues_brasil"),
    )
    base = base.join(
        brasil_avaliacao,
        on=["ano", "serie", "rede"],
        how="left",
    )

    if meta_municipio_df is not None:
        meta_municipio_df = map_meta_rede_to_codigo(meta_municipio_df, "meta_alfabetizacao_municipio")
        meta_municipio = meta_municipio_df.select(
            "ano",
            "id_municipio",
            "rede",
            meta_do_ano(meta_municipio_df).alias("meta_alfabetizacao_municipio"),
            F.col("percentual_participacao").alias("percentual_participacao_municipio"),
        )
        base = base.join(meta_municipio, on=["ano", "id_municipio", "rede"], how="left")

    if meta_uf_df is not None:
        meta_uf_df = map_meta_rede_to_codigo(meta_uf_df, "meta_alfabetizacao_uf")
        meta_uf = meta_uf_df.select(
            "ano",
            "sigla_uf",
            "rede",
            meta_do_ano(meta_uf_df).alias("meta_alfabetizacao_uf"),
            F.col("percentual_participacao").alias("percentual_participacao_uf"),
        )
        base = base.join(meta_uf, on=["ano", "sigla_uf", "rede"], how="left")

    if meta_brasil_df is not None:
        meta_brasil_df = map_meta_rede_to_codigo(meta_brasil_df, "meta_alfabetizacao_brasil")
        meta_brasil = meta_brasil_df.select(
            "ano",
            "rede",
            meta_do_ano(meta_brasil_df).alias("meta_alfabetizacao_brasil"),
            F.col("taxa_alfabetizacao").alias("taxa_alfabetizacao_brasil_oficial"),
            F.col("percentual_participacao").alias("percentual_participacao_brasil"),
        )
        base = base.join(meta_brasil, on=["ano", "rede"], how="left")

    base = add_goal_flag(base, "meta_atingida_municipio", "taxa_alfabetizacao_municipio", "meta_alfabetizacao_municipio")
    base = add_goal_flag(base, "meta_atingida_uf", "taxa_alfabetizacao_uf", "meta_alfabetizacao_uf")
    base = add_goal_flag(base, "meta_atingida_brasil", "taxa_alfabetizacao_brasil_oficial", "meta_alfabetizacao_brasil")

    base = add_goal_flag(base, "meta_atingida_presenca_municipio", "proporcao_presenca", "percentual_participacao_municipio")
    base = add_goal_flag(base, "meta_atingida_presenca_uf", "proporcao_presenca", "percentual_participacao_uf")
    base = add_goal_flag(base, "meta_atingida_presenca_brasil", "proporcao_presenca", "percentual_participacao_brasil")

    # Consistência entre tabelas: taxa recalculada a partir do microdado de
    # aluno (grão fino) vs. taxa publicada na tabela de avaliação de município
    # (grão agregado). Não bloqueia — apenas sinaliza divergência para auditoria.
    base = base.withColumn(
        "diferenca_taxa_observada_municipio_pp",
        F.abs(F.col("taxa_alfabetizacao_observada") - F.col("taxa_alfabetizacao_municipio")),
    ).withColumn(
        "consistencia_taxa_municipio",
        F.when(
            F.col("taxa_alfabetizacao_observada").isNull() | F.col("taxa_alfabetizacao_municipio").isNull(),
            F.lit("Sem dado"),
        )
        .when(F.col("diferenca_taxa_observada_municipio_pp") <= CONSISTENCY_TOLERANCE_PP, F.lit("Consistente"))
        .otherwise(F.lit("Divergente")),
    )

    divergentes = base.filter(F.col("consistencia_taxa_municipio") == "Divergente").count()
    avaliados = base.filter(F.col("consistencia_taxa_municipio").isin("Consistente", "Divergente")).count()
    if avaliados:
        pct = 100 * divergentes / avaliados
        level = log.warning if divergentes else log.info
        level(f"  [DQ:consistencia] taxa observada (aluno) vs. publicada (município): "
              f"{divergentes}/{avaliados} linhas divergentes ({pct:.1f}%, tolerância={CONSISTENCY_TOLERANCE_PP}pp)")

    selectable_columns = [
        "ano",
        "id_municipio",
        "nome_municipio",
        "sigla_uf",
        "nome_uf",
        "nome_regiao",
        "serie_nome",
        "rede",
        "rede_nome",
        "alunos_total",
        "alunos_presentes",
        "alunos_alfabetizados",
        "proporcao_presenca",
        "taxa_alfabetizacao_observada",
        "diferenca_taxa_observada_municipio_pp",
        "consistencia_taxa_municipio",
        "proficiencia_media",
        "proficiencia_media_ponderada",
        "taxa_alfabetizacao_municipio",
        "media_portugues_municipio",
        "meta_alfabetizacao_municipio",
        "percentual_participacao_municipio",
        "meta_atingida_municipio",
        "meta_atingida_presenca_municipio",
        "taxa_alfabetizacao_uf",
        "media_portugues_uf",
        "meta_alfabetizacao_uf",
        "percentual_participacao_uf",
        "meta_atingida_uf",
        "meta_atingida_presenca_uf",
        "taxa_alfabetizacao_brasil",
        "media_portugues_brasil",
        "taxa_alfabetizacao_brasil_oficial",
        "meta_alfabetizacao_brasil",
        "percentual_participacao_brasil",
        "meta_atingida_brasil",
        "meta_atingida_presenca_brasil",
    ]

    existing = [c for c in selectable_columns if c in base.columns]
    return base.select(*existing)


def build_ml_feature_table(tables: dict) -> DataFrame:
    """Tabela de grão aluno para o modelo supervisionado (target: `alfabetizado`).

    Filtra `presenca == 1`: aluno ausente tem `alfabetizado = 0` como valor
    default do INEP, não uma avaliação real (também não tem `proficiencia`) —
    incluí-lo ensinaria o modelo a prever ausência, não alfabetização.

    `proficiencia` e `peso_aluno` ficam de fora das features de propósito: a
    nota de proficiência é o que o INEP usa para derivar `alfabetizado`
    (vazamento de target quase direto).

    Gap conhecido: não há dados socioeconômicos nem populacionais/regionais em
    nenhuma fonte atual (avaliação/meta INEP + diretório IBGE de UF/município)
    — precisam de fonte adicional (ex.: IBGE Cidades, Censo) para entrar aqui.
    """
    aluno_df = tables.get("avaliacao_alfabetizacao_aluno")
    if aluno_df is None:
        raise RuntimeError("Silver de aluno ausente — tabela ML não pode ser construída.")

    meta_municipio_df = tables.get("avaliacao_alfabetizacao_meta_alfabetizacao_municipio")
    meta_uf_df = tables.get("avaliacao_alfabetizacao_meta_alfabetizacao_uf")
    meta_brasil_df = tables.get("avaliacao_alfabetizacao_meta_alfabetizacao_brasil")

    total_alunos = aluno_df.count()
    base = aluno_df.filter(F.col("presenca") == 1)
    presentes = base.count()
    log.warning(f"  [ML] Filtro presenca=1: {presentes}/{total_alunos} alunos mantidos "
                f"({total_alunos - presentes} ausente(s) descartado(s) do treino/teste).")
    log.warning("  [ML] Gap de dados: sem variáveis socioeconômicas/populacionais nas fontes "
                "atuais — requer fonte adicional (ex.: IBGE).")

    if meta_municipio_df is not None:
        # meta_municipio só tem rede 'Municipal' (código 3) — casa direto com o
        # código de rede do próprio aluno.
        meta_municipio_df = map_meta_rede_to_codigo(meta_municipio_df, "meta_alfabetizacao_municipio [ML]")
        meta_municipio = meta_municipio_df.select(
            "ano", "id_municipio", "rede",
            meta_do_ano(meta_municipio_df).alias("meta_alfabetizacao_municipio"),
            F.col("percentual_participacao").alias("percentual_participacao_municipio"),
        )
        base = base.join(meta_municipio, on=["ano", "id_municipio", "rede"], how="left")

    # meta_uf/meta_brasil só têm rede 'Pública' (código 5 = Estadual+Municipal
    # combinados) — não existe aluno individual com rede='5' (só aparece
    # agregado na tabela analítica). Um aluno de rede Estadual(2) ou
    # Municipal(3) está coberto pela meta Pública; Privada(4) fica sem meta,
    # corretamente (não há meta pública para rede privada).
    base = base.withColumn(
        "_rede_publica",
        F.when(F.col("rede").isin("2", "3"), F.lit("5")).otherwise(F.lit(None)),
    )

    if meta_uf_df is not None:
        meta_uf_df = map_meta_rede_to_codigo(meta_uf_df, "meta_alfabetizacao_uf [ML]")
        meta_uf = meta_uf_df.select(
            F.col("ano").alias("_meta_uf_ano"),
            F.col("sigla_uf").alias("_meta_uf_sigla_uf"),
            F.col("rede").alias("_meta_uf_rede"),
            meta_do_ano(meta_uf_df).alias("meta_alfabetizacao_uf"),
            F.col("percentual_participacao").alias("percentual_participacao_uf"),
        )
        base = base.join(
            meta_uf,
            on=(
                (F.col("ano") == F.col("_meta_uf_ano"))
                & (F.col("sigla_uf") == F.col("_meta_uf_sigla_uf"))
                & (F.col("_rede_publica") == F.col("_meta_uf_rede"))
            ),
            how="left",
        ).drop("_meta_uf_ano", "_meta_uf_sigla_uf", "_meta_uf_rede")

    if meta_brasil_df is not None:
        meta_brasil_df = map_meta_rede_to_codigo(meta_brasil_df, "meta_alfabetizacao_brasil [ML]")
        meta_brasil = meta_brasil_df.select(
            F.col("ano").alias("_meta_brasil_ano"),
            F.col("rede").alias("_meta_brasil_rede"),
            meta_do_ano(meta_brasil_df).alias("meta_alfabetizacao_brasil"),
            F.col("percentual_participacao").alias("percentual_participacao_brasil"),
        )
        base = base.join(
            meta_brasil,
            on=(
                (F.col("ano") == F.col("_meta_brasil_ano"))
                & (F.col("_rede_publica") == F.col("_meta_brasil_rede"))
            ),
            how="left",
        ).drop("_meta_brasil_ano", "_meta_brasil_rede")

    selectable_columns = [
        "ano",                                    # indicador temporal
        "id_aluno", "id_escola",                  # chaves (não são features)
        "id_municipio", "nome_municipio",
        "sigla_uf", "nome_uf", "nome_regiao",     # dados territoriais
        "serie", "serie_nome",
        "rede", "rede_nome",
        "caderno", "preenchimento_caderno",       # dados educacionais complementares
        "meta_alfabetizacao_municipio", "percentual_participacao_municipio",
        "meta_alfabetizacao_uf", "percentual_participacao_uf",
        "meta_alfabetizacao_brasil", "percentual_participacao_brasil",  # metas nacionais/estaduais/municipais
        "alfabetizado",                           # target
    ]
    existing = [c for c in selectable_columns if c in base.columns]
    return base.select(*existing)


def write_gold(df: DataFrame, entity: str) -> None:
    table = GOLD_TABLE_TEMPLATE % entity
    partition_cols = ["ano"] if "ano" in df.columns else []

    rows = df.count()
    cols = len(df.columns)

    log.info("=" * 70)
    log.info(f"Gravando Gold: {entity}")
    log.info(f"Destino: {table}")
    log.info(f"shape final Gold=({rows}, {cols})")
    log.info(f"partition={partition_cols or 'nenhuma'}")
    log.info("=" * 70)

    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.saveAsTable(table)

    log.info("=" * 70)
    log.info(f"  -> Gold gravado em {table}")
    log.info("=" * 70)


def main() -> None:
    log.info("=" * 70)
    log.info("Iniciando ETL da camada Gold — Métricas de Alfabetização")
    log.info(f"Origem: {CATALOG}.{SILVER_SCHEMA} | Destino: {CATALOG}.{GOLD_SCHEMA}")
    log.info("=" * 70)

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

    silver_tables = load_silver_tables()

    missing = [e for e in REQUIRED_ENTITIES if e not in silver_tables]
    if missing:
        raise RuntimeError(
            f"Tabelas Silver obrigatórias ausentes: {missing}. "
            "Execute o ETL Silver antes do ETL Gold."
        )

    now = datetime.now(timezone.utc)
    gold_df = unify_gold_table(silver_tables)
    gold_df = gold_df.withColumn("_gold_processing_date", F.lit(now.strftime("%Y-%m-%d")))
    gold_df = gold_df.withColumn("_gold_processing_timestamp", F.lit(now.strftime("%Y%m%d_%H%M%S")))
    gold_df = gold_df.withColumn("_gold_source_catalog", F.lit(CATALOG))
    gold_df = gold_df.withColumn("_pipeline_stage", F.lit("gold"))

    write_gold(gold_df, GOLD_TABLE)

    if "avaliacao_alfabetizacao_aluno" in silver_tables:
        ml_df = build_ml_feature_table(silver_tables)
        ml_df = ml_df.withColumn("_gold_processing_date", F.lit(now.strftime("%Y-%m-%d")))
        ml_df = ml_df.withColumn("_gold_processing_timestamp", F.lit(now.strftime("%Y%m%d_%H%M%S")))
        ml_df = ml_df.withColumn("_gold_source_catalog", F.lit(CATALOG))
        ml_df = ml_df.withColumn("_pipeline_stage", F.lit("gold"))
        write_gold(ml_df, GOLD_ML_TABLE)
    else:
        log.warning("Silver de aluno ausente — tabela ML (grão aluno) não será gerada.")

    log.info("=" * 70)
    log.info("ETL Gold concluído com sucesso.")
    log.info("=" * 70)

if __name__ == "__main__":
    main()
