"""ETL da camada Bronze — Avaliação de Alfabetização (INEP) — Databricks.

Lê os arquivos-fonte brutos (CSV, já em snake_case) direto da raiz do Volume
do Unity Catalog e padroniza apenas nomes de coluna (ASCII). Não há tipagem,
deduplicação ou enriquecimento aqui — isso é responsabilidade da Silver.
Grava como tabela Delta gerenciada em <catalog>.bronze.<entidade>,
particionada por `ano` quando a coluna existir. Os CSVs brutos ficam no
Volume (landing); só a saída de cada camada vira tabela em schema próprio.
"""

import os
import logging
import unicodedata
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

# ---------------------------------------------------------------------------
# Parâmetros (widgets do Databricks) e inicialização do Spark
# ---------------------------------------------------------------------------
dbutils.widgets.text("catalog", "workspace", "Catálogo (Unity Catalog)")
dbutils.widgets.text("landing_schema", "raw", "Schema do Volume de landing")
dbutils.widgets.text("volume", "tech_challenge_fase_3_datalake", "Volume (Unity Catalog)")
dbutils.widgets.text("bronze_schema", "bronze", "Schema de destino da Bronze")

CATALOG = dbutils.widgets.get("catalog")
LANDING_SCHEMA = dbutils.widgets.get("landing_schema")
VOLUME = dbutils.widgets.get("volume")
BRONZE_SCHEMA = dbutils.widgets.get("bronze_schema")

# Landing: raiz do Volume, onde os CSVs brutos foram carregados (sem subpastas).
BASE_PATH = f"/Volumes/{CATALOG}/{LANDING_SCHEMA}/{VOLUME}"
LANDING_PATH = f"{BASE_PATH}/%s"
# Saída da Bronze: tabela Delta gerenciada, não path — <catalog>.bronze.<entidade>.
BRONZE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.%s"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()
# partitionOverwriteMode não é configurável em compute serverless/shared; não é
# necessário aqui, pois cada entidade sempre sobrescreve o path inteiro.
try:
    # spark.sparkContext não existe em compute serverless (Spark Connect) — cosmético, ignora se falhar.
    spark.sparkContext.setLogLevel("WARN")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Contrato de ingestão por entidade: arquivo de origem (relativo à landing),
# delimitador e nº de arquivos na escrita. Todas as fontes já chegam em
# snake_case e delimitador ',' — a Bronze só padroniza nomes de coluna. As
# dimensões de apoio (UF/município) também são ingeridas aqui como entidades
# da Bronze — a Silver deixa de ler CSV bruto diretamente e passa a consumir
# a Bronze, sem bypass de camada.
# ---------------------------------------------------------------------------
BRONZE_SOURCES = {
    "avaliacao_alfabetizacao_uf": {
        "file": "br_inep_avaliacao_alfabetizacao_uf.csv",
        "delimiter": ",",
        "coalesce": 1,
    },
    "avaliacao_alfabetizacao_municipio": {
        "file": "br_inep_avaliacao_alfabetizacao_municipio.csv",
        "delimiter": ",",
        "coalesce": 1,
    },
    "avaliacao_alfabetizacao_meta_alfabetizacao_brasil": {
        "file": "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_brasil.csv",
        "delimiter": ",",
        "coalesce": 1,
    },
    "avaliacao_alfabetizacao_meta_alfabetizacao_uf": {
        "file": "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_uf.csv",
        "delimiter": ",",
        "coalesce": 1,
    },
    "avaliacao_alfabetizacao_meta_alfabetizacao_municipio": {
        "file": "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_municipio.csv",
        "delimiter": ",",
        "coalesce": 1,
    },
    "avaliacao_alfabetizacao_aluno": {
        # Arquivo único (2023+2024 já combinados), colunas já no contrato
        # semântico (ano, id_municipio, id_escola, id_aluno, caderno, serie,
        # rede, presenca, preenchimento_caderno, alfabetizado, proficiencia,
        # peso_aluno) — não precisa de renomeio.
        "file": "br_inep_avaliacao_alfabetizacao_aluno.csv",
        "delimiter": ",",
        "coalesce": None,  # alto volume: preserva paralelismo de escrita do Spark
    },
    "dim_uf": {
        "file": "br_bd_diretorios_brasil_uf.csv",
        "delimiter": ",",
        "coalesce": 1,
    },
    "dim_municipio": {
        "file": "br_bd_diretorios_brasil_municipio.csv",
        "delimiter": ",",
        "coalesce": 1,
    },
}


# ---------------------------------------------------------------------------
# Funções de transformação
# ---------------------------------------------------------------------------
def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def standardize_column_names(df: DataFrame) -> DataFrame:
    """Padroniza nomes de colunas para snake_case ASCII."""
    renamed = {}
    for col in df.columns:
        clean = _strip_accents(col).strip().lower()
        clean = "".join(ch if ch.isalnum() else "_" for ch in clean)
        while "__" in clean:
            clean = clean.replace("__", "_")
        clean = clean.strip("_")
        renamed[col] = clean
    for old, new in renamed.items():
        if old != new:
            df = df.withColumnRenamed(old, new)
    return df


def landing_file_exists(rel_path: str) -> bool:
    """Volumes do Unity Catalog são montados como filesystem normal (FUSE) em
    /Volumes/..., inclusive em compute serverless — dá pra checar com os.path,
    sem precisar de acesso à JVM (spark._jvm/_jsc não existem em serverless)."""
    return os.path.exists(LANDING_PATH % rel_path)


def read_landing(entity: str, config: dict) -> DataFrame:
    """Lê o arquivo bruto de uma entidade; tudo como string (fidelidade da Bronze)."""
    path = LANDING_PATH % config["file"]
    log.info(f"Lendo landing [{entity}]: {path}")
    df = (
        spark.read
        .option("header", "true")
        .option("delimiter", config["delimiter"])
        .option("inferSchema", "false")
        .csv(path)
        .withColumn("_bronze_source_file", F.col("_metadata.file_path"))
    )

    log.info(f"  -> {df.count()} registros | {len(df.columns)} colunas")
    return df


def add_technical_columns(df: DataFrame, entity: str) -> DataFrame:
    """Adiciona colunas de auditoria da Bronze."""
    now = datetime.now(timezone.utc)
    return df \
        .withColumn("_bronze_ingestion_date", F.lit(now.strftime("%Y-%m-%d"))) \
        .withColumn("_bronze_ingestion_timestamp", F.lit(now.strftime("%Y%m%d_%H%M%S"))) \
        .withColumn("_bronze_source_entity", F.lit(entity)) \
        .withColumn("_source_layer", F.lit(VOLUME)) \
        .withColumn("_pipeline_stage", F.lit("bronze"))


def write_bronze(df: DataFrame, entity: str, coalesce_n) -> None:
    """Grava como tabela Delta gerenciada <catalog>.bronze.<entidade>, particionada
    por `ano` quando a coluna existir. overwriteSchema=true porque cada entidade
    é sempre relida por inteiro da landing (sem carga incremental)."""
    table = BRONZE_TABLE % entity
    partition = ["ano"] if "ano" in df.columns else []
    if coalesce_n:
        df = df.coalesce(coalesce_n)
    log.info(f"Gravando Bronze: {table} (partition={partition or 'nenhuma'}, coalesce={coalesce_n})")
    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    if partition:
        writer = writer.partitionBy(*partition)
    writer.saveAsTable(table)
    log.info(f"  -> gravação concluída em {table}")


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def process_entity(entity: str, config: dict) -> None:
    log.info("=" * 70)
    log.info(f"Processando entidade: {entity}")

    df = read_landing(entity, config)
    df = standardize_column_names(df)
    df = add_technical_columns(df, entity)

    write_bronze(df, entity, config.get("coalesce"))


def main() -> None:
    log.info("Iniciando ETL da camada Bronze — Alfabetização INEP")
    log.info(f"Landing: {BASE_PATH}")
    log.info(f"Destino: {CATALOG}.{BRONZE_SCHEMA}")

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{BRONZE_SCHEMA}")

    for entity, config in BRONZE_SOURCES.items():
        if not landing_file_exists(config["file"]):
            log.warning(f"Arquivo ausente na landing — pulando entidade: {entity} "
                        f"({config['file']})")
            continue
        try:
            process_entity(entity, config)
        except Exception as e:
            log.error(f"Falha ao processar {entity}: {e}")
            raise

    log.info("=" * 70)
    log.info("ETL Bronze concluído com sucesso.")


main()
