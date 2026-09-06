"""Configuração central do projeto — caminhos, constantes e dtypes.

Único lugar onde `RANDOM_STATE` é definido: todo split, modelo e amostragem
importa daqui, para que a execução seja reprodutível de ponta a ponta.

Organização de `data/` (Cookiecutter Data Science):

    data/raw/         entrada imutável — os CSVs do INEP e a Gold do Databricks
    data/reference/   dimensões de apoio, pequenas e versionadas
    data/interim/     Parquet por ano, derivado de `raw/`
    data/processed/   dataset final de modelagem

Nada em `raw/` é editado a mão. `reference/` substituiu o antigo `external/`:
`dim_municipio.csv` é derivado *internamente* a partir da Gold, o oposto do que
"external" significa na convenção ("data from third party sources").
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DIR_DATA = Path(os.environ.get("TC_DATA_DIR", str(ROOT / "data"))).expanduser().resolve()
DIR_RAW = DIR_DATA / "raw"
DIR_INEP = DIR_RAW / "inep"
DIR_GOLD = DIR_RAW / "alfabetizacao_aluno"
DIR_REFERENCE = DIR_DATA / "reference"
DIR_INTERIM = DIR_DATA / "interim"
DIR_PROCESSED = DIR_DATA / "processed"

DIR_MODELS = ROOT / "models"
DIR_REPORTS = ROOT / "reports"
DIR_METRICS = DIR_REPORTS / "metrics"
DIR_IMAGES = ROOT / "images"

# Fontes brutas.
#
# O nome do arquivo da Gold carrega a tabela de origem (`alfabetizacao_aluno_features`,
# a `GOLD_ML_TABLE` de `scripts/etl/etl-gold.py`), o período e o timestamp da extração —
# que é o `_gold_processing_timestamp` gravado dentro do próprio CSV. O ETL roda em
# `mode("overwrite")` sem carga incremental, então duas execuções produzem conteúdos
# diferentes; sem o timestamp no nome não há como saber qual extração está em disco.
CSV_GOLD = DIR_GOLD / "alfabetizacao_aluno_features_2023-2024_20260829.csv"
CSV_ALUNO = DIR_INEP / "br_inep_avaliacao_alfabetizacao_aluno.csv"
CSV_META_MUNICIPIO = DIR_INEP / "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_municipio.csv"
CSV_META_UF = DIR_INEP / "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_uf.csv"
CSV_META_BRASIL = DIR_INEP / "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_brasil.csv"
CSV_AGREGADO_MUNICIPIO = DIR_INEP / "br_inep_avaliacao_alfabetizacao_municipio.csv"
CSV_AGREGADO_UF = DIR_INEP / "br_inep_avaliacao_alfabetizacao_uf.csv"

# Derivados (gerados por scripts/prepare_data.py e scripts/build_dim_municipio.py)
PARQUET_GOLD = DIR_INTERIM / "gold_{ano}.parquet"
PARQUET_ALUNO = DIR_INTERIM / "aluno_{ano}.parquet"
CSV_DIM_MUNICIPIO = DIR_REFERENCE / "dim_municipio.csv"
PARQUET_DATASET = DIR_PROCESSED / "dataset_2024.parquet"
# Tabela municipal populacional, ponderada por `peso_aluno` (B2). Separada do
# dataset de modelagem de propósito: é o número que o projeto **publica**
# (ranking de risco, gap de meta, projeção até 2030) e tem de reconciliar com o
# INEP; o dataset é insumo de um classificador em grão aluno.
PARQUET_AGREGADO_MUNICIPAL = DIR_PROCESSED / "agregados_municipais_{ano}.parquet"

# ---------------------------------------------------------------------------
# Constantes de modelagem
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

ANO_ALVO = 2024   # coorte de treino/teste
ANO_LAG = 2023    # fonte exclusiva de features históricas
ANOS = (ANO_LAG, ANO_ALVO)

TARGET_BRUTO = "alfabetizado"                 # 1 = alfabetizado
TARGET = "risco_nao_alfabetizacao"            # 1 - alfabetizado; classe de interesse
GRUPO_CV = "id_municipio"                     # chave de agrupamento em todo split

# `alfabetizado` é um corte determinístico da proficiência, medido nos dois anos:
# max(prof | classe 0) = 742,9998 (2023) e 742,9996 (2024); min(prof | classe 1) = 743,0000.
LIMIAR_ALFABETIZACAO = 743.0

# ---------------------------------------------------------------------------
# Códigos de `rede` nos agregados do INEP — a regra depende do propósito
# ---------------------------------------------------------------------------
# Gold e microdado : 2 = Estadual, 3 = Municipal, 4 = Privada
# agregados do INEP: 2 = Estadual, 3 = Municipal, 5 = Pública, 0 = Total
#
# Não existe uma escolha única boa: as duas linhas medem coisas diferentes e a
# certa depende do que se está fazendo.
#
#   RECONCILIAR (a taxa do agregado bate com a do microdado?) -> rede = 5.
#       É a mais fiel. MAE contra o microdado: 0,99pp em 2023 e 0,37pp em 2024
#       (0,05pp e 0,04pp se a média for ponderada por `peso_aluno`, ver COL_PESO).
#
#   CONSTRUIR LAG (qual era a taxa de 2023 deste município?) -> coalescência 5 -> 3.
#       676 municípios existem em 2024 e não têm microdado de 2023 (23,11% da
#       coorte). No agregado de 2023, `rede = 5` cobre 79 deles e `rede = 3`
#       cobre 625 — os 79 são subconjunto dos 625, e é `rede = 3` quem traz São
#       Paulo. Medido sobre a coorte de 2024:
#           só rede 5      -> resgata 12,33%, sobra um gap de 10,77%
#           coalescência   -> resgata 21,20%, sobra um gap de  1,91%
#       Ou seja, a coalescência recupera 8,86pp a mais da coorte (164 mil alunos)
#       e reduz o gap residual em 5,6x. A fidelidade de `rede = 3` é pior que a
#       de `rede = 5`, mas aceitável: MAE 1,73pp e corr 0,983 contra o microdado.
#
# Toda feature construída pela coalescência precisa carregar a coluna de
# proveniência que diz de qual fonte cada valor veio.
REDE_AGREGADO_RECONCILIACAO = 5          # "Pública" — a mais fiel ao microdado
REDE_AGREGADO_FALLBACK = 3               # "Municipal" — a de maior cobertura
REDE_AGREGADO_COALESCENCIA = (REDE_AGREGADO_RECONCILIACAO, REDE_AGREGADO_FALLBACK)

# Retrocompatibilidade: era a constante única do projeto, hoje é o caso de reconciliação.
REDE_AGREGADO_PUBLICA = REDE_AGREGADO_RECONCILIACAO

# Colunas de auditoria do ETL — descartadas na conversão para Parquet.
COLS_AUDITORIA = (
    "_gold_processing_date",
    "_gold_processing_timestamp",
    "_gold_source_catalog",
    "_pipeline_stage",
)

# ---------------------------------------------------------------------------
# `peso_aluno` — proibido como preditor, obrigatório como peso de agregação
# ---------------------------------------------------------------------------
# É o fator oficial de correção de não-resposta do INEP: reexpande os presentes
# para a população matriculada, município a município (soma dos pesos dividida
# pelos matriculados dá 1,0005 na mediana de 2023 e 1,0000 na de 2024). A taxa que
# o INEP publica é a média de `alfabetizado` ponderada por ele — com o peso, a
# reconciliação cai de 0,99pp para 0,05pp em 2023, e `media_portugues` de 1,05 para 0,07.
#
# Regra do projeto: todo número POPULACIONAL que sai do projeto (taxa municipal,
# taxa por UF, ranking de risco, gap de meta, projeção até 2030) é ponderado por
# `peso_aluno`. O classificador em grão aluno continua sem peso — ali ele seria
# vazamento, por ser calculado a partir da presença observada na própria coorte.
#
# Ressalva a declarar junto de qualquer número ponderado: o fator corrige a
# não-resposta assumindo que o ausente se parece com o presente do mesmo estrato
# (MAR). Se os ausentes forem sistematicamente piores — e a EDA sugere que sim —
# a ponderação padroniza o viés, não o elimina.
COL_PESO = "peso_aluno"

# Nunca podem entrar como preditor. Ver a matriz anti-leakage do PLANO_EXECUCAO.md.
COLS_PROIBIDAS = (
    "proficiencia",        # é o alvo reescrito (corte em 743,0)
    "peso_aluno",          # ver COL_PESO: proibido como preditor, exigido como peso
    "id_aluno",            # não é chave longitudinal e codifica a UF no prefixo
    "id_escola",
    "id_municipio",        # usado só como grupo de CV, jamais como feature
    "nome_municipio",
    "nome_uf",
    "serie",               # constante = 2
    "serie_nome",
    "rede_nome",
    "taxa_alfabetizacao",  # do ano corrente
    # Vazamento contemporâneo — medido, não suposto. As quatro abaixo são
    # registradas no instante da prova que produz o alvo, sobre a mesma coorte
    # que se quer predizer. Ver os vetores 13 e 14 da matriz anti-leakage.
    "preenchimento_caderno",
    "percentual_participacao_municipio",
    "percentual_participacao_uf",
    "percentual_participacao_brasil",
)
