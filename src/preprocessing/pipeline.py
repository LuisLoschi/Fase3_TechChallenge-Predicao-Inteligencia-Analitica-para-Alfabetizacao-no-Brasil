"""Contrato de features e o `ColumnTransformer` que as prepara.

Duas responsabilidades, e só duas: dizer **quais** colunas do dataset são
features, e devolver um `Pipeline` que recebe dado cru e devolve predição. O
pré-processamento é parte do modelo, não uma etapa anterior a ele — o objeto
serializado na Etapa 4 tem de aceitar a linha como ela sai de
`data/processed/dataset_2024.parquet`, sem que o consumidor precise lembrar de
imputar ou escalar nada.

Três decisões medidas nesta etapa e materializadas aqui:

1. **O bloco `esc_*` não entra no conjunto padrão.** `id_escola` é reatribuído a
   cada edição: dos 36.051 identificadores presentes nos dois anos, 2,40% caem no
   mesmo município. Dar a cada escola de 2024 a taxa de uma escola **sorteada da
   mesma UF** rende AUC 0,5601 ± 0,0006 em três seeds, *acima* dos 0,5581 do join
   real; sortear de qualquer UF derruba para 0,4998, o acaso exato. Todo o sinal
   aparente do bloco é sinal de UF.
   Ele fica disponível em `FEATURES_ESCOLA` para a ablação e o controle negativo
   da Etapa 5, não no conjunto de produção.
2. **`min_frequency` só no `caderno`.** O plano previa
   `min_frequency=0,01` em todas as categóricas por causa do `caderno = 43`, que
   tem 12 alunos e só existe em 2024. Aplicado à `sigla_uf`, esse limiar colapsa
   SE, TO, AP e AC — 4 UFs com 10 a 18 mil alunos cada — em um único nível
   "infrequent", justamente na variável de 49pp de amplitude. O limiar vale para
   o `caderno`; para o resto basta `handle_unknown="infrequent_if_exist"`.
3. **Os percentis de proficiência do microdado também ficam de fora.** As duas
   medições do experimento B5 concordam em que o ganho é compatível com zero, e é
   por isso que a decisão se mantém:

   - réplica histórica, `StratifiedGroupKFold(5)` e 3 seeds: **+0,00037** de
     ROC-AUC. Este número **não é evidência válida** — a réplica comparou os
     conjuntos de atributos sobre a coorte inteira de 2024, então os municípios
     depois chamados de reserva participaram da decisão. Fica como registro do
     que foi feito, não como medida de generalização;
   - execução corrigida, reserva separada **antes** da amostragem, 1 seed e 3
     folds só no desenvolvimento: **+0,00135**, com desvio de 0,00276 entre folds
     — ou seja, o ganho é o dobro do desvio abaixo de zero e também não se separa
     do acaso.

   Ver `scripts/experimento_b5.py`, `experimento_b5_replica.csv` (histórico) e
   `experimento_b5_desenvolvimento.csv` (corrente), e a constante
   `FEATURES_PERCENTIS_MICRODADO`.
4. **`FEATURES_PODADAS` para o baseline linear.** No conjunto completo a matriz é
   singular por construção (`iqr = p75 - p25`, `mun_desvio_vs_uf = mun - uf`). O
   subconjunto podado mantém uma métrica de nível por território e tem todo VIF
   abaixo de 3 — medido, não estimado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

# ---------------------------------------------------------------------------
# Contrato de colunas
# ---------------------------------------------------------------------------
# No dataset e fora de X: grupo de CV, peso de agregação populacional e o alvo.
COLS_OPERACIONAIS = (config.GRUPO_CV, config.COL_PESO, config.TARGET)

# Nível e contexto do município em 2023. É onde está o sinal do projeto:
# `mun_media_portugues_lag1` sozinha entrega AUC 0,641, e nenhuma feature de UF,
# de porte ou estrutural chega perto disso.
FEATURES_MUNICIPIO = [
    "mun_taxa_alfab_lag1",
    "mun_media_portugues_lag1",
    "mun_nivel_alfabetizacao_lag1",
    "mun_meta_2024_lag1",
    "mun_taxa_presenca_lag1",
    "mun_n_alunos_lag1",
    "mun_n_escolas_lag1",
    "mun_share_rede_estadual_lag1",
    "mun_desvio_vs_uf",
]

# Forma da distribuição de proficiência municipal de 2023 — **fora** do conjunto
# padrão, pela réplica do experimento B5 (`scripts/experimento_b5.py`).
#
# São as melhores features univariadas do projeto (`mun_prof_p75_lag1` sozinha dá
# AUC 0,658) e, ainda assim, no multivariado somam +0,00037 de ROC-AUC sobre 15
# folds, com IC95 pareado de [−0,00080; +0,00153] e t de Nadeau-Bengio 0,67
# (p = 0,51). O desvio entre folds é 0,0134 — 36 vezes o ganho. A explicação é a
# mesma que a auditoria já tinha dado: `mun_media_portugues_lag1`, que fica no
# conjunto e cobre 98,09% contra 76,88%, mede a mesma distribuição.
#
# Ficam construídas e gravadas no dataset. Se a Etapa 4 quiser reabrir com os
# hiperparâmetros do campeão, é uma linha: `FEATURES_MODELO + FEATURES_PERCENTIS_MICRODADO`.
FEATURES_PERCENTIS_MICRODADO = [
    "mun_prof_p25_lag1",
    "mun_prof_p50_lag1",
    "mun_prof_p75_lag1",
    "mun_prof_iqr_lag1",
    "mun_prof_sd_lag1",
]

FEATURES_UF = [
    "uf_taxa_alfab_lag1",
    "uf_media_portugues_lag1",
    "uf_taxa_presenca_lag1",
]

# Cobertura do lag. `tem_historico_municipio` marca SP, DF e AC (99,92% do gap):
# é quase um indicador de UF, e não nulidade aleatória (B3). `fonte_lag_*` diz de
# qual apuração veio o valor coalescido (B4) — sem ela, três fidelidades
# diferentes convivem na mesma coluna sem rastro.
FEATURES_COBERTURA = [
    "tem_historico_municipio",
    "tem_historico_escola",
    "fonte_lag_municipal",
    "fonte_lag_uf",
]

FEATURES_ESTRUTURAIS = ["rede_grupo", "nome_regiao", "sigla_uf", "caderno"]

# Fora do conjunto padrão — ver a decisão 1 no topo do módulo.
FEATURES_ESCOLA = [
    "esc_taxa_alfab_lag1",
    "esc_taxa_alfab_lag1_suav",
    "esc_taxa_presenca_lag1",
    "esc_n_alunos_lag1",
    "esc_prof_p25_lag1",
    "esc_prof_p50_lag1",
    "esc_prof_p75_lag1",
    "esc_prof_iqr_lag1",
    "esc_prof_sd_lag1",
]

# 20 colunas. Dois blocos construídos ficaram de fora, cada um por uma medição
# desta etapa: os percentis do microdado (réplica de B5) e o de escola (chave
# reciclada). Os dois seguem no dataset, disponíveis para ablação.
FEATURES_MODELO = FEATURES_MUNICIPIO + FEATURES_UF + FEATURES_COBERTURA + FEATURES_ESTRUTURAIS

# Subconjunto do baseline linear. Uma métrica de nível por território
# (`taxa_alfab`, nunca junto de `media_portugues` ou `meta`, que são a mesma
# coisa reescalada). `sigla_uf` sai porque `uf_taxa_alfab_lag1` já é o efeito de
# UF em uma coluna, e 26 dummies ao lado dela reintroduziriam a colinearidade que
# a poda existe para evitar; `mun_desvio_vs_uf` sai por ser combinação linear
# exata das outras duas. Sem métrica de dispersão porque a única disponível,
# `mun_prof_sd_lag1`, saiu junto com o bloco de percentis.
FEATURES_PODADAS = [
    "mun_taxa_alfab_lag1",
    "mun_taxa_presenca_lag1",
    "mun_n_alunos_lag1",
    "mun_share_rede_estadual_lag1",
    "uf_taxa_alfab_lag1",
    "tem_historico_municipio",
    "rede_grupo",
    "nome_regiao",
]

# Categóricas de baixa cardinalidade, todas com o nível mais raro acima de 1% —
# exceto `rede_grupo`, cujo nível "Outros" tem 24 alunos e já é a categoria de
# escape da rede Privada.
COLS_CATEGORICAS = ("rede_grupo", "nome_regiao", "sigla_uf", "fonte_lag_municipal", "fonte_lag_uf")
COLS_CATEGORICAS_RARAS = ("caderno",)


def separar_X_y(
    dados: pd.DataFrame, features: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Devolve `(X, y, grupos)` e falha se alguma coluna proibida entrar em X.

    É a única porta de entrada da matriz de modelagem. A asserção contra
    `config.COLS_PROIBIDAS` fica aqui, e não no notebook, porque é o notebook que
    erra: basta alguém escrever `dados.drop(columns=[TARGET])` uma vez para
    `peso_aluno` e `id_municipio` entrarem no modelo sem que ninguém veja.
    """
    features = list(FEATURES_MODELO if features is None else features)
    faltando = [c for c in features if c not in dados.columns]
    assert not faltando, f"features ausentes do dataset: {faltando}"

    proibidas = set(features) & set(config.COLS_PROIBIDAS)
    assert not proibidas, f"feature proibida em X: {sorted(proibidas)}"
    assert not set(features) & set(COLS_OPERACIONAIS), "coluna operacional em X"

    return dados[features], dados[config.TARGET], dados[config.GRUPO_CV]


def montar_pre_processador(
    features: list[str] | None = None, indicador_de_nulo: bool = True
) -> ColumnTransformer:
    """Imputação, escala e one-hot, ajustados **só** no fold de treino.

    `remainder="drop"` é deliberado: coluna nova na origem não entra no modelo
    por acidente. `add_indicator=True` porque a nulidade aqui é estrutural e
    informativa — 23,11% da coorte não tem histórico municipal, e isso são SP, DF
    e AC inteiros; o indicador é a diferença entre "imputei a mediana" e
    "imputei a mediana e avisei que imputei".

    `indicador_de_nulo=False` existe para o baseline linear. Medido: as dez
    colunas municipais que dependem do microdado somem **em bloco**, com o mesmo
    padrão em 100,00% das linhas, e esse padrão é o complemento exato de
    `tem_historico_municipio`. Com o indicador ligado, o modelo linear recebe
    quatro cópias idênticas da mesma coluna mais a flag, e a matriz volta a ser
    singular — que é justamente o que `FEATURES_PODADAS` existe para evitar.
    Árvore não se importa; regressão logística sim.
    """
    features = list(FEATURES_MODELO if features is None else features)
    categoricas = [c for c in features if c in COLS_CATEGORICAS]
    raras = [c for c in features if c in COLS_CATEGORICAS_RARAS]
    numericas = [c for c in features if c not in categoricas + raras]

    numerico = Pipeline(
        [
            ("imput", SimpleImputer(strategy="median", add_indicator=indicador_de_nulo)),
            ("escala", StandardScaler()),
        ]
    )
    categorico = Pipeline(
        [
            ("imput", SimpleImputer(strategy="constant", fill_value="AUSENTE")),
            ("enc", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False)),
        ]
    )
    # `caderno = 43` tem 12 alunos e não existe em 2023: sem `min_frequency` ele
    # vira uma coluna estimada sobre 12 observações, e sem
    # `handle_unknown="infrequent_if_exist"` um caderno novo em 2025 quebraria o
    # `transform` em produção.
    categorico_raro = Pipeline(
        [
            ("imput", SimpleImputer(strategy="most_frequent")),
            (
                "enc",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numerico, numericas),
            ("cat", categorico, categoricas),
            ("cat_rara", categorico_raro, raras),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def montar_pipeline(
    estimador: BaseEstimator, features: list[str] | None = None, indicador_de_nulo: bool = True
) -> Pipeline:
    """`Pipeline([pre, clf])` — o objeto que a Etapa 4 treina e serializa."""
    return Pipeline(
        [("pre", montar_pre_processador(features, indicador_de_nulo)), ("clf", estimador)]
    )


def nomes_das_features(pipeline: Pipeline) -> list[str]:
    """Nomes das colunas que de fato chegaram ao estimador.

    Sem isto o SHAP da Etapa 5 explicaria `x3` para o stakeholder. Também serve
    de auditoria: se o número de colunas mudar entre execuções, o
    `OneHotEncoder` viu categorias diferentes.
    """
    return list(pipeline[:-1].get_feature_names_out())


def medir_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF por inversão da matriz de correlação, sobre features já numéricas.

    Duplicado de `eda.medir_vif` de propósito: aqui ele roda sobre a saída do
    pré-processador, que é o que o modelo linear de fato recebe, e não sobre as
    colunas cruas.

    Espera a matriz **já imputada**. Quem chamar depois de um `dropna()` derruba
    em bloco os alunos sem histórico municipal — as dez colunas do bloco somem
    juntas — e `tem_historico_municipio` fica constante. A correlação de uma
    coluna sem variância é indefinida e o `pinv` estoura em `SVD did not
    converge`, erro que não diz nada sobre a causa. Daí a checagem explícita.
    """
    constantes = [c for c in X.columns if X[c].std(ddof=0) == 0]
    if constantes:
        raise ValueError(
            f"colunas sem variância, VIF indefinido: {constantes}. "
            "Impute em vez de remover linhas — ver a docstring."
        )
    correlacao = np.corrcoef(X.to_numpy(dtype=float), rowvar=False)
    vif = np.diag(np.linalg.pinv(correlacao))
    return pd.DataFrame({"feature": X.columns, "vif": vif}).sort_values("vif", ascending=False)
