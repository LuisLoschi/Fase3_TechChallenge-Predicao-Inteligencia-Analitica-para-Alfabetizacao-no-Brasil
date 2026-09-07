"""Treino, comparação, tuning e serialização do modelo de risco (Etapa 4).

Executa em estágios independentes, cada um gravando o seu artefato assim que
fecha. A razão é prática: a comparação leva 13 minutos em 16 núcleos e a busca de
hiperparâmetros passa de 45, e um estágio que só grava no fim é um estágio que se
perde inteiro quando a máquina dorme.

    python -m src.modeling.train comparacao   -> reports/metrics/cv_modelos.csv
    python -m src.modeling.train tuning       -> reports/metrics/tuning_lgbm.csv
    python -m src.modeling.train campeao      -> models/campeao.joblib + campeao.json

Desenho de validação, fixado antes de qualquer treino:

    1.851.852 alunos de 2024
      |-- desenvolvimento 82,1%  (4.413 municípios) -> StratifiedGroupKFold(5)
      +-- teste           17,9%  (1.104 municípios) -> tocado uma vez, no fim

A partição é de municípios, não de alunos, então 20% dos municípios viram 17,9%
das linhas — e as prevalências ficam em 0,4067 no desenvolvimento contra 0,3812
no teste. Essa diferença de 2,5pp não é defeito do split: é a heterogeneidade
territorial que o desenho existe para expor, e é por isso que toda métrica de
teste é comparada contra a prevalência do próprio teste, nunca contra a global.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.data import loader
from src.evaluation import metrics as mt
from src.modeling import baselines, split
from src.preprocessing import pipeline as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

CSV_COMPARACAO = config.DIR_METRICS / "cv_modelos.csv"

# Hiperparâmetros de partida, iguais aos da réplica do experimento B5 — a
# comparação entre famílias tem de ser sobre configurações razoáveis e não
# ajustadas, senão ela mede esforço de tuning e não capacidade do modelo.
LGBM_PADRAO = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)

# `min_samples_leaf` alto e profundidade limitada não são economia de tempo: com
# feature municipal e 1,2 milhão de linhas, uma floresta sem freio decora o nível
# de cada município do treino em folhas de dez alunos. O freio é a mesma defesa
# que `min_child_samples` faz no boosting.
RF_PADRAO = dict(
    n_estimators=200,
    max_depth=16,
    min_samples_leaf=500,
    max_features="sqrt",
    n_jobs=-1,
)


def _sem_pipeline(estimador):
    """Marca os estimadores que consomem a coluna crua, sem `ColumnTransformer`."""
    return isinstance(estimador, baselines.BaselineUnivariado)


CANDIDATOS: dict[str, dict[str, Any]] = {
    "dummy_prior": {
        "fabrica": lambda seed: DummyClassifier(strategy="prior"),
        "features": None,
        "papel": "piso absoluto",
    },
    "heuristica_taxa_municipal": {
        "fabrica": lambda seed: baselines.baseline_taxa_municipal(),
        "features": None,
        "papel": "regra de uma variável — a barra real",
    },
    "heuristica_proficiencia_municipal": {
        "fabrica": lambda seed: baselines.baseline_proficiencia_municipal(),
        "features": None,
        "papel": "regra de uma variável, variante contínua",
    },
    "logistica_podada": {
        "fabrica": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        "features": pl.FEATURES_PODADAS,
        "indicador_de_nulo": False,
        "papel": "linear interpretável, subconjunto de VIF < 3",
    },
    "random_forest": {
        "fabrica": lambda seed: RandomForestClassifier(random_state=seed, **RF_PADRAO),
        "features": None,
        "papel": "não-linearidade e interações, sem boosting",
    },
    "lightgbm_padrao": {
        "fabrica": lambda seed: LGBMClassifier(random_state=seed, **LGBM_PADRAO),
        "features": None,
        "papel": "candidato a campeão, hiperparâmetros de partida",
    },
    # A busca roda em 400 mil linhas e 1.277 municípios, onde o desvio entre
    # folds é quase três vezes o da base completa. Um ganho medido lá precisa ser
    # remedido aqui, nos mesmos folds dos demais candidatos, antes de ser
    # chamado de ganho.
    "lightgbm_tunado": {
        "fabrica": lambda seed: LGBMClassifier(random_state=seed, **_hiperparametros_tunados()),
        "features": None,
        "papel": "campeão com hiperparâmetros da busca",
    },
}


def _hiperparametros_tunados() -> dict:
    from src.modeling import tuning

    return tuning.carregar_hiperparametros()


def construir(nome: str, seed: int = config.RANDOM_STATE):
    """Devolve `(modelo, features)` — o modelo já embrulhado no pipeline quando cabe."""
    spec = CANDIDATOS[nome]
    estimador = spec["fabrica"](seed)
    features = list(spec["features"] or pl.FEATURES_MODELO)
    if _sem_pipeline(estimador):
        return estimador, features
    return (
        pl.montar_pipeline(estimador, features, spec.get("indicador_de_nulo", True)),
        features,
    )


def escorar(modelo, X: pd.DataFrame) -> np.ndarray:
    return modelo.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Estágio 1 — comparação em CV agrupada, todos nos mesmos folds
# ---------------------------------------------------------------------------
def avaliar_em_cv(nome: str, dados: pd.DataFrame, particoes, seed=config.RANDOM_STATE) -> pd.DataFrame:
    """Uma linha por fold, com o tempo de ajuste junto — custo é critério de escolha."""
    linhas = []
    for fold, (treino, validacao) in enumerate(particoes):
        modelo, features = construir(nome, seed)
        X, y, _ = pl.separar_X_y(dados, features)
        inicio = time.time()
        modelo.fit(X.iloc[treino], y.iloc[treino])
        escore = escorar(modelo, X.iloc[validacao])
        linhas.append(
            {
                "modelo": nome,
                "papel": CANDIDATOS[nome]["papel"],
                "fold": fold,
                "n_features": len(features),
                **mt.metricas_de_ranking(y.iloc[validacao], escore),
                "segundos": round(time.time() - inicio, 1),
            }
        )
        log.info(
            "%s fold %d: ROC-AUC %.4f | PR-AUC %.4f | Brier %.4f | %.0fs",
            nome, fold, linhas[-1]["roc_auc"], linhas[-1]["pr_auc"],
            linhas[-1]["brier"], linhas[-1]["segundos"],
        )
    return pd.DataFrame(linhas)


def etapa_comparacao(modelos: list[str] | None = None, seed: int = config.RANDOM_STATE) -> None:
    dados = loader.carregar_dataset_modelagem()
    _, y, grupos = pl.separar_X_y(dados)
    dev, _ = split.separar_desenvolvimento_e_teste(grupos, seed=seed)
    desenvolvimento = dados.iloc[dev].reset_index(drop=True)
    _, y_dev, g_dev = pl.separar_X_y(desenvolvimento)
    particoes = split.folds(y_dev, g_dev, seed=seed)
    log.info(
        "desenvolvimento: %d alunos, %d municípios, prevalência %.4f",
        len(desenvolvimento), g_dev.nunique(), y_dev.mean(),
    )

    config.DIR_METRICS.mkdir(parents=True, exist_ok=True)
    acumulado = pd.read_csv(CSV_COMPARACAO) if CSV_COMPARACAO.exists() else pd.DataFrame()
    for nome in modelos or list(CANDIDATOS):
        resultado = avaliar_em_cv(nome, desenvolvimento, particoes, seed)
        acumulado = pd.concat(
            [acumulado[acumulado.get("modelo", pd.Series(dtype=str)) != nome], resultado],
            ignore_index=True,
        )
        acumulado.to_csv(CSV_COMPARACAO, index=False)
        log.info("gravado %s (%d linhas)", CSV_COMPARACAO, len(acumulado))


CSV_PAREADO = config.DIR_METRICS / "comparacao_pareada.csv"

# Cada par responde a uma pergunta, e a ordem é a da escada: piso, regra de uma
# variável, linear, boosting, tuning, família. O último par é o que o relatório
# publica — a distância entre a regra de uma variável e o campeão.
PARES = (
    ("dummy_prior", "heuristica_taxa_municipal"),
    ("heuristica_taxa_municipal", "heuristica_proficiencia_municipal"),
    ("heuristica_taxa_municipal", "logistica_podada"),
    ("logistica_podada", "lightgbm_padrao"),
    ("lightgbm_padrao", "lightgbm_tunado"),
    ("lightgbm_tunado", "random_forest"),
    ("heuristica_taxa_municipal", "lightgbm_tunado"),
)


def comparacoes_pareadas(caminho=CSV_COMPARACAO) -> pd.DataFrame:
    """Delta por fold entre os pares, com o t corrigido de Nadeau-Bengio.

    Pares cujos dois modelos ainda não rodaram são pulados em silêncio: o
    estágio de comparação roda em pedaços, e exigir a tabela completa impediria
    de ver o resultado parcial.
    """
    from src.evaluation import comparacao as cp

    cv = pd.read_csv(caminho)
    presentes = set(cv["modelo"])
    linhas = [cp.comparar_modelos_em_cv(cv, a, b) for a, b in PARES if {a, b} <= presentes]
    tabela = pd.DataFrame(linhas)
    if not tabela.empty:
        tabela.to_csv(CSV_PAREADO, index=False)
    return tabela


def resumo_da_comparacao(caminho=CSV_COMPARACAO) -> pd.DataFrame:
    """Média e desvio entre folds. O desvio é a régua: ganho menor que ele não é ganho."""
    bruto = pd.read_csv(caminho)
    resumo = (
        bruto.groupby(["modelo", "papel"])
        .agg(
            roc_auc=("roc_auc", "mean"),
            roc_auc_dp=("roc_auc", "std"),
            pr_auc=("pr_auc", "mean"),
            pr_auc_dp=("pr_auc", "std"),
            ks=("ks", "mean"),
            brier=("brier", "mean"),
            segundos=("segundos", "mean"),
        )
        .reset_index()
        .sort_values("roc_auc", ascending=False)
    )
    return resumo


def main() -> None:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("estagio", choices=["comparacao", "tuning", "campeao"])
    analisador.add_argument("--modelos", nargs="*", default=None)
    argumentos = analisador.parse_args()

    if argumentos.estagio == "comparacao":
        etapa_comparacao(argumentos.modelos)
        print(resumo_da_comparacao().to_string(index=False))
        print()
        print(comparacoes_pareadas().round(5).to_string(index=False))
    elif argumentos.estagio == "tuning":
        from src.modeling import tuning

        tuning.etapa_tuning()
    else:
        from src.modeling import campeao

        campeao.etapa_campeao()


if __name__ == "__main__":
    main()
