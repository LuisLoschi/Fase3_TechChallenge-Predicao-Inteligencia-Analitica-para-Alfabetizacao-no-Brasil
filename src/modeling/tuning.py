"""Busca de hiperparâmetros do LightGBM, com orçamento declarado.

40 configurações sorteadas do mesmo espaço que `RandomizedSearchCV` percorreria,
avaliadas em `StratifiedGroupKFold(5)` sobre uma subamostra de **municípios
inteiros** do conjunto de desenvolvimento. Sortear alunos individuais colocaria
metade de um município no treino e metade na validação, que é exatamente a
memorização territorial que o desenho existe para impedir (vetor 11 da matriz
anti-leakage).

O laço é escrito à mão em vez de delegado ao `RandomizedSearchCV` por um motivo
só: `ParameterSampler` com a mesma seed sorteia as mesmas 40 configurações, mas
aqui cada uma grava o seu resultado em disco assim que termina. Uma busca de uma
hora e meia que só grava no fim é uma busca que se perde inteira.

Duas escolhas sobre o espaço de busca merecem registro:

- `n_estimators` entra na busca, embora o plano só listasse os sete parâmetros de
  forma. Com `learning_rate` varrendo uma década inteira (0,01 a 0,2), fixar o
  número de árvores confundiria taxa de aprendizado com capacidade: a mesma
  configuração de forma daria resultados opostos em 200 e em 800 árvores.
- `min_child_samples` vai até 2.000 porque é o principal freio disponível. A
  granularidade real do sinal é o município, e há municípios com 8 alunos na
  base — uma folha pequena decora território em vez de aprender a relação.

O ganho da busca é reavaliado depois, nos mesmos folds da comparação e sobre o
desenvolvimento completo. Ganho medido em 400 mil linhas que não sobrevive a 1,5
milhão não é ganho, é ruído de subamostra.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterSampler

from src import config
from src.data import loader
from src.evaluation import metrics as mt
from src.modeling import split
from src.preprocessing import pipeline as pl

log = logging.getLogger(__name__)

CSV_TUNING = config.DIR_METRICS / "tuning_lgbm.csv"
JSON_MELHOR = config.DIR_METRICS / "hiperparametros_campeao.json"

N_ITERACOES = 40
N_LINHAS_TUNING = 400_000

ESPACO = {
    "n_estimators": randint(200, 900),
    "num_leaves": randint(15, 128),
    "max_depth": randint(4, 13),
    "learning_rate": loguniform(0.01, 0.2),
    "min_child_samples": randint(50, 2001),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_lambda": uniform(0.0, 10.0),
}

FIXOS = dict(subsample_freq=1, n_jobs=-1, verbose=-1)


def _configuracoes(n: int = N_ITERACOES, seed: int = config.RANDOM_STATE) -> list[dict]:
    return list(ParameterSampler(ESPACO, n_iter=n, random_state=seed))


def etapa_tuning(
    n_iteracoes: int = N_ITERACOES,
    n_linhas: int = N_LINHAS_TUNING,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    dados = loader.carregar_dataset_modelagem()
    _, y, grupos = pl.separar_X_y(dados)
    dev, _ = split.separar_desenvolvimento_e_teste(grupos, seed=seed)

    posicoes = split.amostrar_municipios(grupos, n_linhas, seed=seed, posicoes=dev)
    amostra = dados.iloc[posicoes].reset_index(drop=True)
    X, y_am, g_am = pl.separar_X_y(amostra)
    particoes = split.folds(y_am, g_am, seed=seed)
    log.info(
        "tuning sobre %d alunos de %d municípios (prevalência %.4f)",
        len(amostra), g_am.nunique(), y_am.mean(),
    )

    config.DIR_METRICS.mkdir(parents=True, exist_ok=True)
    feitas = set()
    linhas: list[dict] = []
    if CSV_TUNING.exists():
        anterior = pd.read_csv(CSV_TUNING)
        linhas = anterior.to_dict("records")
        feitas = set(anterior["iteracao"])

    for i, parametros in enumerate(_configuracoes(n_iteracoes, seed)):
        if i in feitas:
            continue
        inicio = time.time()
        aucs, briers = [], []
        for treino, validacao in particoes:
            modelo = pl.montar_pipeline(LGBMClassifier(random_state=seed, **parametros, **FIXOS))
            modelo.fit(X.iloc[treino], y_am.iloc[treino])
            escore = modelo.predict_proba(X.iloc[validacao])[:, 1]
            resultado = mt.metricas_de_ranking(y_am.iloc[validacao], escore)
            aucs.append(resultado["roc_auc"])
            briers.append(resultado["brier"])
        linhas.append(
            {
                "iteracao": i,
                "roc_auc": float(np.mean(aucs)),
                "roc_auc_dp": float(np.std(aucs, ddof=1)),
                "brier": float(np.mean(briers)),
                "segundos": round(time.time() - inicio, 1),
                **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in parametros.items()},
            }
        )
        pd.DataFrame(linhas).sort_values("iteracao").to_csv(CSV_TUNING, index=False)
        log.info(
            "iteração %d/%d: ROC-AUC %.4f ± %.4f | %.0fs",
            i + 1, n_iteracoes, linhas[-1]["roc_auc"], linhas[-1]["roc_auc_dp"], linhas[-1]["segundos"],
        )

    tabela = pd.DataFrame(linhas).sort_values("roc_auc", ascending=False)
    melhor = tabela.iloc[0]
    parametros = {k: melhor[k] for k in ESPACO}
    parametros = {
        k: (int(v) if k in ("n_estimators", "num_leaves", "max_depth", "min_child_samples") else float(v))
        for k, v in parametros.items()
    }
    JSON_MELHOR.write_text(
        json.dumps(
            {
                "parametros": parametros,
                "fixos": FIXOS,
                "roc_auc_cv_amostra": float(melhor["roc_auc"]),
                "roc_auc_dp_entre_folds": float(melhor["roc_auc_dp"]),
                "n_iteracoes": int(len(tabela)),
                "n_linhas_amostra": int(len(amostra)),
                "n_municipios_amostra": int(g_am.nunique()),
                "seed": seed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("melhor configuração gravada em %s", JSON_MELHOR)
    return tabela


def carregar_hiperparametros() -> dict:
    """Hiperparâmetros do campeão, prontos para `LGBMClassifier(**parametros)`."""
    conteudo = json.loads(JSON_MELHOR.read_text(encoding="utf-8"))
    return {**conteudo["parametros"], **conteudo["fixos"]}


CSV_BORDAS = config.DIR_METRICS / "tuning_bordas.csv"


def refinar_bordas(
    n_estimators=(883, 1500, 2500), reg_lambda=(9.86, 30.0, 100.0), seed: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Grade pequena além da borda, para descobrir se o limite estava atrapalhando.

    A busca escolheu o maior `n_estimators` e o maior `reg_lambda` que o espaço
    oferecia. Ótimo encostado na borda quase sempre significa que a busca queria
    ir além dela, e o que se reporta passa a ser o limite da grade e não o ótimo
    — então a grade tem de ser esticada, nem que seja para confirmar que não
    havia nada do outro lado.

    A varredura é dos dois parâmetros de borda apenas, com o resto fixo no
    vencedor: o objetivo é responder à pergunta da borda, não recomeçar a busca.
    """
    dados = loader.carregar_dataset_modelagem()
    _, y, grupos = pl.separar_X_y(dados)
    dev, _ = split.separar_desenvolvimento_e_teste(grupos, seed=seed)
    posicoes = split.amostrar_municipios(grupos, N_LINHAS_TUNING, seed=seed, posicoes=dev)
    amostra = dados.iloc[posicoes].reset_index(drop=True)
    X, y_am, g_am = pl.separar_X_y(amostra)
    particoes = split.folds(y_am, g_am, seed=seed)

    base = carregar_hiperparametros()
    linhas = []
    for arvores in n_estimators:
        for penalidade in reg_lambda:
            parametros = {**base, "n_estimators": int(arvores), "reg_lambda": float(penalidade)}
            inicio = time.time()
            aucs = []
            for treino, validacao in particoes:
                modelo = pl.montar_pipeline(LGBMClassifier(random_state=seed, **parametros))
                modelo.fit(X.iloc[treino], y_am.iloc[treino])
                aucs.append(
                    mt.metricas_de_ranking(
                        y_am.iloc[validacao], modelo.predict_proba(X.iloc[validacao])[:, 1]
                    )["roc_auc"]
                )
            linhas.append(
                {
                    "n_estimators": int(arvores),
                    "reg_lambda": float(penalidade),
                    "roc_auc": float(np.mean(aucs)),
                    "roc_auc_dp": float(np.std(aucs, ddof=1)),
                    "segundos": round(time.time() - inicio, 1),
                }
            )
            pd.DataFrame(linhas).to_csv(CSV_BORDAS, index=False)
            log.info(
                "borda n_estimators=%d reg_lambda=%.2f: ROC-AUC %.4f",
                arvores, penalidade, linhas[-1]["roc_auc"],
            )
    return pd.DataFrame(linhas)


def diagnostico_de_borda(tabela: pd.DataFrame | None = None, folga: float = 0.05) -> pd.DataFrame:
    """Algum parâmetro ótimo caiu na borda da grade? Se caiu, a grade estava errada.

    Compara o valor escolhido com os limites do espaço amostrado. Um ótimo
    encostado na borda quase sempre significa que a busca queria ir além dela e
    não pôde — e o resultado que se reporta é o do limite, não o do ótimo.
    """
    tabela = pd.read_csv(CSV_TUNING) if tabela is None else tabela
    melhor = tabela.sort_values("roc_auc", ascending=False).iloc[0]
    linhas = []
    for nome in ESPACO:
        baixo, alto = tabela[nome].min(), tabela[nome].max()
        amplitude = alto - baixo
        valor = melhor[nome]
        linhas.append(
            {
                "parametro": nome,
                "escolhido": valor,
                "minimo_amostrado": baixo,
                "maximo_amostrado": alto,
                "na_borda": bool(
                    amplitude > 0
                    and (valor <= baixo + folga * amplitude or valor >= alto - folga * amplitude)
                ),
            }
        )
    return pd.DataFrame(linhas)
