"""Métricas do risco de não alfabetização, com incerteza e sempre estratificadas.

Três regras que o resto do módulo materializa:

1. **Nenhuma métrica sai sozinha.** O intervalo de confiança vem por bootstrap
   reamostrando *municípios*, nunca alunos. Um bootstrap de alunos trataria as
   2.700 crianças de um município como 2.700 observações independentes e
   devolveria um intervalo estreito e falso — o mesmo erro que o split aleatório
   comete, com outra roupa.
2. **Nenhuma métrica agregada sem a estratificada ao lado.** A coluna
   `fonte_lag_municipal` divide a coorte em quatro populações com apurações
   diferentes (B4), e `tem_historico_municipio` separa SP, DF e AC do resto
   (B3). Um único número esconde as três.
3. **O uso real do modelo é ranking, não rótulo.** ROC-AUC é a métrica primária,
   Brier e a curva de calibração são críticos porque as probabilidades viram
   ranking municipal na Etapa 6, e o limiar é escolha operacional — não 0,5.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src import config

N_REAMOSTRAS = 1_000


# ---------------------------------------------------------------------------
# Métricas pontuais
# ---------------------------------------------------------------------------
def ks(y, escore) -> float:
    """Maior distância entre as acumuladas das duas classes."""
    fpr, tpr, _ = roc_curve(y, escore)
    return float(np.max(tpr - fpr))


def metricas_de_ranking(y, escore) -> dict[str, float]:
    """ROC-AUC, PR-AUC, KS e Brier.

    As três primeiras dependem só da ordenação; o Brier depende da escala, e é
    por isso que ele aparece aqui e não junto das métricas de limiar — um modelo
    pode ranquear bem e estar mal calibrado, e a Etapa 6 precisa das duas coisas.
    """
    y = np.asarray(y)
    escore = np.asarray(escore, dtype=float)
    return {
        "roc_auc": float(roc_auc_score(y, escore)),
        "pr_auc": float(average_precision_score(y, escore)),
        "ks": ks(y, escore),
        "brier": float(brier_score_loss(y, escore)),
        "prevalencia": float(y.mean()),
        "n": int(len(y)),
    }


def metricas_no_limiar(y, escore, limiar: float) -> dict[str, float]:
    """Recall, precisão, F1 e a matriz de confusão no limiar escolhido."""
    y = np.asarray(y)
    predito = (np.asarray(escore, dtype=float) >= limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, predito, labels=[0, 1]).ravel()
    recall = vp / (vp + fn) if vp + fn else 0.0
    precisao = vp / (vp + fp) if vp + fp else 0.0
    f1 = 2 * recall * precisao / (recall + precisao) if recall + precisao else 0.0
    return {
        "limiar": float(limiar),
        "recall": float(recall),
        "precisao": float(precisao),
        "f1": float(f1),
        "especificidade": float(vn / (vn + fp)) if vn + fp else 0.0,
        "taxa_de_alerta": float(predito.mean()),
        "vp": int(vp),
        "fp": int(fp),
        "fn": int(fn),
        "vn": int(vn),
    }


def escolher_limiar_por_f1(y, escore) -> float:
    """Limiar que maximiza F1 na curva precision-recall.

    Não é a única escolha defensável, e nem a mais importante: o uso real é
    alocar orçamento, então a `curva_de_capacidade` costuma decidir mais. O F1
    entra como referência neutra, na falta de uma matriz de custo acordada com
    quem paga a intervenção.
    """
    precisao, recall, limiares = precision_recall_curve(y, escore)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * precisao * recall / (precisao + recall)
    f1 = np.nan_to_num(f1[:-1])
    return float(limiares[int(np.argmax(f1))])


def curva_de_capacidade(y, escore, fracoes=(0.05, 0.10, 0.20, 0.30, 0.40, 0.50)) -> pd.DataFrame:
    """Se há verba para atender X% das crianças, quantas em risco o modelo entrega?

    É a tradução da métrica para a unidade em que a decisão acontece. `lift` diz
    quantas vezes o alvo do modelo concentra mais risco que um sorteio.
    """
    y = np.asarray(y)
    ordem = np.argsort(-np.asarray(escore, dtype=float))
    acertos = np.cumsum(y[ordem])
    total = y.sum()
    linhas = []
    for fracao in fracoes:
        corte = max(int(round(fracao * len(y))), 1)
        capturado = acertos[corte - 1]
        linhas.append(
            {
                "fracao_atendida": fracao,
                "n_atendidos": corte,
                "recall": float(capturado / total),
                "precisao": float(capturado / corte),
                "lift": float((capturado / corte) / y.mean()),
            }
        )
    return pd.DataFrame(linhas)


def lift_por_decil(y, escore) -> pd.DataFrame:
    """Concentração de risco por decil do escore, do mais arriscado para o menos."""
    quadro = pd.DataFrame({"y": np.asarray(y), "escore": np.asarray(escore, dtype=float)})
    quadro["decil"] = pd.qcut(
        quadro["escore"].rank(method="first", ascending=False), 10, labels=range(1, 11)
    )
    tabela = (
        quadro.groupby("decil", observed=True)
        .agg(n=("y", "size"), taxa=("y", "mean"))
        .reset_index()
    )
    tabela["lift"] = tabela["taxa"] / quadro["y"].mean()
    return tabela


def curva_calibracao(y, escore, n_bins: int = 20) -> pd.DataFrame:
    """Probabilidade predita x frequência observada, em bins de igual frequência."""
    quadro = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(escore, dtype=float)})
    quadro["bin"] = pd.qcut(quadro["p"].rank(method="first"), n_bins, labels=False)
    return (
        quadro.groupby("bin", observed=True)
        .agg(n=("y", "size"), predito=("p", "mean"), observado=("y", "mean"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Estratificação — obrigatória, não opcional
# ---------------------------------------------------------------------------
def metricas_estratificadas(y, escore, estrato, minimo: int = 500) -> pd.DataFrame:
    """Uma linha por nível do estrato, com ROC-AUC, PR-AUC, KS, Brier e prevalência.

    Estratos com menos de `minimo` alunos, ou com uma única classe, entram com
    métrica nula em vez de sumir: some-los esconderia que existem, e a AUC ali
    seria ruído apresentado como medida.
    """
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    escore = pd.Series(np.asarray(escore, dtype=float)).reset_index(drop=True)
    estrato = pd.Series(np.asarray(estrato)).reset_index(drop=True)

    linhas = []
    for nivel, posicoes in estrato.groupby(estrato, observed=True).groups.items():
        fatia_y = y.loc[posicoes]
        if len(fatia_y) < minimo or fatia_y.nunique() < 2:
            linhas.append(
                {
                    "estrato": nivel,
                    "n": int(len(fatia_y)),
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "ks": np.nan,
                    "brier": np.nan,
                    "prevalencia": float(fatia_y.mean()) if len(fatia_y) else np.nan,
                }
            )
            continue
        linhas.append({"estrato": nivel, **metricas_de_ranking(fatia_y, escore.loc[posicoes])})
    return pd.DataFrame(linhas).sort_values("n", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Incerteza — bootstrap de municípios
# ---------------------------------------------------------------------------
def bootstrap_por_municipio(
    y,
    escore,
    grupos,
    metricas: dict[str, Callable] | None = None,
    n_reamostras: int = N_REAMOSTRAS,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """IC95 percentílico reamostrando municípios com reposição.

    A unidade de reamostragem é o município porque é a unidade de independência
    do desenho: o teste é formado por municípios inteiros e inéditos, e é a
    variação *entre* eles que o intervalo precisa capturar. Reamostrar alunos
    daria um intervalo várias vezes mais estreito e sem significado.
    """
    metricas = metricas or {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "brier": brier_score_loss,
    }
    y = np.asarray(y)
    escore = np.asarray(escore, dtype=float)
    grupos = np.asarray(grupos)

    ordem = np.argsort(grupos, kind="stable")
    y, escore, grupos = y[ordem], escore[ordem], grupos[ordem]
    fronteiras = np.flatnonzero(np.diff(grupos)) + 1
    inicios = np.concatenate(([0], fronteiras))
    fins = np.concatenate((fronteiras, [len(y)]))
    n_municipios = len(inicios)

    rng = np.random.default_rng(seed)
    amostras: dict[str, list[float]] = {nome: [] for nome in metricas}
    for _ in range(n_reamostras):
        sorteados = rng.integers(0, n_municipios, n_municipios)
        posicoes = np.concatenate([np.arange(inicios[m], fins[m]) for m in sorteados])
        yb, eb = y[posicoes], escore[posicoes]
        if yb.min() == yb.max():
            continue
        for nome, funcao in metricas.items():
            amostras[nome].append(float(funcao(yb, eb)))

    return pd.DataFrame(
        [
            {
                "metrica": nome,
                "pontual": float(funcao(y, escore)),
                "ic_baixo": float(np.percentile(amostras[nome], 2.5)),
                "ic_alto": float(np.percentile(amostras[nome], 97.5)),
                "desvio_bootstrap": float(np.std(amostras[nome], ddof=1)),
                "n_reamostras": len(amostras[nome]),
            }
            for nome, funcao in metricas.items()
        ]
    )


# ---------------------------------------------------------------------------
# Grão municipal — é onde a decisão acontece
# ---------------------------------------------------------------------------
def agregar_por_municipio(y, escore, grupos, pesos=None) -> pd.DataFrame:
    """Taxa observada e risco médio predito por município, ponderados por `peso_aluno`.

    A ponderação segue B2: todo número **populacional** que sai do projeto usa o
    fator oficial de não-resposta do INEP. O classificador em grão aluno segue
    sem peso; é só a agregação que o exige.
    """
    quadro = pd.DataFrame(
        {
            "id_municipio": np.asarray(grupos),
            "y": np.asarray(y, dtype=float),
            "p": np.asarray(escore, dtype=float),
            "w": 1.0 if pesos is None else np.asarray(pesos, dtype=float),
        }
    )
    quadro["yw"], quadro["pw"] = quadro["y"] * quadro["w"], quadro["p"] * quadro["w"]
    tabela = quadro.groupby("id_municipio").agg(
        n_alunos=("y", "size"), soma_peso=("w", "sum"), yw=("yw", "sum"), pw=("pw", "sum")
    )
    tabela["taxa_observada"] = tabela["yw"] / tabela["soma_peso"]
    tabela["risco_predito"] = tabela["pw"] / tabela["soma_peso"]
    return tabela.drop(columns=["yw", "pw"]).reset_index()


def r2_municipal(y, escore, grupos, pesos=None, portes=(0, 50, 200)) -> pd.DataFrame:
    """R2 entre risco médio predito e taxa observada, por faixa de porte.

    O R2 sobe com o porte porque a taxa observada de um município de 20 alunos é
    dominada por ruído amostral. Reportar só o número da base completa
    subestimaria o modelo onde ele é usado; reportar só o dos municípios grandes
    o superestimaria em toda parte.
    """
    tabela = agregar_por_municipio(y, escore, grupos, pesos)
    linhas = []
    for porte in portes:
        fatia = tabela[tabela["n_alunos"] >= porte]
        residuo = ((fatia["taxa_observada"] - fatia["risco_predito"]) ** 2).sum()
        total = ((fatia["taxa_observada"] - fatia["taxa_observada"].mean()) ** 2).sum()
        linhas.append(
            {
                "porte_minimo": porte,
                "n_municipios": int(len(fatia)),
                "r2": float(1 - residuo / total) if total else np.nan,
                "correlacao": float(fatia["taxa_observada"].corr(fatia["risco_predito"])),
                "vies_medio": float((fatia["risco_predito"] - fatia["taxa_observada"]).mean()),
            }
        )
    return pd.DataFrame(linhas)
