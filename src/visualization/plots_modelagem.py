"""Gráficos da Etapa 4. Mesma convenção e mesma paleta dos notebooks anteriores.

`medir_*` calcula, `plotar_*` recebe número já medido. Nenhuma função aqui treina
nada: tudo o que aparece no notebook 03 vem dos CSV e JSON de `reports/metrics/`,
que são gravados pelos módulos de `src/modeling/` e `src/evaluation/`. Assim o
notebook reexecuta em segundos e o gráfico nunca discorda da tabela.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src import config
from src.eda import ACENTO, ACENTO_ALERTA, CINZA_CLARO, CINZA_ESCURO, CINZA_MEDIO

DIR_IMAGENS_MODELAGEM = config.DIR_IMAGES / "modelagem"


def salvar_figura(fig: Figure, nome: str) -> str:
    """Grava em `images/modelagem/` e devolve o caminho relativo à raiz."""
    DIR_IMAGENS_MODELAGEM.mkdir(parents=True, exist_ok=True)
    caminho = DIR_IMAGENS_MODELAGEM / f"{nome}.png"
    fig.savefig(caminho)
    return str(caminho.relative_to(config.ROOT)).replace("\\", "/")


def plotar_comparacao_de_modelos(resumo: pd.DataFrame) -> Figure:
    """ROC-AUC médio por modelo, com a barra de erro do desvio entre folds.

    A barra de erro é o argumento principal do gráfico: ela mostra que a
    distância entre o campeão e a regra de uma variável é da ordem de duas vezes
    a oscilação entre partições. Sem ela, a mesma figura contaria uma história de
    superioridade que os dados não sustentam.
    """
    ordenado = resumo.sort_values("roc_auc")
    y = np.arange(len(ordenado))
    cores = [ACENTO if nome.startswith(("lightgbm", "random")) else CINZA_MEDIO for nome in ordenado["modelo"]]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh(y, ordenado["roc_auc"], 0.6, color=cores,
            xerr=ordenado["roc_auc_dp"], error_kw={"ecolor": CINZA_ESCURO, "elinewidth": 1, "capsize": 3})
    ax.axvline(0.5, color=CINZA_ESCURO, linewidth=0.9, linestyle=":")
    barra = float(ordenado.loc[ordenado["modelo"] == "heuristica_taxa_municipal", "roc_auc"].iloc[0])
    ax.axvline(barra, color=ACENTO_ALERTA, linewidth=1.2, linestyle="--")
    ax.text(barra + 0.003, -0.45, "regra de uma variável", color=ACENTO_ALERTA, fontsize=8)

    ax.set_yticks(y, ordenado["modelo"])
    ax.set_xlim(0.45, 0.72)
    ax.set_xlabel("ROC-AUC médio em StratifiedGroupKFold(5) por município")
    ax.set_title("O ganho do modelo sobre a regra de uma variável cabe em duas barras de erro")
    for posicao, valor in zip(y, ordenado["roc_auc"]):
        ax.text(valor + 0.012, posicao, f"{valor:.4f}", va="center", fontsize=9, color=CINZA_ESCURO)
    return fig


def plotar_curva_calibracao(tabela: pd.DataFrame) -> Figure:
    """Probabilidade predita x frequência observada, antes e depois da isotônica."""
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot([0, 1], [0, 1], color=CINZA_MEDIO, linestyle=":", linewidth=1.2, label="calibração perfeita")
    for versao, cor, marcador in (("bruto", CINZA_ESCURO, "s"), ("calibrado", ACENTO, "o")):
        fatia = tabela[tabela["versao"] == versao]
        ax.plot(fatia["predito"], fatia["observado"], marcador + "-", color=cor, markersize=4,
                linewidth=1.4, label=f"LightGBM {versao}")
    ax.set_xlabel("probabilidade predita de não alfabetização")
    ax.set_ylabel("frequência observada no conjunto de teste")
    ax.set_title("Calibração em 20 bins de igual frequência")
    ax.legend(loc="upper left")
    return fig


def plotar_capacidade(capacidade: pd.DataFrame, prevalencia: float) -> Figure:
    """Cobertura e precisão em função da fração de crianças atendidas.

    É a tradução da métrica para a pergunta que o gestor faz: com verba para
    atender X% da coorte, quantas crianças em risco entram na política?
    """
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = capacidade["fracao_atendida"] * 100
    ax.plot(x, capacidade["recall"] * 100, "o-", color=ACENTO, linewidth=2, label="cobertura das crianças em risco")
    ax.plot(x, capacidade["precisao"] * 100, "s-", color=CINZA_ESCURO, linewidth=1.6, label="precisão do alerta")
    ax.plot(x, x, color=CINZA_MEDIO, linestyle=":", linewidth=1.2, label="sorteio aleatório")
    ax.axhline(prevalencia * 100, color=ACENTO_ALERTA, linestyle="--", linewidth=1)
    ax.text(1, prevalencia * 100 + 1.2, f"prevalência {prevalencia:.1%}", color=ACENTO_ALERTA, fontsize=8)
    ax.set_xlabel("% da coorte atendida, do maior risco para o menor")
    ax.set_ylabel("%")
    ax.set_title("Cenários de capacidade orçamentária")
    ax.legend()
    return fig


def plotar_curva_de_aprendizado(tabela: pd.DataFrame) -> Figure:
    """AUC de treino e de validação contra o número de municípios no treino.

    A abscissa é município porque é essa a unidade de informação. Uma curva de
    validação achatada com treino próximo dela é a assinatura de teto
    informacional: mais dado do mesmo tipo não move o número.
    """
    resumo = tabela.groupby("fracao").agg(
        municipios=("n_municipios_treino", "mean"),
        treino=("roc_auc_treino", "mean"),
        validacao=("roc_auc_validacao", "mean"),
        validacao_dp=("roc_auc_validacao", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(resumo["municipios"], resumo["treino"], "s--", color=CINZA_ESCURO, linewidth=1.4, label="treino")
    ax.plot(resumo["municipios"], resumo["validacao"], "o-", color=ACENTO, linewidth=2, label="validação (municípios inéditos)")
    ax.fill_between(
        resumo["municipios"],
        resumo["validacao"] - resumo["validacao_dp"].fillna(0),
        resumo["validacao"] + resumo["validacao_dp"].fillna(0),
        color=ACENTO, alpha=0.15,
    )
    ax.set_xscale("log")
    ax.set_xlabel("municípios no treino (escala log)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Curva de aprendizado — o teto é informacional, não amostral")
    ax.legend()
    return fig


def plotar_permutacao(tabela: pd.DataFrame) -> Figure:
    """As duas distribuições nulas e a AUC real, cada uma na sua escala.

    Painéis separados, e não sobrepostos: o nulo global fica em 0,500 e o nulo
    dentro do município em 0,664, então num eixo compartilhado o segundo colapsa
    contra a linha do modelo real e desaparece — justamente o que ele tem a
    dizer. A distância entre a linha vermelha e o histograma é a resposta de cada
    pergunta, e cada pergunta tem a sua régua.
    """
    real = float(tabela.loc[tabela["tipo"] == "real", "roc_auc"].iloc[0])
    painéis = (
        ("nulo_global", CINZA_MEDIO, "Existe sinal?", "alvo embaralhado globalmente"),
        (
            "nulo_dentro_do_municipio",
            ACENTO,
            "O modelo usa algo além do município?",
            "alvo embaralhado dentro de cada município",
        ),
    )
    fig, eixos = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, (tipo, cor, titulo, rotulo) in zip(eixos, painéis):
        valores = tabela.loc[tabela["tipo"] == tipo, "roc_auc"].to_numpy()
        if not len(valores):
            continue
        ax.hist(valores, bins=15, color=cor, alpha=0.85, edgecolor="white", linewidth=0.5, label=rotulo)
        ax.axvline(real, color=ACENTO_ALERTA, linewidth=2)
        margem = max(real - valores.min(), 0.002) * 0.12
        ax.set_xlim(valores.min() - margem, real + margem)
        ax.annotate(
            f"modelo real\n{real:.4f}",
            xy=(real, ax.get_ylim()[1] * 0.9),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=9,
            color=ACENTO_ALERTA,
        )
        ax.text(
            valores.mean(), ax.get_ylim()[1] * 0.9,
            f"nulo\n{valores.mean():.4f}", ha="center", va="top", fontsize=9, color=CINZA_ESCURO,
        )
        ax.set_title(f"{titulo}\n({rotulo}, {len(valores)} permutações)", fontsize=10)
        ax.set_xlabel("ROC-AUC")
        ax.set_ylabel("permutações")
    fig.tight_layout()
    return fig


def plotar_estratificado(estratificado: pd.DataFrame, dimensao: str, titulo: str) -> Figure:
    """ROC-AUC por estrato, com o tamanho de cada fatia ao lado da barra."""
    fatia = estratificado[estratificado["dimensao"] == dimensao].sort_values("roc_auc")
    y = np.arange(len(fatia))
    fig, ax = plt.subplots(figsize=(8.5, max(2.4, 0.42 * len(fatia) + 1.4)))
    ax.barh(y, fatia["roc_auc"].fillna(0), 0.6, color=CINZA_MEDIO)
    ax.axvline(0.5, color=CINZA_ESCURO, linewidth=0.9, linestyle=":")
    ax.set_yticks(y, fatia["estrato"])
    ax.set_xlim(0.45, max(0.75, float(fatia["roc_auc"].max(skipna=True) or 0.75) + 0.05))
    ax.set_xlabel("ROC-AUC no conjunto de teste")
    ax.set_title(titulo)
    for posicao, (valor, n) in enumerate(zip(fatia["roc_auc"], fatia["n"])):
        rotulo = "sem métrica" if not np.isfinite(valor) else f"{valor:.4f}   n = {n:,}".replace(",", ".")
        ax.text(0.455 if not np.isfinite(valor) else valor + 0.004, posicao, rotulo,
                va="center", fontsize=8.5, color=CINZA_ESCURO)
    return fig


def plotar_risco_municipal(tabela: pd.DataFrame, porte_minimo: int = 200) -> Figure:
    """Risco médio predito x taxa observada, um ponto por município do teste.

    O grão municipal é onde a decisão acontece, e é onde o modelo é bom: o mesmo
    escore que separa duas crianças com AUC 0,66 ordena municípios com
    correlação bem mais alta, porque a média cancela o ruído individual.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6))
    pequenos = tabela[tabela["n_alunos"] < porte_minimo]
    grandes = tabela[tabela["n_alunos"] >= porte_minimo]
    ax.scatter(pequenos["risco_predito"], pequenos["taxa_observada"], s=8, alpha=0.25,
               color=CINZA_CLARO, edgecolors="none", label=f"< {porte_minimo} alunos")
    ax.scatter(grandes["risco_predito"], grandes["taxa_observada"], s=14, alpha=0.55,
               color=ACENTO, edgecolors="none", label=f"≥ {porte_minimo} alunos")
    ax.plot([0, 1], [0, 1], color=CINZA_ESCURO, linestyle=":", linewidth=1.2)
    ax.set_xlabel("risco médio predito no município")
    ax.set_ylabel("taxa observada de não alfabetização (ponderada)")
    ax.set_title("Agregação municipal — onde a predição é utilizável")
    ax.legend(loc="upper left")
    return fig


def plotar_drift_e_logo(drift: pd.DataFrame, logo: pd.DataFrame) -> Figure:
    """Os dois testes de generalização estrutural, lado a lado."""
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4))

    y = np.arange(len(drift))
    cores = [ACENTO_ALERTA if "out-of-time" in nome else CINZA_MEDIO for nome in drift["desenho"]]
    eixos[0].barh(y, drift["roc_auc"], 0.55, color=cores)
    eixos[0].set_yticks(y, drift["desenho"])
    eixos[0].axvline(0.5, color=CINZA_ESCURO, linewidth=0.9, linestyle=":")
    eixos[0].set_xlim(0.45, 0.68)
    eixos[0].set_xlabel("ROC-AUC")
    eixos[0].set_title("Drift — modelo reduzido a território e rede")
    for posicao, valor in zip(y, drift["roc_auc"]):
        eixos[0].text(valor + 0.004, posicao, f"{valor:.4f}", va="center", fontsize=9)

    ordenado = logo.sort_values("roc_auc")
    y = np.arange(len(ordenado))
    eixos[1].barh(y, ordenado["roc_auc"], 0.55, color=CINZA_MEDIO)
    eixos[1].axvline(0.5, color=CINZA_ESCURO, linewidth=0.9, linestyle=":")
    eixos[1].set_yticks(y, ordenado["regiao_de_teste"])
    eixos[1].set_xlim(0.45, 0.72)
    eixos[1].set_xlabel("ROC-AUC na região deixada de fora")
    eixos[1].set_title("LeaveOneGroupOut por região")
    for posicao, valor in zip(y, ordenado["roc_auc"]):
        eixos[1].text(valor + 0.004, posicao, f"{valor:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig
