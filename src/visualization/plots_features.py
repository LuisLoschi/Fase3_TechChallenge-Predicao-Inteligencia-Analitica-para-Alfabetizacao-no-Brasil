"""Gráficos da etapa de engenharia de features, para o notebook 02.

Cada etapa seguinte tem o seu próprio módulo — `plots_modelagem`,
`plots_interpretabilidade` e `plots_estrategia` —, e todos seguem a convenção de
`src/eda.py`, de onde vem também a paleta: `medir_*` calcula e não desenha,
`plotar_*` recebe número já medido e devolve uma `Figure`. Reaproveitar o estilo
em vez de redefini-lo é o que faz o material dos cinco notebooks caber no mesmo
slide.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src import config
from src.eda import ACENTO, ACENTO_ALERTA, CINZA_CLARO, CINZA_ESCURO, CINZA_MEDIO

DIR_IMAGENS_FEATURES = config.DIR_IMAGES / "features"


def salvar_figura(fig: Figure, nome: str) -> str:
    """Grava em `images/features/` e devolve o caminho relativo à raiz."""
    DIR_IMAGENS_FEATURES.mkdir(parents=True, exist_ok=True)
    caminho = DIR_IMAGENS_FEATURES / f"{nome}.png"
    fig.savefig(caminho)
    return str(caminho.relative_to(config.ROOT)).replace("\\", "/")


def plotar_cobertura_por_fonte(fonte: pd.Series) -> Figure:
    """Quanto da coorte cada apuração de 2023 cobre, na ordem da fidelidade.

    Responde à pergunta que decide a coalescência: a feature de nível chega a
    quantos alunos, e com qual qualidade de apuração? Barras empilhadas e não
    lado a lado porque o que importa é o total acumulado até 98,09%.
    """
    ordem = ["gold", "agregado_r5", "agregado_r3", "sem_historico"]
    rotulos = {
        "gold": "Gold 2023 (microdado)",
        "agregado_r5": "agregado INEP, rede 5",
        "agregado_r3": "agregado INEP, rede 3",
        "sem_historico": "sem histórico",
    }
    partes = fonte.value_counts(normalize=True).reindex(ordem).fillna(0)
    cores = [CINZA_ESCURO, CINZA_MEDIO, ACENTO, ACENTO_ALERTA]

    fig, ax = plt.subplots(figsize=(9, 2.6))
    esquerda = 0.0
    for valor, cor, chave in zip(partes, cores, ordem):
        ax.barh([0], [valor], left=esquerda, color=cor, height=0.55)
        if valor > 0.015:
            ax.text(
                esquerda + valor / 2,
                0,
                f"{rotulos[chave]}\n{valor:.2%}",
                ha="center",
                va="center",
                color="white" if chave != "agregado_r3" else "white",
                fontsize=9,
            )
        esquerda += valor
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fração da coorte de 2024")
    ax.grid(False)
    ax.set_title(
        "Coalescência 5 → 3: a feature de nível cobre 98,09% da coorte\n"
        "(só a Gold cobriria 76,88% — o que falta é São Paulo)"
    )
    return fig


def plotar_auc_univariada(quadro: pd.DataFrame, titulo: str, n_maximo: int = 22) -> Figure:
    """Ranking de AUC univariada, com 0,5 marcado e a direção preservada.

    O eixo é a AUC do **risco**: abaixo de 0,5 a feature protege, acima ela
    agrava. Manter a direção em vez de plotar |AUC − 0,5| é o que permite ler o
    sentido do efeito sem consultar outra tabela.
    """
    dados = quadro.dropna(subset=["auc"]).copy()
    dados["distancia"] = (dados["auc"] - 0.5).abs()
    dados = dados.sort_values("distancia").tail(n_maximo)

    cores = [
        ACENTO if f.startswith("mun_") else (CINZA_ESCURO if f.startswith("uf_") else CINZA_MEDIO)
        for f in dados["feature"]
    ]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * len(dados) + 1.2)))
    y = np.arange(len(dados))
    ax.barh(y, dados["auc"] - 0.5, left=0.5, color=cores, height=0.72)
    ax.axvline(0.5, color=CINZA_ESCURO, linewidth=1)
    for yi, (auc, cobertura) in enumerate(zip(dados["auc"], dados["cobertura"])):
        deslocamento = 0.004 if auc < 0.5 else -0.004
        ax.text(
            auc + deslocamento,
            yi,
            f"{auc:.3f}  (cob. {cobertura:.0%})",
            va="center",
            ha="right" if auc < 0.5 else "left",
            fontsize=8,
        )
    ax.set_yticks(y, dados["feature"])
    ax.set_xlabel("ROC-AUC univariada do risco de não alfabetização")
    ax.set_title(titulo)
    return fig


def plotar_controle_embaralhamento(medidas: dict[str, float]) -> Figure:
    """O join por `id_escola` contra o sorteio de uma escola qualquer da UF.

    É o gráfico que sustenta a exclusão do bloco escolar: se a barra do join real
    não supera a do sorteio dentro da UF, o que a feature carrega é território, e
    não escola.
    """
    ordem = ["real", "dentro_uf", "global", "municipal"]
    rotulos = {
        "real": "join real por id_escola",
        "dentro_uf": "valor sorteado na mesma UF",
        "global": "valor sorteado em qualquer UF",
        "municipal": "taxa do município (referência)",
    }
    cores = {"real": ACENTO_ALERTA, "dentro_uf": CINZA_ESCURO, "global": CINZA_CLARO, "municipal": ACENTO}

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    y = np.arange(len(ordem))[::-1]
    valores = [medidas[chave] for chave in ordem]
    ax.barh(y, valores, color=[cores[c] for c in ordem], height=0.65)
    for yi, valor in zip(y, valores):
        ax.text(valor + 0.004, yi, f"{valor:.4f}", va="center", fontsize=9)
    ax.set_yticks(y, [rotulos[c] for c in ordem])
    ax.axvline(0.5, color=CINZA_ESCURO, linewidth=1, linestyle="--")
    ax.set_xlim(0.48, max(valores) + 0.04)
    ax.set_xlabel("ROC-AUC univariada (alfabetização)")
    ax.set_title(
        "O join real não bate o sorteio dentro da UF, e o sorteio entre UFs zera:\n"
        "o lag escolar carrega território, não escola"
    )
    return fig


def plotar_replica_b5(replica: pd.DataFrame) -> Figure:
    """Delta pareado por fold entre os conjuntos de features, com a média marcada.

    Cada ponto é um fold; o que interessa é se a nuvem está claramente de um lado
    do zero. Uma diferença média menor que a dispersão entre folds não é
    diferença — é a variância do desenho de validação.
    """
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    comparacoes = list(replica.columns)
    for i, coluna in enumerate(comparacoes):
        valores = replica[coluna].to_numpy()
        ax.scatter(valores, np.full_like(valores, i, dtype=float), color=CINZA_MEDIO, s=28, zorder=3)
        ax.scatter([valores.mean()], [i], color=ACENTO_ALERTA, s=90, marker="|", linewidths=2.5, zorder=4)
        ax.text(valores.mean(), i + 0.22, f"média {valores.mean():+.4f}", ha="center", fontsize=8)
    ax.axvline(0, color=CINZA_ESCURO, linewidth=1)
    ax.set_yticks(range(len(comparacoes)), comparacoes)
    ax.set_ylim(-0.5, len(comparacoes) - 0.3)
    ax.set_xlabel("Δ ROC-AUC pareado por fold (15 folds: 5 dobras × 3 seeds)")
    ax.set_title("Réplica do experimento B5: o ganho é menor que a dispersão entre folds")
    return fig
