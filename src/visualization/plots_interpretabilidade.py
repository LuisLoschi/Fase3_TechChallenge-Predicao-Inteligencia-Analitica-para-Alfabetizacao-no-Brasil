"""Gráficos da Etapa 5. Mesma convenção dos anteriores: `medir` fica em `src/evaluation`.

Nenhuma função aqui calcula importância. Tudo vem dos CSV de `reports/metrics/`
e do Parquet da amostra de SHAP, gravados por `python -m src.evaluation.interpret`,
para que o gráfico do notebook e a tabela do relatório não possam discordar.

A única exceção é o `beeswarm`, que precisa dos valores individuais e da matriz
transformada para colorir cada ponto — e a matriz é reconstruída a partir do dado
cru guardado na amostra, com o mesmo pré-processador que o campeão usa.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src import config
from src.eda import ACENTO, ACENTO_ALERTA, CINZA_ESCURO, CINZA_MEDIO
from src.evaluation import interpret as it

DIR_IMAGENS_INTERPRET = config.DIR_IMAGES / "interpretabilidade"

# Uma cor por família, em escala de cinza com um acento único. O acento fica na
# família municipal porque é a hipótese H1 que o gráfico existe para julgar.
COR_DA_FAMILIA = {
    "historico_municipal": ACENTO,
    "territorial": CINZA_ESCURO,
    "estrutural": CINZA_MEDIO,
    "metas": "#a8a8a8",
    "historico_escolar": ACENTO_ALERTA,
}


def salvar_figura(fig: Figure, nome: str) -> str:
    """Grava em `images/interpretabilidade/` e devolve o caminho relativo à raiz."""
    DIR_IMAGENS_INTERPRET.mkdir(parents=True, exist_ok=True)
    caminho = DIR_IMAGENS_INTERPRET / f"{nome}.png"
    fig.savefig(caminho)
    return str(caminho.relative_to(config.ROOT)).replace("\\", "/")


def plotar_beeswarm(amostra: pd.DataFrame, pacote: dict, max_colunas: int = 16) -> Figure:
    """O `beeswarm` canônico do SHAP, sobre as colunas pós-`ColumnTransformer`.

    É o gráfico que carrega direção, e não só magnitude: um ponto à direita
    empurra a criança para o risco, e a cor diz se ali o valor da variável era
    alto ou baixo. Ler só o ranking de barras perderia justamente isso — que
    média de português alta do município em 2023 empurra a predição para baixo.

    A escala é log-odds, porque é onde a árvore soma. Um valor de 0,5 desloca a
    predição em meio logit a partir do valor esperado, o que perto da prevalência
    de 0,38 vale algo em torno de 12 pontos percentuais de probabilidade.
    """
    import shap

    matriz = it.reconstruir_matriz(amostra, pacote)
    valores = amostra[[f"shap__{c}" for c in matriz.columns]].to_numpy()
    explicacao = shap.Explanation(
        values=valores,
        base_values=np.full(len(matriz), float(amostra["valor_esperado"].iloc[0])),
        data=matriz.to_numpy(),
        feature_names=list(matriz.columns),
    )
    shap.plots.beeswarm(explicacao, max_display=max_colunas, show=False, color_bar=True)
    fig = plt.gcf()
    fig.set_size_inches(9.5, 6.4)
    fig.axes[0].set_xlabel("valor SHAP — impacto no log-odds de não alfabetização")
    fig.axes[0].set_title(
        f"SHAP em {len(matriz):,} alunos do conjunto de teste".replace(",", "."),
        fontsize=11, fontweight="bold", loc="left",
    )
    return fig


def plotar_barras_por_feature(
    shap_features: pd.DataFrame, n: int = 20
) -> Figure:
    """Importância média por feature de entrada, na coorte inteira e só na fatia Gold.

    As barras são a leitura global e os losangos a leitura restrita a
    `fonte_lag_municipal = gold`, onde o modelo vale AUC 0,6706 contra 0,5598 no
    agregado coalescido. Se os dois desenhos contarem histórias diferentes, a
    barra é média de populações que o modelo entende de formas distintas.
    """
    global_ = shap_features[shap_features["recorte"] == "global"].sort_values("media_do_abs_da_soma").tail(n)
    gold = shap_features[shap_features["recorte"] == "gold"].set_index("feature")
    y = np.arange(len(global_))

    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(global_) + 1.6))
    ax.barh(y, global_["media_do_abs_da_soma"], 0.62,
            color=[COR_DA_FAMILIA[f] for f in global_["familia"]], edgecolor="white", linewidth=0.4)
    ax.plot(gold.loc[global_["feature"], "media_do_abs_da_soma"], y, "D",
            color=ACENTO_ALERTA, markersize=4.5, linestyle="none", label="só onde o lag vem da Gold")
    ax.set_yticks(y, global_["feature"])
    ax.set_xlabel("média do |SHAP| da feature, em log-odds")
    ax.set_title("Importância por feature de entrada — as 89 colunas devolvidas às 20 variáveis")
    ax.legend(loc="lower right")
    for posicao, valor in zip(y, global_["media_do_abs_da_soma"]):
        ax.text(valor + 0.009, posicao, f"{valor:.4f}", va="center", fontsize=8.5, color=CINZA_ESCURO)
    ax.set_xlim(0, float(global_["media_do_abs_da_soma"].max()) * 1.22)
    return fig


def plotar_familias(familias: pd.DataFrame) -> Figure:
    """As duas leituras por família, lado a lado, na régua de cada uma.

    À esquerda a participação no |SHAP| total, que é uma decomposição e soma 100%.
    À direita a queda de ROC-AUC quando a família inteira é embaralhada de uma
    vez, com a linha do piso de ruído do controle negativo. As duas medem coisas
    diferentes: a primeira diz de quanto a família move a predição, a segunda
    quanto ela vale depois de descontado o que as outras já explicam.
    """
    global_ = familias[familias["recorte"] == "global"].sort_values("participacao")
    y = np.arange(len(global_))
    cores = [COR_DA_FAMILIA[f] for f in global_["familia"]]

    fig, eixos = plt.subplots(1, 2, figsize=(12, 3.8))
    eixos[0].barh(y, global_["participacao"] * 100, 0.6, color=cores, edgecolor="white", linewidth=0.4)
    eixos[0].set_yticks(y, global_["familia"])
    eixos[0].set_xlabel("% do |SHAP| total")
    eixos[0].set_title("Quanto a família move a predição")
    for posicao, valor in zip(y, global_["participacao"]):
        eixos[0].text(valor * 100 + 0.8, posicao, f"{valor:.1%}", va="center", fontsize=9, color=CINZA_ESCURO)
    eixos[0].set_xlim(0, float(global_["participacao"].max()) * 118)

    eixos[1].barh(y, global_["queda_permutacao_em_bloco"], 0.6, color=cores, edgecolor="white", linewidth=0.4,
                  xerr=global_["queda_permutacao_dp"], error_kw={"ecolor": CINZA_ESCURO, "elinewidth": 1, "capsize": 3})
    eixos[1].axvline(it.PISO_DE_RUIDO, color=ACENTO_ALERTA, linestyle="--", linewidth=1.2)
    eixos[1].text(it.PISO_DE_RUIDO * 1.15, -0.42, "piso de ruído do controle negativo",
                  color=ACENTO_ALERTA, fontsize=8)
    eixos[1].set_yticks(y, [""] * len(y))
    eixos[1].set_xlabel("queda de ROC-AUC ao embaralhar a família inteira")
    eixos[1].set_title("Quanto a família vale além das outras")
    for posicao, valor in zip(y, global_["queda_permutacao_em_bloco"]):
        eixos[1].text(valor + 0.003, posicao, f"{valor:.4f}", va="center", fontsize=9, color=CINZA_ESCURO)
    eixos[1].set_xlim(0, float(global_["queda_permutacao_em_bloco"].max()) * 1.22)
    fig.tight_layout()
    return fig


def plotar_permutacao(permutacao: pd.DataFrame, n: int = 20) -> Figure:
    """Queda de ROC-AUC por feature no conjunto de teste, contra o piso de ruído.

    A linha vermelha é o que o `caderno` — sorteado entre alunos, sem informação
    possível — produziu na Etapa 4. Barra que não a ultrapassa não distingue
    importância de acaso, e o gráfico existe para que essa fronteira seja lida
    junto com o ranking, e não depois dele.
    """
    ordenado = permutacao.sort_values("queda_media").tail(n)
    y = np.arange(len(ordenado))

    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(ordenado) + 1.6))
    ax.barh(y, ordenado["queda_media"], 0.62,
            color=[COR_DA_FAMILIA[f] for f in ordenado["familia"]], edgecolor="white", linewidth=0.4,
            xerr=ordenado["queda_dp"], error_kw={"ecolor": CINZA_ESCURO, "elinewidth": 0.9, "capsize": 2})
    ax.axvline(it.PISO_DE_RUIDO, color=ACENTO_ALERTA, linestyle="--", linewidth=1.2)
    ax.text(it.PISO_DE_RUIDO * 1.1, -0.6, f"piso de ruído {it.PISO_DE_RUIDO}", color=ACENTO_ALERTA, fontsize=8)
    ax.set_yticks(y, ordenado["feature"])
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel("queda de ROC-AUC ao embaralhar a coluna (escala log)")
    ax.set_title("Permutação no conjunto de teste — 10 repetições, 330.836 alunos")
    return fig


def plotar_coeficientes(coeficientes: pd.DataFrame) -> Figure:
    """Razão de chances da logística podada, com a variação entre folds.

    O modelo tem oito colunas e VIF abaixo de 3, então o coeficiente aqui é
    legível: acima de 1 a variável aumenta a chance de não alfabetização, abaixo
    de 1 a reduz. A barra de erro vem dos cinco folds agrupados por município —
    coeficiente cujo intervalo cruza 1 não sustenta afirmação de direção.
    """
    ordenado = coeficientes.sort_values("coeficiente")
    y = np.arange(len(ordenado))
    cores = [ACENTO_ALERTA if c > 0 else ACENTO for c in ordenado["coeficiente"]]

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(ordenado) + 1.6))
    ax.barh(y, ordenado["coeficiente"], 0.62, color=cores, edgecolor="white", linewidth=0.4,
            xerr=ordenado["coeficiente_dp_entre_folds"],
            error_kw={"ecolor": CINZA_ESCURO, "elinewidth": 1, "capsize": 3})
    ax.axvline(0, color=CINZA_ESCURO, linewidth=1)
    ax.set_yticks(y, ordenado["coluna"])
    ax.set_xlabel("coeficiente em log-odds (numéricas padronizadas)")
    ax.set_title("Logística podada — direção e magnitude, com o desvio entre os 5 folds")
    for posicao, (coeficiente, razao) in enumerate(zip(ordenado["coeficiente"], ordenado["razao_de_chances"])):
        deslocamento = 0.02 if coeficiente >= 0 else -0.02
        ax.text(coeficiente + deslocamento, posicao, f"OR {razao:.2f}", va="center", fontsize=8.5,
                ha="left" if coeficiente >= 0 else "right", color=CINZA_ESCURO)
    limite = float(np.abs(ordenado["coeficiente"]).max()) * 1.45
    ax.set_xlim(-limite, limite)
    return fig


def plotar_dependencia(
    amostra: pd.DataFrame, mapa: pd.DataFrame, features: list[str], n_pontos: int = 12_000
) -> Figure:
    """Forma da relação entre o valor cru da variável e a contribuição SHAP dela.

    O eixo x é o valor **cru**, não o padronizado: uma taxa de 0,45 significa
    alguma coisa para quem lê, e −1,2 desvios não. O eixo y soma os valores SHAP
    de todas as colunas geradas por aquela variável, incluindo o indicador de
    nulo, porque é a contribuição da variável que interessa e não a de uma das
    suas codificações.

    A dispersão vertical em cada x é interação: se a nuvem fosse uma linha, o
    efeito seria aditivo e um modelo linear bastaria. É onde se vê o que o
    boosting comprou sobre a logística.
    """
    fig, eixos = plt.subplots(2, 2, figsize=(11.5, 7.6))
    rng = np.random.default_rng(config.RANDOM_STATE)
    posicoes = rng.choice(len(amostra), min(n_pontos, len(amostra)), replace=False)

    for ax, feature in zip(eixos.ravel(), features):
        colunas = mapa.loc[mapa["feature"] == feature, "coluna"]
        contribuicao = amostra[[f"shap__{c}" for c in colunas]].sum(axis=1).to_numpy()[posicoes]
        valores = amostra[f"x__{feature}"].to_numpy()[posicoes]

        if amostra[f"x__{feature}"].dtype == object or amostra[f"x__{feature}"].nunique() <= 12:
            categorias = pd.Series(valores).astype(str)
            ordem = sorted(categorias.unique())
            ax.boxplot([contribuicao[(categorias == c).to_numpy()] for c in ordem],
                       tick_labels=ordem, showfliers=False,
                       medianprops={"color": ACENTO, "linewidth": 1.6})
            ax.tick_params(axis="x", rotation=20)
        else:
            ax.scatter(valores, contribuicao, s=5, alpha=0.18, color=ACENTO, edgecolors="none")
        ax.axhline(0, color=CINZA_ESCURO, linewidth=0.9, linestyle=":")
        ax.set_xlabel(feature)
        ax.set_ylabel("contribuição SHAP (log-odds)")
        ax.set_title(feature, fontsize=10)
    fig.suptitle("Dependência das quatro variáveis principais — valor cru contra contribuição",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plotar_triangulacao(triangulacao: pd.DataFrame, n: int = 12) -> Figure:
    """Posição de cada feature nas três leituras, ligada por linha.

    Linha horizontal é consenso, linha inclinada é discordância — e é a
    discordância que o gráfico existe para mostrar. Quem sobe muito de uma
    coluna para a outra está sendo lido de formas diferentes por técnicas
    diferentes, o que costuma ser colinearidade ou cardinalidade, e não
    importância para o desfecho.
    """
    tabela = triangulacao.sort_values("posicao_shap").head(n)
    leituras = ["posicao_shap", "posicao_permutacao", "posicao_logistica"]
    rotulos = ["SHAP", "permutação", "logística podada"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for _, linha in tabela.iterrows():
        posicoes = [linha[c] for c in leituras]
        x = [i for i, p in enumerate(posicoes) if pd.notna(p)]
        y = [p for p in posicoes if pd.notna(p)]
        cor = COR_DA_FAMILIA[linha["familia"]]
        largura = 2.2 if linha["veredito"] == "fator" else 1.0
        ax.plot(x, y, "o-", color=cor, linewidth=largura, markersize=5,
                alpha=1.0 if linha["veredito"] == "fator" else 0.55)
        ax.annotate(linha["feature"], xy=(x[0] - 0.06, y[0]), ha="right", va="center", fontsize=8.5, color=cor)
        if len(x) > 1:
            ax.annotate(f"{y[-1]:.0f}º", xy=(x[-1] + 0.06, y[-1]), ha="left", va="center",
                        fontsize=8.5, color=CINZA_MEDIO)

    ax.set_xticks(range(len(leituras)), rotulos)
    ax.set_xlim(-1.5, 2.6)
    ax.set_yticks(range(1, int(tabela[leituras].max().max()) + 1))
    ax.invert_yaxis()
    ax.set_ylabel("posição no ranking (1 = mais importante)")
    ax.set_title("Triangulação — linha reta é consenso, linha inclinada é artefato de leitura")
    ax.grid(axis="x", visible=False)
    return fig
