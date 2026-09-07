"""Gráficos da Etapa 6. Mesma convenção das anteriores: quem mede é `src/modeling/strategic.py`.

Nenhuma função aqui recalcula risco, cluster ou probabilidade de meta. Tudo vem
dos CSV que `python -m src.modeling.strategic` grava, para que o número do slide
e o número da tabela não possam divergir.

A audiência destes gráficos é diferente da dos anteriores. Beeswarm de SHAP fala
com quem modela; um mapa de calor de composição por nível de proficiência fala
com quem assina o orçamento. Por isso quase todo gráfico aqui carrega o eixo em
unidade de negócio — crianças, pontos percentuais, municípios — e não em log-odds.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src import config
from src.eda import ACENTO, ACENTO_ALERTA, CINZA_CLARO, CINZA_ESCURO, CINZA_MEDIO

DIR_IMAGENS_ESTRATEGIA = config.DIR_IMAGES / "estrategia"

COR_DO_CLUSTER = {0: CINZA_CLARO, 1: CINZA_MEDIO, 2: ACENTO_ALERTA}


def salvar_figura(fig: Figure, nome: str) -> str:
    """Grava em `images/estrategia/` e devolve o caminho relativo à raiz."""
    DIR_IMAGENS_ESTRATEGIA.mkdir(parents=True, exist_ok=True)
    caminho = DIR_IMAGENS_ESTRATEGIA / f"{nome}.png"
    fig.savefig(caminho)
    return str(caminho.relative_to(config.ROOT)).replace("\\", "/")


def _rotulo(linha: pd.Series) -> str:
    return f"{linha['nome_municipio']} ({linha['sigla_uf']})"


def plotar_dois_rankings(ranking: pd.DataFrame, n: int = 20) -> Figure:
    """As duas listas lado a lado — a evidência de que responder uma não responde a outra.

    A escala do eixo x é diferente nos dois painéis porque as grandezas são
    diferentes: à esquerda a probabilidade média de uma criança não se
    alfabetizar, à direita quantas crianças isso representa. Forçar o mesmo eixo
    esconderia justamente o que o gráfico existe para mostrar.
    """
    avaliados = ranking[ranking["modelo_avaliado"]]
    por_taxa = avaliados.nsmallest(n, "posicao_por_taxa").iloc[::-1]
    por_volume = avaliados.nsmallest(n, "posicao_por_volume").iloc[::-1]

    fig, eixos = plt.subplots(1, 2, figsize=(13.5, 6.4))
    posicoes = np.arange(n)

    eixos[0].barh(posicoes, por_taxa["risco_suavizado"] * 100, color=ACENTO_ALERTA, height=0.7)
    eixos[0].set_yticks(posicoes, [_rotulo(l) for _, l in por_taxa.iterrows()], fontsize=8)
    eixos[0].set_xlabel("risco médio de não alfabetização (%)")
    eixos[0].set_title(f"Intensidade — os {n} municípios de maior risco", loc="left")
    for posicao, (_, linha) in zip(posicoes, por_taxa.iterrows()):
        eixos[0].text(
            linha["risco_suavizado"] * 100 + 0.6, posicao,
            f"{linha['n_alunos_avaliados']:.0f} alunos · presença {linha['taxa_presenca_2024']:.0%}",
            va="center", fontsize=7, color=CINZA_ESCURO,
        )
    eixos[0].set_xlim(0, avaliados["risco_suavizado"].max() * 100 * 1.45)

    eixos[1].barh(posicoes, por_volume["criancas_em_risco"], color=ACENTO, height=0.7)
    eixos[1].set_yticks(posicoes, [_rotulo(l) for _, l in por_volume.iterrows()], fontsize=8)
    eixos[1].set_xlabel("crianças em risco na população estimada")
    eixos[1].set_title(f"Escala — os {n} municípios com mais crianças em risco", loc="left")
    for posicao, (_, linha) in zip(posicoes, por_volume.iterrows()):
        eixos[1].text(
            linha["criancas_em_risco"] * 1.02, posicao,
            f"risco {linha['risco_suavizado']:.0%} · {linha['posicao_por_taxa']:.0f}º por taxa",
            va="center", fontsize=7, color=CINZA_ESCURO,
        )
    eixos[1].set_xlim(0, por_volume["criancas_em_risco"].max() * 1.55)

    comuns = set(por_taxa["id_municipio"]) & set(por_volume["id_municipio"])
    fig.suptitle(
        f"Municípios em comum entre as duas listas: {len(comuns)}",
        fontsize=10, color=ACENTO_ALERTA, x=0.01, ha="left",
    )
    fig.tight_layout()
    return fig


def plotar_taxa_versus_volume(ranking: pd.DataFrame, topo: int = 50) -> Figure:
    """Onde cada critério busca: risco no eixo y, tamanho da coorte no eixo x.

    O gráfico é o argumento inteiro da pergunta 2 em uma imagem. As duas seleções
    ocupam cantos opostos da nuvem, e nenhuma das duas cobre o que a outra pega.
    """
    avaliados = ranking[ranking["modelo_avaliado"]]
    em_taxa = avaliados["posicao_por_taxa"] <= topo
    em_volume = avaliados["posicao_por_volume"] <= topo

    fig, ax = plt.subplots(figsize=(9.5, 6))
    resto = avaliados[~(em_taxa | em_volume)]
    ax.scatter(
        resto["populacao_estimada"], resto["risco_suavizado"] * 100,
        s=6, alpha=0.18, color=CINZA_MEDIO, edgecolors="none", label="demais municípios",
    )
    ax.scatter(
        avaliados.loc[em_taxa, "populacao_estimada"], avaliados.loc[em_taxa, "risco_suavizado"] * 100,
        s=26, color=ACENTO_ALERTA, edgecolors="none", label=f"{topo} maiores por taxa",
    )
    ax.scatter(
        avaliados.loc[em_volume, "populacao_estimada"], avaliados.loc[em_volume, "risco_suavizado"] * 100,
        s=26, color=ACENTO, edgecolors="none", label=f"{topo} maiores por volume",
    )
    for _, linha in avaliados.nsmallest(4, "posicao_por_volume").iterrows():
        ax.annotate(
            linha["nome_municipio"], (linha["populacao_estimada"], linha["risco_suavizado"] * 100),
            textcoords="offset points", xytext=(-6, 8), fontsize=8, color=ACENTO, ha="right",
        )
    ax.set_xscale("log")
    ax.set_xlabel("população estimada de alunos (escala log)")
    ax.set_ylabel("risco médio de não alfabetização (%)")
    ax.set_title(
        "Intensidade e escala apontam para cantos opostos do país", loc="left"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def plotar_shrinkage(ranking: pd.DataFrame, shrinkage: pd.DataFrame) -> Figure:
    """Quanto cada suavização move o município, em função do porte.

    Os dois painéis usam o mesmo eixo y de propósito. É a comparação que mostra
    por que o `k` de 0,05 do escore do modelo e o `k` de 55 da taxa observada não
    são variações do mesmo número: a primeira suavização é uma linha reta em zero.
    """
    k_modelo = float(shrinkage.loc[shrinkage["quantidade"] == "escore do modelo", "k"].iloc[0])
    k_taxa = float(shrinkage.loc[shrinkage["quantidade"] == "taxa observada", "k"].iloc[0])
    limite = (
        max(
            (ranking["taxa_observada_ajustada"] - ranking["taxa_observada_de_risco"]).abs().max(),
            (ranking["risco_suavizado"] - ranking["risco_medio_oof"]).abs().max(),
        )
        * 100
        * 1.1
    )

    fig, eixos = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    painéis = (
        (eixos[0], "risco_suavizado", "risco_medio_oof",
         f"Escore do modelo — k = {k_modelo:.3f} aluno", ACENTO),
        (eixos[1], "taxa_observada_ajustada", "taxa_observada_de_risco",
         f"Taxa observada de 2024 — k = {k_taxa:.1f} alunos", ACENTO_ALERTA),
    )
    for ax, depois, antes, titulo, cor in painéis:
        ax.scatter(
            ranking["n_alunos_avaliados"], (ranking[depois] - ranking[antes]) * 100,
            s=6, alpha=0.25, color=cor, edgecolors="none",
        )
        ax.axhline(0, color=CINZA_ESCURO, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("alunos avaliados em 2024 (escala log)")
        ax.set_title(titulo, loc="left")
        ax.set_ylim(-limite, limite)
    eixos[0].set_ylabel("deslocamento do shrinkage (pontos percentuais)")
    fig.suptitle(
        "O shrinkage só tem o que corrigir onde há ruído amostral para corrigir",
        fontsize=10, x=0.01, ha="left",
    )
    fig.tight_layout()
    return fig


def plotar_risco_e_presenca(ranking: pd.DataFrame, n_bins: int = 20) -> Figure:
    """A ressalva que acompanha a lista: onde falta gente na prova, a taxa é otimista.

    O eixo y traz as duas medidas juntas. A taxa observada sobe conforme a
    presença sobe, o que é o viés de seleção que a EDA mediu; o risco do modelo
    acompanha em parte, porque ele também aprendeu com taxas otimistas.
    """
    avaliados = ranking[ranking["modelo_avaliado"]].copy()
    # `rank` antes do `qcut` porque 383 municípios têm presença exatamente 1,0 e
    # os cortes de quantil colidiriam no último bin.
    avaliados["bin"] = pd.qcut(
        avaliados["taxa_presenca_2024"].rank(method="first"), n_bins, labels=False
    )
    resumo = avaliados.groupby("bin").agg(
        presenca=("taxa_presenca_2024", "mean"),
        observado=("taxa_observada_de_risco", "mean"),
        modelo=("risco_suavizado", "mean"),
        n=("id_municipio", "size"),
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(resumo["presenca"] * 100, resumo["observado"] * 100, "o-",
            color=ACENTO_ALERTA, linewidth=2, markersize=5, label="taxa observada de risco")
    ax.plot(resumo["presenca"] * 100, resumo["modelo"] * 100, "s--",
            color=ACENTO, linewidth=2, markersize=4, label="risco médio do modelo")
    ax.set_xlabel("taxa de presença municipal na prova de 2024 (%)")
    ax.set_ylabel("risco de não alfabetização (%)")
    ax.set_title(
        "Municípios de baixa presença aparecem piores — e o problema real é maior que o medido",
        loc="left",
    )
    ax.legend()
    fig.tight_layout()
    return fig


def plotar_selecao_de_k(selecao: pd.DataFrame) -> Figure:
    """Silhueta e inércia na mesma figura — o critério e o cotovelo lado a lado."""
    melhor = int(selecao.loc[selecao["silhueta"].idxmax(), "k"])
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.6))

    eixos[0].plot(selecao["k"], selecao["silhueta"], "o-", color=ACENTO, linewidth=2)
    eixos[0].scatter([melhor], [selecao["silhueta"].max()], s=110, facecolors="none",
                     edgecolors=ACENTO_ALERTA, linewidths=2)
    eixos[0].set_xlabel("número de grupos (k)")
    eixos[0].set_ylabel("silhueta média")
    eixos[0].set_title(f"Silhueta — escolhido k = {melhor}", loc="left")
    eixos[0].set_ylim(0, max(0.35, selecao["silhueta"].max() * 1.25))
    eixos[0].axhline(0.25, color=CINZA_MEDIO, linestyle=":", linewidth=1)
    eixos[0].text(selecao["k"].max(), 0.255, "0,25 — separação fraca", ha="right",
                  fontsize=8, color=CINZA_MEDIO)

    eixos[1].plot(selecao["k"], selecao["inercia"], "o-", color=CINZA_ESCURO, linewidth=2)
    eixos[1].set_xlabel("número de grupos (k)")
    eixos[1].set_ylabel("inércia (soma dos quadrados intragrupo)")
    eixos[1].set_title("Cotovelo — sem quebra nítida", loc="left")
    fig.tight_layout()
    return fig


def plotar_perfil_dos_clusters(perfil: pd.DataFrame) -> Figure:
    """Composição por nível de proficiência e as médias que dão nome a cada grupo."""
    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.8))
    posicoes = np.arange(len(perfil))
    rotulos = [f"{linha.cluster_nome}\n{linha.n_municipios} municípios" for linha in perfil.itertuples()]

    esquerda = np.zeros(len(perfil))
    for coluna, cor, nome in (
        ("niveis_0_a_2", ACENTO_ALERTA, "níveis 0 a 2 — pré-alfabético"),
        ("niveis_3_a_5", CINZA_MEDIO, "níveis 3 a 5 — em consolidação"),
        ("niveis_6_a_8", ACENTO, "níveis 6 a 8 — alfabetizado"),
    ):
        eixos[0].barh(posicoes, perfil[coluna], left=esquerda, color=cor, height=0.6, label=nome)
        for posicao, valor, base in zip(posicoes, perfil[coluna], esquerda):
            if valor > 6:
                eixos[0].text(base + valor / 2, posicao, f"{valor:.0f}%", ha="center", va="center",
                              fontsize=9, color="white")
        esquerda = esquerda + perfil[coluna].to_numpy()
    eixos[0].set_yticks(posicoes, rotulos, fontsize=9)
    eixos[0].set_xlabel("distribuição dos alunos por nível de proficiência em 2024 (%)")
    eixos[0].set_title("Composição por nível", loc="left")
    eixos[0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3)

    largura = 0.27
    for deslocamento, coluna, cor, nome in (
        (-largura, "taxa_2023", CINZA_CLARO, "taxa de alfabetização 2023"),
        (0.0, "taxa_2024", CINZA_ESCURO, "taxa de alfabetização 2024"),
        (largura, "taxa_presenca", ACENTO, "presença na prova 2024"),
    ):
        eixos[1].bar(posicoes + deslocamento, perfil[coluna] * 100, largura, color=cor, label=nome)
    eixos[1].set_xticks(posicoes, [linha.cluster_nome for linha in perfil.itertuples()], fontsize=8)
    eixos[1].set_ylabel("%")
    eixos[1].set_title("Desempenho e cobertura", loc="left")
    eixos[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plotar_composicao_regional(composicao: pd.DataFrame, perfil: pd.DataFrame) -> Figure:
    """Onde cada grupo está. Barras empilhadas somam 100% dentro do cluster."""
    ordem = perfil.sort_values("cluster")["cluster_nome"].to_list()
    tabela = composicao.set_index("cluster_nome").loc[ordem]
    regioes = [c for c in tabela.columns]
    cores = [CINZA_CLARO, CINZA_MEDIO, CINZA_ESCURO, ACENTO, ACENTO_ALERTA][: len(regioes)]

    fig, ax = plt.subplots(figsize=(10, 4))
    posicoes = np.arange(len(tabela))
    esquerda = np.zeros(len(tabela))
    for regiao, cor in zip(regioes, cores):
        valores = tabela[regiao].to_numpy() * 100
        ax.barh(posicoes, valores, left=esquerda, color=cor, height=0.55, label=regiao)
        for posicao, valor, base in zip(posicoes, valores, esquerda):
            if valor > 8:
                ax.text(base + valor / 2, posicao, f"{valor:.0f}%", ha="center", va="center",
                        fontsize=8, color="white" if cor != CINZA_CLARO else CINZA_ESCURO)
        esquerda = esquerda + valores
    ax.set_yticks(posicoes, ordem, fontsize=9)
    ax.set_xlabel("composição regional do grupo (%)")
    ax.set_title("Os grupos não são regiões, mas quase", loc="left")
    ax.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    fig.tight_layout()
    return fig


def plotar_gap_de_esforco(metas: pd.DataFrame) -> Figure:
    """Distribuição de `meta_2025 − taxa_2024` — o número mais acionável da etapa."""
    gap = metas["gap_de_esforco_2025"]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.hist(gap.clip(-40, 40), bins=60, color=CINZA_MEDIO, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=CINZA_ESCURO, linewidth=1.2)
    ax.axvline(gap.median(), color=ACENTO_ALERTA, linestyle="--", linewidth=1.6)
    ax.text(gap.median() + 0.8, ax.get_ylim()[1] * 0.9,
            f"mediana {gap.median():+.1f} pp", color=ACENTO_ALERTA, fontsize=9)
    ax.text(-38, ax.get_ylim()[1] * 0.9,
            f"{(gap <= 0).mean():.1%} já superam\na meta de 2025", fontsize=9, color=CINZA_ESCURO)
    ax.set_xlabel("esforço exigido pela meta de 2025 (pontos percentuais sobre a taxa de 2024)")
    ax.set_ylabel("municípios")
    ax.set_title("O esforço mediano exigido é menor que o ruído anual de um município pequeno",
                 loc="left")
    fig.tight_layout()
    return fig


def plotar_incerteza_por_porte(volatilidade: pd.DataFrame, resumo: dict) -> Figure:
    """Oscilação anual medida contra a curva ajustada, com o componente constante ajustado marcado."""
    a = resumo["metas"]["persistencia"]["dispersao_a"]
    c = resumo["metas"]["persistencia"]["dispersao_c"]
    limiar = resumo["metas"]["persistencia"]["porte_referencia_dispersao"]
    gap = resumo["metas"]["gap_de_esforco_2025"]["mediana"]

    grade = np.logspace(np.log10(10), np.log10(5000), 200)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(grade, np.sqrt(a**2 / grade + c**2), color=ACENTO, linewidth=2,
            label=r"ajuste  $\sqrt{a^2/n + c^2}$")
    ax.scatter(volatilidade["porte_mediano"], volatilidade["evolucao_dp"], s=55,
               color=CINZA_ESCURO, zorder=3, label="desvio medido por faixa de porte")
    ax.axhline(c, color=CINZA_MEDIO, linestyle=":", linewidth=1.4)
    ax.text(11, c + 0.4, f"componente constante ajustado {c:.1f} pp", fontsize=8, color=CINZA_MEDIO)
    ax.axhline(gap, color=ACENTO_ALERTA, linestyle="--", linewidth=1.4)
    ax.text(11, gap + 0.4, f"esforço mediano exigido pela meta de 2025: {gap:.1f} pp",
            fontsize=8, color=ACENTO_ALERTA)
    ax.axvline(limiar, color=CINZA_ESCURO, linestyle="-.", linewidth=1)
    ax.text(limiar * 1.08, 24, f"{limiar:.0f} alunos —\nreferência de dispersão, sem corte de elegibilidade",
            fontsize=8, color=CINZA_ESCURO)
    ax.set_xscale("log")
    ax.set_xlabel("alunos avaliados no município (escala log)")
    ax.set_ylabel("desvio-padrão da variação anual (pontos percentuais)")
    ax.set_title("Dispersão residual observada e ajuste na transição 2023–2024", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plotar_validacao_metas(validacao: pd.DataFrame, metas: pd.DataFrame, resumo: dict) -> Figure:
    """Calibração fora do fold e a distribuição das probabilidades de 2025.

    Os dois painéis dizem a mesma coisa por caminhos diferentes: à esquerda, a
    previsão quase não separa quem ficou abaixo da meta de quem não ficou; à
    direita, a probabilidade se espalha por toda a faixa, e o que a resume é
    quantos municípios têm a própria meta dentro do intervalo de 95%.

    O painel da esquerda mede a mesma transição 2023 → 2024 em que os
    parâmetros foram ajustados, com municípios fora do fold. É generalização
    territorial, não desempenho num ano futuro, e o título diz isso.
    """
    auc = resumo["metas"]["validacao_municipal_2023_2024"]["roc_auc"]["projecao_com_shrinkage_e_deriva"]
    dentro = resumo["metas"]["share_com_meta_dentro_do_intervalo_de_95"]

    fig, eixos = plt.subplots(1, 2, figsize=(12.5, 5))
    eixos[0].plot([0, 1], [0, 1], color=CINZA_MEDIO, linestyle=":", linewidth=1.2,
                  label="calibração perfeita")
    eixos[0].plot(validacao["probabilidade_prevista"], validacao["fracao_observada"], "o-",
                  color=ACENTO_ALERTA, linewidth=2, markersize=5, label="decis fora do ajuste")
    eixos[0].set_xlim(0.15, 0.75)
    eixos[0].set_ylim(0.15, 0.75)
    eixos[0].set_xlabel("probabilidade prevista de ficar abaixo da meta de 2024")
    eixos[0].set_ylabel("fração que de fato ficou")
    eixos[0].set_title(f"OOF por município: 2023 → 2024 — ROC-AUC {auc:.4f}".replace(".", ","), loc="left")
    eixos[0].legend(fontsize=9)

    eixos[1].hist(
        [
            metas.loc[metas["meta_dentro_do_intervalo"], "p_abaixo_da_meta_2025"],
            metas.loc[~metas["meta_dentro_do_intervalo"], "p_abaixo_da_meta_2025"],
        ],
        bins=40, stacked=True, color=[CINZA_MEDIO, ACENTO], linewidth=0.4, edgecolor="white",
        label=[
            f"meta dentro do intervalo de 95% — {dentro:.1%}",
            f"meta fora do intervalo — {1 - dentro:.1%}",
        ],
    )
    eixos[1].axvline(0.5, color=CINZA_ESCURO, linewidth=1)
    eixos[1].set_xlabel("P(taxa de 2025 < meta de 2025)")
    eixos[1].set_ylabel("municípios")
    eixos[1].set_title("A resposta é uma probabilidade, nunca um rótulo", loc="left")
    eixos[1].legend(fontsize=9, loc="upper center")
    fig.tight_layout()
    return fig


def plotar_funil(funil: pd.DataFrame) -> Figure:
    """Quantos municípios cada meta anual já encontra atingida com a taxa de 2024."""
    fig, ax = plt.subplots(figsize=(9.5, 5))
    posicoes = np.arange(len(funil))
    ax.bar(posicoes, funil["municipios_que_ja_atingem_com_a_taxa_de_2024"],
           0.6, color=CINZA_MEDIO)
    for posicao, linha in zip(posicoes, funil.itertuples()):
        ax.text(posicao, linha.municipios_que_ja_atingem_com_a_taxa_de_2024 + 40,
                f"{linha.share_que_ja_atinge:.0%}", ha="center", fontsize=9, color=CINZA_ESCURO)
        ax.text(posicao, 60, f"faltam\n{linha.gap_mediano:.1f} pp", ha="center", fontsize=8,
                color="white")
    ax.set_xticks(posicoes, [f"{linha.ano}\nmeta mediana {linha.meta_mediana:.0f}%"
                             for linha in funil.itertuples()], fontsize=9)
    ax.set_ylabel("municípios que já atingiriam a meta com a taxa de 2024")
    ax.set_title("O funil até 2030, sem projetar nada: só a régua ficando mais alta", loc="left")
    fig.tight_layout()
    return fig
