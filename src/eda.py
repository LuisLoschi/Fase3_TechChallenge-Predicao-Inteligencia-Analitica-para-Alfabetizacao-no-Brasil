"""Medições da análise exploratória.

Escrevi cada função daqui para fechar uma decisão de engenharia de features ou de modelagem. Se a saída
de uma delas não muda uma escolha de feature, de validação ou de escopo, ela não
deveria existir. Foi o critério que usei para não encher o notebook de gráfico
decorativo.

O notebook `notebooks/01_eda.ipynb` conta a história; a conta mora aqui, para
que qualquer número do relatório possa ser refeito sem abrir o Jupyter e travado
por teste depois.

A convenção do módulo: `medir_*` devolve número e nunca desenha, `plotar_*`
recebe número já medido e devolve uma `Figure`, `salvar_figura` grava em
`images/eda/`. Manter as duas coisas separadas é o que permite testar a medição
sem renderizar nada.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src import config

# ---------------------------------------------------------------------------
# Estilo: o material é impresso e projetado, então cor não pode carregar sentido
# ---------------------------------------------------------------------------
DIR_IMAGENS_EDA = config.DIR_IMAGES / "eda"

CINZA_ESCURO = "#2b2b2b"
CINZA_MEDIO = "#7a7a7a"
CINZA_CLARO = "#c9c9c9"
ACENTO = "#1f6feb"
ACENTO_ALERTA = "#b3261e"


def aplicar_estilo() -> None:
    """Fixa fonte, grade e tamanho uma vez, no topo do notebook.

    A paleta é cinza com um único acento porque estes gráficos terminam em slide
    projetado e em PDF impresso, onde cor não sobrevive. Quem precisa se destacar
    se destaca por hachura ou por rótulo.
    """
    matplotlib.rcParams.update(
        {
            "figure.figsize": (9, 5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": CINZA_CLARO,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.frameon": False,
        }
    )


def salvar_figura(fig: Figure, nome: str) -> str:
    """Grava em `images/eda/` e devolve o caminho relativo à raiz do repositório.

    Relativo e não absoluto: assim o que o notebook imprime é exatamente o que o
    README consegue referenciar em outra máquina.
    """
    DIR_IMAGENS_EDA.mkdir(parents=True, exist_ok=True)
    caminho = DIR_IMAGENS_EDA / f"{nome}.png"
    fig.savefig(caminho)
    return str(caminho.relative_to(config.ROOT)).replace("\\", "/")


def formatar_milhar(valor: float) -> str:
    """Separador de milhar em português, para rótulo de eixo e texto corrido."""
    return f"{valor:,.0f}".replace(",", ".")


# ---------------------------------------------------------------------------
# Utilitários estatísticos
# ---------------------------------------------------------------------------
def ic_wilson(sucessos, n, z: float = 1.96):
    """Intervalo de Wilson para proporção.

    Toda taxa deste projeto sai daqui com intervalo. Sem ele não dá para separar
    diferença real de flutuação amostral, e boa parte das comparações da EDA,
    entre cadernos, redes e UFs, vive exatamente nessa fronteira.

    Wilson e não o intervalo normal porque a base tem muita unidade pequena com
    taxa perto de 0 ou de 1, onde o normal devolve limite fora de [0, 1].
    """
    k = np.asarray(sucessos, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.full(np.shape(k), np.nan, dtype=float), where=n > 0)
    denominador = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / denominador
    meia_largura = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominador
    return centro - meia_largura, centro + meia_largura


def auc_univariada(y, escore) -> float:
    """ROC-AUC de um preditor isolado, pela estatística U de Mann-Whitney.

    Uso isto antes de qualquer tuning. Feature que sozinha fica em 0,5 raramente
    vira sinal dentro de um GBM, e descobrir isso custa um ranking de postos em
    vez de uma tarde de busca de hiperparâmetro.

    Serve também na direção oposta: variável isolada acima de 0,90 é vazamento
    até prova em contrário. Foi assim que a `proficiencia` do ano corrente
    entregou 1,000 exato e se denunciou sozinha.

    Calculo por postos e não com `roc_auc_score` porque preciso rodar isto sobre
    1,85 milhão de linhas dezenas de vezes, tolerando nulo, sem montar a curva.
    """
    y = np.asarray(y)
    escore = np.asarray(escore, dtype=float)
    valido = ~np.isnan(escore)
    y, escore = y[valido], escore[valido]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    postos = pd.Series(escore).rank().to_numpy()
    return float((postos[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ic_bootstrap_correlacao(x, y, n_reamostras: int = 1000, seed: int = config.RANDOM_STATE):
    """IC percentil de 95% da correlação de Pearson, reamostrando as unidades.

    A persistência municipal sustenta a camada estratégica inteira e vai virar
    slide. Publicá-la como `0,645` seco convidaria alguém a comparar com o
    `0,636` do diagnóstico inicial e concluir que algo piorou, quando os dois cabem no mesmo
    intervalo.

    Reamostro município, não aluno: é o município que é a unidade independente
    aqui, e reamostrar aluno estreitaria o intervalo artificialmente.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valido = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valido], y[valido]
    n = len(x)
    amostras = np.empty(n_reamostras)
    for i in range(n_reamostras):
        idx = rng.integers(0, n, n)
        amostras[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    r = float(np.corrcoef(x, y)[0, 1])
    return r, float(np.percentile(amostras, 2.5)), float(np.percentile(amostras, 97.5))


# ---------------------------------------------------------------------------
# 1-2. Estrutura, cardinalidade e alvo
# ---------------------------------------------------------------------------
def medir_estrutura(df: pd.DataFrame) -> pd.DataFrame:
    """Perfil coluna a coluna: tipo, cardinalidade, nulos, memória e um exemplo.

    Primeira triagem da lista de features. Cardinalidade 1 sai na hora.
    Cardinalidade próxima do número de linhas é identificador, e identificador
    nesta base esconde território no prefixo, então não basta ignorá-lo. O
    percentual de nulos decide quem precisa de `add_indicator`.

    Trago um valor de exemplo em cada linha porque tipo e cardinalidade não
    revelam que `preenchimento_caderno` chega como `float32` valendo 0 ou 1.
    """
    linhas = []
    for coluna in df.columns:
        s = df[coluna]
        linhas.append(
            {
                "coluna": coluna,
                "dtype": str(s.dtype),
                "n_unicos": int(s.nunique(dropna=True)),
                "pct_nulo": float(s.isna().mean()),
                "memoria_mb": float(s.memory_usage(deep=False) / 2**20),
                "exemplo": s.dropna().iloc[0] if s.notna().any() else None,
            }
        )
    return pd.DataFrame(linhas).sort_values("n_unicos").reset_index(drop=True)


def medir_target_por_ano(alvo_por_ano: dict[int, pd.Series]) -> pd.DataFrame:
    """Prevalência do alvo em cada ano, com IC de Wilson.

    Aqui se decide se haverá reamostragem. Numa base 40/60 o SMOTE não corrige
    desequilíbrio nenhum: só desloca a probabilidade predita, e é a probabilidade
    calibrada que o ranking municipal de risco consome.

    Separo por ano porque, com dois anos de histórico, a comparação 2023 × 2024 é
    o único sinal de drift que existe.
    """
    linhas = []
    for ano, serie in alvo_por_ano.items():
        n = int(serie.notna().sum())
        k = int(serie.sum())
        lo, hi = ic_wilson(k, n)
        linhas.append(
            {
                "ano": ano,
                "n_alunos": n,
                "n_alfabetizados": k,
                "taxa_alfabetizado": k / n,
                "taxa_risco": 1 - k / n,
                "ic_low": float(lo),
                "ic_high": float(hi),
            }
        )
    return pd.DataFrame(linhas)


def plotar_target_por_ano(resumo: pd.DataFrame) -> Figure:
    """Barra empilhada da prevalência, com o n de cada ano no eixo.

    Empilhada e não lado a lado porque a pergunta é como a coorte se parte, não
    como dois níveis se comparam. Quem olha precisa ver que a fatia menor é 40% e
    não 5%, é o que encerra a discussão sobre reamostragem.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(resumo))
    ax.bar(x, resumo["taxa_alfabetizado"], 0.55, color=CINZA_MEDIO, label="alfabetizado")
    ax.bar(
        x,
        resumo["taxa_risco"],
        0.55,
        bottom=resumo["taxa_alfabetizado"],
        color=CINZA_ESCURO,
        hatch="///",
        edgecolor="white",
        label="não alfabetizado (classe de interesse)",
    )
    for i, linha in resumo.reset_index(drop=True).iterrows():
        ax.text(i, linha["taxa_alfabetizado"] / 2, f"{linha['taxa_alfabetizado']:.1%}", ha="center", color="white", fontweight="bold")
        ax.text(i, linha["taxa_alfabetizado"] + linha["taxa_risco"] / 2, f"{linha['taxa_risco']:.1%}", ha="center", color="white", fontweight="bold")
    ax.set_xticks(x, [f"{a}\n(n = {formatar_milhar(n)} alunos)" for a, n in zip(resumo["ano"], resumo["n_alunos"])])
    ax.set_ylabel("proporção dos alunos presentes")
    ax.set_ylim(0, 1.2)
    ax.set_title("Base balanceada nos dois anos (~40% em risco): reamostragem é desnecessária")
    ax.legend(loc="upper center", ncols=2)
    return fig


# ---------------------------------------------------------------------------
# 3-5. Taxa por categoria (caderno, região/UF, rede)
# ---------------------------------------------------------------------------
def medir_taxa_por_categoria(df: pd.DataFrame, coluna: str, alvo: str = "alfabetizado") -> pd.DataFrame:
    """Taxa do alvo por nível de uma categórica, com n e IC de Wilson.

    Teste de utilidade de cada categórica. O `n` vem junto por um motivo prático:
    nível raro produz taxa extrema que não significa nada, e a rede Privada, com
    24 alunos e intervalo de 35 pontos de largura, é o lembrete disso.
    """
    agrupado = df.groupby(coluna, observed=True)[alvo].agg(n="size", k="sum")
    agrupado["taxa"] = agrupado["k"] / agrupado["n"]
    lo, hi = ic_wilson(agrupado["k"].to_numpy(), agrupado["n"].to_numpy())
    agrupado["ic_low"], agrupado["ic_high"] = lo, hi
    agrupado["amplitude_ic"] = agrupado["ic_high"] - agrupado["ic_low"]
    return agrupado.reset_index()


def medir_sobreposicao_ic(resumo: pd.DataFrame) -> dict:
    """Procura um valor que esteja dentro do intervalo de confiança de todos os níveis.

    A função nasceu para confirmar o plano: se todos os ICs do `caderno`
    cobrissem um ponto comum, estaria pronto o argumento limpo de que a variável
    é ruído. Ela devolve `False`. Com 85 mil alunos por caderno os intervalos são
    estreitos demais para se encontrarem, e o critério de sobreposição perde o
    sentido nessa escala.

    A amplitude entre taxas volta junto com o veredito porque é ela, 1,75pp
    contra 49pp entre UFs, que sustenta a decisão que a sobreposição não
    sustentou.
    """
    limite_inferior = float(resumo["ic_low"].max())
    limite_superior = float(resumo["ic_high"].min())
    return {
        "n_niveis": int(len(resumo)),
        "taxa_min": float(resumo["taxa"].min()),
        "taxa_max": float(resumo["taxa"].max()),
        "amplitude_taxas_pp": float((resumo["taxa"].max() - resumo["taxa"].min()) * 100),
        "maior_ic_low": limite_inferior,
        "menor_ic_high": limite_superior,
        "todos_ics_se_sobrepoem": limite_inferior <= limite_superior,
    }


def plotar_taxa_por_categoria(
    resumo: pd.DataFrame,
    coluna: str,
    titulo: str,
    rotulo_x: str = "taxa de alfabetização (%)",
    ordenar: bool = True,
    linha_referencia: float | None = None,
    destacar: list | None = None,
) -> Figure:
    """Barras horizontais com IC de 95%, ordenadas pela taxa.

    Horizontal porque são 26 UFs e nome deitado não se lê. O `n` entra no rótulo
    do eixo para que ninguém compare a barra de 24 alunos com a de 1,6 milhão sem
    perceber, e a linha de referência é a média nacional, a única âncora que o
    leitor de política pública já traz de casa.
    """
    dados = resumo.sort_values("taxa") if ordenar else resumo.iloc[::-1]
    altura = max(3.2, 0.30 * len(dados) + 1.4)
    fig, ax = plt.subplots(figsize=(9, altura))
    cores = [ACENTO_ALERTA if destacar and rotulo in destacar else CINZA_MEDIO for rotulo in dados[coluna]]
    y = np.arange(len(dados))
    erro = np.vstack(
        [(dados["taxa"] - dados["ic_low"]).to_numpy() * 100, (dados["ic_high"] - dados["taxa"]).to_numpy() * 100]
    )
    ax.barh(y, dados["taxa"] * 100, color=cores, xerr=erro, error_kw={"ecolor": CINZA_ESCURO, "elinewidth": 1, "capsize": 2})
    ax.set_yticks(y, [f"{v}  (n={formatar_milhar(n)})" for v, n in zip(dados[coluna], dados["n"])])
    ax.set_xlabel(rotulo_x)
    ax.set_title(titulo)
    if linha_referencia is not None:
        ax.axvline(linha_referencia * 100, color=ACENTO_ALERTA, linestyle="--", linewidth=1.2)
        ax.text(linha_referencia * 100 + 0.4, -0.9, f"média nacional {linha_referencia:.1%}", fontsize=8, color=ACENTO_ALERTA)
    return fig


# ---------------------------------------------------------------------------
# 6. Nulos
# ---------------------------------------------------------------------------
def medir_nulos_por_ano(bases: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Percentual de nulos por coluna, quebrado por ano.

    O corte por ano é o ponto. Empilhando os dois anos as metas apareciam com 53%
    de nulo, o que parece falha de coleta; separadas, são 100% em 2023 e 15% em
    2024, e a leitura vira outra: a variável não existia, o INEP passou a
    publicá-la depois.

    A distinção decide a imputação: falta estrutural pede `add_indicator=True`
    porque a ausência informa; falha aleatória de coleta não pede.
    """
    quadro = pd.DataFrame({nome: df.isna().mean() for nome, df in bases.items()})
    quadro.index.name = "coluna"
    return quadro.reset_index()


def plotar_mapa_nulos(quadro: pd.DataFrame, titulo: str) -> Figure:
    dados = quadro.set_index("coluna")
    dados = dados.loc[dados.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.32 * len(dados) + 1.2)))
    imagem = ax.imshow(dados.to_numpy() * 100, aspect="auto", cmap="Greys", vmin=0, vmax=100)
    ax.set_xticks(range(dados.shape[1]), dados.columns)
    ax.set_yticks(range(len(dados)), dados.index, fontsize=9)
    ax.grid(False)
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            valor = dados.iat[i, j] * 100
            ax.text(j, i, f"{valor:.0f}%", ha="center", va="center", fontsize=8, color="white" if valor > 55 else CINZA_ESCURO)
    fig.colorbar(imagem, ax=ax, label="% de nulos")
    ax.set_title(titulo)
    return fig


# ---------------------------------------------------------------------------
# 7. Porte de escola e município
# ---------------------------------------------------------------------------
def medir_porte(df: pd.DataFrame, chave: str) -> pd.DataFrame:
    """Distribuição do número de alunos por escola ou por município.

    Devolvo o erro-padrão da taxa na mediana junto com os percentis porque é ele
    que traduz porte em consequência: a escola mediana avaliou 32 alunos, o que
    dá 8,8pp de erro-padrão numa taxa perto de 0,5, quase um quinto da amplitude
    entre UFs.

    Era a justificativa aritmética da suavização empírico-bayesiana. A aritmética
    continua certa; o item 16 mostra que a conclusão que tirei dela não era.
    """
    contagem = df.groupby(chave, observed=True).size()
    percentis = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    resumo = {"chave": chave, "n_unidades": int(contagem.size), "media": float(contagem.mean())}
    resumo.update({f"p{p}": float(contagem.quantile(p / 100)) for p in percentis})
    resumo["erro_padrao_da_taxa_na_mediana_pp"] = float(100 * np.sqrt(0.25 / contagem.median()))
    return pd.DataFrame([resumo])


def plotar_porte(contagem_escola: pd.Series, contagem_municipio: pd.Series) -> Figure:
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
    for ax, contagem, nome in zip(eixos, [contagem_escola, contagem_municipio], ["escola", "município"]):
        ax.hist(np.log10(contagem.clip(lower=1)), bins=50, color=CINZA_MEDIO, edgecolor="white", linewidth=0.4)
        mediana = float(contagem.median())
        ax.axvline(np.log10(mediana), color=ACENTO_ALERTA, linestyle="--", linewidth=1.4)
        ax.text(np.log10(mediana) + 0.05, ax.get_ylim()[1] * 0.88, f"mediana = {mediana:.0f}", color=ACENTO_ALERTA, fontsize=9)
        ax.set_xlabel(f"alunos avaliados por {nome} em 2023 (escala log10)")
        ax.set_ylabel(f"nº de {nome}s")
        ax.set_title(f"{nome.capitalize()}: metade tem ate {mediana:.0f} alunos avaliados")
    fig.suptitle("Unidades pequenas dominam: a taxa observada de escola é ruidosa por construção", y=1.03, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# 8 / 17. Persistência temporal
# ---------------------------------------------------------------------------
def agregar_taxa(df: pd.DataFrame, chave: str, alvo: str = "alfabetizado") -> pd.DataFrame:
    """Taxa e volume por unidade, no formato que `medir_persistencia` consome.

    Devolvo `n` sempre junto da taxa porque toda leitura de persistência depois
    filtra por porte, e recalcular a contagem lá dentro seria repetir o mesmo
    groupby sobre 1,7 milhão de linhas.
    """
    saida = df.groupby(chave, observed=True)[alvo].agg(n="size", taxa="mean")
    return saida.reset_index()


def medir_persistencia(
    lag: pd.DataFrame,
    atual: pd.DataFrame,
    chave: str,
    faixas_porte: tuple[int, ...] = (0, 20, 50, 200),
) -> pd.DataFrame:
    """`corr(taxa do ano anterior, taxa do ano corrente)`, global e por faixa de porte.

    Mede quanto o passado prediz o presente, que é a pergunta que decide se o
    projeto existe. A quebra por porte serve para diagnosticar a causa quando a
    correlação é baixa: subindo com o porte, o limite é ruído amostral e a
    suavização resolve; estável ou caindo, o limite é outro e a suavização é
    cosmética.

    Vale para município. No grão escola a interpretação desmorona, e não por
    causa do porte: `id_escola` é reciclado entre edições (item 13b), então o
    par que esta função forma ali quase nunca é a mesma escola.
    """
    junto = lag.merge(atual, on=chave, suffixes=("_lag", "_atual"), validate="one_to_one")
    linhas = []
    for minimo in faixas_porte:
        recorte = junto[(junto["n_lag"] >= minimo) & (junto["n_atual"] >= minimo)]
        if len(recorte) < 30:
            continue
        r, lo, hi = ic_bootstrap_correlacao(recorte["taxa_lag"], recorte["taxa_atual"])
        linhas.append(
            {
                "chave": chave,
                "porte_minimo": minimo,
                "n_unidades": len(recorte),
                "corr_pearson": r,
                "ic_low": lo,
                "ic_high": hi,
                "corr_spearman": float(recorte["taxa_lag"].corr(recorte["taxa_atual"], method="spearman")),
            }
        )
    return pd.DataFrame(linhas)


def plotar_persistencia(
    lag: pd.DataFrame, atual: pd.DataFrame, chave: str, titulo: str, porte_minimo: int = 0
) -> Figure:
    junto = lag.merge(atual, on=chave, suffixes=("_lag", "_atual"), validate="one_to_one")
    junto = junto[(junto["n_lag"] >= porte_minimo) & (junto["n_atual"] >= porte_minimo)]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(junto["taxa_lag"] * 100, junto["taxa_atual"] * 100, s=4, alpha=0.12, color=CINZA_ESCURO, edgecolors="none")
    ax.plot([0, 100], [0, 100], color=CINZA_MEDIO, linestyle=":", linewidth=1.2, label="identidade (sem mudança)")
    coef = np.polyfit(junto["taxa_lag"], junto["taxa_atual"], 1)
    grade = np.linspace(junto["taxa_lag"].min(), junto["taxa_lag"].max(), 50)
    ax.plot(grade * 100, np.polyval(coef, grade) * 100, color=ACENTO_ALERTA, linewidth=2, label=f"ajuste (inclinação {coef[0]:.2f})")
    r = float(junto["taxa_lag"].corr(junto["taxa_atual"]))
    ax.set_xlabel("taxa de alfabetização em 2023 (%)")
    ax.set_ylabel("taxa de alfabetização em 2024 (%)")
    ax.set_title(f"{titulo}\nr = {r:.3f} · n = {formatar_milhar(len(junto))} unidades")
    ax.legend(loc="lower right")
    return fig


def plotar_persistencia_por_porte(
    medidas: pd.DataFrame, titulo: str, rotulo_y: str = "corr(taxa 2023, taxa 2024)", limite_inferior: float = 0.0
) -> Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(medidas))
    erro = np.vstack(
        [(medidas["corr_pearson"] - medidas["ic_low"]).to_numpy(), (medidas["ic_high"] - medidas["corr_pearson"]).to_numpy()]
    )
    ax.bar(x, medidas["corr_pearson"], 0.55, color=CINZA_MEDIO, yerr=erro, error_kw={"ecolor": CINZA_ESCURO, "capsize": 3})
    for i, linha in medidas.reset_index(drop=True).iterrows():
        ax.text(i, linha["ic_high"] + 0.03, f"{linha['corr_pearson']:.3f}", ha="center", fontweight="bold")
    ax.set_xticks(
        x, [f"≥ {int(m)} alunos\n({formatar_milhar(n)} unidades)" for m, n in zip(medidas["porte_minimo"], medidas["n_unidades"])]
    )
    ax.set_ylabel(rotulo_y)
    ax.set_ylim(limite_inferior, 1.0)
    ax.axhline(0, color=CINZA_ESCURO, linewidth=0.8)
    ax.set_title(titulo)
    return fig


# ---------------------------------------------------------------------------
# 9. Multicolinearidade
# ---------------------------------------------------------------------------
def medir_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF pela diagonal da inversa da matriz de correlação, sem statsmodels.

    Duas coisas dependem disto. A regressão logística de baseline, onde
    colinearidade infla erro-padrão e chega a inverter o sinal do coeficiente; e
    a leitura de SHAP, onde features correlacionadas diluem
    importância entre si e produzem uma narrativa errada sobre o que o modelo
    usou.

    Caio em `pinv` quando a matriz é singular em vez de estourar, porque
    singularidade aqui é resultado e não erro: quando o VIF vem na casa de 10¹³ e
    alguns valores saem negativos, isso significa que existe combinação linear
    exata entre as colunas: `iqr = p75 - p25`, `desvio = taxa_esc - taxa_mun`.
    Um `LinAlgError` esconderia esse diagnóstico.
    """
    dados = X.dropna()
    correlacao = dados.corr().to_numpy()
    try:
        inversa = np.linalg.inv(correlacao)
    except np.linalg.LinAlgError:
        inversa = np.linalg.pinv(correlacao)
    return (
        pd.DataFrame({"feature": dados.columns, "vif": np.diag(inversa)})
        .sort_values("vif", ascending=False)
        .reset_index(drop=True)
    )


def sugerir_poda_por_correlacao(correlacao: pd.DataFrame, limiar: float = 0.95) -> pd.DataFrame:
    """Pares de features acima do limiar de correlação, ordenados pela intensidade.

    Só metade superior da matriz: listar (a, b) e (b, a) dobraria a tabela sem
    acrescentar par nenhum a podar.
    """
    superior = correlacao.where(np.triu(np.ones(correlacao.shape), k=1).astype(bool))
    pares = superior.stack()
    pares = pares[pares.abs() >= limiar]
    pares = pares.reindex(pares.abs().sort_values(ascending=False).index)
    saida = pares.rename("correlacao").reset_index()
    saida.columns = ["feature_a", "feature_b", "correlacao"]
    return saida


def plotar_heatmap_correlacao(correlacao: pd.DataFrame, titulo: str) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 8.5))
    imagem = ax.imshow(correlacao.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlacao)), correlacao.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(correlacao)), correlacao.index, fontsize=8)
    ax.grid(False)
    for i in range(len(correlacao)):
        for j in range(len(correlacao)):
            valor = correlacao.iat[i, j]
            if abs(valor) >= 0.75 and i != j:
                ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=6.5, color="white")
    fig.colorbar(imagem, ax=ax, label="correlação de Spearman", shrink=0.75)
    ax.set_title(titulo)
    return fig


# ---------------------------------------------------------------------------
# 10. O alvo é a proficiência recortada (A1)
# ---------------------------------------------------------------------------
def medir_corte_proficiencia(df: pd.DataFrame) -> dict:
    """Distância entre o máximo da classe 0 e o mínimo da classe 1.

    Correlação não resolveria esta pergunta: 0,80 entre proficiência e alvo é
    compatível tanto com "relação forte" quanto com "é a mesma variável". O que
    decide é a sobreposição. Se não existe um único aluno da classe 0 acima do
    menor aluno da classe 1, o alvo é a proficiência recortada, e ponto.

    A consequência é dupla e vai nos dois sentidos: a proficiência do ano
    corrente sai das features porque é o alvo reescrito, e a de 2023 vira a
    melhor matéria-prima de feature que a base tem.
    """
    dados = df.dropna(subset=["proficiencia"])
    maximo_classe_0 = float(dados.loc[dados["alfabetizado"] == 0, "proficiencia"].max())
    minimo_classe_1 = float(dados.loc[dados["alfabetizado"] == 1, "proficiencia"].min())
    return {
        "n": int(len(dados)),
        "max_proficiencia_classe_0": maximo_classe_0,
        "min_proficiencia_classe_1": minimo_classe_1,
        "limiar_declarado": config.LIMIAR_ALFABETIZACAO,
        "ha_sobreposicao": bool(maximo_classe_0 >= minimo_classe_1),
        "corr_pearson_com_alvo": float(dados["proficiencia"].corr(dados["alfabetizado"])),
        "auc_univariada": auc_univariada(dados["alfabetizado"].to_numpy(), dados["proficiencia"].to_numpy()),
    }


def plotar_corte_proficiencia(df: pd.DataFrame, medida: dict, amostra: int = 300_000) -> Figure:
    dados = df.dropna(subset=["proficiencia"])
    if len(dados) > amostra:
        dados = dados.sample(amostra, random_state=config.RANDOM_STATE)
    fig, eixos = plt.subplots(1, 2, figsize=(11.5, 4.2))
    bins = np.linspace(float(dados["proficiencia"].min()), float(dados["proficiencia"].max()), 120)
    eixos[0].hist(dados.loc[dados["alfabetizado"] == 0, "proficiencia"], bins=bins, color=CINZA_ESCURO, label="alfabetizado = 0")
    eixos[0].hist(dados.loc[dados["alfabetizado"] == 1, "proficiencia"], bins=bins, color=CINZA_CLARO, label="alfabetizado = 1")
    eixos[0].axvline(config.LIMIAR_ALFABETIZACAO, color=ACENTO_ALERTA, linestyle="--", linewidth=1.4)
    eixos[0].set_xlabel("proficiência (pontos da escala do INEP)")
    eixos[0].set_ylabel(f"nº de alunos (amostra de {formatar_milhar(len(dados))})")
    eixos[0].set_title("As duas classes não se sobrepõem em ponto algum")
    eixos[0].legend()

    janela = dados[dados["proficiencia"].between(config.LIMIAR_ALFABETIZACAO - 3, config.LIMIAR_ALFABETIZACAO + 3)]
    ruido = np.random.default_rng(config.RANDOM_STATE).normal(0, 0.03, len(janela))
    eixos[1].scatter(janela["proficiencia"], janela["alfabetizado"] + ruido, s=3, alpha=0.25, color=CINZA_ESCURO, edgecolors="none")
    eixos[1].axvline(config.LIMIAR_ALFABETIZACAO, color=ACENTO_ALERTA, linestyle="--", linewidth=1.4)
    eixos[1].set_xlabel("proficiência (pontos), janela de ±3 pontos em torno do corte")
    eixos[1].set_ylabel("alfabetizado")
    eixos[1].set_yticks([0, 1])
    eixos[1].set_title(f"Corte determinístico em {config.LIMIAR_ALFABETIZACAO:.1f} pontos")
    fig.suptitle(
        f"O alvo é a proficiência recortada: max(classe 0) = {medida['max_proficiencia_classe_0']:.4f}"
        f" < {medida['min_proficiencia_classe_1']:.4f} = min(classe 1)",
        y=1.04,
        fontweight="bold",
    )
    return fig


# ---------------------------------------------------------------------------
# 11 / 15. Metas municipais (A4)
# ---------------------------------------------------------------------------
def medir_gap_meta(metas: pd.DataFrame, taxa_observada: pd.DataFrame, coluna_meta: str = "meta_alfabetizacao_2024") -> pd.DataFrame:
    """Diferença, em pontos percentuais, entre a taxa observada de 2024 e a meta do ano.

    Aritmética pura, sem modelo, e é justamente por isso que este é o número
    mais confiável da camada estratégica municipal.4. Depois de descobrir que a meta é derivada da
    própria taxa base, prever "quem não bate a meta" virou previsão de ruído; o
    que sobra de acionável é dizer a cada município quantos pontos faltam.

    `validate="one_to_one"` porque a base de metas tem uma linha por
    município-ano; se algum dia ganhar grão por rede, quero que quebre aqui.
    """
    junto = metas.merge(taxa_observada, on="id_municipio", how="inner", validate="one_to_one")
    junto["gap_pp"] = junto["taxa_2024_pct"] - junto[coluna_meta]
    junto["atingiu_meta"] = junto["gap_pp"] >= 0
    return junto


def plotar_taxa_versus_meta(gap: pd.DataFrame, coluna_meta: str = "meta_alfabetizacao_2024") -> Figure:
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    atingiu = gap["atingiu_meta"]
    ax.scatter(gap.loc[atingiu, coluna_meta], gap.loc[atingiu, "taxa_2024_pct"], s=5, alpha=0.25, color=CINZA_MEDIO, edgecolors="none", label="atingiu a meta")
    ax.scatter(gap.loc[~atingiu, coluna_meta], gap.loc[~atingiu, "taxa_2024_pct"], s=5, alpha=0.30, color=ACENTO_ALERTA, edgecolors="none", label="não atingiu")
    ax.plot([0, 100], [0, 100], color=CINZA_ESCURO, linestyle=":", linewidth=1.2, label="meta = observado")
    pct = float((~atingiu).mean())
    ax.set_xlabel("meta de alfabetização para 2024 (%)")
    ax.set_ylabel("taxa de alfabetização observada em 2024 (%)")
    ax.set_title(f"{pct:.1%} dos municípios ficaram abaixo da meta de 2024\n(n = {formatar_milhar(len(gap))} municípios)")
    ax.legend(loc="upper left")
    return fig


def medir_determinismo_das_metas(metas: pd.DataFrame) -> pd.DataFrame:
    """Cardinalidade, dispersão e correlação de cada meta com a taxa base.

    A pergunta é se a meta traz informação nova sobre o município ou apenas
    reescreve a taxa que ele já tinha. Traz zero: a correlação com a taxa base
    fica em 0,97 em todos os anos e a dispersão encolhe até a meta de 2030, que é
    80 para os 5.352 municípios.

    Isso torna a meta legítima como feature, porque é um lag e lag é permitido, e
    ilegítima como alvo preditivo, que era o desenho original da pergunta de
    negócio 4.

    Trato a meta constante como correlação indefinida em vez de deixar o numpy
    devolver `nan` com aviso: desvio zero não tem correlação, e isso é a
    descoberta, não uma falha de cálculo.
    """
    colunas = [c for c in metas.columns if c.startswith("meta_alfabetizacao_")]
    base = metas.dropna(subset=["taxa_alfabetizacao"])
    linhas = []
    for coluna in colunas:
        valido = base.dropna(subset=[coluna])
        # A meta de 2030 é 80 para todo mundo: desvio zero, correlação indefinida.
        constante = metas[coluna].nunique(dropna=True) <= 1
        linhas.append(
            {
                "meta": coluna,
                "n_valores_distintos": int(metas[coluna].nunique()),
                "pct_nulo": float(metas[coluna].isna().mean()),
                "media": float(metas[coluna].mean()),
                "desvio_padrao": float(metas[coluna].std()),
                "corr_com_taxa_base": (
                    np.nan if constante or len(valido) <= 2 else float(valido[coluna].corr(valido["taxa_alfabetizacao"]))
                ),
            }
        )
    return pd.DataFrame(linhas)


def plotar_meta_versus_taxa_base(metas: pd.DataFrame, colunas_meta: tuple[str, ...]) -> Figure:
    base = metas.dropna(subset=["taxa_alfabetizacao"])
    fig, eixos = plt.subplots(1, len(colunas_meta), figsize=(4.2 * len(colunas_meta), 4.2), sharey=True)
    eixos = np.atleast_1d(eixos)
    for ax, coluna in zip(eixos, colunas_meta):
        dados = base.dropna(subset=[coluna])
        r = float(dados[coluna].corr(dados["taxa_alfabetizacao"]))
        ax.scatter(dados["taxa_alfabetizacao"], dados[coluna], s=4, alpha=0.2, color=CINZA_ESCURO, edgecolors="none")
        ax.set_xlabel("taxa de alfabetização base (%)")
        ano = coluna.split("_")[-1]
        ax.set_title(f"meta {ano} · r = {r:.3f}\n{dados[coluna].nunique()} valores distintos")
    eixos[0].set_ylabel("meta de alfabetização (%)")
    fig.suptitle("A meta é uma trajetória determinística da própria taxa base: um lag disfarçado", y=1.03, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# 12. Cobertura do lag
# ---------------------------------------------------------------------------
def medir_cobertura_lag(coorte: pd.DataFrame, lag: pd.DataFrame, chaves: tuple[str, ...] = ("id_escola", "id_municipio")) -> pd.DataFrame:
    """Parcela de unidades e de alunos de 2024 com correspondente em 2023.

    Mede duas coisas que é fácil confundir: quantas *unidades* reaparecem e
    quantos *alunos* elas cobrem. Os dois números divergem porque unidade grande
    e unidade pequena pesam igual na primeira contagem e não na segunda.

    Cuidado ao ler o resultado da chave `id_escola`. Ele mede coincidência de
    identificador, não continuidade de escola, e o item 13b mostra que, no caso
    da escola, as duas coisas não são a mesma. Para `id_municipio`, que é o
    código do IBGE, a leitura é literal.
    """
    linhas = []
    for chave in chaves:
        conhecidas = set(lag[chave].unique())
        tem = coorte[chave].isin(conhecidas)
        linhas.append(
            {
                "chave": chave,
                "unidades_2024": int(coorte[chave].nunique()),
                "unidades_2023": len(conhecidas),
                "pct_unidades_com_historico": float(coorte[chave].drop_duplicates().isin(conhecidas).mean()),
                "pct_alunos_com_historico": float(tem.mean()),
                "alunos_sem_historico": int((~tem).sum()),
            }
        )
    return pd.DataFrame(linhas)


def medir_perfil_sem_historico(coorte: pd.DataFrame, lag: pd.DataFrame, chave: str = "id_escola") -> pd.DataFrame:
    """Compara a taxa de alfabetização de quem tem e de quem não tem histórico.

    Se as taxas divergem, a flag deixa de ser só marcador de imputação e passa a
    ter sinal próprio. A diferença medida é pequena, 1,7pp, e a explicação dela
    não é comportamental: quem não tem histórico municipal é, em 99,9% dos casos,
    aluno de São Paulo, Acre ou Distrito Federal, que não estão no microdado de
    2023. A flag é um indicador de UF disfarçado.
    """
    tem = coorte[chave].isin(set(lag[chave].unique()))
    linhas = []
    for rotulo, mascara in [("com histórico", tem), ("sem histórico", ~tem)]:
        recorte = coorte[mascara]
        n, k = len(recorte), int(recorte["alfabetizado"].sum())
        lo, hi = ic_wilson(k, n)
        linhas.append(
            {
                "grupo": rotulo,
                "chave": chave,
                "n_alunos": n,
                "taxa_alfabetizado": k / n,
                "ic_low": float(lo),
                "ic_high": float(hi),
            }
        )
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 13-14. `id_aluno` (A2 e A3)
# ---------------------------------------------------------------------------
def medir_reciclagem_id_aluno(aluno_lag: pd.DataFrame, aluno_atual: pd.DataFrame) -> dict:
    """Quantos `id_aluno` coincidem entre anos e quantos caem na mesma escola.

    A armadilha mais cara do projeto. O ID coincide em massa, 1,5 milhão de
    casos, e quase nunca aponta para a mesma escola: é sequencial, reciclado a
    cada edição, não é matrícula.

    O que torna isso perigoso não é o erro em si, é que ele *funciona*. O prefixo
    do ID codifica a UF, então o par falso ainda carrega sinal territorial e a
    métrica sobe. Quem fizesse o join veria o AUC melhorar e concluiria que
    acertou.
    """
    a_lag = aluno_lag[["id_aluno", "id_escola", "id_municipio"]].drop_duplicates("id_aluno")
    a_atual = aluno_atual[["id_aluno", "id_escola", "id_municipio"]].drop_duplicates("id_aluno")
    pares = a_lag.merge(a_atual, on="id_aluno", suffixes=("_lag", "_atual"))
    return {
        "ids_2023": int(len(a_lag)),
        "ids_2024": int(len(a_atual)),
        "ids_em_ambos_anos": int(len(pares)),
        "pct_mesma_escola": float((pares["id_escola_lag"] == pares["id_escola_atual"]).mean()),
        "pct_mesmo_municipio": float((pares["id_municipio_lag"] == pares["id_municipio_atual"]).mean()),
        "pares_falsos_estimados": int(len(pares) * (1 - (pares["id_escola_lag"] == pares["id_escola_atual"]).mean())),
    }


def plotar_reciclagem_id_aluno(medida: dict) -> Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    rotulos = ["mesmo id_aluno\nnos dois anos", "…e mesmo município", "…e mesma escola"]
    valores = [
        medida["ids_em_ambos_anos"],
        medida["ids_em_ambos_anos"] * medida["pct_mesmo_municipio"],
        medida["ids_em_ambos_anos"] * medida["pct_mesma_escola"],
    ]
    cores = [CINZA_MEDIO, CINZA_MEDIO, ACENTO_ALERTA]
    y = np.arange(len(rotulos))[::-1]
    ax.barh(y, valores, color=cores)
    for yi, valor in zip(y, valores):
        ax.text(valor + medida["ids_em_ambos_anos"] * 0.01, yi, f"{formatar_milhar(valor)}  ({valor / medida['ids_em_ambos_anos']:.2%})", va="center", fontsize=9)
    ax.set_yticks(y, rotulos)
    ax.set_xlabel("nº de id_aluno")
    ax.set_xlim(0, medida["ids_em_ambos_anos"] * 1.28)
    ax.set_title("`id_aluno` é reciclado a cada ano: não é chave longitudinal (A2)")
    return fig


def medir_estabilidade_id_escola(
    aluno_lag: pd.DataFrame, aluno_atual: pd.DataFrame, dim_municipio: pd.DataFrame
) -> dict:
    """Verifica se um `id_escola` presente nos dois anos é a mesma escola.

    A checagem nasceu por analogia, depois de provado que `id_aluno` é reciclado:
    se um identificador da base é sequencial, por que o outro não seria? A
    resposta é que é — e a essa altura já havia nove features de escola
    construídas sobre o pressuposto contrário.

    O teste é o mais simples possível e não depende de modelo: escola não muda de
    município. Se o mesmo `id_escola` aparece em municípios diferentes em 2023 e
    2024, não é a mesma escola.

    Mede também a correlação de posto entre `id_escola` e `id_municipio` dentro
    de cada ano. Ela explica por que o join falso ainda produz AUC acima do
    acaso: o ID é atribuído em ordem territorial, então o par errado tende a cair
    na mesma UF e carrega sinal de UF, não de escola.
    """
    lag = aluno_lag[aluno_lag["presenca"] == 1] if "presenca" in aluno_lag else aluno_lag
    atual = aluno_atual[aluno_atual["presenca"] == 1] if "presenca" in aluno_atual else aluno_atual

    escola_lag = lag.groupby("id_escola", observed=True)["id_municipio"].first().rename("municipio_lag")
    escola_atual = atual.groupby("id_escola", observed=True)["id_municipio"].first().rename("municipio_atual")
    pares = pd.concat([escola_lag, escola_atual], axis=1, join="inner")

    uf_por_municipio = dim_municipio.set_index("id_municipio")["sigla_uf"]
    mesma_uf = uf_por_municipio.reindex(pares["municipio_lag"]).to_numpy() == uf_por_municipio.reindex(
        pares["municipio_atual"]
    ).to_numpy()
    mesmo_municipio = (pares["municipio_lag"] == pares["municipio_atual"]).to_numpy()

    ids_reaproveitados = set(pares.index[~mesmo_municipio])
    return {
        "escolas_lag": int(escola_lag.size),
        "escolas_atual": int(escola_atual.size),
        "ids_em_ambos_anos": int(len(pares)),
        "pct_mesmo_municipio": float(mesmo_municipio.mean()),
        "pct_mesma_uf": float(mesma_uf.mean()),
        "alunos_com_escola_falsa": int(atual["id_escola"].isin(ids_reaproveitados).sum()),
        "pct_alunos_com_escola_falsa": float(atual["id_escola"].isin(ids_reaproveitados).mean()),
        "spearman_id_escola_municipio_lag": float(
            escola_lag.reset_index()["id_escola"].corr(escola_lag.reset_index()["municipio_lag"], method="spearman")
        ),
        "spearman_id_escola_municipio_atual": float(
            escola_atual.reset_index()["id_escola"].corr(
                escola_atual.reset_index()["municipio_atual"], method="spearman"
            )
        ),
    }


def plotar_estabilidade_id_escola(medida: dict) -> Figure:
    """Compara coincidência de identificador com coincidência de escola.

    A barra de cima é o que um `merge` ingênuo entrega; a de baixo é quanto disso
    é verdade. A distância entre as duas é o tamanho do erro que quase entrou no
    modelo.
    """
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    total = medida["ids_em_ambos_anos"]
    rotulos = ["mesmo id_escola\nnos dois anos", "…e mesma UF", "…e mesmo município"]
    valores = [total, total * medida["pct_mesma_uf"], total * medida["pct_mesmo_municipio"]]
    cores = [CINZA_MEDIO, CINZA_MEDIO, ACENTO_ALERTA]
    y = np.arange(len(rotulos))[::-1]
    ax.barh(y, valores, color=cores)
    for yi, valor in zip(y, valores):
        ax.text(valor + total * 0.01, yi, f"{formatar_milhar(valor)}  ({valor / total:.1%})", va="center", fontsize=9)
    ax.set_yticks(y, rotulos)
    ax.set_xlabel("nº de id_escola")
    ax.set_xlim(0, total * 1.3)
    ax.set_title(
        f"Só {medida['pct_mesmo_municipio']:.1%} dos id_escola repetidos são a mesma escola"
        f" ({medida['pct_mesma_uf']:.0%} caem na mesma UF, e é daí que vem o sinal aparente)"
    )
    return fig


def medir_cobertura_lag_por_uf(coorte: pd.DataFrame, lag: pd.DataFrame, chave: str = "id_municipio") -> pd.DataFrame:
    """Cobertura do histórico de 2023 quebrada por UF da coorte de 2024.

    A cobertura agregada de 77% parecia dispersa: escolas e municípios novos
    espalhados pelo país. Quebrando por UF, ela se revela binária: cada UF tem
    cobertura total ou cobertura zero. O microdado de 2023 simplesmente não tem
    São Paulo, Acre e Distrito Federal.

    A consequência é séria e não é de imputação: a flag `tem_historico_municipio`
    vira um indicador de UF, e um quarto da coorte entra no modelo sem nenhuma
    feature de lag.
    """
    conhecidas = set(lag[chave].unique())
    tem = coorte[chave].isin(conhecidas)
    resumo = (
        coorte.assign(_tem=tem)
        .groupby("sigla_uf", observed=True)
        .agg(n_alunos=("_tem", "size"), pct_com_historico=("_tem", "mean"), n_unidades=(chave, "nunique"))
        .reset_index()
        .sort_values("pct_com_historico")
    )
    return resumo.reset_index(drop=True)


def medir_variancia_entre_municipios(aluno_lag: pd.DataFrame, porte_minimo: int = 20) -> dict:
    """Quanto da variância da taxa escolar é variância *entre* municípios.

    Decomposição clássica de soma de quadrados, ponderada pelo número de escolas.
    Responde se faz sentido investir engenharia de features no grão escola: se a
    maior parte da dispersão entre escolas já é dispersão entre municípios, a
    escola acrescenta pouco mesmo quando medida corretamente.

    Filtro por porte porque escola de 5 alunos infla a variância intramunicipal
    com puro erro amostral e faria a conta mentir para o lado oposto.
    """
    presentes = aluno_lag[aluno_lag["presenca"] == 1] if "presenca" in aluno_lag else aluno_lag
    escola = (
        presentes.groupby(["id_municipio", "id_escola"], observed=True)["alfabetizado"]
        .agg(n="size", taxa="mean")
        .reset_index()
    )
    escola = escola[escola["n"] >= porte_minimo]
    media_geral = float(escola["taxa"].mean())
    por_municipio = escola.groupby("id_municipio", observed=True)["taxa"].agg(["size", "mean"])
    entre = float((((por_municipio["mean"] - media_geral) ** 2) * por_municipio["size"]).sum())
    total = float(((escola["taxa"] - media_geral) ** 2).sum())
    return {
        "porte_minimo": porte_minimo,
        "n_escolas": int(len(escola)),
        "n_municipios": int(por_municipio.shape[0]),
        "variancia_entre_municipios": entre / total,
        "variancia_dentro_do_municipio": 1 - entre / total,
    }


def medir_prefixo_id_aluno(aluno: pd.DataFrame, dim_municipio: pd.DataFrame, amostra: int = 300_000) -> dict:
    """Testa se os dois primeiros dígitos de `id_aluno` reproduzem o código de UF do IBGE.

    Reproduzem, em 100% da amostra. Como numérica, `id_aluno` seria território de
    alta resolução entrando pela porta dos fundos, e atravessaria o split
    agrupado por município, porque a UF é compartilhada entre municípios de
    treino e de teste.

    Amostro 300 mil linhas em vez de rodar na base inteira: o resultado é 1,000
    ou não é, e nesse tipo de verificação binária a amostra decide igual.
    """
    dados = aluno.sample(min(amostra, len(aluno)), random_state=config.RANDOM_STATE)
    prefixo_aluno = dados["id_aluno"].astype("int64") // 10**6
    prefixo_municipio = dados["id_municipio"].astype("int64") // 10**5
    coincide = float((prefixo_aluno == prefixo_municipio).mean())
    uf_por_prefixo = (
        dados.assign(prefixo=prefixo_aluno.to_numpy())
        .merge(dim_municipio[["id_municipio", "sigla_uf"]], on="id_municipio", how="left")
        .groupby("prefixo")["sigla_uf"]
        .agg(n_ufs="nunique", uf_dominante=lambda s: s.mode().iat[0] if len(s.mode()) else None)
    )
    return {
        "n_amostra": int(len(dados)),
        "pct_prefixo_igual_ao_codigo_uf": coincide,
        "n_prefixos": int(uf_por_prefixo.shape[0]),
        "prefixos_com_uma_unica_uf": int((uf_por_prefixo["n_ufs"] == 1).sum()),
        "tabela_prefixo_uf": uf_por_prefixo.reset_index(),
    }


def plotar_prefixo_id_aluno(medida: dict) -> Figure:
    tabela = medida["tabela_prefixo_uf"].sort_values("prefixo")
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(tabela))
    ax.barh(y, tabela["n_ufs"], color=[CINZA_MEDIO if v == 1 else ACENTO_ALERTA for v in tabela["n_ufs"]])
    ax.set_yticks(y, [f"{int(p)} → {uf}" for p, uf in zip(tabela["prefixo"], tabela["uf_dominante"])], fontsize=7)
    ax.set_xlabel("nº de UFs distintas que compartilham o prefixo")
    ax.set_xticks([0, 1, 2])
    ax.set_title(
        f"Cada prefixo de `id_aluno` mapeia uma única UF em {medida['prefixos_com_uma_unica_uf']}/{medida['n_prefixos']} casos"
        ": é proxy territorial (A3)"
    )
    return fig


# ---------------------------------------------------------------------------
# 16. Onde o sinal vive: município, escola, ou nenhum dos dois
# ---------------------------------------------------------------------------
def medir_auc_por_porte_da_escola(
    coorte: pd.DataFrame, feature: str, chave_porte: str = "esc_n_alunos_lag1", faixas: tuple[tuple[int, int], ...] = ((0, 20), (20, 50), (50, 80), (80, 10**9))
) -> pd.DataFrame:
    """AUC univariada de uma feature de lag, estratificada pelo porte da escola em 2023.

    Montei isto para testar a hipótese de que a taxa da escola é fraca por ruído
    amostral: se fosse, escola grande, onde a taxa tem erro-padrão pequeno,
    daria AUC visivelmente maior. Não dá. A AUC fica plana em 0,554–0,560 em
    todas as faixas.

    Interpretei isso, na primeira leitura, como "a escola não carrega informação
    além do território". Estava errado pela metade: o item 13b mostra que a
    própria feature é um join falso, então o que este teste realmente prova é que
    não há como medir sinal escolar com esta base, não que ele não exista.
    """
    linhas = []
    for minimo, maximo in faixas:
        recorte = coorte[(coorte[chave_porte] >= minimo) & (coorte[chave_porte] < maximo)].dropna(subset=[feature])
        if len(recorte) < 1_000:
            continue
        linhas.append(
            {
                "feature": feature,
                "faixa_porte": f"{minimo}–{maximo}" if maximo < 10**9 else f"≥ {minimo}",
                "n_alunos": int(len(recorte)),
                "n_escolas": int(recorte["id_escola"].nunique()),
                "auc": auc_univariada(recorte["alfabetizado"].to_numpy(), recorte[feature].to_numpy()),
            }
        )
    return pd.DataFrame(linhas)


def medir_auc_das_features(coorte: pd.DataFrame, features: list[str], alvo: str = "alfabetizado") -> pd.DataFrame:
    """Ranking de AUC univariada: o que cada feature candidata entrega sozinha.

    Define a barra que o modelo tem de superar. Se o GBM completo não
    bater a melhor feature isolada com folga maior que o desvio entre folds, o
    ganho é marginal, e isso vai no relatório, de preferência escrito por mim,
    não descoberto pelo avaliador.

    Oriento as AUCs abaixo de 0,5 pelo complemento porque 0,45 e 0,55 têm o mesmo
    poder de separação, só que com o sinal trocado; deixar as duas na mesma
    escala evita ler como inútil uma feature que só está invertida.
    """
    linhas = []
    y = coorte[alvo].to_numpy()
    for feature in features:
        escore = coorte[feature].to_numpy(dtype=float)
        linhas.append(
            {
                "feature": feature,
                "pct_disponivel": float(np.isfinite(escore).mean()),
                "auc": auc_univariada(y, escore),
            }
        )
    quadro = pd.DataFrame(linhas)
    # 0,45 separa tanto quanto 0,55; só muda o sinal. Oriento para comparar na mesma escala.
    quadro["auc_orientada"] = quadro["auc"].where(quadro["auc"] >= 0.5, 1 - quadro["auc"])
    return quadro.sort_values("auc_orientada", ascending=False).reset_index(drop=True)


def plotar_auc_por_porte(medidas: pd.DataFrame, titulo: str) -> Figure:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(medidas))
    ax.bar(x, medidas["auc"], 0.55, color=CINZA_MEDIO)
    ax.axhline(0.5, color=ACENTO_ALERTA, linestyle="--", linewidth=1.2)
    ax.text(len(medidas) - 0.5, 0.505, "acaso", color=ACENTO_ALERTA, fontsize=8, ha="right")
    for i, linha in medidas.reset_index(drop=True).iterrows():
        ax.text(i, linha["auc"] + 0.004, f"{linha['auc']:.3f}", ha="center", fontweight="bold")
    ax.set_xticks(x, [f"{f}\n({formatar_milhar(n)} alunos)" for f, n in zip(medidas["faixa_porte"], medidas["n_alunos"])])
    ax.set_xlabel("alunos avaliados na escola em 2023")
    ax.set_ylabel("ROC-AUC univariada")
    ax.set_ylim(0.48, max(0.62, float(medidas["auc"].max()) + 0.03))
    ax.set_title(titulo)
    return fig


def plotar_auc_das_features(quadro: pd.DataFrame, titulo: str) -> Figure:
    dados = quadro.sort_values("auc_orientada")
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.34 * len(dados) + 1.2)))
    y = np.arange(len(dados))
    ax.barh(y, dados["auc_orientada"], color=CINZA_MEDIO)
    ax.axvline(0.5, color=ACENTO_ALERTA, linestyle="--", linewidth=1.2)
    for yi, valor in zip(y, dados["auc_orientada"]):
        ax.text(valor + 0.002, yi, f"{valor:.3f}", va="center", fontsize=9)
    ax.set_yticks(y, dados["feature"])
    ax.set_xlim(0.48, float(dados["auc_orientada"].max()) + 0.03)
    ax.set_xlabel("ROC-AUC univariada (orientada; 0,5 = acaso)")
    ax.set_title(titulo)
    return fig


# ---------------------------------------------------------------------------
# 18. Viés de seleção dos ausentes
# ---------------------------------------------------------------------------
def medir_vies_de_ausencia(aluno: pd.DataFrame, quantis: int = 5) -> pd.DataFrame:
    """Taxa de alfabetização dos presentes, por quintil de taxa de presença do município.

    Não dá para medir o aluno ausente, porque ele não fez a prova. O que dá para medir
    é se o *contexto* de muita ausência anda junto com desempenho pior entre quem
    compareceu, e anda: 53,4% no quintil de menor presença contra 74,6% no de
    maior.

    Isso não prova que o ausente teria ido mal, mas fixa a direção provável do
    viés. A taxa medida é otimista, e é mais otimista exatamente onde o problema
    é maior. Corto municípios com menos de 20 avaliados porque abaixo disso a
    taxa é ruído e contaminaria os quintis.
    """
    municipal = aluno.groupby("id_municipio", observed=True).agg(
        n_total=("presenca", "size"), n_presentes=("presenca", "sum")
    )
    municipal["taxa_presenca"] = municipal["n_presentes"] / municipal["n_total"]
    presentes = aluno[aluno["presenca"] == 1].groupby("id_municipio", observed=True)["alfabetizado"].agg(n="size", k="sum")
    municipal = municipal.join(presentes, how="inner")
    municipal = municipal[municipal["n"] >= 20]
    municipal["quintil_presenca"] = pd.qcut(municipal["taxa_presenca"], quantis, labels=[f"Q{i + 1}" for i in range(quantis)])

    resumo = municipal.groupby("quintil_presenca", observed=True).agg(
        n_municipios=("n", "size"),
        n_alunos_presentes=("n", "sum"),
        k=("k", "sum"),
        taxa_presenca_media=("taxa_presenca", "mean"),
    )
    resumo["taxa_alfabetizado"] = resumo["k"] / resumo["n_alunos_presentes"]
    lo, hi = ic_wilson(resumo["k"].to_numpy(), resumo["n_alunos_presentes"].to_numpy())
    resumo["ic_low"], resumo["ic_high"] = lo, hi
    return resumo.reset_index()


def plotar_vies_de_ausencia(resumo: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    x = np.arange(len(resumo))
    erro = np.vstack(
        [(resumo["taxa_alfabetizado"] - resumo["ic_low"]).to_numpy() * 100, (resumo["ic_high"] - resumo["taxa_alfabetizado"]).to_numpy() * 100]
    )
    ax.bar(x, resumo["taxa_alfabetizado"] * 100, 0.55, color=CINZA_MEDIO, yerr=erro, error_kw={"ecolor": CINZA_ESCURO, "capsize": 3})
    for i, linha in resumo.reset_index(drop=True).iterrows():
        ax.text(i, linha["taxa_alfabetizado"] * 100 + 0.8, f"{linha['taxa_alfabetizado'] * 100:.1f}%", ha="center", fontweight="bold")
    ax.set_xticks(
        x,
        [f"{q}\npresença média {p:.1%}\n({formatar_milhar(n)} municípios)" for q, p, n in zip(resumo["quintil_presenca"], resumo["taxa_presenca_media"], resumo["n_municipios"])],
        fontsize=8,
    )
    ax.set_ylabel("taxa de alfabetização entre os presentes (%)")
    ax.set_xlabel("quintil de taxa de presença do município")
    ax.set_title("Municípios com menos presença também alfabetizam menos: a ausência não é aleatória")
    return fig


def medir_diferenca_ausentes_por_uf(aluno: pd.DataFrame, dim_municipio: pd.DataFrame) -> pd.DataFrame:
    """Taxa de presença e de alfabetização por UF, ordenadas pela presença.

    O ponto aqui não é a correlação entre as duas colunas, que é fraca. É a
    incomparabilidade: Santa Catarina avaliou 70% dos alunos e o Ceará 98%, então
    as duas taxas descrevem populações diferentes e qualquer ranking de UF mistura
    desempenho com cobertura.

    Entra na leitura de fairness, onde reportar erro por subgrupo
    territorial sem essa ressalva seria comparar coisas distintas.
    """
    dados = aluno.merge(dim_municipio[["id_municipio", "sigla_uf"]], on="id_municipio", how="left")
    resumo = dados.groupby("sigla_uf", observed=True).agg(n_total=("presenca", "size"), n_presentes=("presenca", "sum"))
    resumo["taxa_presenca"] = resumo["n_presentes"] / resumo["n_total"]
    presentes = dados[dados["presenca"] == 1].groupby("sigla_uf", observed=True)["alfabetizado"].mean().rename("taxa_alfabetizado")
    return resumo.join(presentes).sort_values("taxa_presenca").reset_index()


# ---------------------------------------------------------------------------
# Protótipo de feature store: vive e morre dentro da EDA
# ---------------------------------------------------------------------------
# A versão de produção mora em `src/preprocessing/feature_store.py`.
# Este protótipo existe porque os itens 9, 12 e 16 precisam medir correlação,
# cobertura e AUC das features candidatas antes de alguém escrevê-las em
# definitivo. É a EDA que decide o desenho delas, não o contrário, e foi
# exatamente aqui, ao medir, que o join por `id_escola` se revelou falso.
PERCENTIS_PROFICIENCIA = (0.25, 0.50, 0.75)


def _agregar_nivel(aluno_lag: pd.DataFrame, chave: str, prefixo: str) -> pd.DataFrame:
    """Agregados de 2023 de uma unidade: escola, município ou UF.

    Duas populações de propósito. A taxa de presença é calculada sobre todas as
    linhas, inclusive `presenca = 0`, que a Gold descarta e que só existem no
    microdado; taxa de alfabetização e percentis de proficiência usam apenas os
    presentes, porque proficiência de quem faltou não existe.

    Misturar as duas populações produziria uma taxa de alfabetização deprimida
    pelos ausentes, que é precisamente o artefato que o filtro `presenca = 1` da
    Gold existe para evitar.
    """
    presenca = aluno_lag.groupby(chave, observed=True)["presenca"].agg(
        **{f"{prefixo}_n_avaliados_lag1": "size", f"{prefixo}_taxa_presenca_lag1": "mean"}
    )
    presentes = aluno_lag[aluno_lag["presenca"] == 1]
    base = presentes.groupby(chave, observed=True).agg(
        **{
            f"{prefixo}_n_alunos_lag1": ("alfabetizado", "size"),
            f"{prefixo}_taxa_alfab_lag1": ("alfabetizado", "mean"),
            f"{prefixo}_prof_media_lag1": ("proficiencia", "mean"),
            f"{prefixo}_prof_sd_lag1": ("proficiencia", "std"),
        }
    )
    percentis = presentes.groupby(chave, observed=True)["proficiencia"].quantile(list(PERCENTIS_PROFICIENCIA)).unstack()
    percentis.columns = [f"{prefixo}_prof_p{int(p * 100)}_lag1" for p in percentis.columns]
    saida = base.join(percentis).join(presenca)
    saida[f"{prefixo}_prof_iqr_lag1"] = saida[f"{prefixo}_prof_p75_lag1"] - saida[f"{prefixo}_prof_p25_lag1"]
    return saida.reset_index()


def montar_agregados_lag(aluno_lag: pd.DataFrame, dim_municipio: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Agregados de escola, município e UF calculados só sobre o microdado de 2023.

    Toda feature do modelo nasce aqui, e nasce anterior ao alvo por construção,
    é a defesa nº 2 da matriz anti-leakage, garantida pelo desenho e não por
    disciplina de quem escreve o código.

    A UF depende de `dim_municipio` porque o microdado do INEP traz
    `id_municipio` e não traz UF. Efeito colateral que só apareceu ao contar as
    linhas: o resultado tem 23 UFs, não 26, porque São Paulo, Acre e Distrito
    Federal não aparecem no arquivo de 2023.
    """
    com_uf = aluno_lag.merge(dim_municipio[["id_municipio", "sigla_uf"]], on="id_municipio", how="left")
    escola = _agregar_nivel(aluno_lag, "id_escola", "esc")
    municipio = _agregar_nivel(aluno_lag, "id_municipio", "mun")
    uf = _agregar_nivel(com_uf, "sigla_uf", "uf")

    escolas_por_municipio = (
        aluno_lag.groupby("id_municipio", observed=True)["id_escola"].nunique().rename("mun_n_escolas_lag1").reset_index()
    )
    share_estadual = (
        aluno_lag.assign(estadual=(aluno_lag["rede"] == 2).astype("int8"))
        .groupby("id_municipio", observed=True)["estadual"]
        .mean()
        .rename("mun_share_rede_estadual_lag1")
        .reset_index()
    )
    municipio = municipio.merge(escolas_por_municipio, on="id_municipio").merge(share_estadual, on="id_municipio")
    return {"escola": escola, "municipio": municipio, "uf": uf}


def montar_coorte_com_lag(
    coorte: pd.DataFrame, agregados: dict[str, pd.DataFrame], mapa_escola_municipio: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Junta a coorte de 2024 aos agregados de 2023, validando cardinalidade em cada merge.

    `validate="many_to_one"` nos três joins e um `assert` de contagem no fim:
    join que multiplica linha em silêncio é a causa mais comum de métrica
    inflada, e a coorte precisa sair daqui com exatamente as linhas com que
    entrou.

    O join por `id_escola` fabrica pares, porque o identificador é reciclado entre
    edições (item 13b). Mantive-o aqui porque a EDA precisa medir o tamanho do
    estrago; ele não deve sobreviver para `src/preprocessing/feature_store.py`.
    """
    n_inicial = len(coorte)
    saida = (
        coorte.merge(agregados["escola"], on="id_escola", how="left", validate="many_to_one")
        .merge(agregados["municipio"], on="id_municipio", how="left", validate="many_to_one")
        .merge(agregados["uf"], on="sigla_uf", how="left", validate="many_to_one")
    )
    assert len(saida) == n_inicial, "o join com os agregados multiplicou linhas"

    saida["tem_historico_escola"] = saida["esc_taxa_alfab_lag1"].notna().astype("int8")
    saida["tem_historico_municipio"] = saida["mun_taxa_alfab_lag1"].notna().astype("int8")
    saida["esc_desvio_vs_municipio"] = saida["esc_taxa_alfab_lag1"] - saida["mun_taxa_alfab_lag1"]
    saida["mun_desvio_vs_uf"] = saida["mun_taxa_alfab_lag1"] - saida["uf_taxa_alfab_lag1"]
    if mapa_escola_municipio is not None:
        saida = saida.merge(mapa_escola_municipio, on="id_escola", how="left", validate="many_to_one")
    return saida


def suavizar_taxa_escola(coorte: pd.DataFrame, k: float = 30.0) -> pd.Series:
    """Puxa a taxa da escola em direção à do município: `(taxa_esc·n + taxa_mun·k) / (n + k)`.

    A ideia era proteger a escola pequena, cuja taxa é quase só ruído, deixando a
    grande falar por si. `k` é o número de alunos-equivalentes de contexto
    municipal que se soma a cada escola.

    O item 16 mostra que a AUC cresce monotonicamente com `k` até o valor da taxa
    municipal pura e nunca a ultrapassa. Ou seja: isto funciona apagando a
    escola. Continua no código como conveniência de imputação, na qual escola sem
    histórico herda o município, e não como recuperação de sinal.
    """
    n = coorte["esc_n_alunos_lag1"].fillna(0)
    taxa_escola = coorte["esc_taxa_alfab_lag1"]
    taxa_municipio = coorte["mun_taxa_alfab_lag1"]
    return ((taxa_escola.fillna(taxa_municipio) * n) + taxa_municipio * k) / (n + k)


def montar_desvio_escolar(aluno_ano: pd.DataFrame) -> pd.DataFrame:
    """Desvio da taxa de cada escola em relação ao próprio município, no mesmo ano.

    Responde ao que a taxa bruta não responde: descontado o contexto municipal, a
    posição relativa de uma escola se repete no ano seguinte? Era o teste tido
    como decisivo, porque não depende de tamanho de amostra da mesma forma que a
    taxa bruta.

    Deu zero. Só que o zero prova menos do que parecia na hora: o par entre anos
    é formado por `id_escola`, e esse identificador é reciclado (item 13b), de
    modo que a conta correlaciona o desvio de uma escola com o de outra.

    Devolve o mesmo formato de `agregar_taxa` para poder ser reaproveitado por
    `medir_persistencia`.
    """
    presentes = aluno_ano[aluno_ano["presenca"] == 1] if "presenca" in aluno_ano else aluno_ano
    escola = presentes.groupby(["id_municipio", "id_escola"], observed=True)["alfabetizado"].agg(n="size", taxa_escola="mean")
    municipio = presentes.groupby("id_municipio", observed=True)["alfabetizado"].mean().rename("taxa_municipio")
    saida = escola.join(municipio, on="id_municipio").reset_index()
    saida["taxa"] = saida["taxa_escola"] - saida["taxa_municipio"]
    return saida[["id_escola", "n", "taxa"]]


def medir_grid_suavizacao(coorte: pd.DataFrame, valores_k: tuple[float, ...] = (0, 5, 10, 20, 30, 50, 100, 200, 500)) -> pd.DataFrame:
    """AUC univariada da taxa escolar suavizada, para cada `k` da grade.

    O plano previa calibrar `k` por validação cruzada. Rodar a grade inteira
    antes mostrou que não há o que calibrar: a curva é monotônica e converge para
    a taxa do município. O melhor `k` é sempre o maior da grade, o que é a
    definição de um hiperparâmetro sem ótimo interior.

    Uma grade barata respondeu o que um `GridSearchCV` teria escondido atrás de
    um número.
    """
    y = coorte["alfabetizado"].to_numpy()
    linhas = []
    for k in valores_k:
        escore = suavizar_taxa_escola(coorte, k=float(k)).to_numpy()
        linhas.append({"k": k, "auc": auc_univariada(y, escore)})
    return pd.DataFrame(linhas)


def plotar_grid_suavizacao(grid: pd.DataFrame, auc_municipio: float, auc_escola: float) -> Figure:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(grid["k"], grid["auc"], marker="o", color=CINZA_ESCURO, linewidth=1.8, label="taxa da escola suavizada")
    ax.axhline(auc_municipio, color=ACENTO_ALERTA, linestyle="--", linewidth=1.3, label=f"taxa do município ({auc_municipio:.3f})")
    ax.axhline(auc_escola, color=CINZA_MEDIO, linestyle=":", linewidth=1.3, label=f"taxa crua da escola ({auc_escola:.3f})")
    ax.set_xscale("symlog")
    ax.set_xlabel("k: peso do município na suavização (0 = escola crua)")
    ax.set_ylabel("ROC-AUC univariada")
    ax.set_title("A suavização melhora monotonicamente até virar a taxa do município,\nnão recupera sinal escolar")
    ax.legend(loc="lower right")
    return fig


def plotar_cardinalidade(estrutura: pd.DataFrame, titulo: str, n_linhas: int) -> Figure:
    """Cardinalidade por coluna em escala log, com constantes e identificadores destacados.

    Log porque a distância entre 1 e 1,85 milhão não cabe em escala linear sem
    achatar as categóricas contra o eixo, que são justamente as que interessam.
    """
    dados = estrutura.sort_values("n_unicos")
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.30 * len(dados) + 1.2)))
    y = np.arange(len(dados))
    cores = [
        ACENTO_ALERTA if n <= 1 else (CINZA_ESCURO if n > 0.5 * n_linhas else CINZA_MEDIO) for n in dados["n_unicos"]
    ]
    ax.barh(y, dados["n_unicos"].clip(lower=1), color=cores)
    ax.set_xscale("log")
    ax.set_yticks(y, dados["coluna"], fontsize=8)
    for yi, n in zip(y, dados["n_unicos"]):
        ax.text(max(n, 1) * 1.25, yi, formatar_milhar(n), va="center", fontsize=8)
    ax.set_xlabel("nº de valores distintos (escala log)")
    ax.set_title(titulo)
    return fig


def plotar_cobertura_lag(cobertura: pd.DataFrame, perfis: pd.DataFrame) -> Figure:
    """Cobertura do histórico ao lado da taxa do alvo em cada grupo.

    Os dois painéis juntos porque a cobertura sozinha não diz se a ausência
    importa. Lado a lado, fica visível que 23% da coorte fica sem lag municipal e
    que esse grupo alfabetiza 1,7pp menos, diferença pequena, e que o item 12
    explica: são São Paulo, Acre e o Distrito Federal.
    """
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(cobertura))
    eixos[0].bar(x, cobertura["pct_alunos_com_historico"] * 100, 0.5, color=CINZA_MEDIO, label="com histórico")
    eixos[0].bar(
        x, (1 - cobertura["pct_alunos_com_historico"]) * 100, 0.5,
        bottom=cobertura["pct_alunos_com_historico"] * 100, color=CINZA_ESCURO, hatch="///", edgecolor="white", label="sem histórico",
    )
    for i, linha in cobertura.reset_index(drop=True).iterrows():
        eixos[0].text(i, linha["pct_alunos_com_historico"] * 50, f"{linha['pct_alunos_com_historico']:.1%}", ha="center", color="white", fontweight="bold")
    eixos[0].set_xticks(x, cobertura["chave"])
    eixos[0].set_ylabel("% dos alunos de 2024")
    eixos[0].set_ylim(0, 118)
    eixos[0].set_title("Cobertura do lag de 2023")
    eixos[0].legend(loc="upper center", ncols=2, fontsize=8)

    x2 = np.arange(len(perfis))
    erro = np.vstack(
        [(perfis["taxa_alfabetizado"] - perfis["ic_low"]).to_numpy() * 100, (perfis["ic_high"] - perfis["taxa_alfabetizado"]).to_numpy() * 100]
    )
    eixos[1].bar(x2, perfis["taxa_alfabetizado"] * 100, 0.5, color=CINZA_MEDIO, yerr=erro, error_kw={"ecolor": CINZA_ESCURO, "capsize": 3})
    for i, linha in perfis.reset_index(drop=True).iterrows():
        eixos[1].text(i, linha["taxa_alfabetizado"] * 100 + 1.2, f"{linha['taxa_alfabetizado']:.1%}", ha="center", fontweight="bold")
    eixos[1].set_xticks(x2, [f"{g}\n({c})" for g, c in zip(perfis["grupo"], perfis["chave"])], fontsize=8)
    eixos[1].set_ylabel("taxa de alfabetização (%)")
    eixos[1].set_ylim(0, 80)
    eixos[1].set_title("Quem não tem histórico alfabetiza pouco menos")
    fig.suptitle("O lag não cobre todo mundo: a ausência precisa de flag, não de imputação silenciosa", y=1.03, fontweight="bold")
    return fig
