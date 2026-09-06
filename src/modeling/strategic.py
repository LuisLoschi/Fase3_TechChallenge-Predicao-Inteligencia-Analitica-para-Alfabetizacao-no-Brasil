"""Produtos municipais retrospectivos e cenários condicionais de metas.

O escore nacional é OOF por município, com seleção histórica compartilhada;
não é validação confirmatória. A inferência de crianças usa contexto territorial.
Metas usam avaliação OOF territorial na transição 2023→2024, sem teste de ano futuro.
"""

from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.data import loader
from src.modeling import campeao, split, tuning
from src.preprocessing import pipeline as pl

log = logging.getLogger(__name__)

PARQUET_OOF_NACIONAL = config.DIR_MODELS / "escores_oof_nacional.parquet"
JSON_OOF_NACIONAL = config.DIR_MODELS / "escores_oof_nacional.json"

CSV_RANKING = config.DIR_REPORTS / "ranking_risco_municipal.csv"
CSV_CLUSTERS = config.DIR_REPORTS / "clusters_municipais.csv"
CSV_METAS = config.DIR_REPORTS / "projecao_metas_municipios.csv"

CSV_SHRINKAGE = config.DIR_METRICS / "estrategia_shrinkage.csv"
CSV_CONCENTRACAO = config.DIR_METRICS / "estrategia_concentracao.csv"
CSV_SELECAO_K = config.DIR_METRICS / "estrategia_selecao_de_k.csv"
CSV_PERFIL_CLUSTERS = config.DIR_METRICS / "estrategia_perfil_clusters.csv"
CSV_REGIAO_CLUSTERS = config.DIR_METRICS / "estrategia_regiao_clusters.csv"
CSV_VOLATILIDADE = config.DIR_METRICS / "estrategia_volatilidade_por_porte.csv"
CSV_VALIDACAO_METAS = config.DIR_METRICS / "estrategia_validacao_metas.csv"
CSV_FUNIL = config.DIR_METRICS / "estrategia_funil_2030.csv"
JSON_ESTRATEGIA = config.DIR_METRICS / "estrategia_resumo.json"

# Onde a Etapa 4 mediu ROC-AUC 0,5173, contra 0,6602 no resto do país. AC e DF
# são as duas unidades sem nenhuma fonte de lag de UF em 2023: o escore existe,
# mas ordena no acaso, e publicar posição para eles seria publicar ruído.
UFS_NAO_AVALIADAS = ("AC", "DF")

# Recorte descritivo de uma contribuição SHAP; não mede a qualidade do ranking.
LIMIAR_FAIXA_ACHATADA = 65.0

PORTES_DE_REFERENCIA = (0, 25, 50, 100, 200, 500, 1000)
TAMANHO_DO_TOPO = 50
GRADE_DE_K = range(3, 9)
ANOS_DE_META = (2025, 2026, 2027, 2028, 2029, 2030)


# ---------------------------------------------------------------------------
# 1. Escore out-of-fold nacional
# ---------------------------------------------------------------------------
def escorar_out_of_fold(seed: int = config.RANDOM_STATE, forcar: bool = False) -> pd.DataFrame:
    """Refaz os cinco folds agrupados sobre a coorte inteira e escora cada aluno.

    O que muda em relação ao artefato da Etapa 4 é a cobertura, não o método: lá
    o escore out-of-fold existe só para o desenvolvimento, porque o teste
    foi reservado para o ajuste, mas consultado na seleção histórica. A medição já aconteceu,
    e um ranking que ignorasse os 1.104 municípios do teste não seria publicável.

    Os hiperparâmetros vêm da busca feita no desenvolvimento, e foram reutilizados nos folds nacionais. O desenvolvimento
    participa da seleção, de modo que o escore nacional não é nested CV. Para uma métrica isso seria
    inaceitável; para um escore descritivo é o preço de cobrir o país inteiro, e
    fica declarado. O invariante que sustenta o ranking continua valendo inteiro:
    nenhum município é escorado pelo modelo que o treinou.
    """
    contrato = {"hash_dataset": campeao.hash_do_dataset(), "seed": seed,
                "features": list(pl.FEATURES_MODELO), "hiperparametros": tuning.carregar_hiperparametros()}
    cache_confere = (JSON_OOF_NACIONAL.exists() and
                    json.loads(JSON_OOF_NACIONAL.read_text(encoding="utf-8"))) == contrato
    if PARQUET_OOF_NACIONAL.exists() and cache_confere and not forcar:
        log.info("escores out-of-fold já existem em %s", PARQUET_OOF_NACIONAL)
        return pd.read_parquet(PARQUET_OOF_NACIONAL)

    inicio = time.time()
    dados = loader.carregar_dataset_modelagem()
    X, y, grupos = pl.separar_X_y(dados)
    particoes = split.folds(y, grupos, seed=seed)
    estimador = pl.montar_pipeline(
        LGBMClassifier(random_state=seed, **tuning.carregar_hiperparametros())
    )

    escore = np.full(len(y), np.nan)
    fold = np.full(len(y), -1, dtype="int8")
    for i, (treino, validacao) in enumerate(particoes):
        parcial = clone(estimador).fit(X.iloc[treino], y.iloc[treino])
        escore[validacao] = parcial.predict_proba(X.iloc[validacao])[:, 1]
        fold[validacao] = i
        log.info(
            "fold %d: %d municípios escorados por um modelo que não os viu (%.0fs)",
            i, grupos.iloc[validacao].nunique(), time.time() - inicio,
        )
    assert np.isfinite(escore).all(), "algum aluno ficou sem escore out-of-fold"

    tabela = pd.DataFrame(
        {
            config.GRUPO_CV: grupos.to_numpy(),
            config.TARGET: y.to_numpy(),
            "escore_oof": escore,
            config.COL_PESO: dados[config.COL_PESO].to_numpy(),
            "fold": fold,
        }
    )
    config.DIR_MODELS.mkdir(parents=True, exist_ok=True)
    tabela.to_parquet(PARQUET_OOF_NACIONAL, index=False)
    JSON_OOF_NACIONAL.write_text(json.dumps(contrato, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(
        "escores out-of-fold: %d alunos, %d municípios, ROC-AUC %.4f (%.0fs)",
        len(tabela), tabela[config.GRUPO_CV].nunique(), roc_auc_score(y, escore),
        time.time() - inicio,
    )
    return tabela


# ---------------------------------------------------------------------------
# 2. Shrinkage empírico bayesiano
# ---------------------------------------------------------------------------
def suavizar_empirico_bayes(
    valor: np.ndarray, n: np.ndarray, variancia_amostral: np.ndarray, prior: np.ndarray | float
) -> tuple[np.ndarray, float, float]:
    """Puxa cada município para o prior na proporção do quanto sua medida é ruidosa.

    Estimador de componentes de variância, na forma clássica de credibilidade: a
    dispersão observada entre municípios é a soma da dispersão verdadeira com o
    ruído amostral, então subtrair o ruído esperado devolve τ², e a razão entre
    ruído e τ² dá o peso `k` — o número de alunos a partir do qual a medida do
    município passa a valer mais que o prior.

    `variancia_amostral` é passado de fora porque a resposta depende do que se
    está suavizando. Numa taxa observada o ruído é binomial, `μ(1−μ)/n`. Na média
    municipal de um escore preditivo é `s²/n`, e `s²` ali é minúsculo: o modelo
    dá quase a mesma probabilidade a todos os alunos do mesmo município. As duas
    contas são a mesma; os dois `k` que elas produzem não são, e a diferença
    entre eles é um resultado desta etapa, não um detalhe de implementação.

    Devolve `(suavizado, k, mu)`.
    """
    valor, n = np.asarray(valor, dtype=float), np.asarray(n, dtype=float)
    variancia_amostral = np.asarray(variancia_amostral, dtype=float)
    prior_vetor = np.full_like(valor, prior, dtype=float) if np.isscalar(prior) else np.asarray(prior, float)

    mu = float(np.average(valor, weights=n))
    var_total = float(np.average((valor - prior_vetor) ** 2, weights=n))
    var_amostral = float(np.average(variancia_amostral, weights=n))
    tau2 = var_total - var_amostral
    assert tau2 > 0, f"ruído amostral ({var_amostral:.6f}) excede a dispersão observada ({var_total:.6f})"

    k = float(np.average(n * variancia_amostral, weights=n) / tau2)
    return (n * valor + k * prior_vetor) / (n + k), k, mu


def _variancia_binomial(taxa: float, n: np.ndarray) -> np.ndarray:
    return taxa * (1 - taxa) / np.asarray(n, dtype=float)


# ---------------------------------------------------------------------------
# 3. Tabela municipal e os dois rankings
# ---------------------------------------------------------------------------
def agregar_municipios(oof: pd.DataFrame) -> pd.DataFrame:
    """Do grão aluno ao grão município, ponderando por `peso_aluno` (regra B2).

    `soma_peso` é a população matriculada reexpandida pelo fator oficial de
    não-resposta do INEP — é ela, e não a contagem de presentes, que dimensiona
    quantas crianças um programa precisaria atender.
    """
    quadro = oof.rename(columns={config.TARGET: "y", "escore_oof": "p", config.COL_PESO: "w"})
    quadro = quadro.assign(yw=lambda d: d.y * d.w, pw=lambda d: d.p * d.w)
    tabela = quadro.groupby(config.GRUPO_CV).agg(
        n_alunos_avaliados=("p", "size"),
        populacao_estimada=("w", "sum"),
        yw=("yw", "sum"),
        pw=("pw", "sum"),
        variancia_do_escore=("p", "var"),
    )
    tabela["taxa_observada_de_risco"] = tabela["yw"] / tabela["populacao_estimada"]
    tabela["risco_medio_oof"] = tabela["pw"] / tabela["populacao_estimada"]
    tabela["variancia_do_escore"] = tabela["variancia_do_escore"].fillna(0.0)
    return tabela.drop(columns=["yw", "pw"]).reset_index()


def _contexto_municipal() -> pd.DataFrame:
    """Nome, território, presença e a feature de 2023 que o modelo mais usa."""
    identificacao = (
        loader.carregar_gold(config.ANO_ALVO, ["id_municipio", "nome_municipio"])
        .drop_duplicates("id_municipio")
    )
    agregado = loader.carregar_agregados_municipais(config.ANO_ALVO)[
        ["id_municipio", "sigla_uf", "nome_regiao", "taxa_presenca", "taxa_alfab_ponderada"]
    ].rename(
        columns={
            "taxa_presenca": "taxa_presenca_2024",
            "taxa_alfab_ponderada": "taxa_alfabetizacao_2024",
        }
    )
    lag = (
        loader.carregar_dataset_modelagem(["id_municipio", "mun_taxa_alfab_lag1"])
        .groupby("id_municipio", as_index=False)
        .first()
    )
    return identificacao.merge(agregado, on="id_municipio").merge(lag, on="id_municipio", how="left")


def construir_ranking(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """As duas ordenações, os dois shrinkages e as ressalvas que viajam com a lista."""
    tabela = agregar_municipios(oof).merge(_contexto_municipal(), on=config.GRUPO_CV, how="left")

    # Shrinkage 1 — sobre o escore do modelo, como o plano exige. A variância
    # amostral aqui é a da média de um escore quase constante dentro do município.
    suavizado, k_modelo, mu_modelo = suavizar_empirico_bayes(
        tabela["risco_medio_oof"],
        tabela["n_alunos_avaliados"],
        tabela["variancia_do_escore"] / tabela["n_alunos_avaliados"],
        prior=float(np.average(tabela["risco_medio_oof"], weights=tabela["n_alunos_avaliados"])),
    )
    tabela["risco_suavizado"] = suavizado

    # Shrinkage 2 — sobre a taxa observada de 2024, com o escore do modelo como
    # prior. É a mistura de credibilidade: município grande fala por si, município
    # de vinte alunos é lido pelo que o contexto prevê para ele.
    ajustado, k_observado, mu_observado = suavizar_empirico_bayes(
        tabela["taxa_observada_de_risco"],
        tabela["n_alunos_avaliados"],
        _variancia_binomial(
            float(np.average(tabela["taxa_observada_de_risco"], weights=tabela["n_alunos_avaliados"])),
            tabela["n_alunos_avaliados"],
        ),
        prior=tabela["risco_medio_oof"].to_numpy(),
    )
    tabela["taxa_observada_ajustada"] = ajustado

    tabela["incerteza_posicao_quantificada"] = False
    tabela["criancas_em_risco"] = tabela["risco_suavizado"] * tabela["populacao_estimada"]
    tabela["modelo_avaliado"] = ~tabela["sigla_uf"].isin(UFS_NAO_AVALIADAS)
    tabela["faixa_shap_taxa_municipal"] = (
        tabela["mun_taxa_alfab_lag1"].isna()
        | (tabela["mun_taxa_alfab_lag1"] < LIMIAR_FAIXA_ACHATADA)
    )

    # Posição e decil só existem para quem o modelo consegue ordenar; em AC e DF
    # o escore existe e a posição não, que é a forma de dizer "não avaliado" sem
    # apagar o município da tabela. Decil 1 é o mais arriscado, como em
    # `metrics.lift_por_decil`.
    avaliados = tabela["modelo_avaliado"]
    for criterio, coluna in (("taxa", "risco_suavizado"), ("volume", "criancas_em_risco")):
        posicao = tabela.loc[avaliados, coluna].rank(ascending=False, method="first")
        tabela[f"posicao_por_{criterio}"] = posicao
        tabela[f"decil_por_{criterio}"] = pd.qcut(posicao, 10, labels=range(1, 11)).astype("float")

    colunas = [
        "id_municipio", "nome_municipio", "sigla_uf", "nome_regiao",
        "n_alunos_avaliados", "populacao_estimada", "taxa_presenca_2024",
        "risco_medio_oof", "risco_suavizado", "posicao_por_taxa", "decil_por_taxa",
        "criancas_em_risco", "posicao_por_volume", "decil_por_volume",
        "taxa_alfabetizacao_2024", "taxa_observada_de_risco", "taxa_observada_ajustada",
        "mun_taxa_alfab_lag1", "modelo_avaliado", "faixa_shap_taxa_municipal", "incerteza_posicao_quantificada",
    ]
    ranking = tabela[colunas].sort_values("posicao_por_taxa", na_position="last")

    diagnostico = {
        "n_municipios": int(len(ranking)),
        "n_nao_avaliados": int((~ranking["modelo_avaliado"]).sum()),
        "ufs_nao_avaliadas": list(UFS_NAO_AVALIADAS),
        "shrinkage": {
            "escore_do_modelo": {"k": k_modelo, "mu": mu_modelo},
            "taxa_observada": {"k": k_observado, "mu": mu_observado},
        },
        "criancas_em_risco_total": float(ranking["criancas_em_risco"].sum()),
    }
    return ranking, diagnostico


def medir_concentracao(ranking: pd.DataFrame, topo: int = TAMANHO_DO_TOPO) -> pd.DataFrame:
    """O que cada ordenação coloca no topo: porte, valor e quantas crianças.

    É a tabela que justifica publicar duas listas. Se as duas selecionassem os
    mesmos municípios, a segunda seria redundante.
    """
    avaliados = ranking[ranking["modelo_avaliado"]]
    linhas = []
    for nome, coluna in (
        ("taxa observada crua", "taxa_observada_de_risco"),
        ("taxa observada ajustada", "taxa_observada_ajustada"),
        ("risco do modelo (publicado)", "risco_suavizado"),
        ("volume de crianças (publicado)", "criancas_em_risco"),
    ):
        fatia = avaliados.nlargest(topo, coluna)
        linhas.append(
            {
                "ordenacao": nome,
                "n_alunos_mediano": float(fatia["n_alunos_avaliados"].median()),
                "municipios_com_menos_de_50_alunos": int((fatia["n_alunos_avaliados"] < 50).sum()),
                "criancas_em_risco": float(fatia["criancas_em_risco"].sum()),
                "share_das_criancas_em_risco": float(
                    fatia["criancas_em_risco"].sum() / avaliados["criancas_em_risco"].sum()
                ),
                "taxa_presenca_mediana": float(fatia["taxa_presenca_2024"].median()),
            }
        )
    tabela = pd.DataFrame(linhas)
    topo_taxa = set(avaliados.nlargest(topo, "risco_suavizado")["id_municipio"])
    topo_volume = set(avaliados.nlargest(topo, "criancas_em_risco")["id_municipio"])
    tabela["municipios_em_comum_com_a_lista_por_taxa"] = [
        len(topo_taxa & set(avaliados.nlargest(topo, coluna)["id_municipio"]))
        for coluna in (
            "taxa_observada_de_risco", "taxa_observada_ajustada",
            "risco_suavizado", "criancas_em_risco",
        )
    ]
    assert len(topo_taxa & topo_volume) == tabela.iloc[3]["municipios_em_comum_com_a_lista_por_taxa"]
    return tabela


# ---------------------------------------------------------------------------
# 4. Agrupamento de municípios
# ---------------------------------------------------------------------------
NIVEIS = tuple(f"proporcao_aluno_nivel_{i}" for i in range(9))

FEATURES_CLUSTER = (
    "taxa_alfabetizacao_2023",
    "taxa_alfabetizacao_2024",
    "media_portugues_2024",
    "taxa_presenca_2024",
    "log_porte",
    "share_rede_estadual",
) + NIVEIS


def montar_base_de_clusterizacao() -> pd.DataFrame:
    """Retrato municipal de 2024, mais a taxa de 2023 para dar a única profundidade disponível.

    A composição por nível de proficiência é a variável que dá textura ao
    agrupamento — ela separa um município que concentra crianças no nível 5 de
    outro que as espalha entre 0 e 8 com a mesma média. E é também a razão de o
    resultado ser um retrato: `proporcao_aluno_nivel_0..8` é 100% nula em 2023,
    então não existe a versão anterior dela para comparar.

    A taxa de 2023 vem da Gold onde há microdado e do agregado do INEP onde não
    há, a mesma coalescência da Etapa 3. Sem ela São Paulo inteiro sairia da
    análise, e sobrariam 4.840 municípios em vez de 5.461.
    """
    agregado = loader.carregar_agregados_municipais(config.ANO_ALVO).rename(
        columns={
            "taxa_alfab_ponderada": "taxa_alfabetizacao_2024",
            "prof_media_ponderada": "media_portugues_2024",
            "taxa_presenca": "taxa_presenca_2024",
        }
    )[
        [
            "id_municipio", "sigla_uf", "nome_regiao", "n_avaliados",
            "taxa_alfabetizacao_2024", "media_portugues_2024", "taxa_presenca_2024",
        ]
    ]

    da_gold = loader.carregar_agregados_municipais(config.ANO_LAG)[
        ["id_municipio", "taxa_alfab_ponderada"]
    ].rename(columns={"taxa_alfab_ponderada": "_taxa_gold"})
    do_inep = loader.carregar_meta_municipio().query("ano == @config.ANO_LAG").set_index(
        "id_municipio"
    )["taxa_alfabetizacao"].div(100).rename("_taxa_inep")

    niveis = loader.carregar_agregado_municipio().query(
        "ano == @config.ANO_ALVO and rede == @config.REDE_AGREGADO_RECONCILIACAO"
    )[["id_municipio", *NIVEIS]]

    coorte = loader.carregar_dataset_modelagem(["id_municipio", "rede_grupo", config.COL_PESO])
    coorte["_estadual"] = (coorte["rede_grupo"] == "Estadual") * coorte[config.COL_PESO]
    share = coorte.groupby("id_municipio").apply(
        lambda g: g["_estadual"].sum() / g[config.COL_PESO].sum(), include_groups=False
    ).rename("share_rede_estadual")

    base = (
        agregado.merge(da_gold, on="id_municipio", how="left")
        .merge(do_inep, on="id_municipio", how="left")
        .merge(niveis, on="id_municipio", how="left")
        .merge(share, on="id_municipio", how="left")
    )
    base["taxa_alfabetizacao_2023"] = base["_taxa_gold"].fillna(base["_taxa_inep"])
    base["log_porte"] = np.log10(base["n_avaliados"])
    return base.drop(columns=["_taxa_gold", "_taxa_inep"])


def escolher_k(matriz: np.ndarray, grade=GRADE_DE_K, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Silhueta e inércia para cada `k` da grade, sobre a matriz já padronizada.

    A inércia é arredondada porque a redução multithread do `KMeans` soma os
    quadrados em ordem variável entre execuções e o último bit oscila. Rótulos,
    silhueta e tamanhos de cluster são idênticos — mas sem o arredondamento o CSV
    muda de MD5 sem que nada tenha mudado, e a checagem de determinismo do
    relatório passaria a acusar um falso positivo por edição.
    """
    linhas = []
    for k in grade:
        modelo = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(matriz)
        linhas.append(
            {
                "k": k,
                "inercia": round(float(modelo.inertia_), 4),
                "silhueta": float(silhouette_score(matriz, modelo.labels_)),
                "menor_cluster": int(np.bincount(modelo.labels_).min()),
            }
        )
    tabela = pd.DataFrame(linhas)
    tabela["queda_de_inercia"] = (-tabela["inercia"].diff()).round(4)
    return tabela


NOMES_DE_CLUSTER = {
    0: "alfabetização consolidada",
    1: "faixa intermediária",
    2: "risco alto e cobertura menor",
}


def clusterizar(seed: int = config.RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """KMeans no grão município, com `k` decidido pela silhueta na grade de 3 a 8.

    Os clusters recebem nome pela ordem da taxa de alfabetização de 2024, e não
    pelo rótulo que o KMeans sorteou — sem isso o mesmo cluster mudaria de nome
    entre execuções e a tabela do relatório deixaria de conferir com o CSV.
    """
    base = montar_base_de_clusterizacao()
    completos = base.dropna(subset=list(FEATURES_CLUSTER)).copy()
    log.info(
        "clusterização sobre %d dos %d municípios da coorte", len(completos), len(base)
    )

    matriz = StandardScaler().fit_transform(completos[list(FEATURES_CLUSTER)])
    selecao = escolher_k(matriz, seed=seed)
    k = int(selecao.loc[selecao["silhueta"].idxmax(), "k"])
    modelo = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(matriz)

    media_por_rotulo = (
        completos.assign(_rotulo=modelo.labels_)
        .groupby("_rotulo")["taxa_alfabetizacao_2024"]
        .mean()
        .sort_values(ascending=False)
    )
    renomear = {rotulo: posicao for posicao, rotulo in enumerate(media_por_rotulo.index)}
    completos["cluster"] = pd.Series(modelo.labels_, index=completos.index).map(renomear)
    completos["cluster_nome"] = completos["cluster"].map(NOMES_DE_CLUSTER)

    perfil = (
        completos.groupby(["cluster", "cluster_nome"])
        .agg(
            n_municipios=("id_municipio", "size"),
            alunos_avaliados=("n_avaliados", "sum"),
            porte_mediano=("n_avaliados", "median"),
            taxa_2023=("taxa_alfabetizacao_2023", "mean"),
            taxa_2024=("taxa_alfabetizacao_2024", "mean"),
            media_portugues=("media_portugues_2024", "mean"),
            taxa_presenca=("taxa_presenca_2024", "mean"),
            share_rede_estadual=("share_rede_estadual", "mean"),
        )
        .reset_index()
    )
    niveis_medios = completos.groupby("cluster")[list(NIVEIS)].mean()
    perfil["niveis_0_a_2"] = niveis_medios[list(NIVEIS[:3])].sum(axis=1).to_numpy()
    perfil["niveis_3_a_5"] = niveis_medios[list(NIVEIS[3:6])].sum(axis=1).to_numpy()
    perfil["niveis_6_a_8"] = niveis_medios[list(NIVEIS[6:])].sum(axis=1).to_numpy()

    composicao = (
        pd.crosstab(completos["cluster_nome"], completos["nome_regiao"], normalize="index")
        .reset_index()
    )

    # A pergunta que o plano manda testar: baixa presença forma um perfil próprio?
    completos["decil_de_presenca"] = pd.qcut(completos["taxa_presenca_2024"], 10, labels=False)
    dispersao = (
        completos[completos["decil_de_presenca"] == 0]["cluster_nome"]
        .value_counts(normalize=True)
        .to_dict()
    )

    saida = completos[
        [
            "id_municipio", "sigla_uf", "nome_regiao", "cluster", "cluster_nome",
            "n_avaliados", *FEATURES_CLUSTER,
        ]
    ].sort_values(["cluster", "id_municipio"])

    diagnostico = {
        "k_escolhido": k,
        "criterio": "maior silhueta na grade de 3 a 8, sobre features padronizadas",
        "silhueta": float(selecao.loc[selecao["k"] == k, "silhueta"].iloc[0]),
        "n_municipios": int(len(completos)),
        "n_fora_por_dado_faltante": int(len(base) - len(completos)),
        "ufs_fora": base[base[list(FEATURES_CLUSTER)].isna().any(axis=1)]["sigla_uf"]
        .value_counts()
        .to_dict(),
        "decil_de_menor_presenca_por_cluster": dispersao,
    }
    return saida, selecao, perfil, composicao, diagnostico


# ---------------------------------------------------------------------------
# 5. Metas — gap, cenário condicional e validação municipal fora do ajuste
# ---------------------------------------------------------------------------
def _base_de_metas() -> pd.DataFrame:
    """Taxas e metas do INEP, na mesma apuração em que as metas foram construídas.

    A taxa vem da base de metas e não dos agregados do projeto de propósito: a
    meta foi calculada sobre a rede Municipal apurada pelo INEP, e comparar com
    a taxa ponderada de todas as redes trocaria a régua no meio da conta. As duas
    medidas concordam com MAE de 0,81pp, e a diferença fica declarada.
    """
    metas = loader.carregar_meta_municipio()
    de_2024 = metas.query("ano == @config.ANO_ALVO").set_index("id_municipio")
    de_2023 = metas.query("ano == @config.ANO_LAG").set_index("id_municipio")

    base = pd.DataFrame(
        {
            "taxa_2023": de_2023["taxa_alfabetizacao"],
            "taxa_2024": de_2024["taxa_alfabetizacao"],
            "meta_2024": de_2024["meta_alfabetizacao_2024"],
            **{
                f"meta_{ano}": de_2024[f"meta_alfabetizacao_{ano}"] for ano in ANOS_DE_META
            },
        }
    ).reset_index()

    porte = loader.carregar_agregados_municipais(config.ANO_ALVO)[
        ["id_municipio", "n_avaliados", "taxa_presenca", "sigla_uf", "nome_regiao"]
    ].rename(columns={"n_avaliados": "n_alunos_avaliados", "taxa_presenca": "taxa_presenca_2024"})
    porte_2023 = loader.carregar_agregados_municipais(config.ANO_LAG)[
        ["id_municipio", "n_avaliados"]
    ].rename(columns={"n_avaliados": "n_alunos_2023"})
    return base.merge(porte, on="id_municipio", how="inner").merge(
        porte_2023, on="id_municipio", how="left"
    )


def projetar_metas(seed: int = config.RANDOM_STATE) -> tuple:
    """Cenários limitados e validação por municípios na mesma transição anual."""
    from src.modeling.metas import gerar_produtos
    total = loader.carregar_agregados_municipais(config.ANO_ALVO).id_municipio.nunique()
    produtos = gerar_produtos(_base_de_metas(), int(total), seed)
    return produtos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(etapa: str = "tudo") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    inicio = time.time()
    config.DIR_METRICS.mkdir(parents=True, exist_ok=True)
    resumo: dict = {
        "hash_dataset": campeao.hash_do_dataset(),
        "seed": config.RANDOM_STATE,
    }
    if campeao.MODELO_CAMPEAO.exists():
        pacote = campeao.carregar_campeao()
        assert pacote["hash_dataset"] == resumo["hash_dataset"], (
            "o Parquet em disco não é o que treinou o campeão — retreine antes de publicar ranking"
        )
        resumo["versoes"] = pacote["versoes"]

    if etapa in ("tudo", "oof", "risco"):
        oof = escorar_out_of_fold(forcar=etapa == "oof")
        resumo["escore"] = {
            "n_alunos": int(len(oof)),
            "n_municipios": int(oof[config.GRUPO_CV].nunique()),
            "roc_auc_out_of_fold": float(roc_auc_score(oof[config.TARGET], oof["escore_oof"])),
        }

    if etapa in ("tudo", "risco"):
        ranking, diagnostico = construir_ranking(oof)
        ranking.to_csv(CSV_RANKING, index=False)
        concentracao = medir_concentracao(ranking)
        concentracao.to_csv(CSV_CONCENTRACAO, index=False)
        pd.DataFrame(
            [
                {"quantidade": "escore do modelo", **diagnostico["shrinkage"]["escore_do_modelo"]},
                {"quantidade": "taxa observada", **diagnostico["shrinkage"]["taxa_observada"]},
            ]
        ).to_csv(CSV_SHRINKAGE, index=False)
        resumo["risco"] = diagnostico
        log.info(
            "ranking: %d municípios, %d não avaliados, k do escore %.3f e k da taxa %.1f",
            diagnostico["n_municipios"], diagnostico["n_nao_avaliados"],
            diagnostico["shrinkage"]["escore_do_modelo"]["k"],
            diagnostico["shrinkage"]["taxa_observada"]["k"],
        )
        print(concentracao.round(4).to_string(index=False))

    if etapa in ("tudo", "clusters"):
        clusters, selecao, perfil, composicao, diagnostico = clusterizar()
        clusters.to_csv(CSV_CLUSTERS, index=False)
        selecao.to_csv(CSV_SELECAO_K, index=False)
        perfil.to_csv(CSV_PERFIL_CLUSTERS, index=False)
        composicao.to_csv(CSV_REGIAO_CLUSTERS, index=False)
        resumo["clusters"] = diagnostico
        log.info("k escolhido: %d (silhueta %.4f)", diagnostico["k_escolhido"], diagnostico["silhueta"])
        print(selecao.round(4).to_string(index=False))
        print(perfil.round(3).to_string(index=False))

    if etapa in ("tudo", "metas"):
        metas, volatilidade, validacao, funil, diagnostico, oof_metas = projetar_metas()
        metas.to_csv(CSV_METAS, index=False)
        oof_metas.to_csv(config.DIR_METRICS / "estrategia_metas_oof.csv", index=False)
        volatilidade.to_csv(CSV_VOLATILIDADE, index=False)
        validacao.to_csv(CSV_VALIDACAO_METAS, index=False)
        funil.to_csv(CSV_FUNIL, index=False)
        resumo["metas"] = diagnostico
        log.info(
            "metas: gap mediano %.2fpp, AUC OOF territorial %.4f, referência descritiva de porte %d",
            diagnostico["gap_de_esforco_2025"]["mediana"],
            diagnostico["validacao_municipal_2023_2024"]["roc_auc"]["projecao_com_shrinkage_e_deriva"],
            diagnostico["persistencia"]["porte_referencia_dispersao"],
        )
        print(volatilidade.round(2).to_string(index=False))
        print(validacao.round(3).to_string(index=False))
        print(funil.round(3).to_string(index=False))

    resumo["segundos"] = round(time.time() - inicio, 1)
    anterior = json.loads(JSON_ESTRATEGIA.read_text(encoding="utf-8")) if JSON_ESTRATEGIA.exists() else {}
    JSON_ESTRATEGIA.write_text(
        json.dumps({**anterior, **resumo}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("camada estratégica concluída em %.0fs", time.time() - inicio)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tudo")
