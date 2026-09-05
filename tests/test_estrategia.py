"""Invariantes da camada estratégica (Etapa 6).

O que pode dar errado aqui não quebra a execução — produz uma lista plausível e
errada. Um escore que viu o próprio município no treino ainda é um número entre
0 e 1; um ranking dominado por municípios de oito alunos ainda é um ranking; uma
probabilidade de meta convertida em rótulo binário ainda cabe numa planilha. Os
testes abaixo travam exatamente essas quatro coisas: a origem out-of-fold do
escore, o comportamento do shrinkage, a reprodutibilidade das duas ordenações e
a recusa em transformar incerteza em rótulo.

Os que dependem dos artefatos da etapa são pulados quando eles não existem, para
que um clone limpo rode `pytest -q` antes de qualquer coisa ser gerada.

Executar:  pytest -q tests/test_estrategia.py
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import loader
from src.modeling import split
from src.modeling import strategic as st


@pytest.fixture(scope="module")
def oof():
    if not st.PARQUET_OOF_NACIONAL.exists():
        pytest.skip("escores ainda não gerados — rode `python -m src.modeling.strategic oof`")
    return pd.read_parquet(st.PARQUET_OOF_NACIONAL)


@pytest.fixture(scope="module")
def ranking():
    if not st.CSV_RANKING.exists():
        pytest.skip("ranking ainda não gerado — rode `python -m src.modeling.strategic risco`")
    return pd.read_csv(st.CSV_RANKING)


@pytest.fixture(scope="module")
def metas():
    if not st.CSV_METAS.exists():
        pytest.skip("projeção ainda não gerada — rode `python -m src.modeling.strategic metas`")
    return pd.read_csv(st.CSV_METAS)


@pytest.fixture(scope="module")
def clusters():
    if not st.CSV_CLUSTERS.exists():
        pytest.skip("clusters ainda não gerados — rode `python -m src.modeling.strategic clusters`")
    return pd.read_csv(st.CSV_CLUSTERS)


@pytest.fixture(scope="module")
def resumo():
    if not st.JSON_ESTRATEGIA.exists():
        pytest.skip("resumo ainda não gerado — rode `python -m src.modeling.strategic`")
    return json.loads(st.JSON_ESTRATEGIA.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# O escore é out-of-fold — sem isso o ranking mede memorização
# ---------------------------------------------------------------------------
def test_cada_municipio_esta_inteiro_em_um_unico_fold(oof):
    """Meio município no treino e meio na validação devolveria a média a ele mesmo."""
    folds_por_municipio = oof.groupby(config.GRUPO_CV)["fold"].nunique()
    assert folds_por_municipio.max() == 1, (
        f"{int((folds_por_municipio > 1).sum())} municípios aparecem em mais de um fold"
    )


def test_os_folds_sao_os_mesmos_que_a_etapa_4_materializou(oof):
    """A partição não pode depender da ordem em que o módulo foi chamado."""
    dados = loader.carregar_dataset_modelagem([config.GRUPO_CV, config.TARGET])
    particoes = split.folds(dados[config.TARGET], dados[config.GRUPO_CV], seed=config.RANDOM_STATE)
    esperado = np.full(len(dados), -1, dtype="int8")
    for i, (_, validacao) in enumerate(particoes):
        esperado[validacao] = i
    assert np.array_equal(oof["fold"].to_numpy(), esperado)


def test_o_escore_cobre_a_coorte_inteira_e_o_artefato_da_etapa_4_nao(oof):
    """Um ranking nacional com 80% dos municípios não é um ranking nacional."""
    assert oof["escore_oof"].notna().all()
    assert oof["escore_oof"].between(0, 1).all()
    nacional = oof[config.GRUPO_CV].nunique()
    assert nacional == 5517

    desenvolvimento = config.DIR_MODELS / "escores_oof_desenvolvimento.parquet"
    if desenvolvimento.exists():
        anterior = pd.read_parquet(desenvolvimento)[config.GRUPO_CV].nunique()
        assert anterior < nacional, "o artefato da Etapa 4 deveria cobrir menos municípios"


# ---------------------------------------------------------------------------
# Shrinkage empírico bayesiano
# ---------------------------------------------------------------------------
def _exemplo_de_taxas(seed: int = 7):
    rng = np.random.default_rng(seed)
    n = rng.integers(5, 5000, 400)
    verdadeira = rng.beta(4, 6, 400)
    taxa = rng.binomial(n, verdadeira) / n
    return taxa, n


def test_o_shrinkage_puxa_todo_municipio_na_direcao_da_media():
    taxa, n = _exemplo_de_taxas()
    mu = float(np.average(taxa, weights=n))
    suavizado, k, _ = st.suavizar_empirico_bayes(taxa, n, st._variancia_binomial(mu, n), prior=mu)

    assert k > 0
    assert np.all(np.abs(suavizado - mu) <= np.abs(taxa - mu) + 1e-12)
    assert np.all(np.sign(suavizado - taxa) * np.sign(mu - taxa) >= 0)


def test_o_shrinkage_puxa_mais_o_municipio_pequeno():
    """Dois municípios com a mesma taxa e portes diferentes não podem ser puxados igual."""
    taxa = np.array([0.9, 0.9, 0.5, 0.5])
    n = np.array([10.0, 5000.0, 10.0, 5000.0])
    mu = 0.5
    suavizado, _, _ = st.suavizar_empirico_bayes(taxa, n, st._variancia_binomial(mu, n), prior=mu)

    deslocamento = np.abs(suavizado - taxa)
    assert deslocamento[0] > deslocamento[1]
    assert deslocamento[1] < 1e-3, "município grande praticamente não deveria se mover"


def test_o_shrinkage_do_escore_do_modelo_e_quase_nulo_e_o_da_taxa_nao(resumo):
    """O achado da etapa, travado: as duas quantidades não têm o mesmo ruído.

    O escore preditivo é quase constante dentro do município, então a média
    municipal dele quase não tem erro amostral e não há o que encolher. A taxa
    observada tem, e por isso o `k` dela é três ordens de grandeza maior.
    """
    k_modelo = resumo["risco"]["shrinkage"]["escore_do_modelo"]["k"]
    k_taxa = resumo["risco"]["shrinkage"]["taxa_observada"]["k"]
    assert k_modelo < 1.0
    assert k_taxa > 20.0
    assert k_taxa / k_modelo > 100


# ---------------------------------------------------------------------------
# Os dois rankings
# ---------------------------------------------------------------------------
def test_o_ranking_e_reprodutivel_com_a_mesma_seed(oof, ranking):
    refeito, _ = st.construir_ranking(oof)
    chave = ranking["id_municipio"].astype("int64")
    gravado = ranking.set_index(chave)["risco_suavizado"].sort_index()
    conferido = refeito.set_index(refeito["id_municipio"].astype("int64"))[
        "risco_suavizado"
    ].sort_index()
    pd.testing.assert_series_equal(gravado, conferido, check_names=False, rtol=1e-12)
    assert refeito.sort_values("posicao_por_taxa")["id_municipio"].to_list() == (
        ranking.sort_values("posicao_por_taxa")["id_municipio"].to_list()
    )


def test_o_topo_do_ranking_nao_e_dominado_por_municipios_minusculos(ranking):
    """A razão de o plano ter tornado o shrinkage obrigatório, medida nas duas réguas."""
    avaliados = ranking[ranking["modelo_avaliado"]]
    topo_do_modelo = avaliados.nlargest(50, "risco_suavizado")
    topo_da_taxa_crua = avaliados.nlargest(50, "taxa_observada_de_risco")

    assert (topo_do_modelo["n_alunos_avaliados"] < 50).sum() <= 5
    assert (topo_da_taxa_crua["n_alunos_avaliados"] < 50).sum() > 10
    assert topo_do_modelo["n_alunos_avaliados"].median() > (
        topo_da_taxa_crua["n_alunos_avaliados"].median()
    )


def test_as_duas_ordenacoes_selecionam_municipios_diferentes(ranking):
    avaliados = ranking[ranking["modelo_avaliado"]]
    por_taxa = set(avaliados.nsmallest(50, "posicao_por_taxa")["id_municipio"])
    por_volume = set(avaliados.nsmallest(50, "posicao_por_volume")["id_municipio"])
    assert len(por_taxa & por_volume) <= 2, "se as listas coincidem, publicar duas é redundante"


def test_ac_e_df_ficam_sem_posicao_no_ranking(ranking):
    """Declarados não avaliados: o escore fica, a posição não."""
    nao_avaliados = ranking[ranking["sigla_uf"].isin(st.UFS_NAO_AVALIADAS)]
    assert len(nao_avaliados) == 23
    assert (~nao_avaliados["modelo_avaliado"]).all()
    assert nao_avaliados["posicao_por_taxa"].isna().all()
    assert nao_avaliados["posicao_por_volume"].isna().all()
    assert nao_avaliados["risco_suavizado"].notna().all()
    assert ranking[ranking["modelo_avaliado"]]["posicao_por_taxa"].notna().all()


def test_criancas_em_risco_e_risco_vezes_populacao(ranking):
    esperado = ranking["risco_suavizado"] * ranking["populacao_estimada"]
    assert np.allclose(ranking["criancas_em_risco"], esperado)
    assert ranking["populacao_estimada"].sum() > ranking["n_alunos_avaliados"].sum()


def test_a_ressalva_da_faixa_achatada_acompanha_a_lista(ranking):
    """A Etapa 5 mediu a taxa municipal sem poder de ordenação abaixo de 65%."""
    esperado = ranking["mun_taxa_alfab_lag1"].isna() | (
        ranking["mun_taxa_alfab_lag1"] < st.LIMIAR_FAIXA_ACHATADA
    )
    assert (ranking["ordenacao_fragil"] == esperado).all()
    topo = ranking[ranking["modelo_avaliado"]].nsmallest(50, "posicao_por_taxa")
    assert topo["ordenacao_fragil"].mean() > 0.5, (
        "se o topo saísse da faixa achatada, a ressalva perderia o sentido"
    )


# ---------------------------------------------------------------------------
# Agrupamento
# ---------------------------------------------------------------------------
def test_o_k_escolhido_e_o_de_maior_silhueta_na_grade(resumo):
    selecao = pd.read_csv(st.CSV_SELECAO_K)
    assert set(selecao["k"]) == set(st.GRADE_DE_K)
    assert int(selecao.loc[selecao["silhueta"].idxmax(), "k"]) == resumo["clusters"]["k_escolhido"]


def test_os_clusters_sao_reprodutiveis_e_nomeados_pela_taxa(clusters):
    media = clusters.groupby("cluster")["taxa_alfabetizacao_2024"].mean()
    assert media.is_monotonic_decreasing, "o rótulo 0 tem de ser o de melhor desempenho"
    assert clusters["cluster_nome"].nunique() == clusters["cluster"].nunique()
    assert clusters["id_municipio"].is_unique


def test_a_presenca_entra_como_variavel_e_nao_como_filtro(clusters):
    """Se ela tivesse virado filtro, os municípios de baixa presença sumiriam."""
    assert "taxa_presenca_2024" in st.FEATURES_CLUSTER
    assert clusters["taxa_presenca_2024"].min() < 0.5
    assert len(clusters) > 5400


# ---------------------------------------------------------------------------
# Metas
# ---------------------------------------------------------------------------
def test_municipio_sem_meta_publicada_nao_aparece_no_csv(metas, ranking):
    publicadas = set(
        loader.carregar_meta_municipio()
        .query("ano == @config.ANO_ALVO and meta_alfabetizacao_2025.notna()")["id_municipio"]
    )
    assert metas["meta_2025"].notna().all()
    assert set(metas["id_municipio"]) <= publicadas
    assert len(metas) < len(ranking), "há municípios na coorte sem meta publicada"


def test_o_gap_de_esforco_e_aritmetica_pura(metas):
    assert np.allclose(
        metas["gap_de_esforco_2025"], metas["meta_2025"] - metas["taxa_2024"], atol=1e-9
    )


def test_a_projecao_e_probabilidade_e_nunca_rotulo(metas):
    probabilidade = metas["p_abaixo_da_meta_2025"]
    assert probabilidade.between(0, 1).all()
    assert probabilidade.nunique() > 1000, "uma probabilidade com poucos valores é um rótulo"
    assert (metas["intervalo_2025_superior"] > metas["intervalo_2025_inferior"]).all()
    assert not any(
        coluna.startswith("vai_atingir") or coluna.startswith("atinge_") for coluna in metas.columns
    )


def test_a_incerteza_engole_a_meta_na_quase_totalidade_dos_municipios(metas, resumo):
    """O produto principal da pergunta 4: o dado não separa quem atinge de quem não."""
    dentro = metas["meta_2025"].between(
        metas["intervalo_2025_inferior"], metas["intervalo_2025_superior"]
    )
    assert (metas["meta_dentro_do_intervalo"] == dentro).all()
    assert dentro.mean() > 0.90
    assert np.isclose(
        resumo["metas"]["share_com_meta_dentro_do_intervalo_de_95"], dentro.mean(), atol=1e-9
    )


def test_o_backtest_ruim_esta_reportado_e_nao_escondido(resumo):
    """AUC abaixo de 0,55 é o argumento da reformulação, não um resultado a maquiar."""
    auc = resumo["metas"]["backtest_2023_2024"]["roc_auc"]
    assert auc["taxa_2023_invertida"] < 0.50, "o preditivo ingênuo é pior que o acaso"
    assert auc["gap_ate_a_meta_sem_modelo"] < 0.50
    assert auc["projecao_com_shrinkage_e_deriva"] < 0.60


def test_a_volatilidade_cai_com_o_porte(resumo):
    volatilidade = pd.read_csv(st.CSV_VOLATILIDADE)
    assert volatilidade["evolucao_dp"].is_monotonic_decreasing
    assert volatilidade["evolucao_dp"].iloc[0] > 20
    assert volatilidade["evolucao_dp"].iloc[-1] < 10
    assert resumo["metas"]["persistencia"]["dispersao_c"] > (
        resumo["metas"]["gap_de_esforco_2025"]["mediana"]
    ), "o piso irredutível da oscilação anual é maior que o esforço que a meta pede"
