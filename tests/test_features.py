"""Invariantes da engenharia de features (Etapa 3).

`tests/test_dados.py` trava o que é verdade sobre as **bases**; aqui se trava o
que é verdade sobre o **dataset de modelagem** e sobre o pipeline que o consome:
cobertura, ausência de vazamento, determinismo e as duas decisões que a Etapa 3
tomou contra o plano — o bloco de escola e o `min_frequency` do `caderno`.

Executar:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config, eda
from src.data import loader
from src.preprocessing import build_dataset, feature_store as fs, pipeline as pl

LINHAS_DATASET_2024 = 1_851_852  # 1.852.788 da Gold menos as 936 sem medida
LINHAS_PREENCHIMENTO_ZERO = 936


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return loader.carregar_dataset_modelagem()


@pytest.fixture(scope="module")
def store() -> dict[str, pd.DataFrame]:
    return fs.montar_feature_store()


# ---------------------------------------------------------------------------
# Anti-vazamento
# ---------------------------------------------------------------------------
def test_nenhuma_coluna_proibida_chega_a_X(dataset):
    """`set(X.columns) & set(COLS_PROIBIDAS)` tem de ser vazio — item da Etapa 3.

    O dataset carrega `id_municipio` e `peso_aluno` de propósito, como colunas
    operacionais: uma é o grupo de CV e a outra o peso de agregação populacional
    exigido por B2. `separar_X_y` é quem garante que nenhuma das duas atravessa
    para a matriz do modelo.
    """
    X, y, grupos = pl.separar_X_y(dataset)
    assert not set(X.columns) & set(config.COLS_PROIBIDAS)
    assert not set(X.columns) & set(pl.COLS_OPERACIONAIS)
    assert len(X) == len(y) == len(grupos) == LINHAS_DATASET_2024
    assert set(y.unique()) == {0, 1}

    with pytest.raises(AssertionError):
        pl.separar_X_y(dataset, ["mun_taxa_alfab_lag1", config.COL_PESO])


def test_toda_feature_de_lag_e_de_2023(dataset, store):
    """A taxa municipal do dataset é a de 2023, não a de 2024.

    A checagem é dupla de propósito: bate com o agregado ponderado de 2023 a menos
    de 0,01pp **e** difere do de 2024 por mais de 1pp na média. Só a primeira
    metade passaria se alguém trocasse o ano da fonte e as duas taxas fossem
    parecidas; só a segunda passaria com uma coluna de lixo.
    """
    municipal = store["municipio"]
    gold = municipal[municipal["fonte_lag_municipal"] == fs.FONTE_GOLD]

    for ano, tolerancia in ((config.ANO_LAG, 0.01), (config.ANO_ALVO, None)):
        tabela = fs.agregados_municipais_ponderados(ano).set_index("id_municipio")
        par = pd.concat(
            [gold["mun_taxa_alfab_lag1"], tabela["taxa_alfab_ponderada"] * 100], axis=1
        ).dropna()
        erro = float((par.iloc[:, 0] - par.iloc[:, 1]).abs().mean())
        if tolerancia is not None:
            assert erro < tolerancia, f"a feature não reproduz {ano}: MAE {erro:.4f}pp"
        else:
            assert erro > 1.0, f"a feature está parecida demais com {ano}: MAE {erro:.4f}pp"


def test_correlacao_com_o_alvo_esta_longe_do_limite(dataset):
    """Nenhuma feature acima de |Spearman| 0,95 com o alvo — o teto da checklist.

    A maior medida é ~0,27. Uma coluna que se aproxime disso é vazamento até
    prova em contrário, não achado: `proficiencia` do ano corrente daria 1,000.
    """
    amostra = dataset.sample(200_000, random_state=config.RANDOM_STATE)
    X, y, _ = pl.separar_X_y(amostra)
    numericas = X.select_dtypes("number")
    correlacao = numericas.corrwith(y, method="spearman").abs()
    assert correlacao.max() < 0.95
    assert correlacao.max() > 0.10, "nenhuma feature tem sinal — o join provavelmente falhou"


def test_pipeline_aprende_so_no_fold_de_treino(dataset):
    """A mediana do imputador vem do treino, não da base inteira.

    É o teste automatizável do vetor 6 da matriz anti-leakage: se o
    `ColumnTransformer` tivesse sido ajustado fora do `Pipeline`, a estatística
    aprendida bateria com a da base completa.
    """
    amostra = dataset.sample(200_000, random_state=config.RANDOM_STATE)
    X, _, _ = pl.separar_X_y(amostra)
    # O "fold" é uma região inteira, e não uma metade aleatória: com feature
    # municipal e 200 mil linhas, duas metades aleatórias têm a mesma mediana
    # até a segunda casa, e o teste passaria sem provar nada.
    treino = X[amostra["nome_regiao"] == "Nordeste"]

    pre = pl.montar_pre_processador().fit(treino)
    imputador = pre.named_transformers_["num"].named_steps["imput"]
    numericas = list(pre.transformers_[0][2])

    aprendido = pd.Series(imputador.statistics_, index=numericas)
    np.testing.assert_allclose(aprendido, treino[numericas].median(), rtol=1e-9)

    diferentes = (aprendido - X[numericas].median()).abs() > 1e-6
    assert diferentes.sum() >= len(numericas) // 2, (
        "o imputador aprendeu a mediana da base inteira, não a do fold de treino"
    )


# ---------------------------------------------------------------------------
# Coorte e cobertura
# ---------------------------------------------------------------------------
def test_linhas_sem_medida_sairam_do_treino(dataset):
    """As 936 linhas de `preenchimento_caderno = 0` não estão no dataset."""
    gold = loader.carregar_gold(ano=config.ANO_ALVO, colunas=["preenchimento_caderno"])
    assert int((gold["preenchimento_caderno"] == 0).sum()) == LINHAS_PREENCHIMENTO_ZERO
    assert len(dataset) == LINHAS_DATASET_2024
    assert "preenchimento_caderno" not in dataset.columns


def test_coalescencia_cobre_98_por_cento_da_coorte(dataset):
    """Nível coalescido em 98,09%; distribucional em 76,88% (só a Gold).

    A diferença entre os dois números é o que a coalescência `5 → 3` compra: as
    features de nível chegam a São Paulo, as que exigem grão aluno não. Um dia
    alguém vai "simplificar" o `_coalescer` — este teste é quem avisa.
    """
    nivel = float(dataset["mun_taxa_alfab_lag1"].notna().mean())
    distribucional = float(dataset["mun_prof_p50_lag1"].notna().mean())
    assert nivel == pytest.approx(0.9809, abs=0.002)
    assert distribucional == pytest.approx(0.7688, abs=0.002)

    fonte = dataset["fonte_lag_municipal"].value_counts(normalize=True)
    assert fonte[fs.FONTE_GOLD] == pytest.approx(0.7688, abs=0.002)
    assert fonte[fs.FONTE_R5] == pytest.approx(0.1234, abs=0.002)
    assert fonte[fs.FONTE_R3] == pytest.approx(0.0886, abs=0.002)
    assert fonte[fs.FONTE_AUSENTE] == pytest.approx(0.0191, abs=0.002)


def test_gap_de_lag_continua_sendo_tres_ufs(dataset):
    """`tem_historico_municipio = 0` são SP, DF e AC — não é nulidade aleatória.

    Consequência prática: a flag é quase colinear com `sigla_uf`, e toda métrica
    da Etapa 4 tem de ser reportada estratificada por ela (B3).
    """
    sem = dataset.loc[dataset["tem_historico_municipio"] == 0, "sigla_uf"]
    assert sem.isin(["SP", "DF", "AC"]).mean() > 0.99
    ufs_sem_lag = set(dataset.loc[dataset["uf_taxa_alfab_lag1"].isna(), "sigla_uf"])
    assert ufs_sem_lag == {"AC", "DF"}, "o agregado de UF do INEP cobre SP em 2023, mas não AC e DF"


def test_nulidade_municipal_e_em_bloco(dataset):
    """As colunas que dependem do microdado somem juntas, e o bloco é a flag.

    Justifica `indicador_de_nulo=False` no baseline linear: com o indicador
    ligado, essas colunas geram cópias idênticas de `tem_historico_municipio` e
    a matriz volta a ser singular.
    """
    amostra = dataset.sample(200_000, random_state=config.RANDOM_STATE)
    bloco = pl.FEATURES_PERCENTIS_MICRODADO + ["mun_taxa_presenca_lag1", "mun_n_alunos_lag1"]
    sem_historico = amostra["tem_historico_municipio"] == 0
    for coluna in bloco:
        assert (amostra[coluna].isna() == sem_historico).all(), f"{coluna} tem nulidade própria"


# ---------------------------------------------------------------------------
# `id_escola` não é chave longitudinal — o achado da Etapa 3
# ---------------------------------------------------------------------------
def test_id_escola_nao_e_chave_longitudinal():
    """Escola não muda de município: se o `id_escola` muda, não é a mesma escola.

    Dos 36.051 identificadores presentes nos dois anos, 2,40% apontam para o
    mesmo município e 80,6% para a mesma UF. O identificador é reatribuído a cada
    edição em ordem territorial, exatamente como `id_aluno` (A2). É por isso que
    o bloco `esc_*` está fora de `FEATURES_MODELO`.
    """
    g23 = loader.carregar_gold(ano=2023, colunas=["id_escola", "id_municipio"])
    g24 = loader.carregar_gold(ano=2024, colunas=["id_escola", "id_municipio"])
    assert (g23.groupby("id_escola")["id_municipio"].nunique() > 1).sum() == 0
    assert (g24.groupby("id_escola")["id_municipio"].nunique() > 1).sum() == 0

    par = pd.concat(
        [
            g23.groupby("id_escola")["id_municipio"].first().rename("m23"),
            g24.groupby("id_escola")["id_municipio"].first().rename("m24"),
        ],
        axis=1,
        join="inner",
    )
    assert len(par) > 30_000
    mesma_escola = float((par["m23"] == par["m24"]).mean())
    assert mesma_escola < 0.10, (
        f"{mesma_escola:.2%} dos id_escola repetidos caem no mesmo município — se subir muito, "
        "reavaliar a decisão de deixar o bloco esc_* fora das features"
    )


def test_lag_de_escola_nao_supera_o_embaralhamento_dentro_da_uf(dataset):
    """Sortear uma escola qualquer da mesma UF prediz igual ou melhor que o join real.

    Controle de embaralhamento, que é o teste decisivo: se o join carregasse
    informação de escola, o valor real bateria o sorteado. Ele não bate — o que
    sobrevive ao embaralhamento é sinal de UF, não de escola.
    """
    y = dataset[config.TARGET].to_numpy()
    auc_real = eda.auc_univariada(y, dataset["esc_taxa_alfab_lag1"].to_numpy(dtype=float))

    rng = np.random.default_rng(config.RANDOM_STATE)
    por_uf = dataset.groupby("sigla_uf")["esc_taxa_alfab_lag1"]
    embaralhado = por_uf.transform(lambda s: rng.permutation(s.to_numpy()))
    auc_embaralhado = eda.auc_univariada(y, embaralhado.to_numpy(dtype=float))

    assert abs(auc_real - 0.5) <= abs(auc_embaralhado - 0.5) + 0.01, (
        f"o join real ({auc_real:.4f}) passou a superar o embaralhado ({auc_embaralhado:.4f}) — "
        "reavaliar a exclusão do bloco esc_*"
    )
    assert not set(pl.FEATURES_MODELO) & set(pl.FEATURES_ESCOLA)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def test_caderno_43_vira_categoria_infrequente(dataset):
    """12 alunos em um caderno que não existe em 2023 não podem virar coluna própria."""
    amostra = dataset.sample(300_000, random_state=config.RANDOM_STATE)
    X, _, _ = pl.separar_X_y(amostra)
    pre = pl.montar_pre_processador().fit(X)
    nomes = list(pre.get_feature_names_out())
    assert "caderno_43" not in nomes
    assert "caderno_infrequent_sklearn" in nomes


def test_uf_pequena_nao_e_colapsada(dataset):
    """SE, TO, AP e AC ficam abaixo de 1% da coorte e **precisam** sobreviver.

    É o motivo de `min_frequency` valer só para o `caderno`: aplicado à
    `sigla_uf`, o limiar de 1% apagaria quatro UFs justamente na variável de
    49pp de amplitude.
    """
    amostra = dataset.sample(300_000, random_state=config.RANDOM_STATE)
    X, _, _ = pl.separar_X_y(amostra)
    nomes = list(pl.montar_pre_processador().fit(X).get_feature_names_out())
    for uf in ("SE", "TO", "AP", "AC"):
        assert f"sigla_uf_{uf}" in nomes


def test_subconjunto_podado_tem_vif_abaixo_de_3(dataset):
    """Todo VIF < 3 no conjunto que vai ao baseline linear.

    No conjunto completo a matriz é singular por construção
    (`mun_desvio_vs_uf = mun_taxa − uf_taxa`), com VIF na casa das centenas.
    """
    amostra = dataset.sample(200_000, random_state=config.RANDOM_STATE)
    X, y, _ = pl.separar_X_y(amostra, pl.FEATURES_PODADAS)
    transformado = pl.montar_pre_processador(pl.FEATURES_PODADAS, indicador_de_nulo=False).fit_transform(X)
    numericas = [c for c in pl.FEATURES_PODADAS if c not in pl.COLS_CATEGORICAS]
    quadro = pd.DataFrame(transformado[:, : len(numericas)], columns=numericas)

    vif = pl.medir_vif(quadro)
    assert vif["vif"].max() < 3.0, f"VIF estourou:\n{vif}"

    # E o baseline linear tem de conseguir convergir sobre ele.
    modelo = pl.montar_pipeline(
        LogisticRegression(max_iter=200, random_state=config.RANDOM_STATE),
        pl.FEATURES_PODADAS,
        indicador_de_nulo=False,
    )
    modelo.fit(X, y)
    assert np.isfinite(modelo[-1].coef_).all()


# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
def test_feature_store_e_deterministica(store):
    """Duas construções seguidas produzem exatamente os mesmos números."""
    outra = fs.montar_feature_store()
    for nivel in store:
        pd.testing.assert_frame_equal(store[nivel], outra[nivel])


def test_dataset_em_disco_reproduz_a_feature_store(dataset, store):
    """O Parquet gravado é o que o código produz hoje, não um resíduo antigo."""
    amostra = dataset.sample(50_000, random_state=config.RANDOM_STATE)
    esperado = amostra["id_municipio"].map(store["municipio"]["mun_taxa_alfab_lag1"])
    pd.testing.assert_series_equal(
        amostra["mun_taxa_alfab_lag1"], esperado, check_names=False, rtol=1e-9
    )
    assert build_dataset.COLS_COORTE  # o contrato de colunas da coorte não sumiu


def test_suavizacao_usa_k_fixo_e_documentado():
    """`k = 30`, sem calibração por CV, e escola sem histórico herda o município.

    A AUC da taxa suavizada cresce monotonicamente com `k` até virar a taxa
    municipal (0,561 em k=0, 0,653 em k=500), então não há ótimo interior a
    calibrar: um `GridSearchCV` devolveria o maior valor da grade.
    """
    assert fs.K_SUAVIZACAO == 30.0
    coorte = pd.DataFrame(
        {
            "esc_n_alunos_lag1": [0.0, 30.0, np.nan],
            "esc_taxa_alfab_lag1": [0.80, 0.80, np.nan],
            "mun_taxa_alfab_lag1": [60.0, 60.0, 60.0],
        }
    )
    suavizada = fs.suavizar_taxa_escola(coorte)
    assert suavizada.iloc[0] == pytest.approx(60.0)   # escola sem aluno: só o município
    assert suavizada.iloc[1] == pytest.approx(70.0)   # n = k: média das duas
    assert suavizada.iloc[2] == pytest.approx(60.0)   # escola sem histórico: só o município


def test_agregado_ponderado_reconcilia_com_o_inep():
    """A tabela populacional da camada estratégica bate com o INEP (B2).

    É a mesma exigência do `test_peso_aluno_e_o_fator_de_nao_resposta_do_inep`,
    aplicada agora ao artefato que a Etapa 6 vai publicar, e não ao microdado
    cru: o que sai do projeto tem de reconciliar a menos de 0,10pp.
    """
    tabela = fs.agregados_municipais_ponderados(config.ANO_LAG).set_index("id_municipio")
    inep = loader.carregar_agregado_municipio(apenas_publica=True)
    inep = inep[inep["ano"] == config.ANO_LAG].set_index("id_municipio")["taxa_alfabetizacao"]

    par = pd.concat([tabela["taxa_alfab_ponderada"] * 100, inep], axis=1).dropna()
    assert len(par) > 4_500
    mae_ponderada = float((par.iloc[:, 0] - par.iloc[:, 1]).abs().mean())
    par_simples = pd.concat([tabela["taxa_alfab_simples"] * 100, inep], axis=1).dropna()
    mae_simples = float((par_simples.iloc[:, 0] - par_simples.iloc[:, 1]).abs().mean())

    assert mae_ponderada < 0.10, f"MAE ponderada de {mae_ponderada:.3f}pp"
    assert mae_ponderada < mae_simples, "o peso deixou de melhorar a reconciliação"
