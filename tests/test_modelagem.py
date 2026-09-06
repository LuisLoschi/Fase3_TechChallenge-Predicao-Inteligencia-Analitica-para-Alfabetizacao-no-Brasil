"""Invariantes da modelagem supervisionada (Etapa 4).

`test_dados.py` trava o que é verdade sobre as bases e `test_features.py` sobre o
dataset. Aqui se trava o que é verdade sobre o **desenho de validação** e sobre o
**artefato entregue**: que nenhum município atravessa um split, que o baseline é
o baseline, que a calibração não reordena nada, e que o campeão em disco
reproduz o número que o relatório publica.

Os testes que dependem do artefato são pulados quando ele não existe, para que um
clone limpo consiga rodar `pytest -q` antes de treinar. Rodar
`python -m src.modeling.train campeao` os liga.

Executar:  pytest -q
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from src import config
from src.data import loader
from src.evaluation import metrics as mt
from src.modeling import baselines, calibracao, campeao, split
from src.preprocessing import pipeline as pl

FAIXA_ESPERADA_ROC_AUC = (0.63, 0.70)


@pytest.fixture(scope="module")
def chaves() -> pd.DataFrame:
    """Só as três colunas que o desenho de validação usa — leitura em segundos."""
    return loader.carregar_dataset_modelagem(
        colunas=[config.GRUPO_CV, config.TARGET, "mun_taxa_alfab_lag1"]
    )


@pytest.fixture(scope="module")
def artefato():
    if not campeao.MODELO_CAMPEAO.exists() or not campeao.JSON_CAMPEAO.exists():
        pytest.skip("campeão ainda não treinado — rode `python -m src.modeling.train campeao`")
    return campeao.carregar_campeao(), json.loads(campeao.JSON_CAMPEAO.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Desenho de validação
# ---------------------------------------------------------------------------
def test_nenhum_municipio_atravessa_o_split(chaves):
    """O invariante que sustenta todo número da etapa (vetor 7 da matriz anti-leakage).

    Sem ele, o modelo memoriza o nível de cada município no treino e o
    reencontra no teste: o diagnóstico mediu 0,667 de ROC-AUC no split aleatório
    contra 0,649 no agrupado, e a diferença é memorização, não aprendizado.
    """
    grupos = chaves[config.GRUPO_CV]
    dev, teste = split.separar_desenvolvimento_e_teste(grupos)
    assert not set(grupos.iloc[dev]) & set(grupos.iloc[teste])
    assert len(dev) + len(teste) == len(grupos)

    y_dev = chaves[config.TARGET].iloc[dev].reset_index(drop=True)
    g_dev = grupos.iloc[dev].reset_index(drop=True)
    for treino, validacao in split.folds(y_dev, g_dev):
        assert not set(g_dev.iloc[treino]) & set(g_dev.iloc[validacao])


def test_split_e_deterministico(chaves):
    """Mesma seed, mesmo split — senão nada do que vem depois é reproduzível."""
    grupos = chaves[config.GRUPO_CV]
    primeiro, _ = split.separar_desenvolvimento_e_teste(grupos)
    segundo, _ = split.separar_desenvolvimento_e_teste(grupos)
    np.testing.assert_array_equal(primeiro, segundo)


def test_amostragem_de_tuning_leva_municipios_inteiros(chaves):
    """Sortear alunos individuais reintroduziria a memorização que o desenho evita.

    A checagem é a definição de "município inteiro": todo município tocado pela
    amostra tem exatamente o mesmo número de linhas nela e no universo.
    """
    grupos = chaves[config.GRUPO_CV]
    dev, _ = split.separar_desenvolvimento_e_teste(grupos)
    posicoes = split.amostrar_municipios(grupos, 300_000, posicoes=dev)

    na_amostra = grupos.iloc[posicoes].value_counts()
    no_universo = grupos.iloc[dev].value_counts()
    pd.testing.assert_series_equal(na_amostra, no_universo.loc[na_amostra.index], check_names=False)
    assert set(posicoes).issubset(set(dev))
    assert 250_000 < len(posicoes) < 400_000


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def test_baseline_univariado_preserva_o_ranking_da_coluna(chaves):
    """A isotônica traduz escala, não cria ordenação.

    O baseline existe para ser a barra honesta da regra de uma variável. Se a
    calibração mudasse a AUC, ele deixaria de ser essa regra e viraria um modelo
    — e a comparação perderia o sentido.
    """
    amostra = chaves.sample(200_000, random_state=config.RANDOM_STATE).reset_index(drop=True)
    y = amostra[config.TARGET]
    modelo = baselines.baseline_taxa_municipal().fit(amostra, y)
    escore = modelo.predict_proba(amostra)[:, 1]

    observado = amostra["mun_taxa_alfab_lag1"].notna()
    direto = -amostra.loc[observado, "mun_taxa_alfab_lag1"].to_numpy()
    assert roc_auc_score(y[observado], escore[observado]) == pytest.approx(
        roc_auc_score(y[observado], direto), abs=1e-6
    )
    # Quem não tem histórico municipal são SP, DF e AC: vão para o meio da fila.
    assert np.allclose(escore[~observado], y.mean())


# ---------------------------------------------------------------------------
# Incerteza
# ---------------------------------------------------------------------------
def test_bootstrap_de_municipios_e_mais_largo_que_o_de_alunos():
    """Reamostrar alunos devolveria um intervalo estreito e falso.

    Com risco correlacionado dentro do município, as 2.700 crianças de uma
    cidade não são 2.700 observações independentes. O teste constrói exatamente
    esse dado e exige que o IC agrupado seja no mínimo duas vezes mais largo —
    é a justificativa numérica da escolha feita em `metrics.py`.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    municipio = np.repeat(np.arange(150), 200)
    nivel = rng.normal(0, 1.2, 150)[municipio]
    escore = 1 / (1 + np.exp(-(nivel + rng.normal(0, 0.5, len(municipio)))))
    y = rng.binomial(1, 1 / (1 + np.exp(-nivel)))

    agrupado = mt.bootstrap_por_municipio(y, escore, municipio, n_reamostras=300)
    individual = mt.bootstrap_por_municipio(
        y, escore, np.arange(len(y)), n_reamostras=300
    )
    largura = lambda tabela: float(  # noqa: E731
        (tabela.set_index("metrica").loc["roc_auc", "ic_alto"]
         - tabela.set_index("metrica").loc["roc_auc", "ic_baixo"])
    )
    assert largura(agrupado) > 2 * largura(individual)


# ---------------------------------------------------------------------------
# Calibração
# ---------------------------------------------------------------------------
def test_calibracao_nao_reordena_e_nao_piora_o_brier(chaves):
    """Isotônica com desempate: mesma AUC, Brier igual ou melhor.

    `aplicar=True` força o caminho da isotônica, porque no campeão ela acaba
    desligada pelo próprio critério — e um teste que exercita o ramo de
    identidade não testaria nada. O que se trava aqui é que, quando ligada, a
    calibração muda a escala e **não** a ordem: se um dia a AUC começar a se
    mover, o ranking municipal da Etapa 6 mudou sem ninguém ter decidido isso.
    """
    from sklearn.linear_model import LogisticRegression

    dados = loader.carregar_dataset_modelagem().sample(
        120_000, random_state=config.RANDOM_STATE
    ).reset_index(drop=True)
    X, y, grupos = pl.separar_X_y(dados, pl.FEATURES_PODADAS)
    particoes = split.folds(y, grupos, n_splits=3)

    base = pl.montar_pipeline(
        LogisticRegression(max_iter=300, random_state=config.RANDOM_STATE),
        pl.FEATURES_PODADAS,
        indicador_de_nulo=False,
    )
    modelo = calibracao.ModeloCalibrado(base, aplicar=True).fit(X, y, particoes=particoes)
    bruto = modelo.escore_bruto(X)
    calibrado = modelo.predict_proba(X)[:, 1]

    assert not np.allclose(calibrado, bruto), "a isotônica não foi aplicada"
    assert roc_auc_score(y, calibrado) == pytest.approx(roc_auc_score(y, bruto), abs=1e-6)
    assert brier_score_loss(y, calibrado) <= brier_score_loss(y, bruto) + 1e-6

    # E a decisão de ligá-la ou não é tomada fora do ajuste da própria curva.
    assert set(modelo.diagnostico_) == {"brier_bruto", "brier_calibrado", "ganho_brier"}


# ---------------------------------------------------------------------------
# O artefato entregue
# ---------------------------------------------------------------------------
def test_campeao_em_disco_reproduz_as_metricas_publicadas(chaves, artefato):
    """O joblib escora o teste e devolve exatamente o ROC-AUC de `campeao.json`.

    É o teste de reprodutibilidade que o plano exige: se alguém retreinar com
    outra seed, mexer no pipeline ou trocar a versão do sklearn e esquecer de
    regravar o JSON, o número do relatório e o do modelo divergem — e este teste
    é o único lugar onde essa divergência aparece.
    """
    pacote, publicado = artefato
    dados = loader.carregar_dataset_modelagem()
    X, y, grupos = pl.separar_X_y(dados)
    _, teste = split.separar_desenvolvimento_e_teste(grupos, seed=pacote["seed"])

    escore = pacote["modelo"].predict_proba(X.iloc[teste])[:, 1]
    assert roc_auc_score(y.iloc[teste], escore) == pytest.approx(
        publicado["teste"]["publicado"]["roc_auc"], abs=1e-9
    )
    assert pacote["hash_dataset"] == publicado["hash_dataset"] == campeao.hash_do_dataset(), (
        "o modelo foi treinado sobre outro arquivo de dados que não o atual"
    )


def test_roc_auc_do_campeao_esta_na_faixa_prevista(artefato):
    """Entre 0,63 e 0,70. Acima de 0,80 é alarme de vazamento, não comemoração.

    A base não tem nenhuma variável sobre a criança — sem nível socioeconômico,
    cor/raça, idade ou frequência. O que se mede é quanto o território determina
    o desfecho individual, e isso tem teto.
    """
    _, publicado = artefato
    auc = publicado["teste"]["publicado"]["roc_auc"]
    baixo, alto = FAIXA_ESPERADA_ROC_AUC
    assert baixo < auc < alto, f"ROC-AUC {auc:.4f} fora da faixa — investigar antes de publicar"


def test_metricas_saem_estratificadas_pelas_duas_dimensoes_obrigatorias(artefato):
    """`tem_historico_municipio` e `fonte_lag_municipal`, exigidas por B3 e B4.

    A mesma feature vale AUC diferente em cada fatia porque são apurações
    diferentes convivendo na mesma coluna. Um número agregado esconde isso.
    """
    if not campeao.CSV_ESTRATIFICADO.exists():
        pytest.skip("estratificação ainda não gerada")
    tabela = pd.read_csv(campeao.CSV_ESTRATIFICADO)
    assert {"tem_historico_municipio", "fonte_lag_municipal"} <= set(tabela["dimensao"])
    assert (tabela.groupby("dimensao")["n"].sum() > 0).all()


def test_caderno_continua_sendo_controle_negativo():
    """Embaralhar o caderno não pode mover a AUC; embaralhar a feature municipal, sim.

    Cadernos são randomizados entre alunos. Importância alta ali é sobreajuste a
    ruído, e o achado a reportar seria esse — não uma leitura educacional sobre
    versões de prova.
    """
    from src.evaluation import rigor

    if not rigor.CSV_INVARIANCIA.exists():
        pytest.skip("teste de invariância ainda não gerado")
    tabela = pd.read_csv(rigor.CSV_INVARIANCIA).set_index("coluna")
    queda_caderno = float(tabela.loc["caderno", "queda_media"])
    queda_municipal = float(tabela.loc["mun_media_portugues_lag1", "queda_media"])

    assert abs(queda_caderno) < 0.002, f"o caderno moveu a AUC em {queda_caderno:.5f}"
    assert queda_municipal > 20 * abs(queda_caderno), (
        "o controle positivo não se separou do negativo — o teste perdeu sensibilidade"
    )
