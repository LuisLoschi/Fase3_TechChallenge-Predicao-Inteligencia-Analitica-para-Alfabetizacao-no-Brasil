"""Regressões dos problemas encontrados na revisão científica."""

import numpy as np
import pandas as pd
import pytest

from scripts.experimento_b5 import selecionar_desenvolvimento
from src import config
from src.modeling import campeao, split
from src.modeling.metas import ProjetorMetas, avaliar_municipios


def _municipios(n=100):
    rng = np.random.default_rng(42)
    taxa = rng.uniform(10, 90, n)
    return pd.DataFrame({
        "id_municipio": np.arange(n), "taxa_2023": taxa,
        "taxa_2024": np.clip(taxa + rng.normal(2, 12, n), 0, 100),
        "meta_2024": taxa + 3, "n_alunos_2023": rng.integers(15, 1000, n),
    })


def test_b5_exclui_reserva_antes_de_amostrar():
    dados = pd.DataFrame({config.GRUPO_CV: np.repeat(np.arange(80), 10), config.TARGET: 0})
    _, teste = split.separar_desenvolvimento_e_teste(dados[config.GRUPO_CV])
    for tamanho in (None, 200):
        dev = selecionar_desenvolvimento(dados, tamanho)
        assert not set(dev[config.GRUPO_CV]) & set(dados.iloc[teste][config.GRUPO_CV])
        assert dev.groupby(config.GRUPO_CV).size().eq(10).all()


def test_desfecho_do_fold_nao_altera_suas_proprias_predicoes():
    base = _municipios()
    original = avaliar_municipios(base).set_index("id_municipio")
    ids = original.index[original.fold == 0]
    alterada = base.copy()
    alterada.loc[alterada.id_municipio.isin(ids), "taxa_2024"] = 0
    conferido = avaliar_municipios(alterada).set_index("id_municipio")
    pd.testing.assert_frame_equal(
        original.loc[ids, ["projecao", "inferior", "superior", "p_abaixo", "baseline_prevalencia"]],
        conferido.loc[ids, ["projecao", "inferior", "superior", "p_abaixo", "baseline_prevalencia"]],
    )


def test_metas_rejeita_municipios_duplicados():
    base = _municipios()
    with pytest.raises(ValueError, match="uma linha"):
        avaliar_municipios(pd.concat([base, base.iloc[:1]]))


def test_projecao_respeita_dominio_e_probabilidade_nas_bordas():
    b = _municipios()
    modelo = ProjetorMetas().fit(b.taxa_2023, b.n_alunos_2023, b.taxa_2024)
    pred = modelo.predict([0, 100, 50, 50], [1, 1, 100, 100], [0, 101, 40, 80])
    assert pred[["projecao", "inferior", "superior"]].ge(0).all().all()
    assert pred[["projecao", "inferior", "superior"]].le(100).all().all()
    assert (pred.inferior <= pred.projecao).all() and (pred.projecao <= pred.superior).all()
    assert pred.p_abaixo.iloc[0] == 0 and pred.p_abaixo.iloc[1] == 1
    assert pred.p_abaixo.iloc[2] < pred.p_abaixo.iloc[3]


@pytest.mark.parametrize("taxa,porte", [([], []), ([101], [10]), ([20], [0]), ([np.nan], [20])])
def test_projecao_rejeita_entradas_invalidas(taxa, porte):
    with pytest.raises(ValueError):
        ProjetorMetas().fit(taxa, porte, taxa)


def test_modelo_historico_carrega_limite_da_validacao():
    if not campeao.MODELO_CAMPEAO.exists():
        pytest.skip("artefato histórico não fornecido")
    status = campeao.carregar_campeao()["status_validacao"]
    assert status["teste_independente_da_selecao"] is False
    assert status["validacao_temporal_do_campeao"] is False
