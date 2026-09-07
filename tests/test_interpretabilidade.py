"""Invariantes da leitura de importância (Etapa 5).

A Etapa 5 não treina nada, então o que pode dar errado aqui não é o modelo: é a
**atribuição**. Um SHAP creditado à variável errada, uma dummy que volta para a
família vizinha, uma família que perde uma coluna no caminho — nada disso quebra
a execução, e tudo isso muda a resposta à pergunta "quais fatores mais impactam a
alfabetização". Os testes abaixo travam a cadeia de atribuição inteira: 89 colunas
para 20 features, 20 features para 5 famílias, e o valor SHAP conferido contra a
própria predição do modelo.

Os que dependem do campeão ou dos CSV da etapa são pulados quando os artefatos
não existem, para que um clone limpo rode `pytest -q` antes de treinar.

Executar:  pytest -q
"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit
from scipy.stats import spearmanr

from src import config
from src.data import loader
from src.evaluation import interpret as it
from src.modeling import campeao, split
from src.preprocessing import pipeline as pl


@pytest.fixture(scope="module")
def pacote():
    if not campeao.MODELO_CAMPEAO.exists():
        pytest.skip("campeão ainda não treinado — rode `python -m src.modeling.train campeao`")
    return campeao.carregar_campeao()


@pytest.fixture(scope="module")
def mapa(pacote):
    return it.mapear_colunas_para_features(pacote["modelo"].estimador_.named_steps["pre"])


@pytest.fixture(scope="module")
def artefatos():
    """Os CSV da etapa. Sem eles não há o que conferir, e o teste se declara pulado."""
    caminhos = {
        "familias": it.CSV_FAMILIAS,
        "permutacao": it.CSV_PERMUTACAO,
        "triangulacao": it.CSV_TRIANGULACAO,
        "coeficientes": it.CSV_COEFICIENTES,
        "shap_features": it.CSV_SHAP_FEATURES,
    }
    faltando = [nome for nome, caminho in caminhos.items() if not caminho.exists()]
    if faltando or not it.JSON_INTERPRET.exists():
        pytest.skip("interpretabilidade ainda não gerada — rode `python -m src.evaluation.interpret`")
    tabelas = {nome: pd.read_csv(caminho) for nome, caminho in caminhos.items()}
    tabelas["resumo"] = json.loads(it.JSON_INTERPRET.read_text(encoding="utf-8"))
    return tabelas


# ---------------------------------------------------------------------------
# O mapa de atribuição
# ---------------------------------------------------------------------------
def test_as_cinco_familias_particionam_as_vinte_features():
    """Partição, não cobertura: nada de fora e nada em duas famílias ao mesmo tempo.

    Uma feature ausente do mapa apareceria no SHAP por coluna e sumiria da leitura
    por família, e a soma das participações continuaria dando 100% — o erro não
    se denunciaria em lugar nenhum do relatório.
    """
    familias = it.familia_de_cada_feature()
    assert len(familias) == len(pl.FEATURES_MODELO) == 20
    assert set(familias["feature"]) == set(pl.FEATURES_MODELO)
    assert not familias["feature"].duplicated().any()
    assert set(familias["familia"]) == {
        "historico_municipal", "territorial", "metas", "historico_escolar", "estrutural"
    }


def test_dummy_com_origem_ambigua_falha_em_vez_de_escolher():
    """Prefixo que casa com duas colunas tem de parar a execução.

    O caso não é hipotético: `fonte_lag_municipal` e `fonte_lag_uf` compartilham
    prefixo, e uma renomeação futura para `fonte_lag` colocaria as dummies de uma
    dentro da outra sem nenhum sintoma visível no gráfico.
    """
    assert it._origem_das_dummies(["rede_grupo_Estadual"], ["rede_grupo", "nome_regiao"]) == ["rede_grupo"]
    with pytest.raises(AssertionError, match="ambígua"):
        it._origem_das_dummies(["fonte_lag_uf_gold"], ["fonte_lag", "fonte_lag_uf"])


def test_o_mapa_reconstroi_as_89_colunas_do_pre_processador(pacote, mapa):
    """Reconstrução conferida posição a posição contra `get_feature_names_out()`.

    A asserção mora dentro de `mapear_colunas_para_features`; aqui se trava o
    número publicado e a composição: 89 colunas, 20 features de origem, e os 12
    indicadores de nulo — as duas colunas `tem_historico_*` não têm nulo e por
    isso não geram indicador.
    """
    assert len(mapa) == len(pacote["nomes_apos_pre_processamento"]) == 89
    assert mapa["coluna"].tolist() == list(pacote["nomes_apos_pre_processamento"])
    assert set(mapa["feature"]) == set(pl.FEATURES_MODELO)
    assert mapa["tipo"].value_counts().to_dict() == {"dummy": 63, "valor": 14, "indicador_de_nulo": 12}
    assert not mapa["familia"].isna().any()


def test_a_matriz_reconstruida_e_a_que_o_modelo_ve(pacote, mapa):
    """O dado cru guardado na amostra reproduz a matriz transformada exatamente.

    É o que permite não gravar 18 MB de matriz padronizada ao lado dos valores
    SHAP. Se a reconstrução divergisse, o `beeswarm` coloriria cada ponto com o
    valor de outra linha.
    """
    if not it.PARQUET_SHAP.exists():
        pytest.skip("amostra de SHAP ainda não gerada")
    amostra = pd.read_parquet(it.PARQUET_SHAP).head(2_000)
    reconstruida = it.reconstruir_matriz(amostra, pacote)
    esperada = pacote["modelo"].estimador_[:-1].transform(
        amostra[[f"x__{c}" for c in pacote["features"]]].rename(columns=lambda c: c[3:])
    )
    assert reconstruida.columns.tolist() == list(pacote["nomes_apos_pre_processamento"])
    np.testing.assert_allclose(reconstruida.to_numpy(), np.asarray(esperada), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# A agregação por família
# ---------------------------------------------------------------------------
def test_agrupamento_mede_cancelamento_dentro_da_familia():
    """Duas colunas que se anulam têm importância somada alta e líquida zero.

    É o caso que justifica reportar `media_do_abs_da_soma` como número principal.
    Empilhar médias de |SHAP| daria a essa família o dobro da importância de uma
    coluna que, na prática, não move a predição de nenhum aluno.
    """
    valores = np.array([0.4, -0.2, 0.9, -0.5])
    amostra = pd.DataFrame(
        {"shap__a": valores, "shap__b": -valores, "shap__c": np.full(4, 0.3)}
    )
    mapa = pd.DataFrame(
        {
            "coluna": ["a", "b", "c"],
            "feature": ["f1", "f1", "f2"],
            "familia": ["par_que_se_anula", "par_que_se_anula", "sozinha"],
        }
    )
    tabela = it._agrupar_shap(amostra, mapa, "familia", "teste").set_index("familia")

    assert tabela.loc["par_que_se_anula", "media_do_abs_da_soma"] == pytest.approx(0.0)
    assert tabela.loc["par_que_se_anula", "soma_das_medias_abs"] == pytest.approx(2 * np.abs(valores).mean())
    assert tabela.loc["par_que_se_anula", "cancelamento_interno"] == pytest.approx(1.0)
    # Família de uma coluna não tem o que cancelar — serve de conferência do zero.
    assert tabela.loc["sozinha", "cancelamento_interno"] == pytest.approx(0.0)
    assert tabela["n_colunas"].sum() == len(mapa)


def test_shap_soma_exatamente_a_predicao_do_modelo(pacote, mapa):
    """Aditividade: base + soma dos 89 valores = log-odds predito, aluno a aluno.

    É a única prova de que o `TreeExplainer` foi aplicado ao estimador certo e
    sobre a matriz certa. Explicar o `Pipeline` inteiro, ou uma matriz montada
    com outra ordem de colunas, produziria valores plausíveis e errados — e
    nenhum gráfico denunciaria isso.
    """
    shap = pytest.importorskip("shap")
    dados = loader.carregar_dataset_modelagem()
    X, _, grupos = pl.separar_X_y(dados)
    _, teste = split.separar_desenvolvimento_e_teste(grupos, seed=pacote["seed"])
    amostra = X.iloc[teste[:3_000]]

    estimador = pacote["modelo"].estimador_
    matriz = pd.DataFrame(estimador[:-1].transform(amostra), columns=pacote["nomes_apos_pre_processamento"])
    explicador = shap.TreeExplainer(estimador.steps[-1][1])
    valores = explicador.shap_values(matriz, check_additivity=False)

    reconstruido = valores.sum(axis=1) + float(np.ravel(explicador.expected_value)[0])
    np.testing.assert_allclose(reconstruido, logit(estimador.predict_proba(amostra)[:, 1]), atol=1e-8)


# ---------------------------------------------------------------------------
# O que a etapa publicou
# ---------------------------------------------------------------------------
def test_o_caderno_continua_abaixo_do_piso_de_ruido(artefatos):
    """O controle negativo da Etapa 4 tem de continuar se comportando como tal.

    Se a permutação do `caderno` passar de 0,002 numa reexecução, o achado a
    reportar é sobreajuste a ruído, e não uma leitura educacional sobre versões
    de prova.
    """
    permutacao = artefatos["permutacao"].set_index("feature")
    assert not bool(permutacao.loc["caderno", "acima_do_piso"])
    assert abs(float(permutacao.loc["caderno", "queda_media"])) < it.PISO_DE_RUIDO
    assert float(permutacao.loc["mun_media_portugues_lag1", "queda_media"]) > 10 * it.PISO_DE_RUIDO


def test_nada_abaixo_do_piso_de_ruido_e_promovido_a_fator(artefatos):
    """A regra de decisão da triangulação, travada contra afrouxamento futuro.

    `fator` é o rótulo que vai virar frase de relatório e recomendação de
    política. Ele exige consenso entre as leituras **e** queda acima do controle
    negativo; sem essa segunda parte, bastaria uma técnica se entusiasmar com uma
    variável para ela virar achado.
    """
    triangulacao = artefatos["triangulacao"]
    fatores = triangulacao[triangulacao["veredito"] == "fator"]
    assert (fatores["permutacao"] > it.PISO_DE_RUIDO).all()
    assert (fatores["posicao_shap"] <= it.TOPO_FEATURES).all()
    assert set(triangulacao["recorte"]) == {"global", "gold"}


def test_a_leitura_na_fatia_gold_nao_contradiz_a_global(artefatos):
    """Se as duas ordenações divergissem, a global seria artefato da coalescência.

    A Etapa 4 mediu AUC 0,6706 onde o lag vem da Gold contra 0,5598 no agregado
    `rede 3`, e alertou que uma leitura única mistura populações que o modelo
    entende de formas diferentes. O teste mede a concordância em vez de assumi-la.
    """
    tabela = artefatos["shap_features"].pivot(
        index="feature", columns="recorte", values="media_do_abs_da_soma"
    )
    correlacao = spearmanr(tabela["global"], tabela["gold"]).statistic
    assert correlacao > 0.9, f"as duas leituras discordam (Spearman {correlacao:.3f})"
    assert tabela["global"].idxmax() == tabela["gold"].idxmax()


def test_a_familia_municipal_domina_as_duas_leituras(artefatos):
    """H1 em forma de asserção: contexto municipal no topo do SHAP e da permutação.

    A hipótese foi escrita na EDA com critério de falseamento explícito. Travá-la
    aqui não é redundância com o relatório: é garantir que uma mudança futura de
    features ou de mapa que derrube H1 apareça como teste vermelho, e não como um
    parágrafo desatualizado.
    """
    familias = artefatos["familias"]
    global_ = familias[familias["recorte"] == "global"].sort_values("participacao", ascending=False)
    assert global_["familia"].iloc[0] == "historico_municipal"
    assert global_["participacao"].iloc[0] > 0.4
    assert global_["participacao"].sum() == pytest.approx(1.0)
    por_permutacao = global_.sort_values("queda_permutacao_em_bloco", ascending=False)
    assert por_permutacao["familia"].iloc[0] == "historico_municipal"


def test_os_coeficientes_da_logistica_sao_estaveis_entre_folds(artefatos):
    """Coeficiente que troca de sinal entre folds não sustenta leitura de direção.

    Vale para os que importam: colunas cuja contribuição é desprezível podem
    oscilar à vontade, porque nada é afirmado sobre elas. O corte é a mediana da
    contribuição, para que o teste não dependa de um limiar arbitrário.
    """
    coeficientes = artefatos["coeficientes"]
    relevantes = coeficientes[coeficientes["contribuicao_dp"] >= coeficientes["contribuicao_dp"].median()]
    instaveis = relevantes.loc[relevantes["troca_de_sinal_entre_folds"], "coluna"].tolist()
    assert not instaveis, f"coeficientes com sinal instável entre folds: {instaveis}"


def test_o_julgamento_das_hipoteses_carrega_o_criterio(artefatos):
    """Veredito sem critério registrado é opinião com aparência de medida."""
    hipoteses = {h["hipotese"]: h for h in artefatos["resumo"]["hipoteses"]}
    assert set(hipoteses) == {"H1", "H2", "H5"}
    for hipotese in hipoteses.values():
        assert hipotese["criterio_de_falseamento"]
        assert hipotese["veredito"] in {"sustentada", "sustentada em parte", "derrubada", "não verificável"}
        assert hipotese["medido"]
    assert artefatos["resumo"]["hash_dataset"] == campeao.hash_do_dataset()
    assert artefatos["resumo"]["permutacao"]["piso_de_ruido"] == it.PISO_DE_RUIDO
    assert artefatos["resumo"]["seed"] == config.RANDOM_STATE
