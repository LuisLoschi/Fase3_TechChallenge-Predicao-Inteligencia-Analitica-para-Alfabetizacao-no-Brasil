"""Testes de integridade dos dados e das armadilhas documentadas no plano.

Cada teste trava uma descoberta do diagnóstico (Etapa 0) para que ela não seja
silenciosamente violada mais tarde. Os testes de vazamento ligados ao split e
ao modelo entram na Etapa 4, em `tests/test_modelagem.py`.

Executar:  pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import loader

LINHAS_GOLD_2024 = 1_852_788
LINHAS_GOLD_2023 = 1_503_058


@pytest.fixture(scope="module")
def aluno_2024() -> pd.DataFrame:
    return loader.carregar_aluno(ano=2024)


@pytest.fixture(scope="module")
def aluno_2023() -> pd.DataFrame:
    return loader.carregar_aluno(ano=2023)


# ---------------------------------------------------------------------------
# Reconciliação entre a Gold e o microdado bruto
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ano,linhas", [(2023, LINHAS_GOLD_2023), (2024, LINHAS_GOLD_2024)])
def test_gold_reconcilia_com_microdado(ano, linhas):
    """A Gold é exatamente `presenca = 1` do microdado — por chave, não por contagem.

    A versão anterior deste teste comparava só `len()`, e uma Gold com o
    subconjunto *errado* e a mesma contagem passaria. Aqui se compara o conjunto
    de `id_aluno` e a igualdade do alvo linha a linha, que é o que de fato
    autoriza usar o microdado como fonte para a coorte da Gold.
    """
    gold = loader.carregar_gold(ano=ano, colunas=["id_aluno", "alfabetizado"])
    micro = loader.carregar_aluno(ano=ano, colunas=["id_aluno", "presenca", "alfabetizado"])
    presentes = micro[micro["presenca"] == 1]

    assert len(gold) == linhas
    assert len(presentes) == linhas
    assert gold["id_aluno"].is_unique, "a Gold duplicou alunos no ETL"
    assert set(gold["id_aluno"]) == set(presentes["id_aluno"]), "conjunto de chaves divergente"

    par = gold.merge(presentes, on="id_aluno", suffixes=("_gold", "_micro"))
    assert len(par) == linhas
    divergentes = int((par["alfabetizado_gold"] != par["alfabetizado_micro"]).sum())
    assert divergentes == 0, f"{divergentes} alunos com alvo diferente entre Gold e microdado"


# ---------------------------------------------------------------------------
# A1 — o alvo é uma função determinística da proficiência
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ano", [2023, 2024])
def test_alvo_e_corte_deterministico_da_proficiencia(ano):
    """`alfabetizado = 1[proficiencia >= 743]`, exatamente, nos dois anos.

    Por isso `proficiencia` do ano corrente não é "quase" vazamento: é o alvo
    reescrito. Se este teste falhar, a definição do alvo mudou na origem.
    """
    df = loader.carregar_aluno(ano=ano, colunas=["presenca", "alfabetizado", "proficiencia"])
    df = df[df["presenca"] == 1].dropna(subset=["proficiencia"])
    maximo_nao_alfabetizados = df.loc[df["alfabetizado"] == 0, "proficiencia"].max()
    minimo_alfabetizados = df.loc[df["alfabetizado"] == 1, "proficiencia"].min()
    assert maximo_nao_alfabetizados < config.LIMIAR_ALFABETIZACAO <= minimo_alfabetizados


def test_proficiencia_ausente_da_gold():
    """A Gold não pode expor `proficiencia` nem `peso_aluno` do ano corrente."""
    colunas = set(loader.carregar_gold(ano=2024).head(1).columns)
    assert not colunas & {"proficiencia", "peso_aluno"}


# ---------------------------------------------------------------------------
# A2 — `id_aluno` NÃO é chave longitudinal
# ---------------------------------------------------------------------------
def test_id_aluno_nao_e_chave_longitudinal(aluno_2023, aluno_2024):
    """IDs repetidos entre anos quase nunca são o mesmo aluno.

    Há ~1,5 milhão de `id_aluno` presentes nos dois anos, mas quase nenhum na
    mesma escola: é ID reciclado a cada ano, não matrícula. Um join por
    `id_aluno` entre 2023 e 2024 produziria milhões de pares falsos.
    """
    a23 = aluno_2023[["id_aluno", "id_escola"]].drop_duplicates("id_aluno")
    a24 = aluno_2024[["id_aluno", "id_escola"]].drop_duplicates("id_aluno")
    pares = a23.merge(a24, on="id_aluno", suffixes=("_23", "_24"))
    assert len(pares) > 1_000_000, "esperado um volume alto de IDs coincidentes"
    mesma_escola = (pares["id_escola_23"] == pares["id_escola_24"]).mean()
    assert mesma_escola < 0.01, (
        f"{mesma_escola:.2%} dos IDs coincidem na mesma escola — se subir muito, "
        "reavaliar a hipótese de que id_aluno não acompanha o aluno"
    )


# ---------------------------------------------------------------------------
# A3 — `id_aluno` codifica a UF no prefixo
# ---------------------------------------------------------------------------
def test_id_aluno_codifica_uf_no_prefixo(aluno_2024):
    """Os 2 primeiros dígitos de `id_aluno` reproduzem o código de UF do IBGE.

    Como numérica, a variável seria um proxy territorial disfarçado — daí ela
    estar em `config.COLS_PROIBIDAS`.
    """
    amostra = aluno_2024.sample(200_000, random_state=config.RANDOM_STATE)
    prefixo_aluno = amostra["id_aluno"].astype("int64") // 10**6
    prefixo_municipio = amostra["id_municipio"].astype("int64") // 10**5
    assert (prefixo_aluno == prefixo_municipio).mean() > 0.99


# ---------------------------------------------------------------------------
# Dimensão territorial e cobertura
# ---------------------------------------------------------------------------
def test_dim_municipio_e_um_para_um():
    dim = loader.carregar_dim_municipio()
    assert dim["id_municipio"].is_unique
    assert dim["sigla_uf"].nunique() == 26, "Roraima está ausente da base — 26 UFs, não 27"


def test_meta_municipio_e_um_por_municipio_ano():
    """A base de metas cobre só a rede Municipal, uma linha por município-ano.

    Pode ser unida à coorte direto por (`ano`, `id_municipio`) sem agregar.
    """
    meta = loader.carregar_meta_municipio()
    assert meta["rede"].unique().tolist() == ["Municipal"]
    assert meta.set_index(["ano", "id_municipio"]).index.is_unique


def test_agregado_municipio_tem_grao_por_rede():
    """O agregado do INEP **não** é um-por-município: tem uma linha por rede.

    Quem fizer merge direto por (`ano`, `id_municipio`) multiplica as linhas da
    coorte. É preciso escolher a rede antes do join.
    """
    agregado = loader.carregar_agregado_municipio()
    assert not agregado.set_index(["ano", "id_municipio"]).index.is_unique
    assert agregado.set_index(["ano", "id_municipio", "rede"]).index.is_unique


def test_target_balanceado_dispensa_reamostragem(aluno_2024):
    """Prevalência ~40/60 entre presentes: SMOTE destruiria a calibração."""
    taxa = aluno_2024.loc[aluno_2024["presenca"] == 1, "alfabetizado"].mean()
    assert 0.55 < taxa < 0.65


def test_ausentes_sao_todos_nao_alfabetizados(aluno_2024):
    """`presenca = 0` implica `alfabetizado = 0` — ausência não é diagnóstico.

    Justifica o filtro `presenca = 1`: incluir ausentes ensinaria o modelo a
    prever falta, não alfabetização.
    """
    ausentes = aluno_2024[aluno_2024["presenca"] == 0]
    assert len(ausentes) > 0
    assert ausentes["alfabetizado"].max() == 0
    assert ausentes["proficiencia"].isna().all()


def test_serie_e_constante(aluno_2024):
    """Cardinalidade 1 → sem variância → descartada das features."""
    assert aluno_2024["serie"].nunique() == 1
    assert "serie" in config.COLS_PROIBIDAS


def test_cobertura_do_lag_de_escola(aluno_2023, aluno_2024):
    """Parcela dos alunos de 2024 cuja escola existe em 2023.

    Sustenta a flag `tem_historico_escola`: o lag não cobre todo mundo, e a
    ausência precisa ser sinalizada em vez de imputada em silêncio.
    """
    escolas_2023 = set(aluno_2023["id_escola"].unique())
    cobertura = aluno_2024["id_escola"].isin(escolas_2023).mean()
    assert 0.70 < cobertura < 0.90


def _taxa_municipal_do_microdado(ano: int, ponderada: bool) -> pd.Series:
    """Taxa de alfabetização por município entre presentes das redes públicas."""
    df = loader.carregar_aluno(
        ano=ano, colunas=["id_municipio", "rede", "presenca", "alfabetizado", "peso_aluno"]
    )
    df = df[(df["presenca"] == 1) & (df["rede"].isin([2, 3]))]
    if not ponderada:
        return df.groupby("id_municipio")["alfabetizado"].mean() * 100
    peso = df[config.COL_PESO]
    somas = df.assign(_num=df["alfabetizado"] * peso, _den=peso).groupby("id_municipio")[["_num", "_den"]].sum()
    return (somas["_num"] / somas["_den"]) * 100


@pytest.mark.parametrize("ano,mae_maxima", [(2023, 1.0), (2024, 0.5)])
def test_rede_5_do_agregado_reconcilia_nos_dois_anos(ano, mae_maxima):
    """`rede = 5` reproduz a taxa das redes 2+3 do microdado — nos DOIS anos.

    A versão anterior só testava 2024, e a nota do `config.py` citava o erro de
    2024 (0,37pp) como se valesse para o projeto todo. Em 2023, que é justamente
    o ano das features de lag, o erro é 2,6x maior: 0,99pp, com 124 municípios
    acima de 5pp. Este teste trava os dois números separadamente.
    """
    micro = _taxa_municipal_do_microdado(ano, ponderada=False)
    agregado = loader.carregar_agregado_municipio(apenas_publica=True)
    agregado = agregado[agregado["ano"] == ano].set_index("id_municipio")["taxa_alfabetizacao"]

    comparacao = pd.concat([micro.rename("micro"), agregado.rename("inep")], axis=1).dropna()
    assert len(comparacao) > 4_500
    assert np.abs(comparacao["micro"] - comparacao["inep"]).mean() < mae_maxima


@pytest.mark.parametrize("ano", [2023, 2024])
def test_peso_aluno_e_o_fator_de_nao_resposta_do_inep(ano):
    """A taxa publicada pelo INEP é a média de `alfabetizado` ponderada por `peso_aluno`.

    Sem o peso a reconciliação municipal fica em ~1pp; com ele cai para ~0,05pp.
    É a prova de que `peso_aluno` não é metadado descartável: é a correção oficial
    de não-resposta, e todo número populacional do projeto tem de usá-la
    (`config.COL_PESO`). Como preditor continua proibido.
    """
    ponderada = _taxa_municipal_do_microdado(ano, ponderada=True)
    agregado = loader.carregar_agregado_municipio(apenas_publica=True)
    agregado = agregado[agregado["ano"] == ano].set_index("id_municipio")["taxa_alfabetizacao"]

    comparacao = pd.concat([ponderada.rename("pond"), agregado.rename("inep")], axis=1).dropna()
    mae = float(np.abs(comparacao["pond"] - comparacao["inep"]).mean())
    assert mae < 0.10, f"MAE ponderada de {mae:.3f}pp — o peso deixou de reconciliar"


def test_coalescencia_de_rede_e_necessaria_para_cobrir_sao_paulo():
    """A coalescência `5 -> 3` recupera 8,86pp da coorte que `rede = 5` não alcança.

    São 676 municípios presentes em 2024 e ausentes do microdado de 2023 — 99,9%
    deles em SP, DF e AC inteiros. O agregado do INEP de 2023 cobre 625 desses
    municípios sob `rede = 3` (Municipal) e só 79 sob `rede = 5` (Pública), que é
    a rede mais fiel mas a de menor cobertura. Medido sobre a coorte de 2024:
    rede 5 sozinha resgata 12,33% e deixa um gap de 10,77%; a coalescência
    resgata 21,20% e deixa 1,91%. Este teste trava a razão de existir de
    `config.REDE_AGREGADO_COALESCENCIA`.
    """
    g23 = loader.carregar_gold(ano=2023, colunas=["id_municipio"])
    g24 = loader.carregar_gold(ano=2024, colunas=["id_municipio"])
    sem_microdado = set(g24["id_municipio"]) - set(g23["id_municipio"])
    assert len(sem_microdado) > 600

    agregado = loader.carregar_agregado_municipio()
    a23 = agregado[(agregado["ano"] == 2023) & agregado["taxa_alfabetizacao"].notna()]
    cobre = {r: sem_microdado & set(a23.loc[a23["rede"] == r, "id_municipio"]) for r in (5, 3)}

    assert len(cobre[5]) < 100, "rede 5 passou a cobrir o gap — revisar a coalescência"
    assert len(cobre[3]) > 600, "rede 3 deixou de cobrir o gap — revisar a coalescência"
    assert cobre[5] <= cobre[3], "rede 5 deixou de ser subconjunto de rede 3 no gap"

    resgate_rede_5 = g24["id_municipio"].isin(cobre[5]).mean()
    resgate_coalescido = g24["id_municipio"].isin(cobre[5] | cobre[3]).mean()
    ganho = resgate_coalescido - resgate_rede_5

    assert ganho > 0.08, f"a coalescência só recupera {ganho:.2%} da coorte — reavaliar"
    residual = g24["id_municipio"].isin(sem_microdado - (cobre[5] | cobre[3])).mean()
    assert residual < 0.03, f"gap residual de {residual:.2%} após a coalescência"


def test_gap_de_lag_sao_tres_ufs_inteiras():
    """O "23% sem histórico" não é ruído difuso: são SP, DF e AC inteiros.

    Consequência de modelagem: `tem_historico_municipio` é, na prática, um
    indicador de "é de São Paulo", colinear com `sigla_uf`. Reportar métrica
    global sem estratificar por essa flag mistura duas populações diferentes.
    """
    g23 = loader.carregar_gold(ano=2023, colunas=["id_municipio"])
    g24 = loader.carregar_gold(ano=2024, colunas=["id_municipio", "sigla_uf"])
    sem_historico = g24["id_municipio"].isin(set(g24["id_municipio"]) - set(g23["id_municipio"]))

    assert 0.20 < sem_historico.mean() < 0.26
    concentracao = g24.loc[sem_historico, "sigla_uf"].isin(["SP", "DF", "AC"]).mean()
    assert concentracao > 0.99, f"só {concentracao:.2%} do gap está em SP/DF/AC"

    ufs_ausentes = set(g24["sigla_uf"]) - set(
        loader.carregar_gold(ano=2023, colunas=["sigla_uf"])["sigla_uf"]
    )
    assert ufs_ausentes == {"SP", "DF", "AC"}


@pytest.mark.parametrize("ano,corr_minima,mae_maxima", [(2023, 0.95, 0.5), (2024, 0.999, 0.05)])
def test_participacao_e_a_presenca_do_proprio_ano(ano, corr_minima, mae_maxima):
    """`percentual_participacao_municipio` de um ano É a taxa de presença desse ano.

    Não é meta. A coluna que vem na linha de 2024 reproduz a presença observada
    em 2024 com corr 1,0000 e MAE 0,002pp — a mesma coorte que se quer predizer,
    medida no instante da prova. Por isso saiu das features e entrou em
    `COLS_PROIBIDAS` (vetor 14 da matriz anti-leakage).

    O teste roda nos dois anos de propósito, porque a distinção é temporal e não
    da coluna: em 2023 a mesma variável reproduz a presença de 2023 (corr 0,971)
    e ali é **lag legítimo**. O que é proibido é a versão do ano corrente.
    """
    assert "percentual_participacao_municipio" in config.COLS_PROIBIDAS
    assert "percentual_participacao_uf" in config.COLS_PROIBIDAS

    micro = loader.carregar_aluno(ano=ano, colunas=["id_municipio", "rede", "presenca"])
    presenca = micro[micro["rede"] == 3].groupby("id_municipio")["presenca"].mean() * 100

    gold = loader.carregar_gold(
        ano=ano, colunas=["id_municipio", "rede", "percentual_participacao_municipio"]
    )
    publicado = (
        gold[gold["rede"] == 3].groupby("id_municipio")["percentual_participacao_municipio"].first()
    )

    par = pd.concat([presenca.rename("obs"), publicado.rename("gold")], axis=1).dropna()
    assert len(par) > 4_000
    assert par["obs"].corr(par["gold"]) > corr_minima
    assert np.abs(par["obs"] - par["gold"]).mean() < mae_maxima


@pytest.mark.parametrize("ano,esperados", [(2023, 249), (2024, 936)])
def test_presentes_sem_proficiencia_tem_alvo_zero_por_ausencia_de_medida(ano, esperados):
    """Alunos com `presenca = 1` e proficiência nula recebem `alfabetizado = 0`.

    É imputação silenciosa do ETL: ausência de medida virou "não alfabetizado",
    o mesmo mecanismo dos `presenca = 0` que a Gold já filtra. São poucas linhas
    (0,02% em 2023 e 0,05% em 2024), mas o alvo delas não é um desfecho — é um
    registro faltante, e por isso saem do treino junto com `preenchimento_caderno`.
    """
    df = loader.carregar_aluno(ano=ano, colunas=["presenca", "alfabetizado", "proficiencia"])
    sem_medida = df[(df["presenca"] == 1) & (df["proficiencia"].isna())]
    assert len(sem_medida) == esperados
    assert sem_medida["alfabetizado"].max() == 0
