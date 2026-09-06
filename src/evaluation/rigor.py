"""Cinco provas que perguntam se o número do campeão significa alguma coisa.

Um ROC-AUC de 0,66 pode ser sinal fraco e real ou pode ser variância bem
apresentada. As provas aqui existem para separar os dois casos, e cada uma
responde a uma pergunta diferente:

    permutação          existe sinal, ou o modelo está aprendendo ruído?
    curva de aprendizado o teto é falta de dado ou falta de informação?
    LOGO por região      generaliza para um território estruturalmente distinto?
    drift 2023 -> 2024   a relação território-risco se mantém entre edições?
    invariância          o modelo ignora o que tem de ignorar?

A prova de permutação tem uma variante que não estava no plano e virou o achado
mais direto da etapa. Embaralhar o alvo globalmente derruba tudo para 0,500 e só
confirma o óbvio. Embaralhar o alvo **dentro de cada município** preserva a taxa
municipal e destrói todo o resto: se a AUC não cair, o modelo não usa nada além
do nível do município — que é uma afirmação bem mais forte, e testável, do que
"o sinal é territorial".
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src import config
from src.data import loader
from src.modeling import split
from src.preprocessing import pipeline as pl

log = logging.getLogger(__name__)

CSV_PERMUTACAO = config.DIR_METRICS / "rigor_permutacao.csv"
CSV_APRENDIZADO = config.DIR_METRICS / "rigor_curva_aprendizado.csv"
CSV_LOGO = config.DIR_METRICS / "rigor_logo_regiao.csv"
CSV_DRIFT = config.DIR_METRICS / "rigor_drift.csv"
CSV_INVARIANCIA = config.DIR_METRICS / "rigor_invariancia.csv"


def _ajustar(parametros, X, y, treino, validacao, features=None, seed=config.RANDOM_STATE):
    modelo = pl.montar_pipeline(LGBMClassifier(random_state=seed, **parametros), features)
    modelo.fit(X.iloc[treino], y.iloc[treino])
    return modelo, modelo.predict_proba(X.iloc[validacao])[:, 1]


# ---------------------------------------------------------------------------
# 1. Permutação — o sinal é real?
# ---------------------------------------------------------------------------
def teste_de_permutacao(
    parametros: dict,
    dados: pd.DataFrame,
    n_permutacoes_global: int = 50,
    n_permutacoes_municipal: int = 20,
    n_splits: int = 3,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Duas distribuições nulas, porque há duas perguntas diferentes.

    `global` embaralha o alvo entre todos os alunos: a hipótese nula é "não há
    relação nenhuma", e a AUC tem de cair para 0,500. `municipal` embaralha o
    alvo **dentro** de cada município, preservando a taxa municipal exata: a
    hipótese nula passa a ser "o modelo não usa nada além do nível do
    município", e é essa que vale a pena testar.

    O p-valor é o percentílico usual, `(1 + #{nulo >= real}) / (1 + n)`, então o
    menor valor observável com 50 permutações é 0,0196.
    """
    X, y, grupos = pl.separar_X_y(dados)
    particoes = split.folds(y, grupos, n_splits=n_splits, seed=seed)
    rng = np.random.default_rng(seed)

    def auc_media(alvo: pd.Series) -> float:
        escores = [
            roc_auc_score(alvo.iloc[validacao], _ajustar(parametros, X, alvo, treino, validacao, seed=seed)[1])
            for treino, validacao in particoes
        ]
        return float(np.mean(escores))

    real = auc_media(y)
    log.info("permutação: AUC real (média de %d folds) %.4f", n_splits, real)

    linhas = [{"tipo": "real", "permutacao": -1, "roc_auc": real}]
    for i in range(n_permutacoes_global):
        embaralhado = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        linhas.append({"tipo": "nulo_global", "permutacao": i, "roc_auc": auc_media(embaralhado)})
        pd.DataFrame(linhas).to_csv(CSV_PERMUTACAO, index=False)

    quadro = pd.DataFrame({"y": y.to_numpy(), "g": grupos.to_numpy()})
    for i in range(n_permutacoes_municipal):
        dentro = quadro.groupby("g")["y"].transform(lambda s: rng.permutation(s.to_numpy()))
        linhas.append(
            {
                "tipo": "nulo_dentro_do_municipio",
                "permutacao": i,
                "roc_auc": auc_media(pd.Series(dentro.to_numpy(), index=y.index)),
            }
        )
        pd.DataFrame(linhas).to_csv(CSV_PERMUTACAO, index=False)

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(CSV_PERMUTACAO, index=False)
    return tabela


def resumo_da_permutacao(tabela: pd.DataFrame | None = None) -> pd.DataFrame:
    tabela = pd.read_csv(CSV_PERMUTACAO) if tabela is None else tabela
    real = float(tabela.loc[tabela["tipo"] == "real", "roc_auc"].iloc[0])
    linhas = []
    for tipo in ("nulo_global", "nulo_dentro_do_municipio"):
        nulo = tabela.loc[tabela["tipo"] == tipo, "roc_auc"].to_numpy()
        if not len(nulo):
            continue
        linhas.append(
            {
                "nulo": tipo,
                "n_permutacoes": len(nulo),
                "roc_auc_real": real,
                "nulo_media": float(nulo.mean()),
                "nulo_dp": float(nulo.std(ddof=1)),
                "nulo_maximo": float(nulo.max()),
                "p_valor": float((1 + (nulo >= real).sum()) / (1 + len(nulo))),
            }
        )
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 2. Curva de aprendizado — o teto é informacional ou amostral?
# ---------------------------------------------------------------------------
def curva_de_aprendizado(
    parametros: dict,
    dados: pd.DataFrame,
    fracoes=(0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0),
    seeds=(42, 7, 2024),
    seed_split: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """AUC de treino e de validação contra o número de **municípios** no treino.

    A abscissa é município, não aluno, porque é essa a unidade de informação: dez
    mil alunos a mais do mesmo município não acrescentam quase nada, e cem
    municípios novos acrescentam muito. Uma curva em número de alunos daria a
    impressão errada de que basta coletar mais dado.

    Cada fração é sorteada com três seeds, e o desvio entre elas é reportado —
    sem isso não há como saber se a diferença entre dois pontos da curva é
    aprendizado ou sorteio.
    """
    X, y, grupos = pl.separar_X_y(dados)
    treino_completo, validacao = split.separar_desenvolvimento_e_teste(grupos, 0.25, seed=seed_split)

    linhas = []
    for fracao in fracoes:
        alvo = int(round(fracao * len(treino_completo)))
        for seed in seeds:
            posicoes = (
                treino_completo
                if fracao >= 1.0
                else split.amostrar_municipios(grupos, alvo, seed=seed, posicoes=treino_completo)
            )
            inicio = time.time()
            modelo, escore = _ajustar(parametros, X, y, posicoes, validacao, seed=config.RANDOM_STATE)
            linhas.append(
                {
                    "fracao": fracao,
                    "seed_amostra": seed,
                    "n_municipios_treino": int(grupos.iloc[posicoes].nunique()),
                    "n_alunos_treino": int(len(posicoes)),
                    "roc_auc_treino": float(
                        roc_auc_score(y.iloc[posicoes], modelo.predict_proba(X.iloc[posicoes])[:, 1])
                    ),
                    "roc_auc_validacao": float(roc_auc_score(y.iloc[validacao], escore)),
                    "segundos": round(time.time() - inicio, 1),
                }
            )
            log.info(
                "curva: %d municípios -> treino %.4f | validação %.4f",
                linhas[-1]["n_municipios_treino"], linhas[-1]["roc_auc_treino"], linhas[-1]["roc_auc_validacao"],
            )
            pd.DataFrame(linhas).to_csv(CSV_APRENDIZADO, index=False)
            if fracao >= 1.0:
                break
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 3. LeaveOneGroupOut por região — o teste mais duro do projeto
# ---------------------------------------------------------------------------
def logo_por_regiao(parametros: dict, dados: pd.DataFrame, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Treina em quatro regiões e testa na quinta, uma vez para cada.

    É mais duro que o split por município porque a região que sobra tem outra
    distribuição de tudo: taxa base, porte, share de rede estadual, cobertura do
    lag. Se a relação "contexto de 2023 → risco em 2024" for a mesma no país, a
    AUC cai pouco; se cair muito, o modelo aprendeu um mapa do Brasil e não uma
    relação.

    A prevalência de cada região é reportada ao lado, porque parte da queda de
    AUC vem de a região sobrante ter uma faixa de risco mais estreita — e isso é
    propriedade da população, não erro do modelo.
    """
    X, y, _ = pl.separar_X_y(dados)
    regiao = dados["nome_regiao"].astype(str)

    linhas = []
    for nome in sorted(regiao.unique()):
        fora = np.flatnonzero((regiao == nome).to_numpy())
        dentro = np.flatnonzero((regiao != nome).to_numpy())
        inicio = time.time()
        _, escore = _ajustar(parametros, X, y, dentro, fora, seed=seed)
        linhas.append(
            {
                "regiao_de_teste": nome,
                "n_treino": int(len(dentro)),
                "n_teste": int(len(fora)),
                "prevalencia_teste": float(y.iloc[fora].mean()),
                "roc_auc": float(roc_auc_score(y.iloc[fora], escore)),
                "segundos": round(time.time() - inicio, 1),
            }
        )
        log.info("LOGO %s: ROC-AUC %.4f", nome, linhas[-1]["roc_auc"])
        pd.DataFrame(linhas).to_csv(CSV_LOGO, index=False)
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 4. Drift — a única validação out-of-time possível nesta base
# ---------------------------------------------------------------------------
FEATURES_DRIFT = ["sigla_uf", "nome_regiao", "rede_grupo"]


def _coorte_territorial(ano: int) -> pd.DataFrame:
    """Alvo e três colunas territoriais, disponíveis identicamente nos dois anos."""
    bruto = loader.carregar_gold(
        ano=ano, colunas=["id_municipio", "sigla_uf", "nome_regiao", "rede", "preenchimento_caderno", "alfabetizado"]
    )
    bruto = bruto[bruto["preenchimento_caderno"] == 1]
    return pd.DataFrame(
        {
            "id_municipio": bruto["id_municipio"].to_numpy(),
            "sigla_uf": bruto["sigla_uf"].astype(str).to_numpy(),
            "nome_regiao": bruto["nome_regiao"].astype(str).to_numpy(),
            "rede_grupo": pd.Series(bruto["rede"].to_numpy()).map({2: "Estadual", 3: "Municipal"}).fillna("Outros").to_numpy(),
            config.TARGET: (1 - bruto["alfabetizado"]).astype("int8").to_numpy(),
        }
    )


def modelo_de_drift(seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Modelo reduzido a território e rede, treinado em 2023 e testado em 2024.

    Validação out-of-time de verdade é impossível aqui: as features de lag dos
    alunos de 2023 exigiriam 2022, que não existe. O que sobra é este modelo
    reduzido, cujas três colunas — UF, região e rede — estão disponíveis
    identicamente nos dois anos. Ele mede a única coisa que interessa saber: a
    relação entre território e risco se mantém de uma edição para a outra?

    Três referências na tabela, e a comparação que importa é a última contra a
    penúltima. Se `2023 -> 2024` ficar perto de `2024 -> 2024 (CV)`, a estrutura
    territorial é estável; se despencar, o campeão precisaria de retreino anual e
    o ranking de 2024 não deveria ser usado para planejar 2025.
    """
    linhas = []
    coorte = {ano: _coorte_territorial(ano) for ano in config.ANOS}

    for ano in config.ANOS:
        base = coorte[ano]
        X, y, grupos = base[FEATURES_DRIFT], base[config.TARGET], base["id_municipio"]
        aucs = []
        for treino, validacao in split.folds(y, grupos, n_splits=3, seed=seed):
            modelo = pl.montar_pipeline(
                LGBMClassifier(random_state=seed, n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1),
                FEATURES_DRIFT,
            )
            modelo.fit(X.iloc[treino], y.iloc[treino])
            aucs.append(roc_auc_score(y.iloc[validacao], modelo.predict_proba(X.iloc[validacao])[:, 1]))
        linhas.append(
            {
                "desenho": f"{ano} -> {ano} (CV agrupada)",
                "n_treino": int(len(base)),
                "n_teste": int(len(base)),
                "prevalencia_teste": float(y.mean()),
                "roc_auc": float(np.mean(aucs)),
                "dp_entre_folds": float(np.std(aucs, ddof=1)),
            }
        )
        log.info("drift %s: %.4f", linhas[-1]["desenho"], linhas[-1]["roc_auc"])

    treino, teste = coorte[config.ANO_LAG], coorte[config.ANO_ALVO]
    modelo = pl.montar_pipeline(
        LGBMClassifier(random_state=seed, n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1),
        FEATURES_DRIFT,
    )
    modelo.fit(treino[FEATURES_DRIFT], treino[config.TARGET])
    escore = modelo.predict_proba(teste[FEATURES_DRIFT])[:, 1]
    linhas.append(
        {
            "desenho": f"{config.ANO_LAG} -> {config.ANO_ALVO} (out-of-time)",
            "n_treino": int(len(treino)),
            "n_teste": int(len(teste)),
            "prevalencia_teste": float(teste[config.TARGET].mean()),
            "roc_auc": float(roc_auc_score(teste[config.TARGET], escore)),
            "dp_entre_folds": np.nan,
        }
    )
    log.info("drift %s: %.4f", linhas[-1]["desenho"], linhas[-1]["roc_auc"])

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(CSV_DRIFT, index=False)
    return tabela


# ---------------------------------------------------------------------------
# 5. Invariância — o modelo ignora o que tem de ignorar?
# ---------------------------------------------------------------------------
def teste_de_invariancia(
    modelo, X: pd.DataFrame, y, colunas=("caderno", "mun_media_portugues_lag1", "rede_grupo"),
    n_repeticoes: int = 5, seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Embaralha uma coluna e mede quanto a AUC se move.

    `caderno` é o controle negativo declarado do projeto: os cadernos são
    randomizados entre alunos, então embaralhar a coluna não pode mudar nada.
    Se mudar, o modelo se sobreajustou a ruído e o resultado é alarme, não
    achado educacional. `mun_media_portugues_lag1` entra como controle positivo,
    para provar que o teste tem sensibilidade — sem ele, uma queda de zero em
    todas as colunas seria indistinguível de um teste quebrado.

    O bloco `esc_*` não aparece aqui porque não está no modelo; ele é controle
    negativo por ablação, medida na Etapa 3.
    """
    rng = np.random.default_rng(seed)
    referencia = float(roc_auc_score(y, modelo.predict_proba(X)[:, 1]))
    linhas = []
    for coluna in colunas:
        quedas = []
        for _ in range(n_repeticoes):
            perturbado = X.copy()
            perturbado[coluna] = rng.permutation(perturbado[coluna].to_numpy())
            quedas.append(referencia - float(roc_auc_score(y, modelo.predict_proba(perturbado)[:, 1])))
        linhas.append(
            {
                "coluna": coluna,
                "roc_auc_referencia": referencia,
                "queda_media": float(np.mean(quedas)),
                "queda_dp": float(np.std(quedas, ddof=1)),
                "n_repeticoes": n_repeticoes,
            }
        )
        log.info("invariância %s: queda %.5f ± %.5f", coluna, linhas[-1]["queda_media"], linhas[-1]["queda_dp"])
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(CSV_INVARIANCIA, index=False)
    return tabela


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
N_LINHAS_PERMUTACAO = 300_000


def main() -> None:
    """`python -m src.evaluation.rigor [prova ...]` — cada prova grava o seu CSV.

    A permutação e a curva de aprendizado rodam sobre subamostras de municípios
    inteiros, não sobre a base completa: são dezenas de ajustes cada, e o que
    elas medem — a forma da distribuição nula e a inclinação da curva — não muda
    de conclusão entre 300 mil e 1,5 milhão de linhas. A invariância e o LOGO
    rodam sobre a base que a decisão usa, porque ali o número em si importa.
    """
    import argparse

    from src.modeling import campeao, tuning

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    todas = ("drift", "logo", "invariancia", "permutacao", "aprendizado")
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("provas", nargs="*", default=None, choices=todas + ("todas",))
    escolhidas = analisador.parse_args().provas or ["todas"]
    if "todas" in escolhidas:
        escolhidas = list(todas)

    parametros = tuning.carregar_hiperparametros()
    dados = loader.carregar_dataset_modelagem()
    _, y, grupos = pl.separar_X_y(dados)
    dev, teste = split.separar_desenvolvimento_e_teste(grupos)

    if "drift" in escolhidas:
        modelo_de_drift()

    if "logo" in escolhidas:
        logo_por_regiao(parametros, dados.iloc[dev].reset_index(drop=True))

    if "invariancia" in escolhidas:
        pacote = campeao.carregar_campeao()
        X_teste, y_teste, _ = pl.separar_X_y(dados.iloc[teste].reset_index(drop=True))
        teste_de_invariancia(pacote["modelo"], X_teste, y_teste)

    amostra = None
    if {"permutacao", "aprendizado"} & set(escolhidas):
        posicoes = split.amostrar_municipios(grupos, N_LINHAS_PERMUTACAO, posicoes=dev)
        amostra = dados.iloc[posicoes].reset_index(drop=True)
        log.info("subamostra de %d alunos para as provas caras", len(amostra))

    if "permutacao" in escolhidas:
        teste_de_permutacao(parametros, amostra)
        print(resumo_da_permutacao().to_string(index=False))

    if "aprendizado" in escolhidas:
        curva_de_aprendizado(parametros, dados.iloc[dev].reset_index(drop=True))


if __name__ == "__main__":
    main()
