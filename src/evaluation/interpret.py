"""Três leituras independentes do campeão, e a triangulação entre elas.

Uma única leitura de importância não decide nada. SHAP diz de quanto cada coluna
moveu a predição, `permutation_importance` diz quanto a métrica perde sem ela, e
o coeficiente linear diz em que direção e com que magnitude ela entra num modelo
que se lê a olho nu. As três discordam com frequência, e é a discordância que
informa: o que aparece no topo de uma só é achado sobre *aquela leitura*, não
sobre alfabetização.

O agrupamento por família não é refinamento. A EDA mediu treze pares de features
com |Spearman| acima de 0,95 e três colunas que são combinação linear exata de
outras; nesse regime o SHAP reparte o crédito entre colunas que dizem a mesma
coisa, e ler coluna a coluna subestima cada família de forma sistemática. O mapa
de 89 colunas pós-`ColumnTransformer` para 20 features de entrada, e delas para
cinco famílias, é construído a partir do transformador ajustado — não por
casamento de string — e a reconstrução é conferida contra
`get_feature_names_out()` a cada execução.

Duas restrições da Etapa 4 atravessam o módulo inteiro:

    o piso de ruído       o `caderno` é controle negativo e cai 0,00031 por
                          permutação; nada abaixo de 0,002 conta como fator
    a coorte é plural     onde o lag vem da Gold a AUC é 0,6706, e no agregado
                          `rede 3` é 0,5598, então toda leitura global vem com a
                          leitura restrita a `fonte_lag_municipal = gold` ao lado

E a ressalva que precede qualquer uso destes números: o que o modelo sabe além
da taxa média do seu município vale 0,0031 de ROC-AUC. A ordenação de features
que sai daqui descreve como um medidor de risco territorial se organiza por
dentro, não os determinantes da alfabetização de uma criança.
"""

from __future__ import annotations

import itertools
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.data import loader
from src.modeling import campeao, split
from src.preprocessing import pipeline as pl

log = logging.getLogger(__name__)

CSV_MAPA = config.DIR_METRICS / "interpret_mapa_de_familias.csv"
CSV_SHAP_COLUNAS = config.DIR_METRICS / "interpret_shap_colunas.csv"
CSV_SHAP_FEATURES = config.DIR_METRICS / "interpret_shap_features.csv"
CSV_FAMILIAS = config.DIR_METRICS / "interpret_familias.csv"
CSV_PERMUTACAO = config.DIR_METRICS / "interpret_permutacao.csv"
CSV_PERMUTACAO_FAMILIA = config.DIR_METRICS / "interpret_permutacao_familia.csv"
CSV_COEFICIENTES = config.DIR_METRICS / "interpret_coeficientes.csv"
CSV_TRIANGULACAO = config.DIR_METRICS / "interpret_triangulacao.csv"
JSON_INTERPRET = config.DIR_METRICS / "interpret_resumo.json"
PARQUET_SHAP = config.DIR_MODELS / "shap_amostra.parquet"

N_AMOSTRA_SHAP = 50_000
N_REPETICOES = 10

# Referência medida na Etapa 4, não escolhida: embaralhar o `caderno` — cadernos
# são sorteados entre alunos — derruba a AUC de teste em 0,00031. Queda menor que
# 0,002 é da ordem do que o controle negativo produz por acaso, e chamar isso de
# importância seria descrever sobreajuste como achado educacional.
PISO_DE_RUIDO = 0.002

# Quantas posições contam como "topo" em cada leitura. Cinco entre vinte features
# e três entre as oito da logística mantêm a mesma proporção, para que a
# triangulação não fique mais exigente numa leitura do que na outra.
TOPO_FEATURES = 5
TOPO_LOGISTICA = 3

# ---------------------------------------------------------------------------
# O mapa: 89 colunas -> 20 features -> 5 famílias
# ---------------------------------------------------------------------------
# A divisão segue a origem do dado, e não a conveniência da narrativa. O bloco
# de metas fica separado do resto do município porque a meta é um lag disfarçado
# (armadilha A4: `corr(meta_2025, taxa_2023) = 0,977`) e misturá-la ao histórico
# municipal esconderia justamente isso; `nome_regiao` e `sigla_uf` ficam com as
# features de UF porque medem o mesmo nível de agregação, acima do município.
FAMILIAS: dict[str, tuple[str, ...]] = {
    "historico_municipal": (
        "mun_taxa_alfab_lag1",
        "mun_media_portugues_lag1",
        "mun_taxa_presenca_lag1",
        "mun_n_alunos_lag1",
        "mun_n_escolas_lag1",
        "mun_share_rede_estadual_lag1",
        "mun_desvio_vs_uf",
    ),
    "metas": ("mun_meta_2024_lag1", "mun_nivel_alfabetizacao_lag1"),
    "territorial": (
        "uf_taxa_alfab_lag1",
        "uf_media_portugues_lag1",
        "uf_taxa_presenca_lag1",
        "sigla_uf",
        "nome_regiao",
    ),
    "historico_escolar": ("tem_historico_escola",),
    "estrutural": (
        "rede_grupo",
        "caderno",
        "tem_historico_municipio",
        "fonte_lag_municipal",
        "fonte_lag_uf",
    ),
}

# Dentro de `estrutural` convivem duas coisas diferentes, e o sub-bloco preserva
# a distinção sem multiplicar famílias: `rede_grupo` é característica da matrícula,
# enquanto as três colunas de proveniência dizem de qual apuração o lag daquele
# aluno veio. `tem_historico_escola` é o caso extremo — a família de histórico
# escolar inteira se resume a uma flag de cobertura, porque o bloco `esc_*` saiu
# do modelo na Etapa 3 (chave de escola reatribuída a cada edição).
SUB_BLOCOS: dict[str, str] = {
    "tem_historico_municipio": "proveniencia_do_lag",
    "fonte_lag_municipal": "proveniencia_do_lag",
    "fonte_lag_uf": "proveniencia_do_lag",
    "tem_historico_escola": "proveniencia_do_lag",
    "caderno": "controle_negativo",
}


def familia_de_cada_feature() -> pd.DataFrame:
    """Uma linha por feature de entrada, com família e sub-bloco.

    A partição é conferida contra `FEATURES_MODELO` em vez de assumida: uma
    feature nova entrando no modelo sem entrar aqui apareceria no SHAP e sumiria
    da leitura por família, que é exatamente o erro que o agrupamento existe para
    evitar.
    """
    linhas = [
        {"feature": feature, "familia": familia, "sub_bloco": SUB_BLOCOS.get(feature, familia)}
        for familia, features in FAMILIAS.items()
        for feature in features
    ]
    mapa = pd.DataFrame(linhas)
    duplicadas = mapa.loc[mapa["feature"].duplicated(), "feature"].tolist()
    assert not duplicadas, f"feature em mais de uma família: {duplicadas}"
    assert set(mapa["feature"]) == set(pl.FEATURES_MODELO), (
        f"o mapa de famílias não cobre FEATURES_MODELO: "
        f"faltando {sorted(set(pl.FEATURES_MODELO) - set(mapa['feature']))}, "
        f"sobrando {sorted(set(mapa['feature']) - set(pl.FEATURES_MODELO))}"
    )
    return mapa


def _origem_das_dummies(saidas: list[str], colunas: list[str]) -> list[str]:
    """Cada dummy volta para a coluna que a gerou, com ambiguidade proibida.

    O `OneHotEncoder` com `verbose_feature_names_out=False` nomeia a saída como
    `<coluna>_<categoria>`, então o prefixo resolve — mas só se um único
    candidato casar. Duas colunas em que uma é prefixo da outra
    (`fonte_lag` e `fonte_lag_uf`, por exemplo) devolveriam crédito para a
    família errada em silêncio, e é isso que a asserção impede.
    """
    origem = []
    for saida in saidas:
        candidatos = [c for c in colunas if saida.startswith(f"{c}_")]
        assert len(candidatos) == 1, f"origem ambígua para {saida!r}: {candidatos}"
        origem.append(candidatos[0])
    return origem


def mapear_colunas_para_features(pre) -> pd.DataFrame:
    """As 89 colunas que chegam ao estimador, cada uma com a sua feature de origem.

    Construído a partir do `ColumnTransformer` **ajustado**, e não de regras sobre
    o nome: o bloco numérico devolve as colunas seguidas dos indicadores de nulo
    que o `SimpleImputer` de fato criou (12 dos 14, porque `tem_historico_*` não
    tem nulo), e o bloco categórico devolve as dummies na ordem das categorias
    vistas no ajuste. A reconstrução é comparada com `get_feature_names_out()`
    posição a posição, então qualquer mudança de pré-processamento quebra aqui em
    vez de aparecer como um SHAP atribuído à variável errada.
    """
    familias = familia_de_cada_feature().set_index("feature")
    linhas = []
    for nome, transformador, colunas in pre.transformers_:
        if transformador == "drop" or nome == "remainder":
            continue
        colunas = list(colunas)
        saidas = list(transformador.get_feature_names_out(colunas))
        ultimo = transformador[-1] if hasattr(transformador, "steps") else transformador

        if isinstance(ultimo, OneHotEncoder):
            origem = _origem_das_dummies(saidas, colunas)
            tipos = ["dummy"] * len(saidas)
            agrupada = [chave for chave, _ in itertools.groupby(origem)]
            assert agrupada == colunas, (
                f"as dummies do bloco {nome!r} não são contíguas por coluna: {agrupada}"
            )
        else:
            imputador = transformador.named_steps["imput"]
            indices = list(imputador.indicator_.features_) if imputador.add_indicator else []
            origem = colunas + [colunas[i] for i in indices]
            tipos = ["valor"] * len(colunas) + ["indicador_de_nulo"] * len(indices)

        assert len(origem) == len(saidas), f"bloco {nome!r}: {len(origem)} origens para {len(saidas)} saídas"
        linhas += [
            {"coluna": saida, "feature": feature, "tipo": tipo, "bloco": nome}
            for saida, feature, tipo in zip(saidas, origem, tipos)
        ]

    mapa = pd.DataFrame(linhas)
    esperado = list(pre.get_feature_names_out())
    assert mapa["coluna"].tolist() == esperado, "a reconstrução não bate com get_feature_names_out()"
    return mapa.join(familias, on="feature").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Contexto: o mesmo split da Etapa 4, sem retreino e sem re-split
# ---------------------------------------------------------------------------
def carregar_contexto() -> dict:
    """Campeão em disco, dataset e o split idêntico ao da Etapa 4.

    O `hash_dataset` gravado no artefato é conferido contra o Parquet atual antes
    de qualquer conta. Explicar um modelo com um arquivo de dados diferente do
    que o treinou produz uma leitura plausível e falsa, e é um erro que nenhuma
    métrica desta etapa denunciaria sozinha.
    """
    pacote = campeao.carregar_campeao()
    atual = campeao.hash_do_dataset()
    assert pacote["hash_dataset"] == atual, (
        f"o campeão foi treinado sobre o dataset {pacote['hash_dataset']} e o "
        f"arquivo em disco é {atual}. Retreine antes de interpretar."
    )

    dados = loader.carregar_dataset_modelagem()
    X, y, grupos = pl.separar_X_y(dados)
    dev, teste = split.separar_desenvolvimento_e_teste(grupos, seed=pacote["seed"])
    split.verificar_grupos_disjuntos(grupos, dev, teste)
    log.info("teste com %d alunos de %d municípios", len(teste), grupos.iloc[teste].nunique())
    return {
        "pacote": pacote,
        "dados": dados,
        "X": X,
        "y": y,
        "grupos": grupos,
        "dev": dev,
        "teste": teste,
    }


# ---------------------------------------------------------------------------
# 1. SHAP
# ---------------------------------------------------------------------------
def amostrar_para_shap(
    contexto: dict, n: int = N_AMOSTRA_SHAP, seed: int = config.RANDOM_STATE
) -> np.ndarray:
    """Posições sorteadas dentro do conjunto de teste, para explicar dado inédito.

    Sorteio de alunos, e não de municípios inteiros como no tuning: aqui não se
    ajusta nada, então não há estrutura de grupo a proteger, e amostrar linhas
    preserva a composição da coorte — inclusive a proporção de cada
    `fonte_lag_municipal`, que é o corte pelo qual a leitura precisa ser refeita.
    """
    teste = contexto["teste"]
    if len(teste) <= n:
        return teste
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(teste, size=n, replace=False))


def valores_shap(contexto: dict, posicoes: np.ndarray) -> pd.DataFrame:
    """Valores SHAP por coluna, em log-odds, mais o dado cru de cada aluno.

    `TreeExplainer` ataca o `LGBMClassifier` diretamente sobre a matriz já
    transformada, e não o `Pipeline` inteiro: a isotônica está desligada no
    artefato (`aplicar_ = False`), então o estimador é o modelo publicado, e
    explicar a árvore é exato em vez de aproximado por amostragem.

    A tabela devolvida carrega três coisas lado a lado — SHAP por coluna
    (`shap__`), valor cru da feature (`x__`) e contexto do aluno. Sem o valor cru
    o gráfico de dependência teria de ser lido na escala padronizada, que não
    significa nada para quem decide orçamento.
    """
    import shap

    pacote = contexto["pacote"]
    modelo = pacote["modelo"]
    estimador = modelo.estimador_
    nomes = pacote["nomes_apos_pre_processamento"]

    X_amostra = contexto["X"].iloc[posicoes]
    matriz = pd.DataFrame(estimador[:-1].transform(X_amostra), columns=nomes)

    inicio = time.time()
    explicador = shap.TreeExplainer(estimador.steps[-1][1])
    explicacao = explicador(matriz, check_additivity=False)
    log.info("SHAP de %d alunos x %d colunas em %.0fs", *matriz.shape, time.time() - inicio)

    valores = pd.DataFrame(explicacao.values, columns=[f"shap__{c}" for c in nomes])
    cru = X_amostra.reset_index(drop=True).add_prefix("x__")
    contextual = contexto["dados"].iloc[posicoes].reset_index(drop=True)
    return pd.concat(
        [
            valores,
            cru,
            pd.DataFrame(
                {
                    "valor_esperado": float(np.ravel(explicador.expected_value)[0]),
                    "escore": modelo.predict_proba(X_amostra)[:, 1],
                    config.TARGET: contextual[config.TARGET].to_numpy(),
                    "fonte_lag_municipal": contextual["fonte_lag_municipal"].astype(str).to_numpy(),
                    "sigla_uf": contextual["sigla_uf"].astype(str).to_numpy(),
                }
            ),
        ],
        axis=1,
    )


def reconstruir_matriz(amostra: pd.DataFrame, pacote: dict) -> pd.DataFrame:
    """Refaz a matriz de 89 colunas a partir do dado cru guardado na amostra.

    Guardar a matriz transformada no Parquet dobraria o arquivo para reproduzir
    em disco algo que o pré-processador do campeão recalcula em décimos de
    segundo. O `beeswarm` precisa dela para colorir cada ponto pelo valor da
    variável, e é o único consumidor.
    """
    cru = amostra[[f"x__{c}" for c in pacote["features"]]].rename(columns=lambda c: c[3:])
    matriz = pacote["modelo"].estimador_[:-1].transform(cru)
    return pd.DataFrame(matriz, columns=pacote["nomes_apos_pre_processamento"])


def importancia_shap_por_coluna(amostra: pd.DataFrame, mapa: pd.DataFrame, recorte: str = "global") -> pd.DataFrame:
    """Média do |SHAP| de cada uma das 89 colunas, com a família ao lado."""
    colunas = [f"shap__{c}" for c in mapa["coluna"]]
    medias = amostra[colunas].abs().mean().to_numpy()
    tabela = mapa.assign(recorte=recorte, n=len(amostra), shap_medio_abs=medias)
    tabela["participacao"] = tabela["shap_medio_abs"] / tabela["shap_medio_abs"].sum()
    return tabela.sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)


def _agrupar_shap(amostra: pd.DataFrame, mapa: pd.DataFrame, chave: str, recorte: str) -> pd.DataFrame:
    """Duas agregações de SHAP por grupo, porque elas respondem a coisas diferentes.

    `soma_das_medias_abs` empilha a média do |SHAP| coluna a coluna: é o que se
    lê num gráfico de barras somado, e ignora que duas colunas do mesmo grupo
    podem empurrar a predição em direções opostas. `media_do_abs_da_soma` soma
    primeiro dentro da linha e tira o módulo depois, que é a contribuição líquida
    do grupo naquele aluno — a leitura correta quando o grupo existe justamente
    porque suas colunas são redundantes entre si.

    A distância entre as duas mede o cancelamento interno. Grupo com uma coluna
    só tem as duas iguais por construção, o que serve de conferência.
    """
    linhas = []
    for grupo, bloco in mapa.groupby(chave, sort=False):
        colunas = [f"shap__{c}" for c in bloco["coluna"]]
        soma_na_linha = amostra[colunas].to_numpy().sum(axis=1)
        linhas.append(
            {
                chave: grupo,
                **({} if chave == "familia" else {"familia": bloco["familia"].iloc[0]}),
                "recorte": recorte,
                "n": len(amostra),
                "n_colunas": len(colunas),
                "n_features": bloco["feature"].nunique(),
                "soma_das_medias_abs": float(amostra[colunas].abs().mean().sum()),
                "media_do_abs_da_soma": float(np.abs(soma_na_linha).mean()),
                "dp_da_soma": float(soma_na_linha.std(ddof=1)),
            }
        )
    tabela = pd.DataFrame(linhas).sort_values("media_do_abs_da_soma", ascending=False)
    tabela["participacao"] = tabela["media_do_abs_da_soma"] / tabela["media_do_abs_da_soma"].sum()
    tabela["cancelamento_interno"] = 1 - tabela["media_do_abs_da_soma"] / tabela["soma_das_medias_abs"]
    return tabela.reset_index(drop=True)


def importancia_shap_por_feature(amostra: pd.DataFrame, mapa: pd.DataFrame, recorte: str = "global") -> pd.DataFrame:
    """Das 89 colunas de volta às 20 features de entrada."""
    return _agrupar_shap(amostra, mapa, "feature", recorte)


def importancia_shap_por_familia(amostra: pd.DataFrame, mapa: pd.DataFrame, recorte: str = "global") -> pd.DataFrame:
    """Das 20 features para as cinco famílias."""
    return _agrupar_shap(amostra, mapa, "familia", recorte)


def leituras_por_recorte(amostra: pd.DataFrame, mapa: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """A mesma leitura na coorte inteira e só onde o lag veio da Gold.

    A Etapa 4 mediu AUC 0,6706 nessa fatia contra 0,5598 no agregado `rede 3`: a
    leitura global é uma média entre populações que o modelo entende de formas
    muito diferentes. Se as duas ordenações divergirem, a global é artefato da
    coalescência e o que vale é a restrita.
    """
    gold = amostra[amostra["fonte_lag_municipal"] == "gold"]
    log.info("recorte gold: %d de %d alunos da amostra", len(gold), len(amostra))
    colunas, features, familias = [], [], []
    for recorte, fatia in (("global", amostra), ("gold", gold)):
        colunas.append(importancia_shap_por_coluna(fatia, mapa, recorte))
        features.append(importancia_shap_por_feature(fatia, mapa, recorte))
        familias.append(importancia_shap_por_familia(fatia, mapa, recorte))
    return {
        "colunas": pd.concat(colunas, ignore_index=True),
        "features": pd.concat(features, ignore_index=True),
        "familias": pd.concat(familias, ignore_index=True),
    }


# ---------------------------------------------------------------------------
# 2. Permutação no conjunto de teste
# ---------------------------------------------------------------------------
def permutacao_por_feature(
    contexto: dict, n_repeticoes: int = N_REPETICOES, seed: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """`permutation_importance` sobre as 20 colunas cruas do conjunto de teste.

    Permutar a feature de entrada, e não a coluna transformada, é o que mantém a
    leitura interpretável: embaralhar `sigla_uf_SP` sozinha deixaria as outras 25
    dummies intactas e mediria uma perturbação impossível. A permutação também
    acontece **antes** do pré-processamento, então o imputador e o encoder veem o
    valor embaralhado, como veriam em produção.

    O conjunto é o de teste, com municípios inéditos. Medir permutação no treino
    devolveria a importância do que o modelo decorou.
    """
    modelo = contexto["pacote"]["modelo"]
    X_teste = contexto["X"].iloc[contexto["teste"]]
    y_teste = contexto["y"].iloc[contexto["teste"]]

    inicio = time.time()
    resultado = permutation_importance(
        modelo, X_teste, y_teste, scoring="roc_auc",
        n_repeats=n_repeticoes, random_state=seed, n_jobs=1,
    )
    referencia = float(roc_auc_score(y_teste, modelo.predict_proba(X_teste)[:, 1]))
    log.info("permutação por feature em %.0fs (referência %.4f)", time.time() - inicio, referencia)

    tabela = pd.DataFrame(
        {
            "feature": list(X_teste.columns),
            "roc_auc_referencia": referencia,
            "queda_media": resultado.importances_mean,
            "queda_dp": resultado.importances_std,
            "n_repeticoes": n_repeticoes,
            "n": len(y_teste),
        }
    ).join(familia_de_cada_feature().set_index("feature"), on="feature")
    tabela["acima_do_piso"] = tabela["queda_media"] > PISO_DE_RUIDO
    return tabela.sort_values("queda_media", ascending=False).reset_index(drop=True)


def permutacao_por_bloco(
    contexto: dict,
    blocos: dict[str, list[str]] | None = None,
    n_repeticoes: int = N_REPETICOES,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Permuta a família inteira de uma vez, com a mesma permutação de linhas.

    É a correção do problema que o plano antecipou. Permutando `mun_taxa_alfab`
    sozinha, `mun_media_portugues` continua na mesa e o modelo recupera quase
    tudo: a queda medida vira uma medida de redundância, não de importância.
    Embaralhando o bloco todo com o mesmo vetor de posições, a relação entre a
    família e o alvo é destruída enquanto a estrutura interna da família é
    preservada — o que se mede passa a ser o que aquele conjunto de colunas
    acrescenta ao resto do modelo.
    """
    modelo = contexto["pacote"]["modelo"]
    X_teste = contexto["X"].iloc[contexto["teste"]]
    y_teste = contexto["y"].iloc[contexto["teste"]].to_numpy()
    blocos = blocos or {familia: list(features) for familia, features in FAMILIAS.items()}

    referencia = float(roc_auc_score(y_teste, modelo.predict_proba(X_teste)[:, 1]))
    rng = np.random.default_rng(seed)
    linhas = []
    for nome, colunas in blocos.items():
        inicio = time.time()
        quedas = []
        for _ in range(n_repeticoes):
            embaralhado = X_teste.copy()
            ordem = rng.permutation(len(embaralhado))
            embaralhado[colunas] = embaralhado[colunas].to_numpy()[ordem]
            quedas.append(referencia - float(roc_auc_score(y_teste, modelo.predict_proba(embaralhado)[:, 1])))
        linhas.append(
            {
                "bloco": nome,
                "n_features": len(colunas),
                "roc_auc_referencia": referencia,
                "queda_media": float(np.mean(quedas)),
                "queda_dp": float(np.std(quedas, ddof=1)),
                "n_repeticoes": n_repeticoes,
            }
        )
        log.info(
            "bloco %s (%d features): queda %.5f ± %.5f em %.0fs",
            nome, len(colunas), linhas[-1]["queda_media"], linhas[-1]["queda_dp"], time.time() - inicio,
        )
    tabela = pd.DataFrame(linhas).sort_values("queda_media", ascending=False)
    tabela["acima_do_piso"] = tabela["queda_media"] > PISO_DE_RUIDO
    return tabela.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Coeficientes da logística podada
# ---------------------------------------------------------------------------
def coeficientes_da_logistica(
    contexto: dict, n_splits: int = split.N_SPLITS, seed: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Coeficientes, razão de chances e estabilidade entre folds do modelo linear.

    O campeão não tem coeficiente para ler, então a terceira leitura vem do
    modelo linear que ficou 0,0098 de AUC atrás dele com oito colunas. Ele roda
    sobre `FEATURES_PODADAS` com `indicador_de_nulo=False` porque no conjunto
    completo a matriz é singular por construção — e um coeficiente de matriz
    singular tem sinal decidido por arredondamento.

    Três colunas de saída, cada uma respondendo a uma pergunta diferente. O
    coeficiente padronizado dá a direção e é comparável entre numéricas, já que o
    `StandardScaler` está no caminho. A razão de chances traduz para a linguagem
    de quem lê relatório. E `contribuicao_dp` — o desvio-padrão de `coef * x` na
    própria amostra — é o único dos três que compara numérica com dummy na mesma
    régua, e por isso é ele que entra na triangulação.

    O ajuste se repete nos cinco folds agrupados por município e o desvio entre
    eles vai na tabela: coeficiente que troca de sinal entre folds não sustenta
    afirmação nenhuma, por maior que seja. Os coeficientes dos folds são alinhados
    **por nome de coluna**, e não por posição, porque um fold em que falte uma
    categoria devolve uma dummy a menos — alinhar por posição ali deslocaria toda
    a comparação em silêncio.
    """
    features = list(pl.FEATURES_PODADAS)
    assert set(features) <= set(contexto["X"].columns), "FEATURES_PODADAS não é subconjunto de FEATURES_MODELO"
    X_dev = contexto["dados"].iloc[contexto["dev"]][features].reset_index(drop=True)
    y_dev = contexto["y"].iloc[contexto["dev"]].reset_index(drop=True)
    g_dev = contexto["grupos"].iloc[contexto["dev"]].reset_index(drop=True)

    def ajustar(posicoes) -> tuple[pd.Series, np.ndarray]:
        modelo = pl.montar_pipeline(
            LogisticRegression(max_iter=1000, random_state=seed), features, indicador_de_nulo=False
        )
        modelo.fit(X_dev.iloc[posicoes], y_dev.iloc[posicoes])
        nomes = pl.nomes_das_features(modelo)
        matriz = np.asarray(modelo[:-1].transform(X_dev.iloc[posicoes]))
        return pd.Series(modelo[-1].coef_.ravel(), index=nomes), matriz

    inicio = time.time()
    coeficientes, matriz = ajustar(np.arange(len(X_dev)))
    por_fold = pd.concat(
        [ajustar(treino)[0] for treino, _ in split.folds(y_dev, g_dev, n_splits, seed)], axis=1
    ).reindex(coeficientes.index)
    log.info("logística podada: %d colunas, %d folds, %.0fs", len(coeficientes), n_splits, time.time() - inicio)

    mapa = pd.DataFrame({"coluna": coeficientes.index})
    mapa["feature"] = [
        next((f for f in features if c == f or c.startswith(f"{f}_")), None) for c in mapa["coluna"]
    ]
    assert mapa["feature"].notna().all(), f"colunas sem origem: {mapa.loc[mapa['feature'].isna(), 'coluna'].tolist()}"

    sinais = np.sign(por_fold.to_numpy())
    tabela = mapa.assign(
        coeficiente=coeficientes.to_numpy(),
        razao_de_chances=np.exp(coeficientes.to_numpy()),
        coeficiente_dp_entre_folds=por_fold.std(axis=1, ddof=1).to_numpy(),
        folds_com_a_coluna=por_fold.notna().sum(axis=1).to_numpy(),
        troca_de_sinal_entre_folds=np.nanmin(sinais, axis=1) != np.nanmax(sinais, axis=1),
        contribuicao_dp=(matriz * coeficientes.to_numpy()).std(axis=0, ddof=1),
    ).join(familia_de_cada_feature().set_index("feature"), on="feature")
    return tabela.sort_values("contribuicao_dp", ascending=False).reset_index(drop=True)


def contribuicao_linear_por_feature(coeficientes: pd.DataFrame) -> pd.DataFrame:
    """Soma da contribuição das dummies de volta na feature de origem.

    O nível de uma categoria não é a variável: o efeito de `nome_regiao` é o que
    as cinco dummies fazem juntas, e reportar a maior delas trocaria a
    importância da variável pela importância de um nível.
    """
    return (
        coeficientes.groupby("feature", as_index=False)
        .agg(
            familia=("familia", "first"),
            n_colunas=("coluna", "size"),
            contribuicao_dp=("contribuicao_dp", "sum"),
            maior_razao_de_chances=("razao_de_chances", "max"),
        )
        .sort_values("contribuicao_dp", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 4. Triangulação
# ---------------------------------------------------------------------------
def triangular(
    shap_features: pd.DataFrame,
    permutacao: pd.DataFrame,
    contribuicao: pd.DataFrame,
    recorte: str = "global",
) -> pd.DataFrame:
    """Cruza as três leituras e classifica cada feature pelo grau de concordância.

    O veredito é conservador de propósito. `fator` exige aparecer no topo de
    todas as leituras que **conseguem** ver aquela feature, e ainda derrubar mais
    que o piso de ruído do controle negativo. A logística só enxerga oito das
    vinte, então para as outras doze a triangulação é de duas pontas e a tabela
    diz isso na coluna `leituras_disponiveis` — reportar consenso de três quando
    houve duas seria inflar a confiança do resultado.

    `achado sobre a leitura` é o rótulo do que aparece no topo de uma leitura só.
    Não é descarte: é a informação de que aquela variável tem peso na mecânica de
    uma técnica e não nas outras, o que costuma ser sintoma de colinearidade ou
    de cardinalidade alta, e não de importância para o desfecho.
    """
    shap_recorte = shap_features[shap_features["recorte"] == recorte]
    tabela = (
        shap_recorte[["feature", "familia", "media_do_abs_da_soma"]]
        .rename(columns={"media_do_abs_da_soma": "shap"})
        .merge(
            permutacao[["feature", "queda_media", "queda_dp", "acima_do_piso"]].rename(
                columns={"queda_media": "permutacao", "queda_dp": "permutacao_dp"}
            ),
            on="feature",
            how="outer",
        )
        .merge(
            contribuicao[["feature", "contribuicao_dp"]].rename(columns={"contribuicao_dp": "logistica"}),
            on="feature",
            how="left",
        )
    )
    tabela["posicao_shap"] = tabela["shap"].rank(ascending=False, method="min")
    tabela["posicao_permutacao"] = tabela["permutacao"].rank(ascending=False, method="min")
    tabela["posicao_logistica"] = tabela["logistica"].rank(ascending=False, method="min")
    tabela["leituras_disponiveis"] = 2 + tabela["logistica"].notna().astype(int)

    no_topo = pd.DataFrame(
        {
            "shap": tabela["posicao_shap"] <= TOPO_FEATURES,
            "permutacao": tabela["posicao_permutacao"] <= TOPO_FEATURES,
            "logistica": tabela["posicao_logistica"] <= TOPO_LOGISTICA,
        }
    )
    disponivel = pd.DataFrame(
        {"shap": True, "permutacao": True, "logistica": tabela["logistica"].notna()}, index=tabela.index
    )
    tabela["leituras_no_topo"] = (no_topo & disponivel).sum(axis=1)
    concorda = ((no_topo & disponivel).sum(axis=1) == disponivel.sum(axis=1)) & no_topo["shap"]

    def rotular(linha, concordou, quantas) -> str:
        if concordou and linha["acima_do_piso"]:
            return "fator"
        if concordou and not linha["acima_do_piso"]:
            return "topo em todas as leituras, mas abaixo do piso de ruído"
        if quantas == 1:
            return "achado sobre a leitura"
        if quantas >= 2:
            return "relevante, sem consenso"
        return "secundária"

    tabela["veredito"] = [
        rotular(linha, bool(c), int(q))
        for (_, linha), c, q in zip(tabela.iterrows(), concorda, tabela["leituras_no_topo"])
    ]
    tabela["recorte"] = recorte
    return tabela.sort_values("permutacao", ascending=False).reset_index(drop=True)


def julgar_hipoteses(
    shap_familias: pd.DataFrame, permutacao_familia: pd.DataFrame, triangulacao: pd.DataFrame
) -> list[dict]:
    """H1, H2 e H5 da EDA, cada uma com o critério de falseamento que ela declarou.

    O julgamento é mecânico porque tem de ser: o critério foi escrito antes de a
    medição existir, e reinterpretá-lo depois de ver o resultado é o modo mais
    comum de uma hipótese nunca cair.

    O veredito tem três valores, e não dois, porque as leituras podem discordar
    entre si. `sustentada em parte` é o caso em que uma leitura confirma e outra
    não — o resultado informativo de verdade, e o que uma escala binária
    esconderia ao arredondar para o lado conveniente.
    """
    def veredito(confirmacoes: list[bool]) -> str:
        if all(confirmacoes):
            return "sustentada"
        return "sustentada em parte" if any(confirmacoes) else "derrubada"

    global_familias = shap_familias[shap_familias["recorte"] == "global"].reset_index(drop=True)
    ordem_shap = global_familias["familia"].tolist()
    ordem_perm = permutacao_familia["bloco"].tolist()
    presenca = triangulacao.set_index("feature").loc["mun_taxa_presenca_lag1"]

    lidera_shap = ordem_shap[0]
    lidera_perm = ordem_perm[0]
    h1 = [lidera_shap == "historico_municipal", lidera_perm == "historico_municipal"]

    posicao_escola_shap = ordem_shap.index("historico_escolar") + 1
    posicao_escola_perm = ordem_perm.index("historico_escolar") + 1
    h2 = [posicao_escola_shap >= len(ordem_shap) - 1, posicao_escola_perm >= len(ordem_perm) - 1]

    h5 = [presenca["posicao_shap"] <= TOPO_FEATURES, bool(presenca["acima_do_piso"])]

    return [
        {
            "hipotese": "H1",
            "enunciado": "o contexto municipal de 2023 é o preditor dominante do risco individual em 2024",
            "criterio_de_falseamento": "uma família não municipal aparecer no topo do SHAP",
            "medido": {
                "familia_no_topo_do_shap": lidera_shap,
                "familia_no_topo_da_permutacao_em_bloco": lidera_perm,
                "ordem_shap": ordem_shap,
                "ordem_permutacao": ordem_perm,
            },
            "veredito": veredito(h1),
        },
        {
            "hipotese": "H2",
            "enunciado": "o histórico escolar verdadeiro acrescentaria sinal além do município?",
            "criterio_de_falseamento": "a família de histórico escolar não aparecer no fim da lista",
            "medido": {
                "posicao_no_shap": posicao_escola_shap,
                "posicao_na_permutacao": posicao_escola_perm,
                "de_um_total_de": len(ordem_shap),
                "ressalva": (
                    "a família se resume a `tem_historico_escola`; o bloco `esc_*` "
                    "é ligado por identificadores reciclados; isso não testa o "
                    "valor de um histórico escolar longitudinal verdadeiro"
                ),
            },
            "veredito": "não verificável",
        },
        {
            "hipotese": "H5",
            "enunciado": "a presença municipal anterior está associada ao risco observado entre presentes",
            "criterio_de_falseamento": (
                "`mun_taxa_presenca_lag1` não aparecer entre as features relevantes — "
                f"lido como estar no topo {TOPO_FEATURES} do SHAP e derrubar mais que "
                f"o piso de ruído de {PISO_DE_RUIDO} por permutação"
            ),
            "medido": {
                "queda_por_permutacao": float(presenca["permutacao"]),
                "piso_de_ruido": PISO_DE_RUIDO,
                "posicao_no_shap": float(presenca["posicao_shap"]),
                "posicao_na_permutacao": float(presenca["posicao_permutacao"]),
                "posicao_na_logistica": (
                    None if pd.isna(presenca["posicao_logistica"]) else float(presenca["posicao_logistica"])
                ),
                "veredito_da_triangulacao": presenca["veredito"],
            },
            "veredito": veredito(h5),
        },
    ]


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def main() -> None:
    """`python -m src.evaluation.interpret` — lê o campeão e grava `reports/metrics/interpret_*`.

    Não treina o campeão, não regenera dado e não toca em `data/`. O único ajuste
    que acontece aqui é o da logística podada, que existe para produzir
    coeficientes e não para competir com nada.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    inicio = time.time()
    config.DIR_METRICS.mkdir(parents=True, exist_ok=True)
    config.DIR_MODELS.mkdir(parents=True, exist_ok=True)

    contexto = carregar_contexto()
    mapa = mapear_colunas_para_features(contexto["pacote"]["modelo"].estimador_.named_steps["pre"])
    mapa.to_csv(CSV_MAPA, index=False)
    log.info("mapa: %d colunas -> %d features -> %d famílias",
             len(mapa), mapa["feature"].nunique(), mapa["familia"].nunique())

    posicoes = amostrar_para_shap(contexto)
    amostra = valores_shap(contexto, posicoes)
    amostra.to_parquet(PARQUET_SHAP, index=False)

    leituras = leituras_por_recorte(amostra, mapa)
    leituras["colunas"].to_csv(CSV_SHAP_COLUNAS, index=False)
    leituras["features"].to_csv(CSV_SHAP_FEATURES, index=False)

    permutacao = permutacao_por_feature(contexto)
    permutacao.to_csv(CSV_PERMUTACAO, index=False)

    blocos = permutacao_por_bloco(contexto)
    blocos.to_csv(CSV_PERMUTACAO_FAMILIA, index=False)

    familias = leituras["familias"].merge(
        blocos[["bloco", "queda_media", "queda_dp", "acima_do_piso"]].rename(
            columns={"bloco": "familia", "queda_media": "queda_permutacao_em_bloco",
                     "queda_dp": "queda_permutacao_dp"}
        ),
        on="familia",
        how="left",
    )
    familias.to_csv(CSV_FAMILIAS, index=False)

    coeficientes = coeficientes_da_logistica(contexto)
    coeficientes.to_csv(CSV_COEFICIENTES, index=False)
    contribuicao = contribuicao_linear_por_feature(coeficientes)

    triangulacao = pd.concat(
        [
            triangular(leituras["features"], permutacao, contribuicao, recorte)
            for recorte in ("global", "gold")
        ],
        ignore_index=True,
    )
    triangulacao.to_csv(CSV_TRIANGULACAO, index=False)

    hipoteses = julgar_hipoteses(
        leituras["familias"], blocos, triangulacao[triangulacao["recorte"] == "global"]
    )
    resumo = {
        "hash_dataset": contexto["pacote"]["hash_dataset"],
        "seed": contexto["pacote"]["seed"],
        "versoes": {
            **contexto["pacote"]["versoes"],
            "shap": __import__("shap").__version__,
        },
        "amostra_shap": {
            "n": int(len(amostra)),
            "n_gold": int((amostra["fonte_lag_municipal"] == "gold").sum()),
            "origem": "conjunto de teste, municípios inéditos",
            "valor_esperado_log_odds": float(amostra["valor_esperado"].iloc[0]),
        },
        "permutacao": {
            "n": int(len(contexto["teste"])),
            "n_repeticoes": N_REPETICOES,
            "piso_de_ruido": PISO_DE_RUIDO,
            "roc_auc_referencia": float(permutacao["roc_auc_referencia"].iloc[0]),
        },
        "familias": {familia: list(features) for familia, features in FAMILIAS.items()},
        "hipoteses": hipoteses,
        "segundos": round(time.time() - inicio, 1),
    }
    JSON_INTERPRET.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    print(familias[familias["recorte"] == "global"].round(5).to_string(index=False))
    print()
    print(triangulacao[triangulacao["recorte"] == "global"].round(5).to_string(index=False))
    for hipotese in hipoteses:
        print(f"{hipotese['hipotese']}: {hipotese['veredito']}")
    log.info("interpretabilidade concluída em %.0fs", time.time() - inicio)


if __name__ == "__main__":
    main()
