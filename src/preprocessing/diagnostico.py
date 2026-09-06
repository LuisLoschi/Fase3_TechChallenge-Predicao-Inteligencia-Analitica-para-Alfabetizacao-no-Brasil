"""Medições sobre o dataset já construído: dicionário, controles e a réplica B5.

Separado de `feature_store.py` de propósito. Lá se **constrói** feature; aqui se
**julga** feature, e o julgamento tem de poder ser refeito sem reconstruir nada.
O notebook 02 narra; a conta mora aqui e é travada por `tests/test_features.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.eda import auc_univariada
from src.preprocessing import pipeline as pl

BLOCOS = {
    "município": pl.FEATURES_MUNICIPIO,
    "município (microdado)": pl.FEATURES_PERCENTIS_MICRODADO,
    "UF": pl.FEATURES_UF,
    "cobertura do lag": pl.FEATURES_COBERTURA,
    "estrutural": pl.FEATURES_ESTRUTURAIS,
    "escola (controle)": pl.FEATURES_ESCOLA,
}


def _bloco_de(feature: str) -> str:
    for bloco, colunas in BLOCOS.items():
        if feature in colunas:
            return bloco
    return "derivada"


def medir_dicionario(dados: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Dicionário de features com cobertura, cardinalidade e AUC univariada.

    A AUC é a do **risco**: abaixo de 0,5 a feature protege. Rodar isto antes de
    qualquer tuning custa segundos e responde duas coisas de uma vez — o que tem
    sinal, e o que tem sinal *demais*. Nada aqui pode passar de 0,90, porque
    acima disso a explicação provável é vazamento e não qualidade.
    """
    # O padrão inclui os dois blocos que ficaram **fora** do modelo: o
    # dicionário serve para justificar a exclusão deles, não para escondê-la.
    padrao = pl.FEATURES_MODELO + pl.FEATURES_PERCENTIS_MICRODADO + pl.FEATURES_ESCOLA
    features = list(padrao if features is None else features)
    y = dados[config.TARGET].to_numpy()
    linhas = []
    for coluna in features:
        serie = dados[coluna]
        categorica = str(serie.dtype) in ("object", "category")
        linhas.append(
            {
                "feature": coluna,
                "bloco": _bloco_de(coluna),
                "tipo": "categórica" if categorica else "numérica",
                "cobertura": float(serie.notna().mean()),
                "n_unicos": int(serie.nunique(dropna=True)),
                "auc": np.nan if categorica else auc_univariada(y, serie.to_numpy(dtype=float)),
            }
        )
    quadro = pd.DataFrame(linhas)
    quadro["forca"] = (quadro["auc"] - 0.5).abs()
    return quadro.sort_values("forca", ascending=False, na_position="last").reset_index(drop=True)


def medir_controle_embaralhamento(dados: pd.DataFrame, n_seeds: int = 3) -> dict[str, float]:
    """O lag de escola contra o sorteio de uma escola qualquer da mesma UF.

    Controle de embaralhamento, o teste que decide se o bloco escolar carrega
    informação de escola. A permutação é feita **dentro da UF** justamente para
    preservar o sinal territorial: o que a comparação isola é o que sobra além
    dele. A permutação global é o piso, e a taxa municipal é o teto.

    Devolve AUC de alfabetização (e não de risco) para poder ser comparada
    diretamente com os números da EDA.
    """
    y = (1 - dados[config.TARGET]).to_numpy()
    real = dados["esc_taxa_alfab_lag1"].to_numpy(dtype=float)
    municipal = dados["mun_taxa_alfab_lag1"].to_numpy(dtype=float)

    dentro, globais = [], []
    for i in range(n_seeds):
        rng = np.random.default_rng(config.RANDOM_STATE + i)
        por_uf = dados.groupby("sigla_uf", observed=True)["esc_taxa_alfab_lag1"]
        dentro.append(
            auc_univariada(y, por_uf.transform(lambda s: rng.permutation(s.to_numpy())).to_numpy(dtype=float))
        )
        globais.append(auc_univariada(y, rng.permutation(real)))

    return {
        "real": auc_univariada(y, real),
        "dentro_uf": float(np.mean(dentro)),
        "dentro_uf_dp": float(np.std(dentro)),
        "global": float(np.mean(globais)),
        "municipal": auc_univariada(y, municipal),
    }


def resumir_replica(replica: pd.DataFrame) -> pd.DataFrame:
    """Média e dispersão da ROC-AUC por conjunto de features, entre folds e seeds."""
    return (
        replica.groupby("conjunto")
        .agg(
            n_features=("n_features", "first"),
            roc_auc=("roc_auc", "mean"),
            dp_entre_folds=("roc_auc", "std"),
            minimo=("roc_auc", "min"),
            maximo=("roc_auc", "max"),
            pr_auc=("pr_auc", "mean"),
        )
        .reset_index()
    )


def comparar_pareado(replica: pd.DataFrame, base: str, alternativo: str, k: int = 5) -> dict:
    """Δ pareado por fold com o t corrigido de Nadeau-Bengio.

    O t pareado comum é **inválido** aqui: os folds compartilham dados de treino,
    a variância do delta é subestimada e o erro tipo I estoura. A correção de
    Nadeau-Bengio infla a variância pelo fator `1/n + n_teste/n_treino`, que com
    `k = 5` vale `1/n + 0,25`.

    Com 15 deltas, um |t| abaixo de ~2,1 não autoriza dizer que um conjunto é
    melhor que o outro — e, mesmo que autorizasse, restaria perguntar se
    0,002 de AUC muda alguma decisão.
    """
    largo = replica.pivot_table(index=["seed", "fold"], columns="conjunto", values="roc_auc")
    delta = (largo[alternativo] - largo[base]).to_numpy()
    n = len(delta)
    correcao = 1 / n + 1 / (k - 1)
    variancia = float(delta.var(ddof=1))
    t = float(delta.mean() / np.sqrt(variancia * correcao)) if variancia > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else np.nan
    ic = stats.t.ppf(0.975, df=n - 1) * np.sqrt(variancia * correcao)
    return {
        "base": base,
        "alternativo": alternativo,
        "n_folds": n,
        "delta_medio": float(delta.mean()),
        "delta_dp": float(delta.std(ddof=1)),
        "folds_a_favor": int((delta > 0).sum()),
        "ic95_baixo": float(delta.mean() - ic),
        "ic95_alto": float(delta.mean() + ic),
        "t_nadeau_bengio": t,
        "p_valor": p,
    }
