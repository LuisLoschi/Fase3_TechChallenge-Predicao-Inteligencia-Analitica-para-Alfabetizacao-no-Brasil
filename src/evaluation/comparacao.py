"""Testes que decidem se a diferença entre dois modelos é diferença ou é ruído.

O desvio entre folds medido nesta base é de 0,011 a 0,015 de ROC-AUC — maior que
quase todo ganho que se espera de tuning. Comparar médias sem teste seria, aqui,
declarar campeão pelo sorteio dos folds.

Duas ferramentas, e a escolha entre elas depende de onde a comparação acontece:

**Na validação cruzada**, o t pareado comum é inválido: os folds compartilham
dados de treino, a variância do delta sai subestimada e o erro tipo I estoura. A
correção de Nadeau-Bengio infla a variância pelo fator `1/n + n_teste/n_treino`,
que com `k = 5` vale `1/n + 0,25`. A implementação é a mesma da Etapa 3, movida
para cá porque agora compara modelos e não conjuntos de features.

**No conjunto de teste**, o instrumento usual seria o teste de DeLong. Ele não
serve nesta base: DeLong supõe observações independentes, e o teste é composto
por municípios inteiros com centenas de alunos correlacionados dentro de cada um.
Aplicado aqui, devolveria um erro-padrão várias vezes menor que o real e um
p-valor de brinquedo. O substituto é o bootstrap pareado de municípios sobre a
**diferença** de AUC, com os dois modelos escorando exatamente a mesma
reamostragem — que respeita a estrutura de grupo e não pede normalidade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src import config


def t_nadeau_bengio(delta: np.ndarray, k: int = 5) -> dict[str, float]:
    """Estatística t corrigida sobre os deltas por fold, com IC95."""
    delta = np.asarray(delta, dtype=float)
    n = len(delta)
    correcao = 1 / n + 1 / (k - 1)
    variancia = float(delta.var(ddof=1))
    t = float(delta.mean() / np.sqrt(variancia * correcao)) if variancia > 0 else np.nan
    margem = stats.t.ppf(0.975, df=n - 1) * np.sqrt(variancia * correcao)
    return {
        "n_folds": n,
        "delta_medio": float(delta.mean()),
        "delta_dp": float(delta.std(ddof=1)),
        "folds_a_favor": int((delta > 0).sum()),
        "ic95_baixo": float(delta.mean() - margem),
        "ic95_alto": float(delta.mean() + margem),
        "t_nadeau_bengio": t,
        "p_valor": float(2 * stats.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else np.nan,
    }


def comparar_modelos_em_cv(
    cv: pd.DataFrame, base: str, alternativo: str, metrica: str = "roc_auc", k: int = 5
) -> dict:
    """Delta pareado por fold entre dois modelos da tabela de comparação.

    Exige que os dois tenham rodado nos **mesmos** folds — que é a razão de
    `split.folds` materializar as partições uma vez e passá-las a todo mundo.
    """
    largo = cv.pivot_table(index="fold", columns="modelo", values=metrica)
    faltando = [nome for nome in (base, alternativo) if nome not in largo.columns]
    assert not faltando, f"modelo ausente da tabela de CV: {faltando}"
    delta = (largo[alternativo] - largo[base]).dropna().to_numpy()
    return {"metrica": metrica, "base": base, "alternativo": alternativo, **t_nadeau_bengio(delta, k)}


def bootstrap_pareado_de_auc(
    y,
    escore_a,
    escore_b,
    grupos,
    n_reamostras: int = 1_000,
    seed: int = config.RANDOM_STATE,
) -> dict[str, float]:
    """IC95 da diferença de ROC-AUC, reamostrando municípios e pareando os modelos.

    O pareamento é o ponto: os dois escores são avaliados na mesma reamostragem,
    então a variação comum entre municípios — que é a maior parte da variância —
    se cancela na diferença. Sem parear, o intervalo seria largo o bastante para
    não rejeitar nada.
    """
    y = np.asarray(y)
    a = np.asarray(escore_a, dtype=float)
    b = np.asarray(escore_b, dtype=float)
    grupos = np.asarray(grupos)

    ordem = np.argsort(grupos, kind="stable")
    y, a, b, grupos = y[ordem], a[ordem], b[ordem], grupos[ordem]
    fronteiras = np.flatnonzero(np.diff(grupos)) + 1
    inicios = np.concatenate(([0], fronteiras))
    fins = np.concatenate((fronteiras, [len(y)]))

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_reamostras):
        sorteados = rng.integers(0, len(inicios), len(inicios))
        posicoes = np.concatenate([np.arange(inicios[m], fins[m]) for m in sorteados])
        yb = y[posicoes]
        if yb.min() == yb.max():
            continue
        deltas.append(roc_auc_score(yb, b[posicoes]) - roc_auc_score(yb, a[posicoes]))

    deltas = np.asarray(deltas)
    observado = float(roc_auc_score(y, b) - roc_auc_score(y, a))
    return {
        "auc_base": float(roc_auc_score(y, a)),
        "auc_alternativo": float(roc_auc_score(y, b)),
        "delta": observado,
        "ic95_baixo": float(np.percentile(deltas, 2.5)),
        "ic95_alto": float(np.percentile(deltas, 97.5)),
        # Fração de reamostragens em que o alternativo não supera a base — a
        # leitura direta de "com que frequência esta vantagem some".
        "p_bootstrap": float((deltas <= 0).mean()),
        "n_reamostras": int(len(deltas)),
    }
