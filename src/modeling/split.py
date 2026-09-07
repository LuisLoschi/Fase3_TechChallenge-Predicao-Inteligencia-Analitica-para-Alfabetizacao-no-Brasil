"""Separação treino/teste e esquema de validação cruzada, sempre por município.

Um `KFold` aleatório aqui produziria um número bonito e falso. As features que
carregam o sinal são municipais — `mun_media_portugues_lag1`, `mun_taxa_alfab_lag1`
—, então alunos do mesmo município são quase réplicas uns dos outros: com split
aleatório o modelo memoriza o nível de cada município no treino e o reencontra no
teste. Agrupando por `id_municipio`, todo município do teste é inédito, e acertar
exige que a *relação* "contexto histórico de 2023 → risco em 2024" generalize
para territórios nunca vistos.

O custo desse rigor foi medido no diagnóstico: 0,667 de ROC-AUC no split aleatório
contra 0,649 no agrupado. A diferença de 0,018 é o tamanho da memorização
territorial que o desenho honesto abre mão de contabilizar.

Três funções e um invariante:

    separar_desenvolvimento_e_teste   80/20 por município, tocado uma vez ao final
    cv_agrupada                       StratifiedGroupKFold(5) para toda a seleção
    amostrar_municipios               subamostra para tuning, municípios inteiros

O invariante é `verificar_grupos_disjuntos`, chamado por dentro das duas
primeiras: nenhum `id_municipio` pode aparecer dos dois lados de um split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src import config

FRACAO_TESTE = 0.20
N_SPLITS = 5


def verificar_grupos_disjuntos(grupos: pd.Series, a: np.ndarray, b: np.ndarray) -> None:
    """Falha se algum município aparecer dos dois lados. Chamado em todo split."""
    comuns = set(grupos.iloc[a].unique()) & set(grupos.iloc[b].unique())
    assert not comuns, f"{len(comuns)} municípios em ambos os lados do split: {sorted(comuns)[:5]}"


def separar_desenvolvimento_e_teste(
    grupos: pd.Series, fracao_teste: float = FRACAO_TESTE, seed: int = config.RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """80% de desenvolvimento e 20% de teste, por município inteiro.

    A fração é de *municípios*, não de alunos, então a partição de linhas fica
    perto de 80/20 mas não exata — municípios têm portes muito diferentes.
    Devolve posições inteiras (`iloc`), não rótulos de índice.
    """
    divisor = GroupShuffleSplit(n_splits=1, test_size=fracao_teste, random_state=seed)
    dev, teste = next(divisor.split(np.zeros(len(grupos)), groups=grupos))
    verificar_grupos_disjuntos(grupos, dev, teste)
    return dev, teste


def cv_agrupada(
    n_splits: int = N_SPLITS, seed: int = config.RANDOM_STATE
) -> StratifiedGroupKFold:
    """`StratifiedGroupKFold` — grupo por município, estratificação pelo alvo.

    A estratificação importa mesmo com a base quase equilibrada (prevalência
    0,4022): a taxa de risco varia de 0,15 a 0,75 entre municípios, e folds
    formados por município sem estratificar chegam a divergir vários pontos
    percentuais entre si, o que vira variância espúria na comparação de modelos.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def folds(
    y: pd.Series, grupos: pd.Series, n_splits: int = N_SPLITS, seed: int = config.RANDOM_STATE
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materializa os folds uma vez, para que todo modelo veja exatamente os mesmos.

    Comparar modelos em folds diferentes mistura a diferença entre eles com a
    diferença entre partições — e o desvio entre folds medido na Etapa 3 (0,0134
    de ROC-AUC) é maior que qualquer ganho que se espere aqui.
    """
    particoes = list(cv_agrupada(n_splits, seed).split(np.zeros(len(y)), y, grupos))
    for treino, validacao in particoes:
        verificar_grupos_disjuntos(grupos, treino, validacao)
    return particoes


def amostrar_municipios(
    grupos: pd.Series,
    n_linhas_alvo: int,
    seed: int = config.RANDOM_STATE,
    posicoes: np.ndarray | None = None,
) -> np.ndarray:
    """Subamostra para tuning sorteando **municípios inteiros**, nunca alunos.

    Sortear alunos individuais quebraria a estrutura de grupo: metade de um
    município no treino e metade na validação reintroduz exatamente a
    memorização que o `StratifiedGroupKFold` existe para impedir (vetor 11 da
    matriz anti-leakage).

    Sorteio uniforme sobre municípios, sem ponderar por porte — ponderar por
    número de alunos faria a amostra ser dominada por capitais e mudaria a
    distribuição de porte, que é uma das features.
    """
    universo = grupos if posicoes is None else grupos.iloc[posicoes]
    tamanhos = universo.value_counts()
    rng = np.random.default_rng(seed)
    ordem = rng.permutation(tamanhos.index.to_numpy())
    acumulado = tamanhos.loc[ordem].cumsum().to_numpy()
    n = int(np.searchsorted(acumulado, n_linhas_alvo) + 1)
    escolhidos = set(ordem[:n].tolist())

    marcados = universo.isin(escolhidos).to_numpy()
    return np.arange(len(grupos))[posicoes][marcados] if posicoes is not None else np.flatnonzero(marcados)
