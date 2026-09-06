"""Os dois pisos que o campeão precisa superar.

O primeiro é o piso absoluto: `DummyClassifier(strategy="prior")` devolve a
prevalência para todo mundo e tem ROC-AUC 0,500 por construção. Ele existe para
que o Brier e o PR-AUC tenham referência — um PR-AUC de 0,53 só significa alguma
coisa ao lado da prevalência de 0,40.

O segundo é o piso que importa. O diagnóstico mediu ROC-AUC 0,654 para "ranquear
os alunos pela taxa de alfabetização de 2023 do seu município", uma regra que
cabe numa linha de SQL e não precisa de modelo nenhum. Esse número, porém, foi
medido em outro esquema de validação, e comparar contra ele seria comparar coisas
diferentes: por isso o baseline é reimplementado aqui como estimador sklearn, para
passar pelos **mesmos folds** e pelo **mesmo conjunto de teste** que os candidatos.

Uma escolha de desenho vale explicação. `BaselineUnivariado` traduz o valor da
coluna em probabilidade com uma isotônica ajustada no fold de treino, em vez de
usar `1 - taxa/100` direto. Duas razões: a AUC é invariante à transformação
monotônica, então nada muda no ranking, mas o Brier passa a ser comparável ao dos
demais modelos — sem isso, o baseline seria punido por estar numa escala que
ninguém pediu que ele acertasse. E a mesma classe passa a servir para
`mun_media_portugues_lag1`, que está em escala de proficiência e não tem leitura
de probabilidade nenhuma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_is_fitted

from src.modeling.calibracao import desempatar


class BaselineUnivariado(ClassifierMixin, BaseEstimator):
    """Ranqueia pelo valor de uma única coluna, sem interação e sem ajuste de forma.

    `sinal = -1` diz que valores altos da coluna significam risco *baixo*, que é
    o caso da taxa de alfabetização e da média de português de 2023.

    Os nulos vão para a prevalência do treino, e não para a mediana da coluna:
    quem não tem histórico municipal são SP, DF e AC inteiros (B3), e atribuir a
    eles o valor mediano do país seria inventar uma informação territorial que a
    regra não tem. Ficar no meio da fila é a leitura honesta de "não sei".
    """

    def __init__(self, coluna: str, sinal: float = -1.0):
        self.coluna = coluna
        self.sinal = sinal

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.prevalencia_ = float(y.mean())
        bruto = self.sinal * X[self.coluna].to_numpy(dtype=float)
        observado = np.isfinite(bruto)
        self.isotonica_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            bruto[observado], y[observado]
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "isotonica_")
        bruto = self.sinal * X[self.coluna].to_numpy(dtype=float)
        risco = np.full(len(bruto), self.prevalencia_)
        observado = np.isfinite(bruto)
        # `desempatar` devolve a ordenação da coluna dentro dos platôs da
        # isotônica. Sem ele o baseline seria avaliado com empates que a regra
        # original não tem, e chegaria à comparação com a AUC subestimada — um
        # erro na direção exata que favoreceria o campeão.
        risco[observado] = desempatar(
            self.isotonica_.predict(bruto[observado]), bruto[observado]
        )
        return np.column_stack([1.0 - risco, risco])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def baseline_taxa_municipal() -> BaselineUnivariado:
    """A regra que o projeto precisa bater: a taxa de 2023 do próprio município."""
    return BaselineUnivariado("mun_taxa_alfab_lag1")


def baseline_proficiencia_municipal() -> BaselineUnivariado:
    """A variante mais forte medida na EDA — mesma ideia, coluna contínua."""
    return BaselineUnivariado("mun_media_portugues_lag1")
