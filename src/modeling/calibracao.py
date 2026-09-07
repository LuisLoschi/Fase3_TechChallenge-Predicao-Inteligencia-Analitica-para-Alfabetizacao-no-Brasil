"""Calibração isotônica ajustada em predições out-of-fold agrupadas por município.

A calibração aqui não é refinamento. As probabilidades do modelo de aluno viram,
na Etapa 6, o ranking municipal de risco: `risco_municipal` é a média das
probabilidades preditas dentro do município. Uma probabilidade descalibrada
desloca essa média de forma desigual entre territórios — um viés de 3pp
concentrado nos municípios de baixa taxa reordena o topo do ranking, que é
justamente a parte que decide orçamento.

Por que não `CalibratedClassifierCV` direto. Ele resolveria o caso i.i.d., mas o
splitter aqui precisa de `groups`, e ajustar a isotônica sobre predições que o
modelo já viu no treino produziria uma curva otimista: o modelo é sempre mais
confiante no dado que memorizou. A construção honesta tem três passos, e é o que
esta classe faz:

    1. predições out-of-fold no desenvolvimento, com `StratifiedGroupKFold`
       por município — cada aluno é escorado por um modelo que nunca viu
       o seu município;
    2. isotônica ajustada nesses escores;
    3. reajuste do estimador no desenvolvimento inteiro.

O passo 3 introduz uma folga conhecida: o modelo final foi treinado em mais dado
que os modelos que geraram os escores da calibração, então é ligeiramente mais
nítido do que a curva supõe. É a mesma folga do `ensemble=False` do sklearn, e é
o preço de não jogar fora 20% do desenvolvimento — o efeito é medido no Brier de
teste, contra o modelo sem calibração, e reportado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_is_fitted

# A isotônica é monotônica **não-decrescente**, e a diferença entre "não
# decrescente" e "crescente" custa AUC: ela achata faixas inteiras de escore em
# um mesmo valor, e cada platô vira um bloco de empates que a ROC conta como meio
# acerto. Medido num modelo linear sobre 120 mil alunos, o achatamento tirou
# 0,0004 de ROC-AUC — pouco, mas na direção errada. A métrica primária do projeto
# é ranking; uma etapa que existe para corrigir a *escala* não tem licença para
# mexer na *ordem*.
#
# O desempate devolve a ordem original dentro de cada platô, com uma perturbação
# de no máximo 2e-9 — nove ordens de grandeza abaixo de qualquer diferença que
# importe para a probabilidade, e portanto invisível no Brier e na curva de
# calibração.
EPSILON_DESEMPATE = 1e-9


def desempatar(calibrado: np.ndarray, bruto: np.ndarray) -> np.ndarray:
    """Reintroduz a ordenação original dentro dos platôs da isotônica.

    A contração por `1 - 2ε` antes de somar o posto não é preciosismo: a
    isotônica satura em 0 e em 1, e um deslocamento aditivo puro produziria
    probabilidades fora de `[0, 1]` que o `brier_score_loss` recusa. Com a
    contração, a imagem continua dentro do intervalo e a ordem fica estrita.
    """
    postos = rankdata(bruto, method="average") / len(bruto)  # em (0, 1]
    return calibrado * (1 - 2 * EPSILON_DESEMPATE) + EPSILON_DESEMPATE * (1 + postos)


def avaliar_calibracao(bruto: np.ndarray, y: np.ndarray, particoes) -> dict[str, float]:
    """A isotônica melhora o Brier? A pergunta não pode ser respondida onde ela foi ajustada.

    Comparar o Brier antes e depois usando a mesma curva que se ajustou àqueles
    escores é circular: a isotônica minimiza exatamente esse erro por
    construção, e o ganho aparente é garantido. A comparação honesta ajusta a
    curva em quatro folds e a mede no quinto, com os mesmos folds agrupados por
    município que produziram os escores.

    Ganho positivo significa que a calibração se paga em dado novo. Ganho
    negativo, que o estimador já saía calibrado e a etapa só adiciona ruído — o
    que é o comportamento esperado de um GBM treinado com logloss numa base
    quase equilibrada.
    """
    from sklearn.metrics import brier_score_loss

    fora_do_ajuste = np.full(len(y), np.nan)
    for treino, validacao in particoes:
        curva = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            bruto[treino], y[treino]
        )
        fora_do_ajuste[validacao] = desempatar(curva.predict(bruto[validacao]), bruto[validacao])

    brier_bruto = float(brier_score_loss(y, bruto))
    brier_calibrado = float(brier_score_loss(y, fora_do_ajuste))
    return {
        "brier_bruto": brier_bruto,
        "brier_calibrado": brier_calibrado,
        "ganho_brier": brier_bruto - brier_calibrado,
    }


class ModeloCalibrado(ClassifierMixin, BaseEstimator):
    """`Pipeline` + isotônica. Recebe dado cru e devolve probabilidade calibrada.

    É o objeto que vai para `models/campeao.joblib`: o consumidor passa a linha
    como ela sai de `data/processed/dataset_2024.parquet` e recebe
    `P(risco_nao_alfabetizacao)`, sem precisar saber que houve imputação,
    one-hot ou calibração pelo caminho.
    """

    def __init__(self, estimador, aplicar: bool | None = None):
        self.estimador = estimador
        self.aplicar = aplicar

    def fit(self, X: pd.DataFrame, y, particoes=None):
        """`particoes` são os folds agrupados por município; sem elas, não calibra.

        Exigir as partições em vez de construí-las aqui é deliberado: quem chama
        já as materializou para a comparação de modelos, e reconstruí-las com
        outra seed faria a calibração ser ajustada em folds diferentes dos que
        escolheram o modelo.
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        assert particoes, "sem folds agrupados não há escore out-of-fold para calibrar"

        oof = np.full(len(y), np.nan)
        for treino, validacao in particoes:
            parcial = clone(self.estimador)
            parcial.fit(X.iloc[treino], y[treino])
            oof[validacao] = parcial.predict_proba(X.iloc[validacao])[:, 1]
        assert np.isfinite(oof).all(), "algum aluno ficou sem escore out-of-fold"

        self.escores_oof_ = oof
        self.calibrador_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)
        self.diagnostico_ = avaliar_calibracao(oof, y, particoes)
        self.aplicar_ = (
            self.diagnostico_["ganho_brier"] > 0 if self.aplicar is None else bool(self.aplicar)
        )
        self.estimador_ = clone(self.estimador).fit(X, y)
        return self

    def escore_bruto(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidade antes da isotônica — o par de comparação do Brier."""
        check_is_fitted(self, "estimador_")
        return self.estimador_.predict_proba(X)[:, 1]

    def calibrar(self, bruto: np.ndarray) -> np.ndarray:
        """Aplica a isotônica, se ela tiver se provado útil, e desempata os platôs."""
        check_is_fitted(self, "calibrador_")
        if not self.aplicar_:
            return bruto
        return desempatar(self.calibrador_.predict(bruto), bruto)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        risco = self.calibrar(self.escore_bruto(X))
        return np.column_stack([1.0 - risco, risco])

    def predict(self, X: pd.DataFrame, limiar: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= limiar).astype(int)
