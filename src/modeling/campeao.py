"""Refit e avaliação retrospectiva do classificador.

A separação protege o ajuste deste módulo. O teste histórico foi consultado
na seleção de atributos; STATUS_VALIDACAO preserva essa limitação.
Limiar e calibração deste módulo são escolhidos no desenvolvimento.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src import config
from src.data import loader
from src.evaluation import metrics as mt
from src.evaluation.protocolo import STATUS_VALIDACAO
from src.modeling import calibracao, split, tuning
from src.preprocessing import pipeline as pl

log = logging.getLogger(__name__)

MODELO_CAMPEAO = config.DIR_MODELS / "campeao.joblib"
JSON_CAMPEAO = config.DIR_METRICS / "campeao.json"
CSV_ESTRATIFICADO = config.DIR_METRICS / "campeao_estratificado.csv"
CSV_CALIBRACAO = config.DIR_METRICS / "campeao_calibracao.csv"
CSV_CAPACIDADE = config.DIR_METRICS / "campeao_capacidade.csv"
CSV_DECIS = config.DIR_METRICS / "campeao_decis.csv"
CSV_R2_MUNICIPAL = config.DIR_METRICS / "campeao_r2_municipal.csv"
PARQUET_OOF = config.DIR_MODELS / "escores_oof_desenvolvimento.parquet"

# Fração da coorte que um programa de intervenção consegue atender. Não é um
# número medido: é uma premissa operacional declarada, e o relatório apresenta a
# curva inteira justamente para que quem tem o orçamento escolha outra.
FRACAO_DE_CAPACIDADE = 0.20


def hash_do_dataset() -> str:
    """MD5 do Parquet de modelagem — o número reportado vale para *este* arquivo."""
    digestor = hashlib.md5()
    with config.PARQUET_DATASET.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            digestor.update(bloco)
    return digestor.hexdigest()


def _quintis_de_presenca(coluna: pd.Series) -> pd.Series:
    """Quintis da taxa de presença municipal de 2023, com os nulos como nível próprio.

    A EDA mediu que a taxa de alfabetização observada é otimista onde a presença
    é baixa — 53,4% no quintil de menor presença contra 74,6% no de maior. Se o
    modelo errar sistematicamente nessa faixa, o ranking de risco subestima o
    problema exatamente onde ele é maior.
    """
    rotulos = ["q1_menor_presenca", "q2", "q3", "q4", "q5_maior_presenca"]
    quintis = pd.qcut(coluna, 5, labels=rotulos)
    return quintis.cat.add_categories(["sem_historico"]).fillna("sem_historico").astype(str)


def etapa_campeao(seed: int = config.RANDOM_STATE) -> dict:
    inicio = time.time()
    dados = loader.carregar_dataset_modelagem()
    X, y, grupos = pl.separar_X_y(dados)
    dev, teste = split.separar_desenvolvimento_e_teste(grupos, seed=seed)
    split.verificar_grupos_disjuntos(grupos, dev, teste)

    X_dev, y_dev, g_dev = X.iloc[dev], y.iloc[dev], grupos.iloc[dev]
    X_teste, y_teste, g_teste = X.iloc[teste], y.iloc[teste], grupos.iloc[teste]
    particoes = split.folds(y_dev.reset_index(drop=True), g_dev.reset_index(drop=True), seed=seed)

    parametros = tuning.carregar_hiperparametros()
    estimador = pl.montar_pipeline(LGBMClassifier(random_state=seed, **parametros))
    modelo = calibracao.ModeloCalibrado(estimador).fit(X_dev, y_dev, particoes=particoes)
    log.info("campeão ajustado em %d alunos de %d municípios", len(X_dev), g_dev.nunique())

    # --- decisões tomadas no desenvolvimento, com escore out-of-fold -------
    oof_bruto = modelo.escores_oof_
    oof_calibrado = modelo.calibrar(oof_bruto)
    log.info(
        "calibração: Brier %.5f bruto contra %.5f isotônico fora do ajuste — aplicar = %s",
        modelo.diagnostico_["brier_bruto"], modelo.diagnostico_["brier_calibrado"], modelo.aplicar_,
    )

    limiar = mt.escolher_limiar_por_f1(y_dev, oof_calibrado)
    # O F1 não conhece orçamento. Numa base de prevalência 0,40 com ranking
    # fraco, maximizá-lo empurra o corte para baixo até alertar quase toda a
    # coorte — matematicamente correto e operacionalmente inútil. O limiar por
    # capacidade é o que uma secretaria consegue executar, e vai ao lado.
    limiar_capacidade = float(np.quantile(oof_calibrado, 1 - FRACAO_DE_CAPACIDADE))
    log.info(
        "limiar por F1 nos escores out-of-fold: %.4f | por capacidade de %.0f%%: %.4f",
        limiar, FRACAO_DE_CAPACIDADE * 100, limiar_capacidade,
    )

    config.DIR_MODELS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id_municipio": g_dev.to_numpy(),
            "risco_nao_alfabetizacao": y_dev.to_numpy(),
            "escore_oof": oof_calibrado,
            "peso_aluno": dados[config.COL_PESO].iloc[dev].to_numpy(),
        }
    ).to_parquet(PARQUET_OOF, index=False)

    # --- avaliação da reserva histórica --------------------------------------
    escore_bruto = modelo.escore_bruto(X_teste)
    escore = modelo.calibrar(escore_bruto)
    escore_isotonico = calibracao.desempatar(
        modelo.calibrador_.predict(escore_bruto), escore_bruto
    )

    ranking = mt.metricas_de_ranking(y_teste, escore)
    ranking_bruto = mt.metricas_de_ranking(y_teste, escore_bruto)
    ranking_isotonico = mt.metricas_de_ranking(y_teste, escore_isotonico)
    no_limiar = mt.metricas_no_limiar(y_teste, escore, limiar)
    no_limiar_capacidade = mt.metricas_no_limiar(y_teste, escore, limiar_capacidade)
    ic = mt.bootstrap_por_municipio(y_teste, escore, g_teste, seed=seed)
    log.info("teste: ROC-AUC %.4f | PR-AUC %.4f | Brier %.4f", ranking["roc_auc"], ranking["pr_auc"], ranking["brier"])

    contexto = dados.iloc[teste]
    estratos = {
        "tem_historico_municipio": contexto["tem_historico_municipio"].astype(int).astype(str),
        "fonte_lag_municipal": contexto["fonte_lag_municipal"].astype(str),
        "nome_regiao": contexto["nome_regiao"].astype(str),
        "sigla_uf": contexto["sigla_uf"].astype(str),
        "rede_grupo": contexto["rede_grupo"].astype(str),
        "quintil_presenca_municipal": _quintis_de_presenca(contexto["mun_taxa_presenca_lag1"]),
        "tem_lag_de_uf": np.where(contexto["uf_taxa_alfab_lag1"].isna(), "AC_e_DF", "demais_ufs"),
    }
    estratificado = pd.concat(
        [
            mt.metricas_estratificadas(y_teste, escore, valores).assign(dimensao=nome)
            for nome, valores in estratos.items()
        ],
        ignore_index=True,
    )[["dimensao", "estrato", "n", "prevalencia", "roc_auc", "pr_auc", "ks", "brier"]]

    calibracao_teste = mt.curva_calibracao(y_teste, escore_isotonico)
    calibracao_bruta = mt.curva_calibracao(y_teste, escore_bruto)
    capacidade = mt.curva_de_capacidade(y_teste, escore)
    decis = mt.lift_por_decil(y_teste, escore)
    r2 = mt.r2_municipal(y_teste, escore, g_teste, dados[config.COL_PESO].iloc[teste])

    config.DIR_METRICS.mkdir(parents=True, exist_ok=True)
    estratificado.to_csv(CSV_ESTRATIFICADO, index=False)
    pd.concat(
        [calibracao_teste.assign(versao="calibrado"), calibracao_bruta.assign(versao="bruto")]
    ).to_csv(CSV_CALIBRACAO, index=False)
    capacidade.to_csv(CSV_CAPACIDADE, index=False)
    decis.to_csv(CSV_DECIS, index=False)
    r2.to_csv(CSV_R2_MUNICIPAL, index=False)

    resumo = {
        "status_validacao": dict(STATUS_VALIDACAO),
        "hiperparametros": parametros,
        "seed": seed,
        "hash_dataset": hash_do_dataset(),
        "n_features_apos_pre_processamento": len(pl.nomes_das_features(modelo.estimador_)),
        "desenvolvimento": {
            "n_alunos": int(len(X_dev)),
            "n_municipios": int(g_dev.nunique()),
            "prevalencia": float(y_dev.mean()),
            "roc_auc_out_of_fold": float(mt.metricas_de_ranking(y_dev, oof_calibrado)["roc_auc"]),
        },
        "calibracao": {
            **modelo.diagnostico_,
            "aplicada": bool(modelo.aplicar_),
            "criterio": (
                "isotônica ajustada em 4 folds e medida no quinto; aplicada só se "
                "o Brier melhorar fora do próprio ajuste"
            ),
        },
        "teste": {
            "n_alunos": int(len(X_teste)),
            "n_municipios": int(g_teste.nunique()),
            "publicado": ranking,
            "sem_calibracao": ranking_bruto,
            "com_isotonica": ranking_isotonico,
            "no_limiar": no_limiar,
            "no_limiar_de_capacidade": no_limiar_capacidade,
            "intervalos_bootstrap": ic.to_dict("records"),
            "r2_municipal": r2.to_dict("records"),
            "capacidade": capacidade.to_dict("records"),
        },
        "limiar": {
            "por_f1": float(limiar),
            "por_capacidade": float(limiar_capacidade),
            "fracao_de_capacidade": FRACAO_DE_CAPACIDADE,
            "criterio": (
                "ambos escolhidos nos escores out-of-fold do desenvolvimento; a "
                "reserva histórica não participa da escolha destes limiares"
            ),
        },
        "segundos": round(time.time() - inicio, 1),
    }
    JSON_CAMPEAO.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(
        {
            "modelo": modelo,
            "limiar_por_f1": float(limiar),
            "limiar_por_capacidade": float(limiar_capacidade),
            "features": list(pl.FEATURES_MODELO),
            "nomes_apos_pre_processamento": pl.nomes_das_features(modelo.estimador_),
            "seed": seed,
            "hash_dataset": resumo["hash_dataset"],
            "versoes": {"sklearn": __import__("sklearn").__version__, "lightgbm": __import__("lightgbm").__version__},
        },
        MODELO_CAMPEAO,
    )
    log.info("campeão serializado em %s (%.0fs no total)", MODELO_CAMPEAO, resumo["segundos"])
    return resumo


def carregar_campeao() -> dict:
    """Artefato do campeão: modelo calibrado, limiar e nomes de feature.

    Os nomes já vêm resolvidos porque é isso que a Etapa 5 precisa — sem eles o
    SHAP explicaria `x17` para quem decide orçamento.
    """
    pacote = joblib.load(MODELO_CAMPEAO)
    pacote["status_validacao"] = dict(STATUS_VALIDACAO)
    return pacote
