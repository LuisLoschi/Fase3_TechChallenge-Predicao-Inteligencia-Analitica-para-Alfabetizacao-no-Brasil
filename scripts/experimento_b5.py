"""Réplica do experimento B5, restrita ao desenvolvimento: os percentis pagam o próprio custo?

A auditoria da Etapa 2.5 mediu +0,0022 de ROC-AUC para o bloco de percentis de
proficiência, com IC pareado incluindo zero — sobre **um** split, **um** conjunto
de hiperparâmetros e **um** modelo. A réplica histórica ampliou o desenho, mas
comparava conjuntos de atributos sobre a coorte inteira de 2024: os municípios
depois chamados de reserva participaram de uma decisão de modelagem, e por isso
as métricas históricas não são uma avaliação confirmatória independente. Aquela
saída está preservada em `experimento_b5_replica.csv` como registro do que foi
feito, não como evidência de generalização.

Esta versão separa a reserva **antes** de qualquer seleção, com a seed fixa do
projeto, e roda `StratifiedGroupKFold` por município apenas sobre o
desenvolvimento. As seeds do argumento alteram só a validação cruzada, nunca a
partição reservada. Isso corrige o fluxo para decisões futuras; não restaura a
independência da reserva já consultada, que depende de amostra nova.

Três conjuntos, e não dois, porque a Etapa 3 descobriu uma segunda pergunta:

    A  sem os percentis de proficiência ............. o microdado ainda entra
                                                       por presença e por peso;
                                                       o que sai é o único bloco
                                                       que **só** ele produz
    B  A + percentis de proficiência do microdado .... a decisão pendente de B5
    C  B + bloco de escola ........................... a ablação do join por
                                                       `id_escola`, que se
                                                       revelou chave reciclada

Custo medido na réplica histórica: ~45 s por ajuste, 45 ajustes, cerca de 35 min
em 16 núcleos. `--n-linhas` reduz o orçamento por municípios inteiros, para
verificar o fluxo sem pagar a execução completa.

Uso:
    python scripts/experimento_b5.py                       # 3 seeds x 5 folds no desenvolvimento
    python scripts/experimento_b5.py --seeds 42 --folds 3 --n-linhas 150000
Saída:
    reports/metrics/experimento_b5_desenvolvimento.csv   (uma linha por seed x fold x conjunto)
    reports/metrics/experimento_b5_desenvolvimento.json  (escopo, seed da reserva e ressalva)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data import loader  # noqa: E402
from src.modeling import campeao, split  # noqa: E402
from src.preprocessing import pipeline as pl  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
)
log = logging.getLogger(__name__)

SEEDS = (42, 7, 2024)
N_SPLITS = 5

# Hiperparâmetros fixos e idênticos entre conjuntos: o experimento compara
# *features*, e deixar a busca solta transformaria a comparação em ruído de
# tuning. Não são os hiperparâmetros do campeão — isso é a Etapa 4.
PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)

CONJUNTOS = {
    "A_sem_percentis": list(pl.FEATURES_MODELO),
    "B_com_percentis": list(pl.FEATURES_MODELO) + pl.FEATURES_PERCENTIS_MICRODADO,
    "C_com_escola": list(pl.FEATURES_MODELO) + pl.FEATURES_PERCENTIS_MICRODADO + pl.FEATURES_ESCOLA,
}

SAIDA = config.DIR_METRICS / "experimento_b5_desenvolvimento.csv"


def selecionar_desenvolvimento(dados: pd.DataFrame, n_linhas: int | None = None) -> pd.DataFrame:
    """Reserva fixa antes de qualquer seleção; seeds do experimento só alteram a CV."""
    grupos = dados[config.GRUPO_CV]
    dev, teste = split.separar_desenvolvimento_e_teste(grupos, seed=config.RANDOM_STATE)
    if n_linhas is not None:
        if n_linhas <= 0:
            raise ValueError("n_linhas deve ser positivo")
        dev = split.amostrar_municipios(grupos, n_linhas, posicoes=dev)
    split.verificar_grupos_disjuntos(grupos, dev, teste)
    return dados.iloc[dev].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="B5 somente no desenvolvimento; não restaura a independência histórica.")
    parser.add_argument("--n-linhas", type=int, help="orçamento aproximado, por municípios inteiros")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    args = parser.parse_args()
    dados = selecionar_desenvolvimento(loader.carregar_dataset_modelagem(), args.n_linhas)
    y, grupos = dados[config.TARGET], dados[config.GRUPO_CV]
    SAIDA.parent.mkdir(parents=True, exist_ok=True)

    linhas: list[dict] = []
    SAIDA.with_suffix(".json").write_text(json.dumps({
        "hash_dataset": campeao.hash_do_dataset(), "seed_reserva": config.RANDOM_STATE,
        "seeds_cv": args.seeds, "folds": args.folds, "n_alunos": len(dados),
        "n_municipios": int(grupos.nunique()), "n_linhas_solicitadas": args.n_linhas,
        "escopo": "desenvolvimento", "execucao_completa": False,
        "ressalva": "A reserva histórica já foi consultada por experimentos anteriores; resultado exploratório.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for seed in args.seeds:
        cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (treino, teste) in enumerate(cv.split(dados, y, grupos)):
            split.verificar_grupos_disjuntos(grupos, treino, teste)
            for nome, features in CONJUNTOS.items():
                X, _, _ = pl.separar_X_y(dados, features)
                inicio = time.time()
                modelo = pl.montar_pipeline(LGBMClassifier(random_state=seed, **PARAMS), features)
                modelo.fit(X.iloc[treino], y.iloc[treino])
                escore = modelo.predict_proba(X.iloc[teste])[:, 1]
                linhas.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "conjunto": nome,
                        "n_features": len(features),
                        "roc_auc": roc_auc_score(y.iloc[teste], escore),
                        "pr_auc": average_precision_score(y.iloc[teste], escore),
                        "n_teste": int(len(teste)),
                        "n_municipios_teste": int(grupos.iloc[teste].nunique()),
                        "segundos": round(time.time() - inicio, 1),
                    }
                )
                log.info("seed %s fold %s %s: ROC-AUC %.4f", seed, fold, nome, linhas[-1]["roc_auc"])
                # Gravação a cada ajuste: 35 minutos é tempo suficiente para a
                # máquina ser desligada no meio.
                pd.DataFrame(linhas).to_csv(SAIDA, index=False)

    manifesto = json.loads(SAIDA.with_suffix(".json").read_text(encoding="utf-8"))
    manifesto["execucao_completa"] = True
    SAIDA.with_suffix(".json").write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("concluído — %d linhas em %s", len(linhas), SAIDA)


if __name__ == "__main__":
    main()
