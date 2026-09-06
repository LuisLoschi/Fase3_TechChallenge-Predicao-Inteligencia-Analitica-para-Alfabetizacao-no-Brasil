"""Monta o dataset de modelagem: coorte de 2024 + feature store de 2023.

Uso:
    python -m src.preprocessing.build_dataset [--force]

Entrada:  `data/interim/{gold,aluno}_{2023,2024}.parquet` e os CSVs do INEP.
Saída:    `data/processed/dataset_2024.parquet` — uma linha por aluno de 2024
          com `preenchimento_caderno = 1`, o alvo invertido e as features de lag.

O arquivo carrega três colunas que **não** são features e estão declaradas em
`pipeline.COLS_OPERACIONAIS`: `id_municipio` (grupo de CV), `peso_aluno` (peso de
agregação populacional, B2) e `alfabetizado` (o alvo original). Quem monta a
matriz do modelo usa `pipeline.separar_X_y`, que remove as três e verifica a
interseção com `config.COLS_PROIBIDAS`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import config
from src.data import loader
from src.preprocessing import feature_store as fs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# Colunas da Gold de 2024 que sobrevivem ao dataset. As demais ou são
# constantes dentro de 2024 (`serie`, `meta_alfabetizacao_brasil`,
# `percentual_participacao_brasil`), ou identificadores descartados (`id_aluno`,
# que codifica a UF no prefixo — A3), ou vazamento contemporâneo medido
# (`percentual_participacao_municipio` e `_uf` — B1).
COLS_COORTE = [
    "id_aluno",           # usado só para trazer `peso_aluno`; não entra no dataset
    "id_escola",          # idem: chave do join do controle negativo, descartada depois
    "id_municipio",
    "sigla_uf",
    "nome_regiao",
    "rede",
    "caderno",
    "preenchimento_caderno",
    "alfabetizado",
]


def carregar_coorte(ano: int = config.ANO_ALVO) -> pd.DataFrame:
    """Coorte de 2024 sem as 936 linhas cujo alvo é ausência de medida.

    São alunos com `presenca = 1`, `preenchimento_caderno = 0` e proficiência
    nula, que o ETL marcou como `alfabetizado = 0`. O alvo deles não é desfecho,
    é registro faltante — o mesmo mecanismo dos `presenca = 0` que a Gold já
    filtra. Ficassem no treino, o modelo aprenderia a prever falha de medição.
    """
    coorte = loader.carregar_gold(ano=ano, colunas=COLS_COORTE)
    n_bruto = len(coorte)
    coorte = coorte[coorte["preenchimento_caderno"] == 1].drop(columns=["preenchimento_caderno"])
    log.info("coorte %d: %d linhas, %d removidas por preenchimento_caderno = 0", ano, len(coorte), n_bruto - len(coorte))
    return coorte.reset_index(drop=True)


def anexar_peso(coorte: pd.DataFrame, ano: int = config.ANO_ALVO) -> pd.DataFrame:
    """Traz `peso_aluno` do microdado do **mesmo ano**, por `id_aluno`.

    Este é o único join por `id_aluno` permitido no projeto, e ele só é válido
    porque acontece **dentro** do mesmo ano: a auditoria provou que o conjunto de
    chaves da Gold é idêntico ao dos presentes do microdado, sem duplicata. Entre
    anos o mesmo join é a armadilha A2 e produziria 1,5 milhão de pares falsos —
    daí o `validate="one_to_one"`, que quebra se alguém trocar o ano.
    """
    micro = loader.carregar_aluno(ano=ano, colunas=["id_aluno", "presenca", config.COL_PESO])
    micro = micro.loc[micro["presenca"] == 1, ["id_aluno", config.COL_PESO]]
    saida = coorte.merge(micro, on="id_aluno", how="left", validate="one_to_one")
    assert len(saida) == len(coorte), "o join do peso multiplicou linhas"
    assert saida[config.COL_PESO].notna().all(), "aluno da Gold sem peso no microdado"
    return saida


def juntar_features(coorte: pd.DataFrame, store: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Três joins `many_to_one` com contagem verificada antes e depois.

    Join que multiplica linha em silêncio é a causa nº 1 de métrica inflada, e a
    coorte precisa sair daqui com exatamente as linhas com que entrou.
    """
    n_inicial = len(coorte)
    saida = (
        coorte.merge(store["escola"], on="id_escola", how="left", validate="many_to_one")
        .merge(store["municipio"], on="id_municipio", how="left", validate="many_to_one")
        .merge(store["uf"], on="sigla_uf", how="left", validate="many_to_one")
    )
    assert len(saida) == n_inicial, "o join com a feature store multiplicou linhas"
    return saida


def derivar_colunas(dados: pd.DataFrame) -> pd.DataFrame:
    """Alvo invertido, flags de cobertura, suavização e contraste com a UF."""
    dados = dados.copy()
    dados[config.TARGET] = (1 - dados[config.TARGET_BRUTO]).astype("int8")

    # `tem_historico_municipio` marca SP, DF e AC — 99,92% do gap (B3). É quase
    # colinear com `sigla_uf` e não é nulidade aleatória: toda métrica reportada
    # sobre ela precisa vir estratificada, senão mistura duas populações.
    dados["tem_historico_escola"] = dados["esc_taxa_alfab_lag1"].notna().astype("int8")
    dados["tem_historico_municipio"] = (
        dados["fonte_lag_municipal"].eq(fs.FONTE_GOLD).fillna(False).astype("int8")
    )
    dados["fonte_lag_municipal"] = dados["fonte_lag_municipal"].fillna(fs.FONTE_AUSENTE)
    dados["fonte_lag_uf"] = dados["fonte_lag_uf"].fillna(fs.FONTE_AUSENTE)

    dados["esc_taxa_alfab_lag1_suav"] = fs.suavizar_taxa_escola(dados)
    dados["mun_desvio_vs_uf"] = dados["mun_taxa_alfab_lag1"] - dados["uf_taxa_alfab_lag1"]

    # A rede Privada tem 24 alunos em 2024. Como categoria própria ela produziria
    # um nível de one-hot estimado sobre 24 observações; agrupada em "Outros" ela
    # some sem que a linha seja descartada.
    dados["rede_grupo"] = (
        dados["rede"].map({2: "Estadual", 3: "Municipal"}).fillna("Outros").astype("category")
    )
    # `caderno` fica `int8` de propósito. Ele é categórico por natureza e é
    # tratado como tal pelo `ColumnTransformer`, que roteia **por nome**
    # (`pipeline.COLS_CATEGORICAS_RARAS`) e não por dtype. Deixá-lo numérico
    # mantém a AUC univariada calculável, e é justamente ela que dá o valor de
    # referência do controle negativo: 0,4987.
    return dados.drop(columns=["id_aluno", "id_escola", "rede", config.TARGET_BRUTO])


def verificar(dados: pd.DataFrame) -> None:
    """Asserções anti-leakage do dataset final. Falha aqui **quebra** o build."""
    proibidas_presentes = set(dados.columns) & set(config.COLS_PROIBIDAS)
    permitidas = {"id_municipio", config.COL_PESO}  # operacionais, nunca em X
    assert proibidas_presentes <= permitidas, f"coluna proibida no dataset: {proibidas_presentes - permitidas}"

    assert dados[config.TARGET].isin((0, 1)).all()
    assert dados["id_municipio"].notna().all(), "linha sem grupo de CV"

    # Nenhuma feature pode estar fortemente associada ao alvo: o limite da
    # checklist é 0,95 e a maior correlação medida na EDA foi 0,272. Uma coluna
    # acima disso é vazamento até prova em contrário, não achado.
    numericas = dados.select_dtypes("number").drop(columns=[config.TARGET, "id_municipio", config.COL_PESO])
    amostra = numericas.join(dados[config.TARGET]).sample(300_000, random_state=config.RANDOM_STATE)
    correlacao = amostra.drop(columns=[config.TARGET]).corrwith(amostra[config.TARGET], method="spearman").abs()
    pior = correlacao.idxmax()
    assert correlacao.max() < 0.95, f"{pior} tem |Spearman| {correlacao.max():.3f} com o alvo — investigar vazamento"
    log.info("maior |Spearman| com o alvo: %.4f (%s)", correlacao.max(), pior)


def construir() -> pd.DataFrame:
    coorte = anexar_peso(carregar_coorte())
    store = fs.montar_feature_store()
    dados = derivar_colunas(juntar_features(coorte, store))
    verificar(dados)
    return dados


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regrava mesmo se o Parquet já existir")
    args = parser.parse_args()

    if config.PARQUET_DATASET.exists() and not args.force:
        log.info("%s já existe, pulando (use --force)", config.PARQUET_DATASET)
        return

    dados = construir()
    config.PARQUET_DATASET.parent.mkdir(parents=True, exist_ok=True)
    dados.to_parquet(config.PARQUET_DATASET, compression="zstd", index=False)
    log.info(
        "dataset gravado: %d linhas x %d colunas | %.0f MB | taxa de risco %.4f",
        len(dados),
        dados.shape[1],
        config.PARQUET_DATASET.stat().st_size / 2**20,
        dados[config.TARGET].mean(),
    )

    # A tabela populacional dos dois anos sai junto: a Etapa 6 precisa de 2024
    # para o ranking observado e de 2023 para a evolução, e nenhuma das duas é
    # feature — por isso vivem fora do dataset de modelagem (B2).
    for ano in config.ANOS:
        tabela = fs.agregados_municipais_ponderados(ano)
        destino = Path(str(config.PARQUET_AGREGADO_MUNICIPAL).format(ano=ano))
        tabela.to_parquet(destino, compression="zstd", index=False)
        log.info("agregado municipal ponderado de %d: %d municípios → %s", ano, len(tabela), destino.name)


if __name__ == "__main__":
    main()
