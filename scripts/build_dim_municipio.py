"""Extrai a dimensão territorial `id_municipio → UF / região` a partir da Gold.

O microdado do INEP só traz `id_municipio`; a Gold é a única base do projeto
que carrega `sigla_uf`, `nome_uf` e `nome_regiao`. Sem essa dimensão não é
possível montar agregados de UF para 2023, que é o ano das features de lag.

O CSV resultante é pequeno e **versionado por exceção** no `.gitignore`.

Uso:
    python scripts/build_dim_municipio.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data import loader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
log = logging.getLogger(__name__)

COLUNAS = ["id_municipio", "nome_municipio", "sigla_uf", "nome_uf", "nome_regiao"]


def main() -> None:
    partes = [loader.carregar_gold(ano=ano, colunas=COLUNAS) for ano in config.ANOS]
    bruto = pd.concat(partes, ignore_index=True)

    # A checagem tem de vir ANTES do `drop_duplicates`: depois dele sobra uma linha
    # por município e `nunique() == 1` passa por construção, provando nada. Um
    # município com duas UFs entre os dois anos indicaria erro na origem.
    conflitos = bruto.groupby("id_municipio")["sigla_uf"].nunique()
    divergentes = conflitos[conflitos > 1]
    assert divergentes.empty, f"município com mais de uma UF na Gold: {list(divergentes.index)[:5]}"

    dim = bruto.drop_duplicates(subset=["id_municipio"])
    dim = dim.sort_values("id_municipio").reset_index(drop=True)

    config.CSV_DIM_MUNICIPIO.parent.mkdir(parents=True, exist_ok=True)
    dim.to_csv(config.CSV_DIM_MUNICIPIO, index=False, encoding="utf-8")
    log.info("%d municípios em %d UFs → %s", len(dim), dim["sigla_uf"].nunique(), config.CSV_DIM_MUNICIPIO)


if __name__ == "__main__":
    main()
