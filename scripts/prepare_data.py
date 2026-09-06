"""Converte os CSVs brutos em Parquet, um arquivo por ano.

Motivação: a Gold tem 615 MB e o microdado 214 MB em CSV. Lidos com dtypes
largos, os dois juntos não cabem confortavelmente na memória disponível.
A conversão é feita em streaming (lotes do pyarrow), com tipos estreitos e
compressão zstd, e o resultado é lido depois em segundos.

Uso:
    python scripts/prepare_data.py [--force]
"""

import argparse
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

TAMANHO_LOTE = 500_000

# ---------------------------------------------------------------------------
# Esquemas — tipos estreitos, escolhidos a partir dos valores reais das bases
# ---------------------------------------------------------------------------
TIPOS_GOLD = {
    "ano": pa.int16(),
    "id_aluno": pa.int32(),
    "id_escola": pa.int32(),
    "id_municipio": pa.int32(),
    "nome_municipio": pa.string(),
    "sigla_uf": pa.string(),
    "nome_uf": pa.string(),
    "nome_regiao": pa.string(),
    "serie": pa.int8(),
    "serie_nome": pa.string(),
    "rede": pa.int8(),
    "rede_nome": pa.string(),
    "caderno": pa.int8(),
    "preenchimento_caderno": pa.float32(),
    "meta_alfabetizacao_municipio": pa.float32(),
    "percentual_participacao_municipio": pa.float32(),
    "meta_alfabetizacao_uf": pa.float32(),
    "percentual_participacao_uf": pa.float32(),
    "meta_alfabetizacao_brasil": pa.float32(),
    "percentual_participacao_brasil": pa.float32(),
    "alfabetizado": pa.int8(),
}

TIPOS_ALUNO = {
    "ano": pa.int16(),
    "id_municipio": pa.int32(),
    "id_escola": pa.int32(),
    "id_aluno": pa.int32(),
    "caderno": pa.int8(),
    "serie": pa.int8(),
    "rede": pa.int8(),
    "presenca": pa.int8(),
    "preenchimento_caderno": pa.int8(),
    "alfabetizado": pa.int8(),
    "proficiencia": pa.float32(),
    "peso_aluno": pa.float32(),
}


def converter(origem: Path, destino_template: str, tipos: dict, descartar: tuple = ()) -> dict:
    """Lê um CSV em lotes e grava um Parquet por ano. Devolve as contagens."""
    if not origem.exists():
        raise FileNotFoundError(f"CSV de origem ausente: {origem}")

    log.info("lendo %s (%.0f MB)", origem.name, origem.stat().st_size / 2**20)
    leitor = pacsv.open_csv(
        origem,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(column_types=tipos, include_columns=list(tipos)),
    )

    escritores: dict[int, pq.ParquetWriter] = {}
    destinos: dict[int, Path] = {}
    contagens: dict[int, int] = {}
    try:
        for lote in leitor:
            tabela = pa.Table.from_batches([lote])
            if descartar:
                tabela = tabela.drop_columns([c for c in descartar if c in tabela.column_names])
            for ano in config.ANOS:
                fatia = tabela.filter(pc.equal(tabela["ano"], ano))
                if fatia.num_rows == 0:
                    continue
                if ano not in escritores:
                    destinos[ano] = Path(destino_template.format(ano=ano))
                    destinos[ano].parent.mkdir(parents=True, exist_ok=True)
                    escritores[ano] = pq.ParquetWriter(destinos[ano], fatia.schema, compression="zstd")
                escritores[ano].write_table(fatia)
                contagens[ano] = contagens.get(ano, 0) + fatia.num_rows
    finally:
        for escritor in escritores.values():
            escritor.close()

    for ano, n in sorted(contagens.items()):
        mb = destinos[ano].stat().st_size / 2**20
        log.info("  %s → %s | %d linhas | %.0f MB", ano, destinos[ano].name, n, mb)
    return contagens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="reconverte mesmo se o Parquet já existir")
    args = parser.parse_args()

    tarefas = (
        ("Gold (grão aluno, alvo de 2024)", config.CSV_GOLD, str(config.PARQUET_GOLD), TIPOS_GOLD, config.COLS_AUDITORIA),
        ("microdado INEP (peso_aluno, presenca e reconciliação)", config.CSV_ALUNO, str(config.PARQUET_ALUNO), TIPOS_ALUNO, ()),
    )

    for descricao, origem, destino, tipos, descartar in tarefas:
        existentes = [Path(destino.format(ano=a)) for a in config.ANOS]
        if all(p.exists() for p in existentes) and not args.force:
            log.info("%s: Parquet já existe, pulando (use --force para refazer)", descricao)
            continue
        log.info("convertendo %s", descricao)
        converter(origem, destino, tipos, descartar)

    log.info("concluído — Parquet em %s", config.DIR_INTERIM)


if __name__ == "__main__":
    main()
