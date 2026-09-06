"""Confere as sete fontes por SHA-256. Não baixa nem publica microdados.

Uso: python scripts/verificar_insumos.py [--gravar-manifesto]
TC_DATA_DIR permite apontar para uma cópia dos dados fornecida pelo grupo.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

MANIFESTO = config.ROOT / "data" / "manifesto_fontes.json"
FONTES = [config.CSV_GOLD, config.CSV_ALUNO, config.CSV_META_MUNICIPIO,
          config.CSV_META_UF, config.CSV_META_BRASIL, config.CSV_AGREGADO_MUNICIPIO,
          config.CSV_AGREGADO_UF]


def sha256(caminho: Path) -> str:
    digestor = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            digestor.update(bloco)
    return digestor.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gravar-manifesto", action="store_true",
                        help="registra explicitamente uma nova versão das fontes locais")
    args = parser.parse_args()
    faltantes = [str(p.relative_to(config.DIR_DATA)) for p in FONTES if not p.is_file()]
    if faltantes:
        raise SystemExit("Insumos ausentes: " + ", ".join(faltantes) +
                         ". Obtenha o pacote de dados do grupo; consulte data/README.md.")
    registros = [{"caminho": p.relative_to(config.DIR_DATA).as_posix(),
                  "bytes": p.stat().st_size, "sha256": sha256(p)} for p in FONTES]
    if args.gravar_manifesto:
        MANIFESTO.write_text(json.dumps({"fontes": registros}, indent=2) + "\n", encoding="utf-8")
    elif not MANIFESTO.exists():
        raise SystemExit("Manifesto ausente; obtenha a versão correspondente ao código.")
    elif json.loads(MANIFESTO.read_text(encoding="utf-8"))["fontes"] != registros:
        raise SystemExit("Fontes diferentes do manifesto. Não misture métricas e versões de dados.")
    print(f"{len(registros)} fontes conferidas por SHA-256.")


if __name__ == "__main__":
    main()
