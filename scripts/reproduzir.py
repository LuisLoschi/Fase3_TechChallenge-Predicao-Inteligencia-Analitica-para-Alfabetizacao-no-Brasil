"""Orquestra etapas locais; 'tudo' inclui treinos demorados, explicitamente selecionados."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ETAPAS = {
    "dados": [["scripts/verificar_insumos.py"], ["scripts/prepare_data.py"],
              ["scripts/build_dim_municipio.py"], ["-m", "src.preprocessing.build_dataset"]],
    "modelos": [["scripts/experimento_b5.py"], ["-m", "src.modeling.train", "tuning"],
                ["-m", "src.modeling.train", "comparacao"], ["-m", "src.modeling.train", "campeao"],
                ["-m", "src.evaluation.rigor"], ["-m", "src.evaluation.interpret"]],
    "relatorios": [["-m", "src.modeling.strategic"], ["scripts/atualizar_relatorios.py"]],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etapa", choices=[*ETAPAS, "tudo"], default="relatorios")
    args = parser.parse_args()
    ambiente = dict(os.environ, PYTHONHASHSEED="42", PYTHONUTF8="1", MPLBACKEND="Agg")
    etapas = ETAPAS if args.etapa == "tudo" else [args.etapa]
    for etapa in etapas:
        for comando in ETAPAS[etapa]:
            print("Executando:", sys.executable, *comando, flush=True)
            subprocess.run([sys.executable, "-B", *comando], cwd=ROOT, env=ambiente, check=True)


if __name__ == "__main__":
    main()
