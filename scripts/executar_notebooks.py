"""Executa os notebooks selecionados com o mesmo Python do ambiente ativo.

O backend fica no `matplotlib_inline`, e não em `Agg`: com `Agg` o notebook roda
sem erro e sai **sem nenhuma figura embutida**, porque as células chamam
`salvar_figura` e o `print` do caminho é a última expressão. O PNG em `images/`
continua igual nos dois casos; o que se perde é o notebook como narrativa legível
por quem não vai executá-lo.
"""

import argparse
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", help="nomes ou caminhos dos notebooks")
    args = parser.parse_args()
    # setdefault não serve aqui: reproduzir.py exporta MPLBACKEND=Agg para os
    # comandos de linha, e herdá-lo esvaziaria as figuras do notebook.
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    for nome in args.notebooks:
        caminho = Path(nome)
        if not caminho.is_absolute():
            caminho = ROOT / caminho
        nb = nbformat.read(caminho, as_version=4)
        km = KernelManager(kernel_name="python3")
        # Evita depender de um kernelspec --user, que não viaja com o clone.
        km._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                     display_name="Python do projeto", language="python")
        NotebookClient(nb, km=km, timeout=600,
                       resources={"metadata": {"path": str(ROOT)}}, allow_errors=False).execute()
        nbformat.write(nb, caminho)
        print(f"Executado sem erros: {caminho.name}", flush=True)


if __name__ == "__main__":
    main()
