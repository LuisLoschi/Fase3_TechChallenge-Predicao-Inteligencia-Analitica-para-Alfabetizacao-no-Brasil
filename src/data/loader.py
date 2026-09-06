"""Leitura das bases do projeto.

Todas as funções devolvem `pandas.DataFrame` e leem do Parquet gerado por
`scripts/prepare_data.py`. Se o Parquet não existir, a mensagem de erro diz
qual comando gera o arquivo — em vez de estourar um `FileNotFoundError` cru.

Convenção de nomes: `carregar_gold` e `carregar_aluno` são as duas bases de
grão aluno; o resto são agregados pequenos, lidos direto do CSV.
"""

from pathlib import Path

import pandas as pd

from src import config

_ERRO_PARQUET = (
    "{caminho} não existe.\n"
    "Gere os arquivos intermediários com:  python scripts/prepare_data.py"
)


def _ler_parquet(caminho: Path, colunas: list[str] | None = None) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(_ERRO_PARQUET.format(caminho=caminho))
    return pd.read_parquet(caminho, columns=colunas)


def carregar_gold(ano: int = config.ANO_ALVO, colunas: list[str] | None = None) -> pd.DataFrame:
    """Gold em grão aluno: coorte e alvo. Já filtrada em `presenca = 1` pelo ETL."""
    return _ler_parquet(Path(str(config.PARQUET_GOLD).format(ano=ano)), colunas)


def carregar_aluno(ano: int = config.ANO_LAG, colunas: list[str] | None = None) -> pd.DataFrame:
    """Microdado bruto do INEP — as três colunas que a Gold descarta.

    Traz `proficiencia`, `presenca` e `peso_aluno`. Fora isso é idêntico à Gold:
    as chaves de 2023 e 2024 batem exatamente com as linhas de `presenca = 1`, e
    `alfabetizado` não diverge em nenhuma das 3.355.846 comparações.

    Papel no projeto, corrigido pela auditoria da Etapa 2.5: é a fonte de
    `peso_aluno` (a correção oficial de não-resposta) e de `presenca` em grão de
    escola, além da base de reconciliação. **Não** é a fonte principal das
    features de lag — medido, o microdado agrega +0,0022 de ROC-AUC sobre um
    conjunto construído só com a Gold, com IC95 pareado incluindo zero.

    Use com `ano=2023` para features; 2024 aqui serve a reconciliação e testes.
    """
    return _ler_parquet(Path(str(config.PARQUET_ALUNO).format(ano=ano)), colunas)


def carregar_dim_municipio() -> pd.DataFrame:
    """`id_municipio → sigla_uf / nome_uf / nome_regiao`, extraída da Gold.

    Necessária porque o microdado do INEP não traz UF, só `id_municipio`;
    sem ela não há como montar agregados de UF para 2023.
    """
    if not config.CSV_DIM_MUNICIPIO.exists():
        raise FileNotFoundError(
            f"{config.CSV_DIM_MUNICIPIO} não existe.\n"
            "Gere com:  python scripts/build_dim_municipio.py"
        )
    return pd.read_csv(config.CSV_DIM_MUNICIPIO, dtype={"id_municipio": "int32"})


def carregar_meta_municipio() -> pd.DataFrame:
    """Metas 2024–2030 por município, mais `nivel_alfabetizacao`.

    Grão: uma linha por (`ano`, `id_municipio`) — a base só cobre a rede
    Municipal. Pode ser unida à coorte direto, sem agregar. Quem tem grão por
    rede é o *agregado* (`carregar_agregado_municipio`), não esta base; travado
    em `test_meta_municipio_e_um_por_municipio_ano`.
    """
    return pd.read_csv(config.CSV_META_MUNICIPIO, dtype={"id_municipio": "int32"})


def carregar_agregado_municipio(apenas_publica: bool = False) -> pd.DataFrame:
    """Taxa, média de português e `proporcao_aluno_nivel_0..8` por município.

    As proporções por nível são 100% nulas em 2023 e preenchidas em 2024 —
    servem à camada estratégica descritiva, nunca como feature preditiva.

    Grão: município x série x rede — sem escolher a rede antes, o join com a
    coorte duplica linhas.

    `apenas_publica=True` filtra `rede = 5`, que é a linha mais **fiel** ao
    microdado (MAE 0,99pp em 2023) e serve à reconciliação. Não é a de maior
    **cobertura**: para construir lag é preciso coalescer `5 → 3`, senão 21,2%
    da coorte de 2024 fica sem histórico. Ver `config.REDE_AGREGADO_COALESCENCIA`.
    """
    df = pd.read_csv(config.CSV_AGREGADO_MUNICIPIO, dtype={"id_municipio": "int32"})
    if apenas_publica:
        df = df[df["rede"] == config.REDE_AGREGADO_PUBLICA].reset_index(drop=True)
    return df


def carregar_agregado_uf(apenas_publica: bool = False) -> pd.DataFrame:
    """Mesmo conteúdo do agregado municipal, no grão UF x série x rede."""
    df = pd.read_csv(config.CSV_AGREGADO_UF)
    if apenas_publica:
        df = df[df["rede"] == config.REDE_AGREGADO_PUBLICA].reset_index(drop=True)
    return df


def carregar_agregados_municipais(ano: int = config.ANO_LAG) -> pd.DataFrame:
    """Tabela municipal ponderada por `peso_aluno`, insumo da camada estratégica."""
    caminho = Path(str(config.PARQUET_AGREGADO_MUNICIPAL).format(ano=ano))
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não existe.\n"
            "Gere com:  python -m src.preprocessing.build_dataset --force"
        )
    return pd.read_parquet(caminho)


def carregar_dataset_modelagem(colunas: list[str] | None = None) -> pd.DataFrame:
    """Dataset final de modelagem (alunos de 2024 + features de 2023)."""
    if not config.PARQUET_DATASET.exists():
        raise FileNotFoundError(
            f"{config.PARQUET_DATASET} não existe.\n"
            "Gere com:  python -m src.preprocessing.build_dataset"
        )
    return pd.read_parquet(config.PARQUET_DATASET, columns=colunas)
