# Dados do projeto

Todo o dado do projeto vive sob `data/`, em quatro camadas. Nada é versionado, com
uma exceção: `data/reference/dim_municipio.csv` (300 KB), pequeno e derivado, para
que um clone limpo tenha a dimensão territorial sem reprocessar os 830 MB.

## Camadas

```
data/
├── raw/                          # entrada imutável — nunca editada a mão
│   ├── inep/                     # 6 CSVs públicos do INEP (216 MB)
│   └── alfabetizacao_aluno/      # a Gold vinda do Databricks (615 MB)
├── reference/                    # dimensões de apoio, pequenas   ← VERSIONADO
├── interim/                      # Parquet por ano, derivado de raw/
├── processed/                    # dataset final de modelagem
└── README.md                                                     ← VERSIONADO
```

| Diretório | Conteúdo | Versionado |
|---|---|---|
| `raw/inep/` | 6 CSVs públicos do INEP, 216 MB | não |
| `raw/alfabetizacao_aluno/` | Gold em grão aluno, 615 MB | não |
| `reference/` | `dim_municipio.csv`, derivado da Gold | **sim** |
| `interim/` | Parquet por ano, gerado de `raw/` | não |
| `processed/` | dataset final de modelagem | não |

**Por que `reference/` e não `external/`.** A convenção Cookiecutter reserva `external/`
para *data from third party sources*. O único arquivo que morava lá é derivado
**internamente**, a partir da própria Gold — o oposto do que o nome diz. `reference/`
descreve o que a coisa é: dimensão de apoio, pequena, versionada, regenerável.

## Como reconstruir, do zero

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1   # bash: source .venv/Scripts/activate
pip install -r requirements.txt

python scripts/prepare_data.py          # CSV → Parquet em data/interim/ (~10 s)
python scripts/build_dim_municipio.py   # dimensão territorial em data/reference/
pytest -q                               # 24 testes de integridade dos dados
```

## Origem de cada base

**`raw/inep/*.csv`** — Avaliação de Alfabetização (ALFA) do INEP, anos 2023 e 2024,
distribuída pelo [Base dos Dados](https://basedosdados.org/) sob o conjunto
`br_inep_avaliacao_alfabetizacao`. São seis arquivos: microdado de aluno, agregados de
município e UF, e metas de município, UF e Brasil.

**`raw/alfabetizacao_aluno/alfabetizacao_aluno_features_2023-2024_20260829.csv`** — não é
base pública: é a saída de um pipeline medalhão (Bronze → Silver → Gold) em
PySpark/Databricks, cujo código está em [`scripts/etl/`](../scripts/etl). A tabela é
produzida por `build_ml_feature_table()` em `scripts/etl/etl-gold.py` e exportada para CSV.

> **Sobre o nome do arquivo.** Ele carrega três coisas que o nome anterior
> (`camada-gold-databricks.csv`) não carregava: a **tabela de origem**
> (`alfabetizacao_aluno_features`, que é a `GOLD_ML_TABLE` do ETL, e não a outra tabela
> Gold, `alfabetizacao_analise_output`), o **período** e o **timestamp da extração** —
> `20260829`, que é o `_gold_processing_timestamp` gravado dentro do próprio CSV. O ETL
> roda em `mode("overwrite")` sem carga incremental, então duas execuções produzem
> conteúdos diferentes; sem o timestamp no nome não há como saber qual está em disco.

> **Pendente de decisão do grupo:** um clone limpo deste repositório não consegue
> reproduzir a Gold sem acesso a um workspace Databricks. As duas saídas possíveis são
> publicar o CSV em um GitHub Release ou Drive e linkar aqui, ou declarar explicitamente
> no README que a execução completa depende da Gold fornecida pelo grupo.

## Contagens de referência

Travadas em `tests/test_dados.py` — se mudarem, algum insumo foi trocado.

| Base | 2023 | 2024 |
|---|---|---|
| Gold (grão aluno, `presenca = 1`) | 1.503.058 | 1.852.788 |
| Microdado (todos, inclusive ausentes) | 1.747.439 | 2.120.560 |
| Municípios na Gold | 4.871 | 5.517 |
| UFs na Gold | 23 | 26 |
| Municípios na dimensão | — | 5.547 em 26 UFs (Roraima ausente) |

**A Gold é o microdado filtrado, e isso é provado, não suposto.** Nos dois anos o conjunto
de `id_aluno` da Gold é idêntico ao dos presentes no microdado, sem duplicata, e
`alfabetizado` não diverge em nenhuma das 3.355.846 comparações. A Gold perde exatamente
512.153 linhas (as de `presenca = 0`) e três colunas: `presenca`, `proficiencia` e
`peso_aluno`.

## Quatro armadilhas de join

1. **Códigos de `rede` divergem entre as bases.** Gold e microdado: `2` Estadual,
   `3` Municipal, `4` Privada. Agregados do INEP: `2`, `3`, `5` Pública, `0` Total.

2. **A escolha da rede no agregado depende do propósito.** Para *reconciliar*, use
   `rede = 5`, que é a mais fiel (MAE 0,99pp em 2023 e 0,37pp em 2024) — é o que
   `loader.carregar_agregado_municipio(apenas_publica=True)` faz. Para *construir lag*,
   é preciso coalescer `5 → 3`: dos 676 municípios que existem em 2024 sem microdado de
   2023, `rede = 5` cobre 79 e `rede = 3` cobre 625, incluindo São Paulo inteiro. Só
   rede 5 resgata 12,33% da coorte; a coalescência resgata 21,20%. Ver
   `config.REDE_AGREGADO_COALESCENCIA`.

3. **O microdado não tem UF**, só `id_municipio`. Qualquer agregado de UF para 2023
   depende de `reference/dim_municipio.csv`, gerado por `scripts/build_dim_municipio.py`.

4. **`peso_aluno` não é descartável.** É o fator oficial de correção de não-resposta do
   INEP: a taxa que o INEP publica é a média de `alfabetizado` ponderada por ele. Com o
   peso a reconciliação municipal cai de 0,99pp para 0,05pp. Todo número populacional do
   projeto é ponderado; como preditor a coluna continua proibida. Ver `config.COL_PESO`.

## Limitações de cobertura a declarar

- **Roraima está ausente** de toda a base: 26 UFs, não 27.
- **SP, DF e AC só existem em 2024** no microdado e na Gold. São 99,9% dos 23,11% de
  alunos de 2024 sem histórico municipal — o gap não é ruído difuso, são três UFs
  inteiras. A coalescência do item 2 reduz o gap residual a 1,91%.
- **79 municípios** têm taxa publicada pelo INEP em 2023 e zero alunos no microdado.
- **O município 5219308** tem 410 alunos em 2023, todos ausentes, e por isso fica fora da
  Gold e da `dim_municipio` (5.547 municípios, contra 5.548 no microdado).
- **1.185 alunos** (249 em 2023, 936 em 2024) têm `presenca = 1` e proficiência nula, e
  receberam `alfabetizado = 0` do ETL. O alvo deles não é desfecho, é registro faltante;
  saem do treino.
