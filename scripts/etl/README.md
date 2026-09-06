# Pipeline Alfabetização INEP — Databricks (Arquitetura Medalhão)

ETL em 3 camadas (Bronze → Silver → Gold) rodando em PySpark no Databricks, com Unity Catalog. Cada camada é uma tabela Delta gerenciada em schema próprio; só os CSVs brutos ficam soltos, num Volume.

## 1. Pré-requisitos

- Workspace Databricks com **Unity Catalog** habilitado.
- Um catálogo já existente (ex.: `workspace`).
- Permissão para criar schema, volume e tabela nesse catálogo.
- Compute: os scripts foram testados em **serverless** (Free Edition). Funcionam também em cluster clássico.

## 2. Estrutura de dados

```
<catalog>                          (ex.: workspace)
├── <landing_schema>.<volume>      (ex.: raw.tech_challenge_fase_3_datalake) → Volume, CSVs brutos
├── <bronze_schema>                (ex.: bronze) → tabelas Delta, 1:1 com o CSV de origem
├── <silver_schema>                (ex.: silver) → tabelas Delta tipadas/tratadas
└── <gold_schema>                  (ex.: gold)   → tabelas Delta analíticas finais
```

### 2.1. Criar o schema e o volume de landing

No SQL Editor do Databricks (ajuste os nomes se for usar outros):

```sql
CREATE CATALOG IF NOT EXISTS workspace;
CREATE SCHEMA IF NOT EXISTS workspace.raw;
CREATE VOLUME IF NOT EXISTS workspace.raw.tech_challenge_fase_3_datalake;
```

Os schemas `bronze`, `silver` e `gold` **não precisam ser criados manualmente** — cada script cria o seu (`CREATE SCHEMA IF NOT EXISTS`) na primeira execução.

### 2.2. Subir os arquivos CSV

Copiar os arquivos abaixo direto para a **raiz** do volume (`/Volumes/workspace/raw/tech_challenge_fase_3_datalake/`, sem subpastas) — pela UI do Catalog Explorer (Upload) ou `databricks fs cp`:

| Arquivo | Conteúdo |
|---|---|
| `br_inep_avaliacao_alfabetizacao_uf.csv` | Avaliação de alfabetização por UF |
| `br_inep_avaliacao_alfabetizacao_municipio.csv` | Avaliação de alfabetização por município |
| `br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_brasil.csv` | Metas nacionais |
| `br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_uf.csv` | Metas estaduais |
| `br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_municipio.csv` | Metas municipais |
| `br_inep_avaliacao_alfabetizacao_aluno.csv` | Microdado de aluno (grão fino, ~3,9M linhas, 2023+2024 já combinados) |
| `br_bd_diretorios_brasil_uf.csv` | Dimensão de apoio: UF (base dos dados) |
| `br_bd_diretorios_brasil_municipio.csv` | Dimensão de apoio: município (base dos dados) |

Todos comma-delimited, já com colunas em snake_case — nenhum precisa de tratamento antes do upload.

## 3. Scripts

| Ordem | Script | Lê de | Grava em |
|---|---|---|---|
| 1 | `etl-bronze.py` | Volume (CSV bruto) | `<catalog>.<bronze_schema>.*` |
| 2 | `etl-silver.py` | `<catalog>.<bronze_schema>.*` | `<catalog>.<silver_schema>.*` |
| 3 | `etl-gold.py` | `<catalog>.<silver_schema>.*` | `<catalog>.<gold_schema>.*` |

Rodar **nessa ordem**, como notebook Databricks (colar o conteúdo do `.py` numa célula) ou como Job Python Script task. Os 3 scripts leem `dbutils`/`spark` do ambiente Databricks — não rodam localmente fora dele.

### 3.1. Parâmetros (widgets)

Cada script expõe widgets com valor default — ajustar na UI do notebook/job se os nomes de catálogo/schema forem diferentes.

**etl-bronze.py**
| Widget | Default |
|---|---|
| `catalog` | `workspace` |
| `landing_schema` | `raw` |
| `volume` | `tech_challenge_fase_3_datalake` |
| `bronze_schema` | `bronze` |

**etl-silver.py**
| Widget | Default |
|---|---|
| `catalog` | `workspace` |
| `bronze_schema` | `bronze` |
| `silver_schema` | `silver` |
| `dq_max_null_pct` | `50` — % máx. de nulo em coluna crítica antes de **interromper** o job |
| `dq_max_orphan_pct` | `20` — % máx. de linha sem correspondência na dimensão (UF/município) antes de **interromper** o job |

**etl-gold.py**
| Widget | Default |
|---|---|
| `catalog` | `workspace` |
| `silver_schema` | `silver` |
| `gold_schema` | `gold` |

### 3.2. O que cada camada faz

- **Bronze**: ingestão crua. Lê o CSV, padroniza nome de coluna (snake_case ASCII), adiciona colunas de auditoria (`_bronze_*`). Sem tipagem, sem dedup. 8 tabelas: as 6 de avaliação/meta/aluno + `dim_uf` e `dim_municipio` (dimensões de apoio, também passam pela Bronze).
- **Silver**: tipagem (`ano`→int, taxas→double), decodifica domínio de `rede`/`serie`, deduplica pela chave de negócio, enriquece com as dimensões de UF/município (join), roda checagem de qualidade (nulo em coluna crítica, integridade referencial — **bloqueiam o job** se passarem do limite configurado).
- **Gold**: 2 tabelas finais:
  - `alfabetizacao_analise_output` — grão ano×município×série×rede, agregada, com metas comparadas e flags `meta_atingida_*`. Uso: dashboard/BI.
  - `alfabetizacao_aluno_features` — grão aluno (só quem fez a prova), com `alfabetizado` como target. Uso: dataset de treino/teste do modelo supervisionado.

## 4. Rodando do zero

1. Criar catálogo/schema/volume (seção 2.1).
2. Subir os 8 CSVs na raiz do volume (seção 2.2).
3. Rodar `etl-bronze.py`.
4. Rodar `etl-silver.py`.
5. Rodar `etl-gold.py`.
6. Conferir: `SELECT * FROM workspace.gold.alfabetizacao_aluno_features LIMIT 10;`

## 5. Notas / limitações conhecidas

- Compute **serverless** não suporta `spark.sparkContext`, `spark._jvm`/`_jsc`, `df.cache()`/`PERSIST`, nem alterar `spark.sql.sources.partitionOverwriteMode` — os scripts já evitam tudo isso (usam `os.path.exists` pra checar arquivo no Volume, `spark.catalog.tableExists` pra checar tabela, sem cache).
- `input_file_name()` não é suportado no Unity Catalog — os scripts usam `_metadata.file_path`.
- Sem carga incremental: toda execução relê a fonte inteira e sobrescreve a tabela de destino (`mode("overwrite")`). Não é seguro rodar só parte do pipeline esperando merge incremental.
- Dados socioeconômicos e populacionais/regionais (citados no enunciado do desafio) **não existem em nenhuma fonte atual** — `alfabetizacao_aluno_features` não os traz; precisaria de fonte externa (ex.: IBGE) pra completar.
