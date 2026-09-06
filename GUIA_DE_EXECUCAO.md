# Guia de execução

Este documento é o mapa operacional do projeto: o que cada arquivo faz, em que ordem as coisas
rodam, o que cada comando lê e escreve, e por que aquele passo existe. O [README](README.md)
apresenta o problema e os resultados; aqui está como reproduzi-los. Antes de citar qualquer número
deste projeto, leia os limites em [`reports/revisao_cientifica.md`](reports/revisao_cientifica.md):
os resultados do classificador são retrospectivos e exploratórios, e os produtos municipais são
cenários condicionais.

---

## Comece aqui

Três caminhos, conforme o tempo disponível. Todos partem do mesmo lugar.

**Você tem 10 minutos e quer avaliar o projeto.** Leia o [README](README.md) até a seção de
Limitações, depois [`reports/revisao_cientifica.md`](reports/revisao_cientifica.md), que lista as
correções aplicadas e as pendências que dependem de terceiros. Em seguida abra
[`reports/aplicacao_estrategica.md`](reports/aplicacao_estrategica.md) na seção 5, que é onde estão
os dois rankings municipais. Não é preciso executar nada — todo número citado tem o artefato
correspondente versionado em [`reports/`](reports).

**Você tem uma hora e quer entender as decisões.** Leia os notebooks na ordem, de 01 a 05. Eles
narram a análise com as saídas já gravadas e não precisam ser executados. Cada um abre com uma
tabela de "pergunta → decisão que ela fecha", e cada achado é numerado. Depois vá para
[`reports/documentacao_tecnica.md`](reports/documentacao_tecnica.md), em especial a seção 9, dos
experimentos descartados.

**Você vai executar.** Siga este guia da próxima seção em diante. Reserve cerca de duas horas de
processamento, quase todo ele concentrado em dois comandos. Quem quiser só a sequência, sem a
justificativa de cada passo, use [`scripts/reproduzir.py`](scripts/reproduzir.py), descrito no fim
da próxima seção.

---

## Pré-requisitos

| Item | Versão | Observação |
|---|---|---|
| Python | 3.14.3 | as wheels de `lightgbm`, `shap` e `numba` foram verificadas nessa versão |
| Memória | 16 GB | a coorte tem 1,85 milhão de linhas; o Parquet é lido inteiro em memória |
| Núcleos | quanto mais, melhor | a comparação de modelos leva 13 minutos em 16 núcleos |

**A dependência que este repositório não resolve sozinho.** A camada Gold, de 615 MB, é a saída de
um pipeline PySpark que roda em Databricks e não está versionada. O código que a produz está em
[`scripts/etl/`](scripts/etl), mas regenerá-la exige acesso a um workspace. Para executar o projeto
é preciso obter o CSV com o grupo e colocá-lo em
`data/raw/alfabetizacao_aluno/alfabetizacao_aluno_features_2023-2024_20260829.csv`. Clonar o
repositório não traz esse arquivo, e o manifesto confere a versão sem substituir o fornecimento.

Os seis CSVs do INEP, que somam 216 MB, são públicos e vêm do
[Base dos Dados](https://basedosdados.org/), conjunto `br_inep_avaliacao_alfabetizacao`. Eles vão em
`data/raw/inep/`. A relação completa está em [`data/README.md`](data/README.md).

**Antes de rodar qualquer coisa, confira as sete fontes.** O ETL grava em `overwrite`, sem carga
incremental, então duas extrações produzem conteúdos diferentes e métricas que não se comparam.

```bash
python scripts/verificar_insumos.py
```

Ele calcula o SHA-256 de cada fonte e confronta com
[`data/manifesto_fontes.json`](data/manifesto_fontes.json), que acompanha o código. Divergência é
erro, não aviso: interrompe a execução em vez de deixar você misturar versões de dados. Só use
`--gravar-manifesto` quando estiver registrando deliberadamente uma extração nova, o que invalida
as métricas publicadas. Se os dados estiverem fora de `data/`, aponte `TC_DATA_DIR` para a cópia.

---

## Instalação

```bash
python -m venv .venv
source .venv/Scripts/activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name tc-fase3
```

O kernel `tc-fase3` importa para quem abre os notebooks no Jupyter: os cinco o fixam no
`kernelspec`, e sem ele o Jupyter cai no Python do sistema — onde `lightgbm` e `shap` não existem.
O sintoma é um `ModuleNotFoundError` no notebook 03 depois de a EDA ter rodado sem reclamar.

Para **reexecutar** os notebooks não é preciso instalar kernel nenhum:

```bash
python scripts/executar_notebooks.py notebooks/05_aplicacao_estrategica.ipynb
```

[`scripts/executar_notebooks.py`](scripts/executar_notebooks.py) monta o `kernelspec` em memória
apontando para o `sys.executable` corrente, roda com `allow_errors=False` e regrava o `.ipynb` com
as saídas. É o caminho que funciona num clone, porque não depende de um kernel registrado com
`--user` na máquina de quem escreveu o notebook.

Conferindo a instalação:

```bash
python -c "import sys, lightgbm, shap; print(sys.prefix)"   # deve apontar para .venv
pytest -q -k dados                                          # 24 testes de integridade das bases
```

---

## O pipeline, em doze comandos

A ordem importa: cada linha consome o que a anterior escreveu.

| # | Comando | O que faz | Tempo |
|---|---|---|---|
| 0 | `python scripts/verificar_insumos.py` | SHA-256 das sete fontes contra o manifesto | segundos |
| 1 | `python scripts/prepare_data.py` | CSV → Parquet, por ano | 9 s |
| 2 | `python scripts/build_dim_municipio.py` | dimensão `município → UF/região` | segundos |
| 3 | `python -m src.preprocessing.build_dataset` | coorte 2024 + atributos de 2023 e cadastro de 2024 | não cronometrado |
| 4 | `python scripts/experimento_b5.py` | ablação de atributos, só no desenvolvimento | ~35 min em 16 núcleos |
| 5 | `python -m src.modeling.train comparacao` | 7 candidatos nos mesmos folds | ~13 min em 16 núcleos |
| 6 | `python -m src.modeling.train tuning` | 40 configurações de LightGBM | **> 45 min** |
| 7 | `python -m src.modeling.train campeao` | refit, reserva, serialização | 189 s |
| 8 | `python -m src.evaluation.rigor` | as cinco provas de robustez | não cronometrado |
| 9 | `python -m src.evaluation.interpret` | SHAP, permutação, famílias | 618 s |
| 10 | `python -m src.modeling.strategic` | ranking, clusters, metas | 131 s |
| 11 | `python scripts/atualizar_relatorios.py` | reescreve os números dos documentos a partir das métricas | segundos |
| — | `python scripts/executar_notebooks.py notebooks/*.ipynb` | regrava as saídas e as 55 figuras | não cronometrado |
| — | `pytest -q` | a suíte inteira | 42 s |

Os tempos são os que os próprios artefatos registraram, em uma máquina de 16 núcleos. Onde não há
medição gravada, o campo diz isso em vez de estimar.

**O orquestrador.** [`scripts/reproduzir.py`](scripts/reproduzir.py) executa esses comandos em três
blocos, com `PYTHONHASHSEED=42` e `MPLBACKEND=Agg` fixados, e para na primeira falha:

```bash
python scripts/reproduzir.py --etapa dados        # passos 0 a 3
python scripts/reproduzir.py --etapa modelos      # passos 4 a 9, o bloco caro
python scripts/reproduzir.py --etapa relatorios   # passos 10 e 11, o padrão
python scripts/reproduzir.py --etapa tudo         # os três, na ordem
```

`--etapa relatorios` é o padrão porque é o bloco que se roda com frequência: ele supõe o
classificador e as métricas de interpretação já em disco. `--etapa modelos` é o que custa mais de
uma hora e meia.

**Atalho legítimo.** Os passos 4, 5 e 6 produzem evidência comparativa e hiperparâmetros, e juntos
consomem quase todo o tempo do pipeline. Os CSVs que eles geram já estão versionados em
`reports/metrics/`, e o passo 7 lê os hiperparâmetros do JSON, não da busca. Para reproduzir só o
modelo e os produtos finais, rode 0, 1, 2, 3, 7, 8, 9, 10, 11.

---

## O que cada passo faz, e por quê

### 0. `scripts/verificar_insumos.py` — a conferência das fontes

**Por que existe.** Todo número publicado aqui vale para uma extração específica. Sem uma
conferência na entrada, um CSV trocado produz métricas plausíveis e silenciosamente incomparáveis
com as do relatório. O script fecha essa porta antes do primeiro `read_csv`.

**Lê** as sete fontes e o manifesto. **Escreve** nada, salvo com `--gravar-manifesto`.

Aceita `TC_DATA_DIR` para apontar a uma cópia dos dados fora do repositório.

### 1. `scripts/prepare_data.py` — CSV para Parquet

**Por que existe.** A Gold tem 615 MB e o microdado 214 MB. Lidos com dtypes largos, os dois juntos
não cabem confortavelmente em memória. A conversão é feita em lotes do pyarrow, com tipos estreitos
e compressão zstd, e derruba 830 MB para 96 MB.

**Lê** `data/raw/alfabetizacao_aluno/*.csv` e `data/raw/inep/*.csv`.
**Escreve** `data/interim/gold_{2023,2024}.parquet` e `aluno_{2023,2024}.parquet`.

Aceita `--force` para reconverter arquivos já existentes.

### 2. `scripts/build_dim_municipio.py` — a dimensão territorial

**Por que existe.** O microdado do INEP traz `id_municipio` e não traz UF. A Gold é a única base do
projeto que carrega `sigla_uf` e `nome_regiao`. Sem esta dimensão não há como montar os agregados de
UF para 2023, que é o ano de todo o bloco histórico.

**Lê** `data/interim/gold_*.parquet`.
**Escreve** `data/reference/dim_municipio.csv` — 5.547 municípios em 26 UFs, Roraima ausente.

É o único arquivo de dados versionado no Git, por exceção declarada no `.gitignore`: tem 300 KB e
poupa quem clona de reprocessar 830 MB só para obter a dimensão.

### 3. `python -m src.preprocessing.build_dataset` — o dataset de modelagem

**Por que existe.** É onde a restrição temporal do projeto vira propriedade do arquivo. A coorte de
2024 recebe todo o bloco de desempenho calculado sobre 2023, então nenhuma medida contemporânea de
proficiência precisa ser lembrada e removida mais tarde.

**O que não é verdade sobre este arquivo.** Nem todos os 20 atributos são de 2023. Caderno, rede e
território vêm do cadastro da própria coorte de 2024, e a meta lida na linha `ano = 2023` tem ano de
referência conhecido, não data de publicação auditada. A tabela por atributo — ano de referência,
disponibilidade e o que ainda falta verificar — está em
[`reports/contrato_temporal.md`](reports/contrato_temporal.md), e é ela que define o momento em que
o escore poderia ser calculado.

**Lê** os quatro Parquet de `data/interim/`, os CSVs do INEP e a dimensão territorial.
**Escreve** `data/processed/dataset_2024.parquet` — 1.851.852 linhas e 41 colunas, das quais 20 são
features. Escreve também os agregados municipais ponderados em `data/processed/`.

Três colunas ficam no arquivo e fora da matriz do modelo: `id_municipio` (grupo da validação
cruzada), `peso_aluno` (peso de agregação populacional) e `alfabetizado` (alvo original). Quem monta
a matriz usa `pipeline.separar_X_y`, que remove as três e falha se alguma coluna proibida sobreviver.

**Cuidado.** O MD5 deste arquivo está gravado dentro de `models/campeao.joblib`. Rodar com `--force`
e obter um Parquet diferente faz os passos 7 a 10 recusarem execução, o que é o comportamento
desejado — mas significa retreinar. Não use `--force` sem intenção.

### 4. `scripts/experimento_b5.py` — a ablação de atributos

**Por que existe.** Para decidir se o bloco de percentis do microdado paga o próprio custo, e se o
bloco de escola acrescenta algo depois de descoberta a reciclagem de `id_escola`. Compara três
conjuntos nos mesmos folds, com hiperparâmetros fixos, para que a diferença medida seja de
atributos e não de tuning.

**A correção que este passo carrega.** A réplica histórica comparava conjuntos sobre a coorte
inteira de 2024: os municípios depois chamados de reserva participaram de uma decisão de modelagem.
Esta versão separa a reserva com a seed fixa do projeto **antes** de qualquer seleção, e as seeds do
argumento alteram apenas a validação cruzada. Isso corrige o fluxo para decisões futuras e não
restaura a independência da reserva já consultada — para isso é preciso uma amostra ainda não usada.

**Escreve** `reports/metrics/experimento_b5_desenvolvimento.csv` e o JSON irmão, que registra
escopo, seed da reserva, hash do dataset e a ressalva. A saída histórica fica preservada em
`experimento_b5_replica.csv` como registro do que foi feito.

`--n-linhas` reduz o orçamento por municípios inteiros, para verificar o fluxo sem pagar os 35
minutos da execução completa.

### 5. `python -m src.modeling.train comparacao` — os sete candidatos

**Por que existe.** Para decidir se o modelo vale a pena, é preciso saber contra o que ele compete. A
heurística "ranqueie pela taxa municipal do ano passado" foi reimplementada como estimador sklearn e
passa pelos mesmos cinco folds dos demais — sem isso, a comparação seria entre números medidos em
desenhos diferentes.

**Lê** o dataset.
**Escreve** `reports/metrics/cv_modelos.csv` (uma linha por modelo × fold) e
`comparacao_pareada.csv` com os testes de Nadeau-Bengio.

### 6. `python -m src.modeling.train tuning` — a busca de hiperparâmetros

**Por que existe.** Para medir se a busca compra alguma coisa. A resposta medida foi que não compra
de forma significante: +0,0022 com IC95 de [−0,0011; +0,0055].

A subamostra é de **municípios inteiros**, nunca de alunos. Metade de um município no treino e
metade na validação reintroduziria a memorização que o desenho inteiro existe para evitar.

**Escreve** `reports/metrics/tuning_lgbm.csv`, `tuning_bordas.csv` e
`hiperparametros_campeao.json`.

### 7. `python -m src.modeling.train campeao` — refit, reserva e serialização

**Por que existe.** É a única passagem do *treinamento* pela reserva. Limiar, hiperparâmetros,
calibração e escolha de família saíram todos do desenvolvimento.

**O que a reserva não é.** Ela não é uma avaliação confirmatória independente, porque a seleção
histórica de atributos consultou a coorte inteira, reserva incluída. O artefato carrega esse limite
consigo: `src/evaluation/protocolo.py` grava `teste_independente_da_selecao: False` e
`validacao_temporal_do_campeao: False` no JSON e no carregador do modelo, para que o número não
viaje sem a ressalva. Trocar a seed não desfaz a consulta anterior.

**Escreve** `models/campeao.joblib`, `models/escores_oof_desenvolvimento.parquet` e cinco CSVs de
métrica mais `campeao.json`.

O `joblib` carrega, além do modelo: os dois limiares, os nomes das 89 colunas que saem do
`ColumnTransformer`, a seed, o MD5 do dataset e as versões de `scikit-learn` e `lightgbm`. É o que
permite ao teste `test_campeao_em_disco_reproduz_as_metricas_publicadas` conferir o número do
relatório contra o modelo em disco a menos de 1e-9.

### 8. `python -m src.evaluation.rigor` — as cinco provas

**Por que existe.** Um ROC-AUC de 0,66 pode ser sinal fraco e real, ou variância bem apresentada.
Cada prova separa um dos casos: permutação, curva de aprendizado, `LeaveOneGroupOut` por região,
drift entre edições e invariância do controle negativo.

A variante que mais importa é a permutação **dentro de cada município**, que preserva a taxa
municipal e destrói o resto. Ela é a medição que reenquadrou o projeto.

**Escreve** cinco CSVs `reports/metrics/rigor_*.csv`.

### 9. `python -m src.evaluation.interpret` — as três leituras

**Por que existe.** Uma leitura de importância sozinha não decide nada. SHAP diz quanto cada coluna
moveu a predição, a permutação diz quanto a métrica perde sem ela, e o coeficiente linear diz em que
direção. A discordância entre elas é informação.

O agrupamento por família não é refinamento: com treze pares de features acima de |Spearman| 0,95, a
leitura coluna a coluna subestima cada família de forma sistemática.

**Lê** `models/campeao.joblib` e o dataset, conferindo o MD5 antes da primeira conta.
**Escreve** oito arquivos `reports/metrics/interpret_*` e `models/shap_amostra.parquet`.

### 10. `python -m src.modeling.strategic` — a camada municipal

**Por que existe.** Muda a unidade de decisão, de aluno para município, que é onde a política pública
acontece. Não treina modelo novo.

Aceita estágios: `oof`, `risco`, `clusters`, `metas` ou `tudo` (padrão).

**Como a projeção de metas é avaliada.** Prior, suavização, deriva, dispersão e baseline são
aprendidos em cinco folds por município e aplicados fora do fold que os gerou. É generalização
territorial dentro da transição 2023 → 2024, não validação de um ano futuro: os parâmetros e o
desfecho vêm da mesma transição. A distribuição de trabalho é censurada em 0–100, porque a versão
gaussiana anterior publicava taxas acima de 100% e limites inferiores negativos.

**Escreve** os três CSVs publicáveis em `reports/`, dez arquivos `reports/metrics/estrategia_*`,
`models/escores_oof_nacional.parquet` e o sidecar `escores_oof_nacional.json`, que guarda hash do
dataset, atributos, seed e parâmetros. O cache OOF só é reaproveitado se os quatro conferirem.

### 11. `scripts/atualizar_relatorios.py` — os números dos documentos

**Por que existe.** Para que nenhum número de relatório seja digitado a mão. Ele lê as métricas em
disco e reescreve os trechos numéricos do README, dos relatórios e do notebook estratégico, na
formatação brasileira. Não ajusta modelo nenhum; roda depois do passo 10.

Um número que muda aqui sem que ninguém tenha rodado o passo 10 é sinal de que os artefatos em
disco não são os que geraram o texto anterior.

**Cuidado com a ordem.** Este passo também é o gerador do notebook 05: ele regrava o `.ipynb` a
partir do texto e das métricas, e portanto **sem saídas**. Rodá-lo e parar aí deixa o notebook
vazio no repositório. Reexecute os notebooks depois dele, nunca antes:

```bash
python scripts/reproduzir.py --etapa relatorios
python scripts/executar_notebooks.py notebooks/05_aplicacao_estrategica.ipynb
```

---

## Mapa de saídas

### Os três produtos de decisão

Estes são os arquivos que um gestor usaria. Cada um traz as colunas de ressalva ao lado das de
resultado, de propósito.

| Arquivo | Linhas | Para que serve |
|---|---:|---|
| [`reports/ranking_risco_municipal.csv`](reports/ranking_risco_municipal.csv) | 5.517 | duas ordenações de risco: por intensidade e por volume de crianças |
| [`reports/clusters_municipais.csv`](reports/clusters_municipais.csv) | 5.461 | os três perfis municipais, para política por perfil em vez de por UF |
| [`reports/projecao_metas_municipios.csv`](reports/projecao_metas_municipios.csv) | 5.352 | gap de esforço até a meta de 2025 e o cenário condicional para ela |

Colunas de ressalva que acompanham o resultado, e o que cada uma quer dizer:

| Coluna | Onde | O que sinaliza |
|---|---|---|
| `modelo_avaliado` | ranking | `False` nos 23 municípios de AC e DF, onde o escore existe e a posição não; o restante do ranking os ignora |
| `faixa_shap_taxa_municipal` | ranking | recorte descritivo do achatamento do SHAP; **não** é um julgamento da ordenação inteira |
| `incerteza_posicao_quantificada` | ranking | é `False` em todas as linhas: a estabilidade das posições não foi medida |
| `natureza_resultado` | projeção | `cenario_condicional_sem_validacao_em_ano_futuro`, escrito por extenso na própria linha |
| `meta_dentro_do_intervalo` | projeção | a meta do município cai dentro do intervalo de 95% |
| `porte_abaixo_referencia_dispersao` | projeção | porte abaixo da referência usada para modelar dispersão; é descrição, não regra de elegibilidade |
| `taxa_presenca_2024` | ambos | a cobertura sobre a qual o número foi calculado |

A coluna `faixa_shap_taxa_municipal` mudou de nome nesta revisão. Ela se chamava `ordenacao_fragil`,
e o nome antigo afirmava mais do que a medição sustentava: o SHAP achatado de **uma** variável
abaixo de 65% não estabelece que a ordenação do modelo inteiro seja frágil naquele recorte, porque
os demais atributos continuam contribuindo. Medir o ranking dentro dessa faixa é trabalho pendente.

### Os relatórios de evidência

Um por etapa, e é neles que estão os números que o README resume.

| Arquivo | Cobre |
|---|---|
| [`reports/auditoria_camada_gold.md`](reports/auditoria_camada_gold.md) | a prova de que a Gold é fiel ao microdado |
| [`reports/feature_engineering.md`](reports/feature_engineering.md) | os 20 atributos e os dois blocos descartados |
| [`reports/contrato_temporal.md`](reports/contrato_temporal.md) | ano de referência e disponibilidade, atributo a atributo |
| [`reports/modelagem.md`](reports/modelagem.md) | os 7 candidatos, a reserva e as provas de rigor |
| [`reports/interpretabilidade.md`](reports/interpretabilidade.md) | SHAP, permutação e a leitura por família |
| [`reports/aplicacao_estrategica.md`](reports/aplicacao_estrategica.md) | as cinco perguntas de negócio |
| [`reports/documentacao_tecnica.md`](reports/documentacao_tecnica.md) | decisões e experimentos descartados |
| [`reports/revisao_cientifica.md`](reports/revisao_cientifica.md) | o que foi corrigido nesta revisão e o que ainda depende de terceiros |
| [`reports/roteiro_video.md`](reports/roteiro_video.md) | o roteiro do vídeo executivo |

Leia [`revisao_cientifica.md`](reports/revisao_cientifica.md) junto com qualquer um dos outros. Ele
é a tabela de correções e, principalmente, a lista de gates que **não** foram fechados: amostra nova
para confirmação prospectiva, dimensão socioeconômica, entrega dos dados à banca, gravação do vídeo
e revisão real por outro integrante do grupo.

### Métricas e artefatos

`reports/metrics/` guarda 38 arquivos, todos gerados por comando e nenhum editado a mão. O prefixo
diz a origem: `experimento_b5_` vem do passo 4, `cv_` e `tuning_` dos passos 5 e 6, `campeao_` do 7,
`rigor_` do 8, `interpret_` do 9 e `estrategia_` do 10. `historico/` guarda as saídas anteriores à
revisão, mantidas como registro do que foi feito e não como evidência corrente.

`models/` guarda o artefato serializado, os escores out-of-fold e o sidecar JSON de proveniência do
cache. Não é versionado.

`images/` guarda 55 figuras, em cinco subdiretórios que correspondem às etapas. São geradas pelos
notebooks a partir dos CSVs de métrica, nunca recalculadas na hora de desenhar — por isso
regenerá-las é só reexecutar os notebooks, sem refazer conta nenhuma.

---

## Estrutura do repositório, comentada

```text
├── data/                       ver data/README.md — proveniência e armadilhas de join
│   ├── raw/                    entrada imutável, 830 MB, fora do Git
│   ├── reference/              dim_municipio.csv — o único dado versionado
│   ├── interim/                Parquet por ano
│   └── processed/              dataset_2024.parquet e os agregados municipais
│
├── notebooks/                  narram a análise; toda lógica vem de src/
│
├── scripts/
│   ├── etl/                    o pipeline PySpark da Fase 2 que produz a Gold
│   ├── verificar_insumos.py    passo 0 — SHA-256 das sete fontes
│   ├── prepare_data.py         passo 1
│   ├── build_dim_municipio.py  passo 2
│   ├── experimento_b5.py       passo 4 — ablação de atributos, só no desenvolvimento
│   ├── atualizar_relatorios.py passo 11 — reescreve os números dos documentos
│   ├── reproduzir.py           o orquestrador dos três blocos
│   └── executar_notebooks.py   reexecuta notebooks sem kernel registrado
│
├── src/
│   ├── config.py               caminhos, RANDOM_STATE e COLS_PROIBIDAS
│   ├── data/loader.py          leitura das bases, com as regras de rede por propósito
│   ├── eda.py                  medições e gráficos da etapa exploratória
│   ├── preprocessing/
│   │   ├── feature_store.py    constrói os agregados históricos de 2023
│   │   ├── build_dataset.py    passo 3
│   │   ├── pipeline.py         ColumnTransformer e o contrato de colunas
│   │   └── diagnostico.py      julga a feature que o feature_store construiu
│   ├── modeling/
│   │   ├── split.py            GroupShuffleSplit e StratifiedGroupKFold por município
│   │   ├── baselines.py        a heurística de uma variável, como estimador sklearn
│   │   ├── train.py            passos 5, 6 e 7
│   │   ├── tuning.py           o espaço de busca
│   │   ├── campeao.py          refit, teste e serialização
│   │   ├── calibracao.py       a isotônica testada e rejeitada
│   │   ├── metas.py            prior, deriva e dispersão das metas, aprendidos fora do fold
│   │   └── strategic.py        passo 10
│   ├── evaluation/
│   │   ├── metrics.py          métricas, bootstrap de municípios e limiares
│   │   ├── comparacao.py       Nadeau-Bengio: se a diferença é diferença ou ruído
│   │   ├── protocolo.py        os limites da evidência, colados ao artefato
│   │   ├── rigor.py            passo 8
│   │   └── interpret.py        passo 9
│   └── visualization/          gráficos por etapa
│
├── tests/                      94 testes; ver a seção abaixo
├── images/                     55 figuras
├── reports/                    relatórios, produtos e métricas
│
├── README.md                   o problema, os resultados e as limitações
├── GUIA_DE_EXECUCAO.md         este arquivo
├── PLANO_EXECUCAO.md           o registro de decisão etapa a etapa
└── requirements.txt
```

**Uma convenção que vale conhecer antes de ler o código.** Em todo o projeto, `medir_*` calcula e
devolve número, `plotar_*` recebe número já medido e devolve uma figura, e `montar_*` constrói
tabela. Nenhuma função de gráfico faz conta. É o que permite os testes cobrirem a medição sem
precisar renderizar nada.

**Os notebooks são finos de propósito.** Eles importam de `src/` e narram; não implementam. Uma
análise que só existe dentro de uma célula não pode ser testada nem reexecutada fora do Jupyter.

---

## Os notebooks

Leia na ordem. Cada um abre com a tabela de perguntas que ele fecha e numera os achados.

| Notebook | Fase CRISP-DM | O que decide |
|---|---|---|
| [`01_eda.ipynb`](notebooks/01_eda.ipynb) | 2, *Data Understanding* | as cinco armadilhas da base e as seis hipóteses analíticas |
| [`02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) | 3, *Data Preparation* | os 20 atributos, e os dois blocos que ficaram de fora |
| [`03_modelagem.ipynb`](notebooks/03_modelagem.ipynb) | 4, *Modeling* | o campeão, os limiares e as provas de rigor |
| [`04_interpretabilidade.ipynb`](notebooks/04_interpretabilidade.ipynb) | 5, *Evaluation* | a importância por família e o que passa do piso de ruído |
| [`05_aplicacao_estrategica.ipynb`](notebooks/05_aplicacao_estrategica.ipynb) | 5 → 6, *Deployment* | as cinco perguntas de negócio |

Todos os cinco estão executados nesta revisão, sem erro e com as saídas gravadas. Para reexecutar,
rode na ordem — o notebook 03 em diante lê artefatos que os passos 5 a 10 precisam ter escrito
antes:

```bash
python scripts/executar_notebooks.py notebooks/01_eda.ipynb notebooks/02_feature_engineering.ipynb     notebooks/03_modelagem.ipynb notebooks/04_interpretabilidade.ipynb     notebooks/05_aplicacao_estrategica.ipynb
```

O notebook 01 é o mais demorado, porque é o único que lê as bases inteiras em vez de artefatos.
Abrir no Jupyter continua funcionando; aí sim é preciso o kernel `tc-fase3`.

---

## Os testes

```bash
pytest -q                       # 94 testes, cerca de 42 s
pytest -q tests/test_dados.py   # só a integridade das bases
```

Eles não testam se o código roda. Testam se as decisões continuam válidas.

| Arquivo | Casos | O que trava |
|---|---:|---|
| `test_dados.py` | 24 | contagens de referência, códigos de rede, e que `id_aluno` não é chave longitudinal |
| `test_features.py` | 17 | interseção vazia com `COLS_PROIBIDAS`, cobertura do lag, VIF do subconjunto podado |
| `test_modelagem.py` | 10 | que o artefato em disco reproduz o ROC-AUC publicado, conferido contra o hash do dataset real |
| `test_interpretabilidade.py` | 12 | que o SHAP soma exatamente a predição, e o mapa de 89 colunas para 5 famílias |
| `test_estrategia.py` | 22 | que o escore é out-of-fold, e que o CSV de metas não contém rótulo binário |
| `test_revisao.py` | 9 | as correções desta revisão: reserva separada antes da seleção, projeção dentro de 0–100, e o artefato carregando o limite da validação |

`test_revisao.py` existe para que as correções não sejam desfeitas em silêncio. Ele trava, entre
outras coisas, que o desfecho de um fold não influencia as próprias predições, que a projeção
recusa entradas inválidas e permanece coerente nas bordas do domínio, e que o carregador do modelo
devolve `teste_independente_da_selecao: False` em vez de omitir a ressalva.

Um teste que falha aqui quase nunca é um bug de código. É um insumo que mudou, ou uma decisão que
alguém reabriu sem perceber.

O único aviso esperado na suíte vem do SHAP, sobre a mudança de formato do `TreeExplainer` para
classificadores binários do LightGBM. Não é falha, e o teste que o dispara confere a soma exata.

---

## Problemas comuns

**`ModuleNotFoundError: lightgbm` dentro do notebook, mas não no terminal.** O Jupyter está num
kernel que não é o do venv. Rode `python -m ipykernel install --user --name tc-fase3` e selecione
`tc-fase3` no notebook.

**O passo 7 ou 9 recusa executar, reclamando de hash.** O `dataset_2024.parquet` em disco não é o
mesmo que treinou o modelo. Ou o dataset foi reconstruído com um insumo diferente, ou o `joblib` é de
outra execução. Refaça a partir do passo 3, na ordem.

**Os testes de contagem falham logo depois de clonar.** Alguma base de `data/raw/` está faltando ou
é de outra extração. O nome do CSV da Gold carrega o timestamp da extração justamente por isso: o
ETL roda em `overwrite` sem carga incremental, então duas execuções produzem conteúdos diferentes.
Rode `python scripts/verificar_insumos.py` antes de investigar qualquer outra coisa: ele responde em
segundos qual das sete fontes divergiu. As contagens esperadas estão em
[`data/README.md`](data/README.md).

**`Fontes diferentes do manifesto`.** O passo 0 encontrou as sete fontes, mas pelo menos uma tem
SHA-256 diferente do registrado. Não contorne com `--gravar-manifesto` para "fazer passar": isso
apenas registra a extração nova e desatualiza todas as métricas publicadas de uma vez. Obtenha a
extração correspondente ao código com o grupo.

**O notebook falha no meio, ou o kernel não sobe.** Use
`python scripts/executar_notebooks.py <caminho>` em vez de rodar pelo Jupyter. Ele usa o Python
corrente, sem depender de um kernel registrado com `--user`, e para na primeira célula com erro em
vez de gravar um notebook meio executado.

**O notebook rodou sem erro e saiu sem nenhuma figura.** É `MPLBACKEND=Agg` vazando para o kernel.
As células chamam `salvar_figura` e imprimem o caminho, então quem exibe a figura no notebook é a
exibição automática do backend inline — com `Agg` ela não acontece, o PNG em `images/` sai igual e
o `.ipynb` fica mudo. `executar_notebooks.py` fixa o backend inline justamente por isso; não
sobrescreva `MPLBACKEND` ao chamá-lo.

**`FileNotFoundError` em `dim_municipio.csv`.** O passo 2 não rodou. Ele é rápido e pode ser
executado isoladamente.

**A busca de hiperparâmetros parece travada.** São mais de 45 minutos, sem barra de progresso por
configuração. Ela grava o CSV só ao final. Se o tempo for um problema, pule os passos 4 e 5 pelo
atalho descrito acima.
