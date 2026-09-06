# Engenharia de features — evidência da Etapa 3

> **Revisão de validade (2026-09-05):** estas medições do classificador são históricas e exploratórias. A seleção supervisionada consultou a coorte inteira; a reserva não é independente dessa seleção. O código corrigido e o protocolo atual estão em [revisao_cientifica.md](revisao_cientifica.md).


O `PLANO_EXECUCAO.md` guarda as decisões; este documento guarda a medição que as
sustenta, no mesmo formato da auditoria da Etapa 2.5. Tudo aqui foi medido sobre
as bases do projeto nesta etapa — nada foi herdado sem reconferir.

Fase CRISP-DM 3, *Data Preparation*. Entregáveis: dataset versionado, catálogo de
features e código de preparação reproduzível.

**Resumo em quatro linhas.** O dataset tem 1.851.852 alunos de 2024 e 41 colunas,
das quais **20 entram no modelo**. Dois blocos previstos pelo plano foram
construídos, medidos e deixados de fora: o de escola, porque `id_escola` é
reciclado entre edições e o join entrega a 76% dos alunos o histórico de outra
escola; e o de percentis de proficiência do microdado, porque a réplica do
experimento B5 mediu +0,0004 de ROC-AUC com intervalo cruzando o zero. O que
sobra é territorial, coberto em 98,09% da coorte e rastreado por uma coluna de
proveniência.

---

## 1. O achado que muda o desenho: `id_escola` não é chave longitudinal

O plano previa nove features de histórico de escola. Elas não existem, porque a
chave que as construiria não aponta para a mesma escola nos dois anos.

O teste não depende de modelo: **escola não muda de município**.

```
id_escola em 2023 ......................... 36.525
id_escola em 2024 ......................... 42.328
presentes nos dois anos ................... 36.051
   …e no mesmo município ..................  2,3966%   (864 escolas)
   …e na mesma UF ......................... 80,6302%
alunos de 2024 que receberiam o agregado
de OUTRA escola ........................... 1.414.592  (76,35% da coorte)
```

Dentro de cada ano o identificador é consistente: zero escolas em mais de um
município em 2023 e em 2024. Entre anos, ele desliza — o padrão é visível a olho:

| id_escola | município em 2023 | município em 2024 |
|---|---|---|
| 60000046 | 1100700 | 1100338 |
| 60000048 | 1100809 | 1100700 |
| 60000049 | 1101104 | 1100809 |
| 60000078 | 1100023 | 1100130 |
| 60000079 | 1100023 | 1100130 |

É o mesmo mecanismo de `id_aluno` (armadilha A2): identificador sequencial
atribuído em ordem territorial a cada edição, e não código de censo. Com 42.328
escolas avaliadas em 2024 contra 36.525 em 2023, cada inserção desloca tudo o que
vem depois dela.

### Controle de embaralhamento

Se o join carregasse informação de escola, o valor real bateria um valor
sorteado. O sorteio é feito **dentro da UF**, para preservar o sinal territorial
e isolar o que sobra além dele — três seeds, AUC univariada de alfabetização
sobre os 1.851.852 alunos:

| Fonte da taxa escolar de 2023 | ROC-AUC |
|---|---|
| join real por `id_escola` | 0,5581 |
| **valor sorteado dentro da mesma UF** | **0,5601** ± 0,0006 |
| valor sorteado em qualquer UF | 0,4998 |
| taxa do município (referência) | 0,6362 |

O sorteio dentro da UF prediz **igual ou melhor** que o join real — 0,5601 contra
0,5581, com desvio de 0,0006 entre as três seeds —, e o sorteio global cai para o
acaso exato. Ou seja: o que a taxa "da escola" carrega é inteiramente recuperável
sorteando uma escola qualquer do mesmo estado. É sinal de UF com outro nome.

Nas 864 escolas cujo identificador cai no mesmo município nos dois anos — as
únicas que *poderiam* ser as mesmas, e ainda assim podem ser coincidência de
ordenação —, a taxa escolar dá AUC 0,5780 contra 0,6208 da taxa municipal na
mesma população. Mesmo no melhor caso possível a escola perde para o município.
(Esta última medição foi feita em rascunho, com o par restrito por município; o
controle de embaralhamento acima é o que está no repositório, em
`diagnostico.medir_controle_embaralhamento`, e é o que os testes travam.)

E, no multivariado, acrescentar o bloco inteiro a um LightGBM custa **−0,00038**
de ROC-AUC em 15 folds (seção 8).

**Decisão.** O bloco `esc_*` sai de `FEATURES_MODELO` e vira controle negativo
declarado, ao lado do `caderno`, disponível em `pipeline.FEATURES_ESCOLA`. Se
receber importância relevante no SHAP da Etapa 5, isso é detector de sobreajuste,
não achado sobre escolas.

**O que isso corrige na EDA.** Os itens 8, 12 e 16 do notebook 01 mediram
persistência escolar (0,258), AUC por porte da escola (0,554 a 0,560) e
persistência do desvio escolar (0,009) usando este join. Os números seguem
corretos como descrição; a *interpretação* muda. A EDA concluiu "o sinal é
territorial, não escolar", e a conclusão se sustenta — por um motivo mais forte
do que ela sabia: não é que a escola tenha pouco sinal, é que a escola nunca foi
medida. A função `eda.medir_estabilidade_id_escola` já existia no módulo, com a
suspeita escrita na docstring, sem nunca ter sido executada no notebook. Esta
etapa a executou.

**O que não muda.** Nada do que a EDA concluiu sobre o grão municipal, que usa
`id_municipio`, código IBGE estável e verificado nos dois anos.

---

## 2. Cobertura: a coalescência `5 → 3` e a proveniência

Medido sobre a coorte de 2024, 1.851.852 alunos:

| Fonte do valor de `mun_taxa_alfab_lag1` | Fração da coorte |
|---|---|
| Gold 2023 (microdado do próprio projeto) | 76,88% |
| agregado do INEP, `rede = 5` | 12,34% |
| agregado do INEP, `rede = 3` | 8,86% |
| sem histórico | 1,91% |

Reproduz exatamente o que a auditoria previu (B4): a coalescência resgata 21,20%
da coorte e deixa 1,91% de gap residual. As features que exigem grão aluno seguem
em 76,88%, porque o agregado publicado não tem distribuição.

Cobertura por feature, medida no dataset final:

| Feature | Cobertura |
|---|---|
| `mun_taxa_alfab_lag1`, `mun_media_portugues_lag1` | 98,09% |
| `uf_taxa_alfab_lag1`, `uf_media_portugues_lag1` | 98,25% |
| `mun_meta_2024_lag1`, `mun_nivel_alfabetizacao_lag1` | 95,33% |
| bloco distribucional e de porte (`mun_prof_*`, presença, contagens) | 76,88% |
| `esc_*` | 78,97% |

**O agregado de UF também precisou coalescer.** A Gold de 2023 tem 23 UFs; o
agregado do INEP de 2023 tem 24, e a que ele acrescenta é São Paulo — 21,35% da
coorte. Sobram **AC e DF**, que não existem em nenhuma fonte de 2023 e ficam com
lag de UF nulo. É limitação da base, não do desenho.

**A coalescência se paga.** Na coorte inteira, com o nulo imputado pela mediana
como o pipeline faz, a AUC univariada de alfabetização vai de 0,6276 (só Gold) a
**0,6346** (coalescida). Mais importante que o delta: sem ela, São Paulo inteiro
entraria no modelo com a mediana nacional no lugar da própria taxa.

**E é por isso que `fonte_lag_municipal` é obrigatória.** A mesma coluna prediz
de forma muito diferente conforme a apuração de origem:

| Fonte | Alunos | AUC (risco) | Risco médio |
|---|---|---|---|
| `gold` | 1.423.733 | 0,3466 | 0,3982 |
| `agregado_r5` | 228.510 | 0,4526 | 0,4213 |
| `agregado_r3` | 164.156 | 0,4199 | 0,4060 |

Métrica agregada sobre isso, sem estratificar, mistura três populações com
fidelidades diferentes.

---

## 3. Ponderação por `peso_aluno` (B2)

A regra do projeto é que todo número populacional é ponderado. Ela vale aqui em
dois lugares, por dois motivos distintos:

**Nos agregados municipais e de UF que viram feature**, porque a coalescência
mistura na mesma coluna a taxa calculada por nós e a taxa publicada pelo INEP.
Sem o peso, a primeira erra 0,97pp em relação à segunda, com máximo de 56,9pp num
único município; com o peso, 0,05pp. É homogeneidade de escala e não ganho
preditivo — medida na mesma população, a taxa ponderada dá AUC 0,3466 e a simples
0,3464.

**Na tabela populacional da camada estratégica**, `agregados_municipais_ponderados`,
gravada em `data/processed/agregados_municipais_{ano}.parquet`, porque é o número
que o projeto publica e que precisa reconciliar com o INEP. Travado em teste a
menos de 0,10pp, com a versão não ponderada gravada ao lado para que a diferença
entre as duas fique visível — ela é a medida do viés de não-resposta que a
ponderação padroniza mas não elimina.

O grão aluno do classificador continua **sem** peso, como manda B2: ali ele seria
derivado da presença observada na própria coorte.

---

## 4. Suavização da escola: `k = 30`, fixo

`(taxa_esc·n + taxa_mun·k) / (n + k)`, com `k = 30` declarado em
`feature_store.K_SUAVIZACAO` e sem calibração por validação cruzada.

A EDA já havia medido a grade inteira: AUC 0,561 em `k = 0`, 0,614 em `k = 30`,
0,653 em `k = 500`, sempre abaixo da taxa municipal pura de 0,654. A curva é
monotônica, o melhor `k` é sempre o maior da grade, e um `GridSearchCV`
devolveria exatamente isso com aparência de otimização.

O achado da seção 1 explica *por quê*: a suavização mistura uma taxa municipal
correta com uma taxa escolar que pertence a outra escola. Quanto maior o `k`,
menos ruído sobra. No dataset final a feature entrega AUC 0,6045 — abaixo dos
0,6362 da taxa municipal que a compõe.

Fica no código como conveniência de imputação, na qual escola sem histórico herda
o contexto do município, e fora de `FEATURES_MODELO` junto com o resto do bloco.

---

## 5. Dicionário de features

AUC univariada do **risco** de não alfabetização: abaixo de 0,5 a feature protege,
acima ela agrava. `AUC alfab.` é o complemento, para comparação direta com os
números da EDA. Medida sobre os 1.851.852 alunos do dataset.

Em itálico, os dois blocos que **não** entram em `FEATURES_MODELO`.

| Feature | Bloco | Cobertura | AUC risco | AUC alfab. |
|---|---|---|---|---|
| *`mun_prof_p75_lag1`* | *microdado* | *76,88%* | *0,3421* | *0,6579* |
| *`mun_prof_p50_lag1`* | *microdado* | *76,88%* | *0,3463* | *0,6537* |
| *`mun_prof_p25_lag1`* | *microdado* | *76,88%* | *0,3516* | *0,6484* |
| `mun_media_portugues_lag1` | município | 98,09% | 0,3589 | 0,6411 |
| `mun_taxa_alfab_lag1` | município | 98,09% | 0,3638 | 0,6362 |
| `mun_meta_2024_lag1` | município | 95,33% | 0,3699 | 0,6301 |
| `mun_nivel_alfabetizacao_lag1` | município | 95,33% | 0,3752 | 0,6248 |
| `uf_media_portugues_lag1` | UF | 98,25% | 0,3825 | 0,6175 |
| `uf_taxa_alfab_lag1` | UF | 98,25% | 0,3951 | 0,6049 |
| *`esc_taxa_alfab_lag1_suav`* | *escola* | *98,09%* | *0,3955* | *0,6045* |
| `mun_taxa_presenca_lag1` | município | 76,88% | 0,4081 | 0,5919 |
| `mun_desvio_vs_uf` | município | 98,09% | 0,4179 | 0,5821 |
| `uf_taxa_presenca_lag1` | UF | 76,90% | 0,4199 | 0,5801 |
| *`esc_prof_p75_lag1`* | *escola* | *78,97%* | *0,4276* | *0,5724* |
| *`esc_prof_p50_lag1`* | *escola* | *78,97%* | *0,4362* | *0,5638* |
| *`esc_taxa_alfab_lag1`* | *escola* | *78,97%* | *0,4419* | *0,5581* |
| *`esc_prof_p25_lag1`* | *escola* | *78,97%* | *0,4453* | *0,5547* |
| *`mun_prof_iqr_lag1`* | *microdado* | *76,88%* | *0,5501* | *0,4499* |
| `mun_n_escolas_lag1` | município | 76,88% | 0,5322 | 0,4678 |
| *`esc_taxa_presenca_lag1`* | *escola* | *78,97%* | *0,4718* | *0,5282* |
| `mun_n_alunos_lag1` | município | 76,88% | 0,5235 | 0,4765 |
| `mun_share_rede_estadual_lag1` | município | 76,88% | 0,5158 | 0,4842 |
| *`mun_prof_sd_lag1`* | *microdado* | *76,88%* | *0,5156* | *0,4844* |
| *`esc_n_alunos_lag1`* | *escola* | *78,97%* | *0,4879* | *0,5121* |
| *`esc_prof_sd_lag1`* | *escola* | *78,95%* | *0,4910* | *0,5090* |
| `tem_historico_municipio` | cobertura do lag | 100% | 0,4936 | 0,5064 |
| `tem_historico_escola` | cobertura do lag | 100% | 0,4953 | 0,5047 |
| *`esc_prof_iqr_lag1`* | *escola* | *78,97%* | *0,4969* | *0,5031* |
| `caderno` | estrutural (controle negativo) | 100% | 0,5013 | 0,4987 |
| `sigla_uf` (26) · `nome_regiao` (5) · `rede_grupo` (3) · `fonte_lag_municipal` (4) · `fonte_lag_uf` (3) | categóricas | 100% | — | — |

Quatro leituras. O topo é inteiramente municipal, como a EDA antecipou. Nada
chega perto de 0,90, que é o resultado esperado quando toda feature vem do ano
anterior — a maior correlação de Spearman com o alvo é 0,268 no dataset inteiro
e 0,240 dentro de `X`, contra o limite de 0,95 da checklist. Porte e dispersão continuam colados em 0,5, com
`mun_n_alunos_lag1` em 0,4765 e `mun_prof_sd_lag1` em 0,4844, e não sustentariam
nada sozinhos. E o `caderno`, controle negativo do desenho, marca 0,5013 — que é
exatamente o que se espera de um caderno randomizado, e o valor de referência
contra o qual a importância dele será lida na Etapa 5.

**As metas confirmam A4 sobre a nova base.** `mun_meta_2024_lag1` correlaciona
0,9685 com a taxa municipal ponderada de 2023 e 0,6315 com a de 2024;
`mun_nivel_alfabetizacao_lag1`, lido da linha de 2023, correlaciona 0,9578 com a
taxa de 2023 e 0,6237 com a de 2024. As duas são a taxa base reescalada — entram
como lag declarado, não como informação nova, e é assim que precisam ser lidas na
interpretabilidade.

---

## 6. Colinearidade e o subconjunto podado

VIF sobre 300 mil linhas, features numéricas, nulo preenchido pela mediana.

**Conjunto completo** — a matriz é praticamente singular, como a EDA já havia
diagnosticado, porque `mun_desvio_vs_uf = mun_taxa − uf_taxa` é combinação linear
exata:

```
mun_taxa_alfab_lag1 1289,1 · uf_taxa_alfab_lag1 660,2 · mun_desvio_vs_uf 590,2
mun_n_escolas_lag1 61,1 · mun_n_alunos_lag1 60,7 · mun_meta_2024_lag1 24,7
uf_media_portugues_lag1 24,6 · mun_media_portugues_lag1 23,4
```

**Subconjunto podado** (`pipeline.FEATURES_PODADAS`), uma métrica de nível por
território, já pré-processado:

| Feature | VIF |
|---|---|
| `mun_taxa_alfab_lag1` | 2,20 |
| `uf_taxa_alfab_lag1` | 2,15 |
| `mun_taxa_presenca_lag1` | 1,20 |
| `tem_historico_municipio` | 1,15 |
| `mun_share_rede_estadual_lag1` | 1,10 |
| `mun_n_alunos_lag1` | 1,10 |

Todo VIF abaixo de 3, como a checklist exige, mais `rede_grupo` e `nome_regiao`
em one-hot. `sigla_uf` fica de fora porque `uf_taxa_alfab_lag1` já é o efeito de
UF em uma coluna contínua, e 26 dummies ao lado dela reintroduziriam a
colinearidade que a poda existe para evitar. O subconjunto ficou sem métrica de
dispersão porque a única disponível, `mun_prof_sd_lag1`, saiu junto com o bloco
de percentis (seção 8) — e ela valia AUC 0,4844 sozinha.

**Um detalhe que só apareceu ao montar o pipeline.** As dez colunas municipais
que dependem do microdado somem **em bloco**, com padrão idêntico em 100,00% das
linhas, e esse padrão é o complemento exato de `tem_historico_municipio`. Com
`add_indicator=True`, o modelo linear receberia quatro cópias da mesma coluna
mais a flag, e a matriz voltaria a ser singular. Daí o parâmetro
`indicador_de_nulo=False` no baseline linear: árvore não se importa, regressão
logística sim.

---

## 7. O pipeline

20 colunas de entrada viram **89** no estimador: 14 numéricas, 12 indicadores de
nulo, 4 + 3 de proveniência, 3 de rede, 5 de região, 26 de UF e 22 de caderno.

**`min_frequency` só no `caderno`.** O plano pedia
`OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01)` por
causa do `caderno = 43`, que tem 12 alunos e não existe em 2023. O parâmetro
resolve o caderno — vira `caderno_infrequent_sklearn` — mas, aplicado à
`sigla_uf`, colapsaria **SE, TO, AP e AC**, as quatro UFs abaixo de 1% da coorte,
com 10 a 18 mil alunos cada, justamente na variável de 49pp de amplitude. O
limiar ficou restrito à categórica que precisa dele, e as duas configurações
estão travadas por teste.

**Indicadores de nulo redundantes.** Os 12 indicadores gerados colapsam em **5**
padrões distintos de nulidade. Foram mantidos porque `add_indicator=True` é a
defesa correta contra uma coluna futura cuja nulidade *não* esteja em bloco, mas
a Etapa 5 precisa lê-los como família: colunas idênticas dividem a importância
entre si e fazem a nulidade parecer irrelevante quando ela não é.

---

## 8. Réplica do experimento B5

Experimento histórico, anterior à correção de `scripts/experimento_b5.py`: `StratifiedGroupKFold(5)` por município, seeds 42, 7
e 2024, folds compartilhados entre os conjuntos, hiperparâmetros fixos e
idênticos, 45 ajustes de LightGBM em 28,5 minutos. Saída bruta em
`reports/metrics/experimento_b5_replica.csv`.

| Conjunto | Features | ROC-AUC | dp entre folds | mín | máx | PR-AUC |
|---|---|---|---|---|---|---|
| **A** — sem os percentis | 20 | **0,6589** | 0,0135 | 0,6386 | 0,6811 | 0,5468 |
| **B** — A + percentis do microdado | 25 | **0,6593** | 0,0134 | 0,6387 | 0,6814 | 0,5472 |
| **C** — B + bloco de escola | 34 | **0,6589** | 0,0134 | 0,6379 | 0,6807 | 0,5466 |

Comparação pareada por fold, com o t corrigido de **Nadeau-Bengio** — o t pareado
comum é inválido aqui, porque os folds compartilham dados de treino e a variância
do delta sai subestimada:

| Comparação | Δ médio | IC95 pareado | t | p | folds a favor |
|---|---|---|---|---|---|
| B − A (percentis) | **+0,00037** | [−0,00080; +0,00153] | 0,67 | 0,51 | 10 de 15 |
| C − B (escola) | **−0,00038** | [−0,00126; +0,00049] | −0,94 | 0,36 | 4 de 15 |
| C − A | −0,00002 | [−0,00091; +0,00088] | −0,04 | 0,97 | 5 de 15 |

E por seed, para mostrar que não é instabilidade de semente:

| Seed | A | B | C |
|---|---|---|---|
| 42 | 0,6585 | 0,6590 | 0,6589 |
| 7 | 0,6594 | 0,6595 | 0,6589 |
| 2024 | 0,6589 | 0,6594 | 0,6589 |

**Decisão: os percentis do microdado saem de `FEATURES_MODELO`.**

O ganho é de +0,0004 de ROC-AUC. É um sexto do +0,0022 que a auditoria mediu num
único split — a diferença entre os dois números é a própria demonstração de por
que a réplica estava na checklist. O intervalo pareado inclui o zero, o p é 0,51,
e o desvio entre folds (0,0134) é 36 vezes o efeito. Pela regra de parcimônia do
projeto, feature que não melhora a métrica além da dispersão do desenho sai.

Três razões somam-se ao número. As cinco colunas cobrem 76,88% contra os 98,09%
de `mun_media_portugues_lag1`, que mede a mesma distribuição e fica no conjunto —
é exatamente o mecanismo que a auditoria havia identificado, o de que a vantagem
univariada evapora no multivariado porque as duas medem a mesma coisa. Elas
faltam precisamente para SP, DF e AC, agravando a redundância de indicadores da
seção 6. E correlacionam acima de 0,95 com colunas que ficam, o que na Etapa 5
diluiria a importância da família municipal entre sinônimos.

O que a decisão **não** significa: o microdado continua sendo fonte obrigatória
do projeto, por `peso_aluno` e por `presenca` em grão de escola e de município. O
que sai é o único bloco que só ele produz, não a dependência.

As colunas continuam construídas e gravadas no dataset. Se a Etapa 4 quiser
reabrir a questão com os hiperparâmetros do campeão, é uma linha:
`FEATURES_MODELO + FEATURES_PERCENTIS_MICRODADO`.

**A ablação do bloco de escola fecha a seção 1 pelo lado multivariado:** −0,00038
de AUC e 4 folds a favor em 15. Um bloco de nove colunas que custa métrica é a
definição de ruído com custo de manutenção.

**Uma leitura que a Etapa 4 vai precisar.** Os três conjuntos ficam em 0,659 de
ROC-AUC com desvio de 0,013 entre folds, dentro da expectativa de 0,65 ± 0,03 do
diagnóstico. Nesse patamar, o que separa um modelo do outro não vai ser lista de
features — e é preciso comparar contra o baseline de uma variável com o **mesmo**
desenho de validação antes de afirmar qualquer superioridade.

---

## 9. Asserções e testes

`tests/test_features.py`, 17 testes novos, somados aos 24 de `test_dados.py`:
**41 passando**.

| Invariante travada | Teste |
|---|---|
| `set(X.columns) & COLS_PROIBIDAS == ∅`, e `separar_X_y` recusa coluna proibida | `test_nenhuma_coluna_proibida_chega_a_X` |
| A feature municipal reproduz 2023 (MAE < 0,01pp) e difere de 2024 (MAE > 1pp) | `test_toda_feature_de_lag_e_de_2023` |
| Maior \|Spearman\| com o alvo abaixo de 0,95 | `test_correlacao_com_o_alvo_esta_longe_do_limite` |
| O imputador aprende a mediana do fold de treino, não da base inteira | `test_pipeline_aprende_so_no_fold_de_treino` |
| As 936 linhas sem medida saíram do treino | `test_linhas_sem_medida_sairam_do_treino` |
| Cobertura de 98,09% / 76,88% e as quatro fontes do lag | `test_coalescencia_cobre_98_por_cento_da_coorte` |
| O gap de lag são SP, DF e AC; sem lag de UF só AC e DF | `test_gap_de_lag_continua_sendo_tres_ufs` |
| A nulidade municipal é em bloco | `test_nulidade_municipal_e_em_bloco` |
| `id_escola` não é chave longitudinal | `test_id_escola_nao_e_chave_longitudinal` |
| O join real não supera o embaralhamento dentro da UF | `test_lag_de_escola_nao_supera_o_embaralhamento_dentro_da_uf` |
| `caderno = 43` vira categoria infrequente | `test_caderno_43_vira_categoria_infrequente` |
| SE, TO, AP e AC sobrevivem ao one-hot | `test_uf_pequena_nao_e_colapsada` |
| Todo VIF do subconjunto podado abaixo de 3, e a logística converge | `test_subconjunto_podado_tem_vif_abaixo_de_3` |
| A feature store é determinística entre execuções | `test_feature_store_e_deterministica` |
| O Parquet em disco é o que o código produz hoje | `test_dataset_em_disco_reproduz_a_feature_store` |
| `k = 30` fixo, com os três casos-limite da fórmula | `test_suavizacao_usa_k_fixo_e_documentado` |
| O agregado ponderado reconcilia com o INEP a menos de 0,10pp | `test_agregado_ponderado_reconcilia_com_o_inep` |

---

## 10. O que fica para a Etapa 4

1. **Estratificar toda métrica** por `tem_historico_municipio` e por
   `fonte_lag_municipal`. Os três estratos têm AUC univariada entre 0,347 e 0,453
   na mesma coluna, e uma média global esconde isso.
2. **Comparar contra o baseline de uma variável no mesmo desenho de validação.**
   O conjunto completo entrega 0,6589 em CV agrupada; `mun_media_portugues_lag1`
   sozinha entrega 0,6411 em AUC univariada sobre a coorte inteira. Os dois
   números não são comparáveis como estão, e a Etapa 4 precisa medir os dois no
   mesmo `StratifiedGroupKFold` antes de dizer quanto o modelo agrega.
3. **AC e DF entram sem lag de UF nenhum.** Vale medir o desempenho nesses dois
   territórios separadamente antes de publicar qualquer ranking que os inclua.
4. **`FEATURES_PODADAS` com `indicador_de_nulo=False`** é obrigatório no baseline
   linear; o conjunto completo é singular.
5. **Dois controles negativos, não um.** `caderno` (AUC 0,5013) e o bloco `esc_*`
   são o par que a Etapa 5 usa como detector de sobreajuste. Se qualquer um dos
   dois aparecer alto no SHAP, o alarme é do modelo, não da educação básica.
6. O dataset tem 1.851.852 linhas e 41 colunas: 20 features, 5 do bloco de
   percentis, 9 do bloco de escola, 3 operacionais (`id_municipio`, `peso_aluno`,
   `risco_nao_alfabetizacao`) e 4 de apoio — contagens de matriculados por nível
   territorial e `uf_n_alunos_lag1`, insumos de diagnóstico que ficam fora do
   modelo porque porte de UF não discrimina aluno.

## Execução corrigida

O script atual reserva os municípios antes de amostrar e comparar atributos. Produz `experimento_b5_desenvolvimento.csv` e manifesto JSON. A execução de verificação usa ~150 mil linhas, uma seed e três folds. O CSV `experimento_b5_replica.csv` acima permanece histórico e não representa o protocolo corrigido.
