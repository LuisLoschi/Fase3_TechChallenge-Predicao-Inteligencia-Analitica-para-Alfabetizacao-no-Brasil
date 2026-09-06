> **Atualização 2026-09-05 — retorno CRISP-DM:** este plano registra decisões históricas. A revisão corrigiu o protocolo de seleção e a projeção municipal. Conclusões anteriores sobre teste intocado, escola, imprevisibilidade e corte de porte foram substituídas em [reports/revisao_cientifica.md](reports/revisao_cientifica.md). Pendem validação prospectiva, dimensão socioeconômica e vídeo.

# Plano de Execução — Tech Challenge Fase 3

> Documento vivo de acompanhamento. Cada etapa é marcada como concluída conforme a execução avança.
> Legenda: `[ ]` pendente · `[~]` em andamento · `[x]` concluído

---

## Progresso geral

| Etapa | Status |
|---|---|
| 0. Diagnóstico dos dados | `[x]` concluído |
| 1. Estrutura do projeto | `[x]` concluído |
| 2. Análise Exploratória (EDA) | `[x]` concluído |
| 2.5. Auditoria da Gold e reorganização de `data/` | `[x]` concluído |
| 3. Feature Engineering | `[x]` concluído |
| 4. Modelagem supervisionada | `[x]` concluído |
| 5. Interpretabilidade | `[x]` concluído |
| 6. Aplicação estratégica | `[x]` concluído |
| 7. Documentação e entregáveis | `[~]` em andamento |
| 8. Revisão científica | `[~]` correções aplicadas; gates externos abertos |

---

## Contexto

A base de entrada é a camada Gold em grão aluno ([data/raw/alfabetizacao_aluno/](data/raw/alfabetizacao_aluno/), 615 MB, 3.355.846 linhas), produzida por um pipeline medalhão em PySpark/Databricks já concluído. **Esse ETL não faz parte deste repositório** — a Gold entra como insumo dado, e sua proveniência fica registrada em [data/README.md](data/README.md).

O enunciado da Fase 3 exige um modelo supervisionado que prevê se um aluno é alfabetizado, com pipeline sklearn completa (imputação, encoding, tratamento de data leakage, pré-processamento integrado ao modelo, validação com generalização), interpretabilidade (Feature Importance e SHAP), EDA que sustente as decisões de modelagem, e respostas a 5 perguntas de negócio.

**Restrição:** nenhuma base externa nova — só derivação a partir das bases já presentes no repositório.

---

## Etapa 0 — Diagnóstico dos dados `[x]`

Profiling executado sobre as bases reais. Todos os números abaixo são **medidos**, não estimados.

- [x] Perfil da Gold em grão aluno
- [x] Perfil das bases INEP de município, UF e metas
- [x] Verificação de viabilidade das features de lag temporal
- [x] Medição do teto de performance com baseline empírico

### Achados

| Fato | Valor | Consequência de modelagem |
|---|---|---|
| Gold grão aluno | 3.355.846 × 25 (2023: 1,50M / 2024: 1,85M) | Volume alto, features rasas |
| Target `alfabetizado` | 59,1% = 1 / 40,9% = 0 | **Balanceado** — dispensa SMOTE/reponderação agressiva |
| `serie` | Constante = 2 (2º ano EF) | Cardinalidade 1 → **descartar** |
| `caderno` | 22 valores, taxa entre 0,587 e 0,604 | Caderno randomizado → **ruído** |
| `rede` | 2=Estadual 13% · 3=Municipal 87% · 4=Privada **24 linhas** | Agrupar Privada em "Outros" |
| `proficiencia` (base bruta) | **corr 0,796** com o target | Vazamento direto — confirma exclusão feita na Gold |
| `presenca=0` | 100% têm `alfabetizado=0` | Confirma o filtro `presenca=1`; ausência ≠ não alfabetizado |
| UFs | 26 — **Roraima ausente** | Limitação a declarar no README |
| Nulos | meta_munic 53,2% · meta_uf 45,8% · particip_munic 13,0% | Nulidade **estrutural** (2023 sem meta) → imputar com indicador |

**Viabilidade do lag temporal (confirmada):**
- 85,2% das escolas de 2024 existem em 2023 → **79,0% dos alunos de 2024** têm histórico de escola
- 87,7% dos municípios coincidem → 76,9% dos alunos têm histórico municipal
- Mediana de 34 alunos/escola-ano (p25 19, p75 56) → agregados de escola são ruidosos e **exigem suavização**

### Quatro armadilhas descobertas (todas verificadas duas vezes, de forma independente)

**A1 — O target é uma função determinística da proficiência.**
`alfabetizado = 1[proficiencia >= 743]`, exatamente. Medido nos dois anos: `max(prof | classe 0) = 742,9996` e `min(prof | classe 1) = 743,0000`. Proficiência não é "quase" vazamento — é o alvo reescrito. Confirma a exclusão feita na Gold, e torna a proficiência de **2023** uma excelente feature de lag.

**A2 — `id_aluno` NÃO é chave longitudinal.**
Há 1.515.671 `id_aluno` presentes nos dois anos, mas apenas **0,12% deles na mesma escola**. É ID sequencial reciclado a cada ano, não matrícula. Qualquer join por `id_aluno` entre 2023 e 2024 produziria 1,5 milhão de pares falsos. É a armadilha mais perigosa do projeto e precisa estar documentada.

**A3 — `id_aluno` codifica a UF no prefixo.** Como variável numérica seria proxy territorial disfarçado. Descartar, não apenas ignorar.

**A4 — As metas 2025–2030 são determinísticas, derivadas da própria taxa base.**
`meta_alfabetizacao_2030 = 80` para todos os 5.352 municípios; as metas são **100% idênticas** nas linhas de 2023 e de 2024; e `corr(meta_2025, taxa_2023) = 0,977`. A meta é simplesmente uma trajetória linear da taxa base até 80 em 2030. Duas consequências: (a) `meta_alfabetizacao_municipio` na Gold é um **lag disfarçado** — legítimo como feature, mas precisa ser declarado como tal; (b) **reformula a pergunta de negócio 4** (ver Etapa 6).

**A5 — `proporcao_aluno_nivel_0..8` é 100% nula em 2023.** Não serve como lag. Substituto melhor: reconstruir **percentis de proficiência** (p25 / p50 / p75 / IQR / desvio-padrão) da escola e do município a partir do microdado de 2023 — mais informativo que a proporção por nível.

### Teto de performance medido

GBM com features de lag 2023, split agrupado por município:

```
ROC-AUC 0,63 – 0,67   |   PR-AUC (classe não-alfabetizado) 0,53 vs. prevalência 0,403

AUC univariada:  proficiência média municipal 2023 ... 0,660
                 taxa municipal 2023 ................. 0,654
                 escola suavizada .................... 0,574 – 0,610
                 proficiência média escola ........... 0,566
                 escola crua ......................... 0,558
                 presença escola ..................... 0,528
```

**A fraqueza da taxa de escola não é ruído amostral.** A explicação natural seria o tamanho: com mediana de 34 alunos por escola, o erro-padrão da taxa fica em torno de 8pp, e daí sairia a suavização empírico-bayesiana como correção obrigatória. O teste derruba essa leitura — a taxa da escola não prediz melhor em escolas grandes (AUC 0,554 com ≥80 alunos contra 0,560 com <20). Se fosse ruído amostral, escolas grandes teriam sinal claramente melhor. O sinal é **territorial, não escolar**: o município carrega a informação e a escola acrescenta pouco. A suavização segue no plano pela melhora de 0,558 para 0,574, como refinamento modesto e não como pilar.

**Expectativa realista: ROC-AUC 0,65 ± 0,03.**

E o dado que precisa estar no README, não escondido: o baseline "ranquear pela taxa municipal do ano anterior" sozinho já entrega **AUC 0,654**, contra 0,649–0,667 do GBM completo. **O modelo mal supera a heurística de uma variável** — um avaliador atento vai notar isso, então a entrega deve ser quem aponta primeiro. A leitura honesta: a base **não tem nenhuma variável sobre o aluno** (sem nível socioeconômico, cor/raça, idade, frequência, histórico escolar), então 0,65 é a medida de quanto o **território** determina o destino individual — o que é, em si, o achado de política pública. Um AUC de 0,95 aqui seria sinal de vazamento, não de qualidade.

E o número que sustenta a camada estratégica: **no grão municipal a persistência é alta — `corr(taxa_2023, taxa_2024) = 0,636`**, subindo para 0,73 em municípios com ≥50 alunos e 0,81 com ≥200. É no município, não no aluno, que a predição funciona.

---

## Decisões travadas

| Decisão | Escolha |
|---|---|
| Ambiente | Local, **venv dedicado** (`.venv/`), Python 3.14.3 + pandas + scikit-learn |
| Escopo | Modelo de aluno **+** camada estratégica municipal |
| Bibliotecas | LightGBM/XGBoost liberados |
| Volume | Treino com 100% dos dados; CSV de 615 MB fora do Git |
| Organização dos dados | Tudo sob `data/` — `raw/` (imutável), `reference/` (versionado), `interim/`, `processed/`. Sem diretório de dados na raiz |
| `peso_aluno` | **Proibido como preditor, obrigatório como peso** em todo número populacional (Etapa 2.5) |
| Rede nos agregados | `rede = 5` para reconciliar; coalescência `5 → 3` para construir lag (Etapa 2.5) |
| Fonte das features de lag | A **Gold**; o microdado entra por `peso_aluno`, `presenca` e reconciliação (Etapa 2.5) |

> **Duas ressalvas:**
> 1. **SHAP** não foi marcado na seleção de bibliotecas, mas o enunciado o recomenda explicitamente e a instrução foi seguir o roteiro à risca. O plano **inclui SHAP**. Para remover: uma linha de `requirements.txt` e uma seção do notebook 04.
> 2. **Risco de bibliotecas resolvido (medido).** O ambiente é Python 3.14.3 e havia dúvida sobre wheels. Resolução de dependências executada: `lightgbm 4.7.0` (`py3-none`), `xgboost 3.4.1` (`py3-none`), `shap 0.52.0` (`cp312-abi3`) e a cadeia `numba 0.67.0` / `llvmlite 0.49.0` (`cp314`) — **todos com wheel binária, nenhum build a partir de sdist**. O campeão é LightGBM e a interpretabilidade usa SHAP, sem fallback. `HistGradientBoostingClassifier` + `permutation_importance` permanecem como plano B apenas se algo quebrar em execução.

---

## Arquitetura da solução

```
data/raw/alfabetizacao_aluno/*.csv  +  data/raw/inep/*.csv
              │
              ▼
   [A] Feature Store temporal          src/preprocessing/
       agregados de 2023 → features dos alunos de 2024
              │
              ▼
   [B] Modelo supervisionado de aluno  src/modeling/
       Pipeline sklearn · GroupKFold por município · SHAP
              │
              ▼
   [C] Camada estratégica municipal    src/modeling/ + src/visualization/
       risco · clusterização · projeção de metas 2025-2030
```

**Dataset de modelagem: alunos de 2024** (1,85M). 2023 é usado exclusivamente como **fonte de features históricas**, nunca como linha de treino do modelo principal. Assim toda feature é temporalmente anterior ao alvo.

---

## Etapa 1 — Estrutura do projeto `[x]`

- [x] **`.gitignore` antes de tudo.** Eram 830 MB de CSV não rastreados, e um `git add .` os teria commitado. A conferência mostrou que nenhum CSV havia entrado no histórico, então deu tempo. Os dois diretórios de dados foram depois unificados sob `data/raw/` na Etapa 2.5, com o `.gitignore` reescrito e reverificado por `git add -An`
- [x] **Venv antes de qualquer instalação.** O interpretador em uso era o Python do sistema (`sys.prefix` em `...\Programs\Python\Python314`); instalar o projeto ali contaminaria o ambiente global e tornaria o `requirements.txt` não confiável
- [x] Dependências instaladas no venv, com as wheels do 3.14 já verificadas (ver ressalva 2)

### Insumos de dados

Sete arquivos, seis do INEP mais a Gold. Os dois de meta de UF e de Brasil chegaram a ser
removidos por não alimentarem o modelo, e voltaram: são insumos obrigatórios do ETL em
`scripts/etl/`, que é quem gera a Gold.

| Arquivo | Papel |
|---|---|
| `data/raw/alfabetizacao_aluno/alfabetizacao_aluno_features_2023-2024_20260829.csv` | Coorte e alvo de 2024, **e fonte principal das features de lag de 2023**; única fonte de `sigla_uf`, `nome_regiao`, `nome_municipio` e metas de UF e Brasil |
| `data/raw/inep/..._municipio.csv` | `taxa_alfabetizacao` e `media_portugues` de 2023 para os 625 municípios que a Gold não tem em 2023 (**inclui São Paulo**); `proporcao_aluno_nivel_0..8` só em 2024, para a clusterização |
| `data/raw/inep/..._aluno.csv` | Microdado: fonte de `peso_aluno` (correção oficial de não-resposta), de `presenca` em grão de escola, e base de reconciliação. **Não** é a fonte principal do lag — ver Etapa 2.5 |
| `data/raw/inep/..._meta_alfabetizacao_municipio.csv` | Metas 2025 a 2030 e `nivel_alfabetizacao`, ausentes da Gold, que sustentam a camada estratégica municipal |
| `data/raw/inep/..._uf.csv` | `uf_media_portugues_lag1`; **tem SP em 2023**, ao contrário da Gold |
| `data/raw/inep/..._meta_alfabetizacao_uf.csv` · `..._meta_alfabetizacao_brasil.csv` | Insumos do ETL; as colunas que deles derivam já vêm materializadas na Gold |

> **Atenção no join.** `municipio.csv` e `uf.csv` têm uma linha por rede, não por território
> (23.995 linhas para cerca de 5,4 mil municípios). É preciso escolher a rede antes do merge,
> senão a coorte duplica. Já a base de *metas* municipais é uma linha por município-ano e
> pode ser unida direto.
>
> **Qual rede escolher depende do propósito** (medido na Etapa 2.5): `rede = 5` para
> reconciliar, coalescência `5 → 3` para construir lag. Ver `config.REDE_AGREGADO_COALESCENCIA`.
>
> **O microdado não tem UF**, apenas `id_municipio`. Todo agregado de UF para 2023 depende de
> `data/reference/dim_municipio.csv`, gerado por `scripts/build_dim_municipio.py`.

### Resultado medido

| Item | Valor |
|---|---|
| Ambiente | `.venv/` com Python 3.14.3; `sklearn 1.8.0`, `lightgbm 4.7.0`, `xgboost 3.4.1`, `shap 0.52.0` |
| Conversão para Parquet | 830 MB → 96 MB em 9 s (`zstd`, um arquivo por ano) |
| Dimensão territorial | 5.547 municípios, 26 UFs (Roraima ausente, como previsto) |
| Testes | 14 passando em 1,9 s (`pytest -q`) |

**Descoberta durante a Etapa 1 — códigos de `rede` divergem entre as bases.** Gold e
microdado usam `4 = Privada`; os agregados do INEP usam `5 = Pública` e `0 = Total`.
Medido contra o microdado de 2024: a linha `rede = 5` reproduz a taxa das redes 2+3 com
erro médio de **0,37pp** e cobre 5.516 municípios, enquanto `rede = 0` só existe para 398.
Fixado em `config.REDE_AGREGADO_PUBLICA` e travado por teste.

**A base de metas municipais não precisa de agregação por rede.** Ela cobre só a rede
Municipal e é única por (`ano`, `id_municipio`), então pode ser unida à coorte direto.
Quem tem grão por rede é o *agregado* `municipio.csv`, com 23.995 linhas — confundir as
duas foi o que fez o primeiro teste de metas falhar, e ali o errado era o teste, não a base.

### Árvore alvo

```
├── data/                       # ← reorganizado na Etapa 2.5
│   ├── raw/                    # entrada imutável — GITIGNORED
│   │   ├── inep/               # 6 CSVs do INEP (216 MB)
│   │   └── alfabetizacao_aluno/   # Gold em grão aluno (615 MB)
│   ├── reference/              # dim_municipio.csv — VERSIONADO
│   ├── interim/                # Parquet por ano — GITIGNORED
│   ├── processed/              # dataset final de modelagem — GITIGNORED
│   └── README.md               # ← VERSIONADO
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelagem.ipynb
│   ├── 04_interpretabilidade.ipynb
│   └── 05_aplicacao_estrategica.ipynb
├── src/
│   ├── config.py
│   ├── data/loader.py
│   ├── preprocessing/
│   │   ├── feature_store.py    # agregados de 2023 + suavização
│   │   ├── build_dataset.py    # Gold 2024 + feature store → dataset
│   │   └── pipeline.py         # ColumnTransformer
│   ├── modeling/
│   │   ├── train.py            # Pipeline sklearn + tuning + joblib
│   │   ├── split.py            # GroupShuffleSplit/GroupKFold por município
│   │   └── strategic.py        # risco, clusterização, projeção de metas
│   ├── evaluation/
│   │   ├── metrics.py          # métricas + calibração + threshold
│   │   └── interpret.py        # SHAP, permutation importance
│   └── visualization/plots_features.py
├── models/                     # GITIGNORED
├── reports/
│   ├── metrics/*.json
│   ├── ranking_risco_municipal.csv
│   ├── projecao_metas_municipios.csv
│   ├── clusters_municipais.csv
│   ├── documentacao_tecnica.md
│   └── roteiro_video.md
├── images/{eda,modelagem,interpretabilidade,estrategia}/
├── scripts/
│   ├── prepare_data.py         # CSV → Parquet
│   ├── build_dim_municipio.py  # dim territorial extraída da Gold
│   └── etl/                    # ETL PySpark que gera a Gold (restaurado ao repo)
├── tests/                      # asserções anti-leakage e de reconciliação
├── requirements.txt
├── .gitignore
└── README.md
```

**Notebooks são finos:** importam de `src/` e chamam funções. Toda lógica reutilizável mora em `src/` — o notebook narra a análise, não a implementa.

---

## Etapa 2 — Análise Exploratória `[x]`

Notebook `01_eda.ipynb` · Gráficos em `images/eda/`

Cada análise amarrada a uma hipótese e a uma decisão de modelagem — sem gráfico decorativo.

- [x] 1. Distribuição do target global e por ano → base balanceada → dispensa reamostragem
- [x] 2. Cardinalidade e variância de cada coluna → `serie` é constante → descarte
- [x] 3. Taxa por `caderno` (barras + IC) → caderno randomizado é ruído → confirma remoção por evidência
- [x] 4. Taxa por região e por UF (ranking) → desigualdade regional → justifica TargetEncoder em `sigla_uf`
- [x] 5. Taxa por rede → Estadual > Municipal → mantém `rede`; agrupa Privada
- [x] 6. Mapa de nulos por coluna e por ano → nulidade estrutural → imputação **com indicador**
- [x] 7. Distribuição de alunos por escola/município → escolas pequenas → **motiva a suavização**
- [x] 8. Dispersão taxa 2023 × 2024 (escola e município) → persistência temporal → valida o lag
- [x] 9. Correlação (Spearman) + VIF entre features derivadas → multicolinearidade → poda
- [x] 10. Distribuição de `proficiencia` × `alfabetizado` → **corte determinístico em 743,0** → documenta o vazamento (A1)
- [x] 11. Taxa observada 2024 × meta 2024 por município → fundamenta a camada estratégica
- [x] 12. Cobertura do lag (% de alunos com histórico) → justifica as flags `tem_historico_*`
- [x] 13. **`id_aluno` entre anos: quantos pares, quantos na mesma escola** → 0,12% → prova que não é chave longitudinal (A2)
- [x] 14. **Prefixo de `id_aluno` × `id_municipio`** → codifica UF → justifica o descarte (A3)
- [x] 15. **Dispersão meta × taxa base** → `corr 0,977`, meta_2030 = 80 para todos → **a meta é um lag disfarçado** (A4)
- [x] 16. AUC da taxa de escola estratificada por porte da escola → não melhora com escola grande → **o sinal é territorial, não escolar**
- [x] 17. Persistência municipal `corr(taxa23, taxa24)` por faixa de porte → 0,64 / 0,73 / 0,81 → sustenta a camada municipal
- [x] 18. Viés de seleção dos 13% ausentes (comparar perfil municipal de alta vs. baixa presença)
- [x] Redigir as hipóteses analíticas formuladas a partir da EDA (H1 a H6, cada uma com critério de falseamento)


### Resultado da Etapa 2

Entregue em `notebooks/01_eda.ipynb` (71 células, todas as 30 de código executadas no kernel
do venv), `src/eda.py` (1.390 linhas, 55 funções) e 23 gráficos em `images/eda/`.

**O que a medição confirmou:** balanceamento do alvo (59,75% alfabetizados em 2024, classe de
interesse em 40,25%), `serie` constante, desigualdade territorial, nulidade estrutural,
escolas pequenas, persistência municipal, corte determinístico em 743,0, cobertura do lag,
`id_aluno` reciclado, prefixo territorial, metas determinísticas, sinal territorial e viés de
ausência. Todas as cinco armadilhas do diagnóstico (A1 a A5) se sustentaram.

**Sete pontos que contrariam ou corrigem o plano:**

1. **O `caderno` não tem ICs sobrepostos.** A amplitude entre os 21 cadernos regulares é de
   1,75pp e a AUC univariada é 0,506, então a conclusão prática não muda, mas o plano
   afirmava sobreposição de intervalos e ela não existe: com 85 mil alunos por caderno, os
   ICs do caderno 12 e do 18 não se tocam. O argumento correto é o tamanho do efeito.
   Apareceu também um `caderno = 43` com 12 alunos, ausente em 2023, que quebraria um
   `OneHotEncoder` ingênuo.
2. **`preenchimento_caderno` é vazamento contemporâneo, não feature de engajamento.** São 936
   alunos (0,05%) com `alfabetizado = 0` por ausência de medida, o mesmo mecanismo dos
   `presenca = 0` que sobreviveu ao filtro da Gold. Além disso a variável é registrada no
   instante da prova que produz o alvo. Sai das features, entra na matriz anti-leakage, e as
   936 linhas saem do treino.
3. **Duas colunas constantes a mais dentro de 2024:** `meta_alfabetizacao_brasil` e
   `percentual_participacao_brasil`.
4. **A suavização da escola não é calibrável por CV.** A AUC cresce monotonicamente com `k`
   (0,561 em k=0, 0,642 em k=100, 0,653 em k=500) e nunca supera a taxa municipal pura de
   0,654. O melhor `k` é o maior da grade, ou seja, a suavização funciona apagando a escola.
   Ela vira imputação documentada com `k` fixo.
5. **`esc_desvio_vs_municipio` sai da lista de features.** AUC univariada de 0,457 e
   `corr(desvio 2023, desvio 2024)` de 0,009 no total, -0,018 em escolas com 80+ alunos. É
   artefato de super-subtração, e como feature induziria leitura causal falsa na Etapa 5.
6. **A colinearidade é exata, não apenas alta.** O VIF do conjunto completo estoura em 10¹³
   com valores negativos porque `iqr` e os desvios são combinações lineares das demais
   colunas. Sem elas o VIF fica entre 1,1 e 30,3, e removendo as `*_prof_media` nenhum passa
   de 2,9. O baseline linear precisa do subconjunto podado; o SHAP precisa agrupar por família.
7. **Números atualizados em relação à Etapa 0**, por diferença de população (lá o microdado
   com `presenca = 1`, aqui a Gold): persistência municipal 0,645 (era 0,636), evolução
   mediana +2,1pp com 42,9% de municípios em queda (era +2,6pp e 41,2%), escola mediana com
   32 alunos (era 34).

**O achado central.** A mesma métrica que é estável no município é ruído na escola, e melhora
com o porte no município enquanto piora com o porte na escola. A persistência municipal vai de
0,645 (todos) a 0,740 (50+ alunos) e 0,809 (200+), enquanto a AUC da taxa escolar fica entre
0,554 e 0,560 em qualquer faixa de porte e o desvio da escola em relação ao município não se
repete de um ano para o outro. Três testes independentes derrubaram a hipótese de ruído
amostral. O esforço de engenharia de features vai para o grão municipal.

**Síntese univariada:** as cinco melhores features são todas municipais, de
`mun_prof_media_lag1` (0,660) a `mun_prof_p25_lag1` (0,648); a melhor puramente escolar é
`esc_prof_p75_lag1` (0,572). Nenhuma chega perto de 0,90, o que descarta vazamento, e nenhuma
de porte ou dispersão passa de 0,55, o que já antecipa ganho pequeno do GBM sobre o baseline
de uma variável.

**Nota de ambiente resolvida durante a Etapa 2.** O único kernel Jupyter registrado apontava
para o Python do sistema, então a primeira execução do notebook rodou fora do venv. Isso
passaria despercebido na EDA, que não usa lightgbm nem shap, e quebraria nas Etapas 3 a 5.
Foi registrado o kernel `tc-fase3` apontando para `.venv`, e o notebook fixa esse kernel no
`kernelspec`. Verificado: `sys.prefix` dentro do kernel é o venv, com lightgbm e shap visíveis.

---

## Etapa 2.5 — Auditoria da camada Gold `[x]`

Auditoria feita antes de a Etapa 3 começar, porque ela ia consumir a Gold e o microdado
simultaneamente sem que ninguém tivesse provado que a Gold é fiel nem medido se a segunda
fonte se paga. **Evidência completa em [`reports/auditoria_camada_gold.md`](reports/auditoria_camada_gold.md)** —
aqui ficam só as decisões que mudam a execução.

**Veredito: a Gold está correta e não será refeita.** Nos dois anos o conjunto de `id_aluno`
é idêntico ao dos presentes no microdado, sem duplicata, com zero divergência de alvo em
3.355.846 comparações. Os defeitos encontrados se resolvem no `feature_store.py`.

### Cinco decisões que a auditoria fecha

| # | Decisão | Efeito na Etapa 3 |
|---|---|---|
| B1 | `percentual_participacao_municipio` e `_uf` **saem das features** — são a presença do próprio ano do alvo (corr 1,00000), não metas | Removidas do bloco "Metas"; vetor 14 da matriz anti-leakage. Custo: −0,0004 de AUC |
| B2 | `peso_aluno` é a correção oficial de não-resposta do INEP: **proibido como preditor, obrigatório como peso** em todo número populacional | Reconciliação cai de 0,99pp para 0,05pp. Reescreve a limitação nº 5 |
| B3 | O gap de 23% sem histórico municipal **são SP, DF e AC inteiros** (99,92%), não ruído difuso | `tem_historico_municipio` é quase colinear com `sigla_uf`; toda métrica passa a ser estratificada por ela |
| B4 | O agregado do INEP tem SP em 2023: **coalescer `rede 5 → 3`** no lag | Gap residual cai de 10,77% para 1,91%. Exige a coluna de proveniência `fonte_lag_municipal` |
| B5 | A **Gold** é a fonte principal do lag; o microdado vale +0,0022 de AUC com IC pareado incluindo zero | O microdado entra por `peso_aluno`, `presenca` e reconciliação. Os percentis de proficiência ficam pendentes de réplica |

### Reorganização de `data/`

Os três diretórios de dados da raiz viraram um, na convenção Cookiecutter Data Science:
`data/{raw/{inep,alfabetizacao_aluno}, reference, interim, processed}`. `external/` virou
`reference/` — o arquivo que morava lá é derivado internamente da Gold, o oposto do que
"external" significa na convenção. O CSV da Gold foi renomeado para
`alfabetizacao_aluno_features_2023-2024_20260829.csv`, que diz a tabela de origem, o período
e o timestamp da extração, em vez da ferramenta que a produziu.

### Executado

- [x] `data/` reorganizado; `.gitignore` reescrito e verificado com `git add -An`
- [x] `config.py` — caminhos, `REDE_AGREGADO_COALESCENCIA`, `COL_PESO`, `COLS_PROIBIDAS` atualizada
- [x] `loader.py` e `build_dim_municipio.py` — docstrings e uma asserção que era vazia por construção
- [x] `tests/test_dados.py` — de 14 para 24 testes; os 7 novos travam B1 a B5
- [x] `prepare_data.py --force` reexecutado dos caminhos novos: mesmas contagens, 24 testes passando

---

## Etapa 3 — Feature Engineering `[x]`

`src/preprocessing/` · Notebook `02_feature_engineering.ipynb` · **Evidência completa em
[`reports/feature_engineering.md`](reports/feature_engineering.md)** — aqui ficam só as
decisões que mudam a execução seguinte.

- [x] `feature_store.py`, agregados de 2023 por escola, município e UF, a partir da Gold
- [x] Coalescência `5 → 3` do agregado do INEP, com `fonte_lag_municipal` (e `fonte_lag_uf`,
      que não estava no plano e recupera São Paulo no grão UF)
- [x] Ponderação por `peso_aluno` nos agregados municipais e de UF; grão aluno sem peso
- [x] Suavização empírico-bayesiana com `k = 30` fixo e documentado, sem calibração
- [x] `build_dataset.py` → `data/processed/dataset_2024.parquet`
- [x] Removidas as 936 linhas de `preenchimento_caderno = 0`
- [x] `pipeline.py`, `ColumnTransformer` com imputação, escala e encoding
- [x] Subconjunto podado do baseline linear, com todo VIF abaixo de 3 (máximo medido: 2,20)
- [x] Asserções anti-leakage, incluindo `set(X.columns) & set(COLS_PROIBIDAS) == ∅`
- [x] Réplica do experimento B5 com `StratifiedGroupKFold(5)` e 3 seeds

### Resultado da Etapa 3

Dataset com **1.851.852 alunos de 2024 e 41 colunas, das quais 20 são features**, todas
derivadas exclusivamente de 2023. Testes: de 24 para **41**, os 17 novos em
`tests/test_features.py`. Notebook com 11 células de código executadas no kernel `tc-fase3`
e 4 gráficos em `images/features/`.

**Três decisões que contrariam o que este plano previa:**

1. **O bloco de escola sai das features — `id_escola` não é chave longitudinal.** Dos 36.051
   identificadores presentes nos dois anos, só **2,40%** caem no mesmo município, e escola não
   muda de município: o identificador é reatribuído a cada edição em ordem territorial, como
   `id_aluno` (A2). O join entregaria a **76,35% da coorte** o histórico de outra escola. Um
   valor sorteado dentro da própria UF prediz igual ao join real (AUC 0,5601 contra 0,5581) e
   sorteado entre UFs cai a 0,4998; no multivariado o bloco custa −0,0004 de AUC. **Invalida as
   nove features de escola do dicionário abaixo**, que viram controle negativo declarado ao lado
   do `caderno`. Reforça, por um motivo mais forte, a conclusão da EDA de que o sinal é
   territorial: não é que a escola tenha pouco sinal, é que a escola nunca foi medida.
2. **Os percentis de proficiência do microdado saem** (a decisão que B5 deixou pendente):
   **+0,00037** de ROC-AUC em 15 folds, IC95 pareado de [−0,0008; +0,0015], t de Nadeau-Bengio
   0,67 e p 0,51 — um sexto do +0,0022 do split único da auditoria, contra desvio de 0,0134
   entre folds. `mun_media_portugues_lag1` mede a mesma distribuição e cobre 98,09% contra
   76,88%. Ficam construídos no dataset; reabrir na Etapa 4 é uma linha.
3. **`min_frequency = 0,01` só no `caderno`.** Aplicado a todas as categóricas, como o plano
   pedia, o limiar colapsaria **SE, TO, AP e AC** — as quatro UFs abaixo de 1% da coorte — na
   variável de 49pp de amplitude. No `caderno` ele resolve o `43` de 12 alunos, como previsto.

**Confirmado como previsto:** a coalescência leva a cobertura de nível a **98,09%** (76,88% sem
ela) e as distribucionais seguem em 76,88%; `k = 30` é imputação e não recuperação de sinal
(AUC 0,6045 contra 0,6362 da taxa municipal que a compõe); a ponderação por `peso_aluno` não
muda AUC (0,3466 contra 0,3464) e vale 0,92pp de reconciliação com o INEP; nenhuma feature passa
de |Spearman| 0,268 com o alvo, contra o limite de 0,95.

### Dicionário final — 20 features, todas de 2023

| Bloco | Colunas | Melhor AUC univariada (alfabetização) |
|---|---|---|
| Município | 9 — taxa, média de português, nível, meta 2024, presença, porte, share de rede, desvio vs. UF | `mun_media_portugues_lag1` **0,641** (cobertura 98,09%) |
| UF | 3 — taxa, média de português, presença | `uf_media_portugues_lag1` 0,618 |
| Cobertura do lag | 4 — `tem_historico_escola`, `tem_historico_municipio`, `fonte_lag_municipal`, `fonte_lag_uf` | 0,506 (são rastro, não sinal) |
| Estrutural | 4 — `rede_grupo`, `nome_regiao`, `sigla_uf`, `caderno` | `caderno` 0,4987 — **controle negativo** |
| *Percentis do microdado* | *5, construídos e fora do modelo* | *`mun_prof_p75_lag1` 0,658, cobertura 76,88%* |
| *Escola* | *9, construídos e fora do modelo* | *`esc_taxa_alfab_lag1_suav` 0,6045* |

Fora de `X` e dentro do arquivo: `id_municipio` (grupo de CV), `peso_aluno` (peso populacional,
B2) e o alvo. `pipeline.separar_X_y` é a única porta de entrada da matriz e falha se qualquer
uma delas atravessar.

### O que a Etapa 4 recebe

- **Estratificar toda métrica** por `tem_historico_municipio` e por `fonte_lag_municipal`: a
  mesma coluna vale AUC 0,347 na fatia da Gold, 0,453 na do agregado `rede 5` e 0,420 na de
  `rede 3`.
- **Comparar contra o baseline de uma variável no mesmo desenho de validação.** O conjunto
  completo entregou 0,6589 em CV agrupada com hiperparâmetros fixos; a melhor feature isolada
  vale 0,641 em AUC univariada. Os dois números ainda não são comparáveis, e a Etapa 4 precisa
  medi-los no mesmo `StratifiedGroupKFold` antes de afirmar ganho.
- **`FEATURES_PODADAS` com `indicador_de_nulo=False`** no baseline linear: as dez colunas
  municipais do microdado somem em bloco, e o indicador vira cópia de `tem_historico_municipio`.
- **AC e DF não têm lag de UF nenhum** — nenhuma fonte de 2023 os cobre. Medir esses dois
  territórios à parte antes de publicar ranking que os inclua.
- **Dois controles negativos, não um:** `caderno` e o bloco `esc_*`.


### Estratégia anti-leakage

| # | Vetor de vazamento | Defesa |
|---|---|---|
| 1 | `proficiencia` do ano corrente, que **é o target reescrito**: corte em 743,0 e AUC 1,000 | Removida; usada só como agregado de 2023 |
| 2 | Agregados do mesmo ano, média do próprio alvo | Todo agregado calculado **exclusivamente sobre 2023** |
| 3 | `proporcao_aluno_nivel_0..8` de 2024 | Fora do modelo; só na camada estratégica descritiva |
| 4 | **Join longitudinal falso por `id_aluno`**: 1,51M de pares, só 0,12% na mesma escola | **Proibido**, travado em `tests/test_dados.py` |
| 5 | **`id_aluno` como numérica**, cujo prefixo codifica a UF | Descartada da lista de features |
| 6 | Imputação e escala ajustadas antes do split | `ColumnTransformer` dentro do `Pipeline`, com `fit` só no treino |
| 7 | Memorização territorial | **`GroupShuffleSplit` e `StratifiedGroupKFold` por `id_municipio`** |
| 8 | Ausentes com `alfabetizado = 0` por default, 267.772 em 2024 | Já filtrados na Gold (`presenca = 1`); em 2023 usados **só** para a taxa de presença |
| 9 | Metas que embutem a taxa observada | Documentado como lag disfarçado (A4); só metas defasadas em relação ao alvo |
| 10 | Seleção de features e hiperparâmetros olhando o teste | Teste tocado **uma única vez**, ao final |
| 11 | Amostragem para tuning quebrando a estrutura de grupo | Amostrar **municípios inteiros**, nunca alunos individuais |
| 12 | IC de métrica artificialmente estreito | Bootstrap reamostrando **municípios**, não alunos |
| 13 | **`preenchimento_caderno`, vazamento contemporâneo**: registrado no instante da prova, com 936 linhas de alvo 0 por ausência de medida | Coluna fora das features; as 936 linhas saem do treino |
| 14 | **`percentual_participacao_municipio` e `_uf`, agregado contemporâneo de presença**: `corr = 1,00000` e `MAE = 0,002pp` com a taxa de presença observada de 2024, sobre a mesma coorte que se quer predizer (B1) | Ambas em `COLS_PROIBIDAS`, travado em `test_participacao_e_a_presenca_do_proprio_ano`. A versão defasada de 2023 é permitida e melhor: AUC 0,5900 contra 0,5701 |
| 15 | **Heterogeneidade de apuração na feature coalescida**: após o `5 → 3`, o mesmo `mun_taxa_alfab_lag1` pode vir de três fontes com fidelidades diferentes | Coluna de proveniência `fonte_lag_municipal` obrigatória no dataset; métricas estratificadas por ela e por `tem_historico_municipio` |
| 16 | **`peso_aluno` usado como preditor** — é derivado da presença observada na própria coorte | Proibido em `X`; permitido apenas como peso de agregação (B2) |
| 17 | **Join falso por `id_escola`**: o identificador é reatribuído a cada edição, e só 2,40% dos repetidos caem no mesmo município — o join daria a 76,35% da coorte o histórico de outra escola, com sinal aparente porque 80,6% ficam na mesma UF | Bloco `esc_*` fora de `FEATURES_MODELO`, mantido como controle negativo; travado em `test_lag_de_escola_nao_supera_o_embaralhamento_dentro_da_uf` |

---

## Etapa 4 — Modelagem supervisionada `[x]`

`src/modeling/` + `src/evaluation/` · Notebook `03_modelagem.ipynb` · **Evidência completa em
[`reports/modelagem.md`](reports/modelagem.md)** — aqui ficam só as decisões que mudam a
execução seguinte.

- [x] `split.py`, `GroupShuffleSplit` 80/20 por `id_municipio` e `StratifiedGroupKFold(5)`
      com folds materializados uma vez e compartilhados por todos os candidatos
- [x] Baseline `DummyClassifier(strategy="prior")` e baseline heurístico de uma variável,
      reimplementado como estimador para passar pelos mesmos folds
- [x] `LogisticRegression` sobre `FEATURES_PODADAS` com `indicador_de_nulo=False`
- [x] `RandomForestClassifier` e `LGBMClassifier` — campeão
- [x] Busca de 40 configurações em `StratifiedGroupKFold(5)`, amostrando municípios inteiros,
      com o resultado remedido no desenvolvimento completo
- [x] Refit no desenvolvimento; calibração isotônica testada e **rejeitada** pelo próprio critério
- [x] Dois limiares documentados: máximo F1 e capacidade de 20%
- [x] `permutation_test_score` com dois nulos; `learning_curve` em municípios; `LeaveOneGroupOut`
      por região; modelo de drift 2023 → 2024; teste de invariância do `caderno`
- [x] IC por bootstrap de **municípios**; toda métrica estratificada; `campeao.joblib` + `reports/metrics/*`

### Resultado da Etapa 4

**ROC-AUC 0,6599 [0,6314; 0,6853]** no teste de 330.836 alunos e 1.104 municípios inéditos,
com PR-AUC 0,5360 contra prevalência 0,3812 e Brier 0,2179. Dentro da faixa fixada antes de
treinar (0,65 ± 0,03). Testes: de 41 para **51**, os 10 novos em `tests/test_modelagem.py`.
Notebook com 20 células de código executadas no kernel `tc-fase3` e 10 gráficos em
`images/modelagem/`.

| Modelo | ROC-AUC em CV | dp entre folds | s/fold |
|---|---:|---:|---:|
| **lightgbm_tunado — campeão** | **0,6611** | 0,0104 | 17 |
| random_forest | 0,6610 | 0,0102 | 89 |
| lightgbm_padrao | 0,6589 | 0,0098 | 11 |
| logistica_podada (8 colunas) | 0,6513 | 0,0113 | 2 |
| heuristica_taxa_municipal — **a barra** | 0,6337 | 0,0135 | 0,6 |
| dummy_prior | 0,5000 | — | — |

**Cinco decisões que contrariam o que este plano previa:**

1. **A barra da regra de uma variável é 0,6337, não 0,654.** Medida no mesmo
   `StratifiedGroupKFold` dos candidatos, ela perde 0,020 — o split agrupado cobra dela o
   mesmo que cobra do modelo. O ganho do campeão é **+0,0274, IC95 [+0,0116; +0,0433],
   p = 0,009**: real, significante e menor que a largura do IC do próprio campeão.
2. **A calibração isotônica foi rejeitada.** O plano previa aplicá-la "se o Brier indicar", e
   ele não indicou: ajustada em quatro folds e medida no quinto, ela **piora** o Brier de
   0,22233 para 0,22244. O LightGBM treinado com logloss já sai calibrado. O artefato
   serializado é o modelo cru, com a curva guardada desligada. Descoberta pelo caminho: a
   isotônica é monotônica *não-decrescente* e seus platôs custavam 0,0004 de AUC,
   penalizando sobretudo o baseline — `calibracao.desempatar` corrige isso.
3. **O limiar de máximo F1 alerta 74% da coorte** e não é utilizável. O limiar publicado é o
   de capacidade: corte nos 20% de maior risco do desenvolvimento, que alerta 13,0% do teste
   e captura 21,1% das crianças em risco com precisão 0,620 (1,63x a prevalência).
4. **Floresta e boosting empatam** (Δ −0,0002, p = 0,78). O campeão é o LightGBM por custo
   — 17 segundos por fold contra 89 — e por `TreeExplainer` exato, não por vencer.
5. **A busca de hiperparâmetros não entrega ganho significante**: +0,0022, IC95
   [−0,0011; +0,0055], p = 0,13, contra amplitude de 0,0076 entre as 40 configurações e
   desvio de 0,0276 entre folds da amostra de busca. Dois parâmetros pararam na borda da
   grade; a extensão até 2.500 árvores e `reg_lambda` 100 não achou nada melhor.

**O achado que reenquadra o projeto.** Embaralhar o alvo **dentro de cada município** —
o que preserva a taxa municipal e destrói todo o resto — derruba a AUC de 0,6667 para
apenas **0,6636**. Tudo o que o modelo sabe além da taxa média do município vale **0,0031**.
É um medidor de risco **territorial**, e não um identificador de crianças em risco dentro de
uma escola. Embaralhando o alvo globalmente o nulo fica em 0,5001 ± 0,0016, então o sinal é
inequívoco — a questão nunca foi se existe, foi de que grão ele é.

**Confirmado como previsto:** a `learning_curve` está achatada (de 979 para 3.309 municípios,
3,4x mais dado, a validação sobe 0,0039), o que confirma teto informacional e não amostral; o
`caderno` se comportou como controle negativo (queda de 0,00031 contra 0,04203 do controle
positivo); e no grão municipal, que é onde a decisão acontece, o R² é **0,624 na base completa
e 0,774 nos municípios com 200+ alunos** — acima do 0,42/0,65 que o plano estimava.

### O que a Etapa 5 recebe

- **`models/campeao.joblib`** com o modelo, os dois limiares, os nomes das **89 colunas**
  pós-`ColumnTransformer`, a seed, o MD5 do dataset e as versões de sklearn e lightgbm.
  `models/escores_oof_desenvolvimento.parquet` traz os escores out-of-fold para a Etapa 6.
- **O agregado esconde três populações.** Onde o lag vem da Gold (86,6% do teste) a AUC é
  0,6706; no agregado `rede 5`, 0,5846; no `rede 3`, 0,5598; sem histórico, 0,5227. A leitura
  de SHAP precisa ser feita também dentro de `fonte_lag_municipal = gold`.
- **AC e DF marcam 0,5173** e devem ser declarados não avaliados em qualquer ranking.
- **O modelo é pior onde o problema é maior**: AUC 0,6181 no quintil de menor presença
  municipal contra 0,7109 no de maior, somando-se ao viés otimista já medido na EDA.
  O Sudeste, quase metade do teste, é a região mais difícil (0,5908) porque é São Paulo
  dependendo do agregado coalescido.
- **A estrutura territorial se desloca entre edições**: o modelo reduzido a UF, região e rede
  cai de 0,6306 dentro de 2024 para 0,6099 quando treinado em 2023. O escore precisa de
  retreino a cada edição, e o ranking de 2024 descreve 2024 — projetar 2026 a partir dele
  assumiria uma estabilidade que a medição desmente.
- **Os dois controles negativos seguem valendo**, e o `caderno` tem valor de referência
  medido: qualquer coisa acima de 0,002 de queda por permutação é alarme de sobreajuste.

---

## Etapa 5 — Interpretabilidade `[x]`

`src/evaluation/interpret.py` · Notebook `04_interpretabilidade.ipynb` · **Evidência
completa em [`reports/interpretabilidade.md`](reports/interpretabilidade.md)** — aqui
ficam só as decisões que mudam a execução seguinte.

Três leituras independentes sobre o campeão em disco, sem retreino: SHAP exato com
`TreeExplainer` em 50 mil alunos sorteados do teste, `permutation_importance` com 10
repetições sobre os 330.836 alunos do teste inteiro, e os coeficientes da logística
podada reajustada no desenvolvimento com os cinco folds agrupados por município. O
`hash_dataset` do artefato é conferido contra o Parquet antes da primeira conta, e a
permutação em bloco por família foi construída porque a leitura coluna a coluna não
resolvia o problema que a EDA havia apontado.

### Resultado da Etapa 5

**O histórico municipal responde por 57,2% do |SHAP| e por uma queda de 0,0828 de
ROC-AUC quando a família é embaralhada inteira; o território acima do município, por
27,6% e 0,0462.** Juntos, 84,8% de tudo o que o modelo usa. Testes: de 51 para **63**,
os 12 novos em `tests/test_interpretabilidade.py`. Notebook com 13 células de código
executadas no kernel `tc-fase3` e 7 gráficos em `images/interpretabilidade/`.

| Família | Features | % do \|SHAP\| | Queda de ROC-AUC em bloco |
|---|---:|---:|---:|
| **histórico municipal** | 7 | **57,2%** | **0,0828 ± 0,0009** |
| territorial | 5 | 27,6% | 0,0462 ± 0,0006 |
| estrutural | 5 | 9,1% | 0,0033 ± 0,0002 |
| metas | 2 | 4,8% | 0,0030 ± 0,0002 |
| histórico escolar | 1 | 1,3% | 0,0001 — abaixo do piso |

**Quatro decisões e achados que contrariam ou refinam o que este plano previa:**

1. **O agrupamento por família não era precaução: o viés que ele corrige foi medido.**
   Somar as quedas das colunas isoladas da família municipal dá 0,0513, e embaralhar a
   família junta dá 0,0828 — 1,61 vez mais. Na territorial a razão é 1,90. Nas três
   famílias sem redundância interna ela fica entre 0,97 e 1,04. O ganho aparece
   exatamente onde a colinearidade está, o que é a prova que faltava.
2. **Só três features passam no critério de "fator", e apenas uma passa nas três
   leituras.** `mun_media_portugues_lag1` e `uf_media_portugues_lag1` estão fora do
   subconjunto podado e por isso são fatores de duas pontas; `mun_taxa_alfab_lag1` é a
   única no topo do SHAP, da permutação e da logística ao mesmo tempo. Outras seis
   entram como *achado sobre a leitura*, e `sigla_uf` é o caso-modelo: 2ª na permutação
   e 6ª no SHAP, porque as 26 dummies diluem uma coisa que a permutação destrói inteira.
3. **A leitura restrita à fatia Gold confirmou a global, e isso era falseável.** A
   Etapa 4 alertou que o agregado esconde três populações e exigiu refazer o SHAP dentro
   de `fonte_lag_municipal = gold`. Refeito, o Spearman entre as duas ordenações é 0,994,
   a primeira colocada é a mesma e as participações por família mudam menos de 2pp. O
   número agregado pode ser publicado sem a ressalva que se esperava precisar.
4. **Dez das vinte features não passam do piso de ruído do controle negativo.**
   `mun_n_escolas_lag1` é 11ª no SHAP e **última** na permutação, com queda de −0,00009:
   embaralhá-la melhora a métrica. É redundância com `mun_n_alunos_lag1`, e o par é
   candidato natural a poda se a Etapa 7 quiser um modelo mais enxuto.

**H1, H2 e H5 se sustentaram.** O contexto municipal lidera as duas leituras por
família (H1); a família de histórico escolar é a última e a única abaixo do piso (H2);
e `mun_taxa_presenca_lag1` é 5ª de 20 no SHAP, com queda de 0,0026 acima do piso e razão
de chances 0,89 na direção esperada (H5). Nenhuma das três caiu, o que merece a
desconfiança devida a resultado confortável: as três nasceram de medições univariadas
sobre a mesma base, então elas e o modelo compartilham a fonte do sinal. O que a etapa
acrescenta é que a ordenação sobrevive ao multivariado, à retirada de cada variável e ao
recorte de população.

**Sobre H2 vale a ressalva de que este é o teste fraco dela.** A família de histórico
escolar se resume a `tem_historico_escola`, porque o bloco `esc_*` saiu do modelo na
Etapa 3. A evidência forte continua sendo aquela ablação, de −0,0004 de AUC.

### O que a Etapa 6 recebe

- **A resposta pronta à pergunta 1**, em uma frase: o que mais impacta o risco de não
  alfabetização é o desempenho do próprio município no ano anterior, proficiência média
  em português acima de tudo e taxa de alfabetização em seguida, com o estado
  respondendo por outro quarto da explicação. E **a resposta à pergunta 5**, por grupo
  de variável, na tabela acima.
- **`mun_taxa_alfab_lag1` é a única variável com consenso nas três leituras**, com razão
  de chances 0,65 por desvio-padrão e desvio entre folds de 1,8% da magnitude. É a que
  pode ir ao slide sem ressalva de método.
- **A taxa municipal não ordena na parte baixa da distribuição.** A dependência mostra
  contribuição achatada em torno de zero abaixo de 65% de alfabetização, e só cai a
  partir dali. O ranking municipal da Etapa 6 herda essa limitação justamente nos
  municípios que mais interessam à política pública, e precisa dizer isso ao lado da
  lista.
- **A rede estadual aparece como fator protetor com OR 0,73**, concordante entre SHAP e
  logística. É associação e não efeito de política: redes estaduais estão concentradas
  em municípios de perfil diferente, e o modelo não separa a rede do lugar.
- **Importância não é causalidade, e falácia ecológica é o risco imediato.** Toda feature
  aqui é contextual. Elevar a média de português de um município não é intervenção; é o
  resultado que se quer, escrito de outra forma.
- **A ordenação descreve 2024.** Somada ao drift medido na Etapa 4, ela não sustenta
  projeção de fatores para 2026 sem reestimação.

---

## Etapa 6 — Aplicação estratégica `[x]`

`src/modeling/strategic.py` · Notebook `05_aplicacao_estrategica.ipynb` · **Evidência
completa em [`reports/aplicacao_estrategica.md`](reports/aplicacao_estrategica.md)** —
aqui ficam só as decisões que mudam a execução seguinte.

As 5 perguntas de negócio do enunciado:

- [x] **1. Quais fatores mais impactam a alfabetização?** — resultado da Etapa 5 traduzido para leitura executiva, com as ressalvas de causalidade, falácia ecológica e drift ao lado
- [x] **2. Quais municípios apresentam maior risco educacional?** — escore out-of-fold **nacional**, shrinkage empírico bayesiano nas duas quantidades candidatas, duas ordenações
- [x] **3. Quais regiões possuem padrões semelhantes?** — KMeans no grão município, `k` pela silhueta na grade de 3 a 8, presença como variável de agrupamento
- [x] **4. Como prever municípios que podem não atingir metas futuras?** — gap de esforço, projeção com incerteza e backtest, nessa ordem de confiabilidade
- [x] **5. Quais variáveis possuem maior influência nos modelos?** — por grupo de variável, com a leitura de quem decide financiar mais dado

### Resultado da Etapa 6

**As duas listas de cinquenta municípios não têm um único município em comum**, e o
decil superior por volume concentra **67,2%** das 837 mil crianças em risco do país.
O agrupamento devolveu três perfis, e o de maior risco reúne 2.142 municípios e
**66,9%** dessas crianças. Em **97,5% dos municípios a meta de 2025 cai dentro do
intervalo de 95% da própria projeção**. Testes: de 63 para **84**, os 21 novos em
`tests/test_estrategia.py`. Notebook com 15 células de código executadas no kernel
`tc-fase3` e 11 gráficos em `images/estrategia/`.

| Entrega | Número que a fecha |
|---|---|
| Escore out-of-fold nacional | 5.517 municípios, ROC-AUC 0,6607, Spearman 0,9881 com o artefato da Etapa 4 |
| Ranking por intensidade | topo 100% no Nordeste; 8,4 mil crianças, 1,0% do total |
| Ranking por volume | 242,7 mil crianças, 29,0% do total; São Paulo é o 1.766º por intensidade |
| Clusterização | `k = 3`, silhueta 0,2487; grupo de risco com 2.142 municípios e 66,9% das crianças |
| Gap de esforço 2025 | mediana +2,24 pp; 43,4% já superam a meta |
| Projeção 2025 | IC95 de 55,7 pp de largura; só 134 municípios ficam fora dele |
| Backtest 2023→2024 | ROC-AUC 0,5445; preditores ingênuos em 0,4876 e 0,4916 |

**Cinco decisões e achados que contrariam ou refinam o que este plano previa:**

1. **O artefato de escores da Etapa 4 não servia, e foi refeito.** Ele cobre 4.413 dos
   5.517 municípios, porque o teste tinha de permanecer intocado. Os cinco folds foram
   refeitos sobre a coorte inteira, com os hiperparâmetros e a seed do campeão, gerando
   `models/escores_oof_nacional.parquet`. O Spearman de **0,9881** nos 4.413 comuns é a
   checagem que autoriza a substituição. Fica declarado que os hiperparâmetros tiveram
   participação indireta do teste: é escore descritivo, não métrica publicada.
2. **O shrinkage sobre o escore do modelo é um não-evento — `k` de 0,046 aluno.**
   O plano o exigia para impedir que o topo fosse ocupado por municípios de oito alunos,
   e o medo estava certo: ordenado pela taxa crua, o topo teria **21 dos 50 com menos de
   50 alunos**, um deles com risco de 100,0%. Só que quem protege é o escore ser
   territorial, não a suavização aplicada a ele — o modelo dá quase a mesma
   probabilidade a todos os alunos do município, com desvio interno de mediana 0,0056, e
   não há erro amostral para encolher. Ordenado pelo escore, o topo tem **2**. A
   suavização que faz diferença é a da taxa observada, com `k` de **55,1 alunos** e
   deslocamento de até 39,4 pp; ela vai no CSV como `taxa_observada_ajustada`.
3. **A silhueta escolheu o menor `k` da grade, e todas as silhuetas são baixas** — 0,2487
   em `k = 3` contra 0,1740 em `k = 4`, sem cotovelo nítido de inércia. O espaço
   municipal é um gradiente de desempenho, não uma coleção de tipos separados. O `k = 5`
   foi inspecionado: o grupo extra se define pelo share de rede estadual no Sul, que é
   fato de rede e não padrão de alfabetização.
4. **A baixa presença não formou um perfil próprio.** O plano previa que sim. O decil de
   menor cobertura se distribui 67,3% no grupo de risco alto e 32,4% no intermediário; a
   presença entrou como gradiente colado ao desempenho, de 88,3% no grupo de risco a
   96,9% no consolidado. O terceiro grupo se chama "risco alto e cobertura menor", e não
   "com baixa participação", por causa disso.
5. **O limiar de porte para meta individual é 119 alunos, não "cerca de 50".** O ponto em
   que `a²/n = c²` no ajuste `sd(n) = √(a²/n + c²)` marca 119, e são **51,6% dos municípios
   com meta publicada e 52,3% da coorte inteira**. O `k` da mistura de credibilidade dá 55 pelo caminho
   independente. A faixa entre 55 e 119 é onde a avaliação individual começa a fazer
   sentido; abaixo de 55 ela não faz nenhum.

**Confirmado como previsto:** o backtest ingênuo reproduziu o diagnóstico quase exato —
0,4876 ordenando pela taxa de 2023 invertida e 0,4916 pelo gap até a meta, contra os
0,485 e 0,488 medidos na Etapa 0, agora sobre a base de metas do INEP; a volatilidade
cai com o porte de 24,19 pp abaixo de 25 alunos para 8,77 pp acima de mil; e a
reconciliação entre a taxa publicada pelo INEP e a ponderada do projeto fica em
MAE 0,81 pp.

**O achado que a política pública leva.** A pergunta 4 devolveu a medida da própria
imprevisibilidade, e ela é o produto: as metas são individualizadas a partir da taxa
base do município, com `corr(meta_2025, taxa_2023) = 0,9765`, então a distância até a
meta carrega pouca informação estrutural. O intervalo de 95% da projeção tem largura
mediana de **55,7 pp** e engole a meta em 5.218 dos 5.352 municípios. Municípios abaixo
de 119 alunos avaliados não deveriam ter metas avaliadas individualmente sem intervalo
publicado ao lado; a unidade de avaliação precisa ser plurianual ou agrupada.

### O que a Etapa 7 recebe

- **Três CSV prontos para publicação**, cada um com as colunas de ressalva ao lado das de
  resultado: `reports/ranking_risco_municipal.csv` (5.517 linhas, com `modelo_avaliado`,
  `ordenacao_fragil`, `taxa_presenca_2024` e `n_alunos_avaliados`),
  `reports/clusters_municipais.csv` (5.461) e `reports/projecao_metas_municipios.csv`
  (5.352, com `meta_dentro_do_intervalo` e `meta_avaliavel_individualmente`).
  *Duas dessas colunas foram renomeadas na revisão, porque os nomes afirmavam mais do que a
  medição sustentava: `ordenacao_fragil` virou `faixa_shap_taxa_municipal` e
  `meta_avaliavel_individualmente` virou `porte_abaixo_referencia_dispersao`. Quem procurar os
  nomes antigos nos CSV não os encontrará; ver o guia de execução.*
- **Quatro limitações novas para a lista do README**, além das dezesseis já declaradas:
  (17) o ranking ordena bem entre decis e mal dentro do decil superior, porque **57,8%
  dos municípios** estão na faixa em que `mun_taxa_alfab_lag1` não discrimina e **os 50 do
  topo estão todos nela**; (18) **AC e DF não são avaliados** e por isso não recebem
  posição, 23 municípios; (19) o escore nacional usa hiperparâmetros com participação
  indireta do teste, é descritivo e não substitui os 0,6599 da Etapa 4; (20) **165
  municípios da coorte não têm meta publicada** e ficam fora do CSV de projeção, com porte
  mediano de 50 alunos, concentrados em RS, SC e MG.
- **A limitação nº 5 do README ganha número novo.** O modelo regride para a média e
  subestima o risco onde ele é maior: 43,4% previstos contra 45,4% observados no quintil
  de menor presença, e 30,2% contra 27,9% no de maior. O viés de seleção da prova e o do
  modelo apontam no mesmo sentido.
- **Para o vídeo**, três números que cabem em trinta segundos: as duas listas de cinquenta
  municípios não têm nenhum em comum; um grupo de 2.142 municípios concentra dois terços
  das crianças em risco do país; e em 97,5% dos municípios a meta de 2025 está dentro da
  margem de erro da própria projeção.
- **O que não foi feito, e fica declarado:** não há intervalo de confiança em torno do
  risco de cada município, nem validação temporal do ranking — com duas edições da prova
  não existe um 2025 contra o qual conferir a ordenação de 2024.

---

## Etapa 7 — Documentação e entregáveis `[~]`

Passada de entregável: os três documentos que vão para avaliação foram escritos a partir da
evidência das seis etapas anteriores, sem recalcular nada e sem citar número que não esteja
num artefato em disco.

### Resultado da Etapa 7

**Os três documentos estão escritos e a verificação técnica de reprodutibilidade passou.**
Falta um item, e ele é decisão do grupo, não tarefa técnica: o repositório Git não contém o
projeto.

| Entregável | Estado |
|---|---|
| `README.md` — as 11 seções exigidas pelo enunciado | escrito, sem nenhum "Em construção" |
| `reports/documentacao_tecnica.md` | escrito, 12 seções |
| `reports/roteiro_video.md` | escrito, cronometrado, com divisão sugerida entre os cinco |
| `reports/auditoria_camada_gold.md` | já entregue na Etapa 2.5 |
| Revisão de reprodutibilidade | executada — ver abaixo |

**O README declara 20 limitações**, as 16 que o plano vinha acumulando mais as 4 que a Etapa 6
acrescentou, agrupadas em quatro blocos: sobre a base, sobre o que a base não permite medir,
sobre o desempenho e sobre os produtos da camada estratégica. As três que limitam o uso dos
CSV publicados (ordenação frágil dentro do decil superior, AC e DF não avaliados, ausência de
intervalo de confiança no ranking) estão no bloco final, e não diluídas no meio da lista.

**A documentação técnica registra os nove experimentos descartados** com o número que motivou
cada descarte, e uma seção própria para as três previsões do plano que a medição negou. Um
experimento negativo bem medido informa tanto quanto um positivo, e é o que separa o registro
analítico de um relatório de resultados.

**O roteiro do vídeo assume o enquadramento honesto no bloco 3**, onde diz que o modelo acerta
pouco no grão da criança e explica por quê, antes de apresentar os produtos municipais. A
ordem importa: a recomendação final sobre metas de municípios pequenos só se sustenta se a
plateia já souber o que o modelo é e o que ele não é.

### Verificação de reprodutibilidade — o que foi conferido

| Checagem | Resultado |
|---|---|
| Suíte de testes | **84 passando** em 100 s *(94 em 42 s após a revisão)* |
| Entry points do pipeline | os 5 módulos `-m` e os 2 scripts têm `__main__` |
| Importação do pacote | 18 módulos de `src/` importam sem erro |
| Links do README | 21 caminhos internos, todos existem em disco |
| Contagem de figuras | 55 PNG em `images/`, distribuídas em 5 subdiretórios |
| Números do README contra artefatos | conferidos contra `reports/metrics/campeao.json` e `cv_modelos.csv` |

Duas correções saíram da conferência. O R² municipal que este plano registrava como 0,625 e
0,777 é **0,624 e 0,774** em `campeao.json`; os documentos de entrega usam os valores do
artefato. E a contagem de figuras, estimada em 56, é 55.

### Revisão de consistência do repositório

Passada de revisão sobre notebooks, docstrings e nomenclatura, com o critério de que o
repositório é lido por um avaliador que nunca viu o projeto.

**Voz.** O notebook 01 falava em primeira pessoa do singular — "descarto", "mantenho",
"verifiquei", "eu preciso provar" — enquanto os notebooks 02 a 05 já usavam voz impessoal.
Ele foi escrito antes da passada de voz da Etapa 3 e ficou para trás. Nove trechos
reescritos, mais quatro docstrings de `src/eda.py` e uma de `montar_agregados_lag`. O
projeto inteiro passa agora no mesmo scan.

**Dois erros de narrativa, ambos contra a saída da própria célula.** O notebook 03 afirmava
R² municipal de 0,625 e 0,777 enquanto a célula duas posições acima imprime 0,6237 e
0,7743; a divergência havia se propagado para `reports/modelagem.md`, para este plano e
para a docstring de `strategic.py`. E a síntese do notebook 01 citava 46,9% de não
atingimento da meta de 2024, contra os **45,7%** que a própria execução imprime. Os dois
foram corrigidos na origem e em todos os lugares para onde tinham viajado.

**Nomenclatura.** `src/visualization/plots.py` era o único dos quatro módulos de gráfico sem
sufixo de etapa, e a docstring dele afirmava cobrir "as etapas de engenharia de features em
diante" — falso, já que cada etapa seguinte tem o seu módulo. Renomeado para
`plots_features.py`, com a docstring corrigida. O notebook 02 foi reexecutado, e as quatro
figuras saíram **byte a byte idênticas**, o que confirma que a renomeação não tocou em
resultado.

**O que foi examinado e não precisou de mudança.** Os 33 módulos têm docstring de módulo
substancial, e elas explicam o porquê da decisão em vez do que a função faz — que é o
padrão exigido. A cobertura de docstring em função pública é de 82,7%, e as 36 ausências se
concentram em `plotar_*` triviais, em `main()` de script e nos métodos de API do sklearn
(`fit`, `predict_proba`). Os notebooks 02 a 05 estão limpos de ponta a ponta.

**Fica registrado sem correção:** `src/eda.py` é dono da paleta de cores e de 19 funções
`plotar_*`, e os três módulos de `src/visualization/` importam as constantes de lá. A
dependência aponta na direção contrária à que a estrutura sugere. Consertar exigiria mover a
identidade visual para um módulo próprio e reexecutar o notebook 01, que é o mais caro do
projeto, sem mudar nenhum resultado — custo que não se justifica com a entrega em cima.

### O bloqueio que sobra, e que é decisão do grupo

**O repositório Git não contém o projeto.** São 32 arquivos rastreados — os da Etapa 6, mais
`README.md`, `PLANO_EXECUCAO.md` e `LICENSE` — contra 69 não rastreados que incluem
`src/config.py`, `requirements.txt`, `.gitignore`, `tests/` inteiro, os notebooks 01 a 04 e os
relatórios das Etapas 2.5 a 5.

O efeito prático é que `HEAD` tem `src/modeling/strategic.py` sem o `src/config.py` de que ele
depende. Um clone limpo não roda, e dois entregáveis explícitos do enunciado — "repositório Git
completo" e "pipeline reproduzível" — não existem enquanto isso não for resolvido.

Não é falha de execução: as Etapas 0 a 5 foram desenvolvidas sem commit, e a Etapa 6 commitou
apenas o que ela mesma produziu, para não reivindicar autoria do trabalho anterior. A saída é um
commit em bloco das etapas anteriores, e quem decide como atribuí-lo é o grupo.

Fica pendente também a identidade Git do repositório, hoje não configurada.

### O que falta para a entrega fechar

1. **Commit em bloco das Etapas 0 a 5**, com a atribuição que o grupo decidir.
2. **Gravar o vídeo** a partir de `reports/roteiro_video.md`, e montar os slides.
3. **Decidir se `PLANO_EXECUCAO.md` entra na entrega.** Ele hoje é documento de trabalho, com
   checkbox e registro de execução. Se entrar, vira relatório narrativo; se não, o README e a
   documentação técnica já carregam tudo o que um avaliador precisa, e ele fica como histórico
   interno.
4. **Publicar a Gold** em Release ou Drive, ou declarar no README que a execução completa depende
   do CSV fornecido pelo grupo. Hoje o README declara a segunda opção.

---

## Etapa 8 — Revisão científica `[~]`

Retorno de *Evaluation* a *Business Understanding* e *Data Preparation/Modeling*, motivado por
uma revisão externa. O registro completo, com a tabela problema → correção → evidência, está em
[`reports/revisao_cientifica.md`](reports/revisao_cientifica.md); aqui fica só o que muda o estado
deste plano.

### O que foi corrigido

**Três correções mudam o que o projeto pode afirmar**, e nenhuma delas é ajuste cosmético:

1. **A seleção de atributos deixou de tocar a reserva.** O B5 histórico comparava conjuntos sobre
   a coorte inteira, então os municípios depois chamados de teste participaram de uma decisão de
   modelagem. O script agora separa a reserva com a seed fixa **antes** de qualquer seleção. Isso
   corrige o fluxo daqui para a frente e **não** restaura a independência já perdida: o artefato
   carrega `teste_independente_da_selecao: False`, e os 0,6599 passam a ser resultado exploratório
   retrospectivo. Recuperar avaliação confirmatória exige amostra não consultada.
2. **A projeção de metas virou cenário condicional avaliado fora do ajuste.** Prior, suavização,
   deriva, dispersão e baseline são aprendidos em cinco folds por município e aplicados fora do
   fold que os gerou. Continua sendo a mesma transição 2023 → 2024: é generalização territorial,
   não validação de ano futuro, e o nome `backtest` saiu do código, dos CSV e das figuras.
3. **A projeção passou a respeitar o domínio de uma taxa.** A versão gaussiana publicava 10 pontos
   acima de 100%, 176 limites inferiores negativos e 2.267 superiores acima de 100%. Com a
   distribuição de trabalho censurada em 0–100, são **zero** em cada uma das três contagens,
   conferido no CSV publicado.

**Quatro correções são de linguagem, e valem tanto quanto.** O corte de 119 alunos deixou de ser
regra de elegibilidade e virou referência de dispersão; a flag `ordenacao_fragil` virou
`faixa_shap_taxa_municipal`, porque o SHAP achatado de uma variável não julga a ordenação inteira;
a hipótese sobre histórico escolar ficou marcada como não verificável, já que a chave não permite
acompanhamento longitudinal; e o roteiro deixou de comparar esforço acumulado com ritmo anual — o
gap de 15,75 pp até 2030 anualiza em 2,625 pp, e a coluna `gap_mediano_anualizado_pp` agora está no
CSV do funil para que a comparação não precise ser refeita de cabeça.

### Verificação desta etapa

| Checagem | Resultado |
|---|---|
| Suíte de testes | **94 passando** em 42 s, com o aviso conhecido do SHAP |
| `tests/test_revisao.py` | 9 casos novos, que travam as correções contra regressão |
| Notebooks 01 a 05 | reexecutados por `scripts/executar_notebooks.py`, sem erro |
| Figuras | 55 PNG regeneradas; a órfã `10_backtest_metas.png` foi removida |
| Orquestrador | `--etapa dados` e `--etapa relatorios` executados de ponta a ponta |
| Insumos | 7 fontes conferidas por SHA-256 contra o manifesto |
| Domínio da projeção | 0 taxas fora de 0–100 e 0 limites impossíveis em 5.352 linhas |

`--etapa modelos` **não** foi reexecutado, e isso é decisão, não pendência esquecida: ele retreina
o campeão e substituiria o artefato cujas métricas estão publicadas e conferidas por hash. O
objetivo da revisão era corrigir o protocolo e a linguagem sem trocar o objeto analisado.

### O que continua aberto, e de quem depende

| Pendência | De quem depende | Critério de fechamento |
|---|---|---|
| Confirmação prospectiva | amostra não consultada, fora do grupo | métricas em dados que nenhuma decisão tocou |
| Dimensão socioeconômica | grupo e professor | fonte integrada e auditada, ou reformulação aceita |
| Disponibilidade temporal das fontes | calendário de publicação do INEP | data real de publicação por atributo, no contrato temporal |
| Entrega dos dados à banca | grupo | as sete fontes acessíveis a quem clonar |
| Vídeo de até cinco minutos | grupo | link acessível e duração verificada |
| Revisão por outro integrante | grupo | PR real, revisado por quem não escreveu |

Nenhuma delas se fecha escrevendo texto, e por isso nenhuma foi marcada como resolvida.

### Seções do README, como entregues

Contexto do problema · Objetivo analítico · Descrição da base utilizada, com dicionário de
features · Estrutura do repositório · Etapas de modelagem, com a matriz anti-leakage resumida ·
Escolha do algoritmo, com a tabela comparativa dos 7 candidatos · Métricas de avaliação, com IC
por bootstrap de municípios e os dois limiares · Interpretação dos resultados, por família ·
Insights encontrados, 8 · Limitações, 20 · Aplicação prática para políticas públicas, com os 3
produtos e uma seção de "como **não** usar" · Possíveis evoluções futuras, 6 · Como reproduzir ·
Entregáveis.

O desvio consciente em relação ao enunciado está declarado no próprio README: ele pede branches
e pull requests, e o grupo decidiu em 2026-09-02 trabalhar em uma branch só.

---

## Git workflow

**Decisão do grupo (2026-09-02): tudo na branch atual, sem branch por etapa.** O plano
previa sete branches com PR; ficou em uma só. O histórico de decisões analíticas passa a
depender inteiramente das mensagens de commit — commits pequenos e descritivos, um por
bloco de decisão, já que não haverá PRs para contar essa história.

---

## Verificação final `[ ]`

Tudo deve rodar do zero num clone limpo:

```bash
python -m venv .venv && source .venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name tc-fase3      # kernel do venv, exigido pelos notebooks
python scripts/prepare_data.py             # CSV para Parquet em data/interim/
python scripts/build_dim_municipio.py      # dimensão territorial em data/reference/
python -m src.preprocessing.build_dataset  # gera data/processed/dataset_2024.parquet
python -m src.modeling.train comparacao    # CV dos 7 candidatos -> reports/metrics/cv_modelos.csv
python -m src.modeling.train tuning        # 40 configuracoes -> hiperparametros_campeao.json
python -m src.modeling.train campeao       # refit, teste, models/campeao.joblib
python -m src.evaluation.rigor             # permutacao, curva, LOGO, drift, invariancia
python -m src.evaluation.interpret         # SHAP, permutacao, familias -> interpret_*
python -m src.modeling.strategic           # escore OOF nacional + os 3 CSVs de reports/
pytest -q                                  # asserções de dados, vazamento e reprodutibilidade
```

**Dados** — 24 testes passando em `tests/test_dados.py` (eram 14 antes da Etapa 2.5)

- [x] Reconciliação Gold × microdado **por conjunto de chaves e igualdade do alvo**, nos dois anos
- [x] `alfabetizado` continua sendo o corte determinístico em 743,0 nos dois anos
- [x] `id_aluno` não é chave longitudinal: menos de 1% dos IDs repetidos caem na mesma escola
- [x] `dim_municipio` é um para um, com 26 UFs
- [x] `rede = 5` do agregado reproduz a taxa das redes 2+3 do microdado **nos dois anos**, com limiar por ano
- [x] A taxa ponderada por `peso_aluno` reconcilia com o INEP a menos de 0,10pp
- [x] A coalescência `5 → 3` recupera 8,86pp da coorte e deixa gap residual de 1,91%
- [x] O gap de lag é SP, DF e AC inteiros — travado contra reinterpretação como ruído difuso
- [x] `percentual_participacao_*` é a presença do próprio ano e está em `COLS_PROIBIDAS`
- [x] As 249 e 936 linhas de alvo imputado por ausência de medida estão identificadas
- [x] O dataset final não tem nenhuma coluna com correlação acima de 0,95 com o alvo (a maior medida na EDA foi 0,272)
- [x] `set(X.columns) & set(config.COLS_PROIBIDAS)` é vazio no dataset final
- [x] As 936 linhas de `preenchimento_caderno = 0` foram removidas do treino
- [x] Toda métrica reportada é estratificada por `tem_historico_municipio`

**Modelo**

- [x] O campeão em disco reproduz as métricas de `reports/metrics/campeao.json` a menos de 1e-9, com asserção em `tests/test_modelagem.py`
- [x] Nenhum `id_municipio` aparece em treino e teste ao mesmo tempo, com asserção em `split.py`
- [x] Nenhum join por `id_aluno` entre anos em lugar nenhum do código (armadilha A2)
- [x] Teste de invariância do `caderno`: importância baixa, e se subir é alarme de sobreajuste
- [x] ROC-AUC do campeão no teste entre 0,63 e 0,70. **Acima de 0,80 investigar vazamento antes de comemorar**
- [x] O campeão supera o baseline heurístico — que no mesmo desenho de validação vale 0,6337, e não os 0,654 medidos noutro esquema. Supera por 0,0274, e isso está no relatório, não escondido
- [x] A `learning_curve` está achatada no fim, o que confirma teto informacional e não amostral

**Entrega**

- [x] Os 5 notebooks executam de ponta a ponta, sem erro, por `scripts/executar_notebooks.py`
- [x] Nenhum notebook depende do Python do sistema, nem de kernel registrado com `--user`
- [x] O README declara as limitações, sem nenhuma seção "Em construção"

---

## Riscos e mitigações

| Risco | Mitigação |
|---|---|
| ~~`lightgbm`/`shap` sem wheel para Python 3.14~~ | **Resolvido** — wheels verificadas para os três pacotes no 3.14 (ver ressalva 2) |
| Instalação fora de venv contaminando o Python do sistema | `.venv/` criado como primeira tarefa da Etapa 1, antes de qualquer `pip install` |
| 615 MB carregados com só ~5 GB de RAM livres | Dtypes otimizados + Parquet + leitura por colunas; medido: 16 GB totais, 16 CPUs |
| AUC "baixo" ser lido como projeto fraco | Enquadrar desde o README: o modelo mede risco estrutural, o teto é propriedade dos dados, e o valor está na camada estratégica |
| Projeção de metas com só 2 pontos no tempo | Declarada como cenário de alerta, não previsão calibrada |
| Escopo grande para o prazo | A ordem das etapas é a ordem de prioridade: 1 a 4 entregam o mínimo exigido, 5 a 7 agregam valor |
| Notebook rodar fora do venv sem ninguém notar | Kernel `tc-fase3` registrado e fixado no `kernelspec` de cada notebook; a Etapa 2 já pegou esse erro uma vez |
| ~~Interpretar importância de feature colineares como achado~~ | **Resolvido** — a leitura por família foi feita e o viés que ela corrige foi medido: embaralhar a família municipal junta derruba 1,61 vez mais que a soma das colunas isoladas, e 1,90 vez na territorial |
| Origem da Gold num clone limpo | **Em aberto.** O CSV tem 615 MB, é gitignored, e o ETL que o gera está em `scripts/etl/` mas depende de um workspace Databricks. Decidir entre publicar o arquivo ou declarar a dependência no README |
