# Modelagem supervisionada — evidência da Etapa 4

> **Revisão de validade (2026-09-05):** estas medições do classificador são históricas e exploratórias. A seleção supervisionada consultou a coorte inteira; a reserva não é independente dessa seleção. O código corrigido e o protocolo atual estão em [revisao_cientifica.md](revisao_cientifica.md).


O `PLANO_EXECUCAO.md` guarda as decisões; este documento guarda a medição que as
sustenta, no mesmo formato da auditoria da Etapa 2.5 e do relatório da Etapa 3.
Tudo aqui foi medido sobre o dataset de 1.851.852 alunos de 2024, com
`RANDOM_STATE = 42`, `scikit-learn 1.8.0` e `lightgbm 4.7.0`.

Fase CRISP-DM 4, *Modeling*. Entregáveis: desenho de teste, modelos treinados com
parâmetros e seeds, avaliação técnica comparativa e justificativa da escolha.

**Resumo em cinco linhas.** O campeão é um LightGBM com 20 features territoriais e
marca **ROC-AUC 0,6599 [0,6314; 0,6853]** no conjunto de teste, contra 0,6337 da
regra de uma variável medida no mesmo desenho de validação. O ganho de 0,027 é
real e pequeno, e é menor que a largura do próprio intervalo de confiança. A
calibração isotônica foi testada e **rejeitada** por não melhorar o Brier fora do
próprio ajuste. No grão municipal, que é onde a decisão de política acontece, o
mesmo modelo explica **R² 0,78** da taxa observada nos municípios com 200 ou mais
alunos avaliados.

---

## 1. O desenho de validação, e o que ele custa

```
1.851.852 alunos de 2024
  ├── desenvolvimento  1.521.016 alunos · 4.413 municípios · prevalência 0,4067
  │     └── StratifiedGroupKFold(5) por município — toda seleção acontece aqui
  └── teste              330.836 alunos · 1.104 municípios · prevalência 0,3812
        └── avaliado retrospectivamente; seleção anterior consultou esta reserva
```

Agrupar por município não é precaução, é a única leitura honesta desta base. As
features que carregam sinal são todas municipais, então num split aleatório os
alunos de um mesmo município cairiam dos dois lados e o modelo reencontraria no
teste um nível que já viu no treino. O diagnóstico mediu o tamanho dessa
memorização: **0,667 no split aleatório contra 0,649 no agrupado**, 0,018 de AUC
que o desenho honesto abre mão de contabilizar.

**A partição de municípios não é a partição de alunos.** Vinte por cento dos
municípios viraram 17,9% das linhas, e a prevalência do risco ficou em 0,4067 no
desenvolvimento contra **0,3812 no teste** — 2,5pp de diferença. Não é defeito do
sorteio: é a heterogeneidade territorial que o desenho existe para expor, já que os
1.104 municípios sorteados são, por acaso, um pouco melhores que a média nacional.
A consequência é operacional: toda métrica de teste é lida contra a prevalência do
teste, e um PR-AUC de 0,54 significa coisas diferentes contra 0,40 e contra 0,38.

---

## 2. A barra real: 0,6337, não 0,654

O diagnóstico media ROC-AUC 0,654 para "ranquear os alunos pela taxa de
alfabetização de 2023 do seu município", e a Etapa 3 registrou a ressalva de que
aquele número vinha de outro esquema de validação. Reimplementada como estimador
`sklearn` e passada pelos **mesmos cinco folds** dos candidatos, a mesma regra vale:

| Modelo | Papel | ROC-AUC | dp entre folds | PR-AUC | Brier | s/fold |
|---|---|---:|---:|---:|---:|---:|
| **lightgbm_tunado** | **campeão** | **0,6611** | 0,0104 | 0,5519 | 0,2223 | 17 |
| random_forest | não-linearidade sem boosting | 0,6610 | 0,0102 | 0,5521 | 0,2224 | 89 |
| lightgbm_padrao | hiperparâmetros de partida | 0,6589 | 0,0098 | 0,5492 | 0,2229 | 11 |
| logistica_podada | linear, 8 colunas, VIF < 3 | 0,6513 | 0,0113 | 0,5389 | 0,2246 | 2 |
| heuristica_proficiencia_municipal | regra de uma variável, contínua | 0,6388 | 0,0144 | 0,5255 | 0,2269 | 0,7 |
| heuristica_taxa_municipal | **a barra** | 0,6337 | 0,0135 | 0,5227 | 0,2278 | 0,6 |
| dummy_prior | piso absoluto | 0,5000 | 0,0000 | 0,4067 | 0,2413 | — |

A queda de 0,654 para 0,6337 é o que o split agrupado cobra da própria regra: ela
também estava sendo medida com municípios repetidos entre treino e teste.

### Comparações pareadas, com o t corrigido de Nadeau-Bengio

O t pareado comum é inválido em validação cruzada — os folds compartilham dados de
treino e a variância do delta sai subestimada. A correção infla a variância pelo
fator `1/n + 1/(k-1)`, que com `k = 5` vale `1/n + 0,25`.

| Base | Alternativo | Δ médio | IC95 | t | p |
|---|---|---:|---|---:|---:|
| dummy | regra de uma variável | +0,1337 | [+0,1085; +0,1588] | 14,74 | 0,0001 |
| regra de uma variável | logística podada | +0,0176 | [+0,0041; +0,0311] | 3,63 | 0,022 |
| logística podada | lightgbm padrão | +0,0076 | [+0,0021; +0,0131] | 3,86 | 0,018 |
| lightgbm padrão | lightgbm tunado | +0,0022 | [−0,0011; +0,0055] | 1,88 | **0,133** |
| lightgbm tunado | random forest | −0,0002 | [−0,0015; +0,0012] | −0,30 | **0,776** |
| **regra de uma variável** | **lightgbm tunado** | **+0,0274** | **[+0,0116; +0,0433]** | **4,80** | **0,009** |

**O ganho do modelo sobre a regra de uma variável é de 0,027, é significante e é
pequeno.** Vinte features de engenharia territorial, uma busca de hiperparâmetros
e um GBM compram menos de três centésimos de AUC sobre uma linha de SQL. Não é
falha de execução: a base não tem **nenhuma** variável sobre a criança — sem nível
socioeconômico, cor/raça, idade, frequência ou histórico escolar. O que 0,66 mede é
quanto o território determina o destino individual, e isso tem teto.

**Floresta e boosting empatam; o desempate é custo.** O delta pareado é de −0,0002
com IC cruzando o zero e p = 0,78. A floresta leva 89 segundos por fold contra 17
do boosting, e o LightGBM tem `TreeExplainer` exato para a Etapa 5. O campeão é o
LightGBM por custo e por ferramental, não por vencer — e isso vai declarado.

**A logística podada chega a 0,6513 com oito colunas.** A distância para o campeão,
0,0098, é da ordem do desvio entre folds. Um scorecard linear resolve quase todo o
problema, e é o caminho a considerar se a explicabilidade virar requisito duro.

---

## 3. A busca de hiperparâmetros cabe dentro do próprio ruído

Quarenta configurações sorteadas do espaço do plano, `StratifiedGroupKFold(5)`
sobre 400.061 alunos de 1.277 **municípios inteiros** — sortear alunos individuais
reintroduziria a memorização que o desenho evita (vetor 11 da matriz anti-leakage).

```
melhor configuração ....... ROC-AUC 0,6646   dp entre folds 0,0276
pior configuração ......... ROC-AUC 0,6570
amplitude da busca ........         0,0076   ← um quarto do desvio entre folds
```

A distância entre a melhor e a pior configuração é **um quarto** da oscilação entre
partições da mesma configuração. Remedido no desenvolvimento completo, o ganho
sobre os hiperparâmetros de partida é de +0,0022 com IC95 [−0,0011; +0,0055] e
p = 0,13. Os parâmetros da busca ficam por serem os melhores disponíveis, não por
terem provado superioridade.

**Dois parâmetros pararam na borda da grade, e a extensão não achou nada.** A busca
escolheu o maior `n_estimators` (883) e o maior `reg_lambda` (9,86) que o espaço
oferecia — sinal clássico de grade curta. Esticando:

| `n_estimators` \ `reg_lambda` | 9,86 | 30 | 100 |
|---|---:|---:|---:|
| 883 | 0,6646 | 0,6647 | 0,6647 |
| 1.500 | 0,6636 | 0,6638 | 0,6638 |
| 2.500 | 0,6627 | 0,6629 | 0,6628 |

Mais árvores piora e a regularização é plana até 100, com diferenças de 1e-4 contra
desvio de 0,028 entre folds. A borda era cosmética e o vencedor se sustenta.

**A forma do vencedor diz mais que o seu valor.** `max_depth = 5` com
`min_child_samples = 1.126` e `learning_rate = 0,020`: a busca convergiu para um
modelo raso e fortemente freado. É coerente com um sinal que é de grupo e não de
indivíduo — folhas pequenas aqui decoram território.

```json
{"n_estimators": 883, "num_leaves": 104, "max_depth": 5,
 "learning_rate": 0.0205, "min_child_samples": 1126, "subsample": 0.697,
 "colsample_bytree": 0.963, "reg_lambda": 9.86, "subsample_freq": 1}
```

`num_leaves = 104` não vincula: com `max_depth = 5` o teto real é 32 folhas.

---

## 4. A calibração foi testada e rejeitada

A calibração importa porque as probabilidades viram, na Etapa 6, o ranking
municipal de risco — `risco_municipal` é a média das probabilidades preditas dentro
do município. Um viés distribuído de forma desigual entre territórios reordena o
topo da lista, que é a parte que decide orçamento.

Ajustar a isotônica nos escores out-of-fold e medi-la **nos mesmos escores**
garantiria ganho por construção: a curva minimiza exatamente esse erro. Ajustando
em quatro folds e medindo no quinto:

```
Brier cru ......................... 0,22233
Brier com isotônica fora do ajuste  0,22244   ← pior em 1,1e-4
```

O LightGBM treinado com logloss numa base de prevalência 0,40 já sai calibrado, e a
etapa só acrescenta ruído. **O artefato serializado é o modelo cru**; a curva
isotônica fica gravada dentro dele, desligada, e o critério de decisão está no
código para quem quiser reabrir.

### Um detalhe que custava AUC em silêncio

A isotônica é monotônica **não-decrescente**, não crescente: ela achata faixas
inteiras de escore num mesmo valor, e cada platô vira um bloco de empates que a ROC
conta como meio acerto. Medido num modelo linear sobre 120 mil alunos, o
achatamento tirava 0,0004 de ROC-AUC — pouco, mas na direção errada, e afetando
sobretudo o baseline de uma variável, cuja coluna tem menos valores distintos.
`calibracao.desempatar` devolve a ordem original dentro de cada platô com uma
perturbação de no máximo 2e-9, invisível na probabilidade e exata no ranking. Sem
ela, a comparação com a barra sairia enviesada a favor do campeão.

---

## 5. O limiar: dois, e nenhum deles é 0,5

Ambos escolhidos nos escores out-of-fold do desenvolvimento. A reserva não participa da escolha dos limiares neste módulo; participou da seleção histórica de atributos.

| Critério | Limiar | Taxa de alerta | Recall | Precisão | F1 |
|---|---:|---:|---:|---:|---:|
| máximo F1 | 0,297 | 74,1% | 0,852 | 0,438 | 0,579 |
| **capacidade de 20%** | **0,525** | **13,0%** | **0,211** | **0,620** | 0,314 |

**O limiar de máximo F1 é matematicamente correto e operacionalmente inútil.** Com
prevalência de 0,38 e ranking fraco, maximizar F1 empurra o corte para baixo até
alertar quase toda a coorte — 74% de alerta não é política pública, é constatação.

O limiar publicado é o de capacidade. Cortando nos 20% de maior risco do
desenvolvimento, ele alerta 13,0% do teste e captura 21,1% das crianças em risco
com precisão de 0,620, ou **1,63 vez a prevalência**. A curva inteira vai junto para
que quem tem o orçamento escolha outro ponto:

| Fração atendida | Cobertura das crianças em risco | Precisão | Lift |
|---:|---:|---:|---:|
| 5% | 8,9% | 0,681 | 1,79 |
| 10% | 16,8% | 0,641 | 1,68 |
| 20% | 30,6% | 0,584 | 1,53 |
| 30% | 42,8% | 0,544 | 1,43 |
| 50% | 63,0% | 0,481 | 1,26 |

---

## 6. Avaliação da reserva retrospectiva

| Métrica | Valor | IC95 (bootstrap de 1.000 reamostras de municípios) |
|---|---:|---|
| **ROC-AUC** | **0,6599** | **[0,6314; 0,6853]** |
| PR-AUC | 0,5360 | [0,5056; 0,5672] |
| Brier | 0,2179 | [0,2118; 0,2237] |
| KS | 0,2218 | — |
| prevalência do teste | 0,3812 | — |

O intervalo reamostra **municípios**, não alunos. Reamostrar alunos trataria as
2.700 crianças de uma cidade como 2.700 observações independentes e devolveria um
intervalo várias vezes mais estreito — o mesmo erro do split aleatório, com outra
roupa; um teste em `tests/test_modelagem.py` trava essa diferença.

O ponto está dentro da faixa fixada antes de treinar, 0,65 ± 0,03. E o dado
desconfortável, que vai no relatório em vez de ficar na nota de rodapé: **a largura
do intervalo, 0,054, é o dobro do ganho do modelo sobre a regra de uma variável.**

### Concentração de risco por decil

| Decil de risco | Taxa observada | Lift |
|---|---:|---:|
| 1 (maior risco) | 64,1% | 1,68 |
| 2 | 52,7% | 1,38 |
| 3 | 46,3% | 1,22 |
| 5 | 35,7% | 0,94 |
| 6 | 38,7% | 1,02 |
| 10 (menor risco) | 15,0% | 0,39 |

As pontas funcionam — razão de 4,3 vezes entre o primeiro e o último decil. **No
meio da fila a ordenação falha**: o decil 6 tem taxa maior que o 5. É onde os
escores se acumulam e onde a informação municipal não distingue mais nada.

---

## 7. A estratificação obrigatória — o agregado esconde três populações

A Etapa 3 deixou isto como contrato, e a medição confirma que era necessário.

### Por fonte do lag municipal (B4)

| Fonte | n | Prevalência | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|
| Gold 2023 (microdado) | 286.358 | 0,3776 | **0,6706** | 0,5436 | 0,2148 |
| agregado INEP, rede 5 | 19.727 | 0,4228 | 0,5846 | 0,4863 | 0,2389 |
| agregado INEP, rede 3 | 23.522 | 0,3876 | 0,5598 | 0,4387 | 0,2363 |
| sem histórico | 1.229 | 0,4516 | 0,5227 | 0,4739 | 0,2501 |

**A coalescência `5 → 3` comprou cobertura pagando em sinal.** Ela levou a
cobertura de nível de 76,9% para 98,1% da coorte, e é o que faz São Paulo ter
feature; o preço é que os 13,4% de alunos resgatados entram no modelo com sinal
quase três vezes mais fraco. Foi um bom negócio, e continua sendo — mas o 0,6599
agregado não vale igualmente para todos, e nenhum relatório pode apresentá-lo como
se valesse.

### Por cobertura de lag (B3) e por lag de UF

| Estrato | n | ROC-AUC |
|---|---:|---:|
| `tem_historico_municipio = 1` | 286.358 | 0,6706 |
| `tem_historico_municipio = 0` (SP, DF, AC) | 44.478 | 0,5718 |
| tem lag de UF | 329.636 | 0,6602 |
| **AC e DF, sem lag de UF em fonte nenhuma** | **1.200** | **0,5173** |

**Acre e Distrito Federal são ruído.** Nenhuma fonte de 2023 os cobre e o modelo não
tem o que dizer sobre eles. Qualquer ranking da Etapa 6 precisa marcá-los como não
avaliados, não posicioná-los.

### Por quintil de taxa de presença municipal em 2023

| Quintil | n | Prevalência | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
| q1, menor presença | 83.583 | 0,4234 | 0,6181 | 0,2334 |
| q2 | 31.475 | 0,4447 | 0,6437 | 0,2319 |
| q3 | 57.001 | 0,4085 | 0,6647 | 0,2215 |
| q4 | 57.054 | 0,3255 | 0,6645 | 0,2030 |
| q5, maior presença | 57.245 | 0,2948 | **0,7109** | 0,1836 |

**O modelo é pior exatamente onde o problema é maior.** A EDA já havia medido que a
taxa observada é otimista onde a presença é baixa — 53,4% de alfabetização no
quintil de menor presença contra 74,6% no de maior. Agora se soma que o modelo
também erra mais ali: 0,6181 contra 0,7109. Os dois vieses apontam para o mesmo
lado, subestimar o problema onde ele é pior, e o ranking municipal precisa carregar
a taxa de presença ao lado do escore.

### Por região e por rede

| Região | n | Prevalência | ROC-AUC |
|---|---:|---:|---:|
| Nordeste | 76.237 | 0,4264 | 0,7178 |
| Centro-Oeste | 23.365 | 0,3043 | 0,6855 |
| Sul | 44.643 | 0,3702 | 0,6725 |
| Norte | 31.789 | 0,4898 | 0,6172 |
| **Sudeste** | **154.802** | 0,3515 | **0,5908** |

**O Sudeste é a região mais difícil e a maior**, quase metade do teste. Não é
coincidência: São Paulo é a maior parte do Sudeste e é justamente quem depende do
agregado coalescido em vez do microdado. As duas estratificações contam a mesma
história por caminhos diferentes.

A rede quase não separa: 0,6606 na Municipal contra 0,6450 na Estadual.

---

## 8. O grão municipal — onde o modelo é bom

A decisão de política não é sobre uma criança, é sobre um município. E a agregação
cancela o ruído individual:

| Porte mínimo | Municípios | R² | Correlação | Viés médio |
|---|---:|---:|---:|---:|
| todos | 1.104 | 0,624 | 0,790 | −0,4pp |
| ≥ 50 alunos | 865 | 0,709 | 0,842 | −0,1pp |
| ≥ 200 alunos | 355 | **0,774** | 0,881 | −0,3pp |

Isto responde ao desconforto da seção 2. A AUC de 0,66 no aluno e o R² de 0,78 no
município não se contradizem: são a mesma informação em dois grãos. **O território
explica bem o território e explica mal a criança** — o que é, em si, o achado de
política pública desta entrega, e o que sustenta a camada estratégica da Etapa 6.

O viés médio negativo em todas as faixas significa que o modelo **subestima**
levemente o risco. É pequeno, mas soma-se ao viés de seleção dos ausentes na mesma
direção.

---

## 9. As cinco provas de rigor

### 9.1 Permutação — e o achado mais direto da etapa

Duas distribuições nulas, porque são duas perguntas. `StratifiedGroupKFold(3)`
sobre 300.703 alunos de municípios inteiros, com o LightGBM do campeão.

| Nulo | Permutações | AUC real | Nulo médio | dp | Máximo do nulo | p |
|---|---:|---:|---:|---:|---:|---:|
| alvo embaralhado globalmente | 50 | 0,6667 | 0,5001 | 0,0016 | 0,5036 | 0,020 |
| **alvo embaralhado dentro de cada município** | 20 | 0,6667 | **0,6636** | 0,0003 | 0,6641 | 0,048 |

A primeira linha responde "existe sinal?" e a resposta é sim, sem ambiguidade: a
distribuição nula fica colada em 0,500 com desvio de 0,0016, e o modelo real está
a mais de cem desvios dela. O p de 0,020 é o menor valor observável com 50
permutações, não uma medida da força do efeito.

A segunda linha é a que informa. Embaralhar o alvo **dentro** de cada município
preserva a taxa municipal exata e destrói todo o resto — a relação entre a criança
individual e qualquer coisa que não seja o seu município. Se o modelo usasse algo
além do nível territorial, a AUC cairia. Ela cai de **0,6667 para 0,6636**.

**Tudo o que o modelo sabe além da taxa média do município vale 0,0031 de ROC-AUC.**
O efeito é detectável — o valor real supera as vinte permutações e o desvio do nulo
é de apenas 0,0003 — e é irrelevante para qualquer decisão. Vinte features, uma
busca de hiperparâmetros e 883 árvores estão, em 99,5% da sua capacidade
discriminante, reproduzindo uma média por município.

Isso não invalida o modelo: reforça o enquadramento. O objeto que esta base permite
construir é um **medidor de risco territorial**, e a Etapa 6 está certa em levar o
resultado para o grão municipal. O que ele não é, e não pode ser vendido como, é um
identificador de crianças em risco dentro de uma mesma escola ou de um mesmo
município.

### 9.2 Curva de aprendizado — o teto é informacional

A abscissa é **município**, não aluno: dez mil alunos a mais do mesmo município não
acrescentam quase nada, cem municípios novos acrescentam. Três seeds por ponto.

| Municípios no treino | Alunos | AUC treino | AUC validação | dp entre seeds |
|---:|---:|---:|---:|---:|
| 79 | 25 mil | 0,6535 | 0,6493 | 0,0103 |
| 204 | 58 mil | 0,6722 | 0,6560 | 0,0035 |
| 409 | 118 mil | 0,6883 | 0,6651 | 0,0033 |
| 979 | 289 mil | 0,6803 | 0,6687 | 0,0015 |
| 1.735 | 601 mil | 0,6727 | 0,6715 | 0,0007 |
| 3.309 | 1,16 mi | 0,6700 | 0,6726 | — |

**De 979 para 3.309 municípios — 3,4 vezes mais dado — a validação sobe 0,0039.**
A curva está achatada, e a extrapolação é direta: dobrar a base compraria algo em
torno de 0,002 de AUC, contra os 0,027 que separam o modelo da regra de uma
variável. O teto não é amostral. Mais crianças das mesmas variáveis não resolvem;
variáveis novas sobre a criança resolveriam.

O gap treino−validação conta a outra metade da história. Ele é máximo em 409
municípios (0,023) e **negativo** com a base inteira (−0,0026): o modelo escolhido
pela busca — `max_depth = 5`, `min_child_samples = 1.126` — é regularizado a ponto
de não conseguir sobreajustar 1,2 milhão de linhas. Não há overfitting a combater
nesta etapa; há informação faltando.

### 9.3 LeaveOneGroupOut por região

Treinar em quatro regiões e testar na quinta é mais duro que o split por município:
a região que sobra tem outra distribuição de taxa base, porte, rede e cobertura de
lag.

| Região de teste | n | Prevalência | ROC-AUC |
|---|---:|---:|---:|
| Nordeste | 389.251 | 0,4324 | 0,7124 |
| Sul | 233.760 | 0,3941 | 0,6196 |
| Sudeste | 589.835 | 0,3838 | 0,6179 |
| Centro-Oeste | 151.840 | 0,3605 | 0,6130 |
| **Norte** | 156.330 | 0,4930 | **0,5814** |

**O modelo generaliza para região nova, mas de forma desigual.** Nenhuma região cai
para o acaso, o que já é informativo — a relação "contexto de 2023 → risco em 2024"
existe fora do território onde foi aprendida. Mas a amplitude é de 0,131 entre
Nordeste e Norte, muito maior que o desvio entre folds do desenho normal.

O Norte a 0,5814 merece a leitura completa: é a região de maior prevalência de risco
(0,4930, contra 0,3605 no Centro-Oeste) e de faixa mais estreita — quando quase
metade das crianças está em risco, há menos a ordenar. Parte da queda é propriedade
da população, não erro do modelo. Mas 0,58 é fraco em termos absolutos, e um
programa nacional que use este escore no Norte precisa saber disso.

### 9.4 Drift — a única validação out-of-time possível

Validação temporal real é impossível nesta base: features de lag para os alunos de
2023 exigiriam 2022, que não existe. O substituto é um modelo reduzido a `sigla_uf`,
`nome_regiao` e `rede_grupo`, três colunas disponíveis identicamente nos dois anos.

| Desenho | Prevalência do teste | ROC-AUC |
|---|---:|---:|
| 2023 → 2023, CV agrupada | 0,4161 | 0,6359 ± 0,0085 |
| 2024 → 2024, CV agrupada | 0,4022 | 0,6306 ± 0,0084 |
| **2023 → 2024, out-of-time** | 0,4022 | **0,6099** |

**A estrutura territorial se desloca entre edições, e o deslocamento é mensurável.**
Comparando o out-of-time com a referência do próprio ano, a queda é de **0,0207** —
0,6099 contra 0,6306. É um terço da distância entre o campeão e o acaso, e é maior
que o ganho do modelo completo sobre a regra de uma variável.

Duas consequências práticas. O modelo precisa de **retreino anual**: um escore
treinado em 2024 e aplicado em 2025 perde algo dessa ordem, e é isso que o plano de
monitoramento tem de vigiar. E o ranking municipal de 2024 descreve 2024 — usá-lo
para planejar 2026 sem reestimar seria assumir uma estabilidade que a medição
desmente. É notável que a estrutura territorial pura já valha 0,63 dentro do mesmo
ano, contra 0,66 do modelo completo: quase todo o sinal do projeto cabe em UF,
região e rede.

### 9.5 Invariância — os controles se comportaram

Cada coluna é embaralhada cinco vezes no conjunto de teste, e se mede quanto a AUC
se move a partir de 0,6599.

| Coluna | Papel | Queda média | dp |
|---|---|---:|---:|
| `mun_media_portugues_lag1` | controle **positivo** | 0,04203 | 0,00082 |
| `rede_grupo` | feature estrutural | 0,00285 | 0,00021 |
| `caderno` | controle **negativo** | **0,00031** | 0,00007 |

O `caderno` é randomizado entre alunos e não deveria carregar informação. Embaralhá-lo
move a AUC em 0,0003 — **135 vezes menos** que o controle positivo, e um décimo do
que o modelo inteiro ganha sobre a taxa municipal embaralhada dentro do município.
A queda não é exatamente zero (é 4,4 desvios acima de zero, com 1,85 milhão de
linhas qualquer coisa é detectável), mas é pequena o bastante para que a coluna
continue sendo o controle negativo declarado. Se ela subir numa próxima execução, o
achado a reportar é sobreajuste, não uma leitura educacional sobre versões de prova.

O controle positivo prova que o teste tem sensibilidade: sem ele, uma queda nula em
todas as colunas seria indistinguível de um teste quebrado.

---

## 10. Reprodutibilidade

| Item | Valor |
|---|---|
| `RANDOM_STATE` | 42, em constante única, propagado a split, folds, busca, amostragens e estimadores |
| Dataset | `data/processed/dataset_2024.parquet`, MD5 gravado dentro de `campeao.json` e do `joblib` |
| Artefato | `models/campeao.joblib` — modelo, dois limiares, nomes das 89 colunas pós-`ColumnTransformer`, seed, hash e versões |
| Versões | `scikit-learn 1.8.0`, `lightgbm 4.7.0`, Python 3.14.3 |
| Escores out-of-fold | `models/escores_oof_desenvolvimento.parquet`, prontos para o ranking municipal da Etapa 6 |
| Testes | `tests/test_modelagem.py`, 10 casos, incluindo a reprodução do ROC-AUC publicado a partir do artefato em disco |

O teste `test_campeao_em_disco_reproduz_as_metricas_publicadas` escora o conjunto
de teste com o `joblib` e compara com `campeao.json` a menos de 1e-9. É o único
lugar onde uma divergência entre o número do relatório e o do modelo apareceria.

---

## 11. O que a Etapa 5 recebe, e com quais ressalvas

**Pronto para consumo:** `models/campeao.joblib` com o `ModeloCalibrado`, que
recebe a linha como ela sai do Parquet de modelagem e devolve
`P(risco_nao_alfabetizacao)`; `pipeline.nomes_das_features` resolve as 89 colunas
que chegam ao estimador, para que o SHAP não explique `x17` a quem decide orçamento.

**Quatro ressalvas:**

1. **Ler SHAP por família, não por coluna.** As vinte features têm treze pares com
   |Spearman| acima de 0,95 e três combinações lineares exatas. Crédito distribuído
   entre colunas que dizem a mesma coisa não é achado sobre alfabetização, é
   artefato de codificação.
2. **`caderno` se comportou como controle negativo** e o teste de invariância
   confirma. Se ele aparecer alto no SHAP, o alarme é de sobreajuste. O bloco
   `esc_*` é o segundo controle negativo, por ablação, e está fora do modelo.
3. **A importância vale muito mais para 86,6% da coorte que para o resto.** Uma
   leitura global mistura a fatia de AUC 0,671 com a de 0,523. O SHAP precisa ser
   lido também dentro de `fonte_lag_municipal = gold`.
4. **Importância não é causalidade.** O SHAP diz o que o modelo usou para prever,
   não o que aconteceria sob intervenção. Todas as features aqui são contextuais, e
   a falácia ecológica é o risco imediato: atribuir a uma criança a característica
   do seu município descreve o contexto, não ela.
