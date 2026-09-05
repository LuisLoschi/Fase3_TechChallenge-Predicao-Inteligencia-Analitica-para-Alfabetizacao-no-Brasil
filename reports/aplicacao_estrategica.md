# Aplicação estratégica — evidência da Etapa 6

O `PLANO_EXECUCAO.md` guarda as decisões; este documento guarda a medição que as
sustenta, no mesmo formato dos relatórios das Etapas 4 e 5. Tudo aqui foi medido
sobre `models/campeao.joblib` e `data/processed/dataset_2024.parquet`, com
`RANDOM_STATE = 42`, `scikit-learn 1.8.0` e `lightgbm 4.7.0`. O `hash_dataset`
gravado dentro do artefato foi conferido contra o Parquet em disco antes da
primeira conta, e nenhum hiperparâmetro foi reescolhido.

Fase CRISP-DM 5, *Evaluation*, com a ponte para a 6, *Deployment*. Entregáveis:
as cinco perguntas do enunciado respondidas em unidade de decisão, a revisão do
processo e os três CSV que a Etapa 7 vai publicar.

**Resumo em cinco linhas.** O risco de não alfabetização é explicado em 57,2%
pelo histórico do próprio município e em 27,6% pelo estado — juntos, 84,8%. O
ranking municipal foi construído sobre escore out-of-fold nacional, e sai em
**duas listas que não têm um único município em comum**: os 50 de maior
intensidade somam 8,4 mil crianças em risco e estão todos no Nordeste; os 50 de
maior volume somam **242,7 mil**, 29,0% do total do país. O agrupamento devolveu
**três** perfis, com silhueta de 0,2487, e o de maior risco concentra 66,9% de
todas as crianças em risco. E a pergunta sobre metas devolveu a medida da própria
imprevisibilidade: **em 97,5% dos municípios a meta de 2025 cai dentro do
intervalo de 95% da projeção**, e o backtest 2023→2024 marca ROC-AUC 0,5445.

---

## 1. O escore que sustenta o ranking

**Pergunta:** o número que ordena os municípios veio de um modelo que já tinha
visto aquele município?

Um município presente no treino recebe de volta a própria média e sobe na lista
por memorização. O artefato que a Etapa 4 deixou pronto,
`models/escores_oof_desenvolvimento.parquet`, resolve isso — mas só para os
**4.413 municípios do desenvolvimento**, porque os 1.104 do teste tinham de
permanecer intocados até a medição final. Essa medição já aconteceu, e um ranking
nacional que ignore um quinto do país não é publicável.

Os cinco folds agrupados por município foram refeitos sobre a coorte inteira, com
os hiperparâmetros e a seed do campeão, gravando
`models/escores_oof_nacional.parquet`.

| Item | Valor |
|---|---|
| Alunos escorados | 1.851.852 |
| Municípios cobertos | **5.517**, contra 4.413 no artefato da Etapa 4 |
| Folds por município | 1 — nenhum município atravessa a partição |
| ROC-AUC out-of-fold | **0,6607**, contra 0,6599 medido no teste |
| Spearman com o artefato anterior | **0,9881** nos 4.413 municípios comuns |
| Tempo | 124 s, cinco ajustes de LightGBM |

A concordância de 0,9881 com o escore da Etapa 4 é a checagem que autoriza a
substituição: a recontagem estendeu a cobertura sem mudar o modelo.

**O que fica declarado, e não escondido.** Os hiperparâmetros vieram da busca
feita no desenvolvimento, então os municípios do teste participaram
indiretamente da escolha deles. Para uma métrica publicada isso seria
inaceitável, e a Etapa 4 não o fez — os 0,6599 dela continuam sendo o número de
desempenho do projeto. Para um escore descritivo, cuja função é ordenar
territórios e não estimar generalização, é o preço de cobrir o país inteiro. O
invariante que sustenta o ranking continua íntegro: **nenhum município é escorado
pelo modelo que o treinou**, e há teste para isso.

---

## 2. Pergunta 1 — quais fatores mais impactam a alfabetização

Nada foi recalculado. A Etapa 5 mediu três leituras independentes do campeão, e
esta seção é a tradução delas para quem decide orçamento.

| Família | O que ela é, em português | % do \|SHAP\| | Queda de ROC-AUC |
|---|---|---:|---:|
| histórico municipal | desempenho do próprio município em 2023 | **57,2%** | **0,0828** |
| territorial | o estado e a região em que o município está | 27,6% | 0,0462 |
| estrutural | rede de ensino e proveniência do dado | 9,1% | 0,0033 |
| metas | meta e nível publicados pelo INEP | 4,8% | 0,0030 |
| histórico escolar | resíduo de escola que sobrou no modelo | 1,3% | 0,0001 |

**A frase para o slide:** o que mais determina o risco de uma criança não se
alfabetizar, neste modelo, é o desempenho do município em que ela estuda no ano
anterior — proficiência média em português acima de tudo, com 33,0% de toda a
movimentação e 3,3 vezes a segunda colocada, e taxa de alfabetização em seguida.
O estado responde por outro quarto. Juntos, 84,8%.

**A frase que precisa vir junto.** Importância não é causalidade. O SHAP diz o
que o modelo usou para prever, não o que aconteceria sob intervenção — e elevar a
média de português de um município não é intervenção, é o resultado que se quer,
escrito de outra forma. Toda feature aqui é contextual, então aplicá-la a uma
criança específica é falácia ecológica. E a ordenação descreve 2024: a Etapa 4
mediu o modelo reduzido a UF, região e rede caindo de 0,6306 dentro de 2024 para
0,6099 quando treinado em 2023, o que desautoriza projetar fatores para 2026 sem
reestimar.

---

## 3. Pergunta 5 — quais variáveis têm maior influência, por grupo

A tabela da seção 2 é a resposta. O que ela não diz, e a decisão de investimento
precisa, é quanto do modelo é redundante: **dez das vinte features não derrubam a
ROC-AUC mais que o controle negativo** quando removidas isoladamente. O
`caderno`, que é sorteado entre alunos e não pode carregar informação, custa
0,00031 quando embaralhado; nada abaixo de 0,002 conta como fator.
`mun_n_escolas_lag1` tem queda **negativa**, de −0,00009: embaralhá-la melhora a
métrica, assinatura de redundância com `mun_n_alunos_lag1`.

**O achado que interessa a quem financia dado.** As vinte features vêm de três
lugares: a apuração municipal de 2023, a apuração de UF de 2023 e o cadastro da
matrícula. Não há **uma única variável sobre a criança** — nem nível
socioeconômico, nem cor ou raça, nem idade, nem frequência, nem histórico
escolar. É isso que põe o teto em 0,66 em vez de 0,85, e é isso que torna a saída
útil territorial em vez de individual. Mais colunas do mesmo tipo não movem o
resultado; dado sobre o aluno moveria.

---

## 4. O shrinkage, e o resultado que contraria o plano

**Pergunta:** o topo do ranking está ocupado por municípios de oito alunos?

O plano tornou a suavização empírico-bayesiana obrigatória exatamente para
impedir isso. Ela foi aplicada às duas quantidades candidatas pelo mesmo
estimador de componentes de variância — a dispersão observada entre municípios
menos o ruído amostral esperado devolve τ², e a razão entre ruído e τ² dá o `k`,
o porte a partir do qual a medida do município passa a valer mais que o prior.

| Quantidade suavizada | Prior | `k` medido | Maior deslocamento |
|---|---|---:|---:|
| escore out-of-fold do modelo | média nacional | **0,046 aluno** | 0,13 pp |
| taxa observada de 2024 | escore do modelo | **55,1 alunos** | 39,4 pp |

**O shrinkage sobre o escore do modelo é um não-evento.** A razão é mecânica e
não estava no plano: o modelo é territorial, dá quase a mesma probabilidade a
todos os alunos do mesmo município — o desvio-padrão do escore dentro do
município tem mediana de 0,0056 — e a média de uma quantidade quase constante não
tem erro amostral para encolher. O `k` de 0,046 aluno diz que a medida do
município vale mais que o prior a partir de um vigésimo de aluno, ou seja,
sempre.

**A suavização que faz diferença é a da taxa observada**, e ali o `k` é de 55,1
alunos. O prior escolhido não é a média nacional e sim o escore do modelo, o que
transforma o estimador numa mistura de credibilidade: município grande fala por
si, município de vinte alunos é lido pelo que o contexto prevê para ele.

**O medo do plano estava certo, e o alvo dele era outro.** Medido sobre as quatro
ordenações possíveis, entre os cinquenta primeiros:

| Ordenação | Alunos, mediana | Com menos de 50 alunos | Crianças em risco |
|---|---:|---:|---:|
| taxa observada crua | 60 | **21** | 0,4% do total |
| taxa observada ajustada | 159 | 4 | 0,9% |
| **risco do modelo — publicado** | 148 | **2** | 1,0% |
| **volume de crianças — publicado** | 6.103 | 0 | **29,0%** |

O ranking construído sobre a taxa crua teria 21 dos 50 primeiros com menos de 50
alunos avaliados, um deles com risco de 100,0% medido sobre uma coorte
minúscula. O ranking do escore tem 2. A proteção que o plano exigia estava certa
na intenção e errada no mecanismo — quem a garante é o fato de o escore ser
contextual, não o shrinkage aplicado a ele.

**Decisão publicada.** `risco_suavizado` é o escore out-of-fold suavizado, como o
plano define, e é ele quem ordena. `taxa_observada_ajustada` vai ao lado, na
mesma linha do CSV, porque é o número que uma secretaria deve usar quando quiser
a medida do próprio município corrigida pelo ruído, e não a estimativa contextual.

---

## 5. Pergunta 2 — quais municípios apresentam maior risco

Duas listas, porque são duas perguntas de orçamento. Intensidade é onde a criança
tem maior probabilidade individual de não se alfabetizar. Escala é onde estão as
crianças.

### Os dez primeiros por intensidade

| # | Município | UF | Alunos | Presença | Risco | Posição por volume |
|---:|---|---|---:|---:|---:|---:|
| 1 | Casa Nova | BA | 430 | 87,8% | **78,8%** | 439º |
| 2 | Poço Redondo | SE | 345 | 87,1% | 77,4% | 482º |
| 3 | Iaçu | BA | 263 | 97,8% | 75,8% | 751º |
| 4 | Riachuelo | SE | 92 | 98,9% | 74,9% | 1.905º |
| 5 | Araci | BA | 331 | 91,2% | 74,6% | 599º |
| 6 | Esplanada | BA | 359 | 95,0% | 74,2% | 559º |
| 7 | Itaporanga d'Ajuda | SE | 345 | 88,4% | 74,2% | 506º |
| 8 | Barra dos Coqueiros | SE | 278 | 86,1% | 74,0% | 616º |
| 9 | Pacatuba | SE | 106 | 89,8% | 73,8% | 1.598º |
| 10 | Saúde | BA | 82 | 94,3% | 73,4% | 2.113º |

### Os dez primeiros por volume

| # | Município | UF | Alunos | Risco | Crianças em risco | Posição por taxa |
|---:|---|---|---:|---:|---:|---:|
| 1 | São Paulo | SP | 94.373 | 43,9% | **47.242** | 1.766º |
| 2 | Rio de Janeiro | RJ | 43.411 | 42,9% | 21.419 | 1.887º |
| 3 | Manaus | AM | 23.125 | 47,1% | 14.389 | 1.406º |
| 4 | Salvador | BA | 12.158 | 63,3% | 7.700 | 316º |
| 5 | Guarulhos | SP | 13.135 | 47,0% | 7.228 | 1.414º |
| 6 | Porto Alegre | RS | 7.389 | 61,8% | 6.353 | 400º |
| 7 | Campo Grande | MS | 10.001 | 53,6% | 6.006 | 898º |
| 8 | Belém | PA | 8.616 | 54,0% | 5.662 | 866º |
| 9 | Belo Horizonte | MG | 16.012 | 29,7% | 5.626 | 3.448º |
| 10 | Curitiba | PR | 12.923 | 32,6% | 5.169 | 3.093º |

**As duas listas de cinquenta não têm um único município em comum.** O primeiro
por intensidade é o 439º por volume; São Paulo, primeiro por volume, é o 1.766º
por intensidade, e a mediana de posição por intensidade entre os cinquenta
maiores por volume é 1.452.

**A concentração é o argumento orçamentário.** Os 50 de maior risco somam **8,4
mil** crianças em situação de risco, 1,0% do total nacional estimado de **837
mil**. Os 50 de maior volume somam **242,7 mil**, 29,0%. O decil superior por
volume concentra **67,2%** de todas as crianças em risco do país; o decil
superior por intensidade, 16,3%.

**A geografia das duas listas é oposta.** Os cinquenta de maior intensidade estão
**todos no Nordeste** — 41 na Bahia e 9 em Sergipe. Os cinquenta de maior volume
são capitais e regiões metropolitanas distribuídas pelas cinco regiões: 22 no
Sudeste, 11 no Nordeste, 6 no Norte, 6 no Sul, 5 no Centro-Oeste.

**Como usar cada uma.** A lista por intensidade dimensiona programas de
assistência técnica, formação de professor e material — intervenções cujo custo
escala com o número de municípios atendidos. A lista por volume dimensiona
orçamento — intervenções cujo custo escala com o número de crianças. Publicar só
uma delas responderia a uma pergunta e daria a impressão de ter respondido às
duas.

---

## 6. As três ressalvas que vão ao lado da lista

### 6.1 AC e DF: declarados não avaliados

A Etapa 4 mediu ROC-AUC **0,5173** no estrato dos 23 municípios de Acre e
Distrito Federal, contra 0,6602 no resto do país. São as duas unidades sem
nenhuma fonte de lag de UF em 2023, e ali o escore ordena no acaso. No CSV eles
mantêm o escore e **não recebem posição nem decil**: `posicao_por_taxa` e
`posicao_por_volume` vêm vazias, e a coluna `modelo_avaliado` é falsa. Travado
por teste.

### 6.2 A faixa em que a taxa municipal não ordena

A Etapa 5 mediu a contribuição SHAP de `mun_taxa_alfab_lag1` **achatada em torno
de zero abaixo de 65% de alfabetização**, caindo só acima disso. Entre dois
municípios ruins, a taxa de 2023 não distingue nada.

**57,8% dos municípios brasileiros estão nessa faixa. E os 50 do topo da lista
por intensidade estão todos nela.** A consequência prática precisa estar escrita
ao lado da tabela: a lista identifica com segurança **que grupo** de municípios
está em risco alto, e distingue mal **qual deles está pior**. Escolher entre o
12º e o 37º colocado com base nessa ordem é escolher com ruído. A coluna
`ordenacao_fragil` marca cada linha nessa condição.

Isso não invalida a lista — invalida um uso dela. Separar o decil superior dos
demais é decisão que o escore sustenta; ordenar dentro do decil superior não é.

### 6.3 Presença na prova

| Quintil de presença | Municípios | Presença média | Risco observado | Risco do modelo |
|---|---:|---:|---:|---:|
| q1, menor | 1.099 | 79,5% | **45,4%** | 43,4% |
| q2 | 1.099 | 88,2% | 38,7% | 37,9% |
| q3 | 1.098 | 91,8% | 36,0% | 36,2% |
| q4 | 1.099 | 94,9% | 35,9% | 36,2% |
| q5, maior | 1.099 | 98,6% | 27,9% | **30,2%** |

Dois erros que apontam no mesmo sentido. O modelo regride para a média: subestima
o risco onde ele é maior (43,4% previstos contra 45,4% observados no q1) e
superestima onde é menor. E a própria taxa observada já era otimista nos
municípios de baixa presença, porque quem falta à prova tende a ser a criança
mais frágil — a EDA mediu 53,4% de alfabetização entre presentes no quintil de
menor presença contra 74,6% no de maior. Somando os dois: **o problema real, nos
municípios de baixa cobertura, é maior que o número da tabela**. A coluna
`taxa_presenca_2024` acompanha cada linha do CSV para que isso seja verificável
caso a caso.

---

## 7. Pergunta 3 — quais regiões têm padrões semelhantes

KMeans no grão município sobre quinze variáveis padronizadas: taxa de
alfabetização de 2023 e de 2024, média de português, taxa de presença, porte,
share de rede estadual e a composição por nível de proficiência de 2024. A
presença entrou como **variável de agrupamento** e não como filtro, conforme o
plano exige. `k` escolhido pela maior silhueta na grade de 3 a 8.

| k | Inércia | Silhueta | Menor cluster |
|---:|---:|---:|---:|
| **3** | 49.635 | **0,2487** | 378 |
| 4 | 44.557 | 0,1740 | 303 |
| 5 | 41.523 | 0,1779 | 295 |
| 6 | 39.159 | 0,1423 | 219 |
| 7 | 37.338 | 0,1395 | 212 |
| 8 | 35.920 | 0,1374 | 208 |

**O critério escolheu o menor `k` da grade, e isso é resultado e não fracasso.**
Silhueta de 0,2487 significa que os grupos se tocam; a inércia não mostra
cotovelo nítido. O espaço municipal brasileiro é um **gradiente de desempenho**,
não uma coleção de tipos separados. Reportar cinco grupos aqui seria fatiar um
contínuo e dar aos pedaços nome de descoberta — e a inspeção do `k = 5` confirma:
o grupo extra que aparece ali se define pelo share de rede estadual em municípios
do Sul, que é fato de rede e não padrão de alfabetização.

### Os três perfis

| | alfabetização consolidada | faixa intermediária | risco alto e cobertura menor |
|---|---:|---:|---:|
| Municípios | 378 | 2.941 | **2.142** |
| Alunos avaliados | 79.415 | 726.949 | **1.010.348** |
| Porte mediano | 98 | 95 | 138 |
| Taxa 2023 | 84,7% | 67,4% | 46,8% |
| Taxa 2024 | **92,6%** | 73,4% | **43,9%** |
| Média de português | 804,3 | 761,4 | 733,1 |
| Presença | 96,9% | 91,5% | 88,3% |
| Alunos nos níveis 0 a 2 | 2,0% | 7,9% | **23,9%** |
| Alunos nos níveis 6 a 8 | **79,7%** | 38,0% | 17,6% |
| Crianças em risco | 9,6 mil (1,1%) | 252,4 mil (29,4%) | **574,0 mil (66,9%)** |

**O grupo de maior risco é onde tudo está.** Reúne 2.142 municípios, mais da
metade dos alunos avaliados do país e **66,9% de todas as crianças em risco**. É
de onde saem **os 50 primeiros da lista por intensidade e 40 dos 50 primeiros por
volume**. E é o único cujo desempenho piorou entre as duas edições, de 46,8% para
43,9%, na direção contrária à do país.

### Composição regional

| Grupo | Centro-Oeste | Nordeste | Norte | Sudeste | Sul |
|---|---:|---:|---:|---:|---:|
| alfabetização consolidada | 10,1% | **67,5%** | 0,5% | 20,6% | 1,3% |
| faixa intermediária | 11,1% | 17,0% | 4,0% | 40,9% | 27,1% |
| risco alto e cobertura menor | 4,7% | **48,0%** | 13,7% | 16,7% | 16,9% |

**O Nordeste ocupa os dois extremos, e a leitura regional agregada apaga isso.**
O grupo consolidado é 67,5% nordestino, e **164 dos 184 municípios cearenses
estão nele**, 89,1%. O grupo de maior risco também é 48% nordestino, e leva **391
dos 407 municípios baianos**, 96,1%, mais 70 dos 75 de Sergipe. Falar em "desempenho do
Nordeste" soma dois retratos opostos e não descreve nenhum dos dois.

**A baixa presença não formou um perfil próprio, e o plano previa que formaria.**
O decil de menor cobertura se distribui 67,3% no grupo de risco alto e 32,4% no
intermediário, em vez de se concentrar num grupo seu. A presença entrou como um
gradiente colado ao desempenho — 88,3% no grupo de maior risco contra 96,9% no
consolidado — e não como eixo independente. O nome do terceiro grupo diz "e
cobertura menor" e não "com baixa participação" por causa disso: a diferença de
cobertura é de 8,6 pontos percentuais entre os extremos, real e modesta.

**Duas cautelas de leitura, medidas.** A composição por nível de proficiência só
existe em 2024 — é 100% nula em 2023 —, então este é um **retrato de um ano** e
não uma trajetória: dois municípios no mesmo grupo podem estar indo em direções
opostas. E 56 municípios ficaram fora por dado faltante, entre eles 22 do Acre e
21 de São Paulo, o que está declarado no resumo em JSON.

---

## 8. Pergunta 4 — municípios que podem não atingir as metas futuras

A pergunta foi reformulada, e a reformulação é sustentada por medição. O
diagnóstico mostrou que as metas municipais são **determinísticas e derivadas da
própria taxa base**: `corr(meta_2025, taxa_2023) = 0,9765`, todas as metas de 2030
iguais a 80% para os 5.352 municípios, e os valores idênticos nas linhas de 2023 e
de 2024. Prever o não atingimento a partir da taxa anterior dá ROC-AUC **abaixo do
acaso**. A entrega vira três produtos, em ordem decrescente de confiabilidade.

Nota de fonte: a taxa usada nesta seção é a **publicada pelo INEP na própria base
de metas**, e não a ponderada do projeto. A meta foi construída sobre a rede
Municipal apurada pelo INEP, e comparar contra a taxa de todas as redes trocaria
a régua no meio da conta. As duas medidas concordam com **MAE de 0,81pp** e
correlação de 0,9894.

### 8a. Gap de esforço — aritmética pura

`meta_2025 − taxa_2024`, sem modelo nenhum. É o número mais acionável do projeto.

| Estatística | Valor |
|---|---:|
| Municípios com meta publicada | 5.352 de 5.517 |
| Já superam a meta de 2025 | **43,4%** |
| Esforço mediano | **+2,24 pp** |
| Percentil 95 | +26,91 pp |
| Mínimo / máximo | −76,28 pp / +68,90 pp |

A distribuição é bimodal por construção. A meta foi desenhada como trajetória
linear da taxa base até 80% em 2030, então quem estava longe recebeu inclinação
alta e quem estava perto recebeu quase nada.

### 8b. Projeção com incerteza — probabilidade, nunca rótulo

Persistência mais deriva nacional, com a taxa de 2024 suavizada por empirismo
bayesiano antes de projetar. A dispersão do resíduo é ajustada como
`sd(n) = √(a²/n + c²)` por máxima verossimilhança: `a²/n` é o ruído de amostragem
da coorte, que encolhe com o porte, e `c` é o que sobra depois dele — mudança
real de política, de rede, de cobertura da prova.

| Faixa de porte | Municípios | Porte mediano | Desvio medido | Ajuste |
|---|---:|---:|---:|---:|
| até 25 alunos | 299 | 20 | **24,19 pp** | 26,04 |
| 25 a 50 | 732 | 38 | 19,46 | 20,08 |
| 50 a 100 | 1.121 | 72 | 16,57 | 16,10 |
| 100 a 200 | 1.057 | 141 | 14,02 | 13,43 |
| 200 a 500 | 898 | 289 | 12,03 | 11,76 |
| 500 a 1.000 | 279 | 668 | 10,71 | 10,75 |
| acima de 1.000 | 225 | 1.709 | **8,77** | 10,24 |

Parâmetros: `a = 107,7`, `c = 9,91 pp`, deriva nacional de **+2,39 pp** entre 2023
e 2024. O ajuste acompanha a medição em toda a faixa e fica levemente
conservador acima de mil alunos.

**O resultado central da pergunta.** O intervalo de 95% da projeção de 2025 tem
largura mediana de **55,7 pontos percentuais**, e a meta cai dentro dele em
**5.218 dos 5.352 municípios — 97,5%**. Só **134** estão suficientemente longe da
própria meta para que o dado autorize uma afirmação: 95 deles quase certamente
ficarão abaixo, com gap mediano de +30,7 pp, e 39 quase certamente ficarão acima,
com gap mediano de −35,3 pp.

**E o piso é quatro vezes o que a meta pede.** Mesmo onde o ruído amostral
praticamente desaparece, o desvio-padrão da variação anual não desce muito:
**8,77 pp** medidos na faixa acima de mil alunos, contra um piso ajustado de
**9,91 pp**. A meta de 2025 pede +2,24 pp da mediana. Num município de vinte
alunos avaliados a oscilação medida é de **24,19 pp**.

### 8c. Backtest 2023 → 2024 — reportado mesmo sendo ruim

O mesmo procedimento, um ano para trás, sobre os 4.611 municípios com taxa nos
dois anos e meta de 2024 publicada. Prevalência de não atingimento: 46,5%.

| Preditor | ROC-AUC |
|---|---:|
| taxa de 2023 invertida | **0,4876** |
| gap até a meta, sem modelo | **0,4916** |
| projeção com shrinkage e deriva | **0,5445** |

**Os preditores ingênuos ficam abaixo do acaso, exatamente como o diagnóstico
mediu** — ali os números foram 0,485 e 0,488, e a réplica sobre a base de metas
do INEP devolve 0,4876 e 0,4916. Não é "fraco", é pior que sortear. A causa está
no desenho da meta: como ela é construída a partir da taxa base, a distância até
ela é quase constante entre municípios, e o que resta para explicar o não
atingimento é ruído de coorte.

**A máquina completa chega a 0,5445 e não salva a pergunta.** O ganho de Brier
sobre simplesmente prever a prevalência é de 0,0015 — de 0,24877 para 0,24728. A
calibração é não monótona no miolo: no quarto decil de probabilidade prevista a
fração observada é *menor* que no segundo. E este número é **otimista**, porque a
deriva e a curva de dispersão foram ajustadas na mesma transição 2023→2024 em que
o backtest é medido. Com dois anos de base não há alternativa, e o viés fica
declarado em vez de estimado para baixo.

### O funil até 2030, sem projetar nada

| Ano | Meta mediana | Já atingem com a taxa de 2024 | Gap mediano |
|---:|---:|---:|---:|
| 2025 | 66,9% | 2.322 — 43,4% | +2,25 pp |
| 2026 | 69,9% | 1.968 — 36,8% | +5,35 pp |
| 2027 | 72,7% | 1.658 — 31,0% | +8,26 pp |
| 2028 | 75,3% | 1.421 — 26,6% | +11,08 pp |
| 2029 | 77,7% | 1.258 — 23,5% | +13,47 pp |
| 2030 | 80,0% | **1.141 — 21,3%** | **+15,75 pp** |

Congelando a taxa de 2024, apenas 21,3% dos municípios atingem a meta de 2030, e
o gap mediano é de +15,75 pp em seis anos contra uma deriva nacional medida de
+2,39 pp por ano. O funil não depende de nenhuma hipótese de projeção: é a régua
subindo contra um retrato parado.

### A conclusão de política pública

**O não atingimento de metas é largamente imprevisível, e a causa está no desenho
da meta, não no modelo.** Elas são individualizadas a partir da própria taxa base
do município, então a distância até a meta carrega pouca informação estrutural e
o resíduo é ruído amostral. Os três preditores testados ficam entre 0,4876 e
0,5445 de ROC-AUC, e o melhor deles empata com prever a prevalência.

**A recomendação que sai daí:** municípios com menos de **119 alunos avaliados**
— o porte em que `a²/n = c²`, ou seja, em que o ruído de amostragem deixa de ser
o componente dominante da oscilação anual, e que são **51,6% dos municípios
brasileiros** — não deveriam ter metas avaliadas individualmente sem intervalo de
confiança publicado ao lado. Para eles, a unidade de avaliação precisa ser
plurianual, ou agrupada por microrregião, ou por grupo de perfil semelhante.
Avaliar um município de trinta alunos contra uma meta de dois pontos percentuais
é premiar e punir variação amostral.

O limiar de 119 é mais exigente que os "cerca de 50" que o plano antecipava, e
os dois números se encontram por caminhos independentes: o `k` da mistura de
credibilidade da seção 4 dá 55 alunos, e o ponto de igualdade das componentes de
variância dá 119. A faixa entre 55 e 119 é onde a avaliação individual começa a
fazer sentido, e abaixo de 55 ela não faz nenhum.

**O que continua utilizável:** o gap de esforço de 8a, que é subtração e não
erra, e a lista dos **134 municípios** cuja meta cai fora do intervalo de 95% —
os únicos para os quais o dado autoriza uma afirmação sobre 2025.

---

## 9. Revisão do processo

A fase de *Evaluation* pede a revisão do que foi feito, e não só do resultado.

- **O artefato é o publicado.** O `hash_dataset` gravado no `joblib` confere com o
  Parquet em disco, e o CLI falha se divergir.
- **O escore é out-of-fold, e há teste para isso.** Cada município cai em
  exatamente um fold, e a partição é reconstruída com a mesma seed dentro do
  teste, não apenas assumida.
- **Nenhuma decisão de modelagem foi reaberta.** Nenhum hiperparâmetro, nenhuma
  feature, nenhum limiar. A única coisa que mudou em relação à Etapa 4 é a
  cobertura do escore, e a concordância de 0,9881 mostra o tamanho da mudança.
- **Duas coisas do plano não se confirmaram, e estão reportadas como não
  confirmadas:** o shrinkage sobre o escore do modelo é um não-evento com `k` de
  0,046 aluno, e a baixa presença não formou um cluster próprio.
- **A pergunta 4 devolveu a medida da imprevisibilidade em vez de uma previsão.**
  Isso é uma resposta, não uma falha, e o CSV não contém nenhuma coluna de rótulo
  binário — há teste que falha se alguém acrescentar uma.

**O que não foi feito, e fica declarado.** Não há intervalo de confiança em torno
do `risco_suavizado` de cada município. O CSV traz `n_alunos_avaliados` e
`taxa_presenca_2024`, que são os dois insumos de quem quiser construí-lo, mas a
incerteza da posição no ranking não está quantificada — e ela é maior do que a
diferença entre posições vizinhas dentro do decil superior, como a seção 6.2
argumenta por outro caminho. Também não há validação temporal do ranking: com
duas edições da prova, não existe um 2025 contra o qual conferir a ordenação de
2024.

---

## 10. Reprodutibilidade

| Item | Valor |
|---|---|
| Comando | `python -m src.modeling.strategic`, 131 s do zero |
| Entradas | `models/campeao.joblib`, `data/processed/dataset_2024.parquet` (MD5 `cc2215565c04d14a5ad97dbfcdda31b5`), agregados municipais e as bases do INEP |
| Saídas publicáveis | `reports/ranking_risco_municipal.csv` (5.517 linhas), `reports/clusters_municipais.csv` (5.461), `reports/projecao_metas_municipios.csv` (5.352) |
| Métricas auxiliares | oito CSV e um JSON em `reports/metrics/estrategia_*` |
| Artefato intermediário | `models/escores_oof_nacional.parquet`, 1.851.852 linhas |
| Gráficos | onze em `images/estrategia/`, gerados pelo notebook a partir dos CSV |
| Notebook | `notebooks/05_aplicacao_estrategica.ipynb`, 15 células de código no kernel `tc-fase3` |
| Versões | `scikit-learn 1.8.0`, `lightgbm 4.7.0`, Python 3.14.3 |
| Testes | `tests/test_estrategia.py`, 21 casos; suíte completa em **84** |
| Determinismo | duas execuções completas do zero produziram os três CSV publicáveis, os oito auxiliares e o Parquet de escores **bit a bit idênticos** (MD5 conferido) |

O `KMeans` roda com `random_state=42` e `n_init=10`, e os clusters são renomeados
pela ordem da taxa de alfabetização de 2024 — sem isso o mesmo grupo mudaria de
nome entre execuções e a tabela deste relatório deixaria de conferir com o CSV.

A checagem de determinismo pegou uma coisa na primeira tentativa: a inércia do
`KMeans` variava no último bit entre execuções, porque a redução multithread soma
os quadrados em ordem diferente. Rótulos, silhueta e tamanhos de cluster eram
idênticos, então nenhuma decisão mudava — mas o CSV trocava de MD5 sem que nada
tivesse mudado, e uma checagem que acusa falso positivo deixa de ser checagem. A
inércia passou a ser gravada arredondada.

---

## 11. O que a Etapa 7 recebe

**Três CSV prontos para publicação**, cada um com as colunas de ressalva ao lado
das colunas de resultado: `modelo_avaliado`, `ordenacao_fragil`,
`taxa_presenca_2024` e `n_alunos_avaliados` no ranking;
`meta_dentro_do_intervalo` e `meta_avaliavel_individualmente` na projeção de
metas.

**As cinco respostas, em uma linha cada:**

| Pergunta | Resposta |
|---|---|
| 1. Fatores | histórico municipal 57,2% do \|SHAP\|, território 27,6% — juntos 84,8% |
| 2. Municípios de risco | duas listas sem sobreposição; 837 mil crianças em risco, 67,2% no decil superior por volume |
| 3. Regiões semelhantes | 3 perfis, silhueta 0,2487; o de maior risco tem 2.142 municípios e 66,9% das crianças em risco |
| 4. Metas futuras | gap mediano +2,24 pp; a meta cai dentro do IC95 em 97,5% dos municípios; backtest 0,5445 |
| 5. Variáveis por grupo | 5 famílias; 10 das 20 features acima do piso de ruído |

**Quatro limitações novas para a lista do README**, além das dezesseis já
declaradas:

1. **O ranking ordena bem entre decis e mal dentro do decil superior.** 57,8% dos
   municípios estão na faixa em que `mun_taxa_alfab_lag1` não discrimina, e os 50
   do topo estão todos nela.
2. **AC e DF não são avaliados**, e por isso não recebem posição — 23 municípios.
3. **O escore nacional usa hiperparâmetros escolhidos com participação indireta
   do teste.** É descritivo, não é a métrica publicada, e a métrica publicada
   continua sendo a da Etapa 4.
4. **165 municípios da coorte não têm meta publicada** e ficam fora do CSV de
   projeção; eles têm porte mediano de 50 alunos e se concentram em RS, SC e MG.

**Para o vídeo**, três números que cabem em trinta segundos: as duas listas de
cinquenta municípios não têm nenhum em comum; um grupo de 2.142 municípios
concentra dois terços das crianças em risco do país; e em 97,5% dos municípios a
meta de 2025 está dentro da margem de erro da própria projeção.
