# Interpretabilidade — a evidência

> **Status de validação deste resultado.** As medições do classificador são
> **retrospectivas e exploratórias**, não confirmatórias. A seleção supervisionada de
> atributos consultou a coorte inteira de 2024, então a reserva de teste **não é
> independente** dessa seleção — e trocar a seed do split não a torna independente, porque
> o que foi consultado foi a coorte, não uma partição dela. O status está fixado em código,
> em `src/evaluation/protocolo.py`, e viaja junto do modelo em `carregar_campeao()`.
> Confirmação prospectiva exige uma amostra ainda não consultada e a documentação da
> disponibilidade temporal das fontes; nenhuma das duas existe hoje. Nada aqui sustenta
> diagnóstico individual, garantia prospectiva ou conclusão causal.


Este documento guarda a medição que sustenta a leitura do modelo, no mesmo formato do
relatório de modelagem. Duas ressalvas valem para tudo o que vem abaixo e estão repetidas
onde importam: **a faixa de SHAP de uma variável não mede o ranking completo**, e **a
incerteza das posições não foi quantificada** — nenhum ranking aqui vem com intervalo.

Tudo aqui foi medido sobre o campeão serializado em `models/campeao.joblib`, com
`RANDOM_STATE = 42`, `scikit-learn 1.8.0`, `lightgbm 4.7.0` e `shap 0.52.0`. Os
números saem de `reports/metrics/interpret_*` e as sete figuras de
`images/interpretabilidade/`; o notebook `04_interpretabilidade.ipynb` reexecuta a
leitura inteira sem recalcular nada.

Fase CRISP-DM 5, *Evaluation*. Entregáveis: leitura global e local do modelo,
importância por grupo de variável, triangulação entre técnicas e a lista do que a
evidência **não** autoriza concluir.

**Resumo em cinco linhas.** O modelo é territorial e a medição não deixa dúvida: o
histórico do próprio município em 2023 responde por **57,2%** do movimento das
predições e por **0,0828** de ROC-AUC quando embaralhado em bloco; somado ao
território acima do município, dá **84,8%** de tudo. Uma única variável — a
proficiência média em português do município — vale **33,0%** do total, 3,3 vezes a
segunda. Das vinte features, **dez** não se distinguem do ruído quando removidas
isoladamente. Três features passam no critério de triangulação e só **uma**
(`mun_taxa_alfab_lag1`) aparece no topo das três leituras disponíveis. Nada disso é
causal, e a seção 12 diz por quê.

---

## 1. O que está sendo explicado, e sobre qual população

**Pergunta:** o artefato em disco é o mesmo que produziu o número publicado?
**Decisão:** seguir com a leitura ou parar e retreinar.

O `hash_dataset` gravado dentro do `joblib` é conferido contra o Parquet atual antes
de qualquer conta. Explicar um modelo com um arquivo de dados diferente do que o
treinou produz uma leitura plausível e falsa, e nenhuma métrica desta etapa
denunciaria isso sozinha. O hash confere: `cc2215565c04d14a5ad97dbfcdda31b5`.

| Item | Valor |
|---|---|
| Features de entrada | 20 |
| Colunas após o `ColumnTransformer` | 89 |
| Isotônica aplicada | não — a modelagem a rejeitou |
| Amostra do SHAP | 50.000 alunos, dos quais 43.263 com lag vindo da Gold |
| Origem da amostra | conjunto de teste, municípios inéditos |
| Valor esperado (log-odds) | −0,4360 |
| Permutação | 330.836 alunos, 10 repetições, ROC-AUC de referência 0,6599 |
| Piso de ruído | 0,002 |

**A amostra é de teste, não de treino.** São 50.000 alunos sorteados entre os 330.836
de municípios inéditos. Explicar o modelo no dado que ele memorizou devolveria a
importância do que foi decorado; a permutação, que roda sobre o teste inteiro, tem a
mesma exigência.

**A isotônica está desligada**, então o `TreeExplainer` ataca o `LGBMClassifier`
diretamente e a explicação é exata, não aproximada por amostragem. Fosse o modelo
calibrado o objeto explicado, seria preciso um explainer agnóstico e caro, com erro
de amostragem entrando na conta.

---

## 2. O mapa de 89 colunas para 20 features e 5 famílias

**Pergunta:** onde cada valor SHAP precisa ser creditado?
**Decisão:** a unidade em que a importância é reportada.

O `ColumnTransformer` transforma 20 features de entrada em 89 colunas: 14 valores
numéricos, 12 indicadores de nulo e 63 dummies. Sem devolver esse crédito à variável
de origem, a leitura compara uma coluna contínua com um vigésimo sexto de uma
variável categórica.

| Família | Features | Colunas | Dummies | Indicadores de nulo | Valores |
|---|---:|---:|---:|---:|---:|
| territorial | 5 | 37 | 31 | 3 | 3 |
| estrutural | 5 | 33 | 32 | 0 | 1 |
| historico_municipal | 7 | 14 | 0 | 7 | 7 |
| metas | 2 | 4 | 0 | 2 | 2 |
| historico_escolar | 1 | 1 | 0 | 0 | 1 |

As cinco features que mais geram colunas: `sigla_uf` (26), `caderno` (22),
`nome_regiao` (5), `fonte_lag_municipal` (4) e `rede_grupo` (3).

O mapa é construído a partir do transformador **ajustado**, e a reconstrução é
conferida posição a posição contra `get_feature_names_out()`. Casamento de string
sozinho colocaria as dummies de `fonte_lag_municipal` dentro de `fonte_lag_uf` em
silêncio — os dois nomes compartilham prefixo.

**Achado 1, a codificação distribui o peso de forma muito desigual entre as
variáveis.** `sigla_uf` ocupa 26 das 89 colunas e `caderno` outras 22, enquanto
`mun_media_portugues_lag1` ocupa duas. Num ranking coluna a coluna, a UF aparece
fatiada em 26 pedaços pequenos e some do topo; a variável municipal aparece inteira.
É um artefato de codificação, e é por isso que a leitura por família deixou de ser
refinamento nesta etapa e virou a unidade principal de reporte.

`reports/metrics/interpret_mapa_de_familias.csv` traz o mapa completo, coluna a
coluna.

---

## 3. SHAP global — magnitude e direção

**Pergunta:** de quanto cada coluna move a predição, e para que lado?
**Decisão:** o conteúdo da resposta executiva sobre fatores.

O `beeswarm` (`01_shap_beeswarm.png`) é o gráfico principal porque carrega direção
junto da magnitude. Barra ordenada por |SHAP| diz quem importa; o enxame diz se valor
alto empurra a criança para o risco ou para longe dele.

| # | Feature | Família | Colunas | \|SHAP\| médio | Participação | Cancelamento interno |
|---:|---|---|---:|---:|---:|---:|
| 1 | `mun_media_portugues_lag1` | historico_municipal | 2 | 0,2709 | 33,0% | 0,0000 |
| 2 | `uf_media_portugues_lag1` | territorial | 2 | 0,0827 | 10,1% | 0,0000 |
| 3 | `mun_taxa_alfab_lag1` | historico_municipal | 2 | 0,0653 | 8,0% | 0,0052 |
| 4 | `rede_grupo` | estrutural | 3 | 0,0574 | 7,0% | 0,0001 |
| 5 | `mun_taxa_presenca_lag1` | historico_municipal | 2 | 0,0557 | 6,8% | 0,0000 |
| 6 | `sigla_uf` | territorial | 26 | 0,0494 | 6,0% | **0,3946** |
| 7 | `nome_regiao` | territorial | 5 | 0,0379 | 4,6% | 0,1920 |
| 8 | `uf_taxa_presenca_lag1` | territorial | 2 | 0,0330 | 4,0% | 0,0251 |
| 9 | `uf_taxa_alfab_lag1` | territorial | 2 | 0,0299 | 3,6% | 0,0068 |
| 10 | `mun_meta_2024_lag1` | metas | 2 | 0,0290 | 3,5% | 0,0012 |

A coluna `cancelamento_interno` mede quanto crédito se perde ao somar as colunas de
uma mesma feature antes de tomar o módulo: em `sigla_uf` são 39,5% e no `caderno`
45,6%, porque dummies da mesma variável têm sinais opostos entre si. Onde a feature
vira uma ou duas colunas, o cancelamento é praticamente zero. Ignorá-lo é o erro que
faz uma variável categórica desaparecer de um ranking coluna a coluna.

**Achado 2, uma variável responde por quase um terço de tudo.**
`mun_media_portugues_lag1` — a proficiência média em português do município em 2023 —
tem |SHAP| médio de **0,271** em log-odds, **3,3 vezes** a segunda colocada e 33,0%
de toda a movimentação do modelo. As três seguintes somadas não a alcançam.

**Achado 3, a direção é a esperada e é forte.** No enxame, valores altos de
proficiência municipal ficam à esquerda, empurrando o log-odds de não alfabetização
para baixo em até 1,8 — o que perto da prevalência de 0,3812 do teste vale dezenas de
pontos percentuais de probabilidade. Nenhuma variável aparece com sinal invertido em
relação ao que a EDA mediu, o que é a primeira conferência de sanidade da etapa.

**Achado 4, o `caderno` se comportou como controle negativo também aqui.** Ele ocupa
22 colunas e some do topo: somadas, ficam em **1,6%** do total e na 15ª posição entre
as 20 features. O `caderno` é sorteado entre alunos e não pode carregar informação;
se tivesse aparecido alto, o achado a reportar seria sobreajuste a ruído, e não uma
leitura educacional sobre versões de prova.

---

## 4. Importância por família — a leitura que responde à pergunta por grupo

**Pergunta:** quanto vale cada bloco de variáveis, somado o crédito que a
colinearidade repartiu?
**Decisão:** a resposta à pergunta 5 do enunciado, que é por grupo de variável.

Duas medidas, porque são duas perguntas. A participação no |SHAP| diz de quanto a
família move a predição. A permutação em bloco — embaralhar todas as colunas da
família de uma vez, com o mesmo vetor de posições — diz quanto ela vale **depois** de
descontado o que as outras famílias já explicam. Figura `03_importancia_por_familia.png`.

| Família | Features | Colunas | Participação \|SHAP\| | Queda em bloco | dp | Acima do piso |
|---|---:|---:|---:|---:|---:|---|
| **historico_municipal** | 7 | 14 | **57,2%** | **0,0828** | 0,0009 | sim |
| territorial | 5 | 37 | 27,6% | 0,0462 | 0,0006 | sim |
| estrutural | 5 | 33 | 9,1% | 0,0033 | 0,0002 | sim |
| metas | 2 | 4 | 4,8% | 0,0030 | 0,0001 | sim |
| historico_escolar | 1 | 1 | 1,3% | 0,0001 | 0,00004 | **não** |

O agrupamento não é escolha de apresentação — o efeito dele foi medido contra a
alternativa:

| Família | Soma das colunas isoladas | Família embaralhada junta | Razão |
|---|---:|---:|---:|
| historico_municipal | 0,0513 | 0,0828 | **1,61×** |
| territorial | 0,0244 | 0,0462 | **1,90×** |
| estrutural | 0,0034 | 0,0033 | 0,97× |
| metas | 0,0031 | 0,0030 | 0,98× |
| historico_escolar | 0,0001 | 0,0001 | 1,04× |

**Achado 5, o histórico municipal domina, e a hipótese H1 se sustenta.** A família
responde por 57,2% do |SHAP| e por uma queda de 0,0828 de ROC-AUC quando embaralhada
inteira — quase o dobro da segunda colocada nas duas réguas. O critério de
falseamento escrito na EDA era "uma família não municipal aparecer no topo do SHAP", e
nenhuma apareceu. As duas ordenações, por SHAP e por permutação, coincidem posição a
posição nas cinco famílias.

**Achado 6, a leitura coluna a coluna subestima exatamente onde há redundância, e o
efeito foi medido.** Somar as quedas das colunas isoladas da família municipal dá
0,0513; embaralhar a família junta dá 0,0828, 1,61 vez mais. Na territorial a razão é
1,90. Já em metas, estrutural e histórico escolar a razão fica entre 0,97 e 1,04 —
famílias sem redundância interna não ganham nada com o agrupamento. É a demonstração
numérica de que o agrupamento corrige um viés real: onde as features dizem a mesma
coisa, permutar uma de cada vez deixa a informação intacta nas outras e a importância
individual sai diluída.

**Achado 7, território acima do município pesa mais do que a EDA sugeria.** A família
territorial — as três features de UF mais `sigla_uf` e `nome_regiao` — vale 27,6% do
|SHAP| e 0,0462 de queda em bloco. Somada à municipal, dá **84,8%** de tudo o que o
modelo usa. As metas ficam em 4,8% e o histórico escolar em 1,3%, este último
**abaixo do piso de ruído** na permutação.

---

## 5. Permutação no conjunto de teste

**Pergunta:** o que a métrica perde quando a coluna é destruída, e não apenas quanto
ela move a predição?
**Decisão:** o que passa da lista de "usado pelo modelo" para a lista de "necessário
ao modelo".

Dez repetições sobre os 330.836 alunos de teste, permutando a feature **crua** antes
do pré-processamento — o imputador e o encoder veem o valor embaralhado, como veriam
em produção. O piso de ruído é 0,002, calibrado pelo `caderno`, que é sorteado entre
alunos e portanto não pode carregar informação. Figura `04_permutacao_no_teste.png`.

| # | Feature | Família | Queda média | dp | Acima do piso |
|---:|---|---|---:|---:|---|
| 1 | `mun_media_portugues_lag1` | historico_municipal | 0,04174 | 0,00062 | sim |
| 2 | `sigla_uf` | territorial | 0,00940 | 0,00015 | sim |
| 3 | `uf_media_portugues_lag1` | territorial | 0,00715 | 0,00015 | sim |
| 4 | `mun_taxa_alfab_lag1` | historico_municipal | 0,00437 | 0,00023 | sim |
| 5 | `mun_meta_2024_lag1` | metas | 0,00299 | 0,00021 | sim |
| 6 | `uf_taxa_alfab_lag1` | territorial | 0,00294 | 0,00015 | sim |
| 7 | `rede_grupo` | estrutural | 0,00283 | 0,00015 | sim |
| 8 | `mun_taxa_presenca_lag1` | historico_municipal | 0,00261 | 0,00015 | sim |
| 9 | `nome_regiao` | territorial | 0,00253 | 0,00011 | sim |
| 10 | `uf_taxa_presenca_lag1` | territorial | 0,00234 | 0,00010 | sim |
| 11 | `mun_n_alunos_lag1` | historico_municipal | 0,00107 | 0,00011 | não |
| 12 | `mun_desvio_vs_uf` | historico_municipal | 0,00100 | 0,00010 | não |
| … | | | | | |
| 14 | `caderno` | estrutural *(controle negativo)* | 0,00030 | 0,00008 | não |
| 16 | `tem_historico_escola` | historico_escolar | 0,00010 | 0,00006 | não |
| 20 | `mun_n_escolas_lag1` | historico_municipal | **−0,00009** | 0,00011 | não |

**Achado 8, dez das vinte features passam do piso de ruído.** As outras dez —
incluindo `mun_desvio_vs_uf`, `mun_n_alunos_lag1`, as três colunas de proveniência do
lag e o próprio `caderno` — não distinguem importância de acaso quando removidas
isoladamente. Elas seguem no modelo porque não atrapalham, não porque façam falta. A
leitura de parcimônia é direta: um modelo com metade das features provavelmente
entregaria a mesma AUC com metade da superfície de monitoramento.

**Achado 9, `mun_n_escolas_lag1` tem queda negativa.** Embaralhar o número de escolas
do município em 2023 **melhora** a AUC em 0,00009, com desvio de 0,00011. O efeito é
indistinguível de zero, mas o sinal importa: a variável é 11ª no SHAP e última na
permutação. O modelo a usa, e o que ela acrescenta não sobrevive a ser retirada —
leitura clássica de coluna redundante com `mun_n_alunos_lag1`.

**Achado 10, e o mais desconfortável: a ordem muda entre as duas leituras.**
`sigla_uf` é 6ª no SHAP e **2ª** na permutação; `mun_meta_2024_lag1` é 10ª no SHAP e
5ª na permutação. A explicação é a mesma nos dois casos — o SHAP fatia a variável
entre suas colunas e a permutação a destrói inteira. É por isso que nenhuma das duas
leituras decide sozinha, e é a razão de existir a triangulação da seção 8.

---

## 6. Coeficientes da logística podada

**Pergunta:** em que direção, e com que magnitude, cada variável entra num modelo que
se lê a olho nu?
**Decisão:** a terceira ponta da triangulação, e o formato em que a resposta chega a
quem não lê SHAP.

O campeão não tem coeficiente. A leitura vem do modelo linear que ficou 0,0098 de AUC
atrás dele com oito features e todo VIF abaixo de 3. Cada coeficiente é reajustado nos
cinco folds agrupados por município, e o desvio entre eles vai ao lado: coeficiente
que troca de sinal entre folds não sustenta afirmação de direção. Features
padronizadas, então as magnitudes são comparáveis entre si. Figura
`05_coeficientes_logistica.png`.

| Coluna | Família | Coeficiente | Razão de chances | dp entre folds | Troca de sinal |
|---|---|---:|---:|---:|---|
| `mun_taxa_alfab_lag1` | historico_municipal | −0,4273 | **0,65** | 0,0078 | não |
| `rede_grupo_Estadual` | estrutural | −0,3160 | 0,73 | 0,0255 | não |
| `nome_regiao_Sudeste` | territorial | −0,3097 | 0,73 | 0,0109 | não |
| `nome_regiao_Sul` | territorial | **+0,2774** | **1,32** | 0,0197 | não |
| `nome_regiao_Centro-Oeste` | territorial | −0,1706 | 0,84 | 0,0247 | não |
| `uf_taxa_alfab_lag1` | territorial | −0,1659 | 0,85 | 0,0102 | não |
| `mun_taxa_presenca_lag1` | historico_municipal | −0,1174 | 0,89 | 0,0085 | não |
| `tem_historico_municipio` | estrutural | −0,0944 | 0,91 | 0,0139 | não |
| `mun_share_rede_estadual_lag1` | historico_municipal | +0,0559 | 1,06 | 0,0049 | não |
| `nome_regiao_Norte` | territorial | −0,0555 | 0,95 | 0,0187 | não |
| `rede_grupo_Municipal` | estrutural | +0,0303 | 1,03 | 0,0227 | **sim** |
| `nome_regiao_Nordeste` | territorial | −0,0280 | 0,97 | 0,0266 | não |
| `mun_n_alunos_lag1` | historico_municipal | +0,0228 | 1,02 | 0,0191 | não |
| `rede_grupo_Outros` | estrutural | −0,0006 | 1,00 | 0,0001 | não *(4 folds)* |

**Achado 11, a direção é inequívoca e estável.** `mun_taxa_alfab_lag1` tem coeficiente
−0,427 e razão de chances **0,65**: um desvio-padrão a mais na taxa de alfabetização
do município em 2023 reduz em 35% a chance de a criança não se alfabetizar em 2024. O
desvio entre folds é 0,008, ou seja, 1,8% da magnitude. A única coluna que troca de
sinal entre folds é `rede_grupo_Municipal`, e a contribuição dela é a 13ª de 14 — não
há afirmação de direção apoiada nela.

**Achado 12, a rede estadual aparece como fator protetor.** `rede_grupo_Estadual` tem
OR **0,73** e a leitura SHAP concorda: matrícula na rede estadual empurra a
contribuição para −0,20 em log-odds, contra +0,05 na municipal. A ressalva é
obrigatória e vai junto de qualquer uso disso: redes estaduais estão concentradas em
municípios de perfil diferente, e o modelo não separa o efeito da rede do efeito de
onde ela está. É associação, não efeito de política.

**Achado 13, o Sul aparece com sinal positivo, e isso surpreendeu.**
`nome_regiao_Sul` tem OR **1,32**: dada a taxa de alfabetização do município e a da UF
em 2023, estar no Sul sobra como risco residual, o oposto do que a taxa bruta da
região sugere. A hipótese mais provável está numa medição que a modelagem já tinha
registrado por outro caminho, a de que Santa Catarina tem 70,1% de presença na prova
contra 98,1% do Ceará. Onde menos crianças comparecem, a taxa municipal de 2023 é
otimista, e o modelo linear compensa isso com um deslocamento regional. Não é achado
sobre a qualidade do ensino no Sul; é achado sobre a comparabilidade das taxas entre
territórios de cobertura muito diferente, e o fato de `mun_taxa_presenca_lag1` entrar
ao lado com OR 0,89, na direção esperada, reforça a leitura. É hipótese com evidência
indireta, não conclusão medida.

---

## 7. A forma da relação nas quatro variáveis principais

**Pergunta:** o efeito é linear, satura, ou muda de regime?
**Decisão:** se a leitura executiva pode ser dita como "quanto mais X, menos risco" ou
precisa de faixa.

O eixo x é o valor **cru** da variável e o eixo y soma os valores SHAP de todas as
colunas que ela gera, incluindo o indicador de nulo. A dispersão vertical em cada x é
interação com o resto do modelo — se a nuvem fosse uma linha, um modelo aditivo
bastaria. As quatro principais por |SHAP| são `mun_media_portugues_lag1`,
`uf_media_portugues_lag1`, `mun_taxa_alfab_lag1` e `rede_grupo`. Figura
`06_dependencia.png`.

**A contribuição de uma variável varia pouco abaixo de 65%.** Essa é a leitura da
figura, e o limite dela precisa ficar na mesma frase: **isso não mede a ordenação do
modelo completo**, que também usa a média de português e as demais variáveis. A
incerteza das posições permanece **não quantificada** — não há intervalo em torno de
nenhuma posição de ranking neste documento. Foi por isso que a coluna que antes se
chamava `ordenacao_fragil` no produto municipal passou a se chamar
`faixa_shap_taxa_municipal`: ela descreve o achatamento do SHAP de **uma** variável
numa faixa, e não julga a ordenação inteira do ranking.

---

## 8. Triangulação — o que é fator e o que é achado sobre a leitura

**Pergunta:** quais variáveis aparecem no topo das três leituras ao mesmo tempo?
**Decisão:** o que pode ser dito como fator na apresentação executiva.

O critério é conservador de propósito: `fator` exige estar no topo de todas as
leituras que **conseguem** ver aquela feature, e ainda derrubar mais que o piso de
ruído do controle negativo. A logística só enxerga oito das vinte features, então para
as outras doze a triangulação é de duas pontas — e a tabela declara isso em
`leituras_disponiveis`, porque reportar consenso de três quando houve duas seria
inflar a confiança do resultado. Figura `07_triangulacao.png`.

| Feature | Família | Pos. SHAP | Pos. permut. | Pos. logística | Leituras | No topo | Veredito |
|---|---|---:|---:|---:|---:|---:|---|
| `mun_media_portugues_lag1` | historico_municipal | 1 | 1 | — | 2 | 2 | **fator** |
| `uf_media_portugues_lag1` | territorial | 2 | 3 | — | 2 | 2 | **fator** |
| `mun_taxa_alfab_lag1` | historico_municipal | 3 | 4 | 1 | 3 | 3 | **fator** |
| `sigla_uf` | territorial | 6 | 2 | — | 2 | 1 | achado sobre a leitura |
| `mun_meta_2024_lag1` | metas | 10 | 5 | — | 2 | 1 | achado sobre a leitura |
| `uf_taxa_alfab_lag1` | territorial | 9 | 6 | 3 | 3 | 1 | achado sobre a leitura |
| `rede_grupo` | estrutural | 4 | 7 | 4 | 3 | 1 | achado sobre a leitura |
| `mun_taxa_presenca_lag1` | historico_municipal | 5 | 8 | 5 | 3 | 1 | achado sobre a leitura |
| `nome_regiao` | territorial | 7 | 9 | 2 | 3 | 1 | achado sobre a leitura |

Distribuição dos vereditos nas 20 features: **3 fatores**, 6 achados sobre a leitura,
11 secundárias.

**Achado 17, três features passam no critério, e só uma passa nas três leituras.**
`mun_media_portugues_lag1` e `uf_media_portugues_lag1` são fatores com duas leituras —
não estão no subconjunto podado, então a logística não as vê. `mun_taxa_alfab_lag1` é
a única que está no topo do SHAP, no topo da permutação **e** em primeiro lugar na
logística. É a variável que sustenta a resposta à pergunta 1 do enunciado sem nenhuma
ressalva de método.

**Achado 18, `sigla_uf` é o caso-modelo de achado sobre a leitura.** Segunda colocada
na permutação, sexta no SHAP, ausente da logística por decisão da engenharia de atributos. Ela não
entra na lista de fatores, e a razão é metodológica: as 26 dummies diluem o SHAP e a
permutação as destrói de uma vez, então as duas técnicas estão medindo objetos
diferentes. Reportá-la como fator descreveria a mecânica do `OneHotEncoder`, não a
alfabetização.

**Achado 19, seis features ficam em "achado sobre a leitura".** Além de `sigla_uf`,
aparecem ali `rede_grupo`, `nome_regiao`, `mun_meta_2024_lag1`, `uf_taxa_alfab_lag1` e
`mun_taxa_presenca_lag1`: estão no topo de uma leitura e não das outras. Todas passam
do piso de ruído, então nenhuma é descartável — mas nenhuma sustenta sozinha uma frase
de recomendação.

---

## 9. A leitura restrita a quem tem histórico de qualidade

**Pergunta:** o ranking global é média de populações que o modelo entende de formas
diferentes?
**Decisão:** publicar a leitura global ou substituí-la pela restrita.

A modelagem mediu ROC-AUC **0,6706** onde o lag vem da Gold (286.358 alunos, 86,6% do
teste) contra **0,5598** no agregado `rede 3` e **0,5227** sem histórico, e deixou o
alerta: uma leitura única mistura populações que o modelo entende de formas muito
diferentes. A verificação é direta — refazer a agregação dentro de
`fonte_lag_municipal = gold` e comparar.

| Feature | Global | Gold | Variação |
|---|---:|---:|---:|
| `mun_media_portugues_lag1` | 0,2709 | 0,2776 | +2,5% |
| `uf_media_portugues_lag1` | 0,0827 | 0,0890 | +7,6% |
| `mun_taxa_alfab_lag1` | 0,0653 | 0,0690 | +5,6% |
| `rede_grupo` | 0,0574 | 0,0526 | −8,4% |
| `mun_taxa_presenca_lag1` | 0,0557 | 0,0630 | **+13,2%** |
| `sigla_uf` | 0,0494 | 0,0554 | +12,2% |

| Família | Global | Gold |
|---|---:|---:|
| historico_municipal | 57,2% | 56,9% |
| territorial | 27,6% | 29,6% |
| estrutural | 9,1% | 7,9% |
| metas | 4,8% | 4,6% |
| historico_escolar | 1,35% | 0,95% |

**Achado 20, a leitura global não é artefato da coalescência — e isso era falseável.**
A correlação de Spearman entre as duas ordenações de features é **0,994**, a primeira
colocada é a mesma e as participações por família mudam pouco. A verificação que a
modelagem exigiu foi feita e passou, o que permite publicar o número agregado sem a
ressalva que se esperava precisar.

**Achado 21, o que mais se move é a presença.** `mun_taxa_presenca_lag1` ganha 13,2%
de importância na fatia Gold e passa da quinta para a quarta posição, enquanto a
família de histórico escolar encolhe de 1,35% para 0,95%. Faz sentido mecânico: onde o
lag veio do microdado, a taxa de presença de 2023 é medida sobre a mesma fonte, e não
herdada de um agregado do INEP.

---

## 10. As três hipóteses, e o que a medição fez com elas

As hipóteses foram escritas com critério de falseamento **antes** desta etapa, na EDA.
Estão em `reports/metrics/interpret_resumo.json` com o número que decidiu cada uma.

| Hipótese | Enunciado | Critério de falseamento | Veredito |
|---|---|---|---|
| **H1** | o contexto municipal de 2023 é o preditor dominante do risco individual em 2024 | uma família não municipal aparecer no topo do SHAP | **sustentada** |
| **H2** | o histórico escolar verdadeiro acrescentaria sinal além do município? | a família de histórico escolar não aparecer no fim da lista | **não verificável** |
| **H5** | a presença municipal anterior está associada ao risco observado entre presentes | `mun_taxa_presenca_lag1` não estar no topo 5 do SHAP e não derrubar mais que 0,002 | **sustentada** |

**H1 é sustentada nas duas réguas.** `historico_municipal` lidera tanto o SHAP quanto
a permutação em bloco, e as duas ordenações de família são idênticas nas cinco
posições.

**H2 é não verificável, e essa é a resposta honesta.** A família de histórico escolar
ficou em 5º de 5 nas duas leituras, e abaixo do piso de ruído. A tentação seria
concluir "o histórico escolar não importa". Não é o que foi medido: a família se
resume a `tem_historico_escola`, uma única flag, porque o bloco `esc_*` é ligado por
identificadores **reciclados** entre edições. O que ficou de fora não é um histórico
escolar longitudinal verdadeiro — é uma chave que não permite acompanhamento. A
pergunta segue aberta e exigiria um vínculo longitudinal válido de aluno e escola, que
esta base não oferece.

**H5 é sustentada, com a ressalva do método.** `mun_taxa_presenca_lag1` está em 5º no
SHAP, cai 0,00261 na permutação (acima do piso de 0,002) e é 5ª na logística, com OR
0,89 na direção esperada. Mas a triangulação a classifica como *achado sobre a
leitura*, não como fator, porque é 8ª na permutação. A associação existe; a hierarquia
dela entre as demais variáveis não é estável entre técnicas.

---

## 11. Reprodutibilidade

| Item | Valor |
|---|---|
| `RANDOM_STATE` | 42, propagado à amostragem do SHAP e às repetições da permutação |
| Dataset | MD5 `cc2215565c04d14a5ad97dbfcdda31b5`, conferido contra o `joblib` antes de qualquer conta |
| Versões | `scikit-learn 1.8.0`, `lightgbm 4.7.0`, `shap 0.52.0` |
| Comando | `python -m src.evaluation.interpret` |
| Custo | 618 s |
| Saídas | 8 CSV + 1 JSON em `reports/metrics/interpret_*`, 7 figuras em `images/interpretabilidade/` |
| Notebook | `notebooks/04_interpretabilidade.ipynb`, que só lê os artefatos e desenha |

**Ressalva de proveniência:** `interpret_resumo.json` recebeu edição manual para gravar o
veredito "não verificável" de H2 sem retreinar o campeão. O conteúdo confere com o que
`src/evaluation/interpret.py` produziria hoje, e a edição está registrada no
`GUIA_DE_EXECUCAO.md`. Os oito CSV são saída direta do comando.

---

## 12. Síntese, e o que não se pode concluir daqui

**A resposta à pergunta 1 do enunciado, em uma frase:** o que mais impacta o risco de
não alfabetização, neste modelo, é o desempenho do próprio município no ano anterior —
proficiência média em português acima de tudo, taxa de alfabetização em seguida —, com
o estado em que o município está respondendo por outro quarto da explicação.

**A resposta à pergunta 5, por grupo de variável:** histórico municipal 57,2%,
território acima do município 27,6%, estrutural 9,1%, metas 4,8%, histórico escolar
1,3%.

**Cinco limites que acompanham obrigatoriamente esses números.**

**Importância não é causalidade.** O SHAP diz o que o modelo usou para prever, não o
que aconteceria se alguém interviesse na variável. Elevar a média de português de um
município não é uma intervenção — é o resultado que se quer, escrito de outra forma. A
pergunta causal exige desenho causal (experimento, diferenças-em-diferenças, variável
instrumental, pareamento), e esta base não o oferece.

**Toda feature aqui é contextual, e aplicá-la a uma criança é falácia ecológica.** O
modelo descreve o território em que a criança está, não a criança. É o mesmo achado
que a modelagem mediu por outro caminho: embaralhando o alvo dentro de cada município, a
AUC cai de 0,6667 para apenas 0,6636. Quase tudo o que o modelo sabe é sobre o
município, e nada sobre o indivíduo — não há na base nível socioeconômico, cor/raça,
idade, frequência ou histórico escolar da criança.

**A incerteza das posições não foi quantificada.** Nenhum ranking deste documento vem
com intervalo. As tabelas de permutação trazem desvio entre repetições, o que dá uma
noção de estabilidade da *queda*, não da *posição*. Onde as quedas são próximas — e da
5ª à 10ª colocada elas estão todas entre 0,0023 e 0,0030 — a ordem entre elas não deve
ser lida como um resultado. O achado 10 mostra que a ordem já muda só por trocar de
técnica.

**A leitura de faixa do SHAP não julga a ordenação do modelo.** A seção 7 descreve a
forma da contribuição de uma variável isolada. Ela não autoriza afirmação sobre a
qualidade do ranking municipal completo, que depende de todas as variáveis juntas.

**O ranking descreve 2024.** O teste de drift mostrou que a estrutura
territorial se desloca entre edições — o modelo reduzido a UF, região e rede cai de
0,6306 dentro de 2024 para 0,6099 quando treinado em 2023. A ordem de importância
medida aqui vale para esta coorte, e projetar 2026 a partir dela assumiria uma
estabilidade que a medição desmente.

**E o limite que vale para tudo acima:** a reserva de teste em que estas medições foram
feitas já havia sido consultada por decisões de seleção anteriores. Nada aqui é
confirmação independente — é leitura descritiva de um modelo retrospectivo. O gate para
mudar isso é uma amostra ainda não consultada, com a disponibilidade temporal das fontes
documentada; até lá, o status fixado em `src/evaluation/protocolo.py` acompanha o modelo
em toda leitura que passe por `carregar_campeao()`.
