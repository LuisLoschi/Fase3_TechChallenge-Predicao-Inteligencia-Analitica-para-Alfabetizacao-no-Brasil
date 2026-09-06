# Documentação técnica

## Contexto e método

CRISP-DM: Evaluation, com as decisões de preparação e de entendimento do problema que a
sustentam. O objetivo, população, custos qualitativos e limites estão no `contrato_temporal.md`.
O projeto classifica registros de alunos, mas seu sinal é predominantemente contextual.

## Dados e qualidade

Gold e seis fontes INEP, anos 2023/2024; manifesto SHA-256 em `data/manifesto_fontes.json`.
Configuração central em `src/config.py`, com `TC_DATA_DIR` opcional.
O pipeline confere cardinalidade dos joins, domínio do alvo, cobertura, reconciliação e
exclusão de medidas ausentes. A auditoria original está em `auditoria_camada_gold.md`.
Roraima não está na base; cobertura histórica varia entre UFs. Não há dado socioeconômico integrado.

## Preparação e prevenção de vazamento

O alvo é `1 - alfabetizado`; a proficiência contemporânea determina o rótulo e fica fora de X.
Identificadores servem para joins/partições, não são preditores. Alunos sem medida válida
são excluídos; o alvo nunca é imputado. Indicadores municipais e de UF usam 2023.
Cadastro e caderno vêm da coorte de 2024; anterioridade real de publicação não foi comprovada.

`Pipeline` contém `ColumnTransformer`, `SimpleImputer`, `StandardScaler` e `OneHotEncoder`.
Esses estimadores são ajustados no treino de cada fold. Categorias inéditas são tratadas
pelo encoder. O artefato serializado recebe as colunas do dataset de modelagem.

### Matriz anti-vazamento

Doze vetores de vazamento identificados no projeto, com a defesa correspondente. Os relatórios
de etapa referenciam esta matriz pelo número do vetor.

| # | Vetor de vazamento | Defesa |
|---|---|---|
| 1 | `proficiencia` do ano corrente, que **é o alvo reescrito**: o rótulo é um corte em 743,0 e a AUC dá 1,000 | removida de `X`; usada só como agregado de 2023 |
| 2 | agregados do mesmo ano, que embutem a média do próprio alvo | todo agregado calculado **exclusivamente sobre 2023** |
| 3 | `proporcao_aluno_nivel_0..8` de 2024 | fora do modelo; só na camada estratégica descritiva |
| 4 | join longitudinal falso por `id_aluno`: 1,51 M de pares, só 0,12% na mesma escola | proibido, travado em `tests/test_dados.py` |
| 5 | `id_aluno` como numérica, cujo prefixo codifica a UF | descartada da lista de atributos |
| 6 | imputação e escala ajustadas antes do split | `ColumnTransformer` dentro do `Pipeline`, com `fit` só no treino do fold |
| 7 | memorização territorial | `GroupShuffleSplit` e `StratifiedGroupKFold` por `id_municipio` |
| 8 | ausentes recebendo `alfabetizado = 0` por omissão | filtrados na Gold (`presenca = 1`); em 2023 usados **só** para a taxa de presença |
| 9 | metas que embutem a taxa observada | tratadas como lag disfarçado; só metas defasadas em relação ao alvo |
| 10 | seleção de atributos e hiperparâmetros olhando a reserva | **defesa não alcançada** — a ablação que decidiu o conjunto de atributos rodou sobre a coorte inteira, e a reserva não é independente dela. O experimento hoje separa a reserva antes de qualquer seleção, o que vale para decisões futuras; a independência deste resultado só se recupera com amostra nova |
| 11 | amostragem para tuning quebrando a estrutura de grupo | amostrar **municípios inteiros**, nunca alunos individuais |
| 12 | intervalo de confiança artificialmente estreito | bootstrap reamostrando **municípios**, não alunos |

O vetor 10 é o único em que a defesa planejada não se sustentou, e essa é a razão de todo o
material deste projeto ser declarado retrospectivo e exploratório.

A ablação que decidiu o conjunto de atributos usou a base inteira, e por isso a reserva não é
independente dela. `scripts/experimento_b5.py` separa a reserva antes de amostrar o
desenvolvimento, e as seeds de validação cruzada não alteram essa partição — o que protege
decisões futuras. Não existe recuperação retroativa de teste intocado com os mesmos dados já
examinados.

## Modelagem individual

Dummy, heurísticas, logística, Random Forest e LightGBM foram comparados em folds agrupados.
Tuning usa municípios do desenvolvimento; calibração e limiares são decididos nesse conjunto.
O LightGBM publicado é congelado e exploratório; o carregador anexa `STATUS_VALIDACAO`.
As métricas estão no README e em `metrics/campeao.json`; bootstrap é por município.
O teste de integração recalcula ROC-AUC e o hash real do dataset.

## Interpretabilidade e uso

SHAP explica o estimador salvo, recuperando nomes das colunas transformadas. Permutação em
famílias considera correlação entre atributos. A importância é associativa e condicional ao modelo.
Histórico escolar verdadeiro não é verificável com IDs reciclados. A presença anterior
prediz associação entre presentes; não identifica o desempenho de alunos ausentes.
Uma curva SHAP de um atributo não prova fragilidade da ordenação completa.

## Produtos municipais

O ranking usa escores OOF nacionais e médias ponderadas por `peso_aluno`.
A validade populacional da ponderação depende de hipóteses sobre não resposta.
Baixa variação dos escores dentro de um município não equivale a baixa incerteza do risco real.
O cache só é reutilizado se hash dos dados, seed, atributos e hiperparâmetros conferirem.

Clusters descrevem 2024; taxas médias dos perfis são médias municipais, não estimativas
populacionais regionais. Risco, perfis e metas estão detalhados em `aplicacao_estrategica.md`.

Metas usam prior, suavização, deriva e dispersão aprendidos fora do fold avaliado.
O porte de origem evita usar informação de tamanho da coorte de destino para prever 2024.
A distribuição normal de trabalho é censurada em 0–100; probabilidades e quantis são coerentes
com essa censura. A validação municipal mede calibração e cobertura; não estabelece estabilidade temporal.
O porte onde a²/n=c² é referência descritiva, sem interpretação causal ou regra de elegibilidade.

## Experimentos descartados

Cada linha registra o número que motivou o descarte. A régua do projeto é a dispersão do
próprio desenho: efeito menor que o desvio entre folds não é efeito.

| # | Experimento | Medição que o descarta | Destino |
|---|---|---|---|
| 1 | `serie` como preditor | constante = 2 (2º ano EF), cardinalidade 1 | fora do dataset |
| 2 | `meta_alfabetizacao_brasil`, `percentual_participacao_brasil` | constantes dentro de 2024 | fora do dataset |
| 3 | `id_aluno` como variável numérica | o prefixo codifica a UF: seria proxy territorial disfarçado | fora, e travado em teste |
| 4 | join longitudinal por `id_aluno` | 1,51 M de pares, só 0,12% na mesma escola | proibido, travado em `tests/test_dados.py` |
| 5 | bloco `esc_*`, 9 atributos de escola | join real 0,5581 contra **0,5601 ± 0,0006** de um valor sorteado dentro da mesma UF, e 0,4998 sorteando de qualquer UF; no multivariado, −0,00038 [−0,00126; +0,00049], p 0,36 | controle negativo declarado |
| 6 | suavização bayesiana da taxa escolar | a AUC cresce monotonicamente com `k` — 0,561 em `k=0`, 0,642 em `k=100`, 0,653 em `k=500` — e nunca alcança os 0,654 da taxa municipal pura: o melhor `k` é o maior da grade, ou seja, a suavização funciona apagando a escola | imputação documentada com `k` fixo |
| 7 | `esc_desvio_vs_municipio` | a persistência do desvio escolar entre 2023 e 2024 é 0,009 — artefato de super-subtração | fora do dataset |
| 8 | percentis de proficiência do microdado, 5 atributos | +0,00037 [−0,00080; +0,00153], p 0,51, contra desvio de 0,0134 entre folds — 36 vezes o efeito; na execução com reserva separada antes da amostragem, +0,00135 com desvio de 0,00276. As duas medições incluem zero | construídos, fora de `FEATURES_MODELO` |
| 9 | `preenchimento_caderno` | registrado no instante da prova que produz o alvo; 936 alunos com rótulo negativo por ausência de medida | vazamento contemporâneo; as 936 linhas saem do treino |
| 10 | `min_frequency = 0,01` em todas as categóricas | colapsaria SE, TO, AP e AC — as quatro UFs abaixo de 1% da coorte — num único nível, na variável de 49 pp de amplitude | mantido só no `caderno` |
| 11 | calibração isotônica | ajustada em quatro folds e medida no quinto, piora o Brier de 0,22233 para 0,22244; os platôs custam ainda 0,0004 de AUC | rejeitada pelo próprio critério; curva guardada desligada |
| 12 | limiar de máximo F1 | alerta 74,1% da coorte | publicado o limiar de capacidade: 12,9% de alerta, recall 21,1%, precisão 0,620 |
| 13 | Random Forest como campeão | empata com o LightGBM (−0,00015 [−0,00150; +0,00120], p 0,776) e custa 89 s por fold contra 17 s | LightGBM, por custo e por `TreeExplainer` exato |
| 14 | `k = 5` na clusterização | o grupo extra se define pelo share de rede estadual no Sul, que é fato de rede e não padrão de alfabetização | `k = 3` |
| 15 | shrinkage sobre o escore do modelo | `k = 0,046` aluno — o escore já é territorial e não há erro amostral para encolher | suavização aplicada à taxa observada, `k = 55,1` |

## Resultados que contrariam a expectativa inicial

Experimentos negativos bem medidos sustentam o desenho tanto quanto os positivos, e vários
aqui redefiniram o produto.

**A suavização necessária não é a do modelo, é a da taxa observada.** O escore atribui
probabilidades quase idênticas aos alunos de um mesmo município, então encolhê-lo é um
não-evento: `k` de 0,046 aluno. A taxa observada é que carrega erro amostral, com `k` de 55,1
alunos. O efeito é operacional e grande: ordenado pela taxa crua, o topo 50 traz **21
municípios com menos de 50 alunos avaliados**, um deles com risco de 100,0%; ordenado pelo
escore, traz **2**.

**A baixa cobertura não forma um perfil próprio.** O decil de menor presença se distribui
**67,3%** no grupo de risco alto e **32,4%** no intermediário, em vez de se concentrar num
agrupamento próprio. A presença entra como gradiente colado ao desempenho — 88,3% de presença
no grupo de risco contra 96,9% no consolidado —, e por isso o terceiro perfil se chama "risco
alto e cobertura menor".

**O espaço municipal é um gradiente, não uma coleção de tipos.** A silhueta escolhe o menor
`k` da grade e todas as silhuetas são baixas: 0,2487 em `k = 3` contra 0,1740 em `k = 4`. A
inércia cai suavemente — 5.078, 3.034, 2.363, 1.821 e 1.418 —, sem cotovelo. Os perfis são
recortes descritivos úteis, não classes naturais.

**A barra correta do baseline é mais alta do que parecia, e o ganho do modelo é pequeno.**
Reimplementada como estimador e passada pelos mesmos folds agrupados, a regra de uma variável
vale 0,6337, e não os 0,654 medidos em outro esquema de validação: o split agrupado cobra dela
o mesmo que cobra do modelo. O ganho do campeão sobre ela é de **+0,0274 [+0,0116; +0,0433],
p 0,009** — real, significante e menor que a largura do intervalo do próprio campeão.

**A busca de hiperparâmetros cabe dentro do próprio ruído.** O ganho sobre os parâmetros de
partida é de +0,00223 [−0,00106; +0,00552], p 0,133, contra amplitude de 0,0076 entre as 40
configurações sorteadas e desvio de 0,0276 entre folds. Os parâmetros escolhidos ficam por
serem os melhores disponíveis, não por superioridade demonstrada.

**Mais dado não é o caminho.** De 979 para 3.309 municípios de treino — 3,4 vezes mais dado —
a validação sobe 0,0039. O teto é informacional, não amostral: a base não tem nenhuma variável
sobre a criança.

**O sinal é de grão municipal, e isso foi medido diretamente.** Embaralhar o alvo dentro de
cada município preserva a taxa municipal e destrói todo o resto; a AUC cai de 0,6667 para
0,6636. Tudo o que o modelo sabe além da média do próprio município vale **0,0031**.
Embaralhando o alvo globalmente o nulo fica em 0,5001 ± 0,0016, então o sinal existe — a
questão nunca foi essa, foi de que grão ele é. No grão municipal o R² é 0,624 na base completa
e 0,774 nos municípios com 200 ou mais alunos avaliados.

**O agrupamento do SHAP por família corrige um viés medido, não hipotético.** Somar as quedas
das colunas isoladas da família municipal dá 0,0513; embaralhar a família junta dá 0,0828,
1,61 vez mais. Na territorial a razão é 1,90. Nas três famílias sem redundância interna fica
entre 0,97 e 1,04 — o ganho aparece exatamente onde a colinearidade está.

**A leitura agregada resiste ao recorte por qualidade do histórico.** Refeito o SHAP apenas
onde o lag vem da Gold, o Spearman entre as duas ordenações é 0,994 e a primeira colocada é a
mesma. Era uma verificação falseável, e passou.

**O `caderno` se comporta como controle negativo em todas as medições.** AUC univariada 0,4987;
permutação no teste derruba 0,00031, contra 0,04203 do controle positivo. A conclusão prática
sobre versões de prova se apoia no tamanho do efeito, não em sobreposição de intervalos: com
cerca de 85 mil alunos por caderno, intervalos de cadernos distintos podem não se tocar sem que
isso tenha significado educacional.

## Reprodução, testes e entrega

`scripts/reproduzir.py --etapa tudo` organiza dados, modelos e relatórios, usando o mesmo Python.
Há execução de testes de código sem microdados em `.github/workflows/testes.yml`.
A integração completa roda localmente com dados e artefatos. CI não substitui essa integração.
Relatórios estratégicos são gerados das métricas; o notebook 05 é atualizado junto deles.
As limitações do trabalho e os três gates abertos — amostra não consultada, disponibilidade
temporal das fontes e dimensão socioeconômica — estão no `README.md`.
