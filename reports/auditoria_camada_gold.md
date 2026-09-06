# Auditoria da camada Gold

> **Revisão de validade (2026-09-05):** estas medições do classificador são históricas e exploratórias. A seleção supervisionada consultou a coorte inteira; a reserva não é independente dessa seleção. O código corrigido e o protocolo atual estão em [revisao_cientifica.md](revisao_cientifica.md).


Conduzida antes de a Etapa 3 escrever a primeira linha de código, por dois analistas em
paralelo e de forma independente. Motivo: a Etapa 3 ia consumir a Gold e o microdado
simultaneamente sem que ninguém tivesse provado que a Gold é fiel, nem medido se o
microdado paga o custo de ser uma segunda fonte.

As decisões que saíram daqui estão no `PLANO_EXECUCAO.md`. Este documento guarda a
evidência que as sustenta.

---

## 1. A Gold é fiel ao microdado

Reconciliação linha a linha, não por contagem:

| Verificação | 2023 | 2024 |
|---|---|---|
| Linhas da Gold | 1.503.058 | 1.852.788 |
| Microdado com `presenca = 1` | 1.503.058 | 1.852.788 |
| Conjunto de `id_aluno` idêntico | sim | sim |
| `id_aluno` duplicado na Gold | 0 | 0 |
| `alfabetizado` divergente no merge | **0** | **0** |
| `rede` / `caderno` / `preenchimento` divergentes | 0 | 0 |

A Gold é o microdado filtrado em `presenca = 1`, enriquecido com dimensões territoriais e
metas. Perde 512.153 linhas e três colunas: `presenca`, `proficiencia`, `peso_aluno`.

Integridade: 0 `id_municipio` fora da faixa IBGE, 0 escola em mais de um município ou rede,
0 divergência de UF ou nome entre os anos nos 4.841 municípios comuns.

**Limiar do alvo, confirmado nos dois anos:**

```
2023: max(prof | classe 0) = 742,999817   min(prof | classe 1) = 743,000000   violações: 0
2024: max(prof | classe 0) = 742,999634   min(prof | classe 1) = 743,000000   violações: 0
```

**Nulidade estrutural da Gold** — as metas não existem na origem para 2023, porque não há
coluna `meta_alfabetizacao_2023` em nenhuma fonte:

| Coluna | 2023 | 2024 |
|---|---|---|
| `meta_alfabetizacao_municipio` | 100,00% nulo | 15,30% |
| `meta_alfabetizacao_uf` | 100,00% nulo | 1,75% |
| `meta_alfabetizacao_brasil` | 100,00% nulo | 0,001% |
| `percentual_participacao_municipio` | 11,59% | 14,21% |

**Veredito:** serve como está. Todos os defeitos encontrados se resolvem a jusante, no
`feature_store.py`, sem tocar no Databricks. Reexportar 615 MB para ganhar uma coluna que
já está no microdado seria trocar risco real por conveniência marginal.

---

## 2. B1 — `percentual_participacao_*` é vazamento contemporâneo

Não são metas: são a taxa de presença do próprio ano, agregada sobre a mesma coorte que se
quer predizer e medida no instante da prova.

```
gold_2024.percentual_participacao_municipio vs presença observada 2024 (rede 3):
    n = 5.352 | corr = 1,00000 | MAE = 0,002 pp | 100,0% dentro de 1 pp
gold_2024.percentual_participacao_municipio vs presença de 2023:
    corr = 0,4699 | MAE = 5,196 pp
```

Estavam listadas como features no bloco "Metas" da Etapa 3, violando a regra nº 2 da própria
matriz anti-leakage ("todo agregado calculado exclusivamente sobre 2023").

Custo de remover: **+0,0004 de AUC** no modelo completo (0,6574 → 0,6578). A versão
defasada é melhor e legítima: presença municipal de 2023 dá AUC 0,5900 contra 0,5701 da
contemporânea, e já vem na própria Gold de 2023 (corr 0,971 com a observada).

Não existe argumento de custo-benefício a favor de manter. Um avaliador que abra o
`etl-gold.py` acha isso em cinco minutos, e aí a credibilidade da matriz inteira cai junto.

---

## 3. B2 — `peso_aluno` é a correção oficial de não-resposta do INEP

A taxa que o INEP publica é a média de `alfabetizado` ponderada por `peso_aluno`:

| Reconciliação com o INEP | Não ponderada | **Ponderada** |
|---|---|---|
| Município 2023 (n=4.871) | 0,9866 pp | **0,0494 pp** |
| Município 2024 (n=5.516) | 0,3732 pp | **0,0394 pp** |
| UF 2023 (n=23) | 0,5768 pp | **0,0160 pp** |
| UF 2024 (n=25) | 0,4740 pp | **0,0433 pp** |
| `media_portugues` | 1,050 | **0,074** |

O peso reexpande os presentes para a população matriculada, município a município:

```
2023: presentes 1.503.058 | soma dos pesos 1.743.618 | matriculados 1.747.439
2024: presentes 1.852.788 | soma dos pesos 2.109.175 | matriculados 2.120.560
soma_peso / matriculados por município: mediana 1,0005 (2023) e 1,0000 (2024)
```

Efeito nos números que o projeto publica: a taxa nacional de 2024 não ponderada é 59,75% e
a ponderada 59,17%, contra 59,2% oficiais do INEP — o número ponderado bate na segunda casa,
o não ponderado erra 0,55pp.

**Ressalva epistemológica, a declarar junto de qualquer número ponderado.** O fator corrige
a não-resposta assumindo que o ausente se parece com o presente do mesmo estrato (MAR). A
própria EDA sugere que não: a alfabetização entre presentes vai de 53,4% no quintil de menor
presença municipal a 74,6% no de maior. A ponderação **padroniza** o viés e reconcilia com a
fonte oficial; não o elimina.

---

## 4. B3 e B4 — o gap de lag é São Paulo, e é recuperável

### O gap não é ruído difuso

Dos 428.157 alunos de 2024 sem histórico municipal (23,11% da coorte):

| UF | Alunos sem histórico | % da UF |
|---|---|---|
| SP | 395.482 | 100% |
| DF | 22.111 | 100% |
| AC | 10.234 | 100% |
| MG / PR / RS | 330 | ~0,1% |

**99,92% em três UFs que simplesmente não existem em 2023.** Consequência:
`tem_historico_municipio` é, na prática, um indicador de "é de São Paulo", colinear com
`sigla_uf`. A taxa-base quase não difere (39,86% contra 41,54% de risco), então não há
atalho para o alvo — o problema é de cobertura de feature, não de vazamento.

### O agregado do INEP tem São Paulo em 2023

Dos 676 municípios sem microdado de 2023, `rede = 5` (Pública) cobre 79 e `rede = 3`
(Municipal) cobre 625 — os 79 são subconjunto dos 625, e é `rede = 3` quem traz SP.

| Estratégia | Resgata | Gap residual |
|---|---|---|
| `rede = 5` sozinha (a regra travada até aqui) | 12,33% | 10,77% |
| **Coalescência `5 → 3`** | **21,20%** | **1,91%** |

Ganho de 8,86pp da coorte (164 mil alunos), gap residual reduzido 5,6x.

Custo: `rede = 3` é menos fiel que `rede = 5` — MAE 1,73pp e corr 0,983, contra 0,99pp e
0,994. Aceitável diante da alternativa, que é não ter feature nenhuma para 21% da coorte.
Daí a exigência da coluna de proveniência `fonte_lag_municipal`.

> **Cuidado com a leitura fácil deste número.** É tentador dizer que `rede = 5` sozinha
> "descartaria 21,2% da coorte", já que 21,20% é o que a coalescência resgata. Não é isso:
> `rede = 5` resgata 12,33% por conta própria, e o que se perde ao dispensar o fallback são
> os 8,86pp de diferença. A confusão entre as duas leituras foi capturada por
> `test_coalescencia_de_rede_e_necessaria_para_cobrir_sao_paulo`, que falha se o ganho
> medido não bater — foi assim que o número errado apareceu antes de chegar ao relatório.

---

## 5. B5 — o microdado não é a matéria-prima das melhores features

Quatro conjuntos sobre os alunos de 2024 (`preenchimento_caderno = 1`), LightGBM idêntico,
`GroupShuffleSplit` 80/20 por `id_municipio`, seed 42, IC por bootstrap pareado
reamostrando os 1.104 municípios de teste:

| Conjunto | Features | ROC-AUC | Δ vs. A (IC95 pareado) |
|---|---|---|---|
| **A** — só Gold 2023 | 16 | **0,6552** | — |
| **A2** — A + participação de 2023 (coluna da própria Gold) | 17 | 0,6569 | +0,0017 [+0,0000; +0,0039] |
| **B** — A + percentis de proficiência do microdado | 32 | 0,6574 | **+0,0022 [−0,0005; +0,0045]** |
| **Bp** — B + participação contemporânea de 2024 | 34 | 0,6578 | +0,0026 [−0,0003; +0,0050] |

A afirmação original da Etapa 3 — "o microdado bruto, **não a Gold**, [é] a matéria-prima
das melhores features" — é falsa duas vezes:

1. A taxa de presença de 2023 **está na Gold** (`percentual_participacao_municipio` da linha
   de 2023, corr 0,971 e MAE 0,19pp contra a observada).
2. A proficiência média municipal **está no agregado do INEP** (`media_portugues`, MAE 0,074
   contra o microdado, para 4.950 municípios em 2023).

A vantagem univariada de `mun_prof_media_lag1` (0,660 contra 0,654 da taxa municipal)
evapora no multivariado porque as duas medem a mesma coisa.

**Ressalva.** O experimento usa um split, um conjunto de hiperparâmetros e um modelo. O
delta é frágil nos dois sentidos. Antes de cortar os percentis de proficiência do
`feature_store.py`, replicar com `StratifiedGroupKFold(5)` e 3 seeds — está na checklist da
Etapa 3. O que **não** depende dessa réplica é a correção da justificativa: a de que o
microdado era insubstituível para o lag estava errada.

---

## 6. Arquitetura de fontes: a terceira resposta

A pergunta "Gold ou Gold + microdado?" tinha uma resposta que nenhuma das duas análises
defendia isoladamente:

- **Gold** → coorte, alvo e a maior parte das features de lag de 2023.
- **Agregado municipal do INEP** → `taxa_alfabetizacao` e `media_portugues` de 2023 para os
  625 municípios que a Gold não cobre. É o que recupera São Paulo.
- **Microdado** → `peso_aluno`, `presenca` em grão de escola, reconciliação e testes. Os
  percentis de proficiência ficam opcionais, com o delta medido reportado.

Consumir Gold + raw simultaneamente é legítimo quando a Gold é *lossy por desenho* e o
consumidor precisa do que ela descartou de propósito — o caso aqui. Vira dívida técnica
quando o motivo do bypass não está escrito, ou quando as fontes podem divergir sem que
ninguém perceba. O primeiro foi fechado por este documento; o segundo, por teste.

---

## 7. Defeitos menores corrigidos

- **`test_gold_reconcilia_com_microdado` provava menos do que se dizia.** Comparava só
  `len()` — uma Gold com o subconjunto errado e a mesma contagem passaria. Reescrito para
  comparar conjunto de chaves e igualdade do alvo, nos dois anos.
- **A nota do `config.py` sobre `rede = 5` citava o número errado.** Os 0,37pp são de 2024;
  em 2023, o ano do lag, é 0,99pp com máximo de 56,9pp e 124 municípios acima de 5pp. O
  teste, que só rodava em 2024, passaria raspando o limiar se estendido (0,9866 < 1,0).
- **Asserção vazia em `build_dim_municipio.py`.** O `groupby(...).nunique() == 1` rodava
  *depois* do `drop_duplicates`, quando já era verdade por construção. Movido para antes.
- **Docstring de `loader.carregar_meta_municipio` contradizia o próprio teste** — dizia
  "uma linha por rede, agregue antes", enquanto `test_meta_municipio_e_um_por_municipio_ano`
  prova o contrário.
- **1.185 alunos com `presenca = 1` e proficiência nula** (249 em 2023, 936 em 2024)
  receberam `alfabetizado = 0` do ETL. É imputação silenciosa: ausência de medida virou
  desfecho. Saem do treino.

## 8. Pendência em aberto

**A taxa nacional de 2023 não reconcilia.** Ponderada dá 57,45%; o
`meta_alfabetizacao_brasil.csv` publica 55,9%. No grão UF a ponderada bate exato (MAE
0,016pp), então a divergência está na construção do número nacional do INEP — possivelmente
média simples entre UFs, que é o que o `etl-gold.py` faz na tabela analítica. Se algum
número nacional de 2023 for citado no vídeo ou no README, investigar antes.
