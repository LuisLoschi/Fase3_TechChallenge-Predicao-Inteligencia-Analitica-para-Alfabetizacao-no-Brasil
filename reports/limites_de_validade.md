# Limites de validade e decisões de protocolo

**CRISP-DM: Evaluation**, com retorno a *Business Understanding* e a *Data Preparation/Modeling*.
Este documento responde a uma pergunta só: o que as medições deste projeto sustentam, e o que
elas não sustentam. As decisões de protocolo que produzem essas garantias vêm descritas com a
evidência que permite verificá-las.

## O que o projeto afirma, e o que não afirma

O classificador é um **artefato retrospectivo e exploratório**. Ele estima risco contextual
associado ao território e serve para priorizar análise; não é diagnóstico de criança, não
autoriza intervenção individual automatizada e não sustenta conclusão causal.

A razão é específica e vale dizer sem rodeio: **a seleção de atributos consultou a coorte inteira
de 2024**, incluindo os municípios que depois formaram a reserva. A reserva, portanto, não é
independente da seleção — e trocar a seed do split não a torna independente, porque o que foi
consultado foi a coorte, não uma partição dela. Nenhuma métrica publicada aqui é confirmação
prospectiva. O status está fixado em código, em `src/evaluation/protocolo.py`, e acompanha o
modelo em `carregar_campeao()`, para que o número não viaje sem a ressalva.

Os produtos municipais apoiam análise. O ranking ordena risco contextual e volume; os perfis são
descritivos e se sobrepõem; a projeção de metas entrega **cenários condicionais**, não previsão
calibrada de 2025.

## Decisões de protocolo, e o que cada uma garante

| Decisão | O que garante | Evidência |
|---|---|---|
| A reserva é separada com a seed fixa do projeto **antes** de qualquer seleção; a validação cruzada roda só no desenvolvimento | que uma decisão de modelagem futura não consulte a reserva | `experimento_b5_desenvolvimento.csv` e o manifesto JSON irmão |
| O limite de validade viaja com o artefato, e não apenas no texto | que quem carrega o modelo receba o status junto | `src/evaluation/protocolo.py`, `tests/test_protocolo.py` |
| Prior, suavização, deriva, dispersão e baseline das metas são aprendidos em cinco folds por município e aplicados fora do fold que os gerou | generalização territorial dentro da transição 2023 → 2024 — não desempenho em ano futuro, já que parâmetros e desfecho vêm da mesma transição | `estrategia_metas_oof.csv` |
| A avaliação das metas usa o porte de 2023 para prever a taxa de 2024 | que a avaliação não se apoie em informação contemporânea ao desfecho | `src/modeling/metas.py` |
| A distribuição de trabalho da projeção é censurada em 0–100, com probabilidades coerentes nas bordas | que o produto publicado não contenha valor impossível para uma taxa | `projecao_metas_municipios.csv` |
| A avaliação das metas vem com bootstrap de 1.000 municípios, ganho pareado de Brier e cobertura | que o resultado seja lido com incerteza, e não como ponto | `estrategia_resumo.json` |
| A faixa de SHAP achatado é marcada como recorte descritivo, e a incerteza das posições é declarada não medida | que uma leitura sobre **uma** variável não seja tomada como julgamento da ordenação inteira | `ranking_risco_municipal.csv` |
| A hipótese sobre histórico escolar é marcada como não verificável | que a ausência de chave longitudinal não seja lida como ausência de sinal | `interpret_resumo.json` |
| O corte de porte é referência de dispersão, sem regra de avaliação individual | que nenhum município seja excluído de política pública por tamanho | `projecao_metas_municipios.csv` |
| O funil publica o gap acumulado e o anualizado lado a lado | que esforço de seis anos não seja comparado com ritmo de um ano | `estrategia_funil_2030.csv` |
| O teste de integração recalcula o hash real do dataset, e não apenas metadados | que o modelo em disco corresponda aos dados que o treinaram | `tests/test_modelagem.py` |
| O cache out-of-fold valida hash, atributos, seed e parâmetros antes de ser reutilizado | que um cache antigo não contamine uma execução nova | sidecar JSON em `models/` |
| As sete fontes são conferidas por SHA-256, e os caminhos são configuráveis por `TC_DATA_DIR` | que métricas de extrações diferentes não sejam misturadas | `scripts/verificar_insumos.py`, `scripts/reproduzir.py` |

**Sobre a força da evidência da ablação de atributos.** A execução registrada em
`experimento_b5_desenvolvimento.csv` usa uma seed, três folds e cerca de 150 mil linhas,
selecionadas por municípios inteiros do desenvolvimento. Ela verifica o fluxo e compara os
conjuntos nesse orçamento; é evidência fina, e não equivale a uma execução de quinze folds. A
comparação que efetivamente decidiu o conjunto de atributos em uso é a que rodou sobre a coorte
inteira, e é dela que vem a falta de independência descrita acima.

**Sobre o modelo salvo.** O classificador e suas métricas são preservados deliberadamente.
Retreinar com os mesmos dados não recuperaria a independência da reserva, já consultada:
produziria números diferentes com o mesmo alcance epistêmico, ao custo de invalidar todas as
métricas publicadas e conferidas por hash. A projeção municipal, essa sim, é recalculada.

## O que continua aberto

- **Confirmação prospectiva.** Depende de uma amostra que nenhuma decisão deste projeto tenha
  consultado. Não existe recuperação retroativa com os mesmos dados já examinados.
- **Disponibilidade temporal das fontes.** O contrato de inferência declara um momento hipotético
  de previsão; a data real de publicação de cada atributo não foi auditada. Ver
  [`contrato_temporal.md`](contrato_temporal.md).
- **Dimensão socioeconômica.** As sete fontes disponíveis não contêm nenhum atributo
  socioeconômico, em nenhum grão. Ver a seção seguinte.
- **Acesso aos dados.** As sete fontes não são versionadas. O manifesto confere a versão de cada
  uma; entregá-las a quem for reproduzir o trabalho continua sendo tarefa do grupo.

## A dimensão socioeconômica

O objetivo do desafio menciona a dimensão socioeconômica, e ela não está no modelo. Isso é
propriedade das fontes disponíveis, não omissão de engenharia.

As sete fontes somam 48 nomes de coluna distintos, quatro deles metadados do próprio ETL.
Nenhum é socioeconômico, em nenhum grão:

| Fonte | Colunas | O que carrega |
|---|---:|---|
| Gold em grão aluno | 25 | identificação, território, rede, caderno, metas e o alvo |
| microdado do aluno | 12 | caderno, série, rede, presença, proficiência e peso |
| agregados de município e UF | 15 cada | taxa, média de português e a distribuição por nível |
| metas de município, UF e Brasil | 13, 12 e 11 | taxa base, as sete metas anuais e participação |

Não há renda, escolaridade do responsável, localização urbana ou rural, nem o Indicador de Nível
Socioeconômico do próprio INEP. `rede` é organização administrativa e não é proxy socioeconômico
defensável: ela separa quem oferta, não quem cursa. Região e UF carregam desigualdade
socioeconômica confundida com tudo mais que varia no território, e chamá-las de dimensão
socioeconômica seria renomear o que já existe.

**Duas saídas, e o custo de cada uma.** Integrar uma fonte externa pede um indicador municipal
anterior ao momento de previsão, com `id_municipio` de sete dígitos como chave — a mesma que une
todas as bases do projeto. O INSE do INEP tem grão de escola, e `id_escola` é chave reciclada
nesta base, o que empurra a escolha para o grão municipal. Exige, na ordem: ano de referência
anterior à previsão, auditoria de cobertura contra os 5.517 municípios da coorte, registro no
manifesto SHA-256 e retreino — o que muda o hash do dataset e substitui as métricas publicadas.
A alternativa é declarar a reformulação do recorte e validá-la academicamente. Nenhuma das duas é
decisão técnica isolada.

## Referências técnicas

A separação anterior à seleção segue a orientação de
[prevenção de leakage do Scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
CDF e quantis normais são calculados pela
[API de distribuição normal do SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html).
A censura em 0–100 é uma decisão deste projeto, não uma garantia de adequação dada pela
biblioteca.
