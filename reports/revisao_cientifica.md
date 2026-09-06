# Revisão científica e correções

**CRISP-DM:** retorno de Evaluation a Business Understanding e Data Preparation/Modeling;
nova Evaluation dos cenários municipais. Entregável: código corrigido, evidências e limites revisados.

## Decisão

Manter o classificador como artefato retrospectivo exploratório e os produtos municipais
como apoio à análise. Não declarar validação prospectiva ou conclusão causal.
O teste histórico foi consultado na seleção e não volta a ser intocado por trocar a seed.

## Correções aplicadas

| Problema | Correção | Evidência |
|---|---|---|
| B5 na coorte inteira | reserva fixa antes da amostragem; CV só no desenvolvimento | `experimento_b5_desenvolvimento.csv` e manifesto JSON |
| Teste histórico tratado como independente | status explícito no JSON e no carregador do modelo | `src/evaluation/protocolo.py` |
| Metas ajustadas e avaliadas na mesma amostra | cinco folds por município; prior, k, deriva, dispersão e baseline aprendidos fora do fold avaliado | `estrategia_metas_oof.csv` |
| Porte contemporâneo na avaliação | usa porte de 2023 para prever a taxa de 2024 | `src/modeling/metas.py` |
| Taxas e intervalos impossíveis | distribuição de trabalho censurada em 0–100; probabilidades coerentes nas bordas | `projecao_metas_municipios.csv` |
| Avaliação pontual das metas | bootstrap de 1.000 municípios, ganho pareado de Brier e cobertura | `estrategia_resumo.json` |
| SHAP isolado usado para julgar ranking inteiro | flag renomeada para recorte descritivo; incerteza das posições explicitamente não medida | `ranking_risco_municipal.csv` |
| Histórico escolar inadequado | hipótese sobre escola marcada não verificável | `interpret_resumo.json` |
| Corte de porte tratado como elegibilidade | referência de dispersão, sem regra de avaliação individual | CSV de metas |
| Esforço acumulado comparado com anual | coluna de gap anualizado e roteiro corrigido | `estrategia_funil_2030.csv` |
| Hash comparava só dois metadados | teste calcula também o hash do dataset real | `tests/test_modelagem.py` |
| Cache OOF sem proveniência | valida hash, atributos, seed e parâmetros antes de reutilizar | sidecar JSON em `models/` |
| Reprodução dependente de caminhos locais | `TC_DATA_DIR`, manifesto SHA-256 e orquestrador | `scripts/verificar_insumos.py`, `scripts/reproduzir.py` |

O B5 corrigido executado nesta revisão usa uma seed, três folds e aproximadamente
150 mil linhas, selecionadas por municípios inteiros do desenvolvimento. Serve para
verificar o fluxo e comparar atributos nesse orçamento; não substitui a réplica histórica
de 15 folds nem muda automaticamente o contrato do classificador salvo.
O artefato individual e suas métricas foram preservados; a projeção municipal foi recalculada.

## Pendências externas e gates

- **Amostra nova e disponibilidade temporal:** necessárias para confirmação prospectiva.
- **Dimensão socioeconômica:** falta fonte integrada; requer complemento ou alinhamento com o professor.
- **Dados para a banca:** o grupo deve fornecer as sete fontes indicadas no manifesto. Não há upload automático.
- **Vídeo:** roteiro corrigido não comprova gravação; falta link e duração verificada.
- **Colaboração remota:** commits locais não comprovam revisão por outro integrante; PR real depende de publicação e revisão.

## Referências técnicas

A separação anterior à seleção segue a orientação de
[prevenção de leakage do Scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
CDF e quantis normais são calculados pela
[API de distribuição normal do SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html).
A censura é uma decisão deste projeto, não uma garantia de adequação dada pela biblioteca.
