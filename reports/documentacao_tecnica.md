# Documentação técnica — versão revisada

## Contexto e método

CRISP-DM: Evaluation com retorno explícito à preparação e ao entendimento do problema.
O objetivo, população, custos qualitativos e limites estão no `contrato_temporal.md`.
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

O B5 histórico usou a base inteira e participou da seleção de atributos. O script corrigido
reserva os mesmos municípios antes de amostrar o desenvolvimento. As seeds de CV não alteram
a reserva. O resultado novo tem arquivo e manifesto próprios; a evidência antiga não foi apagada.
Não existe recuperação retroativa de teste intocado com os mesmos dados já examinados.

## Modelagem individual

Dummy, heurísticas, logística, Random Forest e LightGBM foram comparados em folds agrupados.
Tuning usa municípios do desenvolvimento; calibração e limiares são decididos nesse conjunto.
O LightGBM histórico permanece congelado e exploratório. O carregador anexa `STATUS_VALIDACAO`.
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

## Reprodução, testes e entrega

`scripts/reproduzir.py --etapa tudo` organiza dados, modelos e relatórios, usando o mesmo Python.
Há execução de testes de código sem microdados em `.github/workflows/testes.yml`.
A integração completa roda localmente com dados e artefatos. CI não substitui essa integração.
Relatórios estratégicos são gerados das métricas; o notebook 05 é atualizado junto deles.
As limitações e gates restantes estão em `revisao_cientifica.md`.
