# Roteiro executivo — versão revisada

**CRISP-DM: comunicação de Evaluation.** Público: gestores educacionais.
Duração planejada: 4min a 4min40, a confirmar por ensaio e gravação.
Gravação e link ainda pendentes. Os tempos abaixo são planejamento, não duração medida.

## 0:00–0:40 — A decisão

Como uma secretaria pode escolher onde aprofundar sua análise de alfabetização e
dimensionar o esforço até as metas? Nosso projeto organiza dados públicos para apoiar
essa decisão. Ele oferece duas listas de prioridade territorial, perfis de municípios
e cenários de metas. Cada produto vem acompanhado de seus limites.

Trabalhamos com registros de alunos de 2023 e 2024, integrados na camada Gold da fase
anterior. Em 2024, a base de modelagem tem cerca de 1,85 milhão de registros válidos,
em 5.517 municípios. Essa cobertura não representa todo o país sem ressalvas.

## 0:40–1:30 — O que o modelo consegue medir

O modelo estima risco a partir de contexto educacional e territorial. Não dispomos de
renda da família, frequência individual ou trajetória escolar confiável. Descobrimos
que os identificadores de escola mudam entre edições; juntar pelo mesmo número poderia
associar escolas diferentes. Por isso, esta base não permite avaliar o valor de um
histórico escolar longitudinal verdadeiro.

O sinal preditivo se concentra no território. O resultado é mais útil para apoiar
análises municipais do que para diagnosticar uma criança. Identificar fatores usados
pelo modelo também não prova que intervir neles causará melhora na alfabetização.

## 1:30–2:20 — Duas perguntas, duas listas

A primeira lista ordena a intensidade do risco contextual. A segunda considera o volume
estimado de crianças nesse contexto. São decisões diferentes: atender a um município
com uma taxa elevada não tem a mesma escala de atendimento de um município populoso.

Nos territórios com posição publicada, o modelo estima cerca de 837 mil
crianças em risco contextual. Os cinquenta primeiros por volume concentram cerca de
243 mil. Essas estimativas dependem dos pesos e das
hipóteses sobre alunos que não participaram da prova.

Não quantificamos a incerteza das posições próximas. As listas apoiam investigação e
planejamento; não devem definir automaticamente quais municípios receberão recursos.

## 2:20–3:20 — Metas: separar distância observada de previsão

A distância até a meta de 2025 é diretamente calculável: meta menos taxa de 2024.
O gap mediano é 2,2 pontos percentuais.
Também construímos cenários com incerteza. Corrigimos a avaliação para que cada município
seja previsto por parâmetros ajustados em outros municípios, e todas as taxas e limites
ficam entre zero e cem por cento.

Os intervalos cobriram aproximadamente 94,6 por cento
dos resultados avaliados. Ainda assim, o método distingue pouco quem fica abaixo da
meta, e o ganho sobre um baseline simples é incerto. Essa avaliação usa uma única
transição anual: não comprova que as probabilidades funcionarão em um ano futuro.

Até 2030, o gap mediano é 15,75 pontos em seis anos,
ou 2,625 pontos por ano numa divisão linear.
Não devemos comparar uma distância de seis anos diretamente com um ritmo anual.

## 3:20–4:20 — Como usar e como evoluir

Nossa recomendação é usar os produtos para priorizar análises locais, verificar cobertura
da prova e discutir capacidade de atendimento. Municípios pequenos precisam de contexto
e incerteza, sem um corte automático de elegibilidade baseado em tamanho.

A revisão também identificou que a seleção histórica de atributos consultou o conjunto
depois usado como teste. Corrigimos o fluxo, mas os resultados antigos continuam
exploratórios. O próximo passo é obter uma amostra ainda não consultada, confirmar quando
cada dado estava disponível e integrar a dimensão socioeconômica prevista no desafio.

A entrega organiza evidências para decisão e torna explícito o que ainda precisa ser
validado. Esse cuidado é parte do valor do projeto para o planejamento educacional.

## Slides e ensaio

Usar os gráficos de comparação, duas ordenações e validação municipal atualizados.
`images/estrategia/10_validacao_metas.png` substitui o gráfico antigo, rotulado como backtest.
Ensaiar, cortar exemplos se necessário e conferir duração final de no máximo cinco minutos.
