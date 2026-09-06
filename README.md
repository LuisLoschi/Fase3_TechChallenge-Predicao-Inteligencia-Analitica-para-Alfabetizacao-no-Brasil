# Tech Challenge — Fase 3: Alfabetização no Brasil

Grupo: Eduardo Rossi | Luis Loschi | Luiza Santos | Vitória Santos | Vyctor Correia

**Resultados exploratórios e retrospectivos, com as limitações declaradas em texto e em código.**
O classificador estima risco contextual individual; a aplicação municipal entrega rankings,
perfis e cenários. Comece pelas [limitações do projeto](#limitações-do-projeto) e pelo
[guia de execução](GUIA_DE_EXECUCAO.md).

> **Status de validação, em uma linha:** a seleção supervisionada de atributos consultou a
> coorte inteira de 2024, então a reserva de teste **não é independente** dessa seleção e
> nenhum número aqui é confirmação prospectiva. O status está fixado em
> `src/evaluation/protocolo.py` e acompanha o modelo em `carregar_campeao()`.

## Contexto do problema

Gestores educacionais precisam identificar territórios que merecem análise e dimensionar
o esforço até as metas de alfabetização. O projeto usa dados de 2023 e 2024 da Fase 2.
Predição e associação apoiam investigação; não identificam efeitos causais de políticas.

## Objetivo analítico

Alvo: `risco_nao_alfabetizacao = 1 - alfabetizado`, positivo para não alfabetização.
Unidade: aluno presente com medida válida em 2024. A população modelada contém
1.851.852 registros, 5.517 municípios e 26 UFs. Roraima não está na base.
O uso atual é retrospectivo; o [contrato temporal](reports/contrato_temporal.md) registra
os limites de uso individual e de antecipação de novas coortes.

## Descrição da base utilizada

Sete fontes: Gold em grão aluno; microdado INEP; agregados de município e UF;
metas municipais, estaduais e nacionais. Veja [origens e contagens](data/README.md)
e [manifesto SHA-256](data/manifesto_fontes.json).

São 20 atributos: histórico educacional de 2023, proveniência e cadastro da coorte de 2024.
Não é correto dizer que todos são de 2023. Não foi integrada dimensão socioeconômica;
essa divergência do objetivo do enunciado permanece pendente de complemento ou alinhamento acadêmico.

## Etapas de modelagem

CSV → Parquet → agregados históricos → dataset → comparação e tuning → classificador salvo
→ interpretação → produtos municipais. Código em `src/`, entrada/preparação em `scripts/`.

A Gold foi reconciliada com o microdado. Foram identificados IDs reciclados entre anos,
retirados registros sem medida válida e excluídos proficiência contemporânea, identificadores
e participação da própria prova dos preditores. Os pesos são usados nas agregações populacionais.

`ColumnTransformer` integra imputação, escala e encoding ao modelo. O ajuste usa apenas
o treino de cada fold. A reserva contém municípios diferentes dos do desenvolvimento.
**A seleção de atributos, porém, consultou toda a coorte:** as métricas não são uma
avaliação confirmatória independente. A ablação em `scripts/experimento_b5.py` separa a
reserva antes de qualquer seleção, o que preserva a partição para decisões futuras sem
tornar independente o resultado já publicado.

## Escolha do algoritmo

Foram comparados Dummy, duas heurísticas, logística, Random Forest e LightGBM padrão/tunado.
O LightGBM salvo foi mantido para preservar o artefato analisado e sua interpretação.
Seu empate com Random Forest foi discutido em termos de custo; não se reivindica
ganho comprovado do tuning. A isotônica não foi aplicada pelo critério de Brier no desenvolvimento.
Resultados históricos e comparações pareadas estão em [modelagem](reports/modelagem.md).

## Métricas de avaliação

Reserva retrospectiva: 330836 alunos em 1104 municípios.
Os intervalos abaixo vêm do bootstrap municipal registrado no artefato histórico;
não corrigem a participação prévia da reserva na seleção.

| Métrica | Valor | IC95 registrado |
|---|---:|---|
| ROC-AUC | 0,6599 | [0,6314; 0,6853] |
| Average Precision (`pr_auc` no código) | 0,5360 | [0,5056; 0,5672] |
| Brier | 0,21794 | [0,21174; 0,22321] |

No limiar de capacidade definido no desenvolvimento: precisão 62,01%, recall 21,06%,
F1 0,3144 e taxa de alerta 12,95% no teste. Matriz: VN 188.438, FP 16.270,
FN 99.568, VP 26.560. São estimativas pontuais; não foram calculados ICs para esse limiar.
Um corte escolhido para 20% do desenvolvimento não garante 20% em outra distribuição.

## Interpretação dos resultados

O modelo usa principalmente contexto municipal e de UF. SHAP e permutação em famílias
consideram atributos correlacionados; coeficientes da logística fornecem uma leitura adicional.
Veja [interpretabilidade](reports/interpretabilidade.md). O valor de histórico escolar
verdadeiro não foi testado porque as chaves não permitem acompanhamento longitudinal.

## Insights encontrados

- O vínculo longitudinal de aluno e escola por ID não é válido nesta base.
- O desempenho varia por cobertura do histórico; não se deve ocultar essa diferença numa métrica nacional.
- Ordenações por taxa e por volume respondem a perguntas diferentes sobre prioridade territorial.
- Os cenários de metas discriminam pouco; isso limita os métodos testados, sem provar imprevisibilidade geral.

## Aplicação prática para políticas públicas

As cinco perguntas são respondidas em [aplicação estratégica](reports/aplicacao_estrategica.md).

| Produto | Conteúdo e limite |
|---|---|
| [Ranking](reports/ranking_risco_municipal.csv) | Risco contextual e volume; posições sem incerteza quantificada. |
| [Perfis](reports/clusters_municipais.csv) | Agrupamentos descritivos de 2024, com sobreposição. |
| [Metas](reports/projecao_metas_municipios.csv) | Gap observado e cenários condicionais entre 0% e 100%. |

A projeção de metas foi avaliada em cinco folds municipais: ROC-AUC
0,5455 [0,5283; 0,5631].
Cobertura dos intervalos: 94,60%
[93,95%; 95,21%].
É avaliação territorial na mesma transição, não validação de ano futuro.
O ganho de Brier sobre prevalência do treino inclui zero no intervalo.

## Limitações do projeto

**Sobre a validade do que foi medido.** A seleção supervisionada de atributos consultou a coorte
inteira de 2024: a reserva de teste não é independente dessa seleção, e trocar a seed do split
não a torna independente. Não há validação temporal do classificador completo — features de lag
para 2023 exigiriam 2022, que não existe —, e a única checagem out-of-time possível usa um modelo
reduzido a UF, região e rede. Toda métrica publicada é retrospectiva e exploratória.

**Sobre o que a base não contém.** Não há nenhuma variável sobre a criança: nível socioeconômico,
cor/raça, idade, frequência ou histórico escolar. O modelo é territorial, e atribuir a uma criança
a característica do seu município é falácia ecológica. As chaves de escola são recicladas entre
edições, o que impede acompanhamento longitudinal e torna **não verificável** — não negativa — a
pergunta sobre o valor do histórico escolar. Roraima não está na base. O calendário de publicação
das fontes não foi auditado.

**Sobre quem entra na medição.** Só alunos presentes com medida válida. Ausência não é sinônimo de
não alfabetização, e a taxa municipal de um território com baixa presença é otimista; a ponderação
por `peso_aluno` corrige não resposta sob hipóteses que não são verificáveis aqui. O modelo regride
para a média e subestima o risco justamente onde ele é maior.

**Sobre os produtos municipais.** Não há intervalo de confiança em torno do risco de cada
município, nem validação temporal do ranking: com duas edições da prova não existe um 2025 contra
o qual conferir a ordenação de 2024. A incerteza das posições não foi quantificada em nenhum
produto. Os cenários de metas são **condicionais** à distribuição adotada — normal censurada em
0–100, uma decisão deste projeto e não uma garantia dada por biblioteca — e não garantem
resultados em 2025; a validação deles é territorial, na mesma transição 2023 → 2024, e não é teste
de ano futuro. O porte de referência da dispersão é descritivo e **não** é regra de elegibilidade:
não deve restringir acesso de município nenhum a política pública. Nada no projeto sustenta
conclusão causal.

## Possíveis evoluções futuras

Três gates estão abertos e dependem de coisas que não se resolvem escrevendo texto: **uma amostra
ainda não consultada**, sem a qual não há confirmação prospectiva; a **documentação da
disponibilidade temporal** de cada fonte, que é o que autorizaria uso prospectivo; e a **integração
de uma dimensão socioeconômica**, hoje ausente por falta de fonte integrada. Depois deles vêm medir
a estabilidade do ranking e pactuar capacidade e custo com gestores. Novos algoritmos e tuning
adicional não são a prioridade antes desses pontos — o ganho do modelo sobre uma regra de uma
variável é de 0,027 de ROC-AUC, e o teto não está no algoritmo.

## Estrutura do repositório

`data/`: fontes, referência, intermediários e dataset; `notebooks/`: cinco análises;
`src/preprocessing/`, `src/modeling/`, `src/evaluation/`, `src/visualization/`: código;
`reports/`: documentação e métricas; `images/`: figuras; `tests/`: verificações;
`requirements.txt`: versões; `.gitignore`: exclusão de dados brutos e artefatos grandes.

## Como reproduzir

Use Python 3.14 e um ambiente virtual. No PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/verificar_insumos.py
.\.venv\Scripts\python.exe scripts/reproduzir.py --etapa dados
.\.venv\Scripts\python.exe scripts/reproduzir.py --etapa relatorios
.\.venv\Scripts\python.exe -m pytest -q
```

`--etapa relatorios` requer o classificador e as métricas de interpretação já gerados.
Para refazer tudo, use `--etapa tudo`: inclui treinos e experimentos demorados.
`TC_DATA_DIR` permite mudar a pasta de dados. Um clone requer os sete CSVs fornecidos pelo grupo;
o manifesto confere a versão, mas não substitui o fornecimento. Veja o guia para a sequência completa.

## Entregáveis

Código, notebooks, métricas, figuras, [documentação técnica](reports/documentacao_tecnica.md),
[limites de validade](reports/limites_de_validade.md), [contrato temporal](reports/contrato_temporal.md)
e [roteiro executivo](reports/roteiro_video.md).
A execução completa depende dos sete CSVs de origem, que não são versionados; o
[manifesto SHA-256](data/manifesto_fontes.json) confere a versão de cada um.
