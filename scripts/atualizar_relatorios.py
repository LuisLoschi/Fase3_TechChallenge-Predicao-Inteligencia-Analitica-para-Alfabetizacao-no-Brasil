"""Atualiza documentos e notebook estratégico a partir das métricas em disco.

Não ajusta modelos. Rode após `python -m src.modeling.strategic`.
"""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.preprocessing.pipeline import FEATURES_MODELO


def numero(valor, casas=2):
    return f"{valor:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def escrever(nome, texto):
    (config.ROOT / nome).write_text(texto.strip() + "\n", encoding="utf-8")


def main():
    resumo = json.loads((config.DIR_METRICS / "estrategia_resumo.json").read_text(encoding="utf-8"))
    campeao = json.loads((config.DIR_METRICS / "campeao.json").read_text(encoding="utf-8"))
    metas = resumo["metas"]
    validacao = metas["validacao_municipal_2023_2024"]
    ic = validacao["intervalos_bootstrap"]
    familias = pd.read_csv(config.DIR_METRICS / "interpret_familias.csv").query("recorte == 'global'")
    ranking = pd.read_csv(config.DIR_REPORTS / "ranking_risco_municipal.csv")
    avaliados = ranking[ranking.modelo_avaliado]
    total_risco = avaliados.criancas_em_risco.sum()
    topo = avaliados.nsmallest(50, "posicao_por_volume")
    clusters = pd.read_csv(config.DIR_METRICS / "estrategia_perfil_clusters.csv")
    funil = pd.read_csv(config.DIR_METRICS / "estrategia_funil_2030.csv")
    tabela_familias = "\n".join(f"| {r.familia} | {numero(100*r.participacao,1)}% |" for r in familias.itertuples())
    tabela_clusters = "\n".join(f"| {r.cluster_nome} | {r.n_municipios} | {numero(r.taxa_2024*100,1)}% |" for r in clusters.itertuples())
    estrategia = f"""# Aplicação estratégica

**CRISP-DM: Evaluation.** As cinco perguntas do desafio, respondidas na unidade de decisão
em que a política pública acontece: o município. O classificador sustenta análise
retrospectiva de território; os cenários de metas são condicionais. Uso prospectivo
operacional depende de validação que este trabalho não tem.

## 1. Quais fatores estão associados à alfabetização?

O classificador usa contexto educacional e territorial. As participações abaixo descrevem
a contribuição SHAP do modelo salvo, não efeitos de intervenções.

| Família | Participação no SHAP absoluto agrupado |
|---|---:|
{tabela_familias}

A interpretação é exploratória: a seleção histórica de atributos usou a coorte inteira.
Não há dados socioeconômicos ou medidas individuais de renda, frequência e trajetória.
O histórico escolar verdadeiro não foi avaliado: o identificador de escola é reciclado entre edições.

## 2. Quais municípios apresentam maior risco?

O ranking contém {len(ranking)} municípios, com escores out-of-fold por município.
Os hiperparâmetros e atributos não foram selecionados de forma aninhada nesses folds;
o escore nacional é descritivo, não uma avaliação confirmatória.

Há duas ordenações: intensidade da taxa e volume estimado. Entre os territórios com posição
publicada, o total é {numero(total_risco,0)} crianças em risco contextual estimado;
o top 50 por volume concentra {numero(topo.criancas_em_risco.sum(),0)}.
Isso usa pesos de não resposta e supõe representatividade dos presentes dentro dos estratos.
Não identifica quais crianças ausentes terão o desfecho nem demonstra benefício de uma intervenção.

AC e DF permanecem sem posição publicada, por cobertura e desempenho limitados. Roraima não está na base.
`taxa_presenca_2024` e `n_alunos_avaliados` acompanham cada linha.
`faixa_shap_taxa_municipal` marca apenas taxa anterior abaixo de 65% ou ausente.
Uma contribuição SHAP plana de um atributo não comprova perda de ordenação do modelo completo.
`incerteza_posicao_quantificada = False`: a estabilidade de posições próximas ainda não foi medida.

## 3. Quais regiões apresentam padrões semelhantes?

KMeans descritivo em {resumo['clusters']['n_municipios']} municípios; k={resumo['clusters']['k_escolhido']},
silhueta {numero(resumo['clusters']['silhueta'],4)}, escolhida entre 3 e 8.
Há {resumo['clusters']['n_fora_por_dado_faltante']} municípios excluídos por campos faltantes.
Os grupos se sobrepõem; são perfis exploratórios de 2024, não classes naturais ou permanentes.
As taxas abaixo são médias simples entre municípios do perfil, não taxas populacionais da região.

| Perfil | Municípios | Média municipal da taxa de 2024 |
|---|---:|---:|
{tabela_clusters}

## 4. Como analisar metas futuras?

**Gap de esforço:** meta de 2025 menos taxa de 2024, na mesma apuração municipal do INEP.
São {metas['n_municipios_com_meta_publicada']} municípios com meta e dados disponíveis;
{metas['n_municipios_sem_meta']} ficam fora. Gap mediano de {numero(metas['gap_de_esforco_2025']['mediana'])} pp;
{numero(100*metas['gap_de_esforco_2025']['ja_superam_a_meta'],1)}% já superam essa meta com a taxa de 2024.
É uma diferença observada, sujeita à qualidade e comparabilidade das duas medidas.

**Projeção:** persistência suavizada e deriva estimadas na transição 2023→2024.
O prior, a suavização, a deriva e a dispersão são aprendidos só nos municípios de treino de cada fold.
O porte usado para prever 2024 é de 2023. A prevalência do baseline também vem só do treino.
A aplicação a 2025 usa taxa e porte de 2024, com parâmetros refitados no histórico disponível.

A distribuição de trabalho é T=clip(Z,0,100), com Z normal. `projecao_2025` é sua mediana;
os limites são quantis de 2,5% e 97,5% dessa distribuição. A probabilidade de ficar abaixo
da meta é calculada com a mesma distribuição, respeitando as massas em 0 e 100.
`dp_latente_pp` é a dispersão de Z, não o desvio da taxa censurada.
A estimação da dispersão gaussiana é uma aproximação; a cobertura é avaliada empiricamente.

### Avaliação fora do ajuste

Cinco folds por município na mesma transição anual, com {validacao['n_municipios']} municípios.
**Não é um teste em ano futuro.** A disponibilidade histórica das publicações ainda precisa ser confirmada.

| Métrica | Valor | Intervalo bootstrap de 95% |
|---|---:|---|
| ROC-AUC | {numero(ic['roc_auc']['pontual'],4)} | [{numero(ic['roc_auc']['ic_baixo'],4)}; {numero(ic['roc_auc']['ic_alto'],4)}] |
| Brier | {numero(ic['brier']['pontual'],5)} | [{numero(ic['brier']['ic_baixo'],5)}; {numero(ic['brier']['ic_alto'],5)}] |
| Ganho de Brier sobre prevalência do treino | {numero(ic['ganho_brier_sobre_baseline']['pontual'],5)} | [{numero(ic['ganho_brier_sobre_baseline']['ic_baixo'],5)}; {numero(ic['ganho_brier_sobre_baseline']['ic_alto'],5)}] |
| Cobertura do intervalo nominal de 95% | {numero(100*ic['cobertura_intervalo_95']['pontual'],2)}% | [{numero(100*ic['cobertura_intervalo_95']['ic_baixo'],2)}%; {numero(100*ic['cobertura_intervalo_95']['ic_alto'],2)}%] |

Os intervalos usam 1.000 reamostragens de municípios das predições OOF fixadas.
Não incluem incerteza de refit nem mudanças temporais; os folds compartilham parte do treino.
O ganho de Brier inclui zero. A evidência não sustenta superioridade operacional sobre o baseline.
Os métodos testados discriminam pouco; isso não prova que metas sejam intrinsecamente imprevisíveis.

### Cenários de 2025 e horizonte de 2030

Nos cenários de 2025, a largura mediana é {numero(metas['largura_mediana_do_intervalo_pp'],1)} pp.
A meta fica dentro do intervalo em {numero(100*metas['share_com_meta_dentro_do_intervalo_de_95'],2)}% dos municípios.
Os {metas['n_com_meta_fora_do_intervalo_de_95']} casos fora do intervalo são resultados condicionais ao modelo,
sem garantia de ocorrência. Não se publica rótulo de sucesso ou fracasso.

O porte de referência calculado é {numero(metas['persistencia']['porte_referencia_dispersao'],0)} alunos,
onde a²/n=c². É uma descrição da curva ajustada; não identifica causas da variação
nem determina quando um município pode ser avaliado individualmente. A coluna
`porte_abaixo_referencia_dispersao` é descritiva e não deve restringir acesso a políticas.

Mantendo a taxa de 2024 constante, {numero(100*funil.iloc[-1].share_que_ja_atinge,1)}% já satisfariam a meta de 2030.
O gap mediano até 2030 é {numero(funil.iloc[-1].gap_mediano)} pp em seis anos,
equivalente a {numero(funil.iloc[-1].gap_mediano_anualizado_pp,3)} pp/ano numa divisão linear.
Esse exercício não é previsão da taxa de 2030. Mediana de gaps e deriva ponderada têm agregações distintas;
não se interpreta sua razão como multiplicador de esforço nacional.

## 5. Quais variáveis mais influenciam o modelo?

Média de português municipal, contexto de UF e taxa municipal anterior estão entre as principais.
Consulte `interpret_familias.csv`, `interpret_permutacao.csv` e `interpretabilidade.md`.
Importância é condicionada ao conjunto de atributos e à amostra, com forte correlação entre medidas.
O histórico escolar não é verificável; presença anterior é associação entre os observados,
sem identificação do desfecho dos ausentes.

## Reprodução e limites

`python -m src.modeling.strategic` seguido de `python scripts/atualizar_relatorios.py`.
`estrategia_metas_oof.csv` registra município e fold; `estrategia_validacao_metas.csv`, os decis.
O modelo individual é retrospectivo e exploratório: a reserva de teste foi consultada na seleção
de atributos, e o status está fixado em `src/evaluation/protocolo.py`.
Não há validação temporal completa, dados socioeconômicos, avaliação causal ou incerteza das posições.
"""
    escrever("reports/aplicacao_estrategica.md", estrategia)

    temporal = """# Contrato temporal e de uso

**CRISP-DM: Business Understanding / Data Preparation.** A decisão atual é apoiar
análise retrospectiva de territórios. O usuário é a equipe de planejamento educacional.
O escore não diagnostica crianças e não autoriza intervenção individual automatizada.

Classe positiva: não alfabetizado em 2024, entre alunos presentes com medida válida.
Falso positivo pode desviar capacidade de análise; falso negativo deixa de priorizar um contexto de risco.
Não foram atribuídos custos monetários. A capacidade de 20% é uma hipótese operacional,
não um orçamento aprovado por secretaria. O critério técnico é comparar com baselines
nos mesmos grupos; adoção prospectiva exige evidência adicional e custos acordados.

Momento hipotético de previsão: após divulgação dos dados de 2023 e antes da avaliação de 2024.
Esse cenário **não foi comprovado**: o snapshot foi extraído em 2026 e não contém
um calendário auditado de disponibilidade. Ano de referência não equivale a data de publicação.
Território e rede podem ser conhecidos por matrícula, mas essa disponibilidade é uma premissa.
O caderno é da prova de 2024 e só cabe no artefato retrospectivo até comprovação contrária.

| Atributo | Referência/origem efetiva | Disponibilidade antes da prova de 2024 |
|---|---|---|
"""
    for feature in FEATURES_MODELO:
        if feature in {"rede_grupo", "nome_regiao", "sigla_uf"}:
            origem, disponibilidade = "cadastro da coorte de 2024", "premissa de matrícula; não comprovada"
        elif feature == "caderno":
            origem, disponibilidade = "caderno de prova de 2024", "não comprovada; controle retrospectivo"
        elif feature == "tem_historico_escola":
            origem, disponibilidade = "interseção de IDs 2023/2024 reciclados", "sem significado longitudinal; artefato retrospectivo"
        elif feature in {"mun_meta_2024_lag1", "mun_nivel_alfabetizacao_lag1"}:
            origem, disponibilidade = "linha 2023 da tabela de metas", "publicação e revisões não verificadas"
        else:
            origem, disponibilidade = "apuração de 2023 ou proveniência do join", "publicação não verificada"
        temporal += f"| `{feature}` | {origem} | {disponibilidade} |\n"
    temporal += """

Não foi identificado atributo socioeconômico. O enunciado menciona essa dimensão;
a lacuna permanece até integração de fonte adequada ou alinhamento acadêmico do recorte.
Não se supõe aceite do professor. Sem nova amostra intocada, não há gate de validação
confirmatória. As análises deste trabalho são exploratórias.
"""
    escrever("reports/contrato_temporal.md", temporal)

    teste = campeao["teste"]
    escrever("README.md", f"""# Tech Challenge — Fase 3: Alfabetização no Brasil

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

Reserva retrospectiva: {teste['n_alunos']} alunos em {teste['n_municipios']} municípios.
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
{numero(ic['roc_auc']['pontual'],4)} [{numero(ic['roc_auc']['ic_baixo'],4)}; {numero(ic['roc_auc']['ic_alto'],4)}].
Cobertura dos intervalos: {numero(100*ic['cobertura_intervalo_95']['pontual'],2)}%
[{numero(100*ic['cobertura_intervalo_95']['ic_baixo'],2)}%; {numero(100*ic['cobertura_intervalo_95']['ic_alto'],2)}%].
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
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe scripts/verificar_insumos.py
.\\.venv\\Scripts\\python.exe scripts/reproduzir.py --etapa dados
.\\.venv\\Scripts\\python.exe scripts/reproduzir.py --etapa relatorios
.\\.venv\\Scripts\\python.exe -m pytest -q
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
""")

    escrever("reports/documentacao_tecnica.md", """# Documentação técnica

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
""")

    escrever("reports/roteiro_video.md", f"""# Roteiro executivo

**CRISP-DM: comunicação de Evaluation.** Público: gestores educacionais.
Duração planejada: 4min a 4min40. Os tempos abaixo são planejamento, não duração medida.

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

Nos territórios com posição publicada, o modelo estima cerca de {numero(total_risco/1000,0)} mil
crianças em risco contextual. Os cinquenta primeiros por volume concentram cerca de
{numero(topo.criancas_em_risco.sum()/1000,0)} mil. Essas estimativas dependem dos pesos e das
hipóteses sobre alunos que não participaram da prova.

Não quantificamos a incerteza das posições próximas. As listas apoiam investigação e
planejamento; não devem definir automaticamente quais municípios receberão recursos.

## 2:20–3:20 — Metas: separar distância observada de previsão

A distância até a meta de 2025 é diretamente calculável: meta menos taxa de 2024.
O gap mediano é {numero(metas['gap_de_esforco_2025']['mediana'],1)} pontos percentuais.
Também construímos cenários com incerteza. A avaliação é montada para que cada município
seja previsto por parâmetros ajustados em outros municípios, e todas as taxas e limites
ficam entre zero e cem por cento.

Os intervalos cobriram aproximadamente {numero(100*validacao['cobertura_intervalo_95'],1)} por cento
dos resultados avaliados. Ainda assim, o método distingue pouco quem fica abaixo da
meta, e o ganho sobre um baseline simples é incerto. Essa avaliação usa uma única
transição anual: não comprova que as probabilidades funcionarão em um ano futuro.

Até 2030, o gap mediano é {numero(funil.iloc[-1].gap_mediano,2)} pontos em seis anos,
ou {numero(funil.iloc[-1].gap_mediano_anualizado_pp,3)} pontos por ano numa divisão linear.
Não devemos comparar uma distância de seis anos diretamente com um ritmo anual.

## 3:20–4:20 — Como usar e como evoluir

Nossa recomendação é usar os produtos para priorizar análises locais, verificar cobertura
da prova e discutir capacidade de atendimento. Municípios pequenos precisam de contexto
e incerteza, sem um corte automático de elegibilidade baseado em tamanho.

Vale dizer o que ainda não sabemos. A seleção de atributos consultou o mesmo conjunto que
depois serviu de reserva, então nenhum número aqui é confirmação prospectiva. O próximo
passo é uma amostra ainda não consultada, a confirmação de quando cada dado estava
disponível e a integração da dimensão socioeconômica prevista no desafio.

A entrega organiza evidências para decisão e torna explícito o que ainda precisa ser
validado. Esse cuidado é parte do valor do projeto para o planejamento educacional.

## Slides e ensaio

Usar os gráficos de comparação, as duas ordenações e a validação municipal —
`images/estrategia/10_validacao_metas.png`. Ensaiar, cortar exemplos se necessário e
conferir duração final de no máximo cinco minutos.
""")

    # Notebook de leitura: dados já medidos, sem treinos ocultos.
    import nbformat
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python (tc-fase3)", "language": "python", "name": "tc-fase3"}
    nb.cells = [nbformat.v4.new_markdown_cell(estrategia), nbformat.v4.new_code_cell('''from pathlib import Path
import sys, json
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))
import pandas as pd
from IPython.display import display
from src import config, eda
from src.modeling import strategic as st
from src.visualization import plots_estrategia as plots
eda.aplicar_estilo()
resumo = json.loads(st.JSON_ESTRATEGIA.read_text(encoding="utf-8"))
ranking = pd.read_csv(st.CSV_RANKING)
metas = pd.read_csv(st.CSV_METAS)
display(pd.DataFrame(resumo["metas"]["validacao_municipal_2023_2024"]["intervalos_bootstrap"]).T)
display(metas[["projecao_2025", "intervalo_2025_inferior", "intervalo_2025_superior"]].agg(["min", "max"]))
''')]
    figuras = [
        ("01_dois_rankings", "plots.plotar_dois_rankings(ranking)"),
        ("02_taxa_versus_volume", "plots.plotar_taxa_versus_volume(ranking)"),
        ("03_shrinkage", "plots.plotar_shrinkage(ranking, pd.read_csv(st.CSV_SHRINKAGE))"),
        ("04_risco_e_presenca", "plots.plotar_risco_e_presenca(ranking)"),
        ("05_selecao_de_k", "plots.plotar_selecao_de_k(pd.read_csv(st.CSV_SELECAO_K))"),
        ("06_perfil_dos_clusters", "plots.plotar_perfil_dos_clusters(pd.read_csv(st.CSV_PERFIL_CLUSTERS))"),
        ("07_composicao_regional", "plots.plotar_composicao_regional(pd.read_csv(st.CSV_REGIAO_CLUSTERS), pd.read_csv(st.CSV_PERFIL_CLUSTERS))"),
        ("08_gap_de_esforco", "plots.plotar_gap_de_esforco(metas)"),
        ("09_incerteza_por_porte", "plots.plotar_incerteza_por_porte(pd.read_csv(st.CSV_VOLATILIDADE), resumo)"),
        ("10_validacao_metas", "plots.plotar_validacao_metas(pd.read_csv(st.CSV_VALIDACAO_METAS), metas, resumo)"),
        ("11_funil_2030", "plots.plotar_funil(pd.read_csv(st.CSV_FUNIL))"),
    ]
    for nome, comando in figuras:
        # Sem `display(fig)`: o backend inline ja exibe a figura ao fim da celula,
        # e a chamada explicita a duplicaria. Ver scripts/executar_notebooks.py.
        nb.cells.append(nbformat.v4.new_code_cell(f'fig = {comando}\nprint(plots.salvar_figura(fig, "{nome}"))'))
    nbformat.write(nb, config.ROOT / "notebooks/05_aplicacao_estrategica.ipynb")
    print("README, relatórios, contrato temporal e notebook 05 atualizados das métricas.")


if __name__ == "__main__":
    main()
