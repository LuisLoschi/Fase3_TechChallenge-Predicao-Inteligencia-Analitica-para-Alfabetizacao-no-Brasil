"""Feature store temporal: agregados de 2023 aplicados aos alunos de 2024.

Toda coluna produzida aqui nasce de dados de **2023**. É a defesa nº 2 da matriz
anti-leakage garantida por construção e não por disciplina de quem escreve o
código: não existe caminho neste módulo que leia uma linha de 2024.

Hierarquia de fontes, fechada pela auditoria da Etapa 2.5 (B5):

    Gold 2023 ............ coorte de presentes, alvo, rede e território
    microdado 2023 ....... `presenca` (inclui os ausentes que a Gold descarta),
                           `peso_aluno` e `proficiencia`
    agregado INEP 2023 ... `taxa_alfabetizacao` e `media_portugues` onde a Gold
                           não tem microdado — é o que recupera São Paulo
    metas ................ `meta_alfabetizacao_2024` e `nivel_alfabetizacao`,
                           lidos da **linha de 2023**, nunca da de 2024

Três decisões medidas que este módulo materializa:

1. **`id_escola` não é chave longitudinal** (medido na Etapa 3, ver
   `reports/feature_engineering.md`): dos 36.051 identificadores presentes nos
   dois anos, só 2,40% apontam para o mesmo município. O bloco `esc_*` existe,
   é construído e é medido, mas entra no modelo como **controle negativo
   declarado**, ao lado do `caderno` — não como histórico de escola. Quem quiser
   o conjunto sem ele usa `FEATURES_ESCOLA` para removê-lo.
2. **Agregado municipal e de UF são ponderados por `peso_aluno`** (B2). Sem o
   peso a taxa da Gold e a taxa do agregado do INEP, que a coalescência mistura
   na mesma coluna, ficam em escalas diferentes: 0,99pp de MAE contra 0,05pp.
3. **Coalescência `5 → 3`** no agregado do INEP, com a coluna de proveniência
   `fonte_lag_municipal` obrigatória no dataset final (B4).

**Unidades, porque elas não são uniformes e isso é deliberado.**
`mun_taxa_alfab_lag1`, `uf_taxa_alfab_lag1`, `mun_media_portugues_lag1` e
`mun_meta_2024_lag1` saem em **pontos percentuais**, que é a escala em que o INEP
publica e em que chega metade dos valores depois da coalescência — converter
seria reescrever o número da fonte oficial. `esc_taxa_alfab_lag1`, as taxas de
presença e o share de rede saem em **fração**, porque nunca são comparados com
número publicado. Nenhuma escala afeta modelo de árvore, e o único ponto onde as
duas se encontram, a suavização, faz a conversão explicitamente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data import loader

# `k` da suavização empírico-bayesiana da escola. Fixo e documentado, não
# calibrado: a EDA mediu AUC crescendo monotonicamente com `k` (0,561 em k=0 a
# 0,653 em k=500) sem nunca superar a taxa municipal pura de 0,654. O melhor `k`
# é sempre o maior da grade, porque a suavização não recupera sinal escolar —
# ela substitui a escola pelo município. Calibrar por CV devolveria o maior valor
# da grade e uma falsa impressão de otimização. Fica como conveniência de
# imputação: escola sem histórico herda o contexto do município.
K_SUAVIZACAO = 30.0

PERCENTIS_PROFICIENCIA = (0.25, 0.50, 0.75)

# Proveniência de `mun_taxa_alfab_lag1` e `mun_media_portugues_lag1` depois da
# coalescência. Ordenada da apuração mais fiel para a menos fiel.
FONTE_GOLD = "gold"
FONTE_R5 = "agregado_r5"
FONTE_R3 = "agregado_r3"
FONTE_AUSENTE = "sem_historico"


# ---------------------------------------------------------------------------
# Blocos elementares
# ---------------------------------------------------------------------------
def _media_ponderada(df: pd.DataFrame, chave: str, valor: str, peso: str) -> pd.Series:
    """Média de `valor` ponderada por `peso`, por `chave`, ignorando nulos de valor.

    `groupby.apply` com `np.average` faria isso em uma linha e levaria minutos em
    1,5 milhão de linhas. Somar numerador e denominador separadamente é a mesma
    conta em duas passadas vetorizadas.
    """
    valido = df[[chave, valor, peso]].dropna()
    numerador = (valido[valor] * valido[peso]).groupby(valido[chave]).sum()
    denominador = valido[peso].groupby(valido[chave]).sum()
    return numerador / denominador


def _percentis_proficiencia(micro_lag: pd.DataFrame, chave: str, prefixo: str) -> pd.DataFrame:
    """Forma da distribuição de proficiência de 2023: p25, p50, p75, IQR e desvio.

    Substituem `proporcao_aluno_nivel_0..8`, que é 100% nula em 2023 (A5). São a
    única informação que **só** o microdado tem: nível e presença já vêm da Gold
    e do agregado do INEP. É por isso que a decisão de mantê-las ou não é a
    réplica do experimento B5, e não uma questão de conveniência.

    Calculadas apenas sobre presentes: proficiência de quem faltou não existe.
    """
    presentes = micro_lag[(micro_lag["presenca"] == 1) & micro_lag["proficiencia"].notna()]
    agrupado = presentes.groupby(chave, observed=True)["proficiencia"]
    saida = agrupado.agg(**{f"{prefixo}_prof_sd_lag1": "std"})
    percentis = agrupado.quantile(list(PERCENTIS_PROFICIENCIA)).unstack()
    percentis.columns = [f"{prefixo}_prof_p{int(p * 100)}_lag1" for p in percentis.columns]
    saida = saida.join(percentis)
    saida[f"{prefixo}_prof_iqr_lag1"] = (
        saida[f"{prefixo}_prof_p75_lag1"] - saida[f"{prefixo}_prof_p25_lag1"]
    )
    return saida


def _taxa_presenca(micro_lag: pd.DataFrame, chave: str, prefixo: str) -> pd.DataFrame:
    """Taxa de presença de 2023 — a única feature que precisa dos ausentes.

    É calculada sobre **todas** as linhas do microdado, inclusive `presenca = 0`,
    que a Gold descarta por desenho. Proxy de infraestrutura e de engajamento, e
    a versão legítima da `percentual_participacao_*` que saiu das features por
    ser contemporânea ao alvo (B1): defasada ela vale AUC 0,590 contra 0,570 da
    contemporânea, além de não vazar.
    """
    return micro_lag.groupby(chave, observed=True)["presenca"].agg(
        **{f"{prefixo}_taxa_presenca_lag1": "mean", f"{prefixo}_n_matriculados_lag1": "size"}
    )


# ---------------------------------------------------------------------------
# Agregados por nível territorial
# ---------------------------------------------------------------------------
def agregar_escola(gold_lag: pd.DataFrame, micro_lag: pd.DataFrame) -> pd.DataFrame:
    """Agregados de 2023 por `id_escola`. **Controle negativo, não histórico.**

    Medido nesta etapa: `id_escola` é reatribuído a cada edição em ordem
    territorial. Dos 36.051 identificadores presentes nos dois anos, 2,40% caem
    no mesmo município e 80,6% na mesma UF — o mesmo mecanismo já provado para
    `id_aluno` (A2). O join por `id_escola` entrega a 76,3% dos alunos de 2024 o
    agregado de **outra** escola, quase sempre da mesma UF, e por isso ainda
    produz AUC acima do acaso: o que sobrevive ao embaralhamento é sinal de UF.

    O bloco continua sendo produzido porque a Etapa 5 precisa dele como controle
    negativo: se `esc_*` receber importância relevante no SHAP, isso é detector
    de sobreajuste, exatamente como o `caderno`.
    """
    presentes = gold_lag.groupby("id_escola", observed=True)["alfabetizado"].agg(
        esc_n_alunos_lag1="size", esc_taxa_alfab_lag1="mean"
    )
    return presentes.join(_taxa_presenca(micro_lag, "id_escola", "esc")).join(
        _percentis_proficiencia(micro_lag, "id_escola", "esc")
    )


def agregar_municipio(
    gold_lag: pd.DataFrame,
    micro_lag: pd.DataFrame,
    agregado_inep: pd.DataFrame,
    metas: pd.DataFrame,
) -> pd.DataFrame:
    """Agregados de 2023 por `id_municipio`, com coalescência e proveniência.

    É aqui que está o sinal do projeto: as cinco melhores features univariadas da
    EDA são todas municipais. Três origens entram na mesma tabela:

    * **Gold + microdado** — taxa ponderada, presença, porte, share de rede e a
      forma da distribuição de proficiência. Cobre 4.871 municípios.
    * **agregado do INEP** — `taxa_alfabetizacao` e `media_portugues` para quem a
      Gold não cobre, por coalescência `5 → 3` (B4). São Paulo inteiro entra por
      aqui.
    * **metas** — `meta_alfabetizacao_2024` e `nivel_alfabetizacao` lidos da
      linha de **2023**. A meta é um lag disfarçado (A4, `corr = 0,979` com a
      taxa base), legítima como feature e ilegítima como alvo.
    """
    peso = config.COL_PESO
    base = gold_lag.groupby("id_municipio", observed=True).agg(
        mun_n_alunos_lag1=("alfabetizado", "size"),
        mun_n_escolas_lag1=("id_escola", "nunique"),
        mun_taxa_alfab_gold_lag1=("alfabetizado", "mean"),
        mun_share_rede_estadual_lag1=("rede", lambda s: float((s == 2).mean())),
    )

    # A taxa ponderada é a que o INEP publica (B2). Como a coalescência mistura
    # esta coluna com a do agregado do INEP, usar a não ponderada aqui colocaria
    # duas escalas na mesma feature: MAE de 0,99pp contra 0,05pp.
    presentes = micro_lag[micro_lag["presenca"] == 1]
    base["mun_taxa_alfab_gold_lag1"] = _media_ponderada(
        presentes, "id_municipio", "alfabetizado", peso
    ).reindex(base.index)

    base = base.join(_taxa_presenca(micro_lag, "id_municipio", "mun")).join(
        _percentis_proficiencia(micro_lag, "id_municipio", "mun")
    )

    inep = _pivotar_agregado_inep(agregado_inep, "id_municipio")
    base = base.join(inep, how="outer")
    base = _coalescer(base, "mun", "id_municipio")

    linha_2023 = metas[metas["ano"] == config.ANO_LAG].set_index("id_municipio")
    base = base.join(
        linha_2023[["meta_alfabetizacao_2024", "nivel_alfabetizacao"]].rename(
            columns={
                "meta_alfabetizacao_2024": "mun_meta_2024_lag1",
                "nivel_alfabetizacao": "mun_nivel_alfabetizacao_lag1",
            }
        )
    )
    return base


def agregar_uf(
    gold_lag: pd.DataFrame,
    micro_lag: pd.DataFrame,
    agregado_inep_uf: pd.DataFrame,
    dim_municipio: pd.DataFrame,
) -> pd.DataFrame:
    """Agregados de 2023 por `sigla_uf`, com a mesma coalescência do município.

    O microdado não traz UF, só `id_municipio` — daí a dependência de
    `dim_municipio`. A Gold de 2023 tem 23 UFs; o agregado do INEP tem 24, e a
    que ele acrescenta é justamente São Paulo. Sobram AC e DF sem nenhuma linha
    de 2023 em nenhuma fonte, o que é limitação da base e não do desenho.
    """
    uf_por_municipio = dim_municipio.set_index("id_municipio")["sigla_uf"]
    gold_uf = gold_lag.assign(sigla_uf=gold_lag["id_municipio"].map(uf_por_municipio))
    micro_uf = micro_lag.assign(sigla_uf=micro_lag["id_municipio"].map(uf_por_municipio))

    base = gold_uf.groupby("sigla_uf", observed=True).agg(uf_n_alunos_lag1=("alfabetizado", "size"))
    presentes = micro_uf[micro_uf["presenca"] == 1]
    base["uf_taxa_alfab_gold_lag1"] = _media_ponderada(
        presentes, "sigla_uf", "alfabetizado", config.COL_PESO
    ).reindex(base.index)
    base = base.join(_taxa_presenca(micro_uf, "sigla_uf", "uf"))

    inep = _pivotar_agregado_inep(agregado_inep_uf, "sigla_uf")
    base = base.join(inep, how="outer")
    return _coalescer(base, "uf", "sigla_uf")


# ---------------------------------------------------------------------------
# Coalescência 5 → 3 e proveniência
# ---------------------------------------------------------------------------
def _pivotar_agregado_inep(agregado: pd.DataFrame, chave: str) -> pd.DataFrame:
    """Uma linha por território, com as colunas das redes 5 e 3 lado a lado.

    O agregado do INEP tem grão território × série × rede (23.995 linhas para
    5,4 mil municípios). Sem escolher a rede antes do merge, a coorte duplica —
    é a armadilha de cardinalidade nº 1 do projeto.
    """
    do_ano = agregado[agregado["ano"] == config.ANO_LAG]
    partes = []
    for rede in config.REDE_AGREGADO_COALESCENCIA:
        fatia = do_ano[do_ano["rede"] == rede].set_index(chave)[
            ["taxa_alfabetizacao", "media_portugues"]
        ]
        assert fatia.index.is_unique, f"agregado do INEP duplicado em {chave} para rede {rede}"
        partes.append(fatia.add_suffix(f"_r{rede}"))
    return pd.concat(partes, axis=1)


def _coalescer(base: pd.DataFrame, prefixo: str, chave: str) -> pd.DataFrame:
    """Gold → `rede = 5` → `rede = 3`, registrando de onde veio cada valor.

    A ordem é a da fidelidade medida contra o microdado: a Gold é a própria
    coorte, `rede = 5` erra 0,99pp e `rede = 3` erra 1,73pp. Depois da
    coalescência a mesma coluna carrega três apurações diferentes, e sem
    `fonte_lag_*` não há como estratificar métrica por fidelidade (vetor 15 da
    matriz anti-leakage).

    `fonte_lag_*` descreve `*_taxa_alfab_lag1`. `*_media_portugues_lag1` vem
    **sempre** do agregado do INEP, porque a proficiência média que a Gold
    permitiria calcular é a mesma coisa medida de outro jeito (MAE 0,074) e com
    cobertura menor — 4.871 municípios contra 5.448.
    """
    r5, r3 = config.REDE_AGREGADO_COALESCENCIA
    # A Gold sai em fração e o INEP publica em pontos percentuais; a coluna
    # coalescida precisa de uma escala só, e a escolhida é a do INEP.
    taxa_gold = base.pop(f"{prefixo}_taxa_alfab_gold_lag1") * 100

    fonte = pd.Series(FONTE_AUSENTE, index=base.index, dtype="object")
    fonte[base[f"taxa_alfabetizacao_r{r3}"].notna()] = FONTE_R3
    fonte[base[f"taxa_alfabetizacao_r{r5}"].notna()] = FONTE_R5
    fonte[taxa_gold.notna()] = FONTE_GOLD

    base[f"{prefixo}_taxa_alfab_lag1"] = (
        taxa_gold.fillna(base[f"taxa_alfabetizacao_r{r5}"]).fillna(base[f"taxa_alfabetizacao_r{r3}"])
    )
    base[f"{prefixo}_media_portugues_lag1"] = base[f"media_portugues_r{r5}"].fillna(
        base[f"media_portugues_r{r3}"]
    )
    base[f"fonte_lag_{'municipal' if prefixo == 'mun' else 'uf'}"] = fonte

    descartar = [c for c in base.columns if c.endswith((f"_r{r5}", f"_r{r3}"))]
    return base.drop(columns=descartar).rename_axis(chave)


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------
def montar_feature_store(
    gold_lag: pd.DataFrame | None = None,
    micro_lag: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Devolve os três agregados de 2023 prontos para o join com a coorte de 2024.

    Recebe as bases já carregadas para que o notebook e os testes não paguem duas
    vezes a leitura de 3,2 milhões de linhas.
    """
    if gold_lag is None:
        gold_lag = loader.carregar_gold(
            ano=config.ANO_LAG, colunas=["id_escola", "id_municipio", "rede", "alfabetizado"]
        )
    if micro_lag is None:
        micro_lag = loader.carregar_aluno(
            ano=config.ANO_LAG,
            colunas=["id_municipio", "id_escola", "presenca", "alfabetizado", "proficiencia", config.COL_PESO],
        )

    dim = loader.carregar_dim_municipio()
    agregado_mun = loader.carregar_agregado_municipio()
    agregado_uf = loader.carregar_agregado_uf()
    metas = loader.carregar_meta_municipio()

    return {
        "escola": agregar_escola(gold_lag, micro_lag),
        "municipio": agregar_municipio(gold_lag, micro_lag, agregado_mun, metas),
        "uf": agregar_uf(gold_lag, micro_lag, agregado_uf, dim),
    }


def suavizar_taxa_escola(coorte: pd.DataFrame, k: float = K_SUAVIZACAO) -> pd.Series:
    """`(taxa_esc·n + taxa_mun·k) / (n + k)` — escola puxada para o município.

    Com `k = 30` a escola mediana de 2023, que avaliou 32 alunos, fica com peso
    de 52% da própria taxa. Ver `K_SUAVIZACAO` para por que o `k` é fixo.

    A escala das duas taxas precisa bater: `mun_taxa_alfab_lag1` sai em pontos
    percentuais, porque é a escala em que o INEP publica, e `esc_taxa_alfab_lag1`
    sai em fração. Converter aqui e não na origem mantém cada coluna na unidade
    da própria fonte.
    """
    n = coorte["esc_n_alunos_lag1"].fillna(0)
    taxa_escola = coorte["esc_taxa_alfab_lag1"] * 100
    taxa_municipio = coorte["mun_taxa_alfab_lag1"]
    return (taxa_escola.fillna(taxa_municipio) * n + taxa_municipio * k) / (n + k)


def agregados_municipais_ponderados(
    ano: int, micro: pd.DataFrame | None = None, dim_municipio: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Tabela municipal **populacional** de um ano, ponderada por `peso_aluno` (B2).

    Separada das features de propósito: aqui o número é o que o projeto publica
    (ranking de risco, gap de meta, projeção até 2030) e por isso tem de
    reconciliar com o INEP a menos de 0,10pp; lá o número é insumo de um
    classificador em grão aluno. São usos diferentes da mesma conta.

    Entrega taxa ponderada e não ponderada lado a lado justamente para que a
    diferença entre elas seja visível — é a medida do viés de não-resposta que a
    ponderação padroniza, mas não elimina.
    """
    if micro is None:
        micro = loader.carregar_aluno(
            ano=ano,
            colunas=["id_municipio", "presenca", "alfabetizado", "proficiencia", config.COL_PESO],
        )
    if dim_municipio is None:
        dim_municipio = loader.carregar_dim_municipio()

    presentes = micro[micro["presenca"] == 1]
    tabela = presentes.groupby("id_municipio", observed=True).agg(
        n_avaliados=("alfabetizado", "size"),
        taxa_alfab_simples=("alfabetizado", "mean"),
        prof_media_simples=("proficiencia", "mean"),
    )
    tabela["taxa_alfab_ponderada"] = _media_ponderada(
        presentes, "id_municipio", "alfabetizado", config.COL_PESO
    )
    tabela["prof_media_ponderada"] = _media_ponderada(
        presentes, "id_municipio", "proficiencia", config.COL_PESO
    )
    tabela["soma_pesos"] = presentes.groupby("id_municipio", observed=True)[config.COL_PESO].sum()
    tabela["taxa_presenca"] = micro.groupby("id_municipio", observed=True)["presenca"].mean()
    tabela["ano"] = ano

    uf = dim_municipio.set_index("id_municipio")[["sigla_uf", "nome_regiao"]]
    return tabela.join(uf).reset_index()
