"""Cenários municipais limitados a 0–100%, com avaliação fora do ajuste.

Uma única transição anual permite validar municípios separados, não a estabilidade
em anos futuros. A normal censurada é uma aproximação de trabalho, não um modelo
causal nem uma garantia de cobertura de 95% em 2025.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from src import config


def _validar(taxa, porte) -> tuple[np.ndarray, np.ndarray]:
    taxa, porte = np.asarray(taxa, float), np.asarray(porte, float)
    if taxa.ndim != 1 or porte.shape != taxa.shape or not len(taxa):
        raise ValueError("taxa e porte precisam ser vetores não vazios do mesmo tamanho")
    if not (np.isfinite(taxa).all() and np.isfinite(porte).all()):
        raise ValueError("taxa e porte precisam ser finitos")
    if ((taxa < 0) | (taxa > 100)).any() or (porte <= 0).any():
        raise ValueError("taxa deve estar em [0, 100] e porte deve ser positivo")
    return taxa, porte


class ProjetorMetas:
    """Persistência suavizada + deriva; todos os parâmetros são aprendidos no fit.

    O porte é o observado no ano de origem, disponível antes do desfecho.
    A variância a²/n+c² descreve heterogeneidade; não identifica ruído causal.
    A taxa futura é modelada como clip(Z, 0, 100), Z normal. A projeção pontual
    é a mediana dessa distribuição; os intervalos são seus quantis censurados.
    """

    def fit(self, taxa_origem, porte_origem, taxa_destino) -> "ProjetorMetas":
        taxa, porte = _validar(taxa_origem, porte_origem)
        destino, _ = _validar(taxa_destino, porte_origem)
        fracao = taxa / 100
        self.prior_ = float(np.average(fracao, weights=porte))
        ruido = self.prior_ * (1 - self.prior_)
        variancia = float(np.average((fracao - self.prior_) ** 2, weights=porte))
        tau2 = max(variancia - float(np.average(ruido / porte, weights=porte)), 1e-12)
        self.k_ = float(ruido / tau2)
        suavizada = self.suavizar(taxa, porte)
        self.deriva_ = float(np.average(destino - suavizada, weights=porte))
        residuo = destino - suavizada - self.deriva_

        def objetivo(log_parametros):
            a, c = np.exp(log_parametros)
            variancia_residuo = a * a / porte + c * c
            return float(0.5 * np.sum(np.log(variancia_residuo) + residuo**2 / variancia_residuo))

        ajuste = minimize(objetivo, np.log([100.0, 10.0]), method="L-BFGS-B",
                          bounds=[(-10, 10), (-10, 10)])
        if not ajuste.success or not np.isfinite(ajuste.fun):
            raise RuntimeError(f"ajuste da dispersão não convergiu: {ajuste.message}")
        self.a_, self.c_ = map(float, np.exp(ajuste.x))
        return self

    def suavizar(self, taxa, porte) -> np.ndarray:
        taxa, porte = _validar(taxa, porte)
        return (porte * taxa + self.k_ * self.prior_ * 100) / (porte + self.k_)

    def predict(self, taxa, porte, meta) -> pd.DataFrame:
        taxa, porte = _validar(taxa, porte)
        meta = np.asarray(meta, float)
        if meta.shape != taxa.shape or not np.isfinite(meta).all():
            raise ValueError("meta precisa ser finita e ter o mesmo tamanho da taxa")
        suavizada = self.suavizar(taxa, porte)
        centro = suavizada + self.deriva_
        dp = np.sqrt(self.a_**2 / porte + self.c_**2)
        probabilidade = norm.cdf((meta - centro) / dp)
        # P(clip(Z,0,100) < meta): atenção à massa de probabilidade em 100.
        probabilidade = np.where(meta <= 0, 0.0, np.where(meta > 100, 1.0, probabilidade))
        return pd.DataFrame({
            "taxa_suavizada": suavizada,
            "projecao": np.clip(centro, 0, 100),
            "dp_latente": dp,
            "inferior": np.clip(centro + norm.ppf(0.025) * dp, 0, 100),
            "superior": np.clip(centro + norm.ppf(0.975) * dp, 0, 100),
            "p_abaixo": probabilidade,
        })


def avaliar_municipios(base: pd.DataFrame, n_splits: int = 5,
                       seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """OOF territorial na transição 2023→2024, sem usar y do fold no ajuste."""
    colunas = ["id_municipio", "taxa_2023", "taxa_2024", "meta_2024", "n_alunos_2023"]
    dados = base.dropna(subset=colunas)[colunas].reset_index(drop=True)
    if not dados.id_municipio.is_unique:
        raise ValueError("a avaliação exige uma linha por município")
    partes = []
    for fold, (treino, validacao) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(dados)):
        tr, va = dados.iloc[treino], dados.iloc[validacao]
        modelo = ProjetorMetas().fit(tr.taxa_2023, tr.n_alunos_2023, tr.taxa_2024)
        pred = modelo.predict(va.taxa_2023, va.n_alunos_2023, va.meta_2024)
        pred["id_municipio"] = va.id_municipio.to_numpy()
        pred["fold"] = fold
        pred["ficou_abaixo"] = (va.taxa_2024 < va.meta_2024).astype(int).to_numpy()
        pred["baseline_prevalencia"] = float((tr.taxa_2024 < tr.meta_2024).mean())
        pred["coberto"] = va.taxa_2024.to_numpy() >= pred.inferior
        pred["coberto"] &= va.taxa_2024.to_numpy() <= pred.superior
        pred["gap_ate_meta"] = (va.meta_2024 - va.taxa_2023).to_numpy()
        pred["taxa_2023_invertida"] = -va.taxa_2023.to_numpy()
        partes.append(pred)
    return pd.concat(partes, ignore_index=True)


def resumir_avaliacao(oof: pd.DataFrame, seed: int = config.RANDOM_STATE,
                      n_bootstrap: int = 1000) -> dict:
    """IC por município das predições OOF; não inclui incerteza de refit ou temporal."""
    y, p = oof.ficou_abaixo.to_numpy(), oof.p_abaixo.to_numpy()
    erro = (p - y) ** 2
    erro_base = (oof.baseline_prevalencia.to_numpy() - y) ** 2
    funcoes = {
        "roc_auc": lambda ix: roc_auc_score(y[ix], p[ix]),
        "brier": lambda ix: erro[ix].mean(),
        "ganho_brier_sobre_baseline": lambda ix: (erro_base[ix] - erro[ix]).mean(),
        "cobertura_intervalo_95": lambda ix: oof.coberto.to_numpy()[ix].mean(),
    }
    rng = np.random.default_rng(seed)
    amostras = {k: [] for k in funcoes}
    for _ in range(n_bootstrap):
        ix = rng.integers(0, len(y), len(y))
        if np.unique(y[ix]).size < 2:
            continue
        for nome, funcao in funcoes.items():
            amostras[nome].append(float(funcao(ix)))
    intervalos = {nome: {
        "pontual": float(funcao(np.arange(len(y)))),
        "ic_baixo": float(np.quantile(amostras[nome], .025)),
        "ic_alto": float(np.quantile(amostras[nome], .975)),
    } for nome, funcao in funcoes.items()}
    return {
        "protocolo": "5 folds por município na mesma transição 2023→2024; não é teste de ano futuro",
        "porte_utilizado": "n_alunos_2023",
        "n_municipios": len(oof), "prevalencia_abaixo_da_meta": float(y.mean()),
        "roc_auc": {
            "projecao_com_shrinkage_e_deriva": float(roc_auc_score(y, p)),
            "gap_ate_a_meta_sem_modelo": float(roc_auc_score(y, oof.gap_ate_meta)),
            "taxa_2023_invertida": float(roc_auc_score(y, oof.taxa_2023_invertida)),
        },
        "brier": float(erro.mean()), "brier_da_prevalencia": float(erro_base.mean()),
        "cobertura_intervalo_95": float(oof.coberto.mean()),
        "intervalos_bootstrap": intervalos,
        "n_bootstrap": n_bootstrap,
        "limite_dos_intervalos": "condicionais às predições OOF; folds compartilham treino; não medem drift temporal",
    }


def gerar_produtos(base: pd.DataFrame, n_municipios_total: int,
                   seed: int = config.RANDOM_STATE) -> tuple:
    """Prepara os CSVs e o diagnóstico com o mesmo contrato da camada estratégica."""
    calibracao = base.dropna(subset=["taxa_2023", "taxa_2024", "n_alunos_2023"]).copy()
    modelo = ProjetorMetas().fit(calibracao.taxa_2023, calibracao.n_alunos_2023, calibracao.taxa_2024)
    residuo = calibracao.taxa_2024 - modelo.suavizar(calibracao.taxa_2023, calibracao.n_alunos_2023) - modelo.deriva_
    volatilidade = calibracao.assign(residuo=residuo, faixa=pd.cut(
        calibracao.n_alunos_2023, [0, 25, 50, 100, 200, 500, 1000, np.inf]
    )).groupby("faixa", observed=True).agg(
        n_municipios=("residuo", "size"), porte_mediano=("n_alunos_2023", "median"),
        evolucao_dp=("residuo", "std"),
    ).reset_index()
    volatilidade["faixa"] = volatilidade.faixa.astype(str)
    volatilidade["dp_ajustado"] = np.sqrt(modelo.a_**2 / volatilidade.porte_mediano + modelo.c_**2)

    oof = avaliar_municipios(base, seed=seed)
    diagnostico_oof = resumir_avaliacao(oof, seed=seed)
    decis = oof.assign(decil=pd.qcut(oof.p_abaixo, 10, labels=False, duplicates="drop")).groupby("decil").agg(
        n_municipios=("ficou_abaixo", "size"), probabilidade_prevista=("p_abaixo", "mean"),
        fracao_observada=("ficou_abaixo", "mean"),
    ).reset_index()

    publicavel = base.dropna(subset=["taxa_2024", "meta_2025", "n_alunos_avaliados"]).copy().reset_index(drop=True)
    pred = modelo.predict(publicavel.taxa_2024, publicavel.n_alunos_avaliados, publicavel.meta_2025)
    for origem, destino in {
        "taxa_suavizada": "taxa_2024_suavizada", "projecao": "projecao_2025",
        "dp_latente": "dp_latente_pp", "inferior": "intervalo_2025_inferior",
        "superior": "intervalo_2025_superior", "p_abaixo": "p_abaixo_da_meta_2025",
    }.items():
        publicavel[destino] = pred[origem].to_numpy()
    publicavel["gap_de_esforco_2025"] = publicavel.meta_2025 - publicavel.taxa_2024
    publicavel["meta_dentro_do_intervalo"] = publicavel.meta_2025.between(
        publicavel.intervalo_2025_inferior, publicavel.intervalo_2025_superior)
    publicavel["natureza_resultado"] = "cenario_condicional_sem_validacao_em_ano_futuro"
    limiar = float(np.ceil((modelo.a_ / modelo.c_) ** 2))
    publicavel["porte_abaixo_referencia_dispersao"] = publicavel.n_alunos_avaliados < limiar
    anos = range(2025, 2031)
    funil = pd.DataFrame([{
        "ano": ano, "meta_mediana": float(publicavel[f"meta_{ano}"].median()),
        "municipios_que_ja_atingem_com_a_taxa_de_2024": int((publicavel.taxa_2024 >= publicavel[f"meta_{ano}"]).sum()),
        "share_que_ja_atinge": float((publicavel.taxa_2024 >= publicavel[f"meta_{ano}"]).mean()),
        "gap_mediano": float((publicavel[f"meta_{ano}"] - publicavel.taxa_2024).median()),
        "gap_mediano_anualizado_pp": float((publicavel[f"meta_{ano}"] - publicavel.taxa_2024).median() / (ano - 2024)),
    } for ano in anos])
    diagnostico = {
        "versao_metodo": 2,
        "distribuicao": "normal censurada em [0,100]; projeção pontual = mediana; dp em escala latente",
        "n_municipios_com_meta_publicada": len(publicavel),
        "n_municipios_sem_meta": n_municipios_total - len(publicavel),
        "gap_de_esforco_2025": {
            "mediana": float(publicavel.gap_de_esforco_2025.median()),
            "media": float(publicavel.gap_de_esforco_2025.mean()),
            "ja_superam_a_meta": float((publicavel.gap_de_esforco_2025 <= 0).mean()),
            "p95": float(publicavel.gap_de_esforco_2025.quantile(.95)),
        },
        "persistencia": {"k_shrinkage": modelo.k_, "prior_alfabetizacao": modelo.prior_,
                         "deriva_ponderada_pp": modelo.deriva_, "dispersao_a": modelo.a_,
                         "dispersao_c": modelo.c_, "porte_referencia_dispersao": limiar},
        "validacao_municipal_2023_2024": diagnostico_oof,
        "share_com_probabilidade_entre_40_e_60": float(publicavel.p_abaixo_da_meta_2025.between(.4, .6).mean()),
        "share_com_meta_dentro_do_intervalo_de_95": float(publicavel.meta_dentro_do_intervalo.mean()),
        "n_com_meta_fora_do_intervalo_de_95": int((~publicavel.meta_dentro_do_intervalo).sum()),
        "largura_mediana_do_intervalo_pp": float((publicavel.intervalo_2025_superior - publicavel.intervalo_2025_inferior).median()),
        "share_abaixo_referencia_dispersao": float(publicavel.porte_abaixo_referencia_dispersao.mean()),
        "evolucao_2023_2024": {
            "mediana_pp": float((calibracao.taxa_2024 - calibracao.taxa_2023).median()),
            "dp_pp": float((calibracao.taxa_2024 - calibracao.taxa_2023).std()),
            "share_que_piorou": float((calibracao.taxa_2024 < calibracao.taxa_2023).mean()),
        },
    }
    return publicavel.sort_values("gap_de_esforco_2025", ascending=False), volatilidade, decis, funil, diagnostico, oof
