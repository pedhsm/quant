# mcp/tester.py
# Módulo de permutação para ser consumido pelo scripts/09_run_mcp_macro.py

import pandas as pd
import numpy as np

def _compute_metrics(returns: pd.Series) -> dict[str, float]:
    """
    Função interna para calcular métricas.
    (Lógica adaptada do 'mcp/base.py' original)
    """
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return {"sharpe": float("nan"), "cagr": float("nan"), "vol": float("nan")}
    
    mean = r.mean()
    std = r.std(ddof=0)
    
    sharpe = (mean / std) * np.sqrt(252) if std > 0 else np.nan
    
    cum = r.cumsum()
    if len(cum) == 0:
        cagr = np.nan
    else:
        total = float(np.exp(cum.iloc[-1]))
        years = max(len(r) / 252.0, 1e-9)
        cagr = total ** (1/years) - 1 if total > 0 else np.nan
        
    return {
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "vol": float(std * np.sqrt(252))
    }

def permutation_test(
    signal: pd.Series, 
    metric: str = "sharpe", 
    n_permutations: int = 2000
) -> dict[str, float]:
    """
    Executa um teste de permutação simples sobre a série de retornos.
    
    Projetado para atender aos requisitos do '09_run_mcp_macro.py':
    - Recebe signal, metric, n_permutations
    - Retorna um dict com 'real' (score) e 'pvalue'
    """
    
    real_returns = signal.dropna().astype(float)
    real_metrics = _compute_metrics(real_returns)
    real_score = real_metrics.get(metric.lower(), np.nan)

    if not np.isfinite(real_score):
        print(f"[MCP Warning] Métrica real '{metric}' não é finita (NaN ou Inf). Retornando NaN.")
        return {'real': real_score, 'pvalue': np.nan}

    # 2. Executar permutações
    perm_scores = []
    vals = real_returns.values
    idx = real_returns.index
    
    if len(vals) < 2:
        return {'real': real_score, 'pvalue': np.nan}

    rng = np.random.default_rng(42) # Seed para reprodutibilidade
    for _ in range(int(n_permutations)):
        perm = rng.permutation(vals)
        perm_s = pd.Series(perm, index=idx)
        metrics = _compute_metrics(perm_s)
        perm_scores.append(metrics.get(metric.lower(), np.nan))

    # 3. Calcular p-valor
    perm_scores_arr = np.array(perm_scores)
    perm_scores_arr = perm_scores_arr[np.isfinite(perm_scores_arr)] 
    
    if len(perm_scores_arr) == 0:
        p_value = np.nan #
    else:
        p_value = float(np.mean(perm_scores_arr >= real_score))

    # Retorna o dicionário que '09_run_mcp_macro.py' espera
    return {
        'real': real_score,
        'pvalue': p_value
    }