from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from .ewma import realized_vol, ewma_vol   
from .garch import GARCH11

# -------------------- Helpers (log-returns) --------------------

def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.dropna().astype(float)
    return pd.Series(x, dtype=float).dropna()

def equity_from_log(log_rets: pd.Series, initial: float = 1.0) -> pd.Series:
    r = _to_series(log_rets)
    if r.empty:
        return pd.Series(dtype=float)
    return initial * np.exp(r.cumsum()) # type: ignore

def _ann_from_log(log_rets: pd.Series, periods: int) -> float:
    r = _to_series(log_rets)
    if r.empty:
        return np.nan
    years = len(r) / float(periods)
    if years <= 0:
        return np.nan
    end_equity = float(np.exp(r.sum()))
    return end_equity ** (1.0 / years) - 1.0

def _tot_from_log(log_rets: pd.Series) -> float:
    r = _to_series(log_rets)
    return float(np.exp(r.sum()) - 1.0) if len(r) else np.nan

def _ann_vol_from_log(log_rets: pd.Series, periods: int) -> float:
    r = _to_series(log_rets)
    return float(r.std(ddof=0) * np.sqrt(periods)) if len(r) else np.nan

def _sharpe_from_log(log_rets: pd.Series, periods: int, rf_per_period: float = 0.0) -> float:
    r = _to_series(log_rets)
    if r.empty:
        return np.nan
    excess = r - rf_per_period
    vol = excess.std(ddof=0)
    if not np.isfinite(vol) or vol <= 0:
        return np.nan
    return float(excess.mean() / vol * np.sqrt(periods))

def _sortino_from_log(log_rets: pd.Series, periods: int, rf_per_period: float = 0.0, mar: float = 0.0) -> float:
    r = _to_series(log_rets)
    if r.empty:
        return np.nan
    excess = r - rf_per_period
    downside = np.minimum(excess - mar, 0.0)
    dd = downside.std(ddof=0)
    if not np.isfinite(dd) or dd <= 0:
        return np.nan
    ann_excess = excess.mean() * np.sqrt(periods)
    return float(ann_excess / dd)

def _max_dd_from_log(log_rets: pd.Series) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    eq = equity_from_log(log_rets)
    if eq.empty:
        return (np.nan, None, None)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    trough_idx = int(dd.values.argmin()) 
    mdd = float(dd.iloc[trough_idx])
    peak_idx = int(eq.iloc[:trough_idx + 1].values.argmax())
    t_peak = eq.index[peak_idx] if len(eq.index) > peak_idx else None
    t_trough = eq.index[trough_idx] if len(eq.index) > trough_idx else None
    return (mdd, t_peak, t_trough)

def _calmar_from_log(log_rets: pd.Series, periods: int) -> float:
    cagr = _ann_from_log(log_rets, periods)
    mdd, _, _ = _max_dd_from_log(log_rets)
    mdd_abs = abs(mdd) if np.isfinite(mdd) else np.nan
    if not np.isfinite(cagr) or not np.isfinite(mdd_abs) or mdd_abs <= 0:
        return np.nan
    return float(cagr / mdd_abs)

def _hit_rate(log_rets: pd.Series) -> float:
    r = _to_series(log_rets)
    return float((r > 0).mean()) if len(r) else np.nan

# -------------------- Agregador (C1) --------------------

def compute_metrics_c1(
    log_rets: pd.Series,
    periods: int = 252,
    rf_per_period: float = 0.0,
    include_n: bool = True,
    # --- EWMA ---
    ewma_lambda: float = 0.94,          # lambda RiskMetrics
    ewma_span: Optional[int] = None,    # alternativa: usar "span" se preferir
) -> Dict[str, float]:
    """
    Métricas da BASE (C1) + EWMA:
      - Retorno Total (%)
      - Retorno Anualizado (%)
      - Volatilidade (%) [stdev]
      - Sharpe, Sortino, Calmar
      - Max Drawdown (%)
      - Taxa de Vitória (%)
      - EWMA Vol (%)  [último valor anualizado]
      - EWMA λ
    """
    r = _to_series(log_rets)
    out: Dict[str, float] = {}
    if include_n:
        out["N"] = float(len(r))

    out["Retorno Total (%)"]       = 100.0 * _tot_from_log(r)
    out["Retorno Anualizado (%)"]  = 100.0 * _ann_from_log(r, periods)
    out["Volatilidade (%)"]        = 100.0 * _ann_vol_from_log(r, periods)
    out["Sharpe Ratio"]            = _sharpe_from_log(r, periods, rf_per_period)
    out["Sortino Ratio"]           = _sortino_from_log(r, periods, rf_per_period, mar=0.0)
    out["Calmar Ratio"]            = _calmar_from_log(r, periods)
    mdd, _, _ = _max_dd_from_log(r)
    out["Max Drawdown (%)"]        = 100.0 * mdd
    out["Taxa de Vitória (%)"]     = 100.0 * _hit_rate(r)

    # ---------- EWMA ----------
    try:
        # sua função ewma_vol já anualiza; se não anualizar, ajuste aqui.
        ewma_series = ewma_vol(r, lambda_=ewma_lambda, periods=periods, span=ewma_span)
        if isinstance(ewma_series, (pd.Series, np.ndarray, list)) and len(ewma_series):
            ewma_last = float(pd.Series(ewma_series).iloc[-1])
            out["EWMA Vol (%)"] = 100.0 * ewma_last
        else:
            out["EWMA Vol (%)"] = np.nan
    except Exception:
        out["EWMA Vol (%)"] = np.nan
    out["EWMA λ"] = float(ewma_lambda)

    return out

# -------------------- Ponte com GARCH (existente) --------------------

def garch_fit_and_forecast(
    log_rets: pd.Series,
    annualization: int = 252,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((1e-8, 1e-2), (1e-6, 0.3), (0.5, 0.999))
) -> Dict[str, object]:
    """
    Ajusta GARCH(1,1) do seu metrics/garch.py e retorna:
      - 'params': (omega, alpha, beta)
      - 'cond_vol': pandas.Series (vol condicional anualizada)
      - 'forecast_next': float (previsão 1 step anualizada)
    """
    r = _to_series(log_rets)
    model = GARCH11(annualization=annualization)
    model.fit(r, bounds=bounds)
    return {
        "params": model.params_,
        "cond_vol": model.conditional_vol(),
        "forecast_next": model.forecast_next_vol()
    }
