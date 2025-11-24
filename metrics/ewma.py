# metrics/realized.py
import numpy as np
import pandas as pd

def to_series(x) -> pd.Series:
    if isinstance(x, pd.Series): return x.dropna()
    return pd.Series(x).dropna()

def log_returns(price: pd.Series) -> pd.Series:
    s = to_series(price).astype(float)
    return np.log(s).diff().dropna()

def realized_vol(rets: pd.Series, window: int = 20, annualization: int = 252) -> pd.Series:
    r = to_series(rets)
    return r.rolling(window).std(ddof=0) * np.sqrt(annualization)

def ewma_vol(rets: pd.Series, lam: float = 0.94, annualization: int = 252) -> pd.Series:
    """
    EWMA de RiskMetrics. lam ~ 0.94 (diário). Half-life ≈ ln(2)/ln(1/lam).
    """
    r = to_series(rets)
    var = np.empty(len(r))
    var[:] = np.nan
    if len(r) == 0: return pd.Series(var, index=r.index)
    init_n = min(20, len(r))
    backcast = np.var(r.iloc[:init_n]) if init_n > 1 else 1e-6
    var[0] = backcast
    for t in range(1, len(r)):
        var[t] = lam * var[t-1] + (1 - lam) * (r.iloc[t-1]**2)
    vol = np.sqrt(var) * np.sqrt(annualization)
    return pd.Series(vol, index=r.index)
