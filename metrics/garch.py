# metrics/garch.py
import numpy as np
import pandas as pd

def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series): return x.dropna()
    return pd.Series(x).dropna()

class GARCH11:
    """
    Estimador simples de GARCH(1,1) via grid-search (sem SciPy).
    Foco: gerar série de sigma_t e previsão 1-step para dimensionamento de posição.
    """
    def __init__(self, annualization: int = 252):
        self.annualization = annualization
        self.params_ = None  
        self.sigma2_ = None  

    @staticmethod
    def _loglik(r, omega, alpha, beta, backcast=None):
        T = len(r)
        if T == 0: return np.inf
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1:
            return np.inf
        sigma2 = np.empty(T)
        if backcast is None:
            backcast = np.var(r[:min(20, T)]) if T > 1 else 1e-6
        sigma2[0] = backcast
        for t in range(1, T):
            sigma2[t] = omega + alpha * (r[t-1]**2) + beta * sigma2[t-1]
        ll = -0.5 * np.sum(np.log(2*np.pi) + np.log(sigma2) + (r**2)/sigma2)
        return -ll, sigma2  

    def fit(self, rets: pd.Series, grid_omega=None, grid_alpha=None, grid_beta=None, mean_zero: bool=True):
        r = _to_series(rets).values.astype(float)
        r = r - (0 if mean_zero else np.mean(r))

        if grid_alpha is None: grid_alpha = np.linspace(0.01, 0.20, 10)
        if grid_beta  is None: grid_beta  = np.linspace(0.70, 0.98, 15)
        var_r = np.var(r) if len(r) > 1 else 1e-6
        if grid_omega is None: grid_omega = np.linspace(1e-8, 0.2*var_r, 10)

        best = (np.inf, None, None)
        for a in grid_alpha:
            for b in grid_beta:
                if a + b >= 0.999:  
                    continue
                for w in grid_omega:
                    val, sigma2 = self._loglik(r, w, a, b)
                    if val < best[0]:
                        best = (val, (w, a, b), sigma2)

        self.params_ = best[1]
        self.sigma2_ = best[2]
        return self

    def conditional_vol(self) -> pd.Series:
        if self.sigma2_ is None:
            raise RuntimeError("Chame fit() antes.")
        vol_daily = np.sqrt(self.sigma2_)
        return pd.Series(vol_daily * np.sqrt(self.annualization))

    def forecast_next_vol(self) -> float:
        """Previsão 1-step à frente (anualizada)."""
        if self.params_ is None or self.sigma2_ is None:
            raise RuntimeError("Chame fit() antes.")
        omega, alpha, beta = self.params_
        eps2_last = self.sigma2_[-1] 
        sigma2_next = omega + alpha * eps2_last + beta * self.sigma2_[-1]
        return float(np.sqrt(sigma2_next) * np.sqrt(self.annualization))
