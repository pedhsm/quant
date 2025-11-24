# strats/c1_hedged_portfolio.py
# (Estratégia 4: "Hedge Real")

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pykalman import KalmanFilter # Reusa a lógica de sinal do Kalman

# Definição local de StrategyResult
@dataclass
class StrategyResult:
    signal: pd.Series
    metadata: dict = None

class C1HedgedPortfolio:
    """
    Hipótese de "Hedge Real".
    Usa o mesmo SINAL do C1SpreadKalman (Z-Score do spread).
    Mas a EXECUÇÃO é diferente: em vez de comprar/vender BTC,
    compra/vende o portfólio ponderado completo 
    (ex: w_btc*btc + w_xau*xau + w_ng*ng)
   
    """
    def __init__(self) -> None:
        self.name = "C1HedgedPortfolio"

    def _kalman_zscore(self, s: pd.Series) -> pd.Series:
        """Lógica de Z-Score idêntica ao C1SpreadKalman."""
        kf = KalmanFilter(
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01,
            transition_matrices=[1]
        )
        state_means, _ = kf.filter(s.values)
        state_means = pd.Series(state_means.flatten(), index=s.index)
        resid = s - state_means
        std_resid = resid.rolling(window=30, min_periods=10).std()
        z = resid / (std_resid + 1e-9)
        return z.dropna()

    def generate_signal(self, ohlc: pd.DataFrame, **kwargs) -> StrategyResult:
        """
        Gera sinal de trading.
        'ohlc' deve conter 'spread' (para o sinal)
        e os pesos (ex: 'w_BTC') e retornos (ex: 'btc_ret')
        """
        entry_z = kwargs.get('entry_z', 2.0)
        exit_z = kwargs.get('exit_z', 0.5)

        if 'spread' not in ohlc.columns:
            raise ValueError("Coluna 'spread' não encontrada.")
            
        # 1. Lógica de SINAL (idêntica ao Kalman)
        spread = ohlc['spread'].dropna()
        z = self._kalman_zscore(spread)
        z = z.reindex(spread.index, method='ffill')

        pos = pd.Series(np.nan, index=z.index)
        pos[z < -entry_z] = 1   # Entra long o spread
        pos[z > entry_z] = -1  # Entra short o spread
        pos[z.abs() <= exit_z] = 0 # Sai (flat)
        pos = pos.ffill().fillna(0)

        # 2. Lógica de EXECUÇÃO ("Hedge Real")
        
        # Encontra dinamicamente os pesos e retornos no dataframe
        # (Nomes vêm do 05_build_spread_static.py e 09_run_mcp_macro.py)
        asset_map = {
            "btc": ("w_BTC", "btc_ret"),
            "xau": ("w_XAU-USD", "xau_ret"),
            "ng":  ("w_Gas Natural Futures", "ng_ret")
        }
        
        hedged_ret = pd.Series(0.0, index=ohlc.index)
        
        for key, (weight_col, ret_col) in asset_map.items():
            if weight_col in ohlc.columns and ret_col in ohlc.columns:
                hedged_ret += (ohlc[weight_col] * ohlc[ret_col])
            elif weight_col in ohlc.columns:
                # Se o peso existir mas o retorno não, algo está errado
                print(f"Aviso: Peso '{weight_col}' encontrado, mas retorno '{ret_col}' não.")

        hedged_ret = hedged_ret.fillna(0)
        
        # 3. P&L (posição de ontem * retorno do portfólio hedged de hoje)
        signal = pos.shift(1) * hedged_ret
        signal = signal.fillna(0)
        
        return StrategyResult(signal=signal.rename("ret"), metadata={
            'entry_z': entry_z,
            'exit_z': exit_z
        })