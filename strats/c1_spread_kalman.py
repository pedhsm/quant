# c3_spread_kalman.py (Corrigido para o novo fluxo SSoT)
# Estratégia de Mean Reversion em Spread com filtro de Kalman (Z-Score)

import pandas as pd
import numpy as np
from dataclasses import dataclass # <-- ADICIONADO
from pykalman import KalmanFilter

# --- REMOVIDA a dependência 'from mcp.base import ...' ---

# ADICIONADO: Definição local de StrategyResult
@dataclass
class StrategyResult:
    signal: pd.Series
    metadata: dict = None

class C3SpreadKalman: # <-- REMOVIDO (TradingStrategy)
    """
    Estratégia de Mean Reversion em Spread (estático ou dinâmico).
    Usa Z-Score (via filtro de Kalman) para sinais de entrada/saída.
    """
    def __init__(self) -> None:
        self.name = "C3SpreadKalman"

    def _kalman_zscore(self, s: pd.Series) -> pd.Series:
        """Calcula o z-score usando filtro de Kalman para média e std dinâmicos."""
        # (lógica original)
        kf = KalmanFilter(
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01,
            transition_matrices=[1]
        )
        state_means, _ = kf.filter(s.values)
        state_means = pd.Series(state_means.flatten(), index=s.index)

        # Z-score dinâmico
        resid = s - state_means
        std_resid = resid.rolling(window=30, min_periods=10).std()
        z = resid / (std_resid + 1e-9)
        return z.dropna()

    def generate_signal(self, ohlc: pd.DataFrame, **kwargs) -> StrategyResult:
        """
        Gera sinal de trading.
        'ohlc' é o DataFrame do 09_run_mcp_macro (que já contém 'spread').
        """
        # Pega parâmetros
        entry_z = kwargs.get('entry_z', 2.0)
        exit_z = kwargs.get('exit_z', 0.5)

        # (lógica original)
        if 'spread' not in ohlc.columns:
            raise ValueError("Coluna 'spread' não encontrada. "
                             "Use --spread-csv no 09_run_mcp_macro.py")
            
        spread = ohlc['spread'].dropna()
        z = self._kalman_zscore(spread)
        
        # Alinha z com o índice original
        z = z.reindex(spread.index, method='ffill')

        # Gera posições
        pos = pd.Series(np.nan, index=z.index)
        pos[z < -entry_z] = 1   # Entra long
        pos[z > entry_z] = -1  # Entra short
        pos[z.abs() <= exit_z] = 0 # Sai (flat)

        pos = pos.ffill().fillna(0)

        if 'btc_ret' not in ohlc.columns:
             raise ValueError("c3_spread_kalman precisa de 'btc_ret' no SSoT "
                              "para calcular o P&L.")
        
        # Pega o shift(1) da posição para evitar lookahead
        signal = pos.shift(1) * ohlc['btc_ret']
        signal = signal.fillna(0)
        
        return StrategyResult(signal=signal.rename("ret"), metadata={
            'entry_z': entry_z,
            'exit_z': exit_z
        })