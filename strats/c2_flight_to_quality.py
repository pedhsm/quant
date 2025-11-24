# c2_flight_to_quality.py (Corrigido para o novo fluxo SSoT)
# Estratégia "Risk-Off" que aloca entre BTC e XAU com base no VIX.

import pandas as pd
import numpy as np
from dataclasses import dataclass # <-- ADICIONADO

# --- REMOVIDA a dependência 'from mcp.base import ...' ---

# ADICIONADO: Definição local de StrategyResult
@dataclass
class StrategyResult:
    signal: pd.Series
    metadata: dict = None

class C2FlightToQuality: # <-- REMOVIDO (TradingStrategy)
    """
    Estratégia de Rotação 'Flight to Quality'.
    - VIX alto (Risk-Off): aloca em XAU (Ouro)
    - VIX baixo (Risk-On): aloca em BTC
    """
    def __init__(self) -> None:
        self.name = "C2FlightToQuality"
        # (lógica original)
        self.weights_risk_on = {"btc_ret": 1.0}
        self.weights_risk_off = {"xau_ret": 1.0}
        self.signal_quantile = 0.75 # Quantil do VIX para 'Risk-Off'
    
    def configure(self, **kwargs):
        """Permite que o 09_run_mcp_macro configure os parâmetros."""
        self.weights_risk_on = kwargs.get('weights_risk_on', self.weights_risk_on)
        self.weights_risk_off = kwargs.get('weights_risk_off', self.weights_risk_off)
        self.signal_quantile = kwargs.get('signal_quantile', self.signal_quantile)

    def generate_signal(self, ohlc: pd.DataFrame, **kwargs) -> StrategyResult:
        """
        Gera sinal de trading.
        'ohlc' é o SSoT (clean_prices) que deve conter 'vix', 'btc_ret', 'xau_ret'.
        """
        # (lógica original)
        if 'vix' not in ohlc.columns:
            raise ValueError("Coluna 'vix' não encontrada no SSoT (clean_prices.csv)")
        if 'btc_ret' not in ohlc.columns:
            raise ValueError("Coluna 'btc_ret' não encontrada no SSoT")
        if 'xau_ret' not in ohlc.columns:
            raise ValueError("Coluna 'xau_ret' não encontrada no SSoT")
            
        self.configure(**kwargs) # Aplica parâmetros (ex: --q do 09_run_mcp_macro)

        # Calcula o regime
        q_val = ohlc['vix'].quantile(self.signal_quantile)
        regime = (ohlc['vix'] > q_val).astype(int) # 1 = Risk-Off, 0 = Risk-On

        # Gera P&L
        ret_risk_on = ohlc['btc_ret'] * self.weights_risk_on.get('btc_ret', 1.0)
        ret_risk_off = ohlc['xau_ret'] * self.weights_risk_off.get('xau_ret', 1.0)

        # Aloca o retorno com base no regime (shift 1 para evitar lookahead)
        signal = np.where(
            regime.shift(1) == 1, # Se ontem foi Risk-Off...
            ret_risk_off,         # ...hoje temos o retorno do XAU
            ret_risk_on           # ...senão, hoje temos o retorno do BTC
        )
        
        signal = pd.Series(signal, index=ohlc.index, name="ret").fillna(0)

        return StrategyResult(signal=signal, metadata={
            'vix_quantile_threshold': q_val,
            'signal_quantile': self.signal_quantile
        })