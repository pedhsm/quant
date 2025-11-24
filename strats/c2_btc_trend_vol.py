# c3_btc_trend_vol.py (Corrigido para o novo fluxo SSoT)
# Estratégia de Trend Following + Volatility Targeting no BTC.

import pandas as pd
import numpy as np
from dataclasses import dataclass # <-- ADICIONADO

# --- REMOVIDA a dependência 'from mcp.base import ...' ---

# ADICIONADO: Definição local de StrategyResult
@dataclass
class StrategyResult:
    signal: pd.Series
    metadata: dict = None

class C3BtcTrendVol: # <-- REMOVIDO (TradingStrategy)
    """
    Estratégia de Trend Following no BTC com Volatility Targeting.
    - Sinal de Trend: Média Móvel (MM)
    - Alocação: Inverso da volatilidade (Vol Targeting)
    """
    def __init__(self) -> None:
        self.name = "C3BtcTrendVol"
        # (lógica original)
        self.mm_fast = 5
        self.mm_slow = 20
        self.vol_target = 0.02 # Alvo de 2% de vol diária
        self.vol_lookback = 20
    
    def configure(self, **kwargs):
        """Permite que o 09_run_mcp_macro configure os parâmetros."""
        self.mm_fast = kwargs.get('mm_fast', self.mm_fast)
        self.mm_slow = kwargs.get('mm_slow', self.mm_slow)
        self.vol_target = kwargs.get('vol_target', self.vol_target)

    def generate_signal(self, ohlc: pd.DataFrame, **kwargs) -> StrategyResult:
        """
        Gera sinal de trading.
        'ohlc' é o SSoT (clean_prices) que deve conter 'btc_ret' e 'btc' (preço).
        """
        # (lógica original)
        if 'btc' not in ohlc.columns:
            raise ValueError("Coluna 'btc' (preço) não encontrada no SSoT")
        if 'btc_ret' not in ohlc.columns:
            raise ValueError("Coluna 'btc_ret' (retorno) não encontrada no SSoT")
            
        self.configure(**kwargs) 

        # 1. Sinal de Trend (cruzamento de médias)
        fast = ohlc['btc'].rolling(self.mm_fast).mean()
        slow = ohlc['btc'].rolling(self.mm_slow).mean()
        trend_signal = pd.Series(0, index=ohlc.index)
        trend_signal[fast > slow] = 1  # Long
        trend_signal[fast < slow] = -1 # Short (ou 0 se for long-only)
        
        # 2. Alocação (Vol Targeting)
        # Calcula vol realizada (shift(1) para usar a vol 'de ontem')
        realized_vol = ohlc['btc_ret'].rolling(self.vol_lookback).std().shift(1)
        
        # Alocação é o inverso da vol (com teto de 2x)
        allocation = self.vol_target / (realized_vol + 1e-9)
        allocation = allocation.clip(0, 2.0) # Limita alavancagem

        # 3. Posição Final
        # (shift(1) no sinal de trend para evitar lookahead)
        position = trend_signal.shift(1) * allocation
        position = position.fillna(0)
        
        # 4. P&L
        signal = (position * ohlc['btc_ret']).fillna(0)

        return StrategyResult(signal=signal.rename("ret"), metadata={
            'mm_fast': self.mm_fast,
            'mm_slow': self.mm_slow
        })