# strats/c3_macro_filtered_trend.py
# (Estratégia 1: Trend-Following Macro-Filtrado)

import pandas as pd
import numpy as np
from dataclasses import dataclass

# Definição local de StrategyResult
@dataclass
class StrategyResult:
    signal: pd.Series
    metadata: dict = None

class C3MacroFilteredTrend:
    """
    Evolução do C3BtcTrendVol.
    A lógica de Trend (MM Cross + Vol Target) só é ativada
    se o Filtro Macro (regime_mm50w) for 'BULL'.
   
    """
    def __init__(self) -> None:
        self.name = "C3MacroFilteredTrend"
        self.mm_fast = 5
        self.mm_slow = 20
        self.vol_target = 0.02
        self.vol_lookback = 20
    
    def configure(self, **kwargs):
        self.mm_fast = kwargs.get('mm_fast', self.mm_fast)
        self.mm_slow = kwargs.get('mm_slow', self.mm_slow)
        self.vol_target = kwargs.get('vol_target', self.vol_target)

    def generate_signal(self, ohlc: pd.DataFrame, **kwargs) -> StrategyResult:
        """
        Gera sinal de trading.
        'ohlc' deve conter 'btc', 'btc_ret', 
        'regime_mm50w', e 'STH_flag_proxy'.
        """
        # Verifica se os dados necessários (do regimes.csv) estão presentes
        if 'regime_mm50w' not in ohlc.columns:
            raise ValueError("Coluna 'regime_mm50w' não encontrada. "
                             "Verifique se 09_run_mcp_macro.py está carregando --regimes-csv.")
        if 'STH_flag_proxy' not in ohlc.columns:
            raise ValueError("Coluna 'STH_flag_proxy' não encontrada.")
            
        self.configure(**kwargs) 

        # 1. Lógica de Trend (idêntica à C3BtcTrendVol)
        fast = ohlc['btc'].rolling(self.mm_fast).mean()
        slow = ohlc['btc'].rolling(self.mm_slow).mean()
        trend_signal = pd.Series(0, index=ohlc.index)
        trend_signal[fast > slow] = 1
        trend_signal[fast < slow] = -1
        
        # 2. Lógica de Alocação (idêntica à C3BtcTrendVol)
        realized_vol = ohlc['btc_ret'].rolling(self.vol_lookback).std().shift(1)
        allocation = self.vol_target / (realized_vol + 1e-9)
        allocation = allocation.clip(0, 2.0)
        
        # 3. OS NOVOS FILTROS MACRO
        
        # Filtro 1: Regime Macro (só opera em BULL)
        is_bull_regime = (ohlc['regime_mm50w'] == 'BULL')
        trend_signal[~is_bull_regime] = 0 # Fica FLAT se não for BULL
        
        # Filtro 2: Risco de Topo Local (Filtro STH)
        is_risky_top = (ohlc['STH_flag_proxy'] == 'RISCO_TOPO_LOCAL')
        trend_signal[is_risky_top] = 0 # Fica FLAT se estiver em risco
        
        # 4. Posição Final (com shift(1) para evitar lookahead)
        position = trend_signal.shift(1) * allocation
        position = position.fillna(0)
        
        # 5. P&L
        signal = (position * ohlc['btc_ret']).fillna(0)

        return StrategyResult(signal=signal.rename("ret"), metadata={
            'mm_fast': self.mm_fast,
            'mm_slow': self.mm_slow
        })