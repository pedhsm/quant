# strats/c3_stress_hedge.py
# (Estratégia 3: "Flight-to-Quality Relativo")

import pandas as pd
import numpy as np
from dataclasses import dataclass

# Definição local de StrategyResult (padrão de nossas estratégias)
@dataclass
class StrategyResult:
    signal: pd.Series
    metadata: dict = None

class C3StressHedge:
    """
    Hipótese da Estratégia 3 da IA: "Flight-to-Quality Relativo"
   
    
    Usa o spread de crédito High-Yield (BAMLH0A0HYM2) como indicador 
    de "estresse macro", em vez do VIX.
    
    Quando o estresse dispara (spread > 2 DP acima da média de 1 ano), 
    faz uma aposta relativa: LONG XAU (seguro) e SHORT BTC (risco).
    """
    def __init__(self) -> None:
        self.name = "C3StressHedge"
        self.stress_lookback = 252 # Média móvel de 1 ano
        self.entry_z = 2.0 # Nível de Z-Score para "pânico"
        self.exit_z = 0.5 # Nível de Z-Score para "normalizar"
    
    def configure(self, **kwargs):
        pass # (Sem parâmetros de configure por enquanto)

    def generate_signal(self, ohlc: pd.DataFrame, **kwargs) -> StrategyResult:
        """
        Gera sinal de trading.
        'ohlc' deve conter 'BAMLH0A0HYM2', 'btc_ret', 'xau_ret'.
        """
        stress_col = 'BAMLH0A0HYM2' #
        
        if stress_col not in ohlc.columns:
            raise ValueError(f"Coluna '{stress_col}' não encontrada no SSoT. "
                             "Execute o Step 01 modificado.")
        if 'btc_ret' not in ohlc.columns or 'xau_ret' not in ohlc.columns:
            raise ValueError("btc_ret ou xau_ret não encontrados no SSoT.")
            
        # 1. Calcular o Z-Score do indicador de estresse
        stress = ohlc[stress_col].dropna()
        stress_mean = stress.rolling(self.stress_lookback).mean()
        stress_std = stress.rolling(self.stress_lookback).std()
        
        z_score = (stress - stress_mean) / (stress_std + 1e-9)
        z_score = z_score.reindex(ohlc.index, method='ffill')

        # 2. Gerar Posição
        # Posição 1 = Estresse (LONG XAU / SHORT BTC)
        # Posição 0 = Normal (FLAT)
        pos = pd.Series(np.nan, index=z_score.index)
        pos[z_score > self.entry_z] = 1  # Entra em modo "Pânico"
        pos[z_score < self.exit_z] = 0   # Sai do modo "Pânico"
        pos = pos.ffill().fillna(0)
        
        # 3. Calcular P&L da aposta relativa
        # (Hipótese 1: Ouro Ganha)
        # Posição longa em XAU, Posição curta em BTC
        relative_ret = ohlc['xau_ret'] - ohlc['btc_ret']
        
        # 4. P&L Final (posição de ontem * retorno relativo de hoje)
        signal = pos.shift(1) * relative_ret
        signal = signal.fillna(0)

        return StrategyResult(signal=signal.rename("ret"), metadata={
            'entry_z': self.entry_z,
            'exit_z': self.exit_z
        })