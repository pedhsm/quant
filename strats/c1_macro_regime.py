# c2_macro_regime.py (Corrigido para o novo fluxo SSoT)
# Estratégia de Mean Reversion baseada em quantil de spread

import pandas as pd
import numpy as np
from dataclasses import dataclass # <-- ADICIONADO

# --- REMOVIDA a dependência 'from mcp.base import ...' ---

# ADICIONADO: Definição local de StrategyResult
@dataclass
class StrategyResult:
    signal: pd.Series
    metadata: dict = None

class C2MacroRegime: # <-- REMOVIDO (TradingStrategy)
    """
    Estratégia de Mean Reversion em Spread com filtro de regime macro.
    Usa o quantil do spread para definir bandas de entrada/saída.
    """
    def __init__(self) -> None:
        # (Se você tiver parâmetros __init__, eles permanecem)
        self.name = "C2MacroRegime"

    def _zscore(self, s: pd.Series, lookback: int = 60) -> pd.Series:
        """Calcula z-score simples."""
        roll = s.rolling(lookback)
        return (s - roll.mean()) / (roll.std() + 1e-9)

    def generate_signal(self, ohlc: pd.DataFrame, **kwargs) -> StrategyResult:
        """
        Gera sinal de trading.
        'ohlc' é o DataFrame do 09_run_mcp_macro (que já contém 'spread').
        """
        # Pega parâmetros com valores padrão
        signal_quantile = kwargs.get('signal_quantile', 0.75)
        
        # (lógica original)
        if 'spread' not in ohlc.columns:
            raise ValueError("Coluna 'spread' não encontrada. "
                             "Use --spread-csv no 09_run_mcp_macro.py")
            
        spread = ohlc['spread'].dropna()
        
        # Calcula bandas de quantil
        q_upper = spread.quantile(signal_quantile)
        q_lower = spread.quantile(1.0 - signal_quantile)
        q_mid = spread.median()

        # Gera posições
        # 1 = Long Spread (comprado), -1 = Short Spread (vendido), 0 = Flat
        pos = pd.Series(np.nan, index=spread.index)
        pos[spread < q_lower] = 1  # Entra long
        pos[spread > q_upper] = -1 # Entra short
        pos[spread.between(q_lower, q_upper)] = 0 # Fecha posição
        
        # Preenche dias de 'hold' (ffill)
        pos = pos.ffill().fillna(0)

        # Assume que o "spread" é negociável (ex: long spread = long btc, short xau)
        # O retorno do 'spread' já deve estar no ohlc, ou calculamos aqui
        # Para ser simples, vamos assumir que o 'spread' é o ativo que compramos
        # e usamos 'btc_ret' como o retorno base
        
        if 'btc_ret' not in ohlc.columns:
             raise ValueError("c2_macro_regime precisa de 'btc_ret' no SSoT "
                              "para calcular o P&L.")
        
        # Alinha a série de retornos (btc_ret) com o índice do ohlc
        # Pega o shift(1) da posição para evitar lookahead
        signal = pos.shift(1) * ohlc['btc_ret']
        signal = signal.fillna(0)

        return StrategyResult(signal=signal.rename("ret"), metadata={
            'quantile_upper': q_upper,
            'quantile_lower': q_lower
        })