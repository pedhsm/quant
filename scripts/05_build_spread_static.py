"""
Passo 5: construir SPREAD estático.

Lê dados de um SSoT (Fonte Única da Verdade) e aplica Johansen.
AGORA INCLUI FILTRO DE DATA para que os pesos do spread
sejam calculados no mesmo período da análise de cointegração.

Uso:
  # Usa preços limpos, mas calcula pesos APENAS de 2018 em diante
  python scripts/05_build_spread_static.py --cols "BTC,XAU-USD" --start 2018-01-01
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def hedge_from_johansen(log_df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> np.ndarray:
    """
    Retorna vetor w (pesos) do primeiro eigenvector do Johansen, normalizado c/ último termo = 1.
   
    """
    res = coint_johansen(log_df, det_order=det_order, k_ar_diff=k_ar_diff)
    w = res.evec[:, 0].astype(float)        
    if np.isclose(w[-1], 0.0):
        w = w / (np.abs(w).max() + 1e-12)
    else:
        w = w / w[-1]
    return w

def build_spread(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> pd.Series:
    """
    Constrói spread estático: logS = log(P) @ w; S = exp(logS).
   
    """
    prices = prices.dropna()
    if prices.shape[0] < 250 or prices.shape[1] < 2:
        raise ValueError(f"Dados insuficientes p/ Johansen: shape={prices.shape}")
    logp = np.log(prices)
    w = hedge_from_johansen(logp, det_order=det_order, k_ar_diff=k_ar_diff)
    logS = logp.values @ w
    S = np.exp(logS)
    s = pd.Series(S, index=prices.index, name="spread")
    return s, w, list(prices.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", required=True, help="Lista de colunas, ex: SPY,BTC ou SPY,BTC,XAU-USD")
    
    ap.add_argument(
        "--src", 
        default="data/clean_prices.csv", 
        help="CSV fonte (SSoT). Padrão: data/clean_prices.csv. "
             "Use data/idio_prices.csv para spreads idiossincráticos."
    )
    
    ap.add_argument("--det", type=int, default=0, help="det_order no Johansen (default=0)")
    ap.add_argument("--lags", type=int, default=1, help="k_ar_diff no Johansen (default=1)")
    ap.add_argument("--start", default=None, help="Início (YYYY-MM-DD). Opcional.")
    ap.add_argument("--end",   default=None, help="Fim (YYYY-MM-DD). Opcional.")
    args = ap.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        raise FileNotFoundError(f"Arquivo SSoT não encontrado: {src_path}. "
                               f"Execute os steps anteriores da pipeline.")

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    if len(cols) < 2:
        raise SystemExit("--cols deve ter >= 2 colunas")

    df = pd.read_csv(src_path, index_col=0, parse_dates=True)

    df.index = pd.to_datetime(df.index, utc=True) 
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start, tz='UTC')]
    if args.end:
        df = df[df.index <= pd.Timestamp(args.end, tz='UTC')]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas ausentes em {src_path.name} (após filtro): {missing}")

    sub = df[cols].dropna()
    
    # Modificado para capturar pesos (para a próxima hipótese)
    s, weights, w_cols = build_spread(sub, det_order=args.det, k_ar_diff=args.lags)
    
    # Cria o dicionário de pesos
    weights_dict = {col: w for col, w in zip(w_cols, weights)}

    outdir = Path("data/spreads")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Salva o spread (CSV)
    out_name_base = '_'.join(cols)
    out_csv = outdir / f"{out_name_base}_spread_static.csv"
    s.to_csv(out_csv)
    
    # Salva os pesos (JSON) - para a próxima rodada de hipóteses
    import json
    out_json = outdir / f"{out_name_base}_weights.json"
    with open(out_json, 'w') as f:
        json.dump(weights_dict, f, indent=4)

    print(f"OK: spread salvo em {out_csv} (Fonte: {src_path.name}, Filtro: {args.start}-{args.end})")
    print(f"OK: pesos salvos em {out_json}")
    print(f"Pesos: {weights_dict}")

if __name__ == "__main__":
    main()