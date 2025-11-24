# scripts/09_run_mcp_macro.py 
# Runner MCP. Carrega todos os SSoTs, incluindo o BAMLH0A0HYM2.

import argparse, importlib, os, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, UTC 

DATA = Path("data")

def _import_strategy(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    if not hasattr(mod, class_name):
        raise ImportError(f"Classe '{class_name}' não encontrada em {module_path}")
    return getattr(mod, class_name)

def _load_clean():
    cp = DATA / "clean_prices.csv"
    lr = DATA / "log_returns.csv"
    if not cp.exists() or not lr.exists():
        raise FileNotFoundError("Run Step 01 primeiro: data/clean_prices.csv e data/log_returns.csv não encontrados.")
    prices = pd.read_csv(cp, index_col=0, parse_dates=True)
    rets   = pd.read_csv(lr, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index, utc=True)
    rets.index = pd.to_datetime(rets.index, utc=True)
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    rets   = rets[~rets.index.duplicated(keep="last")].sort_index()
    return prices, rets

def _auto_pick(colnames, keywords):
    low = [c.lower() for c in colnames]
    for i, c in enumerate(low):
        if all(k in c for k in keywords):
            return colnames[i]
    return None

def build_features(
    btc_col=None, 
    xau_col=None, 
    ng_col=None, 
    spread_csv=None,
    spread_weights=None, 
    regimes_csv=None     
):
    prices, rets = _load_clean()

    btc_col = btc_col or _auto_pick(prices.columns, ["btc"]) or _auto_pick(prices.columns, ["bitcoin"])
    xau_col = xau_col or _auto_pick(prices.columns, ["xau"]) or _auto_pick(prices.columns, ["gold","usd"])
    ng_col  = ng_col  or _auto_pick(prices.columns, ["gas","natural"]) or _auto_pick(prices.columns, ["ng"])
    missing = [k for k,v in {"btc":btc_col,"xau":xau_col,"ng":ng_col}.items() if v is None]
    if missing: raise ValueError(f"Não consegui mapear colunas para: {missing}.")

    feat = pd.DataFrame(index=prices.index)
    feat["btc"] = prices[btc_col]
    feat["xau"] = prices[xau_col]
    feat["ng"]  = prices[ng_col]

    macro_cols = ['vix', 'juros_10a', 'dolar', 'petroleo', 'BAMLH0A0HYM2']
    for col in macro_cols:
        if col in prices.columns:
            feat[col] = prices[col]

    feat["btc_ret"] = rets[btc_col]
    feat["xau_ret"] = rets[xau_col]
    feat["ng_ret"]  = rets[ng_col]

    if spread_csv:
        sc = pd.read_csv(spread_csv, index_col=0, parse_dates=True)
        sc.index = pd.to_datetime(sc.index, utc=True)
        scol = sc.columns[0]
        feat["spread"] = pd.to_numeric(sc[scol], errors="coerce")

    if spread_weights:
        with open(spread_weights, 'r') as f: weights = json.load(f)
        for asset_name, weight in weights.items():
            feat[f"w_{asset_name}"] = float(weight)

    if regimes_csv:
        rc = pd.read_csv(regimes_csv, index_col=0, parse_dates=True)
        rc.index = pd.to_datetime(rc.index, utc=True)
        for col in rc.columns:
            if col not in feat.columns: 
                feat[col] = rc[col]

    feat = feat.sort_index()
    return feat

def main():
    ap = argparse.ArgumentParser(description="Passo 09 (MCP) v4")
    ap.add_argument("--strategy-module", required=True)
    ap.add_argument("--strategy-class",  required=True)
    ap.add_argument("--spread-csv", default=None)
    ap.add_argument("--spread-weights", default=None)
    ap.add_argument("--regimes-csv", default="data/btc_regimes.csv")
    ap.add_argument("--btc-col", default=None)
    ap.add_argument("--xau-col", default=None)
    ap.add_argument("--ng-col",  default=None)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--stat", choices=["sharpe","cagr","vol"], default="sharpe")
    args = ap.parse_args()

    ohlc = build_features(
        args.btc_col, 
        args.xau_col, 
        args.ng_col, 
        args.spread_csv,
        args.spread_weights,
        args.regimes_csv
    )

    start_date = pd.Timestamp("2018-01-01", tz='UTC')
    ohlc = ohlc[ohlc.index >= start_date]

    Strategy = _import_strategy(args.strategy_module, args.strategy_class)
    strat = Strategy()
    if hasattr(strat, "configure"):
        try: strat.configure(signal_quantile=args.q)
        except TypeError: pass

    if not hasattr(strat, "generate_signal"):
        raise AttributeError(f"{args.strategy_class} não possui generate_signal(ohlc, **kwargs).")
    
    res = strat.generate_signal(ohlc, signal_quantile=args.q)
    signal = res.signal

    (DATA / "checkpoints").mkdir(parents=True, exist_ok=True)
    signal.to_csv(DATA / "checkpoints/LAST_RUN_SIGNAL.csv")

    try:
        from mcp.tester import permutation_test
        out = permutation_test(signal, metric=args.stat, n_permutations=args.perms)
        print(f"[MCP] {args.stat}: real={out['real']:.6f} | pvalue={out['pvalue']:.6f}")
    except Exception as e:
        print(f"[MCP] aviso: não executei permutação ({e})")

    outdir = DATA / "mcp"; outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.spread_csv).stem if args.spread_csv else "no_spread"
    stamp = datetime.now(UTC).strftime("%Y%m%d") # Corrigido
    out_path = outdir / f"{args.strategy_class}__{tag}__ret_{stamp}.csv"
    pd.DataFrame({"ret": signal}).to_csv(out_path)
    print(f"[OK] {out_path}")

if __name__ == "__main__":
    main()