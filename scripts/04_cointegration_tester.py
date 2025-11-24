# Testa COINTEGRAÇÃO (Engle-Granger e Johansen) em dados SSoT.
#
# Uso (exemplos):
#   # Usa o SSoT padrão (clean_prices) e colunas específicas
#   python scripts/04_cointegration_tester.py --cols BTC XAU-USD "Gas Natural Futures"
#
#   # Usa o SSoT, mas reamostra para semanal (W-FRI) e define uma janela
#   python scripts/04_cointegration_tester.py --cols BTC XAU-USD --freq W-FRI --start 2018-01-01
#
#   # Usa uma SSoT alternativa (idio_prices)
#   python scripts/04_cointegration_tester.py --src data/idio_prices.csv --cols BTC XAU-USD

import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_order

def engle_granger_test(y: pd.Series, x: pd.Series):
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    if df.shape[0] < 25: return None
    Y = df.iloc[:, 0].values
    X = add_constant(df.iloc[:, 1].values)
    model = OLS(Y, X, missing="drop").fit()
    beta = model.params[1]
    resid = model.resid
    adf_res = adfuller(resid, regression="n", maxlag=None, autolag="AIC")
    adf_stat = adf_res[0]
    pvalue = adf_res[1]
    crit = {"1%": adf_res[4].get("1%"), "5%": adf_res[4].get("5%"), "10%": adf_res[4].get("10%")}
    return {"nobs": int(len(resid)), "beta": float(beta), "adf_stat": float(adf_stat), "pvalue": float(pvalue), "crit_1pct": float(crit["1%"]) if crit["1%"] is not None else np.nan, "crit_5pct": float(crit["5%"]) if crit["5%"] is not None else np.nan, "crit_10pct": float(crit["10%"]) if crit["10%"] is not None else np.nan}

def engle_granger_bidirectional(a: pd.Series, b: pd.Series, name_a: str, name_b: str):
    r1 = engle_granger_test(a.rename(name_a), b.rename(name_b))
    r2 = engle_granger_test(b.rename(name_b), a.rename(name_a))
    if r1 is None and r2 is None: return None
    if r1 is None: 
        r2["direction"] = f"{name_b}~{name_a}"
        return r2
    if r2 is None: 
        r1["direction"] = f"{name_a}~{name_b}"
        return r1
    best = r1 if r1["pvalue"] < r2["pvalue"] else r2
    best["direction"] = f"{name_a}~{name_b}" if best is r1 else f"{name_b}~{name_a}"
    return best

def johansen_test(df: pd.DataFrame, det_order=0, max_lags=8):
    # (Código da função johansen_test omitido por brevidade)
    # ...
    if df.shape[1] < 2 or df.shape[0] < 25: return None
    try:
        sel = select_order(df.diff().dropna(), maxlags=max_lags)
        k_ar_diff = sel.aic or 1
        if k_ar_diff is None or k_ar_diff < 1: k_ar_diff = 1
    except Exception:
        k_ar_diff = 1
    cj = coint_johansen(df.values, det_order, k_ar_diff)
    tr_stat = cj.lr1
    tr_crit = cj.cvt[:, 1]
    rank = int(np.sum(tr_stat > tr_crit))
    return {"nobs": int(df.shape[0]), "k_ar_diff": int(k_ar_diff), "trace_stats": [float(x) for x in tr_stat], "trace_crit_5pct": [float(x) for x in tr_crit], "rank_5pct": int(rank), "evec_first": [float(x) for x in cj.evec[:, 0]]}

def main():
    ap = argparse.ArgumentParser(description="Testa cointegração (Engle–Granger e Johansen) a partir de um SSoT (clean_prices).")

    ap.add_argument(
        "--src",
        default="data/clean_prices.csv",
        help="Arquivo SSoT para carregar. Padrão: data/clean_prices.csv"
    )
    ap.add_argument("--cols", nargs="+", required=True,
                help="Colunas do SSoT para testar (ex.: BTC XAU-USD)")
    
    ap.add_argument("--start", default=None, help="Início (YYYY-MM-DD). Opcional.")
    ap.add_argument("--end",   default=None, help="Fim (YYYY-MM-DD). Opcional.")
    ap.add_argument("--freq",  default=None, help="Frequência de reamostra (ex.: D, W-FRI, M). Opcional.")
    ap.add_argument("--out",   default="cointegration_results.csv", help="CSV de saída com resumo.")
    
    args = ap.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        print(f"Erro: Arquivo SSoT não encontrado: {src_path}. Execute o Step 01.", file=sys.stderr)
        sys.exit(1)
        
    df_panel = pd.read_csv(src_path, parse_dates=[0], index_col=0).sort_index()

    if args.cols is None or len(args.cols) < 2:
        print("Erro: use --cols para especificar duas ou mais colunas (ex.: --cols BTC XAU-USD).", file=sys.stderr)
        sys.exit(1)

    missing = [c for c in args.cols if c not in df_panel.columns]
    if missing:
        print(f"Erro: Colunas ausentes no SSoT {src_path.name}: {missing}", file=sys.stderr)
        sys.exit(1)

    df = df_panel[args.cols].copy()
    
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start, tz='UTC')]
    if args.end:
        df = df[df.index <= pd.Timestamp(args.end, tz='UTC')]
    if args.freq:
        df = df.resample(args.freq).last()
    
    df = df.dropna(how="any")

    if df.shape[1] < 2 or df.shape[0] < 25:
        print("Dados insuficientes após filtragem (precisa de ≥2 séries e ≥25 observações).", file=sys.stderr)
        sys.exit(2)

    rows = []
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            A, B = cols[i], cols[j]
            res = engle_granger_bidirectional(df[A], df[B], A, B)
            if res is None:
                rows.append({"pair": f"{A}-{B}", "direction": "", "nobs": df[[A,B]].dropna().shape[0], "beta": np.nan, "adf_stat": np.nan, "pvalue": np.nan, "crit_1pct": np.nan, "crit_5pct": np.nan, "crit_10pct": np.nan, "engle_granger_cointegrated_5pct": False})
            else:
                ok_5 = (res["adf_stat"] < res["crit_5pct"])
                rows.append({"pair": f"{A}-{B}", "direction": res["direction"], "nobs": res["nobs"], "beta": res["beta"], "adf_stat": res["adf_stat"], "pvalue": res["pvalue"], "crit_1pct": res["crit_1pct"], "crit_5pct": res["crit_5pct"], "crit_10pct": res["crit_10pct"], "engle_granger_cointegrated_5pct": bool(ok_5)})

    out_df = pd.DataFrame(rows)

    joh = None
    if df.shape[1] >= 3:
        try:
            joh = johansen_test(df, det_order=0, max_lags=8)
        except Exception as e:
            joh = {"error": str(e)}

    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Engle–Granger salvo em: {out_path} (Fonte={src_path.name}, freq={args.freq}, n={len(df)})")
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    if joh and "error" not in joh:
        print("\n--- Johansen (5%) ---")
        print(f"nobs={joh['nobs']}, k_ar_diff={joh['k_ar_diff']}, rank_5pct={joh['rank_5pct']}")
    elif joh and "error" in joh:
        print("\n[Johansen] erro:", joh["error"])

if __name__ == "__main__":
    main()