#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 01 (Refatorado v2): Script Mestre de ETL e Features.
Lê todos os CSVs de raw_data/, incluindo BAMLH0A0HYM2.csv.

"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)
PRICE_COL_CANDIDATES = ["Último","Fechamento","Close","Adj Close","close","Close Price","Price","Ultimo"]

def _read_euro_price_series(path: Path) -> pd.Series:
    """Lê CSV de preço (formato europeu), ; sep, , decimal."""
    df = pd.read_csv(path, sep=';', engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce", utc=True)
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    price_col = next((col for col in PRICE_COL_CANDIDATES if col in df.columns), None)
    if price_col is None: raise ValueError(f"Não encontrei coluna de preço em {path}")
    s = (
        df.set_index(date_col)[price_col]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
    )
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = Path(path).stem
    s.index = pd.to_datetime(s.index, utc=True)
    s = s[~s.index.duplicated(keep="last")]
    return s

def _read_macro_data(path: Path) -> pd.DataFrame:
    """Lê CSV de macro (formato limpo/FRED), , sep, . decimal."""
    df = pd.read_csv(path, sep=',', engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce", utc=True)
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    
    # Converte colunas para numérico, tratando erros (ex: '.' do FRED)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df[~df.index.duplicated(keep="last")]
    return df

def btc_regime_proxies(
    prices: pd.DataFrame,
    btc_col: str,
    last_halving: str | None = None,
    k_sth: float = 2.0
) -> pd.DataFrame:
    """Gera DataFrame com BTC, MM50W, regime, Mayer Multiple e STH proxy + dias pós-halving."""
    # (Código idêntico ao 01_build_dataset.py original)
    # (Início do código omitido)
    if btc_col not in prices.columns:
        candidates = [c for c in prices.columns if "BTC" in c.upper()]
        if not candidates:
            raise KeyError(f"Coluna do BTC '{btc_col}' não encontrada. Colunas: {list(prices.columns)}")
        btc_col = candidates[0]
    px = prices[btc_col].copy()
    w = px.resample("W-FRI").last()
    mm50w = w.rolling(50, min_periods=10).mean()
    mm50w_daily = mm50w.reindex(px.index, method="ffill")
    regime = pd.Series(
        np.where(px > mm50w_daily, "BULL",
        np.where(px < mm50w_daily, "BEAR", "NA")),
        index=px.index, name="regime_mm50w"
    )
    out = pd.DataFrame({"BTC": px, "MM50W": mm50w_daily, "regime_mm50w": regime}, index=px.index)
    mm200d = px.rolling(200, min_periods=150).mean()
    mayer = px / mm200d
    mayer_flag = np.where(mayer < 0.8, "OPORTUNIDADE",
                   np.where(mayer > 2.4, "EUFORIA", "NEUTRO"))
    out["MM200D"] = mm200d
    out["MayerMultiple"] = mayer
    out["Mayer_flag"] = mayer_flag
    mm155d = px.rolling(155, min_periods=120).mean()
    dev155 = px - mm155d
    sigma155 = dev155.rolling(155, min_periods=120).std()
    sth_upper_proxy = mm155d + k_sth * sigma155
    sth_flag_proxy = np.where(px >= sth_upper_proxy, "RISCO_TOPO_LOCAL",
                       np.where(px > mm155d, "ALTA_SAUDAVEL", "ABAIXO_CUSTO_155D"))
    out["MM155D_proxy"] = mm155d
    out["STH_upper_proxy"] = sth_upper_proxy
    out["STH_flag_proxy"] = sth_flag_proxy
    if last_halving:
        try:
            last = pd.to_datetime(last_halving, utc=True)
            days_since = (px.index - last).days
            out["dias_pos_halving"] = days_since
            out["halving_risco_temporal"] = np.where(days_since > 520, "ALTO", "NORMAL")
        except Exception:
            pass
    return out
    # (Fim do código omitido)


def main():
    ap = argparse.ArgumentParser(description="Step 01 (Refatorado v2): ETL Mestre + Features.")
    ap.add_argument("--raw-dir", default="raw_data", help="Diretório contendo os CSVs brutos")
    ap.add_argument("--btc-col", default="BTC", help="Nome da coluna do BTC")
    ap.add_argument("--last-halving", default=None, help="YYYY-MM-DD (opcional)")
    ap.add_argument("--sth-k", type=float, default=2.0, help="k da banda superior do STH proxy (default=2.0)")
    args = ap.parse_args()

    raw_path = Path(args.raw_dir)
    
    price_files = [
        raw_path / "BTC.csv",
        raw_path / "XAU-USD.csv",
        raw_path / "Gas Natural Futures.csv",
        raw_path / "SPY.csv",
        raw_path / "IDY.csv"
    ]
    price_series = []
    for f in price_files:
        if f.exists():
            print(f"[ETL] Processando {f.name} (Formato Europeu)...")
            price_series.append(_read_euro_price_series(f))

    # 2. Carregar os "dados macro"
    macro_file = raw_path / "macro_data.csv"
    macro_df = None
    if macro_file.exists():
        print(f"[ETL] Processando {macro_file.name} (Formato Limpo)...")
        macro_df = _read_macro_data(macro_file)

    # --- INÍCIO DA NOVA SEÇÃO (Carregar Dados de Estresse) ---
    stress_file = raw_path / "BAMLH0A0HYM2.csv"
    stress_df = None
    if stress_file.exists():
        print(f"[ETL] Processando {stress_file.name} (Formato FRED)...")
        stress_df = _read_macro_data(stress_file)
        # O FRED usa '.' para 'Not Available'. Precisamos limpá-los e preencher.
        stress_df = stress_df.apply(pd.to_numeric, errors='coerce').ffill()
    else:
        print(f"[ETL] Aviso: Arquivo de estresse não encontrado: {stress_file}")
    # --- FIM DA NOVA SEÇÃO ---

    # 3. Alinhar todos os dados
    all_data_frames = price_series
    if macro_df is not None:
        all_data_frames.extend([macro_df[col] for col in macro_df.columns])
    
    # --- INÍCIO DA NOVA SEÇÃO (Adicionar Estresse ao Join) ---
    if stress_df is not None:
        all_data_frames.extend([stress_df[col] for col in stress_df.columns])
    # --- FIM DA NOVA SEÇÃO ---

    clean_panel = pd.concat(all_data_frames, axis=1, join="inner").sort_index().dropna(how="any")
    clean_panel = clean_panel[~clean_panel.index.duplicated(keep="last")]
    
    clean_panel.to_csv(OUT / "clean_prices.csv")
    print(f"\n[OK] data/clean_prices.csv salvo (Shape: {clean_panel.shape})")

    price_cols = [s.name for s in price_series]
    logret = np.log(clean_panel[price_cols]).diff().dropna()
    logret.to_csv(OUT / "log_returns.csv")
    print(f"[OK] data/log_returns.csv salvo (Shape: {logret.shape})")

    try:
        reg = btc_regime_proxies(
            prices=clean_panel,
            btc_col=args.btc_col,
            last_halving=args.last_halving,
            k_sth=args.sth_k
        )
        reg.to_csv(OUT / "btc_regimes.csv")
        print(f"[OK] data/btc_regimes.csv salvo (Shape: {reg.shape})")
    except KeyError as e:
        print(f"[ETL] Aviso: Não foi possível gerar regimes BTC. Erro: {e}")
    except Exception as e:
        print(f"[ETL] Erro inesperado ao gerar regimes BTC: {e}")

    print("\nStep 01 (ETL Mestre) concluído.")

if __name__ == "__main__":
    main()