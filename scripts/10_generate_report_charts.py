import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
import glob

results = {
    "C1_Static (BTC-NG)": {
        "sharpe": -0.273,
        "pvalue": 0.3995,
        "file_glob": "C*MacroRegime__BTC_Gas Natural Futures_spread_static*.csv"
    },
    "C1_Static (Triplet)": {
        "sharpe": -0.134,
        "pvalue": 0.6795,
        "file_glob": "C*MacroRegime__BTC_XAU-USD_Gas Natural Futures_spread_static*.csv"
    },
    
    # --- CLASSE C1: Cointegração Dinâmica ---
    "C1_Kalman (BTC-XAU)": {
        "sharpe": 0.571,
        "pvalue": 0.9735,
        "file_glob": "C3SpreadKalman__BTC_XAU-USD_spread_static*.csv"
    },
    
    # --- CLASSE C1.v2: Hedge Real (Neutro) ---
    "C1.v2_Hedge (BTC-NG)": {
        "sharpe": 0.338,
        "pvalue": 0.9340,
        "file_glob": "C1HedgedPortfolio__BTC_Gas Natural Futures*.csv"
    },
    "C1.v2_Hedge (BTC-XAU)": {
        "sharpe": -0.490,
        "pvalue": 0.3220,
        "file_glob": "C1HedgedPortfolio__BTC_XAU*.csv"
    },

    # --- CLASSE C2: Momentum & Macro ---
    "C2_Trend (Benchmark)": {
        "sharpe": 0.776,
        "pvalue": 0.7270,
        "file_glob": "C3BtcTrendVol__no_spread*.csv"
    },
    "C2_VIX (Risk-Off)": {
        "sharpe": 0.521,
        "pvalue": 0.3260,
        "file_glob": "C2FlightToQuality__no_spread*.csv"
    },
    "C2.v2_MacroFilter": {
        "sharpe": 0.716,
        "pvalue": 0.9995,
        "file_glob": "C3MacroFilteredTrend*.csv"
    },

    # --- CLASSE C3: Crédito ---
    "C3_CreditStress": {
        "sharpe": -0.264,
        "pvalue": 0.2905,
        "file_glob": "C3StressHedge*.csv"
    }
}

DATA_DIR = Path("data/mcp")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

def plot_verdict_chart(results_dict):
    """
    Gera um gráfico de barras comparando Sharpe Real vs. P-Valor
    para todas as estratégias.
    """
    labels = list(results_dict.keys())
    sharpes = [v['sharpe'] for v in results_dict.values()]
    pvalues = [v['pvalue'] for v in results_dict.values()]
    
    x = range(len(labels))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(16, 8)) 
    color1 = '#1f77b4' 
    ax1.set_xlabel('Estratégia', fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio Real (In-Sample)', color=color1, fontweight='bold')
    bar1 = ax1.bar([i - width/2 for i in x], sharpes, width, label='Sharpe Real', color=color1, alpha=0.9)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--') 
    ax2 = ax1.twinx()
    color2 = '#d62728' 
    ax2.set_ylabel('P-Valor (Probabilidade de Sorte)', color=color2, fontweight='bold')
    bar2 = ax2.bar([i + width/2 for i in x], pvalues, width, label='P-Valor MCP', color=color2, alpha=0.9)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 
    ax2.axhline(0.10, color='black', linewidth=1, linestyle=':', 
                label='Nível de Significância (10%)') 
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    
    fig.suptitle('Veredito Final da Pipeline (Sharpe vs. P-Valor)', fontsize=18, y=0.98)

    lines, labels_leg = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels_leg + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    
    fig.tight_layout()
    
    save_path = OUTPUT_DIR / "01_veredicto_sharpe_vs_pvalue.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[OK] Gráfico de Veredito salvo em: {save_path}")

def plot_cumulative_returns(results_dict):
    """
    Gera um gráfico de linha com os retornos log acumulados.
    """
    plt.figure(figsize=(16, 8))
    
    colors = plt.cm.tab10.colors 
    
    for i, (strat_name, data) in enumerate(results_dict.items()):
        try:
            files = list(DATA_DIR.glob(data['file_glob']))
            if not files:
                print(f"AVISO: Arquivo não encontrado para {strat_name} (Pattern: {data['file_glob']})")
                continue
                
            file_path = files[0] 
            
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if 'ret' not in df.columns:
                 cols = df.select_dtypes(include=['float64']).columns
                 if len(cols) > 0:
                     df['ret'] = df[cols[0]]
            
            df['ret'] = pd.to_numeric(df['ret'], errors='coerce').fillna(0)
            cum_log_ret = df['ret'].cumsum()
            
            linestyle = '-' if data['sharpe'] > 0 else ':'
            linewidth = 2 if data['sharpe'] > 0 else 1.5
            
            plt.plot(cum_log_ret.index, cum_log_ret.values, 
                     label=f"{strat_name} (S: {data['sharpe']:.2f}, p: {data['pvalue']:.2f})",
                     linestyle=linestyle, linewidth=linewidth, color=colors[i % len(colors)])
            
        except Exception as e:
            print(f"Erro ao processar {strat_name}: {e}")

    plt.title('Desempenho Acumulado das Estratégias (2018-2025)', fontsize=18)
    plt.ylabel('Retorno Log Acumulado', fontsize=12)
    plt.xlabel('Data', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "02_retorno_acumulado.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[OK] Gráfico de Retornos salvo em: {save_path}")

def main():
    print("--- INICIANDO GERAÇÃO DE GRÁFICOS ---")
    plot_verdict_chart(results)
    plot_cumulative_returns(results)
    print("--- CONCLUÍDO ---")

if __name__ == "__main__":
    main()