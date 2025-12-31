# Systematic Research Framework: Alpha Hypothesis Validation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Research-Active-success)]()

> **Scientific Invalidation of Trading Strategies:** A quantitative pipeline designed to differentiate skill from luck using Monte Carlo Permutation Tests (MCPT) and Kalman Filtering.

## 📉 Project Overview
This repository implements a rigorous backtesting and validation framework applied to a portfolio of **Bitcoin (BTC)**, **Gold (XAU)**, and **Natural Gas (NG)** between 2018 and 2025.

Unlike traditional backtesters that optimize for PnL, this engine focuses on **statistical robustness**. It tests fundamental economic hypotheses (Mean Reversion, Momentum, Risk-Off) and rigorously challenges them against null hypotheses of randomness.

## 🧠 Key Features & Quantitative Methods

### 1. Advanced Signal Engineering
* **Dynamic Cointegration (Kalman Filter):** Instead of static spread trading, we use a state-space model to estimate the dynamic hedge ratio ($\beta_t$) between assets, adapting to structural breaks.
    $$\beta_t = \beta_{t-1} + \omega_t, \quad \omega_t \sim \mathcal{N}(0, Q)$$
* **PCA Alpha Isolation:** Uses Principal Component Analysis to decompose returns and filter out "Beta Risk" (Global Liquidity Factor), isolating idiosyncratic alpha residuals.
* **GARCH Volatility Modeling:** Implementation of heteroscedastic volatility models for dynamic risk targeting.

### 2. Statistical Validation Engine
* **Monte Carlo Permutation Tests (MCPT):** The core of the framework. It generates thousands of synthetic market paths to calculate the **P-Value** of a strategy's Sharpe Ratio, detecting overfitting.
* **Regime Switching:** Logic to detect macro-economic regimes using High-Yield Credit Spreads and VIX.

## 🧪 Hypotheses & Results

We tested three mechanical premises. The framework revealed that despite positive "Real Sharpe Ratios", most strategies failed the statistical significance test ($p > 0.05$).

| Hypothesis | Strategy Class | Method | Real Sharpe | P-Value (MCP) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Reversion** | C1. Dynamic Spread | Kalman Filter (BTC/XAU) | **0.571** | **0.9735** | ❌ **Pure Luck** |
| **Momentum** | C2. Trend | Volatility Targeting | **0.776** | **0.7270** | ❌ **Pure Luck** |
| **Flight-to-Quality** | C3. Macro Hedge | Credit Spreads Filter | **-0.264** | **0.2905** | ❌ **Rejected** |

*See the [Full PDF Report](./Quant_First_(2).pdf) for detailed analysis of why these strategies failed despite looking profitable.*

## 📂 Repository Structure

```bash
systematic-research-framework/
├── mcp/                 # Monte Carlo Permutation engine (The "Judge")
│   └── tester.py        # P-Value calculation logic
├── strats/              # Strategy Implementations
│   ├── c1_spread_kalman.py   # State-space model implementation
│   ├── c1_macro_regime.py    # Macro-economic filter logic
│   └── garch.py              # Volatility modeling
├── metrics/             # Performance analytics (Sharpe, Drawdown, Sortino)
└── data/                # ETL scripts for Single Source of Truth (SSOT)
