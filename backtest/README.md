# AI Trading System - Backtesting Framework

A Python-based backtesting framework for trading strategies. Built as part of an AI trading system project.

## Features

- **Data Fetching**: Pull historical stock data from Yahoo Finance
- **Signal Generators**: Multiple built-in strategies:
  - SMA Crossover (moving average cross)
  - RSI (oversold/overbought)
  - Momentum (trend following)
- **Backtest Engine**: Calculate returns, drawdowns, trade counts
- **Strategy Blending**: Mix multiple strategies with weighted signals
- **Multi-Ticker Testing**: Test across multiple stocks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from base_framework import fetch_stock_data, sma_crossover, run_backtest

# Fetch data
tickers = ["AAPL", "MSFT", "NVDA"]
data = fetch_stock_data(tickers, period="1y")

# Run strategy
for ticker, df in data.items():
    df_sma = sma_crossover(df, short_window=20, long_window=50)
    result = run_backtest(df_sma)
    print(f"{ticker}: {result['total_return_pct']:.2f}%")
```

## Project Structure

```
backtest/
├── base_framework.py    # Core backtesting engine
├── requirements.txt     # Python dependencies
└── .gitignore          # Git ignore rules
```

## Status

🚧 **In Development** - Initial framework complete, strategy testing in progress