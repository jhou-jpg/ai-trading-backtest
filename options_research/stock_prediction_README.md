# Stock Price Prediction Model

## What This Does

- Fetches **real data** from Yahoo Finance (yfinance)
- Predicts whether stock will go **UP or DOWN** tomorrow
- Uses 13 technical indicators as features
- Can be **A/B tested** against your existing strategies

## Quick Start

1. Upload `stock_prediction.py` to Kaggle
2. Enable GPU (Settings → Accelerator → GPU)
3. Run all cells

## Features Used

| Category | Features |
|----------|----------|
| Returns | 1d, 5d, 10d, 20d |
| Moving Averages | Price/SMA ratios |
| Momentum | RSI, MACD |
| Volume | Volume ratio |
| Volatility | Bollinger position, 20d vol |

## Model Output

- **BUY** if probability > 55%
- **SELL** if probability < 45%
- **HOLD** otherwise

## A/B Testing

Compare against your existing strategies:
- SMA Crossover
- RSI
- MACD
- Momentum

Run backtest for each and compare returns + Sharpe ratio!

## Files

- `stock_prediction.py` - Main model
- `stock_prediction_README.md` - This file