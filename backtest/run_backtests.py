"""
Backtest Runner - Test multiple strategies on multiple stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
from strategies import *

# Test different strategies on different stocks
test_configs = [
    ('AAPL', '1y', ['sma_crossover', 'rsi', 'macd', 'momentum', 'bollinger', 'combo']),
    ('NVDA', '1y', ['sma_crossover', 'rsi', 'macd', 'momentum', 'combo']),
    ('MSFT', '1y', ['sma_crossover', 'rsi', 'macd', 'momentum', 'combo']),
    ('TSLA', '1y', ['sma_crossover', 'rsi', 'macd', 'momentum', 'combo']),
    ('GOOGL', '1y', ['sma_crossover', 'rsi', 'macd', 'combo']),
]

results = []

print('Running backtests...')
print('='*70)

for ticker, period, strategies in test_configs:
    print(f'\n{ticker} ({period})')
    print('-'*50)
    
    # Fetch data
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    for strat_name in strategies:
        strat_func = registry.get(strat_name)
        if strat_func:
            df_strat = strat_func(df)
            result = run_backtest(df_strat)
            
            results.append({
                'Ticker': ticker,
                'Strategy': strat_name,
                'Return %': round(result['total_return_pct'], 2),
                'Annual %': round(result['annual_return_pct'], 2),
                'Max DD %': round(result['max_drawdown_pct'], 2),
                'Trades': result['num_trades']
            })
            
            print(f'  {strat_name:15} | Return: {result["total_return_pct"]:>7.2f}% | DD: {abs(result["max_drawdown_pct"]):>6.2f}% | Trades: {result["num_trades"]}')

print('\n' + '='*70)
print('SUMMARY TABLE (Sorted by Return)')
print('='*70)

# Create summary DataFrame
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Return %', ascending=False)

print(df_results.to_string(index=False))

# Best per ticker
print('\n' + '='*70)
print('BEST STRATEGY PER TICKER')
print('='*70)
for ticker, _, _ in test_configs:
    ticker_results = df_results[df_results['Ticker'] == ticker]
    if len(ticker_results) > 0:
        best = ticker_results.iloc[0]
        print(f'{ticker}: {best["Strategy"]} ({best["Return %"]}%)')

# Save to CSV
df_results.to_csv('backtest_results.csv', index=False)
print('\nResults saved to backtest_results.csv')