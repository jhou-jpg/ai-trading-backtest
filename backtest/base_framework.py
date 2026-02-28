"""
AI Trading System - Base Backtesting Framework
Step 1: Core infrastructure for fetching data and running backtests
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Callable
import json

# ============================================================
# DATA FETCHING
# ============================================================

def fetch_stock_data(tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
    """Fetch historical data for multiple tickers."""
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if len(df) > 0:
                data[ticker] = df
                print(f"[+] Fetched {ticker}: {len(df)} days")
            else:
                print(f"[-] No data for {ticker}")
        except Exception as e:
            print(f"[-] Error fetching {ticker}: {e}")
    return data


def fetch_stock_data_range(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for a specific date range."""
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if len(df) > 0:
                data[ticker] = df
        except Exception as e:
            print(f"[-] Error fetching {ticker}: {e}")
    return data


# ============================================================
# SIGNAL GENERATORS
# ============================================================

def sma_crossover(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """Simple Moving Average crossover strategy."""
    result = df.copy()
    result['SMA_Short'] = result['Close'].rolling(window=short_window).mean()
    result['SMA_Long'] = result['Close'].rolling(window=long_window).mean()
    
    # Signal: 1 when short crosses above long, -1 when short crosses below long
    result['Signal'] = 0
    result.loc[result['SMA_Short'] > result['SMA_Long'], 'Signal'] = 1
    result.loc[result['SMA_Short'] < result['SMA_Long'], 'Signal'] = -1
    
    # Position changes
    result['Position'] = result['Signal'].diff()
    return result


def rsi_strategy(df: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
    """RSI-based strategy: buy when oversold, sell when overbought."""
    result = df.copy()
    
    # Calculate RSI
    delta = result['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    result['RSI'] = 100 - (100 / (1 + rs))
    
    # Signal
    result['Signal'] = 0
    result.loc[result['RSI'] < oversold, 'Signal'] = 1  # Buy
    result.loc[result['RSI'] > overbought, 'Signal'] = -1  # Sell
    
    result['Position'] = result['Signal'].diff()
    return result


def momentum(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Momentum strategy: buy when price is trending up."""
    result = df.copy()
    result['Return'] = result['Close'].pct_change(lookback)
    result['Signal'] = 0
    result.loc[result['Return'] > 0, 'Signal'] = 1
    result.loc[result['Return'] < 0, 'Signal'] = -1
    result['Position'] = result['Signal'].diff()
    return result


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(df: pd.DataFrame, initial_capital: float = 10000, 
                 position_size: float = 1.0, commission: float = 0.001) -> Dict:
    """
    Run a backtest on a dataframe with signals.
    
    Args:
        df: DataFrame with 'Close' and 'Position' columns
        initial_capital: Starting capital
        position_size: Fraction of capital to use per trade (0-1)
        commission: Commission rate per trade
    
    Returns:
        Dictionary with performance metrics
    """
    # Use closing prices and position signals
    # Handle multi-level columns from yfinance
    close_col = df['Close']
    if isinstance(close_col, pd.DataFrame):
        # Multi-level column - take first level
        close_col = close_col.iloc[:, 0]
    
    prices = close_col.copy()
    positions = df['Position'].copy()
    
    # Initialize portfolio
    capital = initial_capital
    shares = 0
    position = 0  # 0 = no position, 1 = long
    
    trades = []
    portfolio_values = []
    
    for i in range(len(df)):
        current_price = float(prices.iloc[i])  # Convert to scalar
        signal = positions.iloc[i] if i > 0 else 0
        
        # Calculate current portfolio value
        portfolio_value = capital + shares * current_price
        portfolio_values.append(portfolio_value)
        
        # Execute trades on signal changes
        if signal == 2:  # Buy signal (position went from 0 to 1)
            # Buy shares
            shares_to_buy = int((capital * position_size) / current_price)
            cost = shares_to_buy * current_price * (1 + commission)
            if cost <= capital:
                shares += shares_to_buy
                capital -= cost
                trades.append({'type': 'BUY', 'price': current_price, 'shares': shares_to_buy})
                position = 1
                
        elif signal == -2 and position == 1:  # Sell signal
            # Sell all shares
            proceeds = shares * current_price * (1 - commission)
            trades.append({'type': 'SELL', 'price': current_price, 'shares': shares})
            capital += proceeds
            shares = 0
            position = 0
    
    # Final portfolio value
    final_value = capital + shares * prices.iloc[-1]
    
    # Calculate metrics
    total_return = (final_value - initial_capital) / initial_capital * 100
    days = len(df)
    annual_return = total_return * (365 / days) if days > 0 else 0
    
    # Calculate max drawdown
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    
    # Count trades
    num_trades = len(trades)
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'annual_return_pct': annual_return,
        'max_drawdown_pct': max_drawdown,
        'num_trades': num_trades,
        'trades': trades,
        'portfolio_values': portfolio_values
    }


def run_multi_ticker_backtest(data: Dict[str, pd.DataFrame], strategy_func: Callable,
                               initial_capital: float = 10000, weights: Dict[str, float] = None) -> Dict:
    """
    Run backtest across multiple tickers with optional weighting.
    
    Args:
        data: Dict of ticker -> DataFrame
        strategy_func: Function to generate signals
        initial_capital: Starting capital
        weights: Dict of ticker -> weight (must sum to 1)
    
    Returns:
        Dictionary with per-ticker and blended results
    """
    results = {}
    
    for ticker, df in data.items():
        # Apply strategy
        df_strategy = strategy_func(df)
        
        # Run backtest
        result = run_backtest(df_strategy, initial_capital)
        results[ticker] = result
    
    # Calculate blended portfolio
    if weights is None:
        weights = {ticker: 1/len(results) for ticker in results}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Blend returns (simple weighted average)
    blended_return = sum(results[t]['total_return_pct'] * weights.get(t, 0) for t in results)
    
    return {
        'individual': results,
        'blended_return_pct': blended_return,
        'weights': weights
    }


# ============================================================
# STRATEGY BLENDING
# ============================================================

def blend_strategies(df: pd.DataFrame, strategies: List[Tuple[Callable, float]]) -> pd.DataFrame:
    """
    Blend multiple strategies with weights.
    
    Args:
        df: Price data
        strategies: List of (strategy_function, weight) tuples
    
    Returns:
        DataFrame with blended signal
    """
    result = df.copy()
    blended_signal = pd.Series(0, index=df.index)
    
    total_weight = sum(w for _, w in strategies)
    
    for strategy_func, weight in strategies:
        df_strat = strategy_func(df)
        if 'Signal' in df_strat:
            blended_signal += df_strat['Signal'] * (weight / total_weight)
    
    # Normalize to -1 to 1
    result['Blended_Signal'] = blended_signal.clip(-1, 1)
    result['Position'] = result['Blended_Signal'].diff()
    
    return result


# ============================================================
# VISUALIZATION
# ============================================================

def print_backtest_results(results: Dict, ticker: str = ""):
    """Print backtest results in a readable format."""
    prefix = f"{ticker}: " if ticker else ""
    
    print(f"\n{'='*50}")
    print(f"{prefix}BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"Initial Capital:    ${results['initial_capital']:,.2f}")
    print(f"Final Value:        ${results['final_value']:,.2f}")
    print(f"Total Return:       {results['total_return_pct']:.2f}%")
    print(f"Annual Return:      {results['annual_return_pct']:.2f}%")
    print(f"Max Drawdown:       {results['max_drawdown_pct']:.2f}%")
    print(f"Number of Trades:   {results['num_trades']}")
    print(f"{'='*50}")


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Example: Test a single strategy
    tickers = ["AAPL", "MSFT", "NVDA"]
    
    print("Fetching data...")
    data = fetch_stock_data(tickers, period="1y")
    
    if data:
        print("\nRunning SMA crossover backtest...")
        for ticker, df in data.items():
            df_sma = sma_crossover(df, short_window=20, long_window=50)
            result = run_backtest(df_sma)
            print_backtest_results(result, ticker)