"""
Signal Generator Module
Takes a ticker + strategy → returns a trade signal object
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Callable
from strategies import registry


def fetch_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    # Handle multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df


def get_latest_indicators(df: pd.DataFrame) -> Dict:
    """Calculate and return latest technical indicators."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # SMA
    sma_20 = close.rolling(20).mean().iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1]
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    
    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema_12 - ema_26).iloc[-1]
    signal_line = macd.ewm(span=9, adjust=False).mean().iloc[-1]
    macd_hist = (macd - signal_line).iloc[-1]
    
    # Bollinger Bands
    sma_20_all = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    bb_upper = (sma_20_all + 2 * std_20).iloc[-1]
    bb_lower = (sma_20_all - 2 * std_20).iloc[-1]
    
    # ATR (Average True Range)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    # Volume
    avg_volume = volume.rolling(20).mean().iloc[-1]
    current_volume = volume.iloc[-1]
    
    # Price momentum
    momentum_10 = close.pct_change(10).iloc[-1]
    momentum_20 = close.pct_change(20).iloc[-1]
    
    return {
        'price': close.iloc[-1],
        'sma_20': sma_20,
        'sma_50': sma_50,
        'sma_200': sma_200,
        'rsi': rsi,
        'macd': macd,
        'macd_signal': signal_line,
        'macd_hist': macd_hist,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'atr': atr,
        'volume': current_volume,
        'avg_volume': avg_volume,
        'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
        'momentum_10d': momentum_10,
        'momentum_20d': momentum_20,
    }


def determine_trend(indicators: Dict) -> str:
    """Determine overall market trend."""
    price = indicators['price']
    sma_50 = indicators['sma_50']
    sma_200 = indicators['sma_200']
    
    if price > sma_50 > sma_200:
        return "strong_bull"
    elif price > sma_50:
        return "bull"
    elif price < sma_50 < sma_200:
        return "strong_bear"
    elif price < sma_50:
        return "bear"
    return "neutral"


def calculate_confidence(indicators: Dict, signal_type: str) -> float:
    """
    Calculate confidence score (0-1) for a signal.
    Higher confidence = stronger signal.
    """
    if signal_type == "HOLD":
        return 0.5
    
    confidence = 0.5
    
    # RSI contribution
    rsi = indicators['rsi']
    if signal_type == "BUY" and rsi < 30:
        confidence += 0.2
    elif signal_type == "BUY" and rsi < 40:
        confidence += 0.1
    elif signal_type == "SELL" and rsi > 70:
        confidence += 0.2
    elif signal_type == "SELL" and rsi > 60:
        confidence += 0.1
    
    # MACD contribution
    macd_hist = indicators['macd_hist']
    if signal_type == "BUY" and macd_hist > 0:
        confidence += 0.15
    elif signal_type == "SELL" and macd_hist < 0:
        confidence += 0.15
    
    # Trend contribution
    trend = determine_trend(indicators)
    if signal_type == "BUY" and "bull" in trend:
        confidence += 0.15
    elif signal_type == "SELL" and "bear" in trend:
        confidence += 0.15
    
    return min(1.0, max(0.0, confidence))


def generate_signal(ticker: str, strategy: str = "combo",
                   period: str = "1y", interval: str = "1d") -> Dict:
    """
    Generate a trade signal for a given ticker and strategy.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        strategy: Strategy name from registry
        period: Data period to fetch
        interval: Data interval
    
    Returns:
        Signal dictionary with all details
    """
    # Fetch data
    df = fetch_data(ticker, period, interval)
    
    if len(df) == 0:
        return {
            'ticker': ticker,
            'strategy': strategy,
            'signal_type': 'ERROR',
            'error': 'No data fetched'
        }
    
    # Get latest indicators
    indicators = get_latest_indicators(df)
    
    # Apply strategy
    strategy_func = registry.get(strategy)
    if strategy_func is None:
        return {
            'ticker': ticker,
            'strategy': strategy,
            'signal_type': 'ERROR',
            'error': f'Unknown strategy: {strategy}'
        }
    
    df_strategy = strategy_func(df)
    signal_value = df_strategy['Signal'].iloc[-1]
    
    # Determine signal type
    if signal_value > 0:
        signal_type = "BUY"
    elif signal_value < 0:
        signal_type = "SELL"
    else:
        signal_type = "HOLD"
    
    # Generate entry reason
    entry_reason = generate_entry_reason(signal_type, indicators, strategy)
    
    # Calculate confidence
    confidence = calculate_confidence(indicators, signal_type)
    
    # Determine trend
    trend = determine_trend(indicators)
    
    return {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat(),
        'strategy': strategy,
        'signal_type': signal_type,
        
        # Entry details
        'entry_price': indicators['price'],
        'entry_reason': entry_reason,
        
        # All indicators (for reference)
        'indicators': indicators,
        'trend': trend,
        'confidence': confidence,
        
        # Data for position sizing
        'atr': indicators['atr'],
        'current_price': indicators['price'],
    }


def generate_entry_reason(signal_type: str, indicators: Dict, strategy: str) -> str:
    """Generate human-readable reason for the signal."""
    reasons = []
    
    if signal_type == "BUY":
        rsi = indicators['rsi']
        if rsi < 30:
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 40:
            reasons.append(f"RSI somewhat oversold ({rsi:.1f})")
        
        if indicators['macd_hist'] > 0:
            reasons.append("MACD momentum bullish")
        
        trend = determine_trend(indicators)
        if "bull" in trend:
            reasons.append(f"Trend: {trend}")
            
    elif signal_type == "SELL":
        rsi = indicators['rsi']
        if rsi > 70:
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 60:
            reasons.append(f"RSI somewhat overbought ({rsi:.1f})")
        
        if indicators['macd_hist'] < 0:
            reasons.append("MACD momentum bearish")
        
        trend = determine_trend(indicators)
        if "bear" in trend:
            reasons.append(f"Trend: {trend}")
    
    if not reasons:
        reasons.append(f"Strategy: {strategy}")
    
    return "; ".join(reasons)


def scan_watchlist(tickers: List[str], strategy: str = "combo") -> List[Dict]:
    """
    Scan a watchlist and generate signals for all tickers.
    
    Args:
        tickers: List of stock symbols
        strategy: Strategy to use
    
    Returns:
        List of signal dictionaries
    """
    signals = []
    
    for ticker in tickers:
        try:
            signal = generate_signal(ticker, strategy)
            signals.append(signal)
        except Exception as e:
            signals.append({
                'ticker': ticker,
                'signal_type': 'ERROR',
                'error': str(e)
            })
    
    return signals


def filter_signals(signals: List[Dict], signal_type: str = None, 
                   min_confidence: float = 0.0) -> List[Dict]:
    """
    Filter signals by type and confidence.
    
    Args:
        signals: List of signal dictionaries
        signal_type: Filter by 'BUY', 'SELL', or None for all
        min_confidence: Minimum confidence score (0-1)
    
    Returns:
        Filtered list of signals
    """
    filtered = []
    
    for sig in signals:
        if sig.get('signal_type') == 'ERROR':
            continue
        
        if signal_type and sig.get('signal_type') != signal_type:
            continue
        
        if sig.get('confidence', 0) < min_confidence:
            continue
        
        filtered.append(sig)
    
    return filtered


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Test with single ticker
    print("="*60)
    print("SIGNAL GENERATOR TEST")
    print("="*60)
    
    signal = generate_signal("AAPL", strategy="combo")
    
    print(f"\nTicker: {signal['ticker']}")
    print(f"Strategy: {signal['strategy']}")
    print(f"Signal: {signal['signal_type']}")
    print(f"Entry Price: ${signal['entry_price']:.2f}")
    print(f"Entry Reason: {signal['entry_reason']}")
    print(f"Confidence: {signal['confidence']:.0%}")
    print(f"Trend: {signal['trend']}")
    
    print("\n--- Key Indicators ---")
    ind = signal['indicators']
    print(f"Price: ${ind['price']:.2f}")
    print(f"RSI: {ind['rsi']:.1f}")
    print(f"MACD: {ind['macd']:.2f}")
    print(f"ATR: ${ind['atr']:.2f}")
    print(f"Volume: {ind['volume']:,.0f} ({(ind['volume_ratio']*100):.0f}% of avg)")
    
    # Test watchlist scan
    print("\n" + "="*60)
    print("WATCHLIST SCAN")
    print("="*60)
    
    watchlist = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA"]
    signals = scan_watchlist(watchlist, strategy="combo")
    
    # Filter for BUY signals
    buy_signals = filter_signals(signals, signal_type="BUY", min_confidence=0.5)
    
    print(f"\nFound {len(buy_signals)} BUY signals:")
    for sig in buy_signals:
        print(f"  {sig['ticker']}: {sig['entry_reason']} (confidence: {sig['confidence']:.0%})")