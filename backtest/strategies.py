"""
Multi-Strategy Trading Framework
A modular system with multiple strategies that can be blended and routed
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from functools import wraps

# ============================================================
# STRATEGY REGISTRY
# ============================================================

class StrategyRegistry:
    """Registry for all available trading strategies."""
    
    def __init__(self):
        self.strategies = {}
    
    def register(self, name: str):
        """Decorator to register a strategy."""
        def decorator(func: Callable):
            self.strategies[name] = func
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get(self, name: str) -> Callable:
        """Get a strategy by name."""
        return self.strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all available strategies."""
        return list(self.strategies.keys())

# Global registry
registry = StrategyRegistry()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return data.rolling(window=window).mean()


def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return data.ewm(span=window, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD (Moving Average Convergence Divergence)."""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })


def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    """Bollinger Bands."""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    return pd.DataFrame({
        'upper': sma + (std * num_std),
        'middle': sma,
        'lower': sma - (std * num_std)
    })


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({'k': k, 'd': d})


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = calculate_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx


# ============================================================
# STRATEGIES
# ============================================================

@registry.register("sma_crossover")
def sma_crossover_strategy(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """
    SMA Crossover Strategy
    Buy when short SMA crosses above long SMA, sell when it crosses below.
    """
    result = df.copy()
    result['SMA_Short'] = calculate_sma(result['Close'], short_window)
    result['SMA_Long'] = calculate_sma(result['Close'], long_window)
    
    # Signal: 1 = bullish crossover, -1 = bearish crossover
    result['Signal'] = 0
    result.loc[result['SMA_Short'] > result['SMA_Long'], 'Signal'] = 1
    result.loc[result['SMA_Short'] < result['SMA_Long'], 'Signal'] = -1
    
    # Position changes (2 = buy, -2 = sell)
    result['Position'] = result['Signal'].diff()
    return result


@registry.register("ema_crossover")
def ema_crossover_strategy(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
    """
    EMA Crossover Strategy
    Buy when fast EMA crosses above slow EMA.
    """
    result = df.copy()
    result['EMA_Fast'] = calculate_ema(result['Close'], fast)
    result['EMA_Slow'] = calculate_ema(result['Close'], slow)
    
    result['Signal'] = 0
    result.loc[result['EMA_Fast'] > result['EMA_Slow'], 'Signal'] = 1
    result.loc[result['EMA_Fast'] < result['EMA_Slow'], 'Signal'] = -1
    
    result['Position'] = result['Signal'].diff()
    return result


@registry.register("rsi")
def rsi_strategy(df: pd.DataFrame, period: int = 14, oversold: int = 30, 
                overbought: int = 70) -> pd.DataFrame:
    """
    RSI Strategy
    Buy when RSI < oversold (oversold), sell when RSI > overbought.
    """
    result = df.copy()
    result['RSI'] = calculate_rsi(result['Close'], period)
    
    result['Signal'] = 0
    result.loc[result['RSI'] < oversold, 'Signal'] = 1   # Buy oversold
    result.loc[result['RSI'] > overbought, 'Signal'] = -1  # Sell overbought
    
    result['Position'] = result['Signal'].diff()
    return result


@registry.register("macd")
def macd_strategy(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD Strategy
    Buy when MACD crosses above signal line, sell when it crosses below.
    """
    result = df.copy()
    macd_data = calculate_macd(result['Close'], fast, slow, signal)
    result['MACD'] = macd_data['macd']
    result['MACD_Signal'] = macd_data['signal']
    result['MACD_Hist'] = macd_data['histogram']
    
    result['Signal'] = 0
    result.loc[result['MACD'] > result['MACD_Signal'], 'Signal'] = 1
    result.loc[result['MACD'] < result['MACD_Signal'], 'Signal'] = -1
    
    result['Position'] = result['Signal'].diff()
    return result


@registry.register("bollinger")
def bollinger_strategy(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    """
    Bollinger Bands Strategy
    Buy when price touches lower band, sell when it touches upper band.
    """
    result = df.copy()
    bb = calculate_bollinger_bands(result['Close'], window, num_std)
    result['BB_Upper'] = bb['upper']
    result['BB_Middle'] = bb['middle']
    result['BB_Lower'] = bb['lower']
    
    # Signal: price below lower band = oversold (buy), above upper = overbought (sell)
    result['Signal'] = 0
    result.loc[result['Close'] < result['BB_Lower'], 'Signal'] = 1
    result.loc[result['Close'] > result['BB_Upper'], 'Signal'] = -1
    
    result['Position'] = result['Signal'].diff()
    return result


@registry.register("momentum")
def momentum_strategy(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Momentum Strategy
    Buy when price is trending up (positive momentum), sell when trending down.
    """
    result = df.copy()
    result['Momentum'] = result['Close'].pct_change(lookback)
    
    result['Signal'] = 0
    result.loc[result['Momentum'] > 0, 'Signal'] = 1
    result.loc[result['Momentum'] < 0, 'Signal'] = -1
    
    result['Position'] = result['Signal'].diff()
    return result


@registry.register("mean_reversion")
def mean_reversion_strategy(df: pd.DataFrame, window: int = 20, std_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Mean Reversion Strategy
    Buy when price deviates negatively from moving average, sell on positive deviation.
    """
    result = df.copy()
    result['MA'] = calculate_sma(result['Close'], window)
    result['Std'] = result['Close'].rolling(window=window).std()
    result['Upper_Band'] = result['MA'] + (result['Std'] * std_multiplier)
    result['Lower_Band'] = result['MA'] - (result['Std'] * std_multiplier)
    
    # Z-score
    result['Z_Score'] = (result['Close'] - result['MA']) / result['Std']
    
    result['Signal'] = 0
    result.loc[result['Z_Score'] < -std_multiplier, 'Signal'] = 1   # Buy oversold
    result.loc[result['Z_Score'] > std_multiplier, 'Signal'] = -1   # Sell overbought
    
    result['Position'] = result['Signal'].diff()
    return result


@registry.register("stochastic")
def stochastic_strategy(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
                       oversold: int = 20, overbought: int = 80) -> pd.DataFrame:
    """
    Stochastic Oscillator Strategy
    Buy when %K crosses above %D in oversold territory, sell when it crosses below in overbought.
    """
    result = df.copy()
    stoch = calculate_stochastic(result['High'], result['Low'], result['Close'], k_period, d_period)
    result['Stoch_K'] = stoch['k']
    result['Stoch_D'] = stoch['d']
    
    # Buy when K crosses above D in oversold zone
    # Sell when K crosses below D in overbought zone
    result['Signal'] = 0
    
    # Simple version: just use K level
    result.loc[result['Stoch_K'] < oversold, 'Signal'] = 1
    result.loc[result['Stoch_K'] > overbought, 'Signal'] = -1
    
    result['Position'] = result['Signal'].diff()
    return result


@registry.register("breakout")
def breakout_strategy(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Breakout Strategy
    Buy when price breaks above recent high, sell when it breaks below recent low.
    """
    result = df.copy()
    result['Highest'] = result['High'].rolling(window=lookback).max()
    result['Lowest'] = result['Low'].rolling(window=lookback).min()
    
    result['Signal'] = 0
    result.loc[result['Close'] > result['Highest'].shift(1), 'Signal'] = 1   # Breakout high
    result.loc[result['Close'] < result['Lowest'].shift(1), 'Signal'] = -1   # Breakdown low
    
    result['Position'] = result['Signal'].diff()
    return result


@registry.register("adx_trend")
def adx_trend_strategy(df: pd.DataFrame, period: int = 14, threshold: int = 25) -> pd.DataFrame:
    """
    ADX Trend Strategy
    Buy when ADX > threshold (strong trend), use +DI/-DI for direction.
    """
    result = df.copy()
    result['ADX'] = calculate_adx(result['High'], result['Low'], result['Close'], period)
    
    plus_dm = result['High'].diff()
    minus_dm = -result['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = calculate_atr(result['High'], result['Low'], result['Close'], period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
    
    result['ADX'] = calculate_adx(result['High'], result['Low'], result['Close'], period)
    result['Plus_DI'] = plus_di
    result['Minus_DI'] = minus_di
    
    # Signal: ADX > threshold indicates strong trend
    result['Signal'] = 0
    result.loc[(result['ADX'] > threshold) & (result['Plus_DI'] > result['Minus_DI']), 'Signal'] = 1
    result.loc[(result['ADX'] > threshold) & (result['Plus_DI'] < result['Minus_DI']), 'Signal'] = -1
    
    result['Position'] = result['Signal'].diff()
    return result


@registry.register("combo_trend")
def combo_trend_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combo Trend Strategy
    Combine multiple indicators for a more robust signal.
    Uses SMA, RSI, and MACD together.
    """
    result = df.copy()
    
    # SMA trend
    result['SMA_50'] = calculate_sma(result['Close'], 50)
    result['SMA_200'] = calculate_sma(result['Close'], 200)
    result['SMA_Trend'] = (result['SMA_50'] > result['SMA_200']).astype(int) * 2 - 1
    
    # RSI
    result['RSI'] = calculate_rsi(result['Close'], 14)
    result['RSI_Signal'] = 0
    result.loc[result['RSI'] < 40, 'RSI_Signal'] = 1
    result.loc[result['RSI'] > 60, 'RSI_Signal'] = -1
    
    # MACD
    macd_data = calculate_macd(result['Close'])
    result['MACD_Signal'] = 0
    result.loc[macd_data['macd'] > macd_data['signal'], 'MACD_Signal'] = 1
    result.loc[macd_data['macd'] < macd_data['signal'], 'MACD_Signal'] = -1
    
    # Combine: average all signals
    result['Combined'] = (result['SMA_Trend'] + result['RSI_Signal'] + result['MACD_Signal']) / 3
    
    result['Signal'] = 0
    result.loc[result['Combined'] > 0.3, 'Signal'] = 1
    result.loc[result['Combined'] < -0.3, 'Signal'] = -1
    
    result['Position'] = result['Signal'].diff()
    return result


# ============================================================
# STRATEGY BLENDER
# ============================================================

class StrategyBlender:
    """Blend multiple strategies together."""
    
    def __init__(self, strategies: Dict[str, Tuple[Callable, float]]):
        """
        Args:
            strategies: Dict of {strategy_name: (strategy_function, weight)}
        """
        self.strategies = strategies
        
        # Normalize weights
        total = sum(w for _, w in strategies.values())
        self.weights = {k: v/total for k, v in strategies.items()}
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all strategies and blend signals."""
        result = df.copy()
        blended_signal = pd.Series(0, index=df.index)
        
        for name, (strategy_func, weight) in self.strategies.items():
            df_strat = strategy_func(df)
            if 'Signal' in df_strat:
                blended_signal += df_strat['Signal'] * self.weights[name]
        
        # Normalize to -1 to 1
        result['Blended_Signal'] = blended_signal.clip(-1, 1)
        result['Position'] = result['Blended_Signal'].diff()
        
        return result
    
    def get_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get individual strategy signals."""
        signals = {}
        for name, (strategy_func, _) in self.strategies.items():
            df_strat = strategy_func(df)
            signals[name] = df_strat['Signal']
        return signals


# ============================================================
# STRATEGY ROUTER
# ============================================================

class StrategyRouter:
    """
    Router that selects the best strategy based on market conditions.
    Can use different strategies for different market regimes.
    """
    
    def __init__(self):
        self.strategies = {}
        self.current_strategy = None
        self.market_regime = "neutral"  # bull, bear, neutral, volatile
    
    def detect_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime."""
        # Calculate trend
        sma_50 = calculate_sma(df['Close'], 50)
        sma_200 = calculate_sma(df['Close'], 200)
        
        # Calculate volatility
        returns = df['Close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1]
        avg_volatility = returns.std()
        
        if sma_50.iloc[-1] > sma_200.iloc[-1]:
            trend = "bull"
        elif sma_50.iloc[-1] < sma_200.iloc[-1]:
            trend = "bear"
        else:
            trend = "neutral"
        
        if volatility > avg_volatility * 1.5:
            volatility = "volatile"
        elif volatility < avg_volatility * 0.7:
            volatility = "calm"
        else:
            volatility = "normal"
        
        # Simple regime detection
        if trend == "bull" and volatility in ["calm", "normal"]:
            return "bull"
        elif trend == "bear" and volatility in ["calm", "normal"]:
            return "bear"
        elif volatility == "volatile":
            return "volatile"
        else:
            return "neutral"
    
    def add_strategy(self, regime: str, strategy: Callable, params: dict = None):
        """Add a strategy for a specific market regime."""
        self.strategies[regime] = (strategy, params or {})
    
    def select_strategy(self, df: pd.DataFrame) -> str:
        """Select the best strategy for current market conditions."""
        self.market_regime = self.detect_regime(df)
        
        # Try to find a matching regime strategy
        if self.market_regime in self.strategies:
            self.current_strategy = self.market_regime
        elif "neutral" in self.strategies:
            self.current_strategy = "neutral"
        else:
            self.current_strategy = list(self.strategies.keys())[0] if self.strategies else None
        
        return self.current_strategy
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the selected strategy."""
        regime = self.select_strategy(df)
        
        if regime and regime in self.strategies:
            strategy_func, params = self.strategies[regime]
            return strategy_func(df, **params)
        
        # Fallback to SMA crossover
        return sma_crossover_strategy(df)


# ============================================================
# BACKTESTER
# ============================================================

def run_backtest(df: pd.DataFrame, initial_capital: float = 10000,
                position_size: float = 1.0, commission: float = 0.001) -> Dict:
    """Run backtest on a strategy."""
    # Handle multi-level columns from yfinance
    close_col = df['Close']
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]
    prices = close_col.copy()
    
    positions = df['Position'].copy()
    
    capital = initial_capital
    shares = 0
    position = 0
    
    trades = []
    portfolio_values = []
    
    for i in range(len(df)):
        current_price = float(prices.iloc[i])
        signal = positions.iloc[i] if i > 0 else 0
        
        portfolio_value = capital + shares * current_price
        portfolio_values.append(portfolio_value)
        
        # Buy
        if signal == 2:
            shares_to_buy = int((capital * position_size) / current_price)
            cost = shares_to_buy * current_price * (1 + commission)
            if cost <= capital:
                shares += shares_to_buy
                capital -= cost
                trades.append({'type': 'BUY', 'price': current_price, 'shares': shares_to_buy})
                position = 1
                
        # Sell
        elif signal == -2 and position == 1:
            proceeds = shares * current_price * (1 - commission)
            trades.append({'type': 'SELL', 'price': current_price, 'shares': shares})
            capital += proceeds
            shares = 0
            position = 0
    
    final_value = capital + shares * float(prices.iloc[-1])
    total_return = (final_value - initial_capital) / initial_capital * 100
    days = len(df)
    annual_return = total_return * (365 / days) if days > 0 else 0
    
    # Max drawdown
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'annual_return_pct': annual_return,
        'max_drawdown_pct': max_drawdown,
        'num_trades': len(trades),
        'trades': trades,
        'portfolio_values': portfolio_values
    }


def print_results(results: Dict, name: str = ""):
    """Print backtest results."""
    print(f"\n{'='*50}")
    print(f"{name} BACKTEST RESULTS" if name else "BACKTEST RESULTS")
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
    print("Available Strategies:")
    for name in registry.list_strategies():
        print(f"  - {name}")
    
    # Fetch data
    print("\nFetching data for AAPL...")
    data = yf.download("AAPL", period="1y", progress=False)
    
    # Test individual strategies
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL STRATEGIES")
    print("="*60)
    
    for strategy_name in ['sma_crossover', 'rsi', 'macd', 'momentum']:
        strategy_func = registry.get(strategy_name)
        df_strategy = strategy_func(data)
        result = run_backtest(df_strategy)
        print_results(result, strategy_name.upper())
    
    # Test blended strategy
    print("\n" + "="*60)
    print("TESTING BLENDED STRATEGY")
    print("="*60)
    
    blender = StrategyBlender({
        'sma_crossover': (sma_crossover_strategy, 0.3),
        'rsi': (rsi_strategy, 0.3),
        'macd': (macd_strategy, 0.2),
        'momentum': (momentum_strategy, 0.2)
    })
    
    df_blended = blender.apply(data)
    result = run_backtest(df_blended)
    print_results(result, "BLENDED (SMA+RSI+MACD+Momentum)")
    
    # Test router
    print("\n" + "="*60)
    print("TESTING STRATEGY ROUTER")
    print("="*60)
    
    router = StrategyRouter()
    router.add_strategy("bull", sma_crossover_strategy, {'short_window': 20, 'long_window': 50})
    router.add_strategy("bear", rsi_strategy, {'oversold': 35, 'overbought': 65})
    router.add_strategy("volatile", bollinger_strategy, {'window': 20, 'num_std': 2.5})
    router.add_strategy("neutral", momentum_strategy, {'lookback': 20})
    
    regime = router.detect_regime(data)
    print(f"Detected market regime: {regime.upper()}")
    
    df_routed = router.apply(data)
    result = run_backtest(df_routed)
    print_results(result, f"ROUTED ({regime.upper()})")