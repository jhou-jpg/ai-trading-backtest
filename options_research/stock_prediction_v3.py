"""
Stock Prediction V3 - Scaled Up + Ensemble
==========================================
- More stocks and data
- Feature importance analysis
- Ensemble with SMA/RSI/MACD signals
- Stock-specific thresholds
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PART 1: FEATURE ENGINEERING (Same as V2)
# ============================================================

def create_features(ticker, period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if len(df) == 0:
        raise ValueError(f"No data for {ticker}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Returns
    for w in [1,2,3,5,10,20,50]:
        df[f'return_{w}d'] = df['Close'].pct_change(w)
    
    # MAs
    for w in [5,10,20,50,100,200]:
        df[f'sma_{w}'] = df['Close'].rolling(w).mean()
        df[f'ema_{w}'] = df['Close'].ewm(span=w).mean()
    
    df['price_to_sma20'] = df['Close'] / df['sma_20']
    df['price_to_sma50'] = df['Close'] / df['sma_50']
    df['price_to_sma200'] = df['Close'] / df['sma_200']
    df['sma20_sma50'] = df['sma_20'] / df['sma_50']
    df['sma50_sma200'] = df['sma_50'] / df['sma_200']
    df['golden_cross'] = ((df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(int)
    df['death_cross'] = ((df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1))).astype(int)
    
    # RSI
    for w in [7,14,21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        df[f'rsi_{w}'] = 100 - (100 / (1 + gain / loss))
    
    # MACD
    ema12, ema26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Stochastic
    low14, high14 = df['Low'].rolling(14).min(), df['High'].rolling(14).max()
    df['stoch_k'] = 100 * (df['Close'] - low14) / (high14 - low14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # Bollinger
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2*bb_std
    df['bb_lower'] = bb_mid - 2*bb_std
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.001)
    
    # ATR
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / tr.rolling(20).mean()
    
    # Volatility
    df['vol_10d'] = df['return_1d'].rolling(10).std() * np.sqrt(252)
    df['vol_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
    
    # Volume
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Lag features
    for lag in [1,2,3,5]:
        df[f'return_1d_lag{lag}'] = df['return_1d'].shift(lag)
    
    # Target: 5-day forward return
    df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    df = df.dropna()
    
    exclude = ['target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
    features = [c for c in df.columns if c not in exclude]
    
    return df[features], df['target'], df['Close']


# ============================================================
# PART 2: TRADITIONAL STRATEGY SIGNALS
# ============================================================

def get_traditional_signals(ticker, period="2y"):
    """Get signals from traditional strategies."""
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    signals = {}
    
    # SMA Crossover
    sma20, sma50 = df['Close'].rolling(20).mean(), df['Close'].rolling(50).mean()
    signals['sma_cross'] = (sma20 > sma50).astype(int)
    
    # RSI
    delta = df['Close'].diff()
    rsi = 100 - (100 / (1 + delta.where(delta > 0, 0).rolling(14).mean() / (-delta.where(delta < 0, 0).rolling(14).mean())))
    signals['rsi'] = (rsi < 30).astype(int) - (rsi > 70).astype(int)
    
    # MACD
    macd = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    macd_signal = macd.ewm(span=9).mean()
    signals['macd'] = (macd > macd_signal).astype(int) - (macd < macd_signal).astype(int)
    
    # Combine
    combined = signals['sma_cross'] + signals['rsi'] + signals['macd']
    
    return combined.fillna(0)


# ============================================================
# PART 3: ENSEMBLE MODEL
# ============================================================

class EnsemblePredictor:
    """
    Ensemble of Neural Network + Random Forest + Traditional Strategies
    """
    def __init__(self):
        self.nn_model = None
        self.rf_model = None
        self.scaler = None
        self.feature_names = None
        
    def create_nn(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def fit(self, X, y, epochs=80, batch_size=512):
        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = X.columns.tolist()
        
        # Split
        split = int(len(X) * 0.8)
        X_train, X_val = X_scaled[:split], X_scaled[split:]
        y_train, y_val = y[:split], y[split:]
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Train NN
        print("Training Neural Network...")
        self.nn_model = self.create_nn(X.shape[1]).to(device)
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train.values).reshape(-1,1))
        train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
        
        opt = torch.optim.AdamW(self.nn_model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            self.nn_model.train()
            for x, yb in train_loader:
                x, yb = x.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(self.nn_model(x), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), 1.0)
                opt.step()
        
        # Train RF
        print("Training Random Forest...")
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.rf_model.fit(X_train, y_train)
        
        # Validate both
        self.nn_model.eval()
        with torch.no_grad():
            nn_probs = torch.sigmoid(self.nn_model(torch.FloatTensor(X_val).to(device))).cpu().numpy().flatten()
        rf_probs = self.rf_model.predict_proba(X_val)[:, 1]
        
        nn_acc = ((nn_probs > 0.5) == y_val.values).mean()
        rf_acc = ((rf_probs > 0.5) == y_val.values).mean()
        
        # Ensemble (weighted average)
        ensemble_probs = 0.6 * nn_probs + 0.4 * rf_probs
        ensemble_acc = ((ensemble_probs > 0.5) == y_val.values).mean()
        
        print(f"NN Val Accuracy: {nn_acc:.2%}")
        print(f"RF Val Accuracy: {rf_acc:.2%}")
        print(f"Ensemble Val Accuracy: {ensemble_acc:.2%}")
        
        return ensemble_acc
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.nn_model.eval()
        with torch.no_grad():
            nn_probs = torch.sigmoid(self.nn_model(torch.FloatTensor(X_scaled).to(device)).cpu().numpy().flatten())
        rf_probs = self.rf_model.predict_proba(X_scaled)[:, 1]
        
        return 0.6 * nn_probs + 0.4 * rf_probs
    
    def get_feature_importance(self):
        """Get feature importance from RF."""
        importance = self.rf_model.feature_importances_
        return pd.DataFrame({'feature': self.feature_names, 'importance': importance}).sort_values('importance', ascending=False)


# ============================================================
# PART 4: STOCK-SPECIFIC BACKTEST
# ============================================================

def backtest_stock(ticker, model, period="2y", threshold=0.5):
    """Backtest with stock-specific threshold."""
    features, target, prices = create_features(ticker, period)
    traditional = get_traditional_signals(ticker, period)
    
    # ML predictions
    probs = model.predict(features)
    predictions = (probs > threshold).astype(int)
    
    # Combine with traditional signals (ensemble)
    # If traditional and ML agree, more confident
    ensemble_signal = predictions + traditional.values[-len(predictions):]
    
    # Use ensemble: require ML and at least one traditional to agree
    final_signal = ((ensemble_signal >= 1) | (probs > threshold + 0.1)).astype(int)
    
    returns = features['return_5d'].values
    
    # Strategy returns
    strategy_returns = returns * (final_signal * 2 - 1)
    
    # Only ML
    ml_returns = returns * (predictions * 2 - 1)
    
    # Cumulative
    strat_cum = (1 + strategy_returns).cumprod()
    ml_cum = (1 + ml_returns).cumprod()
    bh_cum = (1 + returns).cumprod()
    
    return {
        'ticker': ticker,
        'ensemble_return': strat_cum[-1] - 1,
        'ml_return': ml_cum[-1] - 1,
        'buyhold_return': bh_cum[-1] - 1,
        'ensemble_sharpe': strategy_returns.mean()/strategy_returns.std() * np.sqrt(252/5),
        'ml_sharpe': ml_returns.mean()/ml_returns.std() * np.sqrt(252/5),
    }


# ============================================================
# PART 5: MAIN
# ============================================================

if __name__ == "__main__":
    # SCALED UP: More stocks
    TICKERS = [
        'AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','JPM','V','WMT',
        'JNJ','PG','UNH','HD','MA','DIS','PYPL','NFLX','ADBE','CRM',
        'INTC','AMD','QCOM','TXN','AVGO','ORCL','IBM','COIN','SQ','SHOP',
        'SNOW','PLTR','U','DDOG','NET','CRWD','ZS','OKTA','MDB','HUBS'
    ]
    
    print("="*60)
    print("GENERATING TRAINING DATA (40 stocks)")
    print("="*60)
    
    all_features, all_targets = [], []
    for t in TICKERS:
        try:
            f, y, _ = create_features(t, "5y")
            all_features.append(f)
            all_targets.append(y)
            print(f"  {t}: {len(f)} samples")
        except Exception as e:
            print(f"  {t}: Error")
    
    X = pd.concat(all_features)
    y = pd.concat(all_targets)
    print(f"\nTotal: {len(X)} samples, {X.shape[1]} features")
    print(f"Class: UP={y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    
    # Train ensemble
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE MODEL")
    print("="*60)
    
    model = EnsemblePredictor()
    acc = model.fit(X, y, epochs=80)
    print(f"\nBest Ensemble Accuracy: {acc:.2%}")
    
    # Feature importance
    print("\n" + "="*60)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("="*60)
    importance = model.get_feature_importance()
    print(importance.head(20).to_string(index=False))
    
    # Backtest with ensemble
    print("\n" + "="*60)
    print("BACKTEST RESULTS (ENSEMBLE)")
    print("="*60)
    
    results = []
    for ticker in ['AAPL','NVDA','TSLA','MSFT','GOOGL','AMZN','META','AMD','COIN','PLTR']:
        try:
            r = backtest_stock(ticker, model)
            results.append(r)
            print(f"\n{ticker}:")
            print(f"  Ensemble: {r['ensemble_return']*100:+6.1f}% (Sharpe: {r['ensemble_sharpe']:.2f})")
            print(f"  ML Only:  {r['ml_return']*100:+6.1f}% (Sharpe: {r['ml_sharpe']:.2f})")
            print(f"  BuyHold:  {r['buyhold_return']*100:+6.1f}%")
        except Exception as e:
            print(f"{ticker}: Error - {e}")
    
    print("\n✅ Done!")