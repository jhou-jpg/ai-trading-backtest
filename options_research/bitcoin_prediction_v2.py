"""
Bitcoin Prediction V2 - Scaled Up
=================================
- More data (3 years hourly)
- More features
- Predict 4-hour direction (less noise than 1h)
- Multiple prediction horizons
- Better risk management
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ENHANCED BITCOIN FEATURES
# ============================================================

def get_bitcoin_data(period="3y", interval="1h"):
    """Get more Bitcoin data."""
    btc = yf.download("BTC-USD", period=period, interval=interval, progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    return btc


def create_features(btc):
    """Enhanced features for Bitcoin."""
    df = btc.copy()
    
    # Returns at MANY horizons
    for w in [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 168]:
        df[f'return_{w}h'] = df['Close'].pct_change(w)
    
    # Moving averages
    for w in [6, 12, 24, 48, 72, 168, 336]:  # Up to 2 weeks
        df[f'sma_{w}'] = df['Close'].rolling(w).mean()
        df[f'ema_{w}'] = df['Close'].ewm(span=w).mean()
    
    # Price relative to MAs
    df['price_sma24'] = df['Close'] / df['sma_24']
    df['price_sma168'] = df['Close'] / df['sma_168']
    df['price_sma336'] = df['Close'] / df['sma_336']
    df['sma24_sma168'] = df['sma_24'] / df['sma_168']
    df['sma168_sma336'] = df['sma_168'] / df['sma_336']
    
    # Golden/Death cross
    df['golden_cross'] = ((df['sma_72'] > df['sma_168']) & (df['sma_72'].shift(1) <= df['sma_168'].shift(1))).astype(int)
    df['death_cross'] = ((df['sma_72'] < df['sma_168']) & (df['sma_72'].shift(1) >= df['sma_168'].shift(1))).astype(int)
    
    # RSI at multiple windows
    for w in [8, 14, 24, 48, 72]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        df[f'rsi_{w}'] = 100 - (100 / (1 + gain / loss))
    
    # MACD at multiple settings
    for fast, slow, sig in [(12, 26, 9), (8, 21, 5), (24, 52, 18)]:
        ema_fast = df['Close'].ewm(span=fast).mean()
        ema_slow = df['Close'].ewm(span=slow).mean()
        df[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
        df[f'macd_{fast}_{slow}_sig'] = df[f'macd_{fast}_{slow}'].ewm(span=sig).mean()
        df[f'macd_{fast}_{slow}_hist'] = df[f'macd_{fast}_{slow}'] - df[f'macd_{fast}_{slow}_sig']
    
    # Stochastic
    for w in [8, 24]:
        low = df['Low'].rolling(w).min()
        high = df['High'].rolling(w).max()
        df[f'stoch_k_{w}'] = 100 * (df['Close'] - low) / (high - low + 0.001)
        df[f'stoch_d_{w}'] = df[f'stoch_k_{w}'].rolling(3).mean()
    
    # Bollinger Bands
    bb_mid = df['Close'].rolling(24).mean()
    bb_std = df['Close'].rolling(24).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.001)
    
    # Volume
    for w in [12, 24, 72]:
        df[f'volume_ma{w}'] = df['Volume'].rolling(w).mean()
        df[f'volume_ratio_{w}'] = df['Volume'] / df[f'volume_ma{w}']
    
    # OBV
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    df['obv_ma24'] = df['obv'].rolling(24).mean()
    df['obv_ratio'] = df['obv'] / (df['obv_ma24'] + 0.001)
    
    # Volatility
    for w in [8, 24, 72, 168]:
        df[f'vol_{w}h'] = df['return_1h'].rolling(w).std() * np.sqrt(24)
    
    # Momentum
    for w in [12, 24, 48]:
        df[f'momentum_{w}h'] = df['Close'] / df['Close'].shift(w)
    
    # Range (high-low)
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    df['range_ma24'] = df['range_pct'].rolling(24).mean()
    
    # Hour of day (cyclical)
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week
    df['dayofweek'] = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Lagged features
    for lag in [1, 2, 3, 4, 6, 12, 24]:
        df[f'return_1h_lag{lag}'] = df['return_1h'].shift(lag)
        df[f'macd_12_26_hist_lag{lag}'] = df['macd_12_26_hist'].shift(lag)
    
    # Target: 4-hour return (less noise than 1h)
    df['target'] = (df['Close'].shift(-4) > df['Close']).astype(int)
    
    df = df.dropna()
    
    exclude = ['target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close', 'hour', 'dayofweek']
    features = [c for c in df.columns if c not in exclude]
    
    return df[features], df['target'], df['Close']


# ============================================================
# MODEL
# ============================================================

class DeepBitcoin(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Input
        self.input = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Hidden blocks
        for i in range(4):
            setattr(self, f'block{i}', nn.Sequential(
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.3)
            ))
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.input(x)
        for i in range(4):
            x = x + getattr(self, f'block{i}')(x)  # Residual
        return self.output(x)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("BITCOIN V2 - SCALED UP")
    print("="*60)
    
    # Get more data
    print("\nDownloading Bitcoin data (3 years)...")
    btc = get_bitcoin_data("3y", "1h")
    print(f"Got {len(btc)} hourly data points")
    
    # Create features
    print("\nCreating enhanced features...")
    X, y, prices = create_features(btc)
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {len(X)}")
    print(f"Class: UP={y.sum()} ({y.mean()*100:.1f}%)")
    
    # Time-based split: 70% train, 30% test
    split = int(len(X) * 0.7)
    
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    print("\n" + "="*60)
    print("TRAINING DEEP MODEL")
    print("="*60)
    
    train_ds = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values).reshape(-1, 1)
    )
    train_loader = DataLoader(train_ds, 256, shuffle=True, drop_last=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = DeepBitcoin(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)
    
    best_acc = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(100):
        model.train()
        for x, yb in train_loader:
            x, yb = x.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(device))) > 0.5).float()
            acc = (preds.squeeze() == torch.FloatTensor(y_test.values).to(device)).float().mean().item()
        
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Test Acc = {acc:.2%}")
        
        if patience_counter >= 15:
            print(f"Early stop at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    print(f"\nBest Test Accuracy: {best_acc:.2%}")
    
    # Backtest with multiple strategies
    print("\n" + "="*60)
    print("BACKTEST (Multiple Thresholds)")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(device))).cpu().numpy().flatten()
    
    returns = X_test['return_4h'].values  # 4-hour returns
    
    for threshold in [0.45, 0.50, 0.55, 0.60]:
        predictions = (probs > threshold).astype(int)
        
        strategy_returns = returns * (predictions * 2 - 1)
        
        # Costs
        costs = 0.001
        strategy_returns = strategy_returns - costs * np.abs(np.diff(np.concatenate([[0], predictions])))
        
        strat_cum = (1 + strategy_returns).cumprod()
        bh_cum = (1 + returns).cumprod()
        
        strat_ret = strat_cum[-1] - 1
        bh_ret = bh_cum[-1] - 1
        
        strat_sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(6*365)  # 4h = 6x/day
        bh_sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(6*365)
        
        print(f"\nThreshold {threshold:.2f}:")
        print(f"  ML: {strat_ret*100:+7.2f}% (Sharpe: {strat_sharpe:.2f})")
        print(f"  B&H: {bh_ret*100:+7.2f}% (Sharpe: {bh_sharpe:.2f})")
    
    print("\n✅ Done!")