"""
Stock Prediction V4 - Fixed + Research-Based
==========================================
- Fixed residual connection bug
- Based on research: what actually works for stock prediction
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
# RESEARCH INSIGHTS:
# - Simple models often beat complex ones for stock prediction
# - Feature selection matters more than model complexity  
# - Ensemble of simple models can help
# - LSTM/GRU good for time series
# ============================================================

def create_features(ticker, period="5y"):
    df = yf.download(ticker, period=period, progress=False)
    if len(df) < 1000:
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Simple but effective features
    for w in [1,2,5,10,20]:
        df[f'return_{w}d'] = df['Close'].pct_change(w)
    
    for w in [20,50,200]:
        df[f'sma_{w}'] = df['Close'].rolling(w).mean()
    
    df['price_to_sma50'] = df['Close'] / df['sma_50']
    df['price_to_sma200'] = df['Close'] / df['sma_200']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    # MACD
    ema12, ema26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Volatility
    df['vol_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
    
    # Volume
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Target
    df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    df = df.dropna()
    
    exclude = ['target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
    features = [c for c in df.columns if c not in exclude]
    
    return df[features], df['target']


# ============================================================
# SIMPLER BUT EFFECTIVE MODELS (Research-based)
# ============================================================

class SimpleMLP(nn.Module):
    """
    Simple MLP - research shows simpler models often work better for stocks.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Residual block with matching dimensions."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class DeepMLP(nn.Module):
    """Deeper MLP with proper residual connections."""
    def __init__(self, input_dim):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Residual blocks (all same dimension)
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.output(x)


class LSTMModel(nn.Module):
    """LSTM for time series - good for sequential patterns."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x: (batch, features) -> (batch, seq_len, features)
        x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        # Take last output
        out = lstm_out[:, -1, :]
        
        return self.fc(out)


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=256):
    """Train any model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}...")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1,1))
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    
    best_acc = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
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
            preds = (torch.sigmoid(model(torch.FloatTensor(X_val).to(device))) > 0.5).float()
            acc = (preds.squeeze() == torch.FloatTensor(y_val).to(device)).float().mean().item()
        
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Val Acc = {acc:.2%}")
        
        if patience_counter >= 15:
            print(f"Early stop at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model, best_acc


def realistic_backtest(model, scaler, test_ticker, train_tickers, train_years_pct=0.4):
    """Out-of-sample backtest."""
    data = create_features(test_ticker, "5y")
    if data is None:
        return None
    
    features, target = data
    
    # Time-based split (no look-ahead)
    split_idx = int(len(features) * train_years_pct)
    test_features = features.iloc[split_idx:]
    test_target = target.iloc[split_idx:]
    
    X_test = scaler.transform(test_features)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.FloatTensor(X_test).to(device))).cpu().numpy().flatten()
    
    predictions = (probs > 0.5).astype(int)
    
    returns = test_features['return_5d'].values
    
    # Transaction costs
    cost = 0.001
    strategy_returns = returns * (predictions * 2 - 1) - cost * np.abs(np.diff(np.concatenate([[0], predictions])))
    buyhold_returns = returns
    
    strategy_cum = (1 + strategy_returns).cumprod()
    bh_cum = (1 + buyhold_returns).cumprod()
    
    strat_ret = strategy_cum[-1] - 1
    bh_ret = bh_cum[-1] - 1
    
    strat_sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252/5)
    bh_sharpe = buyhold_returns.mean() / (buyhold_returns.std() + 1e-8) * np.sqrt(252/5)
    
    accuracy = (predictions == test_target.values).mean()
    
    return {
        'ticker': test_ticker,
        'accuracy': accuracy,
        'strategy_return': strat_ret,
        'buyhold_return': bh_ret,
        'strategy_sharpe': strat_sharpe,
        'buyhold_sharpe': bh_sharpe,
        'num_trades': int(np.sum(np.abs(np.diff(predictions)))),
        'test_samples': len(test_features)
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Training stocks
    TRAIN_TICKERS = [
        'AAPL','MSFT','GOOGL','AMZN','NVDA','META','JPM','V','JNJ','PG',
        'UNH','HD','MA','ADBE','INTC','AMD','QCOM','TXN','AVGO','ORCL'
    ]
    
    # Test on NEW stocks (not in training)
    TEST_TICKERS = ['TSLA', 'COIN', 'PLTR', 'SNOW', 'DDOG']
    
    print("="*60)
    print("PREPARING DATA")
    print("="*60)
    
    all_features, all_targets = [], []
    for t in TRAIN_TICKERS:
        data = create_features(t, "5y")
        if data:
            f, y = data
            f_train = f.iloc[:int(len(f)*0.4)]
            y_train = y.iloc[:int(len(y)*0.4)]
            all_features.append(f_train)
            all_targets.append(y_train)
            print(f"  {t}: {len(f_train)} samples")
    
    X_train_all = pd.concat(all_features)
    y_train_all = pd.concat(all_targets)
    
    print(f"\nTotal: {len(X_train_all)} samples, {X_train_all.shape[1]} features")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)
    y_train_arr = y_train_all.values
    
    # Train/val split
    split = int(len(X_train_scaled) * 0.8)
    X_train, X_val = X_train_scaled[:split], X_train_scaled[split:]
    y_train, y_val = y_train_arr[:split], y_train_arr[split:]
    
    # ============================================================
    # TEST MULTIPLE MODELS
    # ============================================================
    
    models = {
        'SimpleMLP': SimpleMLP(X_train.shape[1]),
        'DeepMLP': DeepMLP(X_train.shape[1]),
    }
    
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    best_model = None
    best_acc = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        trained_model, acc = train_model(model, X_train, y_train, X_val, y_val)
        
        if acc > best_acc:
            best_acc = acc
            best_model = trained_model
            best_name = name
    
    print(f"\nBest: {best_name} with {best_acc:.2%}")
    
    # ============================================================
    # OUT-OF-SAMPLE TEST
    # ============================================================
    
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE BACKTEST")
    print("="*60)
    
    results = []
    for ticker in TEST_TICKERS:
        result = realistic_backtest(best_model, scaler, ticker, TRAIN_TICKERS)
        if result:
            results.append(result)
            print(f"\n{result['ticker']}:")
            print(f"  Accuracy: {result['accuracy']:.1%}")
            print(f"  ML: {result['strategy_return']*100:+6.1f}% (Sharpe: {result['strategy_sharpe']:.2f})")
            print(f"  B&H: {result['buyhold_return']*100:+6.1f}% (Sharpe: {result['buyhold_sharpe']:.2f})")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_ml = np.mean([r['strategy_return'] for r in results])
    avg_bh = np.mean([r['buyhold_return'] for r in results])
    
    print(f"Best Model: {best_name}")
    print(f"Average Accuracy: {avg_acc:.1%}")
    print(f"Average ML Return: {avg_ml*100:+6.1f}%")
    print(f"Average B&H Return: {avg_bh*100:+6.1f}%")
    
    print("\n✅ Done!")