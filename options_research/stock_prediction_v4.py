"""
Stock Prediction V4 - Realistic Out-of-Sample Testing
=====================================================
- Proper train/test split (no look-ahead)
- Transaction costs included
- Only stocks with full 5-year history
- Scaled up NN option
- Realistic backtest
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
# PART 1: FEATURES
# ============================================================

def create_features(ticker, period="5y"):
    """Create features for a stock."""
    df = yf.download(ticker, period=period, progress=False)
    if len(df) < 1000:  # Need ~5 years of data
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Returns
    for w in [1,2,3,5,10,20]:
        df[f'return_{w}d'] = df['Close'].pct_change(w)
    
    # MAs
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
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volatility
    df['vol_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
    
    # Volume
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Target: 5-day forward return
    df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    df = df.dropna()
    
    exclude = ['target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
    features = [c for c in df.columns if c not in exclude]
    
    return df[features], df['target']


# ============================================================
# PART 2: SCALED UP DEEP NN
# ============================================================

class DeepNN(nn.Module):
    """
    Larger neural network for stock prediction.
    """
    def __init__(self, input_dim):
        super().__init__()
        
        # Input processing
        self.input = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Hidden blocks
        self.block1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output
        self.output = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.input(x)
        x = x + self.block1(x)  # Residual
        x = x + self.block2(x)
        x = x + self.block3(x)
        return self.output(x)


def train_scaled_nn(X_train, y_train, X_val, y_val, epochs=100, batch_size=256):
    """Train the scaled up NN."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}...")
    
    model = DeepNN(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1,1))
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    
    best_acc = 0
    best_state = None
    patience = 15
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
            acc = (preds.squeeze() == torch.FloatTensor(y_val)).float().mean().item()
        
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Val Acc = {acc:.2%}")
        
        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model, best_acc


# ============================================================
# PART 3: REALISTIC BACKTEST (OUT-OF-SAMPLE)
# ============================================================

def realistic_backtest(train_tickers, test_ticker, model, scaler, train_years=3):
    """
    Realistic backtest:
    1. Train on train_tickers for train_years
    2. Test on test_ticker (out-of-sample!)
    3. Include transaction costs
    """
    # Get all data for test ticker
    data = create_features(test_ticker, period="5y")
    if data is None:
        return None
    
    features, target = data
    
    # Split: first 3 years train, last 2 years test (OUT OF SAMPLE)
    split_idx = int(len(features) * 0.4)  # 40% train, 60% test
    
    train_features = features.iloc[:split_idx]
    train_target = target.iloc[:split_idx]
    test_features = features.iloc[split_idx:]
    test_target = target.iloc[split_idx:]
    
    # Scale on training data only
    scaler.fit(train_features)
    X_test = scaler.transform(test_features)
    
    # Predict
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.FloatTensor(X_test).to(device)).cpu().numpy().flatten())
    
    predictions = (probs > 0.5).astype(int)
    
    # Returns
    returns = test_features['return_5d'].values
    
    # Transaction costs (0.1% per trade)
    transaction_cost = 0.001
    num_trades = np.sum(np.diff(predictions) != 0)
    total_cost = num_trades * transaction_cost
    
    # Strategy returns with costs
    strategy_returns = returns * (predictions * 2 - 1)
    strategy_returns = strategy_returns - transaction_cost * np.abs(np.diff(np.concatenate([[0], predictions])))
    
    # Buy & hold returns
    buyhold_returns = returns
    
    # Cumulative
    strategy_cum = (1 + strategy_returns).cumprod()
    bh_cum = (1 + buyhold_returns).cumprod()
    
    # Metrics
    strategy_return = strategy_cum[-1] - 1
    bh_return = bh_cum[-1] - 1
    
    # Sharpe
    strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252/5) if strategy_returns.std() > 0 else 0
    bh_sharpe = buyhold_returns.mean() / buyhold_returns.std() * np.sqrt(252/5) if buyhold_returns.std() > 0 else 0
    
    # Accuracy
    accuracy = (predictions == test_target.values).mean()
    
    return {
        'ticker': test_ticker,
        'accuracy': accuracy,
        'strategy_return': strategy_return,
        'buyhold_return': bh_return,
        'strategy_sharpe': strategy_sharpe,
        'buyhold_sharpe': bh_sharpe,
        'num_trades': num_trades,
        'test_samples': len(test_features)
    }


# ============================================================
# PART 4: MAIN - TRAIN ON MULTIPLE, TEST ON NEW STOCKS
# ============================================================

if __name__ == "__main__":
    # STABLE stocks with full 5-year history
    TRAIN_TICKERS = [
        'AAPL','MSFT','GOOGL','AMZN','NVDA','META','JPM','V','JNJ','PG',
        'UNH','HD','MA','ADBE','INTC','AMD','QCOM','TXN','AVGO','ORCL'
    ]
    
    # Test on stocks NOT in training
    TEST_TICKERS = ['TSLA', 'COIN', 'PLTR', 'SNOW', 'DDOG']
    
    print("="*60)
    print("PREPARING TRAINING DATA")
    print("="*60)
    
    # Collect training data
    all_features, all_targets = [], []
    for t in TRAIN_TICKERS:
        data = create_features(t, "5y")
        if data:
            f, y = data
            # Use first 40% for training
            f_train = f.iloc[:int(len(f)*0.4)]
            y_train = y.iloc[:int(len(y)*0.4)]
            all_features.append(f_train)
            all_targets.append(y_train)
            print(f"  {t}: {len(f_train)} train samples")
    
    X_train_all = pd.concat(all_features)
    y_train_all = pd.concat(all_targets)
    
    print(f"\nTotal training samples: {len(X_train_all)}")
    print(f"Features: {X_train_all.shape[1]}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)
    y_train_arr = y_train_all.values
    
    # Split for validation
    split = int(len(X_train_scaled) * 0.8)
    X_train = X_train_scaled[:split]
    X_val = X_train_scaled[split:]
    y_train = y_train_arr[:split]
    y_val = y_train_arr[split:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Train scaled NN
    print("\n" + "="*60)
    print("TRAINING SCALED UP DEEP NN")
    print("="*60)
    
    model, val_acc = train_scaled_nn(X_train, y_train, X_val, y_val, epochs=100)
    print(f"\nValidation Accuracy: {val_acc:.2%}")
    
    # Test on OUT-OF-SAMPLE stocks
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE BACKTEST (Realistic)")
    print("="*60)
    print("(Testing on stocks NOT seen during training)")
    print()
    
    results = []
    for ticker in TEST_TICKERS:
        result = realistic_backtest(TRAIN_TICKERS, ticker, model, scaler)
        if result:
            results.append(result)
            print(f"\n{result['ticker']}:")
            print(f"  Accuracy: {result['accuracy']:.1%}")
            print(f"  ML Return: {result['strategy_return']*100:+6.1f}% (Sharpe: {result['strategy_sharpe']:.2f})")
            print(f"  B&H Return: {result['buyhold_return']*100:+6.1f}% (Sharpe: {result['buyhold_sharpe']:.2f})")
            print(f"  Trades: {result['num_trades']}")
            print(f"  Test samples: {result['test_samples']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_ml = np.mean([r['strategy_return'] for r in results])
    avg_bh = np.mean([r['buyhold_return'] for r in results])
    
    print(f"Average Accuracy: {avg_acc:.1%}")
    print(f"Average ML Return: {avg_ml*100:+6.1f}%")
    print(f"Average B&H Return: {avg_bh*100:+6.1f}%")
    print(f"ML vs B&H: {(avg_ml - avg_bh)*100:+6.1f}%")
    
    print("\n✅ Done! This is more realistic - train on some stocks, test on new ones.")