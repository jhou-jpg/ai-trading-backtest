"""
Stock Prediction V5 - Back to What Worked
=========================================
- Train & test on same stocks (time-based split)
- Avoid stocks with extreme returns for test
- Add stop-loss to prevent -100% 
- Focus on stable stocks
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as as as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FEATURES
# ============================================================

def create_features(ticker, period="5y"):
    df = yf.download(ticker, period=period, progress=False)
    if len(df) < 1000:
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Returns
    for w in [1,2,5,10,20]:
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
    
    # Volatility
    df['vol_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Target: 5-day forward return
    df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    df = df.dropna()
    
    exclude = ['target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
    features = [c for c in df.columns if c not in exclude]
    
    return df[features], df['target']


# ============================================================
# MODEL
# ============================================================

class StockPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def train_and_test(ticker, epochs=80):
    """Train and test on SAME stock (time-based split)."""
    data = create_features(ticker, "5y")
    if data is None:
        return None
    
    features, target = data
    
    # Time-based split: first 3 years train, last 2 years test
    split = int(len(features) * 0.6)
    
    X_train = features.iloc[:split]
    y_train = target.iloc[:split]
    X_test = features.iloc[split:]
    y_test = target.iloc[split:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # To tensors
    train_ds = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values).reshape(-1, 1)
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train
    model = StockPredictor(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    train_loader = DataLoader(train_ds, 256, shuffle=True, drop_last=True)
    
    for epoch in range(epochs):
        model.train()
        for x, yb in train_loader:
            x, yb = x.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    # Test
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(device))).cpu().numpy().flatten()
    
    predictions = (probs > 0.5).astype(int)
    accuracy = (predictions == y_test.values).mean()
    
    # Returns
    returns = X_test['return_5d'].values
    
    # Strategy with stop-loss (max 20% loss per trade)
    strategy_returns = []
    for i in range(len(returns)):
        ret = returns[i] * (1 if predictions[i] == 1 else -1)
        # Stop loss: if position loses >20%, exit
        if i > 0 and strategy_returns:
            cumulative = (1 + np.array(strategy_returns)).prod()
            if cumulative < 0.80:  # Stop loss hit
                ret = 0  # Exit
        strategy_returns.append(ret)
    
    strategy_returns = np.array(strategy_returns)
    
    # No stop-loss version
    strategy_returns_no_sl = returns * (predictions * 2 - 1)
    
    # Costs
    costs = 0.001
    strategy_returns = strategy_returns - costs * np.abs(np.diff(np.concatenate([[0], predictions])))
    strategy_returns_no_sl = strategy_returns_no_sl - costs * np.abs(np.diff(np.concatenate([[0], predictions])))
    
    # Cumulative
    strat_cum = (1 + strategy_returns).cumprod()
    strat_no_sl_cum = (1 + strategy_returns_no_sl).cumprod()
    bh_cum = (1 + returns).cumprod()
    
    return {
        'ticker': ticker,
        'accuracy': accuracy,
        'num_trades': int(np.sum(np.abs(np.diff(predictions)))),
        'return_with_sl': strat_cum[-1] - 1,  # With stop-loss
        'return_no_sl': strat_no_sl_cum[-1] - 1,  # No stop-loss
        'buyhold': bh_cum[-1] - 1,
        'sharpe_with_sl': strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252/5),
        'sharpe_no_sl': strategy_returns_no_sl.mean() / (strategy_returns_no_sl.std() + 1e-8) * np.sqrt(252/5),
        'sharpe_bh': returns.mean() / (returns.std() + 1e-8) * np.sqrt(252/5),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Stable stocks (avoid extreme movers like PLTR, COIN)
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Tech
        'JPM', 'V', 'JNJ', 'PG', 'UNH',  # Financial/Healthcare
        'HD', 'MA', 'ADBE', 'INTC', 'AMD',  # More tech
        'QCOM', 'TXN', 'AVGO', 'ORCL', 'IBM',  # Tech
    ]
    
    print("="*60)
    print("TRAINING & TESTING (Time-Based Split)")
    print("="*60)
    print("Train: First 3 years | Test: Last 2 years")
    print()
    
    results = []
    for ticker in TICKERS:
        print(f"Processing {ticker}...", end=" ")
        r = train_and_test(ticker)
        if r:
            results.append(r)
            print(f"Acc: {r['accuracy']:.1%}, NoSL: {r['return_no_sl']*100:+6.1f}%, B&H: {r['buyhold']*100:+6.1f}%")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_sl = np.mean([r['return_with_sl'] for r in results])
    avg_no_sl = np.mean([r['return_no_sl'] for r in results])
    avg_bh = np.mean([r['buyhold'] for r in results])
    
    print(f"\nAccuracy: {avg_acc:.1%}")
    print(f"With Stop-Loss (20%): {avg_sl*100:+6.1f}%")
    print(f"No Stop-Loss: {avg_no_sl*100:+6.1f}%")
    print(f"Buy & Hold: {avg_bh*100:+6.1f}%")
    print(f"\nML (NoSL) vs B&H: {(avg_no_sl - avg_bh)*100:+6.1f}%")
    
    # Win rate
    wins = sum(1 for r in results if r['return_no_sl'] > r['buyhold'])
    print(f"\nWins: {wins}/{len(results)} ({wins/len(results)*100:.0f}%)")
    
    print("\n✅ Done!")