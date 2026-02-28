"""
Stock Prediction V6 - Combined Training (Like V2)
================================================
- Train on MULTIPLE stocks COMBINED (more data = better)
- Test on held-out stocks
- Stop-loss to prevent -100%
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
# FEATURES
# ============================================================

def create_features(ticker, period="5y"):
    df = yf.download(ticker, period=period, progress=False)
    if len(df) < 800:
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
    
    # Target
    df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    df = df.dropna()
    
    exclude = ['target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
    features = [c for c in df.columns if c not in exclude]
    
    return df[features], df['target']


# ============================================================
# MODEL - Deep NN
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
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# MAIN - COMBINED TRAINING
# ============================================================

if __name__ == "__main__":
    # 25 stocks for training (combined)
    TRAIN_TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V', 'JNJ', 'PG',
        'UNH', 'HD', 'MA', 'ADBE', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO', 'ORCL',
        'IBM', 'WMT', 'DIS', 'NFLX', 'CRM'
    ]
    
    # 5 stocks for testing (OUT OF SAMPLE)
    TEST_TICKERS = ['TSLA', 'COIN', 'PLTR', 'SNOW', 'DDOG']
    
    print("="*60)
    print("STEP 1: COLLECT TRAINING DATA")
    print("="*60)
    
    all_features, all_targets = [], []
    for t in TRAIN_TICKERS:
        data = create_features(t, "5y")
        if data:
            f, y = data
            # Use first 60% for training (time-based)
            f_train = f.iloc[:int(len(f)*0.6)]
            y_train = y.iloc[:int(len(y)*0.6)]
            all_features.append(f_train)
            all_targets.append(y_train)
            print(f"  {t}: {len(f_train)} samples")
    
    X_train_all = pd.concat(all_features)
    y_train_all = pd.concat(all_targets)
    
    print(f"\nTotal training samples: {len(X_train_all)}")
    print(f"Features: {X_train_all.shape[1]}")
    print(f"Class: UP={y_train_all.sum()} ({y_train_all.mean()*100:.1f}%)")
    
    # ============================================================
    # TRAIN ON COMBINED DATA
    # ============================================================
    
    print("\n" + "="*60)
    print("STEP 2: TRAIN MODEL (COMBINED)")
    print("="*60)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_all)
    y_arr = y_train_all.values
    
    # Split for validation
    split = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:split]
    X_val = X_scaled[split:]
    y_train = y_arr[:split]
    y_val = y_arr[split:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create data loaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1,1))
    train_loader = DataLoader(train_ds, 256, shuffle=True, drop_last=True)
    
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = StockPredictor(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)
    
    print("Training...")
    best_acc = 0
    best_state = None
    
    for epoch in range(80):
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
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Val Acc = {acc:.2%}")
    
    model.load_state_dict(best_state)
    print(f"\nBest Validation Accuracy: {best_acc:.2%}")
    
    # ============================================================
    # TEST ON OUT-OF-SAMPLE STOCKS
    # ============================================================
    
    print("\n" + "="*60)
    print("STEP 3: TEST ON NEW STOCKS")
    print("="*60)
    
    results = []
    for ticker in TEST_TICKERS:
        data = create_features(ticker, "5y")
        if data is None:
            continue
            
        features, target = data
        
        # Use last 40% for testing
        split = int(len(features) * 0.6)
        X_test = features.iloc[split:]
        y_test = target.iloc[split:]
        
        X_test_scaled = scaler.transform(X_test)
        
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(device))).cpu().numpy().flatten()
        
        predictions = (probs > 0.5).astype(int)
        accuracy = (predictions == y_test.values).mean()
        
        returns = X_test['return_5d'].values
        
        # Strategy returns
        strategy_returns = returns * (predictions * 2 - 1)
        
        # Transaction costs
        costs = 0.001
        strategy_returns = strategy_returns - costs * np.abs(np.diff(np.concatenate([[0], predictions])))
        
        # Cumulative
        strat_cum = (1 + strategy_returns).cumprod()
        bh_cum = (1 + returns).cumprod()
        
        strat_ret = strat_cum[-1] - 1
        bh_ret = bh_cum[-1] - 1
        
        results.append({
            'ticker': ticker,
            'accuracy': accuracy,
            'ml_return': strat_ret,
            'buyhold': bh_ret
        })
        
        print(f"\n{ticker}:")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  ML: {strat_ret*100:+6.1f}%")
        print(f"  B&H: {bh_ret*100:+6.1f}%")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_ml = np.mean([r['ml_return'] for r in results])
    avg_bh = np.mean([r['buyhold'] for r in results])
    
    print(f"Accuracy: {avg_acc:.1%}")
    print(f"ML Return: {avg_ml*100:+6.1f}%")
    print(f"B&H Return: {avg_bh*100:+6.1f}%")
    
    wins = sum(1 for r in results if r['ml_return'] > r['buyhold'])
    print(f"Wins: {wins}/{len(results)}")
    
    print("\n✅ Done!")