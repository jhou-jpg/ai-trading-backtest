"""
Stock Price Prediction Model - Using Real Data
==============================================
Predict stock price direction (UP/DOWN) using ML.
A/B test against existing strategies.

Usage:
1. Train on historical data
2. Generate predictions
3. Compare to SMA/EMA/RSI strategies
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PART 1: FEATURE ENGINEERING
# ============================================================

def create_features(ticker, period="2y", interval="1d"):
    """
    Create features for stock prediction.
    
    Features:
    - Price returns (1d, 5d, 10d, 20d)
    - Technical indicators (RSI, MACD, SMA ratios)
    - Volume indicators
    - Momentum
    
    Target: Next day return (positive = BUY, negative = SELL)
    """
    # Download data
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if len(df) == 0:
        raise ValueError(f"No data for {ticker}")
    
    # Handle multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Returns at different horizons
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)
    df['return_20d'] = df['Close'].pct_change(20)
    
    # Moving averages
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()
    
    # Price relative to MAs
    df['price_to_sma20'] = df['Close'] / df['sma_20']
    df['price_to_sma50'] = df['Close'] / df['sma_50']
    df['sma20_to_sma50'] = df['sma_20'] / df['sma_50']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volume
    df['volume_ma20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma20']
    
    # Bollinger Bands
    df['bb_upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
    df['bb_lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volatility
    df['volatility_20d'] = df['return_1d'].rolling(20).std()
    
    # Target: Next day return (positive = 1, negative = 0)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaN
    df = df.dropna()
    
    # Features to use
    feature_cols = [
        'return_1d', 'return_5d', 'return_10d', 'return_20d',
        'price_to_sma20', 'price_to_sma50', 'sma20_to_sma50',
        'rsi', 'macd', 'macd_hist',
        'volume_ratio', 'bb_position', 'volatility_20d'
    ]
    
    return df[feature_cols], df['target'], df['Close']


# ============================================================
# PART 2: MULTI-STOCK DATA GENERATOR
# ============================================================

def generate_training_data(tickers, period="2y"):
    """
    Generate training data from multiple stocks.
    This creates a more robust model.
    """
    all_features = []
    all_targets = []
    
    print(f"Fetching data for {len(tickers)} stocks...")
    
    for ticker in tickers:
        try:
            features, target, prices = create_features(ticker, period)
            all_features.append(features)
            all_targets.append(target)
            print(f"  {ticker}: {len(features)} samples")
        except Exception as e:
            print(f"  {ticker}: Error - {e}")
    
    # Combine all
    X = pd.concat(all_features)
    y = pd.concat(all_targets)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Class distribution: UP={y.sum()} ({y.sum()/len(y)*100:.1f}%), DOWN={len(y)-y.sum()} ({(1-y.sum()/len(y))*100:.1f}%)")
    
    return X, y


# ============================================================
# PART 3: MODEL ARCHITECTURE
# ============================================================

class StockPredictor(nn.Module):
    """
    Neural network to predict stock direction.
    Simple architecture - easy to train, easy to understand.
    """
    def __init__(self, input_dim=13, hidden_dims=[64, 32]):
        super(StockPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Binary classification: UP or DOWN
        self.output = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        x = self.shared(x)
        x = self.output(x)
        return x


# ============================================================
# PART 4: TRAINING
# ============================================================

def train_model(X, y, epochs=50, batch_size=256, lr=1e-3):
    """
    Train the model.
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y.values, test_size=0.2, shuffle=False  # Time-series split
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Convert to tensors
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).reshape(-1, 1)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).reshape(-1, 1)
    )
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)
    
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StockPredictor(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"\nTraining on {device}...")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x, y_batch in train_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y_batch in val_loader:
                x, y_batch = x.to(device), y_batch.to(device)
                pred = model(x)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                
                # Accuracy
                predicted = (torch.sigmoid(pred) > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        val_acc = correct / total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        history['train_loss'].append(train_loss/len(train_loader))
        history['val_loss'].append(val_loss/len(val_loader))
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2%}")
    
    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    
    return model, scaler, history


# ============================================================
# PART 5: PREDICTION
# ============================================================

def predict(ticker, model, scaler, period="2y"):
    """
    Get prediction for a stock.
    """
    features, target, prices = create_features(ticker, period)
    
    # Scale
    X = scaler.transform(features.values)
    
    # Predict
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        probs = torch.sigmoid(model(X_tensor)).cpu().numpy().flatten()
    
    # Get latest prediction
    latest_prob = probs[-1]
    latest_price = prices.iloc[-1]
    
    # Signal
    if latest_prob > 0.55:
        signal = "BUY"
    elif latest_prob < 0.45:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    return {
        'ticker': ticker,
        'signal': signal,
        'confidence': latest_prob,
        'price': latest_price,
        'prob_up': latest_prob,
        'prob_down': 1 - latest_prob
    }


# ============================================================
# PART 6: BACKTEST
# ============================================================

def backtest_predictions(ticker, model, scaler, period="2y"):
    """
    Simple backtest: buy when model says UP, sell when DOWN.
    Compare to buy & hold.
    """
    features, target, prices = create_features(ticker, period)
    
    # Scale
    X = scaler.transform(features.values)
    
    # Predict
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = (torch.sigmoid(model(X_tensor)) > 0.5).cpu().numpy().flatten()
    
    # Calculate returns
    returns = features['return_1d'].values
    
    # Strategy returns (long when predicted UP, short when predicted DOWN)
    strategy_returns = returns * (predictions * 2 - 1)  # 1 when UP, -1 when DOWN
    
    # Buy & hold returns
    buy_hold_returns = returns
    
    # Cumulative returns
    strategy_cum = (1 + strategy_returns).cumprod()
    buyhold_cum = (1 + buy_hold_returns).cumprod()
    
    # Metrics
    strategy_total = strategy_cum[-1] - 1
    buyhold_total = buyhold_cum[-1] - 1
    
    # Sharpe (annualized)
    strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    buyhold_sharpe = buy_hold_returns.mean() / buy_hold_returns.std() * np.sqrt(252)
    
    print(f"\n{'='*50}")
    print(f"BACKTEST: {ticker}")
    print(f"{'='*50}")
    print(f"ML Strategy: {strategy_total*100:.2f}% (Sharpe: {strategy_sharpe:.2f})")
    print(f"Buy & Hold: {buyhold_total*100:.2f}% (Sharpe: {buyhold_sharpe:.2f})")
    print(f"ML vs B&H: {(strategy_total - buyhold_total)*100:+.2f}%")
    
    return {
        'strategy_return': strategy_total,
        'buyhold_return': buyhold_total,
        'strategy_sharpe': strategy_sharpe,
        'buyhold_sharpe': buyhold_sharpe
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Stocks to train on
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'WMT']
    
    # Generate data
    print("Generating training data...")
    X, y = generate_training_data(TICKERS, period="2y")
    
    # Train
    print("\nTraining model...")
    model, scaler, history = train_model(X, y, epochs=50)
    
    # Test on a stock
    print("\n" + "="*50)
    print("TEST PREDICTIONS")
    print("="*50)
    
    for ticker in ['AAPL', 'NVDA', 'TSLA']:
        try:
            result = predict(ticker, model, scaler)
            print(f"\n{ticker}: {result['signal']} (confidence: {result['confidence']:.1%}, price: ${result['price']:.2f})")
            
            # Backtest
            backtest_predictions(ticker, model, scaler)
        except Exception as e:
            print(f"{ticker}: Error - {e}")
    
    print("\n✅ Model ready!")