"""
Stock Prediction Model V2 - Enhanced
====================================
More features, deeper network, better training.
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
# PART 1: ENHANCED FEATURES
# ============================================================

def create_features_v2(ticker, period="5y", interval="1d"):
    """
    Enhanced feature engineering with more indicators.
    """
    # Download data
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if len(df) == 0:
        raise ValueError(f"No data for {ticker}")
    
    # Handle multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # ============= PRICE FEATURES =============
    # Returns at multiple horizons
    for window in [1, 2, 3, 5, 10, 20, 50]:
        df[f'return_{window}d'] = df['Close'].pct_change(window)
    
    # ============= MOVING AVERAGES =============
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{window}'] = df['Close'].rolling(window).mean()
        df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
    
    # Price relative to MAs
    df['price_to_sma20'] = df['Close'] / df['sma_20']
    df['price_to_sma50'] = df['Close'] / df['sma_50']
    df['price_to_sma200'] = df['Close'] / df['sma_200']
    df['sma20_to_sma50'] = df['sma_20'] / df['sma_50']
    df['sma50_to_sma200'] = df['sma_50'] / df['sma_200']
    
    # Golden cross / Death cross
    df['golden_cross'] = ((df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(int)
    df['death_cross'] = ((df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1))).astype(int)
    
    # ============= MOMENTUM =============
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # RSI at different windows
    for window in [7, 21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_hist_change'] = df['macd_hist'].diff()
    
    # Stochastic
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * (df['Close'] - low14) / (high14 - low14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # ROC (Rate of Change)
    for window in [10, 20]:
        df[f'roc_{window}'] = ((df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window)) * 100
    
    # ============= VOLATILITY =============
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_20'] = tr.rolling(20).mean()
    df['atr_ratio'] = df['atr_14'] / df['atr_20']
    
    # Historical volatility
    df['volatility_10d'] = df['return_1d'].rolling(10).std() * np.sqrt(252)
    df['volatility_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
    df['volatility_60d'] = df['return_1d'].rolling(60).std() * np.sqrt(252)
    
    # ============= VOLUME =============
    df['volume_ma10'] = df['Volume'].rolling(10).mean()
    df['volume_ma20'] = df['Volume'].rolling(20).mean()
    df['volume_ma50'] = df['Volume'].rolling(50).mean()
    df['volume_ratio_10'] = df['Volume'] / df['volume_ma10']
    df['volume_ratio_20'] = df['Volume'] / df['volume_ma20']
    
    # OBV (On Balance Volume)
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    df['obv_ma10'] = df['obv'].rolling(10).mean()
    df['obv_ratio'] = df['obv'] / df['obv_ma10']
    
    # ============= PRICE ACTION =============
    # Candlestick patterns
    df['body'] = abs(df['Close'] - df['Open'])
    df['upper_shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['lower_shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['body_ratio'] = df['body'] / (df['High'] - df['Low'] + 0.001)
    
    # High-Low range
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    
    # ============= LAG FEATURES =============
    for lag in [1, 2, 3, 5]:
        df[f'return_1d_lag{lag}'] = df['return_1d'].shift(lag)
        df[f'rsi_14_lag{lag}'] = df['rsi_14'].shift(lag)
        df[f'macd_hist_lag{lag}'] = df['macd_hist'].shift(lag)
    
    # ============= TARGET =============
    # Predict next 5 days return (more signal)
    df['target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    # Drop NaN
    df = df.dropna()
    
    # Select features (exclude target and Close)
    exclude_cols = ['target', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    return df[feature_cols], df['target'], df['Close']


# ============================================================
# PART 2: DATA GENERATION
# ============================================================

def generate_training_data(tickers, period="5y"):
    """Generate training data from multiple stocks."""
    all_features = []
    all_targets = []
    
    print(f"Fetching data for {len(tickers)} stocks...")
    
    for ticker in tickers:
        try:
            features, target, prices = create_features_v2(ticker, period)
            all_features.append(features)
            all_targets.append(target)
            print(f"  {ticker}: {len(features)} samples")
        except Exception as e:
            print(f"  {ticker}: Error - {e}")
    
    X = pd.concat(all_features)
    y = pd.concat(all_targets)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Class: UP={y.sum()} ({y.sum()/len(y)*100:.1f}%), DOWN={len(y)-y.sum()} ({(1-y.sum()/len(y))*100:.1f}%)")
    
    return X, y


# ============================================================
# PART 3: DEEP MODEL
# ============================================================

class DeepStockPredictor(nn.Module):
    """
    Deeper neural network with residual connections.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=4, dropout=0.3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Hidden layers with residual connections
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Output
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        
        return self.output(x)


class TransformerStockPredictor(nn.Module):
    """
    Transformer-based model for stock prediction.
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding (simple)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, features)
        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch, 1, features)
        
        # Embed
        x = self.input_embedding(x)  # (batch, 1, d_model)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :1, :]
        
        # Transform
        x = self.transformer(x)  # (batch, 1, d_model)
        
        # Output
        x = x.squeeze(1)  # (batch, d_model)
        return self.output(x)


# ============================================================
# PART 4: TRAINING WITH MORE RESOURCES
# ============================================================

def train_model_v2(X, y, model_type='deep', epochs=100, batch_size=512, lr=1e-4):
    """
    Train with more resources and better hyperparameters.
    """
    # Time-series split (last 20% for validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # To tensors
    train_ds = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values).reshape(-1, 1)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val.values).reshape(-1, 1)
    )
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size)
    
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if model_type == 'transformer':
        model = TransformerStockPredictor(input_dim=X.shape[1]).to(device)
    else:
        model = DeepStockPredictor(input_dim=X.shape[1], hidden_dim=256, num_layers=6, dropout=0.3).to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"\nTraining {model_type} model...")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
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
                
                predicted = (torch.sigmoid(pred) > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        val_acc = correct / total
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2%}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    
    return model, scaler, best_val_acc


# ============================================================
# PART 5: BACKTEST
# ============================================================

def backtest(ticker, model, scaler, period="2y"):
    """Run backtest on a stock."""
    features, target, prices = create_features_v2(ticker, period)
    
    X = scaler.transform(features.values)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        probs = torch.sigmoid(model(X_tensor)).cpu().numpy().flatten()
    
    # Predictions
    predictions = (probs > 0.5).astype(int)
    
    # Returns
    returns = features['return_5d'].values  # 5-day returns
    
    # Strategy returns
    strategy_returns = returns * (predictions * 2 - 1)
    
    # Cumulative
    strategy_cum = (1 + strategy_returns).cumprod()
    buyhold_cum = (1 + returns).cumprod()
    
    # Metrics
    strategy_total = strategy_cum[-1] - 1
    buyhold_total = buyhold_cum[-1] - 1
    
    strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252/5)
    buyhold_sharpe = returns.mean() / returns.std() * np.sqrt(252/5)
    
    print(f"\n{ticker}:")
    print(f"  ML Strategy: {strategy_total*100:+.1f}% (Sharpe: {strategy_sharpe:.2f})")
    print(f"  Buy & Hold:  {buyhold_total*100:+.1f}% (Sharpe: {buyhold_sharpe:.2f})")
    
    return {
        'strategy': strategy_total,
        'buyhold': buyhold_total,
        'strategy_sharpe': strategy_sharpe,
        'buyhold_sharpe': buyhold_sharpe
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # More stocks for more data
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 
        'V', 'WMT', 'JNJ', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'NFLX',
        'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO', 'ORCL', 'IBM'
    ]
    
    print("="*60)
    print("GENERATING TRAINING DATA")
    print("="*60)
    X, y = generate_training_data(TICKERS, period="5y")
    
    print("\n" + "="*60)
    print("TRAINING DEEP MODEL")
    print("="*60)
    model, scaler, best_acc = train_model_v2(X, y, model_type='deep', epochs=100)
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    for ticker in ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL']:
        try:
            backtest(ticker, model, scaler)
        except Exception as e:
            print(f"{ticker}: Error - {e}")
    
    print("\n✅ Done!")
