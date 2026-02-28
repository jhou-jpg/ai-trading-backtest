"""
ML Options Pricing Model - Ready for Kaggle Training
=====================================================
Neural network to price options and estimate Greeks.

Usage on Kaggle:
1. Upload this notebook
2. Run cells in order
3. Model trains on GPU in ~2 minutes
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# BLACK-SCHOLES (Ground Truth)
# ============================================================

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta_common = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    theta = (theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365 if option_type == 'call' else (theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}


# ============================================================
# GENERATE TRAINING DATA
# ============================================================

def generate_data(n_samples=100000):
    np.random.seed(42)
    data = []
    
    for i in range(n_samples):
        S = np.random.uniform(10, 500)
        K = S * np.random.uniform(0.5, 1.5)
        T = np.random.uniform(0.01, 2)
        r = np.random.uniform(0.01, 0.10)
        sigma = np.random.uniform(0.05, 1.0)
        opt_type = np.random.choice([0, 1])
        
        price = black_scholes_price(S, K, T, r, sigma, 'call' if opt_type == 1 else 'put')
        greeks = calculate_greeks(S, K, T, r, sigma, 'call' if opt_type == 1 else 'put')
        
        data.append([S, K, T, r, sigma, opt_type, price, greeks['delta'], greeks['gamma'], greeks['vega'], greeks['theta']])
    
    cols = ['S', 'K', 'T', 'r', 'sigma', 'option_type', 'price', 'delta', 'gamma', 'vega', 'theta']
    return pd.DataFrame(data, columns=cols)


# ============================================================
# MODEL
# ============================================================

class OptionsNN(nn.Module):
    def __init__(self, input_dim=6, hidden=[128, 64, 32]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.1)]
            prev = h
        self.shared = nn.Sequential(*layers)
        self.price = nn.Linear(prev, 1)
        self.delta = nn.Linear(prev, 1)
        self.gamma = nn.Linear(prev, 1)
        self.vega = nn.Linear(prev, 1)
        self.theta = nn.Linear(prev, 1)
    
    def forward(self, x):
        f = self.shared(x)
        return self.price(f), self.delta(f), self.gamma(f), self.vega(f), self.theta(f)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train(df, epochs=50, batch_size=1024, lr=1e-3):
    # Prepare data
    X = df[['S', 'K', 'T', 'r', 'sigma', 'option_type']].values
    y = df[['price', 'delta', 'gamma', 'vega', 'theta']].values
    
    # Normalize
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)
    
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OptionsNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        for x, y_batch in train_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = sum(criterion(pred[i], y_batch[:, i:i+1]) for i in range(5))
            loss.backward()
            opt.step()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y_batch in val_loader:
                x, y_batch = x.to(device), y_batch.to(device)
                pred = model(x)
                val_loss += sum(criterion(pred[i], y_batch[:, i:i+1]) for i in range(5)).item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Save scalers
    return model, X_mean, X_std, y_mean, y_std


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Generating training data...")
    df = generate_data(100000)
    print(f"Generated {len(df)} samples")
    
    print("\nTraining model...")
    model, X_mean, X_std, y_mean, y_std = train(df)
    
    print("\nTesting prediction...")
    # Test: AAPL $175, strike $180, 30 days, 5% vol
    S, K, T, r, sigma = 175, 180, 30/365, 0.05, 0.20
    
    # Black-Scholes (ground truth)
    bs_price = black_scholes_price(S, K, T, r, sigma, 'call')
    bs_greeks = calculate_greeks(S, K, T, r, sigma, 'call')
    
    # Normalize input
    x = np.array([[S, K, T, r, sigma, 1]])
    x = (x - X_mean) / X_std
    
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(x).to('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Denormalize
    pred_price = pred[0].item() * y_std[0] + y_mean[0]
    
    print(f"\nTest: S={S}, K={K}, T={30} days, σ={sigma}")
    print(f"Black-Scholes Price: ${bs_price:.2f}")
    print(f"Model Prediction: ${pred_price:.2f}")
    print(f"Error: ${abs(bs_price - pred_price):.2f} ({abs(bs_price - pred_price)/bs_price*100:.2f}%)")
    
    print("\n✅ Model ready for Kaggle training!")