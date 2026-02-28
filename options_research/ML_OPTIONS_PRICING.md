# Options Pricing with ML - Research Notes

## Traditional Methods

### Black-Scholes Model
- Assumes log-normal distribution of stock prices
- Closed-form solution for European options
- Limitations: Assumes constant volatility, no transaction costs

### Binomial/Trinomial Trees
- Discretize price movements over time
- More flexible but computationally expensive

### Monte Carlo Simulation
- Simulate thousands of price paths
- Good for complex payoffs but slow

---

## ML Approaches to Options Pricing

### 1. Deep Learning Approximation
**Papers:**
- "Deep Learning for Generic Object Detection" (not relevant)
- Look for: "Neural Network Black-Scholes", "Deep Options Pricing"

**Approach:**
- Train neural network to approximate Black-Scholes or Monte Carlo prices
- Input: S, K, T, r, σ (spot, strike, time, rate, volatility)
- Output: Option price

**Pros:** Fast inference after training
**Cons:** Needs lots of training data

---

### 2. **Masked Learning / Masked Attention (Innovative!)**

**What is masking in ML?**
- Used in transformer models (BERT, GPT) to hide/mask certain inputs
- Model learns to predict missing or masked tokens

**Could this apply to options pricing?**

**Hypothesis:** Use masked attention to handle:
1. **Missing/implied volatility** - Mask the volatility input, let model learn to infer it
2. **Greeks estimation** - Mask specific Greek inputs to predict them
3. **Incomplete market data** - Use attention to focus on available data

**Related papers to explore:**
- "Attention is All You Need" (Transformer)
- "BERT: Pre-training of Deep Bidirectional Transformers"
- "TabTransformer: Tabular Data Modeling with Transformers"

**Architecture idea:**
```
Input Features: [S, K, T, r, σ, volume, IV, etc.]
    ↓
Embedding Layer
    ↓
Transformer Encoder (with masked attention)
    ↓
Output: [Price, Delta, Gamma, Vega, etc.]
```

---

### 3. **Reinforcement Learning for Options**

**Approach:**
- Treat options trading as RL problem
- Agent learns optimal exercise/hedging strategies

**Key Papers:**
- "TradingAgents: Multi-Agent LLM Financial Trading" (found in search)
- "Deep Reinforcement Learning in Options Market Making"

---

### 4. **Graph Neural Networks for Options Chains**

**Idea:**
- Options on same underlying form a "chain" or graph
- GNN can learn relationships between strikes/expirations

---

### 5. **Uncertainty Quantification**

**Use:**
- Dropout at inference time (MC Dropout)
- Ensemble methods
- Bayesian neural networks

**Benefit:** 
- Get confidence intervals on option prices
- Useful for risk management

---

## Does Masking Make Sense?

### **Yes, here's why:**

1. **Handling missing data** - Real market data has gaps (no IV for all strikes)
2. **Learning implicit relationships** - Attention can learn vol surface structure
3. **Greeks as masks** - Mask specific inputs to predict Greeks
4. **Robustness** - Training with masked inputs improves generalization

### **Implementation Ideas:**

#### Option 1: Volatility Imputation
```
Input: [S, K, T, r, market_price]
Mask: [0, 0, 0, 0, 1]  (mask the IV)
Output: Implied σ

Then use σ to price option
```

#### Option 2: Full Pricing Model
```
Input: [S, K, T, r, σ] with random masking during training
Transformer with causal masking
Output: Price
```

#### Option 3: Greeks via Masking
```
Train separate heads:
- Price head
- Delta head (mask S, predict ∂V/∂S)
- Gamma head (mask S twice, predict ∂²V/∂S²)
```

---

## Recommended Architecture

```
┌─────────────────────────────────────────────────┐
│              Options Pricing Model              │
├─────────────────────────────────────────────────┤
│  Input:                                         │
│  - Spot (S)                                     │
│  - Strike (K)                                   │
│  - Time to expiry (T)                           │
│  - Risk-free rate (r)                           │
│  - Volatility (σ)                               │
│  - Option type (call/put)                       │
│  - Volume / Open Interest                       │
│  - IV from market                               │
├─────────────────────────────────────────────────┤
│  Layer 1: Embedding (numerical → vectors)       │
│  Layer 2: Positional encoding (for time series) │
│  Layer 3-6: Transformer blocks                  │
│  Layer 7: Output heads                          │
│    - Price prediction                           │
│    - Delta (1st derivative)                     │
│    - Gamma (2nd derivative)                     │
│    - Vega (vol sensitivity)                     │
│    - Confidence interval                        │
└─────────────────────────────────────────────────┘
```

---

## Data Sources for Training

1. **Historical options data**
   - CBOE Data Shop
   - OptionMetrics
   - Free: Yahoo Finance options chains

2. **Synthetic data**
   - Generate via Black-Scholes
   - Add noise for realism

3. **Real-time (for production)**
   - Stream from IBKR API

---

## Libraries to Use

- **PyTorch** / **PyTorch Lightning** - Training
- **Transformers** (HuggingFace) - For transformer architecture
- **FinRL** - Reinforcement learning for finance
- **QuantLib** - Traditional pricing benchmarks

---

## Next Steps

1. [ ] Gather historical options data
2. [ ] Build baseline Black-Scholes model
3. [ ] Create simple NN pricing model
4. [ ] Add transformer with masking
5. [ ] Train and benchmark
6. [ ] Add to trading system

---

## Key Papers to Read

1. "Neural Network Black-Scholes" - Early ML options pricing
2. "Deep Learning for Volatility Surface" - IV modeling
3. "Transformer Networks for Financial Time Series"
4. "Monte Carlo Options Pricing with Deep Learning"

---

*Last updated: Feb 2026*
*Status: Researching*