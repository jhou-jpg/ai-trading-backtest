# ML Options Pricing Model - Kaggle Setup

## Quick Start

1. **Go to Kaggle**: https://www.kaggle.com/new-notebook
2. **Upload this file**: Copy `options_model.py` content to a new notebook
3. **Enable GPU**: Settings → Accelerator → GPU (P100 or T4)
4. **Run**: Click "Run All"

## What This Does

1. **Generates 100K synthetic options** using Black-Scholes
2. **Trains neural network** to predict:
   - Option Price
   - Delta (price sensitivity)
   - Gamma (delta sensitivity)
   - Vega (vol sensitivity)
   - Theta (time decay)
3. **Validates** against Black-Scholes

## Expected Results

- Training time: ~2 minutes on GPU
- Price prediction error: < 1%
- Ready for real options data after training

## To Test with Real Data

After training, you can fetch real options from Yahoo Finance:

```python
import yfinance as yf

# Get options chain for AAPL
aapl = yf.Ticker("AAPL")
calls = aapl.option_calls()

# Use model to predict prices
```

## Files

- `options_model.py` - Main training code (this file)
- `test.txt` - Placeholder

## Next Steps

1. Run on Kaggle with GPU
2. Add real options data from Yahoo Finance
3. Add transformer architecture with masking
4. Deploy in trading system