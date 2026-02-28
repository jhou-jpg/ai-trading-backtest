"""
Kalshi Prediction Markets Integration
=====================================
Connect your ML predictions to live trading on Kalshi.

Kalshi is a prediction market where you can trade:
- Crypto (BTC, ETH ranges)
- Elections
- Economics (CPI, unemployment)
- Weather

SETUP:
1. Go to kalshi.com and create account
2. Get API credentials from Settings -> API
3. Use the API key and secret (NOT private key)
"""

import requests
import hashlib
import hmac
import time
import json
from typing import Dict, List, Optional

# ============================================================
# KALSHI API CLIENT
# ============================================================

class KalshiClient:
    """Client for Kalshi trading API."""
    
    BASE_URL = "https://api.kalshi.com"
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
    
    def _sign_request(self, method: str, path: str, body: str = "") -> Dict:
        """Sign request with HMAC-SHA256."""
        timestamp = str(int(time.time()))
        
        # Create signature
        msg = timestamp + method + path + body
        signature = hmac.new(
            self.api_secret.encode(),
            msg.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "KALSHI-KEY": self.api_key,
            "KALSHI-SIGNATURE": signature,
            "KALSHI-TIMESTAMP": timestamp
        }
    
    def get_markets(self, ticker: str = None) -> List[Dict]:
        """Get available markets."""
        path = "/v1/markets"
        if ticker:
            path += f"?ticker={ticker}"
        
        headers = self._sign_request("GET", path)
        headers["Content-Type"] = "application/json"
        
        resp = requests.get(self.BASE_URL + path, headers=headers)
        return resp.json()
    
    def get_market(self, market_ticker: str) -> Dict:
        """Get specific market details."""
        path = f"/v1/markets/{market_ticker}"
        headers = self._sign_request("GET", path)
        headers["Content-Type"] = "application/json"
        
        resp = requests.get(self.BASE_URL + path, headers=headers)
        return resp.json()
    
    def get_order_book(self, market_ticker: str) -> Dict:
        """Get order book for a market."""
        path = f"/v1/markets/{market_ticker}/order_book"
        headers = self._sign_request("GET", path)
        
        resp = requests.get(self.BASE_URL + path, headers=headers)
        return resp.json()
    
    def place_order(self, market_ticker: str, side: str, price: float, 
                   count: int = 1) -> Dict:
        """Place an order.
        
        Args:
            market_ticker: Market ticker (e.g., "BTC-24DEC31-85000")
            side: "buy" or "sell"
            price: Price (0-100)
            count: Number of contracts
        """
        path = "/v1/orders"
        
        order = {
            "market_ticker": market_ticker,
            "side": side,
            "price": price,
            "count": count
        }
        
        body = json.dumps(order)
        headers = self._sign_request("POST", path, body)
        headers["Content-Type"] = "application/json"
        
        resp = requests.post(self.BASE_URL + path, headers=headers, data=body)
        return resp.json()
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        path = "/v1/positions"
        headers = self._sign_request("GET", path)
        
        resp = requests.get(self.BASE_URL + path, headers=headers)
        return resp.json()
    
    def get_balance(self) -> Dict:
        """Get account balance."""
        path = "/v1/balance"
        headers = self._sign_request("GET", path)
        
        resp = requests.get(self.BASE_URL + path, headers=headers)
        return resp.json()


# ============================================================
# EXAMPLE: TRADING BTC RANGE
# ============================================================

def get_btc_markets(client: KalshiClient) -> List[Dict]:
    """Get all BTC-related markets."""
    # List all markets and filter for BTC
    all_markets = client.get_markets()
    
    btc_markets = []
    for market in all_markets.get("markets", []):
        if "BTC" in market.get("ticker", ""):
            btc_markets.append(market)
    
    return btc_markets


def trade_signal(client: KalshiClient, market_ticker: str, 
                signal: str, price: float = 50, count: int = 10):
    """
    Execute trade based on ML signal.
    
    Args:
        client: KalshiClient
        market_ticker: Market to trade
        signal: "BUY" or "SELL"
        price: Price to bid
        count: Number of contracts
    """
    if signal == "BUY":
        result = client.place_order(market_ticker, "buy", price, count)
    elif signal == "SELL":
        result = client.place_order(market_ticker, "sell", price, count)
    else:
        print("HOLD - No trade")
        return
    
    print(f"Order result: {result}")
    return result


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Your API credentials from kalshi.com
    # Get from: Settings -> API
    API_KEY = "6f1d007c-3933-40bd-b570-bb7806f4c8ac"
    API_SECRET = "YOUR_API_SECRET_HERE"  # <-- Replace with your secret
    
    print("="*60)
    print("KALSHI INTEGRATION")
    print("="*60)
    print("\nTo use this:")
    print("1. Create account at kalshi.com")
    print("2. Go to Settings -> API")
    print("3. Get your API Key and API Secret")
    print("4. Replace API_KEY and API_SECRET above")
    print("\nExample markets available:")
    print("- BTC-24DEC31-85000 (BTC > $85k on Dec 31)")
    print("- ETH-24DEC31-4500 (ETH > $4500 on Dec 31)")
    print("- TRUMP-2024 (Will Trump win 2024?)")
    print("- CPI-24-12 (December 2024 CPI)")
    
    # Uncomment to test:
    # client = KalshiClient(API_KEY, API_SECRET)
    # balance = client.get_balance()
    # print(f"\nYour balance: {balance}")