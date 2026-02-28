"""
Kalshi Integration - Safe Version
=================================
"""

import requests
import hashlib
import hmac
import time
import json

# ============================================================
# YOUR CREDENTIALS - Replace these!
# ============================================================
API_KEY = "6f1d007c-3933-40bd-b570-bb7806f4c8ac"
API_SECRET = "ks_your_secret_here"  # Replace with your secret

BASE_URL = "https://api.kalshi.com"

def sign_request(method, path, body=""):
    timestamp = str(int(time.time()))
    msg = timestamp + method + path + body
    signature = hmac.new(API_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()
    return {
        "KALSHI-KEY": API_KEY,
        "KALSHI-SIGNATURE": signature,
        "KALSHI-TIMESTAMP": timestamp
    }

def get_markets():
    headers = sign_request("GET", "/v1/markets")
    headers["Content-Type"] = "application/json"
    resp = requests.get(BASE_URL + "/v1/markets", headers=headers)
    return resp.json()

def get_market(ticker):
    headers = sign_request("GET", f"/v1/markets/{ticker}")
    headers["Content-Type"] = "application/json"
    resp = requests.get(BASE_URL + f"/v1/markets/{ticker}", headers=headers)
    return resp.json()

def place_order(ticker, side, price, count):
    path = "/v1/orders"
    order = {"market_ticker": ticker, "side": side, "price": price, "count": count}
    body = json.dumps(order)
    headers = sign_request("POST", path, body)
    headers["Content-Type"] = "application/json"
    resp = requests.post(BASE_URL + path, headers=headers, data=body)
    return resp.json()

# Test
if __name__ == "__main__":
    print("Testing Kalshi connection...")
    
    # Get markets
    markets = get_markets()
    print(f"\nGot {len(markets.get('markets', []))} markets")
    
    # Find BTC markets
    btc_markets = [m for m in markets.get('markets', []) if 'BTC' in m.get('ticker', '')]
    print(f"\nBTC markets found: {len(btc_markets)}")
    for m in btc_markets[:5]:
        print(f"  {m.get('ticker')}: {m.get('title')}")