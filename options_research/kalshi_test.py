"""
Kalshi Test with RSA Private Key
================================
"""

import requests
import hashlib
import hmac
import time
import json

# YOUR CREDENTIALS
API_KEY = "6f1d007c-3933-40bd-b570-bb7806f4c8ac"

# Your RSA private key as the secret
RSA_PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEArPwBFSKrUUMh0PdYK4o1lDHNRKqbHryqLodf6VV2j4x1x7Pp
wDjtER/Q8hsR4z0nCVe5UoHo3IOQICd9e6hwWZGmqwRBtgReJvWicCl/A/iS0dJb
v1OjzuNLXM3xD5nIqWfBPg0JIQlgbeaPmfvbxd836T6zODFVgzM0g4HnOZYOlc94
6pDOTBZGpuoaLac3T2WnLDMQ2lEjfdlBn8VVEH1dcHebtczrMqZEeoGKT900qU5w
Dd0TDoFnVdpKg2YTU4AZ0cdHVkaBf1xlXVe0Lc0VrDzwn4BbVu/HqsGJCupKKNpU
wrq5nDibsKZbM46fUWiPSbHJoeQQ6ErljRxW4QIDAQABAoIBAAWiE+OdJD4PtRdf
I/hlq6FT+S7DbYIT5mjV0LpCUGDpU+E5IHi5gdgnvz/7FgXeim8Onm2A5yqIgEDZ
IwghWpNCynjRjoA/u9SB9ZRjIuHoDi6WA+stl+0spVoe0hBOKOTXXklygY8ZWpIJ
oDiFT/QCyl577mMl5svQduUnbaEIOQzmVwdLm2d2L/UttqFcgO/YZB6K3NYAXGWL
tHTjfMhLVFDZ0GvxDiDhh14R4trm6MZ1ofn8xGPz8njtv0mA6wqyMqvwg7enUCdx
i8SmSpqZd8FAXrZJOtRE7u5BL/1zD/ZtMxYQ8Zd4zmgADy057ZuSfQ68pwCNgnpp
q77C/M8CgYEAyzg0/+q+URnL8buyda1dzL7Rm2UNt/pKMOsXWu87rBR24HQluW2t
1pTT3FkU7vEYSlZ1ntfqyVOXvNDFVF85QVQtGXekgSKQ39gpVwjLsPxg/IpCmYk5
/hw26lByRos1wZIr+Jf2lpH/RDxoEEXKIrPB8MoNwM6MWEXqTh2AEx8CgYEA2emA
fc0bE6ZBC4YJTbneQyiFBCD+MnJJUFkMO1hmHQubGNK4c36K/W7mv76XqLr7QLnB
/mstKOBlyVzZ6yztUdVui+atuJts4gmEW9tEY9kwKLhndkGQnxBTwpnFSMerukAi
rE3J9KXvdxCgWmiRJZHdKYmIHd9qoVANNIXQVf8CgYAHxUxxhL5mR4A+7Bh02348
uAc30/NkV5PCrxqjhYZYnCe3iXlvz7vX+rTnNhjQ0jNFlGzG+CaoMCQbOjhxc3qy
/s8CrEqEDZhQlultxI5VZDEpNvg4+sBW8SlAaHcWL3iMwFQiG114gHisWUr5ZFHI
ZetCk9dWyg8fyPyepA4jrQKBgC1Jbb63RY1L2/C9JmnU2vAyF5LIGIv7XGkqWHRs
5qvaoZ0DDfpSrigFSEdJINOcKGNHN53cQEJigETc2x0Y9SkwpgzFIA1hn7tKJwvA
AfKOTIfp6vaUoa2tAvtKYcnCVVobwhj1AaeqZJ3mAq1HVgLs0X20a2lp6QslS9lC
hg2ZAoGARmIHydZPuYK/OwwMScWEkJqoyB+niNZdWqZL8BdXC74AbfgV+RKziYE5
n5z2/NA+uEgVC6lo6aNl/lk4pcQjHskUskQrEAth0szgy/nnNq2Q0/rlc2okABLL
sYs0ZJ9UZzvgGYV0DpWMMqaR4OXFrTretLcK0QTxRIfxaoi8w94=
-----END RSA PRIVATE KEY-----"""

BASE_URL = "https://api.kalshi.com"

def sign_request(method, path, body=""):
    timestamp = str(int(time.time()))
    msg = timestamp + method + path + body
    # Sign with RSA private key
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
    
    # Load private key
    from cryptography.hazmat.primitives import serialization
    private_key = serialization.load_pem_private_key(
        RSA_PRIVATE_KEY.encode(),
        password=None,
        backend=default_backend()
    )
    
    # Sign the message
    signature = private_key.sign(
        msg.encode(),
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    signature_hex = signature.hex()
    
    return {
        "KALSHI-KEY": API_KEY,
        "KALSHI-SIGNATURE": signature_hex,
        "KALSHI-TIMESTAMP": timestamp
    }

def get_markets():
    headers = sign_request("GET", "/v1/markets")
    headers["Content-Type"] = "application/json"
    resp = requests.get(BASE_URL + "/v1/markets", headers=headers)
    return resp.json()

def place_order(ticker, side, price, count):
    path = "/v1/orders"
    order = {"market_ticker": ticker, "side": side, "price": price, "count": count}
    body = json.dumps(order)
    headers = sign_request("POST", path, body)
    headers["Content-Type"] = "application/json"
    resp = requests.post(BASE_URL + path, headers=headers, data=body)
    return resp.json()

if __name__ == "__main__":
    print("Testing Kalshi connection with RSA key...")
    
    try:
        markets = get_markets()
        print(f"\nSuccess! Got {len(markets.get('markets', []))} markets")
        
        # Find BTC markets
        btc = [m for m in markets.get('markets', []) if 'BTC' in m.get('ticker', '')]
        print(f"\nBTC markets: {len(btc)}")
        for m in btc[:5]:
            print(f"  {m.get('ticker')}: {m.get('title')}")
            
    except Exception as e:
        print(f"Error: {e}")