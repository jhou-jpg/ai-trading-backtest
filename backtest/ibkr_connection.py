"""
IBKR Connection Module
Connect to Interactive Brokers Trader Workstation (TWS) API
"""

from ib_insync import IB, Stock, Forex, Option, Future
import asyncio
from typing import Dict, List, Optional, Any
import pandas as pd

# Default connection settings
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT_PAPER = 7496  # Paper trading
DEFAULT_PORT_LIVE = 7497   # Live trading
DEFAULT_CLIENT_ID = 1


class IBKRConnection:
    """Interactive Brokers connection manager."""
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT_PAPER, 
                 client_id: int = DEFAULT_CLIENT_ID):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.connected = False
    
    def connect(self, timeout: int = 10) -> bool:
        """Connect to TWS."""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            print(f"[+] Connected to IBKR on port {self.port}")
            return True
        except Exception as e:
            print(f"[-] Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from TWS."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            print("[*] Disconnected from IBKR")
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.connected and self.ib.isConnected()
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        if not self.is_connected():
            return {"error": "Not connected"}
        
        account = self.ib.accountValues()
        # Parse into dict
        data = {}
        for kv in account:
            data[kv.tag] = kv.value
        return data
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        if not self.is_connected():
            return []
        
        positions = self.ib.positions()
        pos_list = []
        for pos in positions:
            pos_list.append({
                'symbol': pos.contract.symbol,
                'secType': pos.contract.secType,
                'position': pos.position,
                'avgCost': pos.avgCost
            })
        return pos_list
    
    def get_portfolio(self) -> List[Dict]:
        """Get portfolio items."""
        if not self.is_connected():
            return []
        
        portfolio = self.ib.portfolio()
        port_list = []
        for item in portfolio:
            port_list.append({
                'symbol': item.contract.symbol,
                'position': item.position,
                'marketValue': item.marketValue,
                'avgCost': item.avgCost,
                'unrealizedPNL': item.unrealizedPNL,
                'realizedPNL': item.realizedPNL
            })
        return port_list
    
    def get_contract_details(self, symbol: str, secType: str = 'STK') -> Dict:
        """Get contract details for a symbol."""
        if not self.is_connected():
            return {"error": "Not connected"}
        
        # Create contract
        if secType == 'STK':
            contract = Stock(symbol, 'SMART', 'USD')
        elif secType == 'Forex':
            contract = Forex(symbol)
        else:
            contract = Stock(symbol, 'SMART', 'USD')
        
        # Get details
        details = self.ib.reqContractDetails(contract)
        if details:
            d = details[0]
            return {
                'symbol': d.contract.symbol,
                'secType': d.contract.secType,
                'strike': d.contract.strike,
                'expiry': d.contract.lastTradeDateOrContractMonth,
                'right': d.contract.right,
                'multiplier': d.contract.multiplier,
                'marketName': d.marketName,
                'minTick': d.minTick,
                'orderTypes': d.orderTypes,
                'validExchanges': d.validExchanges
            }
        return {}
    
    def place_market_order(self, symbol: str, quantity: int, action: str) -> Optional[Any]:
        """
        Place a market order.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            quantity: Number of shares
            action: 'BUY' or 'SELL'
        
        Returns:
            Order execution result
        """
        if not self.is_connected():
            print("[-] Not connected")
            return None
        
        # Create contract
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Create order
        from ib_insync import MarketOrder
        order = MarketOrder(action, quantity)
        
        # Place order
        trade = self.ib.placeOrder(contract, order)
        
        # Wait for execution
        self.ib.waitOnUpdate()
        
        if trade.orderStatus.status == 'Filled':
            print(f"[+] {action} {quantity} {symbol} filled @ {trade.orderStatus.avgFillPrice}")
            return trade
        else:
            print(f"[-] Order status: {trade.orderStatus.status}")
            return trade
    
    def get_market_data(self, symbols: List[str]) -> Dict:
        """Get real-time market data for symbols."""
        if not self.is_connected():
            return {"error": "Not connected"}
        
        contracts = [Stock(sym, 'SMART', 'USD') for sym in symbols]
        tickers = self.ib.reqTickers(*contracts)
        
        data = {}
        for ticker in tickers:
            data[ticker.contract.symbol] = {
                'bid': ticker.bid,
                'ask': ticker.ask,
                'last': ticker.last,
                'close': ticker.close,
                'modelGreeks': ticker.modelGreeks if ticker.modelGreeks else None
            }
        return data
    
    def get_historical_data(self, symbol: str, duration: str = '1 Y', 
                           barSize: str = '1 day') -> pd.DataFrame:
        """Get historical data."""
        if not self.is_connected():
            return pd.DataFrame()
        
        contract = Stock(symbol, 'SMART', 'USD')
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=barSize,
            whatToShow='ADJUSTED_LAST',
            useRTH=True
        )
        
        if bars:
            df = pd.DataFrame(bars)
            df['date'] = df['date'].dt.tz_localize(None)
            return df
        return pd.DataFrame()
    
    def cancel_all_orders(self):
        """Cancel all open orders."""
        if not self.is_connected():
            return
        for order in self.ib.openOrders():
            self.ib.cancelOrder(order)


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Create connection (use port 7496 for paper, 7497 for live)
    ib = IBKRConnection(port=DEFAULT_PORT_PAPER)
    
    # Connect
    if ib.connect():
        print("\n--- Account Info ---")
        account = ib.get_account_info()
        print(f"Cash: {account.get('NetLiquidation', 'N/A')}")
        
        print("\n--- Positions ---")
        positions = ib.get_positions()
        for pos in positions:
            print(f"{pos['symbol']}: {pos['position']} shares @ ${pos['avgCost']:.2f}")
        
        print("\n--- Market Data (AAPL, MSFT) ---")
        data = ib.get_market_data(['AAPL', 'MSFT'])
        for sym, info in data.items():
            print(f"{sym}: Bid={info['bid']}, Ask={info['ask']}, Last={info['last']}")
        
        # Disconnect
        ib.disconnect()
    else:
        print("[-] Make sure TWS is running and API access is enabled!")
        print("    1. Open TWS")
        print("    2. Go to File -> Settings -> API")
        print("    3. Enable 'Enable ActiveX and Socket Clients'")
        print("    4. Note the port (7496 for paper, 7497 for live)")