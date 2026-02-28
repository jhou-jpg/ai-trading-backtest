"""
Trade Executor Module
Executes trades via Interactive Brokers API
"""

from typing import Dict, List, Optional
from datetime import datetime
import json
import os


# ============================================================
# ORDER TYPES
# ============================================================

class OrderType:
    """Available order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide:
    """Order sides."""
    BUY = "BUY"
    SELL = "SELL"


class OrderTimeInForce:
    """Time in force options."""
    DAY = "DAY"           # Good for the day only
    GTC = "GTC"           # Good until cancelled
    IOC = "IOC"           # Immediate or cancel
    FOK = "FOK"           # Fill or kill


# ============================================================
# TRADE EXECUTOR
# ============================================================

class TradeExecutor:
    """
    Executes trades via IBKR API or paper trading for testing.
    """
    
    def __init__(self, use_ibkr: bool = True, paper_trading: bool = True):
        """
        Args:
            use_ibkr: Whether to use real IBKR connection
            paper_trading: Use paper trading account (True) or live (False)
        """
        self.use_ibkr = use_ibkr
        self.paper_trading = paper_trading
        self.port = 7496 if paper_trading else 7497  # Paper=7496, Live=7497
        
        self.ib = None
        self.connected = False
        
        # Trade log
        self.trade_history: List[Dict] = []
        
        # Pending orders
        self.pending_orders: List[Dict] = []
        
        # Open positions (for paper trading simulation)
        self.positions: Dict[str, Dict] = {}
        
        # Load trade history if exists
        self._load_history()
    
    def connect(self, host: str = "127.0.0.1", client_id: int = 1) -> bool:
        """
        Connect to IBKR TWS.
        
        Args:
            host: TWS host address
            client_id: Client ID for connection
        
        Returns:
            True if connected successfully
        """
        if not self.use_ibkr:
            print("[*] IBKR disabled, using paper trading mode")
            self.connected = True
            return True
        
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect(host, self.port, clientId=client_id)
            self.connected = True
            account_type = "paper" if self.paper_trading else "live"
            print(f"[+] Connected to IBKR ({account_type}) on port {self.port}")
            return True
        except Exception as e:
            print(f"[-] Connection failed: {e}")
            print("    Falling back to paper trading mode")
            self.use_ibkr = False
            self.connected = True
            return False
    
    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self.connected:
            self.ib.disconnect()
        self.connected = False
        self._save_history()
        print("[*] Disconnected")
    
    def is_connected(self) -> bool:
        """Check if connected."""
        if self.use_ibkr and self.ib:
            return self.ib.isConnected()
        return self.connected
    
    # ==================== ORDER EXECUTION ====================
    
    def place_market_order(self, ticker: str, shares: int, 
                          action: str = "BUY") -> Dict:
        """
        Place a market order.
        
        Args:
            ticker: Stock symbol
            shares: Number of shares
            action: 'BUY' or 'SELL'
        
        Returns:
            Order result dictionary
        """
        order_id = self._generate_order_id()
        
        if not self.is_connected():
            return {
                'success': False,
                'error': 'Not connected'
            }
        
        try:
            if self.use_ibkr:
                return self._place_ibkr_order(ticker, shares, action, "MARKET")
            else:
                return self._place_paper_order(ticker, shares, action, "MARKET")
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker,
                'action': action,
                'shares': shares
            }
    
    def place_limit_order(self, ticker: str, shares: int, 
                         limit_price: float, action: str = "BUY") -> Dict:
        """
        Place a limit order.
        
        Args:
            ticker: Stock symbol
            shares: Number of shares
            limit_price: Limit price
            action: 'BUY' or 'SELL'
        
        Returns:
            Order result dictionary
        """
        if not self.is_connected():
            return {
                'success': False,
                'error': 'Not connected'
            }
        
        try:
            if self.use_ibkr:
                return self._place_ibkr_order(ticker, shares, action, "LIMIT", 
                                             limit_price=limit_price)
            else:
                return self._place_paper_order(ticker, shares, action, "LIMIT",
                                              limit_price=limit_price)
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker
            }
    
    def place_stop_order(self, ticker: str, shares: int,
                        stop_price: float, action: str = "SELL") -> Dict:
        """
        Place a stop order (stop loss).
        
        Args:
            ticker: Stock symbol
            shares: Number of shares
            stop_price: Stop price
            action: 'BUY' or 'SELL'
        
        Returns:
            Order result dictionary
        """
        if not self.is_connected():
            return {
                'success': False,
                'error': 'Not connected'
            }
        
        try:
            if self.use_ibkr:
                return self._place_ibkr_order(ticker, shares, action, "STOP",
                                             stop_price=stop_price)
            else:
                return self._place_paper_order(ticker, shares, action, "STOP",
                                              stop_price=stop_price)
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _place_ibkr_order(self, ticker: str, shares: int, action: str,
                         order_type: str, limit_price: float = None,
                         stop_price: float = None) -> Dict:
        """Place order via IBKR API."""
        from ib_insync import Stock, MarketOrder, LimitOrder, StopOrder
        
        # Create contract
        contract = Stock(ticker, 'SMART', 'USD')
        
        # Create order based on type
        if order_type == "MARKET":
            order = MarketOrder(action, shares)
        elif order_type == "LIMIT" and limit_price:
            order = LimitOrder(action, shares, limit_price)
        elif order_type == "STOP" and stop_price:
            order = StopOrder(action, shares, stop_price)
        else:
            order = MarketOrder(action, shares)
        
        # Place order
        trade = self.ib.placeOrder(contract, order)
        
        # Wait for execution (with timeout)
        self.ib.waitOnUpdate(timeout=10)
        
        # Check result
        if trade.orderStatus.status == 'Filled':
            result = {
                'success': True,
                'order_id': trade.order.orderId,
                'ticker': ticker,
                'action': action,
                'shares': shares,
                'filled_price': trade.orderStatus.avgFillPrice,
                'timestamp': datetime.now().isoformat()
            }
        else:
            result = {
                'success': False,
                'order_id': trade.order.orderId,
                'status': trade.orderStatus.status,
                'ticker': ticker,
                'error': f"Order {trade.orderStatus.status}"
            }
        
        self._log_trade(result)
        return result
    
    def _place_paper_order(self, ticker: str, shares: int, action: str,
                          order_type: str, limit_price: float = None,
                          stop_price: float = None) -> Dict:
        """Simulate paper trade."""
        # Get current price (simulated)
        current_price = self._get_simulated_price(ticker)
        
        # For market orders, execute at current price
        # For limit orders, only execute if price is met
        # For stop orders, track for later execution
        
        executed_price = current_price
        
        if order_type == "LIMIT" and limit_price:
            if action == "BUY" and current_price > limit_price:
                return {
                    'success': False,
                    'ticker': ticker,
                    'reason': f'Price {current_price} above limit {limit_price}'
                }
            elif action == "SELL" and current_price < limit_price:
                return {
                    'success': False,
                    'ticker': ticker,
                    'reason': f'Price {current_price} below limit {limit_price}'
                }
            executed_price = limit_price
        
        # Execute the trade
        if action == "BUY":
            # Update position
            if ticker in self.positions:
                existing = self.positions[ticker]
                total_shares = existing['shares'] + shares
                avg_price = ((existing['shares'] * existing['avg_price']) + 
                           (shares * executed_price)) / total_shares
                self.positions[ticker] = {
                    'shares': total_shares,
                    'avg_price': avg_price,
                    'current_price': executed_price
                }
            else:
                self.positions[ticker] = {
                    'shares': shares,
                    'avg_price': executed_price,
                    'current_price': executed_price
                }
        elif action == "SELL":
            if ticker in self.positions:
                self.positions[ticker]['shares'] -= shares
                if self.positions[ticker]['shares'] <= 0:
                    del self.positions[ticker]
        
        result = {
            'success': True,
            'order_id': self._generate_order_id(),
            'ticker': ticker,
            'action': action,
            'shares': shares,
            'filled_price': executed_price,
            'order_type': order_type,
            'timestamp': datetime.now().isoformat(),
            'mode': 'PAPER'
        }
        
        self._log_trade(result)
        return result
    
    # ==================== POSITION MANAGEMENT ====================
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current open positions."""
        if self.use_ibkr and self.connected:
            try:
                positions = {}
                for pos in self.ib.positions():
                    positions[pos.contract.symbol] = {
                        'shares': pos.position,
                        'avg_price': pos.avgCost,
                        'current_price': self._get_last_price(pos.contract.symbol)
                    }
                return positions
            except:
                pass
        
        return self.positions
    
    def has_position(self, ticker: str) -> bool:
        """Check if we have an open position in a ticker."""
        positions = self.get_positions()
        return ticker in positions and positions[ticker]['shares'] > 0
    
    def get_position(self, ticker: str) -> Optional[Dict]:
        """Get position for a specific ticker."""
        positions = self.get_positions()
        return positions.get(ticker)
    
    # ==================== ORDER MANAGEMENT ====================
    
    def cancel_order(self, order_id: int) -> Dict:
        """Cancel an order."""
        if self.use_ibkr and self.connected:
            try:
                for order in self.ib.openOrders():
                    if order.orderId == order_id:
                        self.ib.cancelOrder(order)
                        return {'success': True, 'order_id': order_id}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        # Check pending orders (paper)
        for i, order in enumerate(self.pending_orders):
            if order.get('order_id') == order_id:
                self.pending_orders.pop(i)
                return {'success': True, 'order_id': order_id}
        
        return {'success': False, 'error': 'Order not found'}
    
    def cancel_all_orders(self) -> Dict:
        """Cancel all open orders."""
        if self.use_ibkr and self.connected:
            try:
                self.ib.cancelAllOrders()
                return {'success': True}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        self.pending_orders = []
        return {'success': True}
    
    # ==================== UTILITY METHODS ====================
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        import uuid
        return f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    
    def _get_simulated_price(self, ticker: str) -> float:
        """Get simulated current price (would fetch from IB in real mode)."""
        # In paper mode, use a mock price
        # In real mode, would fetch live price
        import yfinance as yf
        try:
            data = yf.download(ticker, period="1d", progress=False)
            if len(data) > 0:
                if isinstance(data['Close'], pd.DataFrame):
                    return float(data['Close'].iloc[-1, 0])
                return float(data['Close'].iloc[-1])
        except:
            pass
        
        # Fallback: return a mock price
        mock_prices = {
            'AAPL': 175.0, 'MSFT': 400.0, 'NVDA': 500.0,
            'GOOGL': 140.0, 'TSLA': 200.0, 'AMD': 120.0
        }
        return mock_prices.get(ticker, 100.0)
    
    def _get_last_price(self, ticker: str) -> float:
        """Get last traded price."""
        return self._get_simulated_price(ticker)
    
    def _log_trade(self, trade: Dict):
        """Log trade to history."""
        self.trade_history.append(trade)
        self._save_history()
    
    def _load_history(self):
        """Load trade history from file."""
        try:
            if os.path.exists('trade_history.json'):
                with open('trade_history.json', 'r') as f:
                    self.trade_history = json.load(f)
        except:
            self.trade_history = []
    
    def _save_history(self):
        """Save trade history to file."""
        try:
            with open('trade_history.json', 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except:
            pass
    
    def get_trade_history(self, limit: int = None) -> List[Dict]:
        """Get trade history."""
        if limit:
            return self.trade_history[-limit:]
        return self.trade_history


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    import pandas as pd
    
    print("="*60)
    print("TRADE EXECUTOR TEST")
    print("="*60)
    
    # Create executor (use paper trading for testing)
    executor = TradeExecutor(use_ibkr=False, paper_trading=True)
    executor.connect()
    
    # Test market order
    print("\n--- Test: Buy AAPL ---")
    result = executor.place_market_order("AAPL", 10, "BUY")
    print(f"Result: {result}")
    
    # Test another buy
    print("\n--- Test: Buy MSFT ---")
    result = executor.place_market_order("MSFT", 5, "BUY")
    print(f"Result: {result}")
    
    # Check positions
    print("\n--- Current Positions ---")
    positions = executor.get_positions()
    for ticker, pos in positions.items():
        print(f"  {ticker}: {pos['shares']} shares @ ${pos['avg_price']:.2f}")
    
    # Test sell
    print("\n--- Test: Sell AAPL ---")
    result = executor.place_market_order("AAPL", 5, "SELL")
    print(f"Result: {result}")
    
    # Check positions again
    print("\n--- Updated Positions ---")
    positions = executor.get_positions()
    for ticker, pos in positions.items():
        print(f"  {ticker}: {pos['shares']} shares @ ${pos['avg_price']:.2f}")
    
    # Check trade history
    print("\n--- Trade History ---")
    history = executor.get_trade_history()
    for trade in history:
        print(f"  {trade['action']} {trade['shares']} {trade['ticker']} @ ${trade['filled_price']:.2f}")
    
    executor.disconnect()