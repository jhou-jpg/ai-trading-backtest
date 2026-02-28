"""
Trading System - Main Orchestrator
Ties together: Signal Generator + Position Sizer + Trade Executor + Risk Manager
"""

import time
from typing import Dict, List, Optional, Callable
from datetime import datetime
import pandas as pd
import yfinance as yf


# Import our modules
from signal_generator import generate_signal, scan_watchlist, filter_signals
from position_sizer import PositionSizer
from trade_executor import TradeExecutor
from risk_manager import RiskManager


# ============================================================
# TRADING SYSTEM CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    # Portfolio
    'initial_capital': 100_000,
    
    # Strategy
    'default_strategy': 'combo',
    'min_confidence': 0.50,
    
    # Position sizing
    'max_risk_per_trade': 0.02,       # 2%
    'max_position_size': 0.10,        # 10%
    'default_stop_loss_pct': 0.10,    # 10%
    'reward_risk_ratio': 2.0,         # 2:1
    
    # Execution
    'use_ibkr': False,                # Use paper trading
    'paper_trading': True,
    
    # Risk
    'max_daily_loss': 0.02,
    'max_open_positions': 10,
    
    # Scanning
    'scan_interval_seconds': 300,     # 5 minutes
}


class TradingSystem:
    """
    Main trading system that orchestrates all components.
    
    Flow:
    1. Scan watchlist for signals
    2. Validate signals with position sizer
    3. Check with risk manager
    4. Execute trades via executor
    5. Monitor open positions
    """
    
    def __init__(self, config: Dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Initialize components
        self.position_sizer = PositionSizer(
            portfolio_value=self.config['initial_capital'],
            config={
                'max_risk_per_trade': self.config['max_risk_per_trade'],
                'max_position_size': self.config['max_position_size'],
                'default_stop_loss_pct': self.config['default_stop_loss_pct'],
                'reward_risk_ratio': self.config['reward_risk_ratio'],
            }
        )
        
        self.trade_executor = TradeExecutor(
            use_ibkr=self.config['use_ibkr'],
            paper_trading=self.config['paper_trading']
        )
        
        self.risk_manager = RiskManager(
            portfolio_value=self.config['initial_capital'],
            config={
                'max_daily_loss': self.config['max_daily_loss'],
                'max_open_positions': self.config['max_open_positions'],
                'default_stop_loss_pct': self.config['default_stop_loss_pct'],
            }
        )
        
        # State
        self.watchlist: List[str] = []
        self.running = False
        self.last_scan_time = None
        
        # Callbacks
        self.on_signal = None      # Called when new signal found
        self.on_trade = None       # Called when trade executed
        self.on_risk = None        # Called when risk check fails
    
    # ==================== CONNECTION ====================
    
    def connect(self) -> bool:
        """Connect to broker."""
        return self.trade_executor.connect()
    
    def disconnect(self):
        """Disconnect from broker."""
        self.trade_executor.disconnect()
    
    # ==================== WATCHLIST ====================
    
    def set_watchlist(self, tickers: List[str]):
        """Set the watchlist to scan."""
        self.watchlist = tickers
        print(f"[*] Watchlist set: {tickers}")
    
    def add_to_watchlist(self, ticker: str):
        """Add ticker to watchlist."""
        if ticker not in self.watchlist:
            self.watchlist.append(ticker)
    
    def remove_from_watchlist(self, ticker: str):
        """Remove ticker from watchlist."""
        if ticker in self.watchlist:
            self.watchlist.remove(ticker)
    
    # ==================== SCANNING ====================
    
    def scan(self) -> Dict:
        """
        Scan watchlist for tradeable signals.
        
        Returns:
            Dictionary with scan results
        """
        self.last_scan_time = datetime.now()
        
        if not self.watchlist:
            return {
                'success': False,
                'error': 'Watchlist empty'
            }
        
        # Generate signals for all tickers
        signals = scan_watchlist(
            self.watchlist, 
            strategy=self.config['default_strategy']
        )
        
        # Filter for actionable signals
        buy_signals = filter_signals(
            signals, 
            signal_type='BUY',
            min_confidence=self.config['min_confidence']
        )
        
        sell_signals = filter_signals(
            signals,
            signal_type='SELL',
            min_confidence=0.3
        )
        
        return {
            'success': True,
            'timestamp': self.last_scan_time.isoformat(),
            'all_signals': signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_scanned': len(self.watchlist)
        }
    
    # ==================== SIGNAL PROCESSING ====================
    
    def process_buy_signal(self, signal: Dict) -> Dict:
        """
        Process a BUY signal: validate, size position, execute.
        
        Args:
            signal: Signal from scan
        
        Returns:
            Execution result
        """
        ticker = signal['ticker']
        
        # Check with risk manager first
        can_trade, reason = self.risk_manager.can_trade(ticker)
        if not can_trade:
            return {
                'success': False,
                'reason': reason,
                'ticker': ticker
            }
        
        # Get current price
        current_price = signal.get('current_price', signal['entry_price'])
        
        # Validate with position sizer
        validation = self.position_sizer.validate_position(signal, current_price)
        
        if not validation.get('valid', False):
            return {
                'success': False,
                'reason': validation.get('reason', 'Validation failed'),
                'ticker': ticker
            }
        
        # Execute trade
        result = self.trade_executor.place_market_order(
            ticker=ticker,
            shares=validation['shares'],
            action='BUY'
        )
        
        if result.get('success'):
            # Add to risk manager
            self.risk_manager.add_position(
                ticker=ticker,
                shares=validation['shares'],
                entry_price=current_price,
                stop_loss=validation['stop_loss'],
                take_profit=validation['take_profit']
            )
            
            # Callback
            if self.on_trade:
                self.on_trade({
                    'action': 'BUY',
                    'ticker': ticker,
                    'shares': validation['shares'],
                    'price': current_price,
                    'result': result
                })
        
        return result
    
    def process_sell_signal(self, signal: Dict) -> Dict:
        """
        Process a SELL signal: check if we have position, close it.
        
        Args:
            signal: Signal from scan
        
        Returns:
            Execution result
        """
        ticker = signal['ticker']
        
        # Check if we have a position
        if not self.trade_executor.has_position(ticker):
            return {
                'success': False,
                'reason': 'No position to close',
                'ticker': ticker
            }
        
        # Get current price
        current_price = signal.get('current_price', signal['entry_price'])
        
        # Get position size
        position = self.trade_executor.get_position(ticker)
        shares = position['shares']
        
        # Execute sell
        result = self.trade_executor.place_market_order(
            ticker=ticker,
            shares=shares,
            action='SELL'
        )
        
        if result.get('success'):
            # Remove from risk manager
            self.risk_manager.remove_position(
                ticker=ticker,
                exit_price=current_price,
                reason='signal_sell'
            )
            
            # Callback
            if self.on_trade:
                self.on_trade({
                    'action': 'SELL',
                    'ticker': ticker,
                    'shares': shares,
                    'price': current_price,
                    'result': result
                })
        
        return result
    
    # ==================== POSITION MONITORING ====================
    
    def check_positions(self) -> List[Dict]:
        """
        Check all open positions for risk triggers.
        
        Returns:
            List of positions with triggered risk rules
        """
        def get_price(ticker):
            try:
                data = yf.download(ticker, period="1d", progress=False)
                if len(data) > 0:
                    if isinstance(data['Close'], pd.DataFrame):
                        return float(data['Close'].iloc[-1, 0])
                    return float(data['Close'].iloc[-1])
            except:
                pass
            return 0
        
        triggered = self.risk_manager.check_all_positions(get_price)
        
        # Process triggered positions
        for item in triggered:
            ticker = item['ticker']
            signals = item['signals']
            position = item['position']
            
            if signals['stop_loss'] or signals['trailing_stop']:
                # Close at market
                current_price = get_price(ticker)
                self.trade_executor.place_market_order(ticker, position['shares'], 'SELL')
                self.risk_manager.remove_position(ticker, current_price, 'stop_loss')
                
                if self.on_risk:
                    self.on_risk({
                        'type': 'stop_loss',
                        'ticker': ticker,
                        'price': current_price
                    })
            
            elif signals['profit_take']:
                # Take profit
                current_price = get_price(ticker)
                self.trade_executor.place_market_order(ticker, position['shares'], 'SELL')
                self.risk_manager.remove_position(ticker, current_price, 'profit_take')
                
                if self.on_risk:
                    self.on_risk({
                        'type': 'profit_take',
                        'ticker': ticker,
                        'price': current_price
                    })
        
        return triggered
    
    # ==================== MAIN LOOP ====================
    
    def run_once(self):
        """Run one iteration of the trading loop."""
        # 1. Scan for signals
        scan_result = self.scan()
        
        # 2. Process buy signals
        for signal in scan_result.get('buy_signals', []):
            result = self.process_buy_signal(signal)
            if result.get('success'):
                print(f"[+] BUY {result['shares']} {signal['ticker']} @ ${result.get('filled_price', 'MARKET')}")
        
        # 3. Process sell signals
        for signal in scan_result.get('sell_signals', []):
            result = self.process_sell_signal(signal)
            if result.get('success'):
                print(f"[-] SELL {signal['ticker']} @ ${result.get('filled_price', 'MARKET')}")
        
        # 4. Check open positions
        triggered = self.check_positions()
        
        # 5. Portfolio risk check
        risk_status = self.risk_manager.check_portfolio_risk()
        
        return {
            'scan': scan_result,
            'positions_checked': len(self.risk_manager.get_positions()),
            'risk_status': risk_status
        }
    
    def start(self, iterations: int = None):
        """
        Start the trading system.
        
        Args:
            iterations: Number of iterations (None = run forever)
        """
        self.running = True
        iteration = 0
        
        print("="*60)
        print("TRADING SYSTEM STARTED")
        print("="*60)
        print(f"Capital: ${self.config['initial_capital']:,.2f}")
        print(f"Strategy: {self.config['default_strategy']}")
        print(f"Paper Trading: {self.config['paper_trading']}")
        print(f"Watchlist: {self.watchlist}")
        print("="*60)
        
        while self.running:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            try:
                result = self.run_once()
                print(f"Positions: {result['positions_checked']}")
                print(f"Risk OK: {result['risk_status']['limits_ok']}")
                
            except Exception as e:
                print(f"Error: {e}")
            
            # Check if should stop
            if iterations and iteration >= iterations:
                break
            
            # Wait before next iteration
            if self.running:
                time.sleep(self.config['scan_interval_seconds'])
        
        print("\n" + "="*60)
        print("TRADING SYSTEM STOPPED")
        print("="*60)
    
    def stop(self):
        """Stop the trading system."""
        self.running = False
    
    # ==================== STATUS ====================
    
    def get_status(self) -> Dict:
        """Get current system status."""
        positions = self.risk_manager.get_positions()
        risk = self.risk_manager.check_portfolio_risk()
        
        return {
            'running': self.running,
            'watchlist': self.watchlist,
            'positions': positions,
            'num_positions': len(positions),
            'portfolio_value': risk['portfolio_value'],
            'daily_pnl': risk['daily_pnl'],
            'risk_ok': risk['limits_ok'],
            'last_scan': self.last_scan_time.isoformat() if self.last_scan_time else None
        }


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("TRADING SYSTEM TEST")
    print("="*60)
    
    # Create system
    config = {
        'initial_capital': 100_000,
        'default_strategy': 'combo',
        'min_confidence': 0.50,
        'use_ibkr': False,       # Paper trading
        'paper_trading': True,
    }
    
    system = TradingSystem(config)
    
    # Connect
    system.connect()
    
    # Set watchlist
    system.set_watchlist(['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'TSLA'])
    
    # Run one iteration (like a scan)
    print("\n--- Running Single Scan ---")
    result = system.run_once()
    
    print(f"\nScan Results:")
    print(f"  Buy signals: {len(result['scan'].get('buy_signals', []))}")
    print(f"  Sell signals: {len(result['scan'].get('sell_signals', []))}")
    print(f"  Positions: {result['positions_checked']}")
    print(f"  Risk OK: {result['risk_status']['limits_ok']}")
    
    # Show status
    status = system.get_status()
    print(f"\nSystem Status:")
    print(f"  Portfolio: ${status['portfolio_value']:,.2f}")
    print(f"  Daily P&L: ${status['daily_pnl']:,.2f}")
    print(f"  Open Positions: {status['num_positions']}")
    
    # Check positions
    if status['positions']:
        print("\nOpen Positions:")
        for ticker, pos in status['positions'].items():
            print(f"  {ticker}: {pos['shares']} shares @ ${pos['avg_price']:.2f} "
                  f"(P&L: ${pos['pnl']:.2f})")
    
    system.disconnect()