"""
Risk Manager Module
Monitors open positions and manages risk
"""

import pandas as pd
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import json
import os


# ============================================================
# RISK LIMITS
# ============================================================

DEFAULT_RISK_LIMITS = {
    # Maximum portfolio loss before stopping trading
    'max_daily_loss': 0.02,           # 2% per day
    'max_weekly_loss': 0.05,          # 5% per week
    'max_monthly_loss': 0.10,         # 10% per month
    
    # Position limits
    'max_position_size_pct': 0.10,    # 10% of portfolio in one position
    'max_open_positions': 10,
    
    # Stop loss defaults
    'default_stop_loss_pct': 0.10,    # 10% trailing stop
    'use_trailing_stop': True,
    'trailing_stop_pct': 0.08,        # 8% trailing stop
    
    # Profit taking
    'use_profit_taking': True,
    'profit_take_pct': 0.15,          # Take profit at 15% gain
    
    # Risk per trade
    'max_risk_per_trade': 0.02,       # 2% max risk per trade
    
    # Trading hours
    'trading_start_hour': 9,          # 9:30 AM ET
    'trading_end_hour': 16,           # 4:00 PM ET
}


class RiskManager:
    """
    Monitor and manage trading risk.
    """
    
    def __init__(self, portfolio_value: float, config: Dict = None):
        """
        Args:
            portfolio_value: Starting portfolio value
            config: Risk configuration
        """
        self.portfolio_value = portfolio_value
        self.starting_value = portfolio_value
        self.config = {**DEFAULT_RISK_LIMITS, **(config or {})}
        
        # Track daily P&L
        self.daily_pnl = 0
        self.daily_trades = []
        self.last_reset_date = datetime.now().date()
        
        # Open positions tracking
        self.open_positions: Dict[str, Dict] = {}
        
        # Callbacks for risk events
        self.on_risk_limit = None  # Called when risk limit hit
        self.on_stop_loss = None   # Called when stop loss triggered
        self.on_profit_take = None # Called when profit target hit
        
        # Load state
        self._load_state()
    
    # ==================== POSITION MANAGEMENT ====================
    
    def add_position(self, ticker: str, shares: int, entry_price: float,
                    stop_loss: float = None, take_profit: float = None) -> Dict:
        """
        Add a new position to track.
        
        Args:
            ticker: Stock symbol
            shares: Number of shares
            entry_price: Entry price
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
        
        Returns:
            Position dictionary
        """
        # Use default stop loss if not provided
        if stop_loss is None:
            stop_loss = entry_price * (1 - self.config['default_stop_loss_pct'])
        
        if take_profit is None and self.config['use_profit_taking']:
            take_profit = entry_price * (1 + self.config['profit_take_pct'])
        
        position = {
            'ticker': ticker,
            'shares': shares,
            'entry_price': entry_price,
            'entry_date': datetime.now().isoformat(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': entry_price,
            'pnl': 0,
            'pnl_pct': 0,
            'trailing_stop': stop_loss if self.config['use_trailing_stop'] else None
        }
        
        self.open_positions[ticker] = position
        self._save_state()
        
        return position
    
    def update_position(self, ticker: str, current_price: float) -> Dict:
        """
        Update position with current price and check risk rules.
        
        Args:
            ticker: Stock symbol
            current_price: Current market price
        
        Returns:
            Position dictionary with updated values and any risk signals
        """
        if ticker not in self.open_positions:
            return {'error': 'Position not found'}
        
        pos = self.open_positions[ticker]
        
        # Update current price
        pos['current_price'] = current_price
        
        # Calculate P&L
        pos['pnl'] = (current_price - pos['entry_price']) * pos['shares']
        pos['pnl_pct'] = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
        
        # Update trailing stop
        if self.config['use_trailing_stop'] and current_price > pos['entry_price']:
            # Move stop loss up as price increases
            profit = current_price - pos['entry_price']
            new_stop = pos['entry_price'] + (profit * 0.5)  # Keep 50% of profits
            if new_stop > (pos.get('trailing_stop') or 0):
                pos['trailing_stop'] = new_stop
        
        # Check for risk signals
        signals = self._check_position_risk(pos)
        
        self._save_state()
        
        return {
            'position': pos,
            'signals': signals
        }
    
    def remove_position(self, ticker: str, exit_price: float, reason: str) -> Dict:
        """
        Close and remove a position.
        
        Args:
            ticker: Stock symbol
            exit_price: Exit price
            reason: Reason for exit (profit_take, stop_loss, manual, etc.)
        
        Returns:
            Position P&L
        """
        if ticker not in self.open_positions:
            return {'error': 'Position not found'}
        
        pos = self.open_positions[ticker]
        
        # Calculate final P&L
        pnl = (exit_price - pos['entry_price']) * pos['shares']
        pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
        
        # Record trade
        trade_record = {
            'ticker': ticker,
            'shares': pos['shares'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': reason,
            'exit_date': datetime.now().isoformat(),
            'holding_period': (datetime.now() - datetime.fromisoformat(pos['entry_date'])).days
        }
        
        # Update daily P&L
        self.daily_pnl += pnl
        self.daily_trades.append(trade_record)
        
        # Remove position
        del self.open_positions[ticker]
        
        self._save_state()
        
        return trade_record
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get all open positions."""
        return self.open_positions.copy()
    
    def get_position(self, ticker: str) -> Optional[Dict]:
        """Get specific position."""
        return self.open_positions.get(ticker)
    
    # ==================== RISK CHECKS ====================
    
    def _check_position_risk(self, position: Dict) -> Dict:
        """Check if position triggers any risk rules."""
        signals = {
            'stop_loss': False,
            'profit_take': False,
            'trailing_stop': False
        }
        
        current_price = position['current_price']
        
        # Check stop loss
        if current_price <= position['stop_loss']:
            signals['stop_loss'] = True
        
        # Check trailing stop
        if position.get('trailing_stop') and current_price <= position['trailing_stop']:
            signals['trailing_stop'] = True
        
        # Check take profit
        if position.get('take_profit') and current_price >= position['take_profit']:
            signals['profit_take'] = True
        
        return signals
    
    def check_all_positions(self, get_price_func: Callable[[str], float]) -> List[Dict]:
        """
        Check all positions for risk triggers.
        
        Args:
            get_price_func: Function that returns current price for a ticker
        
        Returns:
            List of positions with triggered risk rules
        """
        triggered = []
        
        for ticker in list(self.open_positions.keys()):
            current_price = get_price_func(ticker)
            result = self.update_position(ticker, current_price)
            
            if 'signals' in result:
                signals = result['signals']
                if signals['stop_loss'] or signals['trailing_stop'] or signals['profit_take']:
                    triggered.append({
                        'ticker': ticker,
                        'position': result['position'],
                        'signals': signals
                    })
        
        return triggered
    
    def check_portfolio_risk(self) -> Dict:
        """
        Check overall portfolio risk limits.
        
        Returns:
            Dictionary with risk status
        """
        # Reset daily if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0
            self.daily_trades = []
            self.last_reset_date = current_date
        
        # Calculate current portfolio value
        positions_value = sum(
            pos['current_price'] * pos['shares'] 
            for pos in self.open_positions.values()
        )
        
        # Total value = cash + positions (simplified)
        # In real system, would track actual cash
        current_value = self.portfolio_value + positions_value + self.daily_pnl
        
        # Calculate drawdown
        drawdown = (self.starting_value - current_value) / self.starting_value
        daily_loss = self.daily_pnl / self.portfolio_value
        
        # Check limits
        risk_status = {
            'portfolio_value': current_value,
            'daily_pnl': self.daily_pnl,
            'daily_loss_pct': daily_loss * 100,
            'drawdown_pct': drawdown * 100,
            'num_positions': len(self.open_positions),
            'max_positions': self.config['max_open_positions'],
            'limits_ok': True,
            'warnings': [],
            'stop_trading': False
        }
        
        # Check daily loss limit
        if daily_loss < -self.config['max_daily_loss']:
            risk_status['warnings'].append('Daily loss limit hit!')
            risk_status['limits_ok'] = False
        
        # Check drawdown
        if drawdown > self.config['max_monthly_loss']:
            risk_status['warnings'].append('Maximum drawdown exceeded!')
            risk_status['stop_trading'] = True
            risk_status['limits_ok'] = False
        
        # Check position count
        if len(self.open_positions) >= self.config['max_open_positions']:
            risk_status['warnings'].append('Max positions reached')
        
        return risk_status
    
    def can_trade(self, ticker: str = None) -> tuple[bool, str]:
        """
        Check if we can take a new trade.
        
        Returns:
            (can_trade: bool, reason: str)
        """
        # Check portfolio risk
        portfolio_status = self.check_portfolio_risk()
        
        if portfolio_status['stop_trading']:
            return False, "Portfolio risk limits exceeded - trading stopped"
        
        if portfolio_status['warnings']:
            return False, portfolio_status['warnings'][0]
        
        # Check position count
        if len(self.open_positions) >= self.config['max_open_positions']:
            return False, "Maximum number of positions reached"
        
        # Check if already have position in this ticker
        if ticker and ticker in self.open_positions:
            return False, f"Already have position in {ticker}"
        
        return True, "OK"
    
    # ==================== STATE MANAGEMENT ====================
    
    def _get_state_file(self) -> str:
        return "risk_manager_state.json"
    
    def _save_state(self):
        """Save state to file."""
        state = {
            'portfolio_value': self.portfolio_value,
            'starting_value': self.starting_value,
            'daily_pnl': self.daily_pnl,
            'last_reset_date': str(self.last_reset_date),
            'open_positions': self.open_positions
        }
        try:
            with open(self._get_state_file(), 'w') as f:
                json.dump(state, f, indent=2)
        except:
            pass
    
    def _load_state(self):
        """Load state from file."""
        try:
            if os.path.exists(self._get_state_file()):
                with open(self._get_state_file(), 'r') as f:
                    state = json.load(f)
                    self.portfolio_value = state.get('portfolio_value', self.portfolio_value)
                    self.starting_value = state.get('starting_value', self.portfolio_value)
                    self.daily_pnl = state.get('daily_pnl', 0)
                    self.last_reset_date = pd.to_datetime(state.get('last_reset_date', str(datetime.now().date()))).date()
                    self.open_positions = state.get('open_positions', {})
        except:
            pass
    
    def reset_daily(self):
        """Reset daily tracking (call at start of trading day)."""
        self.daily_pnl = 0
        self.daily_trades = []
        self.last_reset_date = datetime.now().date()
        self._save_state()
    
    def get_daily_summary(self) -> Dict:
        """Get summary of today's trading."""
        return {
            'daily_pnl': self.daily_pnl,
            'num_trades': len(self.daily_trades),
            'trades': self.daily_trades,
            'open_positions': len(self.open_positions)
        }


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("RISK MANAGER TEST")
    print("="*60)
    
    # Create risk manager with $100,000 portfolio
    rm = RiskManager(portfolio_value=100_000)
    
    # Add some positions
    print("\n--- Adding Positions ---")
    
    rm.add_position(
        ticker="AAPL",
        shares=100,
        entry_price=175.00,
        stop_loss=157.50,  # 10% stop
        take_profit=201.25  # 15% target
    )
    
    rm.add_position(
        ticker="NVDA",
        shares=20,
        entry_price=500.00,
        stop_loss=450.00,
        take_profit=575.00
    )
    
    print(f"Open positions: {list(rm.open_positions.keys())}")
    
    # Simulate price updates
    print("\n--- Simulating Price Updates ---")
    
    # AAPL drops to stop loss
    result = rm.update_position("AAPL", 155.00)
    print(f"AAPL @ $155: P&L = ${result['position']['pnl']:.2f} ({result['position']['pnl_pct']:.1f}%)")
    print(f"  Signals: {result['signals']}")
    
    # NVDA rises
    result = rm.update_position("NVDA", 550.00)
    print(f"NVDA @ $550: P&L = ${result['position']['pnl']:.2f} ({result['position']['pnl_pct']:.1f}%)")
    print(f"  Signals: {result['signals']}")
    
    # Close AAPL position (stop loss)
    print("\n--- Closing AAPL (Stop Loss) ---")
    exit_result = rm.remove_position("AAPL", 155.00, "stop_loss")
    print(f"Closed AAPL: P&L = ${exit_result['pnl']:.2f} ({exit_result['pnl_pct']:.1f}%)")
    
    # Check portfolio risk
    print("\n--- Portfolio Risk Check ---")
    risk = rm.check_portfolio_risk()
    print(f"Portfolio Value: ${risk['portfolio_value']:,.2f}")
    print(f"Daily P&L: ${risk['daily_pnl']:.2f}")
    print(f"Daily Loss: {risk['daily_loss_pct']:.2f}%")
    print(f"Drawdown: {risk['drawdown_pct']:.2f}%")
    print(f"Open Positions: {risk['num_positions']}/{risk['max_positions']}")
    print(f"Limits OK: {risk['limits_ok']}")
    
    # Check if we can trade
    print("\n--- Can Trade Check ---")
    can_trade, reason = rm.can_trade("TSLA")
    print(f"Can trade TSLA: {can_trade} ({reason})")
    
    can_trade, reason = rm.can_trade("AAPL")
    print(f"Can trade AAPL: {can_trade} ({reason})")