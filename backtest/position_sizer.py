"""
Position Sizer Module
Calculates position size based on risk rules
"""

import pandas as pd
from typing import Dict, Optional
from decimal import Decimal, ROUND_DOWN


# ============================================================
# RISK CONFIGURATION
# ============================================================

DEFAULT_RISK_CONFIG = {
    # Maximum portfolio % to risk per trade
    'max_risk_per_trade': 0.02,        # 2%
    
    # Maximum portfolio % in a single position
    'max_position_size': 0.10,         # 10%
    
    # Maximum number of open positions
    'max_open_positions': 10,
    
    # Stop loss as % of entry price
    'default_stop_loss_pct': 0.10,     # 10%
    
    # Take profit as multiple of risk (reward:risk ratio)
    'reward_risk_ratio': 2.0,          # 2:1
    
    # Minimum reward:risk ratio to accept
    'min_reward_risk': 1.5,
    
    # Use fixed dollar amount instead of % risk
    'use_fixed_risk': False,
    'fixed_risk_amount': 500,          # $500 per trade
}


class PositionSizer:
    """
    Calculate optimal position size based on risk management rules.
    """
    
    def __init__(self, portfolio_value: float, config: Dict = None):
        """
        Args:
            portfolio_value: Total portfolio value in dollars
            config: Risk configuration dictionary
        """
        self.portfolio_value = portfolio_value
        self.config = {**DEFAULT_RISK_CONFIG, **(config or {})}
    
    def calculate_stop_loss(self, entry_price: float, 
                           stop_loss_pct: float = None,
                           atr_multiplier: float = None,
                           atr: float = None) -> float:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Entry price for the position
            stop_loss_pct: Stop loss as % (e.g., 0.10 = 10%)
            atr_multiplier: Multiply ATR for stop loss
            atr: Average True Range value
        
        Returns:
            Stop loss price
        """
        if atr_multiplier is not None and atr is not None:
            # ATR-based stop loss
            return entry_price - (atr * atr_multiplier)
        else:
            # Percentage-based stop loss
            pct = stop_loss_pct or self.config['default_stop_loss_pct']
            return entry_price * (1 - pct)
    
    def calculate_take_profit(self, entry_price: float, 
                             stop_loss: float,
                             reward_risk_ratio: float = None) -> float:
        """
        Calculate take profit price based on reward:risk ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            reward_risk_ratio: Target reward:risk ratio
        
        Returns:
            Take profit price
        """
        ratio = reward_risk_ratio or self.config['reward_risk_ratio']
        risk = entry_price - stop_loss
        return entry_price + (risk * ratio)
    
    def calculate_position_size(self, entry_price: float, 
                               stop_loss: float,
                               current_price: float = None) -> Dict:
        """
        Calculate position size based on risk parameters.
        
        Uses the formula:
        position_size = risk_amount / risk_per_share
        
        Where:
        - risk_amount = portfolio * max_risk_per_trade
        - risk_per_share = entry_price - stop_loss
        
        Args:
            entry_price: Planned entry price
            stop_loss: Stop loss price
            current_price: Current market price (for validation)
        
        Returns:
            Dictionary with position details
        """
        # Calculate risk amount
        if self.config['use_fixed_risk']:
            risk_amount = self.config['fixed_risk_amount']
        else:
            risk_amount = self.portfolio_value * self.config['max_risk_per_trade']
        
        # Ensure we don't exceed max position size
        max_position_value = self.portfolio_value * self.config['max_position_size']
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return {
                'shares': 0,
                'capital_required': 0,
                'risk_amount': 0,
                'error': 'Stop loss equals entry price'
            }
        
        # Calculate raw position size
        raw_shares = risk_amount / risk_per_share
        
        # Round down to whole shares
        shares = int(Decimal(str(raw_shares)).quantize(Decimal('1'), rounding=ROUND_DOWN))
        
        # Calculate actual capital required
        actual_entry = current_price if current_price else entry_price
        capital_required = shares * actual_entry
        
        # Check if within position limits
        if capital_required > max_position_value:
            # Recalculate based on max position
            shares = int(max_position_value / actual_entry)
            capital_required = shares * actual_entry
            actual_risk = shares * risk_per_share
        
        # Verify reward:risk ratio
        reward = abs(self.calculate_take_profit(entry_price, stop_loss) - entry_price)
        actual_rr_ratio = reward / risk_per_share if risk_per_share > 0 else 0
        
        return {
            'shares': shares,
            'capital_required': capital_required,
            'risk_amount': risk_amount,
            'risk_per_share': risk_per_share,
            'stop_loss': stop_loss,
            'take_profit': self.calculate_take_profit(entry_price, stop_loss),
            'reward_risk_ratio': actual_rr_ratio,
            'risk_pct_of_portfolio': (shares * risk_per_share) / self.portfolio_value,
        }
    
    def validate_position(self, signal: Dict, current_price: float) -> Dict:
        """
        Validate if a signal is worth trading based on position sizing.
        
        Args:
            signal: Signal from signal_generator
            current_price: Current market price
        
        Returns:
            Validation result with position details or rejection reason
        """
        entry_price = signal.get('entry_price', current_price)
        
        # Calculate stop loss (use signal ATR if available, otherwise default)
        atr = signal.get('atr')
        if atr:
            # Use 2x ATR for stop
            stop_loss = self.calculate_stop_loss(
                entry_price, 
                atr_multiplier=2.0,
                atr=atr
            )
        else:
            stop_loss = self.calculate_stop_loss(entry_price)
        
        # Check if stop loss is too tight
        stop_distance_pct = abs(entry_price - stop_loss) / entry_price
        if stop_distance_pct < 0.02:  # Less than 2%
            return {
                'valid': False,
                'reason': f'Stop loss too tight: {stop_distance_pct:.1%}'
            }
        
        # Calculate position size
        position = self.calculate_position_size(entry_price, stop_loss, current_price)
        
        if position.get('shares', 0) == 0:
            return {
                'valid': False,
                'reason': 'Position size too small'
            }
        
        # Check reward:risk ratio
        if position.get('reward_risk_ratio', 0) < self.config['min_reward_risk']:
            return {
                'valid': False,
                'reason': f'Reward:risk too low: {position["reward_risk_ratio"]:.2f}'
            }
        
        # All checks passed
        return {
            'valid': True,
            'ticker': signal['ticker'],
            'signal_type': signal['signal_type'],
            'entry_price': current_price,
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit'],
            'shares': position['shares'],
            'capital_required': position['capital_required'],
            'risk_amount': position['risk_amount'],
            'reward_risk_ratio': position['reward_risk_ratio'],
            'confidence': signal.get('confidence', 0),
            'entry_reason': signal.get('entry_reason', ''),
        }
    
    def calculate_portfolio_allocation(self, positions: list) -> Dict:
        """
        Calculate current portfolio allocation across positions.
        
        Args:
            positions: List of open position dictionaries
        
        Returns:
            Allocation summary
        """
        total_value = self.portfolio_value
        allocated = 0
        position_details = []
        
        for pos in positions:
            pos_value = pos.get('shares', 0) * pos.get('current_price', 0)
            allocated += pos_value
            
            position_details.append({
                'ticker': pos.get('ticker'),
                'value': pos_value,
                'pct': (pos_value / total_value) * 100 if total_value > 0 else 0
            })
        
        return {
            'total_portfolio': total_value,
            'allocated': allocated,
            'cash': total_value - allocated,
            'allocated_pct': (allocated / total_value) * 100 if total_value > 0 else 0,
            'positions': position_details,
            'num_positions': len(positions)
        }
    
    def can_open_new_position(self, current_positions: int) -> bool:
        """Check if we can open a new position based on limits."""
        return current_positions < self.config['max_open_positions']


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("POSITION SIZER TEST")
    print("="*60)
    
    # Create position sizer with $100,000 portfolio
    sizer = PositionSizer(portfolio_value=100_000)
    
    # Example: AAPL at $175 with 10% stop loss
    entry_price = 175.00
    stop_loss = 175 * 0.90  # $157.50
    
    print(f"\n--- Example: AAPL @ ${entry_price} ---")
    print(f"Stop Loss: ${stop_loss}")
    
    position = sizer.calculate_position_size(entry_price, stop_loss)
    
    print(f"\nPosition Details:")
    print(f"  Shares: {position['shares']}")
    print(f"  Capital Required: ${position['capital_required']:,.2f}")
    print(f"  Risk Amount: ${position['risk_amount']:,.2f}")
    print(f"  Stop Loss: ${position['stop_loss']:.2f}")
    print(f"  Take Profit: ${position['take_profit']:.2f}")
    print(f"  Reward:Risk Ratio: {position['reward_risk_ratio']:.2f}")
    print(f"  Risk % of Portfolio: {position['risk_pct_of_portfolio']:.2%}")
    
    # Test validation with a signal
    print("\n" + "="*60)
    print("SIGNAL VALIDATION TEST")
    print("="*60)
    
    # Simulated signal from signal_generator
    test_signal = {
        'ticker': 'NVDA',
        'signal_type': 'BUY',
        'entry_price': 500.00,
        'atr': 15.50,
        'confidence': 0.75,
        'entry_reason': 'RSI oversold (28); MACD bullish'
    }
    
    result = sizer.validate_position(test_signal, current_price=500.00)
    
    print(f"\nSignal: {test_signal['ticker']} - {test_signal['signal_type']}")
    print(f"Valid: {result.get('valid', False)}")
    
    if result.get('valid'):
        print(f"  Shares: {result['shares']}")
        print(f"  Capital: ${result['capital_required']:,.2f}")
        print(f"  Risk: ${result['risk_amount']:,.2f}")
        print(f"  Stop: ${result['stop_loss']:.2f}")
        print(f"  Target: ${result['take_profit']:.2f}")
        print(f"  R:R Ratio: {result['reward_risk_ratio']:.2f}")
    else:
        print(f"  Rejected: {result.get('reason')}")
    
    # Test portfolio allocation
    print("\n" + "="*60)
    print("PORTFOLIO ALLOCATION TEST")
    print("="*60)
    
    open_positions = [
        {'ticker': 'AAPL', 'shares': 100, 'current_price': 175},
        {'ticker': 'MSFT', 'shares': 50, 'current_price': 400},
        {'ticker': 'NVDA', 'shares': 20, 'current_price': 500},
    ]
    
    allocation = sizer.calculate_portfolio_allocation(open_positions)
    
    print(f"\nPortfolio: ${allocation['total_portfolio']:,.2f}")
    print(f"Allocated: ${allocation['allocated']:,.2f} ({allocation['allocated_pct']:.1f}%)")
    print(f"Cash: ${allocation['cash']:,.2f}")
    print(f"Positions: {allocation['num_positions']}")