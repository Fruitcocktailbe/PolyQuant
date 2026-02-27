"""
Position Sizing for PolyQuant 2.0

This module handles position sizing using the Modified Kelly Criterion.
It ensures we never risk too much on any single trade.

THE KELLY CRITERION:
--------------------
The Kelly Criterion tells us the optimal fraction of capital to bet:

    f* = (p * b - q) / b

Where:
- f* = optimal fraction of capital to bet
- p = probability of winning
- b = odds received (net profit if win)
- q = probability of losing (1 - p)

WHY "MODIFIED"?
---------------
Full Kelly is too aggressive for most traders:
1. It assumes perfect probability estimates (we don't have those)
2. It doesn't account for correlation between bets
3. Drawdowns can be severe

We use "Half Kelly" as a conservative default, and additional
constraints based on order book depth.

CONSTRAINTS:
------------
1. Maximum 50% of order book depth (avoid moving the market)
2. Maximum 5% of capital per trade (diversification)
3. Maximum 25% of capital total exposure (correlation risk)

USAGE:
------
    sizer = PositionSizer(capital=10000)
    
    size = sizer.calculate_size(
        probability=0.65,
        odds=1.5,
        order_book_depth=5000,
    )
    
    print(f"Recommended position: ${size}")
"""

from decimal import Decimal
from typing import NamedTuple

from pydantic import BaseModel, Field

from polyquant.data import OrderBook, ProposedTrade
from polyquant.utils import config, get_logger

logger = get_logger(__name__)


class PositionLimits(NamedTuple):
    """Position limit thresholds."""
    max_single_trade_pct: float  # Max % of capital per trade
    max_total_exposure_pct: float  # Max % of capital total
    max_orderbook_depth_pct: float  # Max % of order book
    kelly_fraction: float  # Fraction of full Kelly to use


class PositionSize(BaseModel):
    """
    Calculated position size with explanation.
    
    Attributes:
        recommended_size: The recommended position size in dollars
        kelly_size: What full Kelly would suggest
        limited_by: What constraint limited the size
        probability: The input probability
        expected_value: Expected value of the trade
    """
    recommended_size: float
    kelly_size: float
    limited_by: str = "none"
    probability: float
    expected_value: float
    
    @property
    def is_positive_ev(self) -> bool:
        """Whether the trade has positive expected value."""
        return self.expected_value > 0


class PositionSizer:
    """
    Position sizing using Modified Kelly Criterion.
    
    Calculates optimal position sizes while respecting risk limits.
    
    Example:
        sizer = PositionSizer(capital=10000)
        
        # Calculate for a trade with 65% win probability at 1.5:1 odds
        result = sizer.calculate_size(
            probability=0.65,
            odds=1.5,
            order_book_depth=5000,
        )
        
        print(f"Recommended: ${result.recommended_size}")
        print(f"Limited by: {result.limited_by}")
    """
    
    # Default conservative limits
    DEFAULT_LIMITS = PositionLimits(
        max_single_trade_pct=0.05,  # 5% per trade
        max_total_exposure_pct=0.25,  # 25% total
        max_orderbook_depth_pct=0.5,  # 50% of order book
        kelly_fraction=0.5,  # Half Kelly
    )
    
    def __init__(
        self,
        capital: float = 10000.0,
        limits: PositionLimits | None = None,
        current_exposure: float = 0.0,
    ):
        """
        Initialize the position sizer.
        
        Args:
            capital: Total capital available
            limits: Position limit thresholds
            current_exposure: Current open exposure
        """
        self.capital = capital
        self.limits = limits or self.DEFAULT_LIMITS
        self.current_exposure = current_exposure
        
        # Use config for orderbook depth cap if available
        if hasattr(config, 'orderbook_depth_cap'):
            self.limits = PositionLimits(
                max_single_trade_pct=self.limits.max_single_trade_pct,
                max_total_exposure_pct=self.limits.max_total_exposure_pct,
                max_orderbook_depth_pct=config.orderbook_depth_cap,
                kelly_fraction=self.limits.kelly_fraction,
            )
        
        logger.info(
            "PositionSizer initialized",
            capital=capital,
            limits=self.limits._asdict(),
        )
    
    def calculate_size(
        self,
        probability: float,
        odds: float,
        order_book_depth: float,
    ) -> PositionSize:
        """
        Calculate the recommended position size.
        
        Uses Modified Kelly Criterion with multiple constraints:
        1. Kelly optimal (adjusted by fraction)
        2. Single trade limit (% of capital)
        3. Total exposure limit
        4. Order book depth limit
        
        Args:
            probability: Estimated probability of the outcome (0-1)
            odds: Decimal odds (e.g., 1.5 means win $1.50 per $1 risked)
            order_book_depth: Available liquidity in dollars
            
        Returns:
            PositionSize with recommended size and explanation
        """
        # Validate inputs
        probability = max(0.001, min(0.999, probability))
        odds = max(0.01, odds)
        order_book_depth = max(0, order_book_depth)
        
        # Price band validation - reject trades at extreme prices
        # Price = 1 / odds for fair odds approximation
        implied_price = 1.0 / odds if odds > 0 else 0.5
        if implied_price < 0.02 or implied_price > 0.98:
            logger.warning(
                "Price band rejection",
                implied_price=implied_price,
                reason="Extreme price suggests resolution or broken market",
            )
            return PositionSize(
                recommended_size=0,
                kelly_size=0,
                limited_by="price_band",
                probability=probability,
                expected_value=0,
            )
        
        # Calculate expected value
        ev = probability * odds - (1 - probability)
        
        if ev <= 0:
            # Negative EV trade - don't take it
            logger.debug(
                "Negative EV trade rejected",
                probability=probability,
                odds=odds,
                ev=ev,
            )
            return PositionSize(
                recommended_size=0,
                kelly_size=0,
                limited_by="negative_ev",
                probability=probability,
                expected_value=ev,
            )
        
        # Step 1: Calculate full Kelly size
        # Kelly fraction: f* = (p * b - q) / b
        q = 1 - probability
        full_kelly_fraction = (probability * odds - q) / odds
        full_kelly_size = full_kelly_fraction * self.capital
        
        # Step 2: Apply Kelly reduction
        kelly_size = full_kelly_size * self.limits.kelly_fraction
        
        # Step 3: Apply constraints
        constraints = {
            "kelly": kelly_size,
            "single_trade_limit": self.capital * self.limits.max_single_trade_pct,
            "total_exposure_limit": (
                self.capital * self.limits.max_total_exposure_pct - self.current_exposure
            ),
            "orderbook_limit": order_book_depth * self.limits.max_orderbook_depth_pct,
        }
        
        # Find the binding constraint
        limiting_constraint = min(constraints.items(), key=lambda x: x[1])
        recommended_size = max(0, limiting_constraint[1])
        
        logger.debug(
            "Position size calculated",
            probability=probability,
            odds=odds,
            kelly_size=kelly_size,
            recommended_size=recommended_size,
            limited_by=limiting_constraint[0],
        )
        
        return PositionSize(
            recommended_size=recommended_size,
            kelly_size=kelly_size,
            limited_by=limiting_constraint[0],
            probability=probability,
            expected_value=ev,
        )
    
    def calculate_for_trade(
        self,
        trade: ProposedTrade,
        order_book: OrderBook,
        probability: float,
    ) -> PositionSize:
        """
        Calculate position size for a specific proposed trade.
        
        Convenience method that extracts odds from the order book.
        
        Args:
            trade: The proposed trade
            order_book: Current order book
            probability: Estimated probability
            
        Returns:
            PositionSize recommendation
        """
        # Calculate odds from order book prices
        if trade.side.value == "buy":
            price = order_book.best_ask or 0.5
            # Buying at 'price' to win 1: odds = 1/price
            odds = 1.0 / price if price > 0 else 2.0
        else:
            price = order_book.best_bid or 0.5
            # Selling at 'price': we get price, win (1-price) if outcome doesn't happen
            odds = price / (1 - price) if price < 1 else 1.0
        
        # Get order book depth
        if trade.side.value == "buy":
            depth = order_book.total_ask_depth()
        else:
            depth = order_book.total_bid_depth()
        
        # Convert share depth to dollar depth
        dollar_depth = depth * price if depth else 0
        
        return self.calculate_size(probability, odds, dollar_depth)
    
    def update_exposure(self, amount: float) -> None:
        """Update current exposure after a trade."""
        self.current_exposure += amount
        logger.debug("Exposure updated", new_exposure=self.current_exposure)
    
    def reset_exposure(self) -> None:
        """Reset exposure to zero (e.g., after positions close)."""
        self.current_exposure = 0
        logger.debug("Exposure reset")
