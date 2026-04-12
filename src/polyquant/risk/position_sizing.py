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

EMPIRICAL KELLY (FROM RESEARCH):
--------------------------------
Standard Kelly assumes your edge is known with certainty. In practice,
the solver's profit guarantee has variance. We adjust:

    f_empirical = f_kelly × (1 - CV_edge)

Where CV_edge = std(edge) / mean(edge) is the coefficient of variation
of recent edge estimates. High CV → uncertain edge → smaller positions.

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

import math
from collections import deque
from typing import NamedTuple

from pydantic import BaseModel, Field

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
        recommended_size: The recommended position size in shares (number of tokens to buy/sell)
        kelly_size: What full Kelly would suggest (dollars, internal pre-conversion value)
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
        
        print(f"Recommended: {result.recommended_size} shares")
        print(f"Limited by: {result.limited_by}")
    """
    
    # Default conservative limits now mapped to centralized config
    @property
    def DEFAULT_LIMITS(self) -> PositionLimits:
        return PositionLimits(
            max_single_trade_pct=config.max_single_trade_pct,
            max_total_exposure_pct=config.max_total_exposure_pct,
            max_orderbook_depth_pct=config.orderbook_depth_cap,
            kelly_fraction=config.kelly_fraction,
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

        # Empirical Kelly: track edge estimates to compute CV
        self._edge_history: deque[float] = deque(maxlen=50)
        self._cv_edge: float = 0.0  # Coefficient of variation of edge

        
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
        # Price = 1 / (odds + 1) for fair odds approximation
        implied_price = 1.0 / (odds + 1.0) if odds > 0 else 0.5
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
        
        # Step 2: Apply Kelly reduction (Half Kelly)
        kelly_size = full_kelly_size * self.limits.kelly_fraction

        # Step 2b: Empirical Kelly adjustment
        # f_empirical = f_kelly × (1 - CV_edge)
        # When edge estimates are volatile (high CV), size down further
        self._edge_history.append(ev)
        self._update_cv_edge()
        
        if self._cv_edge > 0.1 and len(self._edge_history) >= 5:
            empirical_factor = max(0.2, 1.0 - self._cv_edge)  # Floor at 20%
            kelly_size *= empirical_factor
            logger.debug(
                "Empirical Kelly adjustment applied",
                cv_edge=f"{self._cv_edge:.3f}",
                factor=f"{empirical_factor:.3f}",
                kelly_before=f"{full_kelly_size * self.limits.kelly_fraction:.2f}",
                kelly_after=f"{kelly_size:.2f}",
            )
        
        # Step 3: Apply constraints
        constraints = {
            "kelly": kelly_size,
            "single_trade_limit": self.capital * self.limits.max_single_trade_pct,
            "total_exposure_limit": (
                self.capital * self.limits.max_total_exposure_pct - self.current_exposure
            ),
            "orderbook_limit": order_book_depth * self.limits.max_orderbook_depth_pct,
        }
        
        # Find the binding constraint (still in dollars at this point)
        limiting_constraint = min(constraints.items(), key=lambda x: x[1])
        dollar_size = max(0.0, limiting_constraint[1])

        # Convert dollar stake → shares. price_per_share = 1 / (odds + 1) works for both
        # BUY (buy YES at ask, odds = 1/price - 1) and SELL (buy NO at 1-bid, odds = bid/(1-bid))
        # because fw_solver maps each side's per-share cost into `odds` consistently.
        price_per_share = 1.0 / (odds + 1.0) if odds > 0 else 0.5
        recommended_size = dollar_size / price_per_share if price_per_share > 0 else 0.0

        logger.debug(
            "Position size calculated",
            probability=probability,
            odds=odds,
            kelly_size=kelly_size,
            dollar_size=dollar_size,
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
    
    def calculate_dutching_sizes(
        self,
        odds_list: list[float],
        depth_list: list[float],
    ) -> list[PositionSize]:
        """
        Calculate risk-free arbitrage sizes across a group of mutually exclusive outcomes.
        
        This uses the Dutching method: staking optimally across all legs such that
        a guaranteed profit is locked in, regardless of the event's outcome.
        
        Args:
            odds_list: Decimal odds for each leg.
            depth_list: Available liquidity (in dollars) for each leg.
            
        Returns:
            A list of PositionSize objects, one for each leg in order.
            If no risk-free arbitrage exists (implied probability sum >= 1.0),
            returns sizes of 0.
        """
        if not odds_list or len(odds_list) != len(depth_list):
            logger.warning("Invalid inputs to dutching sizing", count_odds=len(odds_list), count_depth=len(depth_list))
            return [PositionSize(recommended_size=0, kelly_size=0, limited_by="invalid_input", probability=0, expected_value=0)] * len(odds_list)
            
        # Calculate implied probabilities: p = 1 / (odds + 1)
        implied_probs = [1.0 / (odds + 1.0) if odds > 0 else 0.5 for odds in odds_list]
        sum_implied = sum(implied_probs)
        
        # If sum >= 1.0, there is no risk-free arbitrage
        if sum_implied >= 1.0:
            logger.debug("No Dutching arbitrage exists", sum_implied=sum_implied)
            return [
                PositionSize(recommended_size=0, kelly_size=0, limited_by="no_arb", probability=p, expected_value=0)
                for p in implied_probs
            ]
            
        # Expected percentage profit on total capital staked across all legs
        percent_profit = (1.0 / sum_implied) - 1.0
        
        # Calculate maximum possible total stake T across the entire ring
        # 1. Capital constraints
        max_total_exposure = self.capital * self.limits.max_total_exposure_pct - self.current_exposure
        max_single_trade_exposure = self.capital * self.limits.max_single_trade_pct # Treating the entire ring as one synthetic 'trade'
        max_t = min(max_total_exposure, max_single_trade_exposure)
        
        # 2. Liquidity constraints (bottleneck detection)
        # stake_i = T * (p_i / sum_implied)
        # We require stake_i <= depth_i * limit  =>  T <= (depth_i * limit * sum_implied) / p_i
        limiting_constraint = "capital_limits"
        for i, (p_i, depth_i) in enumerate(zip(implied_probs, depth_list)):
            leg_max_stake = depth_i * self.limits.max_orderbook_depth_pct
            max_t_for_leg = (leg_max_stake * sum_implied) / p_i
            
            if max_t_for_leg < max_t:
                max_t = max_t_for_leg
                limiting_constraint = f"leg_{i}_liquidity"
                
        # Generate final sizes
        final_sizes = []
        for p_i in implied_probs:
            stake_i = max_t * (p_i / sum_implied)
            # number of shares = stake / price
            # Since p_i = 1 / (odds + 1), and price = 1 / (odds + 1), p_i is exactly the price
            shares_i = stake_i / p_i if p_i > 0 else 0.0
            
            final_sizes.append(PositionSize(
                recommended_size=max(0.0, shares_i),
                kelly_size=0.0, # N/A for Dutching
                limited_by=limiting_constraint,
                probability=p_i,
                expected_value=percent_profit # Expected percentage margin
            ))
            
        logger.debug(
            "Dutching position size calculated",
            sum_implied=f"{sum_implied:.4f}",
            guaranteed_margin=f"{percent_profit*100:.2f}%",
            total_stake=max_t,
            limited_by=limiting_constraint
        )
        return final_sizes
    
    def update_exposure(self, amount: float) -> None:
        """Update current exposure after a trade."""
        self.current_exposure += amount
        logger.debug("Exposure updated", new_exposure=self.current_exposure)
    
    def reset_exposure(self) -> None:
        """Reset exposure to zero (e.g., after positions close)."""
        self.current_exposure = 0
        logger.debug("Exposure reset")

    def _update_cv_edge(self) -> None:
        """
        Update the coefficient of variation of edge estimates.
        
        CV = std(edge) / mean(edge)
        High CV means our edge estimates are noisy → size down.
        """
        if len(self._edge_history) < 3:
            self._cv_edge = 0.0
            return

        edges = list(self._edge_history)
        mean_edge = sum(edges) / len(edges)

        if abs(mean_edge) < 1e-10:
            self._cv_edge = 1.0  # Edge is ~0, maximum uncertainty
            return

        variance = sum((e - mean_edge) ** 2 for e in edges) / len(edges)
        std_edge = math.sqrt(variance)
        self._cv_edge = abs(std_edge / mean_edge)

    @property
    def edge_uncertainty(self) -> float:
        """Current edge uncertainty (CV). Higher = more uncertain."""
        return self._cv_edge
