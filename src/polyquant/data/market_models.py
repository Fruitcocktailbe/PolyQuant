"""
Data Models for PolyQuant 2.0

This module contains all Pydantic data models used throughout the system.

MODELS:
-------
- Market: Prediction market with outcomes
- Outcome: Individual outcome with price
- OrderBook: Bid/ask order book
- OrderLevel: Single price level
- MarketDependency: Logical dependency between markets
- ProposedTrade: Trade recommendation
- ArbitrageOpportunity: Detected arbitrage

USAGE:
------
    from polyquant.data import Market, OrderBook, ProposedTrade
    
    market = Market(
        market_id="abc123",
        question="Will X happen?",
        outcomes=[Outcome(outcome_id="yes", name="Yes", price=0.65)]
    )
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    """Side of an order: buy or sell."""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """
    Time-in-force for order execution on Polymarket CLOB.
    
    FOK: Fill-or-Kill - Fill completely or cancel (ALL-OR-NOTHING, safest for arb)
    FAK: Fill-and-Kill - Fill as much as possible, cancel remainder (partial IOC)
    GTC: Good-til-Cancelled - Stays in book (DANGEROUS for arb, never use)
    GTD: Good-til-Date - Stays in book until expiration
    """
    FOK = "FOK"
    FAK = "FAK"
    GTC = "GTC"
    GTD = "GTD"


class Outcome(BaseModel):
    """
    A single outcome within a prediction market.
    
    Attributes:
        outcome_id: Unique identifier for this outcome
        name: Human-readable name (e.g., "Yes", "No", "Trump")
        price: Current market price (0-1)
        token_id: On-chain token identifier
    """
    outcome_id: str
    name: str
    price: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    token_id: str = ""


class MicrostructureSignal(BaseModel):
    """
    Real-time order book signals.
    """
    imbalance: float = 0.0  # (BidVol - AskVol) / (BidVol + AskVol)
    spread: float = 0.0     # Ask - Bid
    weighted_midpoint: float = 0.0
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class CorrelationSignal(BaseModel):
    """
    Statistical relationship between two markets.
    """
    leader_id: str
    laggard_id: str
    correlation: float  # -1.0 to 1.0
    lead_time_seconds: float = 0.0
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Market(BaseModel):
    """
    A prediction market from Polymarket.
    
    Attributes:
        market_id: Unique market identifier
        question: The market question
        description: Detailed description and resolution criteria
        outcomes: List of possible outcomes
        volume: Total trading volume in dollars
        liquidity: Current liquidity in dollars
        end_date: When the market closes
        resolved: Whether the market has resolved
    """
    market_id: str
    question: str
    description: str = ""
    outcomes: list[Outcome] = Field(default_factory=list)
    volume: float = 0.0
    liquidity: float = 0.0
    end_date: datetime | None = None
    resolved: bool = False
    
    # Phase 5: Enhanced Metadata
    negrisk: bool = False
    group_id: str | None = None
    market_type: str | None = None

    # Phase 6: Conditional market support
    conditional_parent_id: str | None = None  # Parent market if conditional
    resolution_source: str | None = None      # UMA, Polymarket, custom

    # Phase 8: Strategy Signals
    microstructure: "MicrostructureSignal | None" = None
    
    def get_outcome(self, outcome_id: str) -> Outcome | None:
        """Get an outcome by ID."""
        for o in self.outcomes:


            if o.outcome_id == outcome_id:
                return o
        return None


class OrderLevel(BaseModel):
    """
    A single price level in an order book.
    
    Attributes:
        price: Price at this level (0-1)
        size: Available size in shares
    """
    price: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    size: Decimal = Field(default=Decimal("0"), ge=0)


class OrderBook(BaseModel):
    """
    Order book for a single outcome.
    
    Attributes:
        outcome_id: Which outcome this order book is for
        bids: Buy orders (sorted by price descending)
        asks: Sell orders (sorted by price ascending)
        timestamp: When this snapshot was taken
    """
    outcome_id: str
    bids: list[OrderLevel] = Field(default_factory=list)
    asks: list[OrderLevel] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def best_bid(self) -> Decimal | None:
        """Highest bid price."""
        if self.bids:
            return max(b.price for b in self.bids)
        return None
    
    @property
    def best_ask(self) -> Decimal | None:
        """Lowest ask price."""
        if self.asks:
            return min(a.price for a in self.asks)
        return None
    
    @property
    def spread(self) -> Decimal | None:
        """Bid-ask spread."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is not None and ask is not None:
            return ask - bid
        return None
    
    @property
    def mid_price(self) -> Decimal | None:
        """Mid-market price."""
        bid = self.best_bid
        ask = self.best_ask
        if bid is not None and ask is not None:
            return (bid + ask) / Decimal("2")
        return None
    
    def total_bid_depth(self) -> Decimal:
        """Total bid size across all levels."""
        return sum((b.size for b in self.bids), Decimal("0"))
    
    def total_ask_depth(self) -> Decimal:
        """Total ask size across all levels."""
        return sum((a.size for a in self.asks), Decimal("0"))

    def get_vwap(self, side: OrderSide, size: Decimal) -> Decimal | None:
        """
        Calculate the Volume-Weighted Average Price for a given order size.

        This is critical for execution. The top bid/ask price is not the price
        you will actually get for a large order; you will "walk the book".

        Args:
            side: OrderSide.BUY to calculate for buying, OrderSide.SELL for selling.
            size: The order size in number of tokens.

        Returns:
            The VWAP if enough liquidity exists, None otherwise.
        """
        book = self.asks if side == OrderSide.BUY else self.bids
        if not book:
            return None

        total_cost = Decimal("0")
        remaining_size = size
        for level in book:
            filled = min(remaining_size, level.size)
            total_cost += filled * level.price
            remaining_size -= filled
            if remaining_size <= 0:
                break

        if remaining_size > 0:
            return None  # Not enough liquidity

        return total_cost / size


class MarketDependency(BaseModel):
    """
    A logical dependency between two market outcomes.
    
    Represents a rule like: "If market A resolves to X, then market B
    must resolve to Y with some probability constraint."
    
    Attributes:
        source_market_id: First market
        source_outcome: Outcome in first market (e.g., "Yes")
        target_market_id: Second market
        target_outcome: Outcome in second market
        relationship: Type (implies, excludes, correlates)
        confidence: How confident we are in this dependency
    """
    source_market_id: str
    source_outcome: str
    target_market_id: str
    target_outcome: str
    relationship: str = "implies"  # implies, excludes, correlates
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""


class ProposedTrade(BaseModel):
    """
    A proposed trade from the optimizer.
    
    Attributes:
        market_id: Which market
        outcome_id: Which outcome to trade
        side: Buy or sell
        size: Number of shares
        limit_price: Maximum/minimum price to accept
        time_in_force: Order execution policy (default: IOC for safety)
        priority: Execution order (lower = first, illiquid legs prioritized)
        reason: Why this trade was proposed
    """
    market_id: str = ""  # Optional: CLOB uses outcome_id (token_id) for submission
    outcome_id: str
    side: OrderSide
    size: Decimal = Field(default=Decimal("0"), ge=0)
    limit_price: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    time_in_force: TimeInForce = TimeInForce.FOK  # FOK = all-or-nothing (safest for arb)
    priority: int = Field(default=0, ge=0)  # Lower = execute first
    reason: str = ""
    exchange: str = "polymarket"
    
    @property
    def notional_value(self) -> Decimal:
        """Dollar value of this trade."""
        return self.size * self.limit_price


class ArbitrageOpportunity(BaseModel):
    """
    A detected arbitrage opportunity.

    Represents a set of trades that together guarantee a profit
    regardless of market outcomes.

    Attributes:
        markets: List of market IDs involved
        trades: Proposed trades to execute
        expected_profit: Expected profit in dollars
        guaranteed_profit: Minimum guaranteed profit (from formula)
        confidence: Overall confidence score
        roi: Return on investment (expected_profit / capital_deployed)
        capital_efficiency: Profit per second (for ranking by capital turnover)
        detected_at: When this was detected
    """
    markets: list[str] = Field(default_factory=list)
    trades: list[ProposedTrade] = Field(default_factory=list)
    expected_profit: Decimal = Field(default=Decimal("0"))
    guaranteed_profit: Decimal = Field(default=Decimal("0"))
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    roi: float = Field(default=0.0, description="Return on investment as fraction")
    capital_efficiency: float = Field(default=0.0, description="Profit per second (estimated)")
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def trade_count(self) -> int:
        """Number of trades in this opportunity."""
        return len(self.trades)
    
    @property
    def total_notional(self) -> Decimal:
        """Total notional value of all trades."""
        return sum((t.notional_value for t in self.trades), Decimal("0"))
