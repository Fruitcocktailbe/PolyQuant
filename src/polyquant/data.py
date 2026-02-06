"""
PolyQuant Data Models and Polymarket Client

This module defines the core data structures and the PolymarketClient for
interacting with the Polymarket Gamma API (market discovery) and CLOB
(order book and execution).

ARCHITECTURE OVERVIEW (from Research Papers):
---------------------------------------------
The system operates on the principle of finding and exploiting arbitrage
opportunities in prediction markets. This requires:

1.  **Market Discovery (Gamma API)**: Fetch active markets and their metadata.
2.  **Real-Time Data (CLOB WebSocket)**: Maintain ultra-low-latency L2 order
    book state for each outcome token.
3.  **Execution (CLOB REST)**: Submit orders to capture arbitrage profits.

KEY DATA STRUCTURES:
--------------------
- `Market`: Represents a prediction market (e.g., "Will X win?").
- `MarketOutcome`: A specific outcome within a market (e.g., "Yes", "No").
- `OrderBook`: L2 order book (bids and asks with prices and sizes).
- `ProposedTrade`: A trade recommendation from the optimization engine.

USAGE:
------
    async with PolymarketClient() as client:
        markets = await client.get_active_markets(limit=100)
        for market in markets:
            token_ids = [t.get("token_id") for t in market.tokens]
            await client.subscribe_to_order_books(token_ids)
        
        # ... later, get order book state
        book = await client.get_order_book(token_id)
        print(f"Best Bid: {book.best_bid}, Best Ask: {book.best_ask}")
"""

import asyncio
import json
import logging
import time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import websockets
from pydantic import BaseModel, Field

from polyquant.utils import config

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class OrderSide(str, Enum):
    """Represents the side of an order (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


class MarketOutcome(BaseModel):
    """
    Represents a single outcome within a prediction market.
    
    Attributes:
        name: The human-readable name of the outcome (e.g., "Yes", "Donald Trump").
        outcome_id: The unique token ID for this outcome on Polymarket (CLOB).
        price: The current market price for this outcome (0.0 to 1.0).
    """
    name: str
    outcome_id: str
    price: float


class Market(BaseModel):
    """
    Represents a prediction market on Polymarket.

    A market consists of a question and a set of mutually exclusive outcomes.
    The sum of the probabilities of all outcomes should ideally equal 1.0,
    but deviations create arbitrage opportunities.

    Attributes:
        market_id: The condition_id used to identify this market on the CLOB.
        question: The human-readable question the market is predicting.
        description: Optional longer description of the market.
        outcomes: A list of `MarketOutcome` objects.
        volume: The total trading volume in USD.
        tokens: Raw token data from the Gamma API (for advanced use).
    """
    market_id: str
    question: str
    description: Optional[str] = None
    outcomes: List[MarketOutcome] = []
    volume: float = 0.0
    tokens: List[Dict[str, Any]] = []


class OrderBook(BaseModel):
    """
    Represents an L2 (Level 2) order book for a single outcome token.

    The order book contains all resting bids and asks at various price levels.
    This data is essential for:
    1.  Calculating VWAP (Volume-Weighted Average Price) for execution.
    2.  Determining available liquidity at a given price.
    3.  Detecting single-market arbitrage (sum of best bids/asks != 1.0).

    Attributes:
        market_id: The token_id this order book corresponds to.
        best_bid: The highest price a buyer is willing to pay.
        best_ask: The lowest price a seller is willing to accept.
        bids: A list of (price, size) tuples, sorted descending by price.
        asks: A list of (price, size) tuples, sorted ascending by price.
        last_updated: Unix timestamp of the last update.
    """
    market_id: str = ""
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bids: List[Tuple[float, float]] = []
    asks: List[Tuple[float, float]] = []
    last_updated: float = Field(default_factory=time.time)

    def get_vwap(self, side: OrderSide, size: float) -> Optional[float]:
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

        total_cost = 0.0
        remaining_size = size
        for price, level_size in book:
            filled = min(remaining_size, level_size)
            total_cost += filled * price
            remaining_size -= filled
            if remaining_size <= 0:
                break
        
        if remaining_size > 0:
            return None  # Not enough liquidity
        
        return total_cost / size


class ProposedTrade(BaseModel):
    """
    A trade recommendation generated by the optimization engine.

    Attributes:
        market_id: The condition_id of the market.
        outcome_id: The token_id of the specific outcome to trade.
        side: Whether to BUY or SELL.
        size: The number of tokens to trade.
        limit_price: The maximum/minimum price to pay/receive.
    """
    market_id: str
    outcome_id: str
    side: OrderSide
    size: float
    limit_price: float


class ArbitrageOpportunity(BaseModel):
    """
    Represents a detected arbitrage opportunity.

    This is the output of the solver phase, containing all the trades
    needed to capture the arbitrage.

    Attributes:
        markets: List of market_ids involved.
        trades: List of ProposedTrade objects to execute.
        expected_profit: The guaranteed minimum profit (in USD).
        confidence: A score from the validation phase (0.0 to 1.0).
    """
    markets: List[str]
    trades: List[ProposedTrade]
    expected_profit: Decimal
    confidence: float


class MarketDependency(BaseModel):
    """
    Represents a logical dependency between two markets.

    Example: "If Trump wins Pennsylvania, he must also win the Presidency."
    This is identified by the Logic Architect agent and used by the IP solver
    to define the constraints of the marginal polytope.

    Attributes:
        source_market_id: The market that implies the dependency.
        source_outcome: The outcome in the source market.
        target_market_id: The market that is constrained by the dependency.
        target_outcome: The corresponding outcome in the target market.
        relationship: A string describing the relationship (e.g., "IMPLIES").
        confidence: The LLM's confidence in this dependency (0.0 to 1.0).
    """
    source_market_id: str
    source_outcome: str
    target_market_id: str
    target_outcome: str
    relationship: str
    confidence: float


# =============================================================================
# Polymarket Client
# =============================================================================

class PolymarketClient:
    """
    Asynchronous client for interacting with Polymarket.

    This client provides:
    1.  **Market Discovery**: Fetch active markets from the Gamma API.
    2.  **Real-Time Order Books**: Subscribe to L2 order book updates via WebSocket.
    3.  **Order Execution**: (Stubbed) Submit orders to the CLOB REST API.

    Usage:
        async with PolymarketClient() as client:
            markets = await client.get_active_markets()
            # ...
    """
    GAMMA_API_URL = "https://gamma-api.polymarket.com"

    def __init__(self):
        """Initialize the client. Call `async with` to activate."""
        self.clob_url = config.polymarket_clob_url
        self.ws_url = config.polymarket_ws_url
        self.http_client: Optional[httpx.AsyncClient] = None
        self.ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self.order_books: Dict[str, OrderBook] = {}
        self.subscribed_markets: Set[str] = set()
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "PolymarketClient":
        """Initialize HTTP client and start running."""
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self._running = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Cleanup: cancel WS task, close connections."""
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        if self.ws_connection:
            await self.ws_connection.close()
        if self.http_client:
            await self.http_client.aclose()

    # -------------------------------------------------------------------------
    # Market Discovery (Gamma API)
    # -------------------------------------------------------------------------
    async def get_active_markets(
        self, limit: int = 100, min_liquidity: float = 1000.0
    ) -> List[Market]:
        """
        Fetch active markets from the Gamma API.

        Args:
            limit: Maximum number of markets to return.
            min_liquidity: Minimum trading volume in USD.

        Returns:
            A list of Market objects, sorted by volume (descending).
        """
        if not self.http_client:
            raise RuntimeError("Client not initialized. Use 'async with'.")

        params = {
            "limit": limit,
            "active": "true",
            "closed": "false",
            "volume_min": min_liquidity,
            "order": "volume_desc",
        }

        try:
            response = await self.http_client.get(
                f"{self.GAMMA_API_URL}/markets", params=params
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_markets(data)
        except Exception as e:
            logger.error(f"Error fetching markets from Gamma API: {e}")
            return []

    def _parse_markets(self, data: List[Dict[str, Any]]) -> List[Market]:
        """Parse raw Gamma API response into Market objects."""
        markets = []
        for item in data:
            try:
                tokens = item.get("tokens", [])
                outcomes_raw = item.get("outcomes", [])
                if isinstance(outcomes_raw, str):
                    outcomes_raw = json.loads(outcomes_raw)

                if not tokens or len(tokens) != len(outcomes_raw):
                    continue

                market_outcomes = [
                    MarketOutcome(
                        name=outcomes_raw[idx],
                        outcome_id=token.get("token_id", ""),
                        price=float(token.get("price", 0.0)),
                    )
                    for idx, token in enumerate(tokens)
                ]

                markets.append(
                    Market(
                        market_id=item.get("condition_id", ""),
                        question=item.get("question", "Unknown Question"),
                        description=item.get("description"),
                        outcomes=market_outcomes,
                        volume=float(item.get("volume", 0.0)),
                        tokens=tokens,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse market item: {e}")
        return markets

    # -------------------------------------------------------------------------
    # Real-Time Order Books (CLOB WebSocket)
    # -------------------------------------------------------------------------
    async def subscribe_to_order_books(self, token_ids: List[str]) -> None:
        """
        Subscribe to real-time L2 order book updates for the given tokens.

        Args:
            token_ids: A list of outcome token IDs.
        """
        if not self._running:
            return

        new_ids = [tid for tid in token_ids if tid not in self.subscribed_markets]
        if not new_ids:
            return

        self.subscribed_markets.update(new_ids)

        if self.ws_connection is None or self.ws_connection.closed:
            if self._ws_task:
                self._ws_task.cancel()
            self._ws_task = asyncio.create_task(self._ws_handler())
            await asyncio.sleep(1)  # Allow connection to establish

        if self.ws_connection and not self.ws_connection.closed:
            msg = {"type": "market", "assets": new_ids, "action": "subscribe"}
            await self.ws_connection.send(json.dumps(msg))
            logger.info(f"Subscribed to {len(new_ids)} order book channels")

    async def _ws_handler(self) -> None:
        """Main WebSocket event loop with reconnection logic."""
        while self._running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.ws_connection = ws
                    logger.info("Connected to Polymarket CLOB WebSocket")

                    if self.subscribed_markets:
                        msg = {
                            "type": "market",
                            "assets": list(self.subscribed_markets),
                            "action": "subscribe",
                        }
                        await ws.send(json.dumps(msg))

                    async for message in ws:
                        self._process_ws_message(message)

            except Exception as e:
                logger.error(f"WebSocket error: {e}. Reconnecting in 5s...")
                self.ws_connection = None
                await asyncio.sleep(5)

    def _process_ws_message(self, message: str) -> None:
        """Process an incoming WebSocket message and update order books."""
        try:
            data = json.loads(message)
            if data.get("event_type") == "book":
                asset_id = data.get("asset_id")
                if not asset_id:
                    return

                bids = sorted(
                    [(float(b["price"]), float(b["size"])) for b in data.get("bids", [])],
                    key=lambda x: x[0],
                    reverse=True,
                )
                asks = sorted(
                    [(float(a["price"]), float(a["size"])) for a in data.get("asks", [])],
                    key=lambda x: x[0],
                )

                self.order_books[asset_id] = OrderBook(
                    market_id=asset_id,
                    best_bid=bids[0][0] if bids else None,
                    best_ask=asks[0][0] if asks else None,
                    bids=bids,
                    asks=asks,
                )
        except Exception as e:
            logger.debug(f"Error processing WS message: {e}")

    async def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Get the cached order book for a specific token."""
        return self.order_books.get(token_id)

    # -------------------------------------------------------------------------
    # Order Execution (CLOB REST API) - STUB
    # -------------------------------------------------------------------------
    async def place_order(self, trade: ProposedTrade) -> Dict[str, Any]:
        """
        Place an order on the CLOB.

        WARNING: This method is currently a STUB. Real order placement requires
        L1/L2 authentication (deriving API keys from a private wallet key).

        Args:
            trade: A ProposedTrade object from the solver.

        Returns:
            A dict with the order result (currently simulated).
        """
        if not self.http_client:
            raise RuntimeError("Client not initialized.")

        # TODO(critical): Implement L1/L2 signing logic.
        # See: https://docs.polymarket.com/#authentication
        payload = {
            "token_id": trade.outcome_id,
            "price": str(trade.limit_price),
            "size": str(trade.size),
            "side": trade.side.value.upper(),
            "order_type": "LIMIT",
            "condition_id": trade.market_id,
        }
        logger.warning(
            "Order placement is a STUB. Credentials required. Payload: %s", payload
        )
        return {"status": "stub_simulated", "order_id": "mock_id"}
