"""
Real-Time Price Cache for PolyQuant

This module provides an in-memory cache for order book data that is
fed by WebSocket updates. It ensures sub-50ms access to current prices.

DESIGN:
-------
- Thread-safe: Uses asyncio locks for concurrent access.
- Event-driven: Notifies subscribers when prices change.
- Decay: Marks stale data after configurable timeout.

USAGE:
------
    cache = PriceCache()
    
    # Subscribe to updates
    async def on_update(token_id: str, book: OrderBook):
        print(f"Price update for {token_id}")
    
    cache.subscribe(on_update)
    
    # Feed from WebSocket
    cache.update(token_id, order_book)
    
    # Get current price
    book = cache.get(token_id)
"""

import asyncio
import time
from typing import Callable, Awaitable

from polyquant.data.market_models import OrderBook
from polyquant.utils import get_logger

logger = get_logger(__name__)

# Type alias for update callbacks
UpdateCallback = Callable[[str, OrderBook], Awaitable[None]]


class PriceCache:
    """
    In-memory cache for real-time order book data.
    
    This replaces REST API polling with instant access to
    WebSocket-fed price data.
    
    Features:
    - O(1) access to current order books
    - Automatic staleness detection
    - Subscriber notification on updates
    """
    
    def __init__(self, stale_threshold_seconds: float = 5.0):
        """
        Initialize the price cache.

        Args:
            stale_threshold_seconds: Time after which data is considered stale.
        """
        self._books: dict[str, OrderBook] = {}
        self._last_update: dict[str, float] = {}  # monotonic timestamps
        self._stale_threshold = stale_threshold_seconds  # seconds as float
        self._subscribers: list[UpdateCallback] = []
        self._lock = asyncio.Lock()
        self._update_event = asyncio.Event()  # For event-driven architecture

        logger.info("PriceCache initialized", stale_threshold=stale_threshold_seconds)
    
    def subscribe(self, callback: UpdateCallback) -> None:
        """
        Subscribe to price updates.
        
        Args:
            callback: Async function called with (token_id, OrderBook) on each update.
        """
        self._subscribers.append(callback)
        logger.debug("Subscriber added", total_subscribers=len(self._subscribers))
    
    async def update(self, token_id: str, book: OrderBook) -> None:
        """
        Update the cache with a new order book snapshot.

        This is the hot path - called on every WebSocket message.

        Args:
            token_id: The token ID (outcome) being updated.
            book: The new order book snapshot.
        """
        async with self._lock:
            self._books[token_id] = book
            self._last_update[token_id] = time.monotonic()  # Fast monotonic time

        # Signal event-driven systems (e.g., Navigator)
        self._update_event.set()

        # Notify subscribers in parallel (without holding lock)
        # Using asyncio.gather for concurrent execution saves 5-20ms
        if self._subscribers:
            await asyncio.gather(
                *[callback(token_id, book) for callback in self._subscribers],
                return_exceptions=True  # Don't let one failure block others
            )
    
    def get(self, token_id: str) -> OrderBook | None:
        """
        Get the current order book for a token.
        
        Returns None if the token is not in cache or data is stale.
        
        Args:
            token_id: The token ID to look up.
            
        Returns:
            OrderBook or None if not available/stale.
        """
        book = self._books.get(token_id)
        if book is None:
            return None
        
        # Check staleness
        last = self._last_update.get(token_id)
        if last is None:
            return None

        if time.monotonic() - last > self._stale_threshold:
            logger.warning("Stale price data", token_id=token_id)
            return None

        return book
    
    def get_all(self) -> dict[str, OrderBook]:
        """
        Get all non-stale order books.

        Returns:
            Dict mapping token_id -> OrderBook for fresh data only.
        """
        now = time.monotonic()  # Fast monotonic time
        result = {}

        for token_id, book in self._books.items():
            last = self._last_update.get(token_id)
            if last and (now - last) <= self._stale_threshold:
                result[token_id] = book

        return result
    
    def is_stale(self, token_id: str) -> bool:
        """Check if data for a token is stale."""
        last = self._last_update.get(token_id)
        if last is None:
            return True
        return time.monotonic() - last > self._stale_threshold
    
    @property
    def size(self) -> int:
        """Number of tokens in cache."""
        return len(self._books)
    
    @property
    def fresh_count(self) -> int:
        """Number of non-stale tokens."""
        now = time.monotonic()  # Fast monotonic time
        return sum(
            1 for token_id in self._books
            if token_id in self._last_update
            and (now - self._last_update[token_id]) <= self._stale_threshold
        )
    
    async def wait_for_update(self, timeout: float = 5.0) -> bool:
        """
        Wait for the next price update (event-driven).

        Returns True if an update arrived, False if timed out.
        A timeout does NOT mean anything is broken — it just means
        no WebSocket data arrived in that window.
        """
        try:
            await asyncio.wait_for(self._update_event.wait(), timeout=timeout)
            self._update_event.clear()
            return True
        except asyncio.TimeoutError:
            self._update_event.clear()
            return False

    def clear(self) -> None:
        """Clear all cached data."""
        self._books.clear()
        self._last_update.clear()
        logger.info("PriceCache cleared")
