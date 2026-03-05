"""
Polymarket API Client for PolyQuant 2.0

This module provides an async client for interacting with Polymarket's APIs.

POLYMARKET USES 4 APIS:
-----------------------
1. Gamma API (https://gamma-api.polymarket.com)
   - Market discovery and metadata
   - Event information and categories

2. CLOB API (https://clob.polymarket.com)
   - Prices and order books
   - Trading/order submission

3. Data API (https://data-api.polymarket.com)
   - Positions and activity
   - Historical data

4. WebSocket (wss://ws-subscriptions-clob.polymarket.com)
   - Real-time price updates
   - Live order book changes

USAGE:
------
    async with PolymarketClient() as client:
        # Uses Gamma API for discovery
        markets = await client.get_active_markets(limit=100)
        
        # Uses CLOB API for order books
        for market in markets:
            order_books = await client.get_all_order_books(market)
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Awaitable

import asyncio
import orjson
import httpx
from decimal import Decimal


from polyquant.data.market_models import (
    Market,
    OrderBook,
    OrderLevel,
    Outcome,
    ProposedTrade,
    OrderSide,
)
from polyquant.utils import config, get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Polymarket REST Client
# -----------------------------------------------------------------------------
class PolymarketClient:
    """
    Async client for Polymarket APIs.
    
    Uses separate endpoints for different operations:
    - Gamma API: Market discovery and metadata
    - CLOB API: Order books, prices, trading
    - Data API: Positions and history
    
    Example:
        client = PolymarketClient()
        
        async with client:
            markets = await client.get_active_markets(min_liquidity=1000)
            
            for market in markets:
                print(f"{market.question}: {market.liquidity}")
    """
    
    def __init__(self):
        """Initialize the Polymarket client with all API endpoints."""
        # API 1: Gamma - Market discovery
        self.gamma_url = config.polymarket_gamma_url
        
        # API 2: CLOB - Order books and trading
        self.clob_url = config.polymarket_clob_url
        
        # API 3: Data - Positions and history
        self.data_url = config.polymarket_data_url
        
        # API 4: WebSocket - Real-time updates
        self.ws_url = config.polymarket_ws_url
        
        # HTTP clients (initialized in __aenter__)
        self._gamma_client: httpx.AsyncClient | None = None
        self._clob_client: httpx.AsyncClient | None = None
        self._data_client: httpx.AsyncClient | None = None

        # py-clob-client SDK (initialized once in __aenter__, reused for all orders)
        self._sdk_client: Any = None
        
        # WebSocket client for real-time updates
        self._ws_client: Any = None

        # Rate limiting for CLOB order submission
        self._order_semaphore = asyncio.Semaphore(5)    # Max 5 concurrent orders
        self._order_rate_limit = asyncio.Semaphore(10)   # Max 10 orders/second
        self._rate_limit_task: asyncio.Task | None = None
        
        logger.info(
            "PolymarketClient initialized",
            gamma_url=self.gamma_url,
            clob_url=self.clob_url,
            data_url=self.data_url,
        )

    async def get_history(self, market_id: str, fidelity: int = 60) -> list[dict[str, Any]]:
        """
        Fetch historical prices for a market.
        
        Args:
            market_id: The market (condition) ID or token ID
            fidelity: Time resolution in minutes (default 60)
            
        Returns:
            List of price points [{"t": timestamp, "p": price}, ...]
        """
        # Endpoint: /prices-history?interval=1h&market=...
        # Note: Actual endpoint might vary. Using CLOB prices-history as standard.
        if not self._clob_client:
            return []
            
        try:
            # Example: https://clob.polymarket.com/prices-history?interval=1h&market=...
            # This is a best-effort guess at the endpoint schema.
            # Using 1h interval for correlation.
            resp = await self._clob_client.get(
                "/prices-history",
                params={"market": market_id, "interval": "1h"}
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("history", [])
        except Exception as e:
            logger.warning(f"Failed to fetch history for {market_id}: {e}")
            return []
    
    async def _retry_get(
        self,
        client: httpx.AsyncClient,
        path: str,
        params: dict[str, Any] | None = None,
        max_retries: int | None = None,
    ) -> httpx.Response:
        """
        GET request with retry + exponential backoff + 429 handling.

        Args:
            client: The httpx.AsyncClient to use.
            path: URL path (appended to client's base_url).
            params: Query parameters.
            max_retries: Override config.http_max_retries.

        Returns:
            httpx.Response on success.

        Raises:
            httpx.HTTPError: After all retries exhausted.
        """
        retries = max_retries if max_retries is not None else config.http_max_retries
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                response = await client.get(path, params=params)

                # Handle 429 rate limiting
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 2))
                    backoff = max(retry_after, 1.0 * (2 ** attempt))
                    logger.warning(
                        "Rate limited (429), backing off",
                        path=path,
                        attempt=attempt + 1,
                        backoff_seconds=backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue

                # Permanent errors: do NOT retry
                if response.status_code in (400, 401, 403, 404):
                    response.raise_for_status()

                response.raise_for_status()
                return response

            except httpx.HTTPError as e:
                last_error = e
                if attempt < retries:
                    # More aggressive backoff for generic HTTP errors (e.g. 502, ReadTimeout)
                    # Start at 2s, 4s, 8s...
                    backoff = 2.0 * (2 ** attempt)
                    
                    error_type = type(e).__name__
                    if isinstance(e, httpx.HTTPStatusError):
                        error_type = f"HTTP {e.response.status_code}"
                        
                    logger.warning(
                        f"Request failed ({error_type}), retrying",
                        path=path,
                        attempt=attempt + 1,
                        backoff_seconds=backoff,
                        error=str(e),
                    )
                    await asyncio.sleep(backoff)

        raise last_error  # type: ignore[misc]

    async def __aenter__(self) -> "PolymarketClient":
        """Initialize HTTP clients for all APIs."""
        headers = {"Accept": "application/json"}

        # Gamma API client (market discovery) — 10s timeout
        self._gamma_client = httpx.AsyncClient(
            base_url=self.gamma_url,
            timeout=config.gamma_timeout_seconds,
            headers=headers,
            verify=True,
        )

        # CLOB API client (order books, trading) — 5s timeout
        self._clob_client = httpx.AsyncClient(
            base_url=self.clob_url,
            timeout=config.clob_timeout_seconds,
            headers=headers,
            verify=True,
        )

        # Data API client (positions, history)
        self._data_client = httpx.AsyncClient(
            base_url=self.data_url,
            timeout=config.gamma_timeout_seconds,
            headers=headers,
            verify=True,
        )

        # Initialize py-clob-client SDK (cached for all order submissions)
        # This avoids re-deriving the signing key on every order (~50ms savings)
        try:
            from py_clob_client.client import ClobClient
            
            raw_key = config.polygon_private_key.get_secret_value()
            # Sanitize: strip whitespace and 0x prefix
            private_key = raw_key.strip()
            if private_key.startswith("0x"):
                private_key = private_key[2:]
            
            if private_key and len(private_key) >= 64:
                self._sdk_client = ClobClient(
                    host=self.clob_url,
                    chain_id=137,  # Polygon Mainnet
                    key=private_key,
                )
                logger.info("CLOB SDK client initialized (live trading ready)")
            elif raw_key:
                logger.warning(
                    "POLYGON_PRIVATE_KEY provided but seems invalid or too short. "
                    "Running in paper mode only."
                )
            else:
                logger.info("No POLYGON_PRIVATE_KEY — SDK client not initialized (paper mode only)")
        except ImportError:
            logger.warning(
                "py-clob-client not installed — live trading unavailable. "
                "Run: pip install py-clob-client"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize CLOB SDK: {e}")
        
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Close all HTTP clients."""
        if self._gamma_client:
            await self._gamma_client.aclose()
        if self._clob_client:
            await self._clob_client.aclose()
        if self._data_client:
            await self._data_client.aclose()
    
    async def get_active_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        min_liquidity: float = 0.0,
    ) -> tuple[list[Market], int]:
        """
        Fetch active markets from Polymarket.
        
        Args:
            limit: Maximum number of markets to return
            min_liquidity: Minimum liquidity threshold in dollars
            
        Returns:
            Tuple of (List of Market objects, raw_count_fetched)
        """
        if not self._gamma_client:
            raise RuntimeError("Client not initialized. Use 'async with client:'")
        
        logger.debug("Fetching active markets", limit=limit, min_liquidity=min_liquidity)
        
        try:
            # Fetch markets from GAMMA API (not CLOB!) — with retry
            response = await self._retry_get(
                self._gamma_client,
                "/markets",
                params={
                    "limit": limit,
                    "offset": offset,
                    "active": True,
                },
            )
            data = response.json()
            
            markets = []
            for item in data:
                # Parse market data
                market = self._parse_market(item)
                
                # Apply liquidity filter
                if market.liquidity >= min_liquidity:
                    markets.append(market)
            
            logger.info(
                f"Fetched {len(data)} markets ({len(markets)} after filtering)",
                total=len(data),
                after_filter=len(markets),
            )
            
            return markets, len(data)
            
        except httpx.HTTPError as e:
            logger.error("Failed to fetch markets", error=str(e))
            return [], 0
    
    async def get_active_events(
        self,
        min_liquidity: float = 1000.0,
        max_events: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Fetch active events from Polymarket, sorted by liquidity descending.
        
        Uses the /events endpoint which returns pre-grouped markets within
        each event, drastically reducing the number of API calls needed.
        Automatically stops when event liquidity drops below the threshold.
        
        Args:
            min_liquidity: Stop fetching when event liquidity drops below this.
            max_events: Maximum events to return (0 = all above threshold).
            
        Returns:
            List of event dicts, each containing parsed Market objects in 'markets' key.
        """
        if not self._gamma_client:
            raise RuntimeError("Client not initialized. Use 'async with client:'")
        
        logger.info(
            "Fetching active events",
            min_liquidity=min_liquidity,
            max_events=max_events if max_events > 0 else "ALL",
        )
        
        all_events: list[dict[str, Any]] = []
        offset = 0
        batch_size = 100
        
        try:
            while True:
                response = await self._retry_get(
                    self._gamma_client,
                    "/events",
                    params={
                        "limit": batch_size,
                        "offset": offset,
                        "active": True,
                        "closed": False,
                        "order": "liquidity",
                        "ascending": False,
                    },
                )
                data = response.json()
                
                if not data:
                    break
                
                hit_threshold = False
                for event_data in data:
                    event_liq = float(event_data.get("liquidity", 0) or 0)
                    
                    # Since sorted by liquidity desc, once we're below threshold, stop
                    if event_liq < min_liquidity:
                        hit_threshold = True
                        break
                    
                    # Parse markets within the event
                    raw_markets = event_data.get("markets", [])
                    parsed_markets = [self._parse_market(m) for m in raw_markets]
                    
                    all_events.append({
                        "event_id": event_data.get("id", ""),
                        "title": event_data.get("title", ""),
                        "slug": event_data.get("slug", ""),
                        "liquidity": event_liq,
                        "volume": float(event_data.get("volume", 0) or 0),
                        "markets": parsed_markets,
                        # API returns camelCase: negRiskMarketID
                        "neg_risk_market_id": (
                            event_data.get("negRiskMarketID")
                            or event_data.get("neg_risk_market_id")
                        ),
                        "tags": [t.get("label", "") for t in event_data.get("tags", [])],
                    })
                
                offset += len(data)
                
                logger.debug(
                    "Events batch fetched",
                    batch=len(data),
                    total_so_far=len(all_events),
                    last_liquidity=f"${float(data[-1].get('liquidity', 0) or 0):,.0f}",
                )
                
                # Stop conditions
                if hit_threshold:
                    break
                if len(data) < batch_size:
                    break
                if max_events > 0 and len(all_events) >= max_events:
                    all_events = all_events[:max_events]
                    break
            
            total_markets = sum(len(e["markets"]) for e in all_events)
            logger.info(
                "Events fetched",
                events=len(all_events),
                total_markets=total_markets,
                api_calls=offset // batch_size + 1,
            )
            
            return all_events
            
        except httpx.HTTPError as e:
            logger.error("Failed to fetch events", error=str(e))
            return []
    
    async def get_order_book(self, token_id: str) -> OrderBook | None:
        """
        Get the order book for a specific token.
        
        Args:
            token_id: The token ID for the outcome
            
        Returns:
            OrderBook or None if not found
        """
        if not self._clob_client:
            raise RuntimeError("Client not initialized")
        
        try:
            # Use CLOB API for order books — with retry
            response = await self._retry_get(
                self._clob_client,
                "/book",
                params={"token_id": token_id},
            )
            data = response.json()

            return self._parse_order_book(token_id, data)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Market resolved/delisted — not an error, just no book available
                return None
            logger.warning("Failed to get order book", token_id=token_id, error=str(e))
            return None
        except httpx.HTTPError as e:
            logger.warning("Failed to get order book", token_id=token_id, error=str(e))
            return None
    
    async def get_all_order_books(self, market: Market) -> dict[str, OrderBook]:
        """
        Get order books for all outcomes in a market.
        
        Args:
            market: The market to get order books for
            
        Returns:
            Dict mapping outcome_id -> OrderBook
        """
        result = {}
        
        for outcome in market.outcomes:
            if outcome.token_id:
                ob = await self.get_order_book(outcome.token_id)
                if ob:
                    result[outcome.outcome_id] = ob
        
        return result
    
    async def get_positions(self, user_address: str = "") -> list[dict[str, Any]]:
        """
        Get positions from Data API.
        
        Args:
            user_address: Wallet address (optional via Data API auth)
            
        Returns:
            List of position dictionaries
        """
        if not self._data_client:
            raise RuntimeError("Client not initialized")
            
        try:
            # Note: Data API usually requires distinct authentication or
            # specific query params. For now we use the general endpoint structure.
            response = await self._data_client.get(
                "/positions",
                params={"user": user_address} if user_address else {},
            )
            # 404 is expected until we have real auth/endpoints
            if response.status_code == 404:
                return []
                
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning("Failed to get positions", error=str(e))
            return []

    # -------------------------------------------------------------------------
    # Order Execution (CLOB REST API) - REAL IMPLEMENTATION
    # -------------------------------------------------------------------------
    async def place_order(self, trade: ProposedTrade) -> dict[str, Any]:
        """
        Place an order on the CLOB using the cached py-clob-client SDK.

        Uses Fill-or-Kill (FOK) by default for all-or-nothing execution.
        The SDK client is initialized once in __aenter__ and reused here
        to avoid re-deriving the signing key on every order.

        The sync SDK calls are wrapped in asyncio.to_thread() so they
        don't block the Navigator's event loop during EIP-712 signing.

        Args:
            trade: A ProposedTrade object from the solver.

        Returns:
            A dict with the order result from the CLOB API.
        """
        if not self._sdk_client:
            logger.error(
                "CLOB SDK not initialized. Check POLYGON_PRIVATE_KEY and "
                "py-clob-client installation."
            )
            return {"status": "error", "reason": "sdk_not_initialized"}

        # Rate limit: max 5 concurrent, 10/second
        await self._order_rate_limit.acquire()
        async with self._order_semaphore:
            try:
                from py_clob_client.order_builder.constants import BUY, SELL
                from py_clob_client.clob_types import OrderType

                # Map our TimeInForce to Polymarket's OrderType enum
                order_type_map = {
                    "FOK": OrderType.FOK,
                    "FAK": OrderType.FOK,  # FAK not always available, fall back to FOK
                    "GTC": OrderType.GTC,
                    "GTD": OrderType.GTC,  # GTD needs expiration, fall back to GTC
                }
                order_type = order_type_map.get(
                    trade.time_in_force.value, OrderType.FOK
                )

                side = BUY if trade.side == OrderSide.BUY else SELL

                # Create and sign the order (sync) — run in thread to avoid blocking
                sdk = self._sdk_client
                order = await asyncio.to_thread(
                    sdk.create_order,
                    token_id=trade.outcome_id,
                    price=float(trade.limit_price),
                    size=float(trade.size),
                    side=side,
                )

                # Submit to CLOB with FOK (sync) — run in thread
                result = await asyncio.to_thread(
                    sdk.post_order, order, order_type
                )

                logger.info(
                    "Order submitted",
                    order_id=result.get("orderID"),
                    status=result.get("status"),
                    order_type=trade.time_in_force.value,
                    outcome_id=trade.outcome_id,
                    side=trade.side.value,
                    size=trade.size,
                    price=trade.limit_price,
                )

                return {
                    "status": "submitted",
                    "order_id": result.get("orderID"),
                    "raw_response": result,
                }

            except ImportError:
                logger.error("py-clob-client not installed. Run: pip install py-clob-client")
                return {"status": "error", "reason": "missing_dependency"}

            except Exception as e:
                logger.error("Order placement failed", error=str(e), outcome_id=trade.outcome_id)
                return {"status": "error", "reason": str(e)}

    async def place_orders(self, trades: list[ProposedTrade]) -> list[dict[str, Any]]:
        """
        Place multiple orders on the CLOB using a single batch request to reduce latency.
        """
        if not self._sdk_client:
            logger.error("CLOB SDK not initialized.")
            return [{"status": "error", "reason": "sdk_not_initialized"}] * len(trades)

        await self._order_rate_limit.acquire()
        async with self._order_semaphore:
            try:
                from py_clob_client.order_builder.constants import BUY, SELL
                from py_clob_client.clob_types import OrderType

                order_type_map = {
                    "FOK": OrderType.FOK,
                    "FAK": OrderType.FOK,
                    "GTC": OrderType.GTC,
                    "GTD": OrderType.GTC,
                }
                # Use same order type for the batch based on the first trade
                order_type = order_type_map.get(trades[0].time_in_force.value, OrderType.FOK)
                sdk = self._sdk_client

                # Create orders concurrently
                async def make_order(t: ProposedTrade):
                    side = BUY if t.side == OrderSide.BUY else SELL
                    return await asyncio.to_thread(
                        sdk.create_order,
                        token_id=t.outcome_id,
                        price=float(t.limit_price),
                        size=float(t.size),
                        side=side,
                    )
                    
                orders = await asyncio.gather(*(make_order(t) for t in trades))

                # Submit to CLOB (sync) — run in thread
                # Assuming post_orders returns a list of responses
                results = await asyncio.to_thread(sdk.post_orders, orders, order_type)
                
                # Make sure results is a list
                if not isinstance(results, list):
                    # Fallback if SDK returns a single dict with an array
                    if isinstance(results, dict) and "result" in results:
                         results = results["result"]
                    else:
                         results = [results] * len(trades)

                parsed_results = []
                for i, trade in enumerate(trades):
                    res = results[i] if i < len(results) else {}
                    
                    logger.info(
                        "Batch order submitted",
                        order_id=res.get("orderID"),
                        status=res.get("status"),
                        outcome_id=trade.outcome_id,
                        size=trade.size,
                    )
                    
                    if res.get("error"):
                         parsed_results.append({
                             "status": "error",
                             "reason": res.get("errorMsg", "Unknown error"),
                         })
                    else:
                         parsed_results.append({
                             "status": "submitted",
                             "order_id": res.get("orderID"),
                             "raw_response": res,
                         })

                return parsed_results

            except Exception as e:
                logger.error("Batch placement failed", error=str(e))
                return [{"status": "error", "reason": str(e)}] * len(trades)

    async def cancel_all_orders(self) -> dict[str, Any]:
        """
        Cancel all open orders on the CLOB and verify cancellation.

        Called by the kill switch — bypasses rate limiting since this is
        an emergency path that must execute immediately.

        P1-H4: After cancel_all, polls open orders to confirm none remain.
        """
        if not self._sdk_client:
            logger.error("Cannot cancel: SDK not initialized")
            return {"status": "error", "reason": "sdk_not_initialized"}
        try:
            result = await asyncio.to_thread(self._sdk_client.cancel_all)
            logger.critical("ALL ORDERS CANCELLED", result=result)

            # H4: Verify cancellation — poll for open orders
            verified = False
            for attempt in range(3):
                await asyncio.sleep(0.5 * (attempt + 1))
                try:
                    open_orders = await asyncio.to_thread(
                        self._sdk_client.get_orders
                    )
                    if not open_orders or len(open_orders) == 0:
                        verified = True
                        logger.info("Cancel verification: all orders confirmed cancelled")
                        break
                    else:
                        logger.warning(
                            "Cancel verification: orders still open",
                            remaining=len(open_orders),
                            attempt=attempt + 1,
                        )
                        # Retry cancel
                        await asyncio.to_thread(self._sdk_client.cancel_all)
                except Exception as e:
                    logger.warning("Cancel verification poll failed", error=str(e))

            if not verified:
                logger.critical(
                    "CANCEL VERIFICATION FAILED — orders may still be live on CLOB"
                )
                raise RuntimeError(
                    "Cancel verification failed after 3 attempts — "
                    "open orders may remain on CLOB. Manual intervention required."
                )

            return {
                "status": "cancelled",
                "verified": verified,
                "raw_response": result,
            }
        except Exception as e:
            logger.critical("CANCEL_ALL FAILED — manual intervention required", error=str(e))
            return {"status": "error", "reason": str(e)}

    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        """
        Poll CLOB for the status of a specific order.

        Returns:
            Dict with at least 'status' key: 'matched', 'delayed', 'live', 'cancelled', etc.
        """
        if not self._sdk_client:
            return {"status": "error", "reason": "sdk_not_initialized"}
        try:
            result = await asyncio.to_thread(
                self._sdk_client.get_order, order_id
            )
            return result if isinstance(result, dict) else {"status": "unknown", "raw": result}
        except Exception as e:
            logger.warning("get_order_status failed", order_id=order_id, error=str(e))
            return {"status": "error", "reason": str(e)}

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """
        Cancel a specific order on the CLOB.

        Used to clean up ghost/delayed orders that were never confirmed.
        """
        if not self._sdk_client:
            return {"status": "error", "reason": "sdk_not_initialized"}
        try:
            result = await asyncio.to_thread(
                self._sdk_client.cancel, order_id
            )
            logger.info("Order cancelled", order_id=order_id, result=result)
            return {"status": "cancelled", "raw_response": result}
        except Exception as e:
            logger.warning("cancel_order failed", order_id=order_id, error=str(e))
            return {"status": "error", "reason": str(e)}

    def get_stale_tokens(self) -> set[str]:
        """Return set of token IDs with detected WS sequence gaps."""
        if hasattr(self, 'ws_client') and self.ws_client:
            return getattr(self.ws_client, '_stale_assets', set()).copy()
        return set()

    async def get_usdc_balance(self) -> Decimal:
        """
        Fetch the current USDC balance/allowance from the CLOB.

        This is called once at startup and refreshed periodically in the
        background (every 60s). It is NEVER called on the hot path.

        Returns:
            Current USDC balance as Decimal, or Decimal("0") on failure.
        """
        if not self._sdk_client:
            logger.info("SDK not initialized (Paper Mode). Returning default $10,000.00 paper balance.")
            return Decimal("10000.00")

        try:
            # py-clob-client exposes get_balance_allowance for USDC
            result = await asyncio.to_thread(
                self._sdk_client.get_balance_allowance,
            )
            # result typically has {'balance': '...', 'allowance': '...'}
            raw_balance = result.get("balance", "0")
            balance = Decimal(str(raw_balance))
            logger.info("USDC balance fetched", balance=str(balance))
            return balance
        except Exception as e:
            logger.error("Failed to fetch USDC balance", error=str(e))
            return Decimal("0")

    def is_ws_healthy(self, max_age_seconds: float = 30.0) -> bool:
        """Whether the WebSocket connection is receiving fresh data."""
        if hasattr(self, 'ws_client') and self.ws_client:
            return self.ws_client.is_connection_healthy(max_age_seconds=max_age_seconds)
        return False  # No WS = not healthy

    
    def _parse_market(self, data: dict[str, Any]) -> Market:
        """Parse API response into Market object."""
        outcomes = []
        
        # In the /events endpoint, tokens aren't provided directly. 
        # Instead, we get outcomes, outcomePrices, and clobTokenIds as JSON-encoded strings
        tokens = data.get("tokens")
        if tokens:
            for token in tokens:
                outcomes.append(
                    Outcome(
                        outcome_id=token.get("token_id", ""),
                        name=token.get("outcome", "Unknown"),
                        price=Decimal(str(token.get("price", "0.5"))),
                        token_id=token.get("token_id", ""),
                    )
                )
        else:
            # Try to parse from the flattened event market structure
            try:
                import json
                outcome_names = data.get("outcomes", "[]")
                outcome_prices = data.get("outcomePrices", "[]")
                clob_token_ids = data.get("clobTokenIds", "[]")
                
                if isinstance(outcome_names, str):
                    outcome_names = json.loads(outcome_names)
                if isinstance(outcome_prices, str):
                    outcome_prices = json.loads(outcome_prices)
                if isinstance(clob_token_ids, str):
                    clob_token_ids = json.loads(clob_token_ids)
                    
                if isinstance(outcome_names, list) and isinstance(outcome_prices, list):
                    for i, name in enumerate(outcome_names):
                        price = outcome_prices[i] if i < len(outcome_prices) else "0.5"
                        token_id = clob_token_ids[i] if isinstance(clob_token_ids, list) and i < len(clob_token_ids) else ""
                        
                        outcomes.append(
                            Outcome(
                                outcome_id=token_id,
                                name=str(name),
                                price=Decimal(str(price)),
                                token_id=token_id,
                            )
                        )
            except Exception as e:
                logger.warning("Failed to parse flat outcomes strings", error=str(e), market=data.get("condition_id"))
        
        # Parse end date
        end_date = None
        if data.get("end_date_iso"):
            try:
                end_date = datetime.fromisoformat(
                    data["end_date_iso"].replace("Z", "+00:00")
                )
            except ValueError:
                pass
        
        return Market(
            market_id=data.get("condition_id", ""),
            question=data.get("question", ""),
            description=data.get("description", ""),
            outcomes=outcomes,
            volume=float(data.get("volume", 0)),
            liquidity=float(data.get("liquidity", 0)),
            end_date=end_date,
            resolved=data.get("closed", False),
            
            # Phase 5: Enhanced Metadata Extraction
            negrisk=data.get("neg_risk", False) or data.get("negrisk", False),
            group_id=data.get("group_id") or data.get("series_id"),
            market_type=data.get("type") or data.get("market_type"),

            # Phase 6: Conditional market support
            conditional_parent_id=data.get("conditional_id") or data.get("parent_market"),
            resolution_source=data.get("resolution_source") or data.get("uma_resolution_source"),
        )
    
    def _parse_order_book(self, token_id: str, data: dict[str, Any]) -> OrderBook:
        """Parse API response into OrderBook object."""
        bids = []
        asks = []
        
        for bid in data.get("bids", []):
            bids.append(
                OrderLevel(
                    price=Decimal(str(bid.get("price", "0"))),
                    size=Decimal(str(bid.get("size", "0"))),
                )
            )
        
        for ask in data.get("asks", []):
            asks.append(
                OrderLevel(
                    price=Decimal(str(ask.get("price", "0"))),
                    size=Decimal(str(ask.get("size", "0"))),
                )
        )
        
        return OrderBook(
            outcome_id=token_id,
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
        )

    def is_ws_healthy(self, max_age_seconds: float = 30.0) -> bool:
        """
        Check if the WebSocket connection is healthy.
        
        Delegates to PolymarketWSClient.is_connection_healthy.
        Returns False if no WS client exists.
        """
        ws = getattr(self, "ws_client", None)
        if ws is None:
            return False
        return ws.is_connection_healthy(max_age_seconds=max_age_seconds)

    @property
    def ws_stale_assets(self) -> set[str]:
        """Get the set of asset IDs with potentially stale WS data (sequence gaps)."""
        ws = getattr(self, "ws_client", None)
        if ws is None:
            return set()
        return ws._stale_assets

class PolymarketWSClient:
    """
    WebSocket client for real-time Polymarket CLOB data.
    
    Manages connections, subscriptions, and updates for Level 2 order books.
    
    Usage:
        ws_client = PolymarketWSClient()
        
        async def on_update(book: OrderBook):
            print(f"Update for {book.outcome_id}: {book.mid_price}")
            
        await ws_client.connect()
        await ws_client.subscribe(["token_id_1", "token_id_2"], on_update)
    """
    
    def __init__(self):
        self.ws_url = config.polymarket_ws_url
        self._ws = None
        self._callbacks: dict[str, list[Callable[[OrderBook], Awaitable[None]]]] = {}
        self._limit_client = None  # Helper for parsing
        self._running = False
        self._task: asyncio.Task | None = None

        # Reconnection tracking
        self._subscribed_tokens: list[str] = []  # Track tokens for re-subscription
        self._reconnect_backoff = 1.0  # Start with 1 second
        self._max_backoff = 60.0  # Max 60 seconds between retries
        self._reconnect_attempts = 0

        # Connection health monitoring
        self._last_message_time: float = 0.0  # time.monotonic() of last message
        self._connection_healthy = False

        # Full order book tracking — maintained by merging snapshots + deltas
        self._books: dict[str, OrderBook] = {}  # asset_id → latest full book

        # Sequence tracking: detect missed messages → stale orderbook
        self._last_sequence: dict[str, int] = {}  # asset_id → last seen sequence
        self._sequence_gaps: int = 0  # Total gaps detected
        self._stale_assets: set[str] = set()  # Assets with potentially stale data
        
    async def connect(self) -> None:
        """Establish WebSocket connection."""
        import websockets
        import time

        logger.info("Connecting to Polymarket WS", url=self.ws_url)
        try:
            self._ws = await websockets.connect(self.ws_url)
            self._running = True
            self._last_message_time = time.monotonic()  # Initialize health tracking
            self._connection_healthy = True
            self._task = asyncio.create_task(self._listen())
            logger.info("Connected to Polymarket WS")
        except Exception as e:
            logger.error("Failed to connect to WS", error=str(e))
            raise

    async def _reconnect(self) -> None:
        """
        Reconnect to WebSocket and re-subscribe to all tokens.

        This is called automatically by _listen() when connection drops.
        """
        import websockets
        import json
        import time

        logger.info(
            "Reconnecting to Polymarket WS",
            url=self.ws_url,
            attempt=self._reconnect_attempts + 1
        )

        try:
            # Establish new connection
            self._ws = await websockets.connect(self.ws_url)
            self._last_message_time = time.monotonic()  # Reset health tracking
            self._connection_healthy = True
            logger.info("WebSocket reconnected successfully")

            # Re-subscribe to all previously subscribed tokens
            if self._subscribed_tokens:
                logger.info(
                    "Re-subscribing to tokens after reconnect",
                    count=len(self._subscribed_tokens)
                )

                msg = {
                    "assets_ids": self._subscribed_tokens,
                    "type": "market"
                }

                await self._ws.send(json.dumps(msg))
                logger.info("Re-subscription complete")

        except Exception as e:
            logger.error("Reconnection failed", error=str(e))
            self._ws = None
            self._connection_healthy = False
            raise
        
    async def subscribe(
        self,
        token_ids: list[str],
        callback: Callable[[OrderBook], Awaitable[None]]
    ) -> None:
        """
        Subscribe to Level 2 updates for specific tokens.

        Args:
            token_ids: List of outcome token IDs
            callback: Async function to call with updated OrderBook
        """
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        import json

        # Register callbacks
        for tid in token_ids:
            if tid not in self._callbacks:
                self._callbacks[tid] = []
            self._callbacks[tid].append(callback)

            # Track subscribed tokens for re-subscription after reconnect
            if tid not in self._subscribed_tokens:
                self._subscribed_tokens.append(tid)

        # Send subscription message
        # Protocol: https://docs.polymarket.com/#websocket-subscriptions
        msg = {
            "assets_ids": token_ids,
            "type": "market"
        }

        await self._ws.send(json.dumps(msg))
        logger.info("Subscribed to tokens", count=len(token_ids))
        
    async def _listen(self) -> None:
        """Main listener loop with automatic reconnection."""
        logger.info("WS listener started")

        while self._running:
            try:
                # Ensure we have a connection
                if not self._ws:
                    await self._reconnect()
                    continue

                # Reset backoff on successful connection
                self._reconnect_backoff = 1.0
                self._reconnect_attempts = 0

                # Iterate over messages
                async for msg_str in self._ws:
                    # Update connection health tracking
                    import time
                    self._last_message_time = time.monotonic()
                    self._connection_healthy = True

                    try:
                        # orjson.loads() is 2-3x faster than stdlib json
                        data = orjson.loads(msg_str)
                        await self._handle_message(data)
                    except (ValueError, TypeError) as e:
                        # orjson raises ValueError for invalid JSON
                        logger.warning("Received invalid JSON from WS", error=str(e))
                    except Exception as e:
                        logger.error("Error handling WS message", error=str(e))

            except Exception as e:
                logger.error("WebSocket connection dropped", error=str(e))

                # Close current connection
                if self._ws:
                    try:
                        await self._ws.close()
                    except:
                        pass
                    self._ws = None

                # Reconnect with exponential backoff
                if self._running:
                    logger.warning(
                        "Attempting reconnection",
                        backoff_seconds=self._reconnect_backoff,
                        attempt=self._reconnect_attempts + 1
                    )
                    await asyncio.sleep(self._reconnect_backoff)

                    # Exponential backoff: double each time, up to max
                    self._reconnect_backoff = min(
                        self._reconnect_backoff * 2,
                        self._max_backoff
                    )
                    self._reconnect_attempts += 1
                
    async def _handle_message(self, data: Any) -> None:
        """
        Handle incoming WebSocket messages.
        
        Polymarket sends two event types:
        - 'book':         Full order book snapshot → replace entire book
        - 'price_change':  Delta update → merge into existing book
        
        We maintain a full book per asset in self._books and always
        emit the FULL merged book to callbacks, never a partial.
        """
        # API sometimes sends a list, sometimes a dict
        items = data if isinstance(data, list) else [data]
            
        for item in items:
            # Check for asset_id and order book data
            asset_id = item.get("asset_id")
            if not asset_id:
                continue

            event_type = item.get("event_type", "book")  # default to snapshot

            # --- Sequence tracking ---
            seq = item.get("sequence") or item.get("seq")
            if seq is not None:
                try:
                    seq_num = int(seq)
                    prev = self._last_sequence.get(asset_id)
                    if prev is not None and seq_num > prev + 1:
                        gap = seq_num - prev - 1
                        self._sequence_gaps += gap
                        self._stale_assets.add(asset_id)
                        logger.warning(
                            "WS sequence gap detected — orderbook may be stale",
                            asset_id=asset_id,
                            expected=prev + 1,
                            received=seq_num,
                            missed=gap,
                            total_gaps=self._sequence_gaps,
                        )
                        # On gap, request a fresh snapshot via REST in background
                        # (book will self-heal on next full snapshot from WS)
                    else:
                        # Sequence is continuous, clear stale flag
                        self._stale_assets.discard(asset_id)
                    self._last_sequence[asset_id] = seq_num
                except (ValueError, TypeError):
                    pass  # Non-integer sequence, skip tracking

            bids_raw = item.get("bids", [])
            asks_raw = item.get("asks", [])

            if not bids_raw and not asks_raw:
                continue

            if event_type == "price_change" and asset_id in self._books:
                # DELTA: merge changed levels into existing book
                book = self._merge_book_delta(asset_id, bids_raw, asks_raw)
            else:
                # SNAPSHOT (or first message for this asset): full replacement
                book = self._parse_ws_book(asset_id, bids_raw, asks_raw)
                self._books[asset_id] = book

            # Notify callbacks with the FULL merged book
            if asset_id in self._callbacks:
                for cb in self._callbacks[asset_id]:
                    try:
                        await cb(book)
                    except Exception as e:
                        logger.error("Callback failed", error=str(e))

    def _merge_book_delta(
        self, asset_id: str, bids_raw: list, asks_raw: list
    ) -> OrderBook:
        """
        Merge delta changes into the existing full book.
        
        Rules:
        - size > 0 at a price: update/add that level
        - size == 0 at a price: remove that level
        
        Returns the FULL updated book (not just the delta).
        """
        existing = self._books[asset_id]
        
        # Build dicts keyed by price for O(1) lookups
        bid_map: dict[Decimal, Decimal] = {b.price: b.size for b in existing.bids}
        ask_map: dict[Decimal, Decimal] = {a.price: a.size for a in existing.asks}
        
        # Apply bid deltas
        for b in bids_raw:
            try:
                price = Decimal(str(b.get("price", "0")))
                size = Decimal(str(b.get("size", "0")))
                if size > 0:
                    bid_map[price] = size  # Add or update
                else:
                    bid_map.pop(price, None)  # Remove
            except (ValueError, TypeError):
                continue
        
        # Apply ask deltas
        for a in asks_raw:
            try:
                price = Decimal(str(a.get("price", "0")))
                size = Decimal(str(a.get("size", "0")))
                if size > 0:
                    ask_map[price] = size  # Add or update
                else:
                    ask_map.pop(price, None)  # Remove
            except (ValueError, TypeError):
                continue
        
        # Reconstruct sorted OrderBook
        book = OrderBook(
            outcome_id=asset_id,
            bids=sorted(
                [OrderLevel(price=p, size=s) for p, s in bid_map.items()],
                key=lambda x: x.price, reverse=True
            ),
            asks=sorted(
                [OrderLevel(price=p, size=s) for p, s in ask_map.items()],
                key=lambda x: x.price
            ),
            timestamp=datetime.utcnow()
        )
        
        # Store the merged book
        self._books[asset_id] = book
        return book

    def _parse_ws_book(self, token_id: str, bids_raw: list, asks_raw: list) -> OrderBook:
        """Parse raw WS data into a full OrderBook (for snapshot events)."""
        bids = []
        for b in bids_raw:
            try:
                price = Decimal(str(b.get("price", "0")))
                size = Decimal(str(b.get("size", "0")))
                if size > 0:  # Skip zero-size levels in snapshots
                    bids.append(OrderLevel(price=price, size=size))
            except (ValueError, TypeError):
                continue
                
        asks = []
        for a in asks_raw:
            try:
                price = Decimal(str(a.get("price", "0")))
                size = Decimal(str(a.get("size", "0")))
                if size > 0:  # Skip zero-size levels in snapshots
                    asks.append(OrderLevel(price=price, size=size))
            except (ValueError, TypeError):
                continue
                
        return OrderBook(
            outcome_id=token_id,
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
            timestamp=datetime.utcnow()
        )

    def is_connection_healthy(self, max_age_seconds: float = 30.0) -> bool:
        """
        Check if WebSocket connection is healthy.

        A connection is considered unhealthy if:
        - Not connected (!_connection_healthy)
        - No message received for max_age_seconds

        Args:
            max_age_seconds: Maximum seconds since last message (default 30)

        Returns:
            True if connection is healthy, False otherwise
        """
        import time

        if not self._connection_healthy:
            return False

        if self._last_message_time == 0.0:
            # Never received a message
            return False

        age_seconds = time.monotonic() - self._last_message_time

        if age_seconds > max_age_seconds:
            logger.warning(
                "Connection health check failed",
                age_seconds=age_seconds,
                max_age_seconds=max_age_seconds,
                reason="No messages received recently"
            )
            return False

        return True

    def get_connection_age(self) -> float:
        """
        Get time in seconds since last WebSocket message.

        Returns:
            Seconds since last message, or -1 if never received
        """
        import time

        if self._last_message_time == 0.0:
            return -1.0

        return time.monotonic() - self._last_message_time

    async def close(self):
        """Close the connection."""
        self._running = False
        self._connection_healthy = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
