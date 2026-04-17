"""
Limitless Exchange API Client

Provides an async REST client for connecting to Limitless on the Base network.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from decimal import Decimal

import httpx
from eth_account import Account
from eth_account.messages import encode_typed_data
from web3 import Web3

from polyquant.utils import config, get_logger

logger = get_logger(__name__)

CHAIN_ID_BASE = 8453
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# Internal encoding convention for Limitless token_ids.
#
# Limitless's /markets/{slug}/orderbook endpoint returns a single YES-denominated
# book. We synthesize the NO side by inverting prices (1 - p). To keep a single
# token_id namespace across the rest of PolyQuant, we suffix the slug with a
# LOGICAL side marker: LIMITLESS_YES_SUFFIX for the raw book, LIMITLESS_NO_SUFFIX
# for the inverted synthetic NO book.
#
# This is OUR convention, not an API-imposed ordering. All code that constructs
# Limitless token_ids must use limitless_yes_token() / limitless_no_token() —
# never raw "_0"/"_1" string literals, which are easy to typo into a silent
# sign flip.
LIMITLESS_YES_SUFFIX = "_yes"
LIMITLESS_NO_SUFFIX = "_no"


def limitless_yes_token(slug: str) -> str:
    return f"{slug}{LIMITLESS_YES_SUFFIX}"


def limitless_no_token(slug: str) -> str:
    return f"{slug}{LIMITLESS_NO_SUFFIX}"


def limitless_market_to_market(raw: Dict[str, Any]) -> "Any | None":
    """
    Build a polyquant.data.market_models.Market instance from a raw Limitless
    dict (as returned by `LimitlessClient.get_markets`).

    Used by Gap 5b Limitless-first discovery so standalone Limitless binary
    markets can be clustered as `native_partition` on the Limitless side
    (YES + NO = 1), unlocking pure-Limitless intra-market arb.

    Returns None if the dict is missing a slug or lacks the minimum fields
    needed to construct a Market. All fields are best-effort — downstream
    code already handles missing end_date / volume gracefully.
    """
    from datetime import datetime as _dt
    from decimal import Decimal as _Dec
    from polyquant.data.market_models import Market, Outcome

    slug = raw.get("slug")
    if not isinstance(slug, str) or not slug:
        return None
    title = raw.get("title") or ""
    if not title:
        return None

    def _parse_ts(val: Any) -> "_dt | None":
        if val is None or val == "":
            return None
        if isinstance(val, (int, float)):
            ts = float(val)
            if ts > 1e12:
                ts /= 1000.0
            try:
                return _dt.fromtimestamp(ts)
            except (OSError, ValueError, OverflowError):
                return None
        if isinstance(val, str):
            s = val.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            try:
                return _dt.fromisoformat(s)
            except ValueError:
                return None
        return None

    end_raw = (
        raw.get("expirationDate")
        or raw.get("expirationTimestamp")
        or raw.get("endDate")
    )
    end_date = _parse_ts(end_raw)

    try:
        vol = float(raw.get("volumeFormatted") or raw.get("volume") or 0)
        liq = float(raw.get("liquidityFormatted") or raw.get("liquidity") or 0)
    except (TypeError, ValueError):
        vol, liq = 0.0, 0.0
    # Limitless sometimes ships raw micro-USDC; normalize the same way
    # exchange_matcher._limitless_dollars does.
    if vol > 1_000_000:
        vol /= 1_000_000
    if liq > 1_000_000:
        liq /= 1_000_000

    yes_outcome = Outcome(
        outcome_id=limitless_yes_token(slug),
        name="Yes",
        price=_Dec("0.5"),
        token_id=limitless_yes_token(slug),
    )
    no_outcome = Outcome(
        outcome_id=limitless_no_token(slug),
        name="No",
        price=_Dec("0.5"),
        token_id=limitless_no_token(slug),
    )

    return Market(
        market_id=slug,
        question=title,
        description=raw.get("description") or "",
        outcomes=[yes_outcome, no_outcome],
        volume=vol,
        liquidity=liq,
        end_date=end_date,
        resolved=False,
        slug=slug,
        resolution_source=(
            raw.get("resolutionSource")
            or raw.get("rules")
            or None
        ),
    )


def parse_limitless_token(token_id: str) -> tuple[str, bool] | None:
    """
    Parse a Limitless token_id into (slug, is_yes).

    Returns None if the token_id does not use the PolyQuant Limitless encoding.
    Accepts both the current "_yes"/"_no" logical suffixes and the legacy
    "_0"/"_1" numeric suffixes for manifests written before the convention
    migration.
    """
    if token_id.endswith(LIMITLESS_YES_SUFFIX):
        return token_id[: -len(LIMITLESS_YES_SUFFIX)], True
    if token_id.endswith(LIMITLESS_NO_SUFFIX):
        return token_id[: -len(LIMITLESS_NO_SUFFIX)], False
    # Legacy numeric form.
    try:
        slug, side_str = token_id.rsplit("_", 1)
        side = int(side_str)
        if side in (0, 1):
            return slug, side == 0
    except ValueError:
        pass
    return None

class LimitlessClient:
    """Async client for Limitless Exchange APIs on Base."""
    
    def __init__(self):
        self.api_url = config.limitless_api_url
        self.api_key = config.limitless_api_key.get_secret_value()
        raw_key = config.base_private_key.get_secret_value()
        # Sanitize: strip whitespace and 0x prefix
        self.private_key = raw_key.strip()
        if self.private_key.startswith("0x"):
            self.private_key = self.private_key[2:]
            
        self._rest_client: httpx.AsyncClient | None = None
        self._base_nonce: int | None = None
        
        if self.private_key and len(self.private_key) >= 64:
            try:
                self.account = Account.from_key(self.private_key)
                self.address = self.account.address
            except Exception as e:
                logger.warning(f"Failed to initialize Limitless account with key: {e}")
                self.account = None
                self.address = ZERO_ADDRESS
        else:
            self.account = None
            self.address = ZERO_ADDRESS
            
        logger.info("LimitlessClient initialized", api_url=self.api_url, address=self.address)

    async def __aenter__(self) -> "LimitlessClient":
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            
        self._rest_client = httpx.AsyncClient(
            base_url=self.api_url,
            timeout=15.0, # Using explicit timeout as clob_timeout may not be in config
            headers=headers,
            verify=True
        )
        
        # Pre-warm nonce to avoid execution latency
        try:
            self._base_nonce = await self.get_base_network_nonce()
            logger.info("Limitless base nonce pre-warmed", nonce=self._base_nonce)
        except Exception as e:
            logger.error(f"Failed to pre-warm Limitless nonce: {e}")
            self._base_nonce = None
            
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._rest_client:
            await self._rest_client.aclose()

    async def _retry_request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Execute HTTP request with retry logic for 429 errors."""
        retries = config.http_max_retries
        last_error = None
        
        if not self._rest_client:
            raise RuntimeError("Client not initialized. Use 'async with client:'")

        for attempt in range(retries + 1):
            try:
                response = await self._rest_client.request(method, path, **kwargs)
                
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 2))
                    backoff = max(retry_after, 1.0 * (2 ** attempt))
                    logger.warning("Limitless Rate limited (429)", path=path, backoff=backoff)
                    await asyncio.sleep(backoff)
                    continue
                    
                response.raise_for_status()
                return response
                
            except httpx.HTTPError as e:
                last_error = e
                if attempt < retries:
                    backoff = 0.5 * (2 ** attempt)
                    await asyncio.sleep(backoff)
                    
        raise last_error # type: ignore

    async def get_markets(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch all active markets from Limitless via pagination."""
        all_markets = []
        page = 1

        while True:
            path = "/markets/active"
            res = await self._retry_request("GET", path, params={
                "limit": limit,
                "page": page
            })

            data = res.json()
            batch = data.get("data", []) or data.get("markets", [])
            
            if not batch:
                break
                
            all_markets.extend(batch)
            page += 1
            
            if len(batch) < limit:
                break
                
        logger.info(f"Fetched {len(all_markets)} active markets from Limitless.")
        return all_markets

    async def get_market(self, slug: str) -> Dict[str, Any]:
        """Fetch a specific market."""
        res = await self._retry_request("GET", f"/markets/{slug}")
        return res.json()

    async def get_order_book(self, token_id: str) -> Optional[Any]:
        """
        Fetch the order book for a given Limitless market outcome.

        Args:
            token_id: Must be built via limitless_yes_token(slug) or
                limitless_no_token(slug). Legacy "{slug}_0"/"{slug}_1" token
                ids are still accepted for backward compatibility with older
                constraint manifests but produce a warning.

        The Limitless /orderbook endpoint returns YES-denominated bids/asks.
        For the NO side we synthesize the book by inverting prices (p -> 1-p)
        and swapping bid<->ask, under the partition invariant p_yes + p_no = 1.
        """
        from polyquant.data.market_models import OrderBook, OrderLevel

        parsed = parse_limitless_token(token_id)
        if parsed is None:
            logger.error(f"Invalid Limitless token_id format: {token_id}")
            return None
        slug, is_yes = parsed
        if token_id.endswith(("_0", "_1")) and not token_id.endswith((LIMITLESS_YES_SUFFIX, LIMITLESS_NO_SUFFIX)):
            logger.warning(
                "Legacy numeric Limitless token_id — regenerate manifests to migrate",
                token_id=token_id,
            )

        try:
            res = await self._retry_request("GET", f"/markets/{slug}/orderbook")
            data = res.json()
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {slug}: {e}")
            return None

        book = OrderBook(outcome_id=token_id)

        raw_bids = data.get("bids", [])
        raw_asks = data.get("asks", [])

        if is_yes:
            for b in raw_bids:
                book.bids.append(OrderLevel(price=Decimal(str(b["price"])), size=Decimal(str(b["size"]))))
            for a in raw_asks:
                book.asks.append(OrderLevel(price=Decimal(str(a["price"])), size=Decimal(str(a["size"]))))
        else:
            # Synthetic NO book: YES Ask at price p becomes NO Bid at 1-p
            # (paying 1-p for NO is the complement of being willing to sell
            # YES at p), and vice versa. Only keep levels that stay inside
            # the (0, 1) probability range after inversion.
            for a in raw_asks:
                inv_price = Decimal("1.0") - Decimal(str(a["price"]))
                if inv_price > 0:
                    book.bids.append(OrderLevel(price=inv_price, size=Decimal(str(a["size"]))))
            for b in raw_bids:
                inv_price = Decimal("1.0") - Decimal(str(b["price"]))
                if inv_price < 1:
                    book.asks.append(OrderLevel(price=inv_price, size=Decimal(str(b["size"]))))

        book.bids.sort(key=lambda x: x.price, reverse=True)
        book.asks.sort(key=lambda x: x.price)

        return book
        
    async def get_usdc_balance(self) -> Decimal:
        """Fetch USDC balance on Base via RPC with failover, scaled by 1e6."""
        if not self.address or self.address == ZERO_ADDRESS:
            return Decimal("0.0")

        usdc_address = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

        # Minimal ERC20 balanceOf ABI call: 0x70a08231 + padded address
        data = f"0x70a08231000000000000000000000000{self.address[2:]}"
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": usdc_address, "data": data}, "latest"],
            "id": 1
        }

        try:
            resp = await self._rpc_call(payload)
            result = resp.get("result", "0x0")
            if result == "0x":
                result = "0x0"
            return Decimal(int(result, 16)) / Decimal(1e6)
        except Exception as e:
            logger.error(f"Failed to fetch Base USDC balance: {e}")
            return Decimal("0.0")

    async def _rpc_call(self, payload: dict) -> dict:
        """Execute an RPC call with failover across configured providers."""
        urls = [config.base_rpc_url] + list(config.base_rpc_fallback_urls)
        last_error = None

        async with httpx.AsyncClient() as client:
            for url in urls:
                try:
                    res = await client.post(url, json=payload, timeout=5.0)
                    res.raise_for_status()
                    return res.json()
                except Exception as e:
                    last_error = e
                    logger.warning(f"RPC call failed on {url}: {e}")
                    continue

        raise RuntimeError(f"All RPC providers failed. Last error: {last_error}")

    async def get_base_network_nonce(self) -> int:
        """Fetch the current transaction count (nonce) from Base RPC with failover.

        Uses "safe" block tag (~1 min lag on Base L2). "finalized" lags 12+ hours
        on Base because it means finalized to L1.
        """
        if not self.address or self.address == ZERO_ADDRESS:
            return int(time.time())

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getTransactionCount",
            "params": [self.address, "safe"],
            "id": 1
        }

        try:
            data = await self._rpc_call(payload)
            result = data.get("result", "0x0")
            if result == "0x":
                result = "0x0"
            return int(result, 16)
        except Exception as e:
            logger.error(f"Failed to fetch Base network nonce: {e}")
            return int(time.time())
                
    def sign_order(self, order_data: Dict[str, Any], verifying_contract: str) -> str:
        """Sign an order using EIP-712 for Limitless CTF Exchange."""
        if not self.private_key:
            raise RuntimeError("Cannot sign order without base_private_key configured.")
            
        domain = {
            "name": "Limitless CTF Exchange",
            "version": "1",
            "chainId": CHAIN_ID_BASE,
            "verifyingContract": Web3.to_checksum_address(verifying_contract),
        }

        types = {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "Order": [
                {"name": "salt", "type": "uint256"},
                {"name": "maker", "type": "address"},
                {"name": "signer", "type": "address"},
                {"name": "taker", "type": "address"},
                {"name": "tokenId", "type": "uint256"},
                {"name": "makerAmount", "type": "uint256"},
                {"name": "takerAmount", "type": "uint256"},
                {"name": "expiration", "type": "uint256"},
                {"name": "nonce", "type": "uint256"},
                {"name": "feeRateBps", "type": "uint256"},
                {"name": "side", "type": "uint8"},
                {"name": "signatureType", "type": "uint8"},
            ],
        }

        message = {
            "salt": int(order_data.get("salt", time.time_ns())),
            "maker": Web3.to_checksum_address(order_data["maker"]),
            "signer": Web3.to_checksum_address(order_data["signer"]),
            "taker": Web3.to_checksum_address(order_data.get("taker", ZERO_ADDRESS)),
            "tokenId": int(order_data["tokenId"]),
            "makerAmount": int(order_data["makerAmount"]),
            "takerAmount": int(order_data["takerAmount"]),
            "expiration": int(order_data.get("expiration", int(time.time()) + 60)),
            "nonce": int(order_data.get("nonce", 0)),
            "feeRateBps": int(order_data.get("feeRateBps", 0)),
            "side": int(order_data["side"]),
            "signatureType": int(order_data.get("signatureType", 0)),
        }

        typed_data = {
            "types": types,
            "primaryType": "Order",
            "domain": domain,
            "message": message,
        }

        encoded = encode_typed_data(typed_data)
        signed = Account.sign_message(encoded, private_key=self.private_key)
        return signed.signature.hex()

    async def place_order(self, market: Dict[str, Any], token_id: str, price: Decimal, size: Decimal, is_buy: bool) -> Dict[str, Any]:
        """
        Create, sign, and submit an FOK order to Limitless.
        Using 1e6 for USDC scale.
        """
        if self._base_nonce is None:
            self._base_nonce = await self.get_base_network_nonce()
            
        assert self._base_nonce is not None
        current_nonce = self._base_nonce
        self._base_nonce += 1
        
        # BUY (side 0): maker pays USDC, receives shares
        # SELL (side 1): maker pays shares, receives USDC
        if is_buy:
            maker_amount = int((price * size * Decimal(10**6)).to_integral_value())
            taker_amount = int((size * Decimal(10**6)).to_integral_value())
            side = 0
        else:
            maker_amount = int((size * Decimal(10**6)).to_integral_value())
            taker_amount = int((price * size * Decimal(10**6)).to_integral_value())
            side = 1
            
        order_payload = {
            "maker": self.address,
            "signer": self.address,
            "taker": ZERO_ADDRESS,
            "tokenId": token_id,
            "makerAmount": maker_amount,
            "takerAmount": taker_amount,
            "side": side,
            "salt": time.time_ns(),
            "expiration": int(time.time()) + config.order_expiration_seconds,
            "nonce": current_nonce,
            "feeRateBps": 0,
            "signatureType": 0,
            "orderType": "FOK"
        }
        
        verifying_contract = market.get("exchangeContract", ZERO_ADDRESS)
        
        signature = self.sign_order(order_payload, verifying_contract)
        
        submit_payload = {
            "order": order_payload,
            "signature": signature
        }
        
        res = await self._retry_request("POST", "/orders", json=submit_payload)
        return res.json()
