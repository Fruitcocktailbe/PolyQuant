"""
Configuration Management for PolyQuant 2.0

This module handles all configuration loading from environment variables.
It uses Pydantic for validation and type safety.

DESIGN DECISIONS:
-----------------
1. All config is loaded from environment variables for security (no hardcoded secrets)
2. Sensible defaults are provided where appropriate
3. Pydantic validates all values at startup to fail fast on misconfigurations
4. Config is a singleton - imported once and reused throughout the application

USAGE:
------
    from polyquant.utils.config import config
    
    print(config.gemini_api_key)
    print(config.extraction_alpha)
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class PolyQuantConfig(BaseSettings):
    """
    Central configuration for the PolyQuant system.
    
    All values are loaded from environment variables. See .env.example for
    a complete list of available configuration options.
    
    Attributes:
        gemini_api_key: API key for Google Gemini (Discovery, Validator)
        alchemy_api_key: API key for Alchemy (blockchain data)
        extraction_alpha: Target arbitrage extraction efficiency (0-1)
        max_drawdown: Maximum drawdown before kill switch (0-1)
        orderbook_depth_cap: Max position as fraction of order book (0-1)
        latency_target_ms: Target latency in milliseconds
        trading_mode: 'paper' for simulation, 'live' for real trading
    """
    
    # Tell Pydantic where to find our .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # Environment variables are case-insensitive
        extra="ignore",  # Ignore extra fields in .env
    )
    
    # =========================================================================
    # API Keys (stored as SecretStr for security - won't be printed in logs)
    # =========================================================================
    
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Google Gemini API Key",
    )

    openrouter_api_key: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "OpenRouter API key (optional, format: sk-or-v1-...). When set, "
            "free-tier OpenRouter models are added to the fallback chain, "
            "giving the pipeline a second-provider safety net when Google AI "
            "Studio's daily quota is exhausted. Get a key at openrouter.ai."
        ),
    )

    alchemy_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Alchemy API key for Polygon blockchain data"
    )

    polygon_private_key: SecretStr = Field(
        default=SecretStr(""),
        description="Polygon wallet private key for EIP-712 signing (0x...)"
    )

    base_private_key: SecretStr = Field(
        default=SecretStr(""),
        description="Base wallet private key for EIP-712 signing on Limitless (0x...)"
    )
    
    limitless_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Limitless API Key (lmts_...)"
    )
    
    # =========================================================================
    # Polymarket Connection Settings
    # =========================================================================
    
    # API 1: Gamma API - Market discovery and metadata
    polymarket_gamma_url: str = Field(
        default="https://gamma-api.polymarket.com",
        description="Polymarket Gamma API for market discovery and metadata"
    )
    
    # API 2: CLOB API - Prices, order books, and trading
    polymarket_clob_url: str = Field(
        default="https://clob.polymarket.com",
        description="Polymarket CLOB API for prices, orderbooks, and trading"
    )
    
    # API 3: Data API - Positions, activity, and history
    polymarket_data_url: str = Field(
        default="https://data-api.polymarket.com",
        description="Polymarket Data API for positions, activity, and history"
    )
    
    # API 4: WebSocket - Real-time updates
    polymarket_ws_url: str = Field(
        default="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        description="Polymarket WebSocket for real-time price and order updates"
    )

    gamma_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        description="Timeout for Gamma API requests in seconds"
    )

    clob_timeout_seconds: float = Field(
        default=5.0,
        ge=1.0,
        description="Timeout for CLOB API requests in seconds"
    )

    http_max_retries: int = Field(
        default=3,
        ge=0,
        description="Max HTTP retries for REST APIs"
    )

    # =========================================================================
    # Limitless Connection Settings
    # =========================================================================

    limitless_api_url: str = Field(
        default="https://api.limitless.exchange",
        description="Limitless REST API base URL"
    )

    limitless_ws_url: str = Field(
        default="wss://stream.limitless.exchange",
        description="Limitless WebSocket for real-time orderbooks"
    )

    base_rpc_url: str = Field(
        default="https://mainnet.base.org",
        description="Base RPC URL for fetching Limitless USDC balances"
    )

    base_rpc_fallback_urls: list[str] = Field(
        default_factory=list,
        description="Backup Base RPC URLs for failover (comma-separated in env: BASE_RPC_FALLBACK_URLS)"
    )

    # =========================================================================
    # Redis Cache Configuration
    # =========================================================================
    
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for caching and state management"
    )
    
    # =========================================================================
    # Trading Parameters
    # =========================================================================
    
    polymarket_taker_fee_pct: float = Field(
        default=0.02,
        description="Polymarket Taker Fee as a percentage (2% on the resting side per Polymarket docs)"
    )

    limitless_taker_fee_pct: float = Field(
        default=0.03,
        description="Limitless Taker Fee — conservative max. Real fees ramp 0.03%→3% as events approach resolution; default to the max to avoid approving rings that become unprofitable near expiry."
    )

    polygon_gas_per_tx: float = Field(
        default=0.05,
        description="Estimated Polygon gas cost per execution in USD/USDC (tune during congestion)"
    )

    base_gas_per_tx: float = Field(
        default=0.01,
        description="Estimated Base gas cost per execution in USD/USDC"
    )

    max_in_flight_capital: float = Field(
        default=500.0,
        description="Maximum total value of pending (unsettled) order capital across all exchanges."
    )
    
    extraction_alpha: float = Field(
        default=0.9,
        ge=0.0,  # Greater than or equal to 0
        le=1.0,  # Less than or equal to 1
        description="Target extraction efficiency (0.9 = capture 90% of arbitrage)"
    )
    
    max_drawdown: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Kill switch triggers when drawdown exceeds this (0.15 = 15%)"
    )
    
    orderbook_depth_cap: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Max position size as fraction of order book depth (0.5 = 50%)"
    )

    # ===== Dutching Strategy Enhancements =====
    max_partition_size: int = Field(
        default=5,
        ge=2,
        description="Maximum number of outcomes in a partition for dutching (prevents O(N²) blowup)"
    )

    partial_fill_probability: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Expected probability of partial fill (for unwind cost model)"
    )

    unwind_spread_estimate: float = Field(
        default=0.06,
        ge=0.0,
        le=0.5,
        description="Expected average unwind spread (3%→6%→9% average = 6%)"
    )
    
    latency_target_ms: int = Field(
        default=30,
        ge=1,
        description="Target decision-to-mempool latency in milliseconds"
    )
    
    ws_max_age_ms: int = Field(
        default=200,
        ge=10,
        description="Dead Man's Switch: max age of websocket messages before halting"
    )

    soft_staleness_start_ms: int = Field(
        default=50,
        ge=1,
        description="Staleness penalty begins at this quote age (ms). Linear decay from 1.0 at this threshold to 0.0 at ws_max_age_ms. Set above typical WS tick interval (~30-80 ms) to avoid penalizing normal jitter."
    )

    order_expiration_seconds: int = Field(
        default=6,
        ge=1,
        description="Limitless order expiration deadline (approx 3 Polygon blocks)"
    )
    
    vwap_slippage_limit: float = Field(
        default=0.05,
        ge=0.0,
        description="Maximum VWAP slippage in dollars before aborting trade"
    )
    
    solver_timeout_seconds: int = Field(
        default=30,
        ge=1,
        description="Maximum time for SCIP solver before timing out"
    )
    
    # =========================================================================
    # Frank-Wolfe Algorithm Parameters (from research papers)
    # =========================================================================
    
    initial_epsilon: float = Field(
        default=0.1,
        ge=0.001,
        le=0.5,
        description="Initial contraction parameter for Barrier Frank-Wolfe (Part 2)"
    )
    
    fw_min_profit: float = Field(
        default=0.05,
        ge=0.0,
        description="Minimum profit in USD for Frank-Wolfe arbitrage detection"
    )
    
    fw_max_iterations: int = Field(
        default=150,
        ge=10,
        description="Maximum Frank-Wolfe iterations before stopping"
    )

    # =========================================================================
    # Risk & Position Sizing
    # =========================================================================

    sizing_strategy: Literal["kelly", "dutching"] = Field(
        default="dutching",
        description="Position sizing strategy: 'kelly' (directional EV bets) or 'dutching' (risk-free arbitrage)"
    )

    kelly_fraction: float = Field(
        default=0.5,
        ge=0.01, le=1.0,
        description="Fraction of full Kelly to use (0.5 = Half Kelly)"
    )

    max_single_trade_pct: float = Field(
        default=0.05,
        ge=0.01, le=1.0,
        description="Max fraction of capital per single trade (0.05 = 5%)"
    )

    max_total_exposure_pct: float = Field(
        default=0.25,
        ge=0.01, le=1.0,
        description="Max fraction of capital for total open exposure (0.25 = 25%)"
    )

    # =========================================================================
    # Correlation Trading (statistical leader-laggard signals)
    # =========================================================================

    correlation_execution_enabled: bool = Field(
        default=False,
        description="Master switch for correlation-signal trade execution. "
                    "When False, signals are logged but never executed."
    )

    correlation_kelly_fraction: float = Field(
        default=0.25,
        ge=0.0, le=1.0,
        description="Extra multiplicative fraction applied to Kelly size for correlation "
                    "trades (stacks on top of kelly_fraction). Conservative because "
                    "correlation is statistical, not guaranteed arbitrage."
    )

    correlation_max_probability: float = Field(
        default=0.75,
        ge=0.5, le=0.99,
        description="Hard cap on the effective probability fed to the position sizer "
                    "for correlation trades, regardless of signal strength."
    )

    correlation_outcome_delay_minutes: float = Field(
        default=10.0,
        ge=0.1,
        description="Minutes to wait after a correlation trade before resolving the signal "
                    "outcome (laggard caught up = correct) and updating pair accuracy."
    )

    correlation_max_signal_gap_s: float = Field(
        default=5.0,
        ge=0.1,
        description="Max seconds between the previous and current tick for correlation "
                    "signal deltas to be trusted. Longer gaps mean stale price info."
    )

    # =========================================================================
    # Market Discovery & Filtering
    # =========================================================================

    min_liquidity: float = Field(
        default=1000.0,
        ge=0.0,
        description="Minimum event liquidity in USD for Map Maker scanning"
    )

    zombie_low_threshold: float = Field(
        default=0.02,
        ge=0.0, le=0.5,
        description="Lower zombie price threshold (reject extreme prices)"
    )

    zombie_high_threshold: float = Field(
        default=0.98,
        ge=0.5, le=1.0,
        description="Upper zombie price threshold (reject extreme prices)"
    )
    
    enable_semantic_matching: bool = Field(
        default=True,
        description="Enable heavy AI-based semantic matching for cross-exchange arbitrage. Disable on low-RAM systems."
    )

    min_liquidity_matcher_polymarket: float = Field(
        default=2500.0,
        ge=0.0,
        description="ExchangeMatcher: minimum Polymarket market liquidity to consider for cross-exchange matching."
    )

    min_liquidity_matcher_limitless: float = Field(
        default=500.0,
        ge=0.0,
        description="ExchangeMatcher: minimum Limitless market liquidity to consider for cross-exchange matching. "
                    "Lower than Polymarket because the Limitless universe is ~20x smaller; a $2500 floor "
                    "left only 129/976 markets visible to the matcher."
    )

    cross_exchange_expiry_tolerance_hours: float = Field(
        default=24.0,
        ge=0.0,
        description="ExchangeMatcher: maximum allowed drift between Polymarket and Limitless resolution "
                    "deadlines (hours). 24h is appropriate for daily-resolution markets; tighten to ~1h "
                    "for minute-resolution or crypto-snapshot markets. The old 1-week slack was far too loose."
    )

    crypto_expiry_tolerance_hours: float = Field(
        default=1.0,
        ge=0.0,
        description="ExchangeMatcher: tighter expiry tolerance (hours) applied when either side of a "
                    "candidate pair has domain=='crypto'. BTC/ETH price snapshot markets resolve on "
                    "point-in-time prices that can diverge meaningfully inside the 24h default window, "
                    "so crypto pairs need minute-to-hour precision. Set equal to the default if you "
                    "want the old flat behaviour."
    )

    rejection_cache_ttl_days: int = Field(
        default=14,
        ge=0,
        description="ExchangeMatcher: days to honor a cached LLM rejection before re-verifying the pair. "
                    "0 disables TTL (permanent rejection — the legacy behaviour). Default 14 days trades "
                    "LLM cost against the risk that market wording/liquidity evolves and unlocks a real "
                    "match that was previously unverifiable."
    )

    excluded_tags: list[str] = Field(
        default_factory=list,
        description="Discovery: event tag labels to skip during clustering. Case-insensitive exact "
                    "match on the event's tag labels. Use to cut whole domains (e.g. ['Sports']) on "
                    "low-resource hosts without code changes."
    )

    enable_limitless_discovery: bool = Field(
        default=True,
        description="Discovery: after cross-exchange matching, emit native_partition clusters for "
                    "standalone Limitless binary markets that have NO Polymarket counterpart. "
                    "Closes the pure-Limitless intra-market arb hole. Disable on rate-limited hosts."
    )

    min_liquidity_limitless_discovery: float = Field(
        default=500.0,
        ge=0.0,
        description="Discovery: minimum Limitless market liquidity to consider for standalone "
                    "Limitless-first discovery (Gap 5b). Matches min_liquidity_matcher_limitless "
                    "by default so the same universe is searched."
    )

    cross_event_similarity_threshold: float = Field(
        default=0.40,
        ge=0.0, le=1.0,
        description="Gap 1: cosine-similarity threshold (MiniLM) for grouping mechanical-cluster "
                    "representatives into cross_event_logical candidates. 0.40 is an initial guess; "
                    "raise to cut LLM cost if cross-event cluster volume explodes, lower to "
                    "surface more inter-event implications."
    )

    cross_event_representatives_per_cluster: int = Field(
        default=3,
        ge=1, le=10,
        description="Gap 1: top-K YES tokens by liquidity to pull from each mechanical cluster as "
                    "cross-event representatives. 3 balances coverage against LLM cost."
    )

    cross_event_confidence_floor: float = Field(
        default=0.7,
        ge=0.0, le=1.0,
        description="Gap 1: minimum LLM confidence to persist a cross_event_logical constraint. "
                    "Floor keeps LLM noise out of the solver without requiring a second validator pass."
    )

    cross_event_pool_strategy: str = Field(
        default="all_pairs_ann",
        description="Gap 1 coverage strategy for cross-event logical clustering. "
                    "'representative_top_k' (legacy): draw top-K markets from each "
                    "mechanical cluster by liquidity and semantic-cluster those. "
                    "'all_pairs_ann' (new default): embed EVERY polar market and "
                    "find neighbours across cluster boundaries — ~6-8x more "
                    "candidate pairs evaluated without the top-K coverage ceiling."
    )

    cross_event_all_pairs_top_k: int = Field(
        default=10,
        ge=1, le=50,
        description="When cross_event_pool_strategy='all_pairs_ann', number of "
                    "nearest neighbours to retain per market before connected-"
                    "component grouping. Higher values catch weaker links at the "
                    "cost of LLM budget. 10 is a reasonable default for 500 markets."
    )

    cross_event_max_cluster_size: int = Field(
        default=6,
        ge=2, le=20,
        description="Upper bound on cross-event cluster size. Larger groups "
                    "blow up the LLM prompt and rarely yield more useful "
                    "constraints than the pairwise links that spawned them."
    )

    # =========================================================================
    # LLM Configuration
    # =========================================================================

    llm_model: str = Field(
        default="gemini-2.5-flash-lite",
        description="Primary LLM model (Google AI Studio). Default for any caller that doesn't specify a per-agent model."
    )

    llm_model_discovery: str = Field(
        default="gemini-2.5-flash-lite",
        description="Model for DiscoveryAgent clustering (hot path, 15 RPM / 1000 RPD on free tier)."
    )

    llm_model_logic: str = Field(
        default="gemini-2.5-pro",
        description="Model for LogicArchitect constraint extraction (reasoning-heavy, 5 RPM / 100 RPD on free tier)."
    )

    llm_model_validator: str = Field(
        default="gemini-2.5-flash",
        description="Model for ValidatorAgent (balanced quality/quota, 10 RPM / 250 RPD on free tier)."
    )

    llm_model_matcher: str = Field(
        default="gemini-2.5-flash-lite",
        description="Model for ExchangeMatcher Polymarket<->Limitless LLM matching (high volume, shares flash-lite pool with discovery)."
    )

    llm_fallback_models: list[str] = Field(
        default_factory=lambda: [
            # ----- Tier 1: Google AI Studio free tier (same key as primary) -----
            # Useful when one Google tier's RPM is hit but others still have
            # headroom. All share the same DAILY quota, though — when the day
            # cap is gone, all three go dark.
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",

            # ----- Tier 2: OpenRouter FREE router -----
            # One magic id — OpenRouter auto-routes to whichever free model is
            # currently live AND supports the capabilities we ask for (JSON mode,
            # etc.). Cleaner than hardcoding rotating `:free` ids, which go 404
            # whenever OpenRouter rotates its catalog. Requires OPENROUTER_API_KEY.
            # Free-tier cap: 50 req/day (1000 with a one-time $10 deposit).
            # See https://openrouter.ai/docs/guides/routing/routers/free-models-router
            "openrouter/free",
        ],
        description=(
            "Fallback chain when primary returns empty/invalid JSON or is "
            "rate-limited. Two tiers: Google-free (direct) → OpenRouter-free "
            "router (auto-picks a live free model). Intentionally free-only — "
            "no paid models so the chain can never accidentally spend OpenRouter "
            "credit during tests. On 404 'no endpoints found' a model is "
            "dead-listed for the rest of the run."
        )
    )

    llm_temperature: float = Field(
        default=0.3,
        ge=0.0, le=2.0,
        description="LLM Temperature for generation"
    )

    validator_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0, le=1.0,
        description="Minimum confidence score from Validator Agent to proceed"
    )

    # =========================================================================
    # Cross-market partition detection (Map Maker)
    # =========================================================================

    partition_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0, le=1.0,
        description="Minimum LLM verification confidence to accept a cross-market partition cluster."
    )

    partition_embedding_threshold: float = Field(
        default=0.80,
        ge=0.0, le=1.0,
        description="Cosine similarity threshold for Layer 2 embedding-based partition candidate clustering."
    )

    partition_end_date_window_days: int = Field(
        default=30,
        ge=1,
        description="Layer 2 hard prefilter: markets whose end_dates differ by more than this many days never cluster together."
    )

    partition_min_cluster_size: int = Field(
        default=2,
        ge=2,
        description="Minimum candidate cluster size for cross-market partition detection."
    )

    partition_max_cluster_size: int = Field(
        default=10,
        ge=2,
        description="Maximum candidate cluster size sent to the LLM verifier (LLM may prune to fewer)."
    )

    partition_manifest_ttl_hours: int = Field(
        default=6,
        ge=1,
        description="Layer 3 dedup window: skip candidate clusters whose constraint manifest was written within this window."
    )

    # =========================================================================
    # Solver Fine-tuning
    # =========================================================================

    scip_gap: float = Field(
        default=0.01,
        ge=0.0, le=1.0,
        description="SCIP Optimality Gap limit"
    )

    min_trade_size: float = Field(
        default=0.01,
        ge=0.0,
        description="Minimum trade size in USD to execute"
    )

    # =========================================================================
    # Logging Configuration
    # =========================================================================
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity level"
    )
    
    log_json: bool = Field(
        default=False,
        description="Enable JSON-formatted logs for production"
    )
    
    # =========================================================================
    # Execution Mode
    # =========================================================================
    
    trading_mode: Literal["paper", "live"] = Field(
        default="paper",
        description="'paper' for simulation, 'live' for real trading"
    )
    
    private_rpc_url: str | None = Field(
        default=None,
        description="Optional private RPC for low-latency submission"
    )

    # =========================================================================
    # Dashboard API Server
    # =========================================================================

    api_server_host: str = Field(
        default="0.0.0.0",
        description="Bind host for the dashboard API server"
    )

    api_server_port: int = Field(
        default=8000,
        ge=1, le=65535,
        description="Bind port for the dashboard API server"
    )

    # =========================================================================
    # Supervisor (`run` mode) - periodic MapMaker inside the Navigator loop
    # =========================================================================

    map_interval_seconds: int = Field(
        default=3600,
        ge=1,
        description="Seconds between MapMaker runs in supervisor (`run`) mode"
    )

    map_run_on_start: bool = Field(
        default=True,
        description="Run MapMaker immediately on supervisor startup, before the first interval"
    )

    map_limit: int = Field(
        default=500,
        ge=0,
        description="Markets per MapMaker run in supervisor mode (0 = no limit)"
    )

    map_min_liquidity: float = Field(
        default=1000.0,
        ge=0.0,
        description="Minimum liquidity filter for MapMaker in supervisor mode"
    )


@lru_cache(maxsize=1)
def get_config() -> PolyQuantConfig:
    """
    Load and cache the configuration.
    
    Uses lru_cache to ensure config is loaded only once and reused.
    This is important because loading from .env is I/O bound.
    
    Returns:
        PolyQuantConfig: The validated configuration object
        
    Raises:
        ValidationError: If required environment variables are missing or invalid
    """
    return PolyQuantConfig()


# Create a convenient alias for importing
# Usage: from polyquant.utils.config import config
config = get_config()
