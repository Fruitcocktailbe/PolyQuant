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
        deepseek_api_key: API key for DeepSeek (Logic Architect)
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
        default=0.00,
        description="Polymarket Taker Fee as a percentage (e.g. 0.001 is 10bps)"
    )

    limitless_taker_fee_pct: float = Field(
        default=0.00,
        description="Limitless Taker Fee as a percentage"
    )

    polygon_gas_per_tx: float = Field(
        default=0.01,
        description="Estimated Polygon gas cost per execution in USD/USDC"
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

    orderbook_depth_cap_liquid: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Depth cap for liquid markets (>$10k total depth)"
    )

    orderbook_depth_cap_illiquid: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Depth cap for illiquid markets (<$1k total depth)"
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
    
    min_profit_threshold: float = Field(
        default=0.05,
        ge=0.0,
        description="Minimum profit in USD to consider trading (filter noise)"
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

    # =========================================================================
    # LLM Configuration
    # =========================================================================

    llm_model: str = Field(
        default="openrouter/free",
        description="LLM Model identifier (e.g., 'openrouter/free', 'google/gemini-2.0-flash-exp:free')"
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
