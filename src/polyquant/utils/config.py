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
    
    print(config.openai_api_key)
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
        openai_api_key: API key for OpenAI (GPT-4o, o1-preview)
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
    
    openai_api_key: SecretStr = Field(
        description="OpenAI API key for GPT-4o and o1-preview models"
    )
    
    deepseek_api_key: SecretStr = Field(
        description="DeepSeek API key for the Logic Architect (R1 model)"
    )
    
    alchemy_api_key: SecretStr = Field(
        description="Alchemy API key for Polygon blockchain data"
    )
    
    # =========================================================================
    # Polymarket Connection Settings
    # =========================================================================
    
    polymarket_clob_url: str = Field(
        default="https://clob.polymarket.com",
        description="Polymarket CLOB REST API endpoint"
    )
    
    polymarket_ws_url: str = Field(
        default="wss://ws-subscriptions-clob.polymarket.com/ws",
        description="Polymarket WebSocket endpoint for real-time data"
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
    
    latency_target_ms: int = Field(
        default=30,
        ge=1,
        description="Target decision-to-mempool latency in milliseconds"
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
