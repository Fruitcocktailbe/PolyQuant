"""
Data module exports for PolyQuant 2.0.

This module provides all data models and clients used throughout the system.
"""

from polyquant.data.market_models import (
    ArbitrageOpportunity,
    Market,
    MarketDependency,
    OrderBook,
    OrderLevel,
    OrderSide,
    Outcome,
    ProposedTrade,
)
from polyquant.data.polymarket_client import PolymarketClient
from polyquant.data.auth import PolymarketAuth, ApiCredentials

__all__ = [
    # Core Models
    "Market",
    "Outcome",
    "OrderBook",
    "OrderLevel",
    "OrderSide",
    # Dependencies
    "MarketDependency",
    # Trading
    "ProposedTrade",
    "ArbitrageOpportunity",
    # Client
    "PolymarketClient",
    # Auth
    "PolymarketAuth",
    "ApiCredentials",
]
