"""
Trade Store — Atomic, ACID-compliant trade persistence.

Records every fill into a local SQLite database with full transactional
integrity. Uses WAL mode for concurrent reads during writes.

PERFORMANCE:
  All writes happen via asyncio.to_thread() AFTER the trade is executed.
  This is a fire-and-forget operation that does NOT touch the hot path.
  Typical write latency: ~1ms (WAL mode, local SSD).

USAGE:
    store = TradeStore()
    await store.initialize()

    # After executor completes...
    await store.record_fills(fills)

    # Query for UI...
    recent = await store.get_recent_trades(limit=50)
"""

import asyncio
import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from polyquant.utils import get_logger

logger = get_logger(__name__)

# Default path: .polyquant/trades.db in project root
DEFAULT_DB_PATH = Path(".polyquant") / "trades.db"


class TradeStore:
    """
    SQLite-backed trade persistence with ACID transactions.

    All writes are dispatched to a thread pool so they never block
    the async event loop. WAL mode allows concurrent reads.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """
        Create the database and tables if they don't exist.
        Call once at startup (cold path).
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await asyncio.to_thread(self._open_connection)
        await asyncio.to_thread(self._create_tables)
        logger.info("TradeStore initialized", db_path=str(self._db_path))

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")     # Concurrent reads
        conn.execute("PRAGMA synchronous=NORMAL")   # Good durability, fast writes
        conn.execute("PRAGMA busy_timeout=5000")     # Wait up to 5s on lock
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        assert self._conn
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS fills (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id        TEXT NOT NULL,
                outcome_id      TEXT NOT NULL,
                side            TEXT NOT NULL,
                size            TEXT NOT NULL,
                limit_price     TEXT NOT NULL,
                filled_size     TEXT NOT NULL,
                filled_price    TEXT NOT NULL,
                notional        TEXT NOT NULL,
                fill_quality    REAL,
                fill_time       TEXT NOT NULL,
                source          TEXT DEFAULT 'constraint',
                cluster_id      TEXT,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_fills_outcome
                ON fills(outcome_id);
            CREATE INDEX IF NOT EXISTS idx_fills_time
                ON fills(fill_time DESC);
        """)
        self._conn.commit()

    async def record_fills(
        self,
        fills: list[Any],
        source: str = "constraint",
        cluster_id: str = "",
    ) -> None:
        """
        Atomically record a batch of fills in a single transaction.

        If ANY insert fails, the entire batch is rolled back (ACID).
        Runs in a background thread — does NOT block the event loop.

        Args:
            fills: List of Fill dataclass objects from executor.
            source: Origin of the opportunity ('constraint', 'correlation').
            cluster_id: The constraint cluster that generated the trade.
        """
        if not fills:
            return

        await asyncio.to_thread(
            self._insert_fills_sync, fills, source, cluster_id
        )

    def _insert_fills_sync(
        self,
        fills: list[Any],
        source: str,
        cluster_id: str,
    ) -> None:
        """Synchronous batch insert inside a transaction."""
        assert self._conn

        try:
            cursor = self._conn.cursor()
            cursor.execute("BEGIN TRANSACTION")

            for fill in fills:
                cursor.execute(
                    """
                    INSERT INTO fills (
                        order_id, outcome_id, side, size, limit_price,
                        filled_size, filled_price, notional, fill_quality,
                        fill_time, source, cluster_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fill.order_id,
                        fill.trade.outcome_id,
                        fill.trade.side.value,
                        str(fill.trade.size),
                        str(fill.trade.limit_price),
                        str(fill.filled_size),
                        str(fill.filled_price),
                        str(fill.notional),
                        fill.fill_quality,
                        fill.fill_time.isoformat(),
                        source,
                        cluster_id,
                    ),
                )

            cursor.execute("COMMIT")
            logger.info("Fills recorded atomically", count=len(fills))

        except Exception as e:
            self._conn.rollback()
            logger.error(
                "Fill recording failed — transaction rolled back",
                error=str(e),
                fills_attempted=len(fills),
            )

    async def get_recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Fetch recent trades for the UI dashboard.

        Returns:
            List of trade dicts, newest first.
        """
        return await asyncio.to_thread(self._query_recent_sync, limit)

    def _query_recent_sync(self, limit: int) -> list[dict[str, Any]]:
        assert self._conn
        cursor = self._conn.execute(
            "SELECT * FROM fills ORDER BY fill_time DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    async def get_trade_summary(self) -> dict[str, Any]:
        """
        Get aggregate trade statistics for the UI.

        Returns:
            Dict with total_trades, total_notional, avg_fill_quality, etc.
        """
        return await asyncio.to_thread(self._query_summary_sync)

    def _query_summary_sync(self) -> dict[str, Any]:
        assert self._conn
        row = self._conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                COALESCE(SUM(CAST(notional AS REAL)), 0) as total_notional,
                COALESCE(AVG(fill_quality), 0) as avg_fill_quality,
                MIN(fill_time) as first_trade,
                MAX(fill_time) as last_trade
            FROM fills
        """).fetchone()

        if row:
            return dict(row)
        return {
            "total_trades": 0,
            "total_notional": 0,
            "avg_fill_quality": 0,
            "first_trade": None,
            "last_trade": None,
        }

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await asyncio.to_thread(self._conn.close)
            self._conn = None
            logger.info("TradeStore closed")
