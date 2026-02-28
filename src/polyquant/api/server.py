"""
Sidecar UI Server

This module implements a lightweight FastAPI server that runs alongside the main
PolyQuant loop. It provides real-time updates via WebSockets and handles
user commands (like the Kill Switch).
"""

import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="PolyQuant Sidecar")

# CORS for React frontend (dev mode)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Global State Monitor
# -----------------------------------------------------------------------------

class SystemState(BaseModel):
    status: str = "OFFLINE"
    net_liquidation_value: float = 0.0
    active_solvers: int = 0
    global_latency_ms: float = 0.0
    kill_switch_active: bool = False
    active_positions: List[Dict[str, Any]] = []
    clusters: List[Dict[str, Any]] = []  # Discovered by MapMaker
    opportunities: List[Dict[str, Any]] = []  # Detected by Navigator
    logs: List[str] = []

class Monitor:
    """
    Singleton-like monitor that holds the state and broadcasts updates.
    """
    def __init__(self):
        self.state = SystemState()
        self.active_connections: List[WebSocket] = []
        self._log_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send initial state
        await websocket.send_json({"type": "state_update", "data": self.state.model_dump()})

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast to all clients in parallel (non-blocking per-client)."""
        if not self.active_connections:
            return
        
        async def safe_send(ws: WebSocket):
            try:
                await asyncio.wait_for(ws.send_json(message), timeout=1.0)
            except Exception:
                pass  # Dead connection, ignore
        
        # Send to all clients in parallel with timeout
        await asyncio.gather(*[safe_send(ws) for ws in self.active_connections], return_exceptions=True)

    async def update_status(self, **kwargs):
        """
        Update fields in the global state and broadcast the change.
        Usage: await monitor.update_status(status="RUNNING", global_latency_ms=23.5)
        """
        updated = False
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
                updated = True
        
        if updated:
            # Fire-and-forget: don't block pipeline waiting for broadcasts
            asyncio.create_task(self.broadcast({"type": "state_update", "data": self.state.model_dump()}))

    async def log(self, message: str, level: str = "INFO"):
        """Append a log message and broadcast it."""
        log_entry = f"[{level}] {message}"
        # Keep only last 100 logs
        self.state.logs.append(log_entry)
        if len(self.state.logs) > 100:
            self.state.logs.pop(0)
        
        # Fire-and-forget log broadcast
        asyncio.create_task(self.broadcast({"type": "log", "data": log_entry}))

    def trigger_kill_switch(self):
        self.state.kill_switch_active = True
        logger.critical("KILL SWITCH TRIGGERED FROM UI")
        # In a real sync scenario, we'd also need to notify the main loop immediately
        # The main loop checks this flag.

    def reset_kill_switch(self):
        self.state.kill_switch_active = False
        logger.info("Kill switch reset")

# Global instance
monitor = Monitor()

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await monitor.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Handle incoming commands
            if data.get("command") == "panic_sell":
                monitor.trigger_kill_switch()
                await monitor.broadcast({"type": "alert", "message": "KILL SWITCH ACTIVATED"})
    except WebSocketDisconnect:
        monitor.disconnect(websocket)

@app.get("/status")
async def get_status():
    return monitor.state

@app.post("/kill")
async def kill_switch_http():
    """Alternative HTTP endpoint for the kill switch."""
    monitor.trigger_kill_switch()
    return {"status": "triggered"}

# -----------------------------------------------------------------------------
# Database Endpoints (TradeStore)
# -----------------------------------------------------------------------------

_trade_store = None

def set_trade_store(store: Any):
    """Inject the TradeStore instance into the API server."""
    global _trade_store
    _trade_store = store

@app.get("/api/trades")
async def get_recent_trades():
    """Return recent executed trades from the ACID database."""
    if _trade_store:
        return await _trade_store.get_recent_trades(limit=50)
    return []

@app.get("/api/trade-summary")
async def get_trade_summary():
    """Return aggregated trading statistics."""
    if _trade_store:
        return await _trade_store.get_trade_summary()
    return {"total_trades": 0, "total_notional": 0.0}
