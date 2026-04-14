"""
Sidecar UI Server

This module implements a lightweight FastAPI server that runs alongside the main
PolyQuant loop. It provides real-time updates via WebSockets and handles
user commands (like the Kill Switch).
"""

import asyncio
import os
import pathlib
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
    mapped_pairs: List[Dict[str, Any]] = []  # Cross-exchange pairs from ExchangeMatcher
    opportunities: List[Dict[str, Any]] = []  # Detected by Navigator
    trades_executed: List[Dict[str, Any]] = []  # Finalized fills
    pipeline_stage: str = "IDLE"  # "IDLE", "DISCOVERY", "LOGIC", "MATCHING", "COMPLETE"
    pipeline_events: List[Dict[str, Any]] = []  # Chronological MapMaker activity feed
    llm_progress: Dict[str, Dict[str, Any]] = {
        "LOGIC":    {"done": 0, "total": 0, "current": ""},
        "MATCHING": {"done": 0, "total": 0, "current": ""},
    }
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

    async def record_trade(self, fill: Dict[str, Any]):
        """Record an executed trade and broadcast."""
        # Add to state
        self.state.trades_executed.append(fill)
        # Keep last 50
        if len(self.state.trades_executed) > 50:
            self.state.trades_executed.pop(0)
        
        # Broadcast
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

    async def emit_pipeline_event(
        self,
        stage: str,
        event_type: str,
        message: str,
        detail: str = "",
        duration: float | None = None,
    ):
        """
        Emit a structured pipeline event for the UI timeline.

        Args:
            stage: Pipeline stage (DISCOVERY, LOGIC, MATCHING, COMPLETE)
            event_type: One of: info, llm_start, llm_success, llm_fail,
                        cache_hit, validation_fail, match, api_fetch
            message: Short summary line
            detail: Optional extra detail (topic, reason, etc.)
            duration: Optional elapsed time in seconds
        """
        import datetime as _dt

        event = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "stage": stage,
            "type": event_type,
            "message": message,
            "detail": detail,
        }
        if duration is not None:
            event["duration"] = round(duration, 2)

        self.state.pipeline_events.append(event)
        # Keep last 200 events
        if len(self.state.pipeline_events) > 200:
            self.state.pipeline_events = self.state.pipeline_events[-200:]

        asyncio.create_task(
            self.broadcast({"type": "state_update", "data": self.state.model_dump()})
        )

    async def update_llm_progress(
        self,
        phase: str,
        *,
        done: int | None = None,
        total: int | None = None,
        current: str | None = None,
    ):
        """
        Update one phase of llm_progress without clobbering the other.

        Today matching and reasoning run sequentially in map_maker.build_map, but
        the copy-then-write pattern keeps this helper safe to call from concurrent
        coroutines if that ever changes.
        """
        current_progress = self.state.llm_progress
        bucket = dict(current_progress.get(phase, {"done": 0, "total": 0, "current": ""}))
        if done is not None:
            bucket["done"] = done
        if total is not None:
            bucket["total"] = total
        if current is not None:
            bucket["current"] = current
        new_progress = {**current_progress, phase: bucket}
        await self.update_status(llm_progress=new_progress)

    def trigger_kill_switch(self):
        self.state.kill_switch_active = True
        logger.critical("KILL SWITCH TRIGGERED FROM UI")
        # In a real sync scenario, we'd also need to notify the main loop immediately
        # The main loop checks this flag.

    def reset_kill_switch(self):
        self.state.kill_switch_active = False
        logger.info("Kill switch reset")

class WebLogHandler(logging.Handler):
    """
    Custom logging handler that routes logs to the Monitor.
    """
    def __init__(self, monitor_instance: Monitor):
        super().__init__()
        self.monitor = monitor_instance
        # Don't log our own WebSocket broadcasts or we'll loop infinitely
        self.addFilter(logging.Filter("polyquant.api.server"))
        # Only allow info and above for the UI to save bandwidth
        self.setLevel(logging.INFO)

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Use fire-and-forget task to avoid blocking the logging thread
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.monitor.log(msg, record.levelname))
            except RuntimeError:
                # No event loop running (e.g. during shutdown), just skip
                pass
        except Exception:
            self.handleError(record)

class CancelledErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and issubclass(record.exc_info[0], asyncio.CancelledError):
            return False
        if "CancelledError" in str(getattr(record, 'message', '')) or "CancelledError" in str(record.msg):
            return False
        return True

def setup_web_logging():
    """Attach the WebLogHandler to the polyquant root logger."""
    pq_logger = logging.getLogger("polyquant")
    handler = WebLogHandler(monitor)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    pq_logger.addHandler(handler)
    logger.info("Web logging handler attached to 'polyquant' logger")

    # Silence Starlette/Uvicorn CancelledError tracebacks on shutdown
    cancelled_filter = CancelledErrorFilter()
    logging.getLogger("uvicorn.error").addFilter(cancelled_filter)
    logging.getLogger("uvicorn.lifespan").addFilter(cancelled_filter)
    logging.getLogger("uvicorn").addFilter(cancelled_filter)
    logging.getLogger("asyncio").addFilter(cancelled_filter)


# Module-level singletons so repeat calls in the same process reuse one server.
# The supervisor (`run` mode) calls start_api_server() directly, and Navigator's
# __aenter__ / MapMaker's standalone main() also each call it. With this guard,
# the second+ call returns the existing (server, task) tuple instead of trying
# to bind a second uvicorn to the same port.
_running_server: "uvicorn.Server | None" = None  # type: ignore[name-defined]
_running_task: "asyncio.Task[None] | None" = None


async def start_api_server(
    host: str | None = None,
    port: int | None = None,
):
    """
    Boot the dashboard server. Returns (server, task) for graceful shutdown.

    Idempotent within a process: if already running, returns the existing
    handles unchanged. Host/port default to config.api_server_host /
    config.api_server_port; explicit args override for tests.
    """
    global _running_server, _running_task
    import uvicorn

    if _running_server is not None and _running_task is not None and not _running_task.done():
        return _running_server, _running_task

    # Resolve bind target. Import locally so tests can monkeypatch config.
    from polyquant.utils.config import config as _cfg
    resolved_host = host if host is not None else _cfg.api_server_host
    resolved_port = port if port is not None else _cfg.api_server_port

    config_uv = uvicorn.Config(app, host=resolved_host, port=resolved_port, log_level="warning")
    server = uvicorn.Server(config_uv)
    task = asyncio.create_task(server.serve(), name="uvicorn_server")

    # Observe startup instead of blindly sleeping: poll server.started and
    # re-raise any exception from the server task (bind errors, etc.) in the
    # caller's frame — uvicorn otherwise swallows them into a SystemExit(1)
    # that kills the whole interpreter with no actionable message.
    deadline = asyncio.get_running_loop().time() + 5.0
    while not server.started:
        if task.done():
            # server.serve() exited before signalling started — surface the cause.
            exc = task.exception()
            if isinstance(exc, OSError) and getattr(exc, "errno", None) in (48, 98, 10048):
                raise RuntimeError(
                    f"API server cannot bind to {resolved_host}:{resolved_port} — address already in use.\n"
                    f"Another PolyQuant process is probably still holding the port. Stop it with:\n"
                    f"    lsof -iTCP:{resolved_port} -sTCP:LISTEN    # find the PID\n"
                    f"    kill <pid>\n"
                    f"Or pick a different port:\n"
                    f"    API_SERVER_PORT=8001 python -m polyquant.main run"
                ) from exc
            if exc is not None:
                raise RuntimeError(
                    f"API server failed to start on {resolved_host}:{resolved_port}: {exc}"
                ) from exc
            raise RuntimeError(
                f"API server task exited during startup on {resolved_host}:{resolved_port} with no exception"
            )
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError(
                f"API server did not become ready within 5s on {resolved_host}:{resolved_port}"
            )
        await asyncio.sleep(0.05)

    setup_web_logging()
    logger.info(f"🌐 API server started on http://{resolved_host}:{resolved_port}")
    _running_server = server
    _running_task = task
    return server, task

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
            elif data.get("command") == "reset":
                monitor.reset_kill_switch()
                await monitor.broadcast({"type": "alert", "message": "SYSTEM RESUMED"})
    except WebSocketDisconnect:
        monitor.disconnect(websocket)

@app.get("/status")
async def get_status():
    return monitor.state

@app.get("/api/status")
async def get_api_status():
    """Alias so /api/status also works."""
    return monitor.state

@app.post("/kill")
async def kill_switch_http():
    """Alternative HTTP endpoint for the kill switch."""
    monitor.trigger_kill_switch()
    return {"status": "triggered"}

@app.post("/reset")
async def reset_kill_switch_http():
    """Alternative HTTP endpoint to reset the kill switch."""
    monitor.reset_kill_switch()
    return {"status": "reset"}

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

# -----------------------------------------------------------------------------
# Knowledge Map Endpoints (ConstraintStore)
# -----------------------------------------------------------------------------

_constraint_store = None

def set_constraint_store(store: Any):
    """Inject the ConstraintStore instance into the API server."""
    global _constraint_store
    _constraint_store = store

@app.get("/api/clusters/{cluster_id}")
async def get_cluster_details(cluster_id: str):
    """Return the full ConstraintManifest for a specific cluster."""
    if _constraint_store:
        manifest = await _constraint_store.load_manifest(cluster_id)
        if manifest:
            return manifest.model_dump()
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Cluster not found")

# -----------------------------------------------------------------------------
# Static File Serving (Built-in UI)
# -----------------------------------------------------------------------------

# Discover web/dist directory relative to this file
# src/polyquant/api/server.py -> 4 levels up to root
BASE_DIR = pathlib.Path(__file__).parent.parent.parent.parent
STATIC_DIR = BASE_DIR / "web" / "dist"

if STATIC_DIR.exists():
    logger.info(f"Serving built-in UI from {STATIC_DIR}")
    # Mount assets if they exist
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    
    @app.get("/")
    async def serve_index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/{rest_of_path:path}")
    async def serve_spa_fallback(rest_of_path: str):
        """Fallback for React SPA routing."""
        # Don't intercept API or WS calls
        if rest_of_path.startswith(("api", "ws", "status", "kill")):
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "not found"}, status_code=404)
             
        # Check if file exists in dist (e.g. favicon.ico)
        file_path = STATIC_DIR / rest_of_path
        if file_path.is_file():
            return FileResponse(file_path)
            
        # Otherwise serve index.html for SPA
        return FileResponse(STATIC_DIR / "index.html")
else:
    logger.warning(f"UI build directory not found at {STATIC_DIR}. Run 'npm run build' in the web folder to enable built-in UI.")
