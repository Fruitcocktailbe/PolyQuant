"""
IPC Client to interface with the ultra-low latency Rust execution sidecar.
"""

import json
import time
import uuid
from typing import Optional

import zmq

from polyquant.data import ProposedTrade
from polyquant.utils import get_logger

logger = get_logger(__name__)


class RustClient:
    """Client for routing execution payloads to the Rust sidecar via ZeroMQ."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555", timeout_ms: int = 5000):
        self._endpoint = endpoint
        self._timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._socket: Optional[zmq.Socket] = None
        self._request_count = 0
        self._connect()

    def _connect(self):
        """Create and configure a fresh REQ socket."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass

        self._socket = self._context.socket(zmq.REQ)

        # Configure timeout to prevent hanging the event loop if sidecar is dead
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        # Disable Nagle's algorithm for low-latency small messages
        self._socket.setsockopt(zmq.TCP_NODELAY, 1)
        # Prevent OOM from malformed messages (1MB limit)
        self._socket.setsockopt(zmq.MAXMSGSIZE, 1_048_576)

        try:
            self._socket.connect(self._endpoint)
            logger.info("RustClient connected to ZeroMQ endpoint", endpoint=self._endpoint)
        except Exception as e:
            logger.error("Failed to connect to Rust OMS", error=str(e))

    def _reconnect(self):
        """Tear down dead socket and create a fresh one.

        ZMQ REQ/REP requires strict send-recv alternation. After a timeout
        on recv, the socket enters an error state (EFSM) and cannot send
        again. The only recovery is to close and recreate.
        """
        logger.warning("Reconnecting ZMQ socket after error")
        self._connect()

    def dispatch_trades(
        self,
        trades: list[ProposedTrade],
        signal_timestamp_us: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Send a batch of trades to the Rust sidecar.

        Args:
            trades: List of trades to execute.
            signal_timestamp_us: Timestamp (microseconds since epoch) of when
                the price signal was observed. If None, stamps at dispatch time
                (less accurate — prefer passing detection-time timestamp).

        Returns the parsed JSON dictionary from the rust sidecar response,
        or None on timeout/error.
        """
        signal_ts = signal_timestamp_us or int(time.time() * 1_000_000)
        request_id = str(uuid.uuid4())
        self._request_count += 1

        payload = []
        for trade in trades:
            trade_dict = trade.model_dump(mode="json")
            # Force string representation for precision-critical decimals
            trade_dict["size"] = str(trade.size)
            trade_dict["limit_price"] = str(trade.limit_price)
            # Ensure side is uppercase string
            trade_dict["side"] = str(trade.side.value).upper()
            # Staleness check: Rust will reject if too old
            trade_dict["signal_timestamp_us"] = signal_ts
            # Idempotency key: Rust dedup cache rejects duplicate batch IDs
            trade_dict["request_id"] = request_id
            payload.append(trade_dict)

        payload_bytes = json.dumps(payload).encode("utf-8")

        try:
            logger.debug(
                f"Dispatching {len(trades)} trades to Rust Sidecar over ZMQ",
                request_id=request_id,
            )
            self._socket.send(payload_bytes)

            # Wait for acknowledgment
            response_bytes = self._socket.recv()
            response_data = json.loads(response_bytes.decode("utf-8"))

            logger.debug("Received response from Rust Sidecar", response=response_data)
            return response_data

        except zmq.error.Again:
            logger.error(
                "Timeout waiting for Rust OMS response. Sidecar may be offline or overloaded.",
                request_id=request_id,
            )
            # Socket is now in bad state (EFSM) — must reconnect
            self._reconnect()
            return None
        except Exception as e:
            logger.error("IPC communication error with Rust OMS", error=str(e))
            self._reconnect()
            return None

    def send_halt(self, reason: str = "kill_switch_triggered") -> Optional[dict]:
        """Send halt command to Rust sidecar. Called by KillSwitch on trigger."""
        command = {"command": "halt", "reason": reason}
        payload_bytes = json.dumps(command).encode("utf-8")

        try:
            logger.critical("Sending HALT to Rust OMS", reason=reason)
            self._socket.send(payload_bytes)
            response_bytes = self._socket.recv()
            response_data = json.loads(response_bytes.decode("utf-8"))
            logger.info("Rust OMS halted", response=response_data)
            return response_data
        except zmq.error.Again:
            logger.error("Timeout sending HALT to Rust OMS")
            self._reconnect()
            return None
        except Exception as e:
            logger.error("Failed to send HALT to Rust OMS", error=str(e))
            self._reconnect()
            return None

    def send_reset(self) -> Optional[dict]:
        """Clear halt state in Rust sidecar."""
        command = {"command": "reset"}
        payload_bytes = json.dumps(command).encode("utf-8")

        try:
            logger.info("Sending RESET to Rust OMS")
            self._socket.send(payload_bytes)
            response_bytes = self._socket.recv()
            return json.loads(response_bytes.decode("utf-8"))
        except Exception as e:
            logger.error("Failed to send RESET to Rust OMS", error=str(e))
            self._reconnect()
            return None

    def send_status(self) -> Optional[dict]:
        """Query Rust sidecar status (running/halted)."""
        command = {"command": "status"}
        payload_bytes = json.dumps(command).encode("utf-8")

        try:
            self._socket.send(payload_bytes)
            response_bytes = self._socket.recv()
            return json.loads(response_bytes.decode("utf-8"))
        except Exception:
            self._reconnect()
            return None

    def close(self):
        """Cleanup sockets"""
        if self._socket:
            self._socket.close()
        self._context.term()
