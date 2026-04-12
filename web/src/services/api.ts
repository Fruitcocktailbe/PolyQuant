export interface LLMPhaseProgress {
    done: number;
    total: number;
    current: string;
}

export interface SystemState {
    status: string;
    net_liquidation_value: number;
    active_solvers: number;
    global_latency_ms: number;
    kill_switch_active: boolean;
    active_positions: any[];
    clusters: any[];
    mapped_pairs: any[];
    opportunities: any[];
    trades_executed: any[];
    pipeline_stage: string;
    pipeline_events: any[];
    llm_progress: Record<string, LLMPhaseProgress>;
    logs: string[];
}

const getApiHost = () => {
    if (typeof window !== 'undefined') {
        const hostname = window.location.hostname;
        // If we're on a remote server, use that hostname
        if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
            return hostname;
        }
    }
    return 'localhost';
};

const host = getApiHost();
export const WS_URL = `ws://${host}:8000/ws`;
export const API_URL = `http://${host}:8000`;

type Listener = (state: SystemState) => void;

class ApiService {
    private ws: WebSocket | null = null;
    private listeners: Set<Listener> = new Set(); // Use Set for O(1) add/remove
    private reconnectTimeout: number | null = null;
    private state: SystemState = {
        status: "OFFLINE",
        net_liquidation_value: 0,
        active_solvers: 0,
        global_latency_ms: 0,
        kill_switch_active: false,
        active_positions: [],
        clusters: [],
        mapped_pairs: [],
        opportunities: [],
        trades_executed: [],
        pipeline_stage: "IDLE",
        pipeline_events: [],
        llm_progress: {
            LOGIC:    { done: 0, total: 0, current: "" },
            MATCHING: { done: 0, total: 0, current: "" },
        },
        logs: [],
    };

    connect() {
        // Prevent duplicate connections
        if (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING) {
            return;
        }

        this.ws = new WebSocket(WS_URL);

        this.ws.onopen = () => {
            console.log("Connected to Sidecar UI");
            if (this.reconnectTimeout) {
                clearTimeout(this.reconnectTimeout);
                this.reconnectTimeout = null;
            }
        };

        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);

                if (message.type === "state_update") {
                    // Shallow merge to preserve log array reference if unchanged
                    const newState = { ...this.state, ...message.data };

                    // Only notify if state actually changed
                    if (this.hasStateChanged(newState)) {
                        this.state = newState;
                        this.notify();
                    }
                } else if (message.type === "log") {
                    // Efficiently append log
                    const newLogs = this.state.logs.length >= 100
                        ? [...this.state.logs.slice(1), message.data]
                        : [...this.state.logs, message.data];

                    this.state = { ...this.state, logs: newLogs };
                    this.notify();
                }
            } catch (e) {
                console.error("Failed to parse WebSocket message:", e);
            }
        };

        this.ws.onclose = () => {
            console.log("Disconnected. Reconnecting in 3s...");
            this.reconnectTimeout = window.setTimeout(() => this.connect(), 3000);
        };

        this.ws.onerror = (err) => {
            console.error("WebSocket error:", err);
        };
    }

    private hasStateChanged(newState: SystemState): boolean {
        const newProg = newState.llm_progress || {};
        const oldProg = this.state.llm_progress || {};
        const progChanged = ["LOGIC", "MATCHING"].some(phase => {
            const a = newProg[phase] || { done: 0, total: 0, current: "" };
            const b = oldProg[phase] || { done: 0, total: 0, current: "" };
            return a.done !== b.done || a.total !== b.total || a.current !== b.current;
        });

        // Quick check on primitive values (most common changes)
        return (
            newState.status !== this.state.status ||
            newState.net_liquidation_value !== this.state.net_liquidation_value ||
            newState.active_solvers !== this.state.active_solvers ||
            newState.global_latency_ms !== this.state.global_latency_ms ||
            newState.kill_switch_active !== this.state.kill_switch_active ||
            newState.logs.length !== this.state.logs.length ||
            newState.clusters.length !== this.state.clusters.length ||
            newState.mapped_pairs.length !== this.state.mapped_pairs.length ||
            newState.opportunities.length !== this.state.opportunities.length ||
            newState.trades_executed.length !== this.state.trades_executed.length ||
            newState.pipeline_stage !== this.state.pipeline_stage ||
            (newState.pipeline_events?.length || 0) !== (this.state.pipeline_events?.length || 0) ||
            ((newState.pipeline_events?.length ? newState.pipeline_events[newState.pipeline_events.length - 1].timestamp : "") !==
             (this.state.pipeline_events?.length ? this.state.pipeline_events[this.state.pipeline_events.length - 1].timestamp : "")) ||
            progChanged
        );
    }

    subscribe(listener: Listener): () => void {
        this.listeners.add(listener);
        // Immediately send current state
        listener(this.state);

        return () => {
            this.listeners.delete(listener);
        };
    }

    private notify() {
        // Use requestAnimationFrame to batch rapid updates
        requestAnimationFrame(() => {
            this.listeners.forEach(listener => listener(this.state));
        });
    }

    async triggerKillSwitch(): Promise<void> {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ command: "panic_sell" }));
        } else {
            // Fallback to HTTP
            await fetch(`${API_URL}/kill`, { method: "POST" });
        }
    }

    async resetKillSwitch(): Promise<void> {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ command: "reset" }));
        } else {
            await fetch(`${API_URL}/reset`, { method: "POST" });
        }
    }

    async fetchCluster(clusterId: string): Promise<any> {
        try {
            const response = await fetch(`${API_URL}/api/clusters/${clusterId}`);
            if (!response.ok) throw new Error("Network response was not ok");
            return await response.json();
        } catch (error) {
            console.error("Failed to fetch cluster details:", error);
            return null;
        }
    }

    disconnect() {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
        }
        this.ws?.close();
    }
}

export const api = new ApiService();
