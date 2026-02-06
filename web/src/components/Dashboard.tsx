import { useEffect, useState, useMemo, useCallback, memo } from 'react';
import { api, SystemState } from '../services/api';
import { LogTerminal } from './LogTerminal';
import { KillSwitch } from './KillSwitch';
import { Activity, Cpu, DollarSign, Wifi, Layers } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

// Memoized stat card to prevent re-renders
const StatCard = memo<{ icon: React.ReactNode; label: string; value: string | number; color?: string }>(
    ({ icon, label, value, color = 'text-white' }) => (
        <div className="panel flex flex-col justify-center">
            <span className="text-xs text-gray-500 uppercase flex items-center gap-2">
                {icon} {label}
            </span>
            <span className={`text-2xl font-mono ${color}`}>{value}</span>
        </div>
    )
);
StatCard.displayName = 'StatCard';

// Memoized header to prevent re-renders from parent state changes
const Header = memo<{ status: string; latency: number; nlv: number }>(({ status, latency, nlv }) => {
    const statusColor = useMemo(() => {
        if (status.includes('RUNNING')) return 'text-[#00FF94]';
        if (status.includes('STOPPED')) return 'text-[#FF2A6D]';
        return 'text-gray-500';
    }, [status]);

    const latencyColor = latency < 50 ? 'text-[#00FF94]' : 'text-[#FFD600]';

    return (
        <header className="h-16 border-b border-[#2D3339] flex items-center justify-between px-6 bg-[#15191E]">
            <div className="flex items-center gap-4">
                <div className="w-3 h-3 rounded-full bg-[#00F0FF] animate-pulse shadow-[0_0_10px_#00F0FF]" />
                <h1 className="text-xl font-bold tracking-widest font-header">
                    POLYQUANT <span className="text-[#00F0FF] text-xs align-top">2.0</span>
                </h1>
                <div className={`ml-8 font-mono font-bold tracking-wide ${statusColor}`}>
                    [{status}]
                </div>
            </div>

            <div className="flex items-center gap-8 text-sm font-mono">
                <div className="flex items-center gap-2 text-gray-400">
                    <Wifi className="w-4 h-4" />
                    <span className={latencyColor}>{latency}ms</span>
                </div>
                <div className="flex items-center gap-2 bg-black/30 px-3 py-1 rounded border border-gray-800">
                    <DollarSign className="w-4 h-4 text-[#00FF94]" />
                    <span className="text-white font-bold">${nlv.toLocaleString()}</span>
                </div>
            </div>
        </header>
    );
});
Header.displayName = 'Header';

// Memoized chart to prevent re-renders when other state changes
const EquityChart = memo<{ data: { time: number; value: number }[] }>(({ data }) => (
    <div className="panel flex-1 flex flex-col min-h-0">
        <div className="flex justify-between items-center mb-4">
            <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                <Activity className="w-4 h-4 text-[#00F0FF]" /> Equity Curve
            </h3>
        </div>
        <div className="flex-1 w-full min-h-0">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                    <XAxis dataKey="time" hide />
                    <YAxis domain={['auto', 'auto']} hide />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#15191E', border: '1px solid #2D3339' }}
                        itemStyle={{ color: '#00FF94' }}
                    />
                    <Line
                        type="monotone"
                        dataKey="value"
                        stroke="#00FF94"
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    </div>
));
EquityChart.displayName = 'EquityChart';

// Initial state constant (defined outside component to avoid recreation)
const INITIAL_STATE: SystemState = {
    status: 'OFFLINE',
    net_liquidation_value: 0,
    active_solvers: 0,
    global_latency_ms: 0,
    kill_switch_active: false,
    logs: [],
};

export const Dashboard: React.FC = () => {
    const [state, setState] = useState<SystemState>(INITIAL_STATE);
    const [equityHistory, setEquityHistory] = useState<{ time: number; value: number }[]>([]);

    // Stable callback for state updates
    const handleStateUpdate = useCallback((newState: SystemState) => {
        setState(newState);

        if (newState.net_liquidation_value > 0) {
            setEquityHistory(prev => {
                const newPoint = { time: Date.now(), value: newState.net_liquidation_value };
                // Only update if value changed (prevents unnecessary chart re-renders)
                if (prev.length > 0 && prev[prev.length - 1].value === newPoint.value) {
                    return prev;
                }
                return [...prev, newPoint].slice(-50);
            });
        }
    }, []);

    useEffect(() => {
        api.connect();
        const unsubscribe = api.subscribe(handleStateUpdate);
        return () => unsubscribe();
    }, [handleStateUpdate]);

    return (
        <div className="h-screen w-screen flex flex-col bg-[#0B0E11] text-[#E6E6E6] overflow-hidden">
            <Header
                status={state.status}
                latency={state.global_latency_ms}
                nlv={state.net_liquidation_value}
            />

            <main className="flex-1 p-4 grid grid-cols-12 grid-rows-12 gap-4 overflow-hidden">
                {/* Left Col: Logs & Activity */}
                <div className="col-span-4 row-span-12 flex flex-col gap-4">
                    <div className="grid grid-cols-2 gap-4 h-24">
                        <StatCard
                            icon={<Cpu className="w-3 h-3" />}
                            label="Active Solvers"
                            value={state.active_solvers}
                            color="text-[#00F0FF]"
                        />
                        <StatCard
                            icon={<Layers className="w-3 h-3" />}
                            label="Mkts Scanned"
                            value={450}
                        />
                    </div>
                    <div className="flex-1 min-h-0">
                        <LogTerminal logs={state.logs} />
                    </div>
                </div>

                {/* Center Col: Equity & Risk */}
                <div className="col-span-5 row-span-12 flex flex-col gap-4">
                    <EquityChart data={equityHistory} />
                    <div className="h-1/3">
                        <KillSwitch active={state.kill_switch_active} />
                    </div>
                </div>

                {/* Right Col: Positions */}
                <div className="col-span-3 row-span-12 panel flex flex-col">
                    <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-4 border-b border-gray-800 pb-2">
                        Active Positions
                    </h3>
                    <div className="flex-1 overflow-y-auto">
                        <div className="text-xs text-gray-500 text-center mt-10">
                            Waiting for opportunities...
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};
