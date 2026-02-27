import { useEffect, useState, useMemo, useCallback, memo } from 'react';
import { api, SystemState } from '../services/api';
import { LogTerminal } from './LogTerminal';
import { KillSwitch } from './KillSwitch';
import { MarketList } from './MarketList';
import { Activity, Cpu, DollarSign, Wifi, Layers } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

// Memoized stat card to prevent re-renders
const StatCard = memo<{ icon: React.ReactNode; label: string; value: string | number; color?: string; subValue?: string }>(
    ({ icon, label, value, color = 'text-white', subValue }) => (
        <div className="bg-charcoal/50 border border-white/5 backdrop-blur-sm p-4 flex flex-col justify-between hover:border-neon-cyan/30 transition-colors group">
            <div className="flex justify-between items-start mb-2">
                <span className="text-[10px] uppercase tracking-[0.2em] text-gray-500 font-bold flex items-center gap-2 group-hover:text-neon-cyan transition-colors">
                    {icon} {label}
                </span>
                {subValue && <span className="text-[10px] text-neon-green font-mono">{subValue}</span>}
            </div>
            <span className={`text-2xl font-mono tracking-tight font-medium ${color} drop-shadow-[0_0_10px_rgba(0,0,0,0.5)]`}>
                {value}
            </span>
        </div>
    )
);
StatCard.displayName = 'StatCard';

// Memoized header
const Header = memo<{ status: string; latency: number; nlv: number }>(({ status, latency, nlv }) => {
    const statusColor = useMemo(() => {
        if (status.includes('RUNNING')) return 'text-neon-green drop-shadow-[0_0_8px_rgba(0,255,148,0.5)]';
        if (status.includes('STOPPED')) return 'text-neon-red drop-shadow-[0_0_8px_rgba(255,0,85,0.5)]';
        return 'text-gray-500';
    }, [status]);

    const latencyColor = latency < 50 ? 'text-neon-green' : 'text-neon-purple';

    return (
        <header className="h-14 border-b border-white/10 bg-black/40 backdrop-blur-md flex items-center justify-between px-6 z-10">
            <div className="flex items-center gap-6">
                <div className="flex items-center gap-3">
                    <div className="w-2 h-2 rounded-none bg-neon-cyan animate-pulse shadow-neon-cyan" />
                    <h1 className="text-lg font-bold tracking-[0.2em] text-white">
                        POLYQUANT <span className="text-neon-cyan text-[10px] align-top opacity-80">v2.0</span>
                    </h1>
                </div>

                <div className="h-4 w-[1px] bg-white/10" />

                <div className={`font-mono text-xs font-bold tracking-wider ${statusColor}`}>
                    [{status}]
                </div>
            </div>

            <div className="flex items-center gap-6 text-xs font-mono">
                <div className="flex items-center gap-2 text-gray-400 group">
                    <Wifi className="w-3 h-3 group-hover:text-white transition-colors" />
                    <span className={latencyColor}>{latency}ms</span>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 bg-neon-green/10 border border-neon-green/20">
                    <DollarSign className="w-3 h-3 text-neon-green" />
                    <span className="text-white font-bold tracking-wide">${nlv.toLocaleString()}</span>
                </div>
            </div>
        </header>
    );
});
Header.displayName = 'Header';

// Memoized chart
const EquityChart = memo<{ data: { time: number; value: number }[] }>(({ data }) => (
    <div className="bg-charcoal/50 border border-white/5 backdrop-blur-sm flex-1 flex flex-col min-h-0 relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-4 opacity-50 pointer-events-none">
            <Activity className="w-32 h-32 text-white/5" />
        </div>

        <div className="p-4 border-b border-white/5 flex justify-between items-center bg-black/20">
            <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-[0.2em] flex items-center gap-2">
                <Activity className="w-3 h-3 text-neon-purple" /> Equity Curve
            </h3>
            <div className="flex gap-2">
                <span className="text-[10px] text-gray-600 font-mono">LIVE FEED</span>
                <div className="w-1.5 h-1.5 bg-neon-green rounded-full animate-ping" />
            </div>
        </div>

        <div className="flex-1 w-full min-h-0 p-2">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                    <XAxis dataKey="time" hide />
                    <YAxis domain={['auto', 'auto']} hide />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: 'rgba(5, 5, 5, 0.95)',
                            border: '1px solid rgba(255, 255, 255, 0.1)',
                            boxShadow: '0 0 20px rgba(0,0,0,0.5)',
                            fontSize: '12px',
                            fontFamily: 'monospace'
                        }}
                        itemStyle={{ color: '#00ff94' }}
                        labelStyle={{ display: 'none' }}
                        formatter={(value: number) => [`$${value.toFixed(2)}`, 'Equity']}
                    />
                    <Line
                        type="stepAfter"
                        dataKey="value"
                        stroke="#00ff94"
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                        strokeDasharray="0"
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
    markets: [],
    logs: [],
};

export const Dashboard: React.FC = () => {
    const [state, setState] = useState<SystemState>(INITIAL_STATE);
    const [equityHistory, setEquityHistory] = useState<{ time: number; value: number }[]>([]);
    const [showRawData, setShowRawData] = useState(false);

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
        <div className="h-screen w-screen flex flex-col bg-obsidian text-gray-300 font-sans selection:bg-neon-cyan/30 selection:text-neon-cyan overflow-hidden bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-charcoal via-obsidian to-obsidian">
            <Header
                status={state.status}
                latency={state.global_latency_ms}
                nlv={state.net_liquidation_value}
            />

            <main className="flex-1 p-3 grid grid-cols-12 grid-rows-12 gap-3 overflow-hidden">
                {/* Left Col: Logs & Activity */}
                <div className="col-span-3 row-span-12 flex flex-col gap-3">
                    <div className="grid grid-cols-1 gap-3 h-auto">
                        <StatCard
                            icon={<Cpu className="w-3 h-3" />}
                            label="Active Solvers"
                            value={state.active_solvers}
                            color="text-neon-cyan"
                            subValue="+2.5%"
                        />
                        <StatCard
                            icon={<Layers className="w-3 h-3" />}
                            label="Mkts Scanned"
                            value={450}
                            color="text-white"
                        />
                    </div>
                    <div className="flex-1 min-h-0 bg-charcoal/50 border border-white/5 backdrop-blur-sm flex flex-col">
                        <LogTerminal logs={state.logs} />
                    </div>
                </div>

                {/* Center Col: Equity & Risk */}
                <div className="col-span-6 row-span-12 flex flex-col gap-3">
                    <EquityChart data={equityHistory} />
                    <div className="h-48">
                        <KillSwitch active={state.kill_switch_active} />
                    </div>
                </div>

                {/* Right Col: Positions */}
                <div className="col-span-3 row-span-12 bg-charcoal/50 border border-white/5 backdrop-blur-sm flex flex-col relative overflow-hidden">
                    <div className="p-4 border-b border-white/5 bg-black/20 flex justify-between items-center">
                        <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-[0.2em]">
                            Active Positions
                        </h3>
                        <div className="px-2 py-0.5 bg-neon-cyan/10 text-neon-cyan text-[10px] font-mono border border-neon-cyan/20">
                            LIVE
                        </div>
                    </div>

                    <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col min-h-0">
                        <MarketList markets={state.markets} />
                    </div>

                    <div className="border-t border-white/5 p-2 bg-black/20">
                        <button
                            onClick={() => setShowRawData(!showRawData)}
                            className="text-[10px] text-gray-500 hover:text-neon-cyan transition-colors flex items-center gap-1 uppercase tracking-tighter"
                        >
                            <Layers className="w-3 h-3" />
                            {showRawData ? 'Hide Raw State' : 'Show Raw State'}
                        </button>

                        {showRawData && (
                            <pre className="mt-2 text-[8px] font-mono text-neon-green/70 bg-black/50 p-2 rounded overflow-auto max-h-40 custom-scrollbar">
                                {JSON.stringify(state, null, 2)}
                            </pre>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
};
