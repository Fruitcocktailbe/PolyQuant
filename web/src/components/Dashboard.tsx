import React, { useState, useEffect, useCallback, useMemo, memo, useRef } from 'react';
import {
    LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer
} from 'recharts';
import { Cpu, Layers, Activity, Wifi, DollarSign } from 'lucide-react';
import { api, SystemState } from '../services/api';
import { PipelineMonitor } from './PipelineMonitor';
import { TradeLog } from './TradeLog';
import { LogTerminal } from './LogTerminal';
import { KillSwitch } from './KillSwitch';

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

// Memoized Cluster card
const ClusterCard = memo<{ cluster: any; onClick?: (id: string) => void }>(({ cluster, onClick }) => (
    <div
        className={`p-3 border-b border-white/5 transition-colors ${onClick ? 'cursor-pointer hover:bg-white/10 hover:border-neon-cyan/50' : 'hover:bg-white/5'}`}
        onClick={() => onClick && onClick(cluster.id)}
    >
        <div className="flex justify-between items-center mb-1">
            <span className="text-[10px] font-bold text-neon-cyan uppercase tracking-wider truncate mr-2">{cluster.topic}</span>
            <span className="text-[10px] text-gray-600 font-mono shrink-0">#{cluster.id.slice(0, 4)}</span>
        </div>
        <div className="flex justify-between items-center">
            <span className="text-[10px] text-gray-400">{cluster.count} Markets</span>
            <span className="text-[10px] px-1.5 py-0.5 bg-neon-green/10 text-neon-green rounded-none border border-neon-green/20 uppercase tracking-tighter">
                {cluster.status || 'ACTIVE'}
            </span>
        </div>
    </div>
));

// Cluster Details Modal Component
const ClusterDetailsModal = memo<{ clusterId: string | null; onClose: () => void }>(({ clusterId, onClose }) => {
    const [details, setDetails] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!clusterId) {
            setDetails(null);
            return;
        }

        let isMounted = true;
        setLoading(true);
        api.fetchCluster(clusterId).then(data => {
            if (isMounted) {
                setDetails(data);
                setLoading(false);
            }
        });

        return () => { isMounted = false; };
    }, [clusterId]);

    if (!clusterId) return null;

    return (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-6 sm:p-10">
            <div className="bg-obsidian border border-neon-cyan/30 w-full max-w-3xl max-h-[90vh] flex flex-col overflow-hidden shadow-[0_0_30px_rgba(0,255,240,0.15)] rounded-sm">
                <div className="p-4 border-b border-white/10 bg-black/40 flex justify-between items-center shrink-0">
                    <div className="flex items-center gap-3">
                        <Layers className="w-4 h-4 text-neon-cyan" />
                        <div>
                            <h2 className="text-xs font-bold text-white uppercase tracking-widest leading-none mb-1">
                                {details?.topic || `Cluster ${clusterId.slice(0, 8)}`}
                            </h2>
                            <span className="text-[9px] font-mono text-gray-500 uppercase">ID: {clusterId}</span>
                        </div>
                    </div>
                    <button onClick={onClose} className="text-gray-500 hover:text-white transition-colors">✕</button>
                </div>

                <div className="flex-1 overflow-auto p-0 custom-scrollbar relative bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-charcoal/30 to-transparent">
                    {loading ? (
                        <div className="flex items-center justify-center h-full text-neon-cyan/50 font-mono text-xs animate-pulse">
                            Loading internal structure...
                        </div>
                    ) : details ? (
                        <div className="p-5 space-y-6">
                            {/* Stats */}
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                <div className="border border-white/5 bg-black/20 p-3">
                                    <div className="text-[9px] text-gray-500 uppercase tracking-widest mb-1">Constraints</div>
                                    <div className="text-lg font-mono text-neon-green">{details.constraints?.length || 0}</div>
                                </div>
                                <div className="border border-white/5 bg-black/20 p-3">
                                    <div className="text-[9px] text-gray-500 uppercase tracking-widest mb-1">Linked Markets</div>
                                    <div className="text-lg font-mono text-neon-purple">{details.market_ids?.length || 0}</div>
                                </div>
                                <div className="border border-white/5 bg-black/20 p-3">
                                    <div className="text-[9px] text-gray-500 uppercase tracking-widest mb-1">Constraint Ext.</div>
                                    <div className="text-lg font-mono text-white">{details.extractor?.slice(0, 10) || 'None'}</div>
                                </div>
                                <div className="border border-white/5 bg-black/20 p-3">
                                    <div className="text-[9px] text-gray-500 uppercase tracking-widest mb-1">Validation Status</div>
                                    <div className="text-lg font-mono text-neon-cyan text-sm mt-1">{details.is_valid ? 'VALID' : 'INVALID'}</div>
                                </div>
                            </div>

                            {/* Constraints Detail */}
                            <div>
                                <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-[0.2em] mb-3 flex items-center gap-2 border-b border-white/10 pb-2">
                                    <Activity className="w-3 h-3 text-neon-green" /> Constraint Equations
                                </h3>
                                <div className="space-y-2">
                                    {details.constraints?.map((constraint: any, idx: number) => (
                                        <div key={idx} className="bg-black/40 border border-white/5 p-3 hover:border-white/10 transition-colors">
                                            <div className="flex justify-between items-start mb-2">
                                                <span className="text-[10px] text-gray-500 uppercase">{constraint.type || 'Prio 0'}</span>
                                                <div className="text-[10px] font-mono whitespace-nowrap overflow-x-auto custom-scrollbar pb-1 text-right max-w-[70%]">
                                                    {Object.entries(constraint.coefficients || {}).map(([token, coeff]: [string, any], i, arr) => (
                                                        <span key={token} className="inline-block">
                                                            <span className={coeff > 0 ? "text-neon-cyan" : "text-neon-red"}>{coeff > 0 ? '+' : ''}{Number(coeff).toFixed(2)}</span>
                                                            <span className="text-gray-400">×</span>
                                                            <span className="text-white" title={token}>{token.slice(0, 5)}...</span>
                                                            {i < arr.length - 1 ? "  " : ""}
                                                        </span>
                                                    ))}
                                                    <span className="text-gray-500 mx-2">{constraint.operator || '<='}</span>
                                                    <span className="text-neon-green">{constraint.rhs?.toFixed(2) || '0.00'}</span>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                    {(!details.constraints || details.constraints.length === 0) && (
                                        <div className="text-[10px] text-gray-600 font-mono text-center py-4 italic">No constraints extracted.</div>
                                    )}
                                </div>
                            </div>

                            {/* Markets Map */}
                            <div>
                                <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-[0.2em] mb-3 flex items-center gap-2 border-b border-white/10 pb-2">
                                    <Cpu className="w-3 h-3 text-neon-purple" /> Markets Discovered
                                </h3>
                                <div className="space-y-2">
                                    {Object.entries(details.market_exchanges || {}).map(([marketId, exchange]: [string, any]) => {
                                        const title = details.market_titles?.[marketId] || "Unknown Market";
                                        const isLimitless = String(exchange).toLowerCase().includes('limitless');
                                        return (
                                            <div key={marketId} className="bg-black/40 border border-white/5 p-3 hover:border-white/10 transition-colors flex justify-between items-center">
                                                <div className="flex flex-col gap-1 w-[70%]">
                                                    <span className="text-xs text-white truncate" title={title}>{title}</span>
                                                    <span className="text-[9px] text-gray-500 font-mono select-all">ID: {marketId}</span>
                                                </div>
                                                <div className="flex items-center">
                                                    <span className={`text-[8px] px-2 py-1 border font-bold uppercase tracking-tighter ${isLimitless ? 'bg-neon-purple/20 text-neon-purple border-neon-purple/40' : 'bg-neon-cyan/20 text-neon-cyan border-neon-cyan/40'}`}>
                                                        {exchange as string}
                                                    </span>
                                                </div>
                                            </div>
                                        );
                                    })}
                                    {Object.keys(details.market_exchanges || {}).length === 0 && (
                                        <div className="text-[10px] text-gray-600 font-mono text-center py-4 italic">No markets linked yet.</div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center h-full text-red-400 font-mono text-xs gap-2 p-10 text-center">
                            Failed to load cluster details.
                            <span className="text-gray-500 text-[10px]">The API may be unavailable or the cluster ID is invalid.</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
});
ClusterDetailsModal.displayName = 'ClusterDetailsModal';

// Memoized Opportunity ticker
const OpportunityCard = memo<{ opp: any }>(({ opp }) => {
    const isCrossExchange = opp.trades?.some((t: any) => t.exchange === 'limitless');
    const badgeColor = isCrossExchange ? 'bg-neon-purple/20 text-neon-purple border-neon-purple/40' : 'bg-neon-cyan/20 text-neon-cyan border-neon-cyan/40';
    const badgeLabel = isCrossExchange ? 'CROSS-EXCHANGE' : 'INTRA-MARKET';

    return (
        <div className={`p-3 border-l-2 mb-2 animate-in slide-in-from-right duration-300 ${isCrossExchange ? 'bg-neon-purple/5 border-neon-purple' : 'bg-neon-cyan/5 border-neon-cyan'}`}>
            <div className="flex justify-between items-start mb-2">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className={`text-[7px] px-1 py-0.5 border font-bold uppercase tracking-tighter ${badgeColor}`}>
                            {badgeLabel}
                        </span>
                    </div>
                    <div className="text-[10px] font-bold text-white uppercase tracking-widest">Opportunity Detected</div>
                    <div className="text-[8px] text-gray-500 font-mono">{opp.cluster_id}</div>
                </div>
                <div className="text-neon-green font-mono font-bold text-sm">+${opp.profit?.toFixed(2) || '0.00'}</div>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-2">
                {opp.trades && opp.trades.slice(0, 2).map((t: any, idx: number) => (
                    <div key={idx} className="text-[8px] font-mono text-gray-400 border border-white/5 p-1 bg-black/20 truncate">
                        {t.side} {t.size} @ ${t.price} <span className="opacity-50 text-[6px]">({t.exchange?.slice(0, 4)})</span>
                    </div>
                ))}
            </div>
        </div>
    );
});

// Initial state constant (defined outside component to avoid recreation)
const INITIAL_STATE: SystemState = {
    status: 'OFFLINE',
    net_liquidation_value: 0,
    active_solvers: 0,
    global_latency_ms: 0,
    kill_switch_active: false,
    active_positions: [],
    clusters: [],
    mapped_pairs: [],
    opportunities: [],
    trades_executed: [],
    pipeline_stage: 'IDLE',
    pipeline_events: [],
    llm_progress: {
        LOGIC:    { done: 0, total: 0, current: '' },
        MATCHING: { done: 0, total: 0, current: '' },
    },
    logs: [],
};

export const Dashboard: React.FC = () => {
    const [state, setState] = useState<SystemState>(INITIAL_STATE);
    const [equityHistory, setEquityHistory] = useState<{ time: number; value: number }[]>([]);
    const [showRawData, setShowRawData] = useState(false);
    const [activeTab, setActiveTab] = useState<'chart' | 'terminal' | 'pipeline'>('chart');
    const [selectedClusterId, setSelectedClusterId] = useState<string | null>(null);
    const prevStageRef = useRef<string>('IDLE');

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

    // Auto-switch to pipeline tab when MapMaker starts a new run
    useEffect(() => {
        const prev = prevStageRef.current;
        const curr = state.pipeline_stage;
        if (prev === 'IDLE' && curr === 'DISCOVERY') {
            setActiveTab('pipeline');
        }
        prevStageRef.current = curr;
    }, [state.pipeline_stage]);

    return (
        <div className="h-screen w-screen flex flex-col bg-obsidian text-gray-300 font-sans selection:bg-neon-cyan/30 selection:text-neon-cyan overflow-hidden bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-charcoal via-obsidian to-obsidian">
            <Header
                status={state.status}
                latency={state.global_latency_ms}
                nlv={state.net_liquidation_value}
            />

            <main className="flex-1 p-3 grid grid-cols-12 grid-rows-12 gap-3 overflow-hidden">
                {/* Left Col: Logs & Activity */}
                <div className="col-span-2 row-span-12 flex flex-col gap-3">
                    <div className="grid grid-cols-1 gap-3 h-auto">
                        <StatCard
                            icon={<Cpu className="w-3 h-3" />}
                            label="Solvers"
                            value={state.active_solvers}
                            color="text-neon-cyan"
                        />
                        <StatCard
                            icon={<Layers className="w-3 h-3" />}
                            label="Clusters"
                            value={state.clusters.length}
                            color="text-white"
                        />
                    </div>
                    <div className="flex-1 min-h-0 bg-charcoal/50 border border-white/5 backdrop-blur-sm flex flex-col p-4">
                        <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-4">Quick Stats</div>
                        <div className="space-y-4">
                            <div className="flex justify-between items-center border-b border-white/5 pb-2">
                                <span className="text-[10px] text-gray-500">Pipeline</span>
                                <span className="text-[10px] text-neon-cyan font-mono">{state.pipeline_stage}</span>
                            </div>
                            <div className="flex justify-between items-center border-b border-white/5 pb-2">
                                <span className="text-[10px] text-gray-500">Opportunities</span>
                                <span className="text-[10px] text-white font-mono">{state.opportunities.length}</span>
                            </div>
                            <div className="flex justify-between items-center border-b border-white/5 pb-2">
                                <span className="text-[10px] text-gray-500">Trades</span>
                                <span className="text-[10px] text-neon-green font-mono">{state.trades_executed.length}</span>
                            </div>
                        </div>
                        <div className="mt-auto pt-4 border-t border-white/5">
                            <div className="text-[8px] text-gray-600 font-mono">SYSTEM_ID: PQ-V2-MAIN</div>
                        </div>
                    </div>
                </div>

                {/* Center Col: Equity & Risk */}
                <div className="col-span-6 row-span-12 flex flex-col gap-3">
                    <div className="flex-[0.65] flex flex-col min-h-0">
                        <div className="flex-1 flex flex-col min-h-0">
                            <div className="flex h-10 border-b border-white/5 bg-black/20">
                                <button
                                    onClick={() => setActiveTab('chart')}
                                    className={`px-6 text-[10px] font-bold tracking-widest transition-all border-b-2 ${activeTab === 'chart' ? 'border-neon-cyan text-white bg-white/5' : 'border-transparent text-gray-500 hover:text-gray-300'}`}
                                >
                                    EQUITY_CURVE
                                </button>
                                <button
                                    onClick={() => setActiveTab('terminal')}
                                    className={`px-6 text-[10px] font-bold tracking-widest transition-all border-b-2 ${activeTab === 'terminal' ? 'border-neon-purple text-white bg-white/5' : 'border-transparent text-gray-500 hover:text-gray-300'}`}
                                >
                                    SYSTEM_TERMINAL
                                </button>
                                <button
                                    onClick={() => setActiveTab('pipeline')}
                                    className={`px-6 text-[10px] font-bold tracking-widest transition-all border-b-2 flex items-center gap-2 ${activeTab === 'pipeline' ? 'border-neon-green text-white bg-white/5' : 'border-transparent text-gray-500 hover:text-gray-300'}`}
                                >
                                    PIPELINE_FEED
                                    {['DISCOVERY', 'LOGIC', 'MATCHING'].includes(state.pipeline_stage) && (
                                        <span className="relative flex h-2 w-2">
                                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-neon-green opacity-75" />
                                            <span className="relative inline-flex rounded-full h-2 w-2 bg-neon-green" />
                                        </span>
                                    )}
                                    {(state.pipeline_events?.length || 0) > 0 && activeTab !== 'pipeline' && (
                                        <span className="text-[8px] text-gray-500 font-mono">({state.pipeline_events.length})</span>
                                    )}
                                </button>
                            </div>
                            <div className="flex-1 min-h-0">
                                {activeTab === 'chart' && (
                                    <EquityChart data={equityHistory} />
                                )}
                                {activeTab === 'terminal' && (
                                    <div className="h-full bg-charcoal/50 border border-white/5 border-t-0 backdrop-blur-sm">
                                        <LogTerminal logs={state.logs} />
                                    </div>
                                )}
                                {activeTab === 'pipeline' && (
                                    <PipelineMonitor stage={state.pipeline_stage} mappedPairs={state.mapped_pairs} pipelineEvents={state.pipeline_events || []} llmProgress={state.llm_progress} />
                                )}
                            </div>
                        </div>
                    </div>
                    <div className="flex-[0.35] min-h-0 flex flex-col gap-3">
                        <TradeLog trades={state.trades_executed} />
                        <div className="h-20 shrink-0">
                            <KillSwitch active={state.kill_switch_active} />
                        </div>
                    </div>
                </div>

                {/* Right Col: Discovery & Reasoning Feed */}
                <div className="col-span-4 row-span-12 flex flex-col gap-3 overflow-hidden">
                    {/* Top: Discovered Clusters (MapMaker Output) */}
                    <div className="flex-[0.3] bg-charcoal/50 border border-white/5 backdrop-blur-sm flex flex-col min-h-0 relative overflow-hidden">
                        <div className="p-3 border-b border-white/5 bg-black/20 flex justify-between items-center">
                            <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-[0.2em]">
                                Knowledge Map
                            </h3>
                            <div className="px-2 py-0.5 bg-neon-cyan/10 text-neon-cyan text-[8px] font-mono border border-neon-cyan/20">
                                MAPMAKER
                            </div>
                        </div>
                        <div className="flex-1 overflow-y-auto custom-scrollbar">
                            {state.clusters.length === 0 && (
                                <div className="p-4 text-center text-[10px] text-gray-600 font-mono mt-4">
                                    No clusters discovered yet...
                                </div>
                            )}
                            {state.clusters.map((c, idx) => (
                                <ClusterCard key={idx} cluster={c} onClick={setSelectedClusterId} />
                            ))}
                        </div>
                    </div>

                    {/* Bottom: Live Opportunities (Navigator Output) - SPLIT INTO 2 COLUMNS */}
                    <div className="flex-[0.7] flex flex-col gap-3 min-h-0 relative">
                        <div className="flex-1 grid grid-cols-2 gap-3 min-h-0">
                            {/* Intra-Market Arbitrage */}
                            <div className="bg-charcoal/50 border border-white/5 backdrop-blur-sm flex flex-col min-h-0 relative overflow-hidden">
                                <div className="p-2 border-b border-white/5 bg-black/40 flex justify-between items-center">
                                    <h3 className="text-[9px] font-bold text-neon-cyan uppercase tracking-widest truncate">INTRA-MARKET</h3>
                                </div>
                                <div className="flex-1 overflow-y-auto p-2 custom-scrollbar">
                                    {state.opportunities.filter(o => !o.trades?.some((t: any) => t.exchange === 'limitless')).length === 0 && (
                                        <div className="text-center text-[8px] text-gray-600 font-mono mt-10">Scanning...</div>
                                    )}
                                    {[...state.opportunities]
                                        .filter(o => !o.trades?.some((t: any) => t.exchange === 'limitless'))
                                        .reverse()
                                        .map((opp, idx) => (
                                            <OpportunityCard key={idx} opp={opp} />
                                        ))}
                                </div>
                            </div>

                            {/* Cross-Exchange Arbitrage */}
                            <div className="bg-charcoal/50 border border-white/5 backdrop-blur-sm flex flex-col min-h-0 relative overflow-hidden">
                                <div className="p-2 border-b border-white/5 bg-black/40 flex justify-between items-center">
                                    <h3 className="text-[9px] font-bold text-neon-purple uppercase tracking-widest truncate">CROSS-EXCHANGE</h3>
                                </div>
                                <div className="flex-1 overflow-y-auto p-2 custom-scrollbar">
                                    {state.opportunities.filter(o => o.trades?.some((t: any) => t.exchange === 'limitless')).length === 0 && (
                                        <div className="text-center text-[8px] text-gray-600 font-mono mt-10">Searching pairs...</div>
                                    )}
                                    {[...state.opportunities]
                                        .filter(o => o.trades?.some((t: any) => t.exchange === 'limitless'))
                                        .reverse()
                                        .map((opp, idx) => (
                                            <OpportunityCard key={idx} opp={opp} />
                                        ))}
                                </div>
                            </div>
                        </div>

                        <div className="shrink-0 p-1 flex justify-center border-t border-white/5 bg-black/20">
                            <button
                                onClick={() => setShowRawData(!showRawData)}
                                className="text-[9px] text-gray-500 hover:text-neon-cyan transition-colors flex items-center gap-1 uppercase tracking-tighter"
                            >
                                <Layers className="w-2 h-2" />
                                DEBUG_MODE
                            </button>
                        </div>
                    </div>
                </div>

                {/* Raw Debug Modal Overlay (Conditional) */}
                {showRawData && (
                    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-10">
                        <div className="bg-obsidian border border-white/10 w-full max-w-4xl max-h-full flex flex-col overflow-hidden shadow-2xl">
                            <div className="p-4 border-b border-white/5 flex justify-between items-center">
                                <span className="text-xs font-mono text-neon-cyan">SYSTEM_STATE.JSON</span>
                                <button onClick={() => setShowRawData(false)} className="text-gray-500 hover:text-white">✕</button>
                            </div>
                            <pre className="flex-1 overflow-auto p-6 text-[10px] font-mono text-neon-green/80 custom-scrollbar">
                                {JSON.stringify(state, null, 2)}
                            </pre>
                        </div>
                    </div>
                )}

                {/* Interactive Cluster Details Modal */}
                <ClusterDetailsModal
                    clusterId={selectedClusterId}
                    onClose={() => setSelectedClusterId(null)}
                />
            </main>
        </div>
    );
};
