import React, { useRef, useEffect, memo } from 'react';
import { Search, Loader2, CheckCircle2, Link2, ExternalLink } from 'lucide-react';

interface MappedPair {
    polymarket_question: string;
    limitless_title: string;
    polymarket_id: string;
    limitless_id: string;
    similarity: number;
}

interface PipelineMonitorProps {
    stage: string;
    mappedPairs: MappedPair[];
}

const STAGES = [
    { key: 'DISCOVERY', label: 'Market Discovery', icon: Search },
    { key: 'LOGIC', label: 'Logical Analysis', icon: Search },
    { key: 'MATCHING', label: 'Cross-Exchange Mapping', icon: Link2 },
    { key: 'COMPLETE', label: 'Building Complete', icon: CheckCircle2 },
];

export const PipelineMonitor: React.FC<PipelineMonitorProps> = memo(({ stage, mappedPairs }) => {
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [mappedPairs]);

    return (
        <div className="bg-charcoal/50 border border-white/5 backdrop-blur-sm flex-1 flex flex-col min-h-0 relative overflow-hidden group">
            {/* Header / Stepper */}
            <div className="p-4 border-b border-white/5 bg-black/20 flex gap-4 overflow-x-auto custom-scrollbar">
                {STAGES.map((s, idx) => {
                    const isActive = stage === s.key;
                    const isCompleted = STAGES.findIndex(st => st.key === stage) > idx;
                    const Icon = s.icon;

                    return (
                        <div key={s.key} className="flex items-center gap-2 min-w-fit">
                            <div className={`p-1.5 rounded-none border ${isActive ? 'bg-neon-cyan/20 border-neon-cyan text-neon-cyan animate-pulse' :
                                    isCompleted ? 'bg-neon-green/10 border-neon-green/30 text-neon-green' :
                                        'bg-white/5 border-white/10 text-gray-600'
                                }`}>
                                <Icon className="w-3 h-3" />
                            </div>
                            <span className={`text-[10px] font-bold uppercase tracking-widest ${isActive ? 'text-white' : isCompleted ? 'text-neon-green/70' : 'text-gray-600'
                                }`}>
                                {s.label}
                            </span>
                            {idx < STAGES.length - 1 && <div className="ml-2 w-4 h-[1px] bg-white/5" />}
                        </div>
                    );
                })}
            </div>

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col p-4 overflow-hidden">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-[0.2em] flex items-center gap-2">
                        <Link2 className="w-3 h-3 text-neon-cyan" /> Cross-Exchange Discovery
                    </h3>
                    <span className="text-[10px] text-neon-green font-mono">{mappedPairs.length} PAIRS FOUND</span>
                </div>

                <div
                    ref={scrollRef}
                    className="flex-1 overflow-y-auto custom-scrollbar space-y-2 pr-2"
                >
                    {mappedPairs.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center text-gray-600 gap-4 opacity-50">
                            <Loader2 className="w-8 h-8 animate-spin" />
                            <span className="text-[10px] uppercase tracking-widest font-bold">
                                {stage === 'MATCHING' ? 'Searching for Limitless matches...' : 'Waiting for Matcher stage...'}
                            </span>
                        </div>
                    ) : (
                        mappedPairs.map((pair, idx) => (
                            <div
                                key={idx}
                                className="p-3 bg-black/40 border border-white/5 hover:border-neon-cyan/30 transition-all group animate-in slide-in-from-bottom-2 duration-300"
                            >
                                <div className="flex justify-between items-start mb-2 gap-4">
                                    <div className="flex-1">
                                        <div className="text-[10px] text-white font-medium mb-1 line-clamp-1">{pair.polymarket_question}</div>
                                        <div className="text-[8px] text-neon-cyan font-mono uppercase tracking-tighter flex items-center gap-1">
                                            Polymarket <ExternalLink className="w-2 h-2" /> {pair.polymarket_id.slice(0, 12)}...
                                        </div>
                                    </div>
                                    <div className="bg-neon-cyan/10 border border-neon-cyan/20 px-2 py-1 flex flex-col items-center">
                                        <div className="text-[8px] text-neon-cyan font-bold leading-none">{(pair.similarity * 100).toFixed(0)}%</div>
                                        <div className="text-[6px] text-neon-cyan/70 uppercase font-black leading-none mt-1">SIM</div>
                                    </div>
                                </div>

                                <div className="h-[1px] w-full bg-gradient-to-r from-neon-cyan/20 to-transparent my-2" />

                                <div className="flex-1">
                                    <div className="text-[10px] text-gray-400 font-medium mb-1 line-clamp-1">{pair.limitless_title}</div>
                                    <div className="text-[8px] text-neon-purple font-mono uppercase tracking-tighter flex items-center gap-1">
                                        Limitless <ExternalLink className="w-2 h-2" /> {pair.limitless_id.slice(0, 12)}...
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Footer Overlay */}
            <div className="p-3 bg-black/40 border-t border-white/5 flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 bg-neon-green rounded-full animate-ping" />
                    <span className="text-[8px] text-gray-500 font-bold uppercase tracking-widest">Pipeline Live</span>
                </div>
                <div className="text-[8px] text-gray-600 font-mono tracking-tighter">
                    POLYQUANT_MAPPING_V2.0 // BYPASSING_GEOBLOCK=TRUE
                </div>
            </div>
        </div>
    );
});

PipelineMonitor.displayName = 'PipelineMonitor';
