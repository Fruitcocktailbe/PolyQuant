import React, { useRef, useEffect, memo, useMemo } from 'react';
import {
    Search, Loader2, CheckCircle2, Link2, Zap, AlertTriangle,
    Database, Brain, Globe, Clock
} from 'lucide-react';

interface PipelineEvent {
    ts: string;
    stage: string;
    type: string;   // info | llm_start | llm_success | llm_fail | cache_hit | validation_fail | match | api_fetch
    message: string;
    detail?: string;
    duration?: number;
}

interface PipelineMonitorProps {
    stage: string;
    mappedPairs: any[];
    pipelineEvents: PipelineEvent[];
}

const STAGES = [
    { key: 'DISCOVERY', label: 'Discovery', icon: Search },
    { key: 'LOGIC', label: 'Logic', icon: Brain },
    { key: 'MATCHING', label: 'Matching', icon: Link2 },
    { key: 'COMPLETE', label: 'Complete', icon: CheckCircle2 },
];

/** Color & icon lookup per event type */
const EVENT_STYLES: Record<string, { color: string; borderColor: string; icon: React.ReactNode; label: string }> = {
    info:            { color: 'text-gray-300',   borderColor: 'border-gray-600',         icon: <Clock className="w-3 h-3 text-gray-400" />,           label: 'INFO' },
    llm_start:       { color: 'text-neon-cyan',  borderColor: 'border-neon-cyan/40',     icon: <Brain className="w-3 h-3 text-neon-cyan" />,           label: 'LLM' },
    llm_success:     { color: 'text-neon-green',  borderColor: 'border-neon-green/40',   icon: <CheckCircle2 className="w-3 h-3 text-neon-green" />,   label: 'LLM ✓' },
    llm_fail:        { color: 'text-neon-red',    borderColor: 'border-neon-red/40',     icon: <AlertTriangle className="w-3 h-3 text-neon-red" />,    label: 'LLM ✗' },
    cache_hit:       { color: 'text-yellow-400',  borderColor: 'border-yellow-500/40',   icon: <Zap className="w-3 h-3 text-yellow-400" />,            label: 'CACHE' },
    validation_fail: { color: 'text-orange-400',  borderColor: 'border-orange-500/40',   icon: <AlertTriangle className="w-3 h-3 text-orange-400" />,  label: 'REJECT' },
    match:           { color: 'text-neon-purple',  borderColor: 'border-neon-purple/40', icon: <Link2 className="w-3 h-3 text-neon-purple" />,         label: 'MATCH' },
    api_fetch:       { color: 'text-blue-400',    borderColor: 'border-blue-500/40',     icon: <Globe className="w-3 h-3 text-blue-400" />,            label: 'API' },
};

const STAGE_COLORS: Record<string, string> = {
    DISCOVERY: 'bg-neon-cyan/20 text-neon-cyan',
    LOGIC:     'bg-neon-purple/20 text-neon-purple',
    MATCHING:  'bg-neon-green/20 text-neon-green',
    COMPLETE:  'bg-white/10 text-white',
};

/** Single event card in the timeline */
const EventCard = memo<{ event: PipelineEvent }>(({ event }) => {
    const style = EVENT_STYLES[event.type] || EVENT_STYLES.info;
    const stageColor = STAGE_COLORS[event.stage] || 'bg-white/5 text-gray-400';
    const time = useMemo(() => {
        try {
            return new Date(event.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        } catch {
            return '--:--:--';
        }
    }, [event.ts]);

    return (
        <div className={`flex items-start gap-2 p-2 border-l-2 ${style.borderColor} bg-black/30 hover:bg-white/5 transition-colors animate-in slide-in-from-bottom-1 duration-200`}>
            {/* Icon */}
            <div className="pt-0.5 shrink-0">{style.icon}</div>

            {/* Body */}
            <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                    {/* Stage badge */}
                    <span className={`text-[7px] px-1.5 py-0.5 font-black uppercase tracking-widest rounded-none ${stageColor}`}>
                        {event.stage}
                    </span>
                    {/* Type badge */}
                    <span className={`text-[7px] font-bold uppercase tracking-wider ${style.color}`}>
                        {style.label}
                    </span>
                    {/* Duration */}
                    {event.duration !== undefined && (
                        <span className="text-[8px] text-gray-500 font-mono ml-auto shrink-0">
                            {event.duration.toFixed(1)}s
                        </span>
                    )}
                </div>
                <div className={`text-[10px] mt-0.5 ${style.color} font-medium leading-snug break-words`}>
                    {event.message}
                </div>
                {event.detail && (
                    <div className="text-[9px] text-gray-500 mt-0.5 leading-snug break-words">
                        {event.detail}
                    </div>
                )}
            </div>

            {/* Timestamp */}
            <div className="text-[8px] text-gray-600 font-mono shrink-0 pt-0.5">{time}</div>
        </div>
    );
});
EventCard.displayName = 'EventCard';


export const PipelineMonitor: React.FC<PipelineMonitorProps> = memo(({ stage, mappedPairs: _mappedPairs, pipelineEvents }) => {
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll when new events arrive (only if user is near bottom)
    useEffect(() => {
        const el = scrollRef.current;
        if (!el) return;
        const isNearBottom = el.scrollHeight - el.scrollTop <= el.clientHeight + 80;
        if (isNearBottom) {
            el.scrollTop = el.scrollHeight;
        }
    }, [pipelineEvents.length]);

    const eventCount = pipelineEvents.length;

    return (
        <div className="bg-charcoal/50 border border-white/5 backdrop-blur-sm flex-1 flex flex-col min-h-0 relative overflow-hidden group">
            {/* Header / Stepper */}
            <div className="p-3 border-b border-white/5 bg-black/20 flex gap-3 overflow-x-auto custom-scrollbar items-center">
                {STAGES.map((s, idx) => {
                    const isActive = stage === s.key;
                    const isCompleted = STAGES.findIndex(st => st.key === stage) > idx;
                    const Icon = s.icon;

                    return (
                        <div key={s.key} className="flex items-center gap-1.5 min-w-fit">
                            <div className={`p-1 rounded-none border ${isActive ? 'bg-neon-cyan/20 border-neon-cyan text-neon-cyan animate-pulse' :
                                    isCompleted ? 'bg-neon-green/10 border-neon-green/30 text-neon-green' :
                                        'bg-white/5 border-white/10 text-gray-600'
                                }`}>
                                <Icon className="w-2.5 h-2.5" />
                            </div>
                            <span className={`text-[9px] font-bold uppercase tracking-widest ${isActive ? 'text-white' : isCompleted ? 'text-neon-green/70' : 'text-gray-600'
                                }`}>
                                {s.label}
                            </span>
                            {idx < STAGES.length - 1 && <div className="ml-1 w-3 h-[1px] bg-white/5" />}
                        </div>
                    );
                })}

                <div className="ml-auto flex items-center gap-2">
                    <Database className="w-3 h-3 text-gray-600" />
                    <span className="text-[9px] text-gray-500 font-mono">{eventCount} events</span>
                </div>
            </div>

            {/* Timeline Feed */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto custom-scrollbar space-y-1 p-2"
            >
                {eventCount === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-gray-600 gap-4 opacity-50">
                        <Loader2 className="w-8 h-8 animate-spin" />
                        <span className="text-[10px] uppercase tracking-widest font-bold">
                            Waiting for MapMaker to start...
                        </span>
                    </div>
                ) : (
                    pipelineEvents.map((evt, idx) => (
                        <EventCard key={`${idx}-${evt.ts}`} event={evt} />
                    ))
                )}
            </div>

            {/* Footer */}
            <div className="p-2 bg-black/40 border-t border-white/5 flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 bg-neon-green rounded-full animate-ping" />
                    <span className="text-[8px] text-gray-500 font-bold uppercase tracking-widest">Pipeline Timeline</span>
                </div>
                <div className="text-[8px] text-gray-600 font-mono tracking-tighter">
                    POLYQUANT_MAPMAKER_V2.0
                </div>
            </div>
        </div>
    );
});

PipelineMonitor.displayName = 'PipelineMonitor';
