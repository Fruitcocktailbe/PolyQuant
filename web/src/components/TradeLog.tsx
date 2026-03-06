import React, { useEffect, useRef } from 'react';
import { DollarSign, CheckCircle, Clock } from 'lucide-react';

interface TradeLogProps {
    trades: any[];
}

export const TradeLog: React.FC<TradeLogProps> = ({ trades }) => {
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [trades.length]);

    return (
        <div className="flex-1 flex flex-col bg-charcoal/30 border border-white/5 backdrop-blur-sm overflow-hidden min-h-0">
            <div className="h-10 px-4 border-b border-white/5 flex items-center justify-between bg-black/20">
                <div className="flex items-center gap-2">
                    <Clock className="w-3.5 h-3.5 text-neon-cyan" />
                    <span className="text-[10px] uppercase tracking-widest font-bold text-gray-400">Execution History</span>
                </div>
                <span className="text-[10px] font-mono text-neon-cyan/50">{trades.length} FILLS</span>
            </div>

            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto p-2 space-y-1 font-mono text-[11px] custom-scrollbar"
            >
                {trades.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-gray-500 gap-3 opacity-60 mt-8 mb-8">
                        <DollarSign className="w-10 h-10 text-gray-600" />
                        <span className="tracking-widest uppercase font-bold text-[10px]">Awaiting Fills...</span>
                    </div>
                ) : (
                    trades.map((trade, i) => (
                        <div
                            key={`${trade.order_id}-${i}`}
                            className="bg-black/40 border border-white/5 p-2 flex items-center justify-between hover:border-neon-cyan/20 transition-colors"
                        >
                            <div className="flex items-center gap-3">
                                <div className={`w-1 h-6 ${trade.trade?.side?.toLowerCase() === 'buy' ? 'bg-neon-green' : 'bg-neon-red'}`} />
                                <div className="flex flex-col">
                                    <div className="flex items-center gap-2">
                                        <span className={trade.trade?.side?.toLowerCase() === 'buy' ? 'text-neon-green' : 'text-neon-red'}>
                                            {trade.trade?.side?.toUpperCase() || 'UNKNOWN'}
                                        </span>
                                        <span className="text-white font-bold">{trade.trade?.outcome_id || 'Unknown Outcome'}</span>
                                        <span className="text-gray-500">@</span>
                                        <span className="text-neon-cyan">{trade.filled_price?.toFixed(3) || '0.000'}</span>
                                    </div>
                                    <div className="flex items-center gap-2 text-[9px] text-gray-500">
                                        <span className="uppercase">{trade.trade?.exchange || 'unknown'}</span>
                                        <span>•</span>
                                        <span>ID: {trade.order_id?.slice(0, 8) || 'N/A'}</span>
                                        <span>•</span>
                                        <span>{trade.fill_time ? new Date(trade.fill_time).toLocaleTimeString() : 'N/A'}</span>
                                    </div>
                                </div>
                            </div>
                            <div className="text-right">
                                <div className="text-white font-bold">{trade.filled_size} shares</div>
                                <div className="text-neon-green text-[9px] flex items-center justify-end gap-1">
                                    <CheckCircle className="w-2.5 h-2.5" />
                                    FILLED
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};
