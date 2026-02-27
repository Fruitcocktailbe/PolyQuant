import { memo } from 'react';
import { ShoppingCart, TrendingUp } from 'lucide-react';

interface Market {
    id: string;
    question: string;
    liquidity: number;
    topic: string;
}

interface MarketListProps {
    markets: Market[];
}

export const MarketList = memo<MarketListProps>(({ markets }) => {
    if (!markets || markets.length === 0) {
        return (
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                <div className="text-xs text-gray-600 font-mono text-center mt-20 flex flex-col items-center gap-4">
                    <div className="w-12 h-12 border border-dashed border-gray-700 rounded-full animate-spin-slow flex items-center justify-center">
                        <span className="w-1 h-1 bg-gray-500 rounded-full" />
                    </div>
                    <div className="space-y-1">
                        <p className="text-neon-cyan/50 animate-pulse">Scanning Markets...</p>
                        <p className="text-[9px] text-gray-700 italic">Phase 1: Discovery Active</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="flex-1 overflow-y-auto p-2 custom-scrollbar space-y-2">
            <div className="px-2 py-1 flex justify-between items-center border-b border-white/5 mb-2">
                <span className="text-[10px] font-mono text-gray-500 uppercase">Live Pipeline</span>
                <span className="text-[10px] font-mono text-neon-green">{markets.length} Markets</span>
            </div>
            {markets.map((market) => (
                <div
                    key={market.id}
                    className="bg-black/30 border border-white/5 p-3 hover:border-neon-cyan/30 transition-all group relative overflow-hidden"
                >
                    <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
                        <ShoppingCart className="w-8 h-8 text-white" />
                    </div>

                    <div className="flex justify-between items-start mb-1 gap-2">
                        <span className="text-[9px] uppercase tracking-wider text-neon-cyan font-bold truncate">
                            {market.topic || 'General'}
                        </span>
                        <span className="text-[9px] font-mono text-gray-500 whitespace-nowrap">
                            ID: {market.id.slice(-6)}
                        </span>
                    </div>

                    <p className="text-xs text-gray-200 line-clamp-2 mb-2 font-medium leading-relaxed">
                        {market.question}
                    </p>

                    <div className="flex justify-between items-center text-[10px] font-mono">
                        <div className="flex items-center gap-1.5 text-neon-green">
                            <TrendingUp className="w-3 h-3" />
                            ${market.liquidity.toLocaleString()}
                        </div>
                        <div className="text-gray-500 italic">
                            Liquidity
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
});

MarketList.displayName = 'MarketList';
