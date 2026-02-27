import { useEffect, useRef, memo, useMemo } from 'react';
import { Terminal } from 'lucide-react';

interface LogTerminalProps {
    logs: string[];
}

// Memoized log line prevents re-render of unchanged logs
const LogLine = memo<{ log: string }>(({ log }) => {
    const timestamp = useMemo(() => new Date().toLocaleTimeString(), []);

    return (
        <div className="text-gray-300 break-words hover:bg-gray-900/50 px-1 rounded">
            <span className="text-gray-600 mr-2">[{timestamp}]</span>
            {log}
        </div>
    );
});

LogLine.displayName = 'LogLine';

export const LogTerminal = memo<LogTerminalProps>(({ logs }) => {
    const bottomRef = useRef<HTMLDivElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    // Auto-scroll only if already at bottom (don't interrupt manual scrolling)
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 50;
        if (isAtBottom) {
            bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs.length]);

    // Only render last 50 logs for performance (older logs are in memory but not DOM)
    const visibleLogs = useMemo(() => logs.slice(-50), [logs]);

    return (
        <div className="h-full flex flex-col font-mono text-xs bg-black/40 text-gray-300 relative">
            <div className="flex items-center gap-2 p-3 border-b border-white/5 bg-black/20">
                <Terminal className="w-3 h-3 text-neon-cyan" />
                <span className="uppercase text-[10px] tracking-[0.2em] font-bold text-gray-500">System Logs</span>
                {logs.length > 50 && (
                    <span className="text-[10px] text-gray-700 ml-auto">
                        LIVE | TAILING
                    </span>
                )}
            </div>

            <div
                ref={containerRef}
                className="flex-1 overflow-y-auto space-y-0.5 p-2 font-mono scrollbar-hide"
            >
                {visibleLogs.length === 0 && (
                    <div className="text-gray-700 italic text-[10px] p-2">Waiting for pipeline events...</div>
                )}
                {visibleLogs.map((log, i) => (
                    <LogLine
                        key={`${logs.length - visibleLogs.length + i}-${log.slice(0, 20)}`}
                        log={log}
                    />
                ))}
                <div ref={bottomRef} />
            </div>
        </div>
    );
});

LogTerminal.displayName = 'LogTerminal';
