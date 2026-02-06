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
        <div className="panel h-full flex flex-col font-mono text-sm bg-black border-gray-800">
            <div className="flex items-center gap-2 mb-2 text-gray-500 border-b border-gray-800 pb-2">
                <Terminal className="w-4 h-4" />
                <span className="uppercase text-xs tracking-wider">System Logs</span>
                {logs.length > 50 && (
                    <span className="text-xs text-gray-600 ml-auto">
                        Showing last 50 of {logs.length}
                    </span>
                )}
            </div>

            <div
                ref={containerRef}
                className="flex-1 overflow-y-auto space-y-1 scrollbar-thin scrollbar-thumb-gray-800 p-2"
            >
                {visibleLogs.length === 0 && (
                    <div className="text-gray-600 italic">No logs received yet...</div>
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
