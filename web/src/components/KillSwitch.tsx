import { useState, memo, useCallback } from 'react';
import { api } from '../services/api';
import { AlertTriangle, XOctagon } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface KillSwitchProps {
    active: boolean;
}

export const KillSwitch = memo<KillSwitchProps>(({ active }) => {
    const [showConfirm, setShowConfirm] = useState(false);
    const [triggered, setTriggered] = useState(false);

    const handlePanic = useCallback(async () => {
        await api.triggerKillSwitch();
        setTriggered(true);
        setShowConfirm(false);
    }, []);

    const openConfirm = useCallback(() => setShowConfirm(true), []);
    const closeConfirm = useCallback(() => setShowConfirm(false), []);

    if (active || triggered) {
        return (
            <div className="panel flex items-center justify-center bg-red-900/20 border-red-500 animate-pulse h-full">
                <XOctagon className="w-8 h-8 text-red-500 mr-3" />
                <span className="text-xl font-bold text-red-500 tracking-widest">
                    KILL SWITCH ACTIVE - HALTING
                </span>
            </div>
        );
    }

    return (
        <>
            <button
                onClick={openConfirm}
                className="w-full h-full min-h-[60px] bg-red-950 border-2 border-red-600 text-red-500 hover:bg-red-600 hover:text-white transition-all duration-200 flex items-center justify-center gap-3 group relative overflow-hidden"
            >
                <div className="absolute inset-0 bg-stripes opacity-10 group-hover:opacity-20"></div>
                <AlertTriangle className="w-6 h-6" />
                <span className="text-lg font-bold font-mono tracking-widest">PANIC SELL ALL</span>
            </button>

            <AnimatePresence>
                {showConfirm && (
                    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            className="bg-[#1a0f0f] border-2 border-red-500 p-8 max-w-md w-full shadow-[0_0_50px_rgba(255,42,109,0.3)] rounded-lg text-center"
                        >
                            <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-6" />
                            <h2 className="text-2xl font-bold text-white mb-2">CONFIRM LIQUIDATION</h2>
                            <p className="text-gray-400 mb-8">
                                Are you sure? This will immediately stop all solvers and attempt to market sell all open positions. This action cannot be undone.
                            </p>

                            <div className="flex gap-4">
                                <button
                                    onClick={closeConfirm}
                                    className="flex-1 py-3 px-6 bg-transparent border border-gray-600 text-gray-300 hover:border-white hover:text-white transition-colors font-mono"
                                >
                                    CANCEL
                                </button>
                                <button
                                    onClick={handlePanic}
                                    className="flex-1 py-3 px-6 bg-red-600 hover:bg-red-700 text-white font-bold tracking-wider shadow-lg shadow-red-900/50 transition-all font-mono"
                                >
                                    CONFIRM EXECUTION
                                </button>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </>
    );
});

KillSwitch.displayName = 'KillSwitch';
