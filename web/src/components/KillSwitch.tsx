import { useState, memo, useCallback } from 'react';
import { api } from '../services/api';
import { AlertTriangle, XOctagon } from 'lucide-react';
// @ts-ignore
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

    if (active || (triggered && active)) {
        return (
            <div className="h-full flex items-center justify-center bg-neon-red/10 border border-neon-red/50 shadow-neon-red animate-pulse relative overflow-hidden group">
                <div className="absolute inset-0 bg-stripes opacity-20" />
                <div className="z-10 flex flex-col items-center">
                    <XOctagon className="w-6 h-6 text-neon-red mb-1" />
                    <span className="text-sm font-bold text-white tracking-[0.2em] drop-shadow-md">
                        SYSTEM HALTED
                    </span>
                    <button
                        onClick={async () => {
                            await api.resetKillSwitch();
                            setTriggered(false);
                        }}
                        className="mt-2 px-3 py-1 bg-charcoal border border-neon-red/50 text-[10px] text-white hover:bg-neon-red hover:text-white transition-colors uppercase tracking-widest font-bold z-20"
                    >
                        Resume Trading
                    </button>
                </div>
            </div>
        );
    }

    return (
        <>
            <button
                onClick={openConfirm}
                className="w-full h-full bg-neon-red/5 hover:bg-neon-red/10 border border-neon-red/20 hover:border-neon-red text-neon-red transition-all duration-300 flex flex-col items-center justify-center gap-2 group relative overflow-hidden"
            >
                <div className="absolute inset-0 bg-stripes opacity-[0.05] group-hover:opacity-10 transition-opacity"></div>
                <AlertTriangle className="w-6 h-6 group-hover:scale-110 transition-transform" />
                <span className="text-sm font-bold font-mono tracking-[0.2em] group-hover:text-white transition-colors">EMERGENCY STOP</span>
            </button>

            <AnimatePresence>
                {showConfirm && (
                    <div className="fixed inset-0 bg-black/90 backdrop-blur-md z-50 flex items-center justify-center p-4">
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            className="bg-charcoal border border-neon-red/50 p-8 max-w-md w-full shadow-neon-red rounded-none relative"
                        >
                            <div className="absolute top-0 left-0 w-full h-1 bg-neon-red shadow-neon-red" />

                            <AlertTriangle className="w-16 h-16 text-neon-red mx-auto mb-6 animate-pulse" />
                            <h2 className="text-2xl font-bold text-white mb-2 text-center tracking-widest font-header">CONFIRM PURGE</h2>
                            <p className="text-gray-400 mb-8 text-center text-sm font-mono border-l-2 border-neon-red/30 pl-4 mx-4">
                                This will immediately stop all active solvers and attempt to liquidate all positions at market price.
                            </p>

                            <div className="flex gap-4 font-mono text-sm">
                                <button
                                    onClick={closeConfirm}
                                    className="flex-1 py-3 px-6 bg-transparent border border-gray-700 text-gray-400 hover:text-white hover:border-white transition-all uppercase tracking-wider"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handlePanic}
                                    className="flex-1 py-3 px-6 bg-neon-red/10 border border-neon-red text-neon-red hover:bg-neon-red hover:text-white hover:shadow-neon-red transition-all uppercase tracking-wider font-bold"
                                >
                                    Execute
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
