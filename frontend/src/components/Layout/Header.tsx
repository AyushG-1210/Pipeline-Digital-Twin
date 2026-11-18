import { useState } from 'react';
import { usePipelineStore } from '../../store/usePipelineStore';
import { useThemeStore } from '../../store/useThemeStore';
import { formatTime } from '../../utils/colors';
import LiveFeedGalleryModal from '../CVPanel/LiveFeedGalleryModal';

/**
 * Application header — segment health summary, live camera feeds launcher, and connection status.
 */
export default function Header() {
    const wsConnected = usePipelineStore(state => state.wsConnected);
    const lastUpdate = usePipelineStore(state => state.lastUpdate);
    const segments = usePipelineStore(state => state.segments);
    const theme = useThemeStore(state => state.theme);
    const toggleTheme = useThemeStore(state => state.toggleTheme);

    const [isGalleryOpen, setIsGalleryOpen] = useState(false);

    const segmentArray = Array.from(segments.values());
    const criticalCount = segmentArray.filter(s => s.integrity < 0.3).length;
    const warningCount = segmentArray.filter(s => s.integrity >= 0.3 && s.integrity < 0.6).length;
    const healthyCount = segmentArray.length - criticalCount - warningCount;

    return (
        <header className="bg-surface border-b border-line px-6 py-3.5">
            <div className="flex items-center justify-between gap-6">
                <div>
                    <h1 className="text-lg font-semibold text-ink tracking-tight">
                        Pipeline Integrity Digital Twin
                    </h1>
                    <p className="text-xs text-ink-muted mt-0.5">
                        Physics-informed inspection, live camera feed &amp; degradation forecasting
                    </p>
                </div>

                <div className="flex items-center gap-4 sm:gap-5">
                    {/* Live Camera Feed & YOLO Predictions Button */}
                    <button
                        type="button"
                        onClick={() => setIsGalleryOpen(true)}
                        className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-line bg-surface hover:bg-surface-sunken text-xs font-semibold text-ink transition-all shadow-sm group"
                        title="Open Live Optical Camera Feeds & YOLOv8 Defect Predictions Gallery"
                    >
                        <div className="relative flex items-center justify-center">
                            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                        </div>
                        <svg className="w-4 h-4 text-primary group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                        <span>Live Feed &amp; CV</span>
                        <span className="px-1.5 py-0.2 rounded-full text-[10px] font-mono bg-emerald-100 text-emerald-800 border border-emerald-300 font-bold">
                            20 CAM
                        </span>
                    </button>

                    <div className="w-px h-6 bg-line hidden sm:block" />

                    {/* Benchmark summary */}
                    <div className="flex items-center gap-3 text-xs">
                        <span className="flex items-center gap-1.5" title="Healthy segments">
                            <span className="w-2 h-2 rounded-full" style={{ background: '#4C9A78' }} />
                            <span className="text-ink-soft font-medium">{healthyCount}</span>
                        </span>
                        <span className="flex items-center gap-1.5" title="Warning tier">
                            <span className="w-2 h-2 rounded-full" style={{ background: '#D98E2B' }} />
                            <span className="text-ink-soft font-medium">{warningCount}</span>
                        </span>
                        <span className="flex items-center gap-1.5" title="Critical alert">
                            <span className="w-2 h-2 rounded-full" style={{ background: '#D6473C' }} />
                            <span className="text-ink-soft font-medium">{criticalCount}</span>
                        </span>
                    </div>

                    <div className="w-px h-6 bg-line" />

                    {/* Connection status */}
                    <div className="flex items-center gap-2">
                        <div
                            className="w-1.5 h-1.5 rounded-full"
                            style={{ background: wsConnected ? '#68B0AB' : '#D6473C' }}
                        />
                        <span className="text-xs text-ink-soft">
                            {wsConnected ? 'Live' : 'Disconnected'}
                        </span>
                    </div>

                    {lastUpdate && (
                        <div className="text-xs font-mono text-ink-muted hidden md:block">
                            {formatTime(lastUpdate)}
                        </div>
                    )}

                    <div className="w-px h-6 bg-line" />

                    {/* Theme toggle */}
                    <button
                        type="button"
                        onClick={toggleTheme}
                        aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                        title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                        className="relative w-10 h-5.5 rounded-full border border-line bg-surface-sunken shrink-0
                                   transition-colors hover:border-accent/40 focus-visible:outline-accent"
                        style={{ height: '1.375rem' }}
                    >
                        <span
                            className="absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-primary shadow-sm
                                       flex items-center justify-center transition-transform duration-200"
                            style={{ transform: theme === 'dark' ? 'translateX(1.125rem)' : 'translateX(0)' }}
                        >
                            {theme === 'dark' ? (
                                <svg className="w-2.5 h-2.5 text-canvas" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                                </svg>
                            ) : (
                                <svg className="w-2.5 h-2.5 text-canvas" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1zm6 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1z" />
                                </svg>
                            )}
                        </span>
                    </button>
                </div>
            </div>

            {/* Live Feed Gallery Modal */}
            <LiveFeedGalleryModal
                isOpen={isGalleryOpen}
                onClose={() => setIsGalleryOpen(false)}
            />
        </header>
    );
}
