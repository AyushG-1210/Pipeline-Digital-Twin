import { useState, useEffect } from 'react';
import type { LiveFeedItem, LiveFeedSummary } from '../../types';

interface LiveFeedViewerProps {
    initialImageFile?: string;
    onSelectImage?: (imageFile: string) => void;
    compact?: boolean;
}

const CLASS_COLORS: Record<string, { bg: string; text: string; border: string }> = {
    rust: { bg: '#FEF2F2', text: '#DC2626', border: '#FECACA' },
    corrosion_zone: { bg: '#F0FDFA', text: '#0D9488', border: '#99F6E4' },
    none: { bg: '#F3F4F6', text: '#6B7280', border: '#E5E7EB' },
};

const SEVERITY_COLORS: Record<string, { bg: string; text: string }> = {
    HIGH: { bg: '#FEE2E2', text: '#991B1B' },
    MODERATE: { bg: '#FEF3C7', text: '#92400E' },
    LOW: { bg: '#FEF9C3', text: '#854D0E' },
};

export default function LiveFeedViewer({ initialImageFile, compact = false }: LiveFeedViewerProps) {
    const [feedItems, setFeedItems] = useState<LiveFeedItem[]>([]);
    const [summary, setSummary] = useState<LiveFeedSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Active image selection
    const [currentIndex, setCurrentIndex] = useState(0);
    // View mode: 'annotated' | 'raw' | 'split'
    const [viewMode, setViewMode] = useState<'annotated' | 'raw' | 'split'>('annotated');

    // Fetch live feed detections & summary
    useEffect(() => {
        let mounted = true;

        async function fetchFeed() {
            try {
                // Try backend endpoint first
                let data = null;
                try {
                    const res = await fetch('http://localhost:3001/api/cv/live-feed');
                    if (res.ok) {
                        data = await res.json();
                    }
                } catch {
                    // Fall back to public static JSON
                }

                if (!data) {
                    const [resDet, resSum] = await Promise.all([
                        fetch('/data/detections.json'),
                        fetch('/data/summary.json'),
                    ]);
                    if (!resDet.ok) throw new Error('Could not load detections data');
                    const detJson = await resDet.json();
                    const sumJson = resSum.ok ? await resSum.json() : null;
                    data = {
                        success: true,
                        summary: sumJson,
                        items: detJson.map((d: LiveFeedItem) => ({
                            ...d,
                            raw_image_url: `/data/live_feed/${d.image_file}`,
                            annotated_image_url: `/data/annotated_feed/annotated_${d.image_file}`,
                        })),
                    };
                }

                if (mounted && data?.items) {
                    setFeedItems(data.items);
                    setSummary(data.summary || null);

                    if (initialImageFile) {
                        const idx = data.items.findIndex((item: LiveFeedItem) => item.image_file === initialImageFile);
                        if (idx >= 0) setCurrentIndex(idx);
                    }
                }
            } catch (err: unknown) {
                if (mounted) {
                    setError(err instanceof Error ? err.message : 'Failed to load live feed');
                }
            } finally {
                if (mounted) setLoading(false);
            }
        }

        fetchFeed();
        return () => { mounted = false; };
    }, [initialImageFile]);

    // Handle initialImageFile updates from parent
    useEffect(() => {
        if (initialImageFile && feedItems.length > 0) {
            const idx = feedItems.findIndex(i => i.image_file === initialImageFile);
            if (idx >= 0) setCurrentIndex(idx);
        }
    }, [initialImageFile, feedItems]);

    const activeItem = feedItems[currentIndex];

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center p-8 text-ink-muted">
                <svg className="animate-spin w-6 h-6 mb-2 text-primary" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                <span className="text-xs">Loading live pipeline camera feed…</span>
            </div>
        );
    }

    if (error || !activeItem) {
        return (
            <div className="p-4 rounded-lg bg-red-50 text-red-700 text-xs font-mono">
                {error || 'No live feed images available'}
            </div>
        );
    }

    const peakDetection = activeItem.detections && activeItem.detections.length > 0
        ? [...activeItem.detections].sort((a, b) => b.confidence - a.confidence)[0]
        : null;

    const sectionNum = activeItem.image_file.replace(/[^0-9]/g, '');

    return (
        <div className="flex flex-col space-y-3">
            {/* Top Toolbar: Section Selector + View Mode */}
            <div className="flex flex-wrap items-center justify-between gap-2 bg-surface-sunken p-2.5 rounded-lg border border-line">
                <div className="flex items-center gap-2">
                    <span className="flex h-2 w-2 relative">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                    </span>
                    <span className="text-xs font-bold text-ink tracking-wide font-mono">
                        CAM SECTION {sectionNum}
                    </span>
                    <span className="text-[10px] text-ink-muted hidden sm:inline">
                        ({currentIndex + 1} of {feedItems.length})
                    </span>
                </div>

                {/* Navigation & Mode toggles */}
                <div className="flex items-center gap-1.5">
                    {/* View Mode Buttons */}
                    <div className="flex rounded-md border border-line bg-surface p-0.5 text-xs">
                        <button
                            type="button"
                            onClick={() => setViewMode('annotated')}
                            className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                                viewMode === 'annotated' ? 'bg-primary text-white shadow-sm' : 'text-ink-muted hover:text-ink'
                            }`}
                            title="Show YOLOv8 AI segmentation and bounding box overlay"
                        >
                            AI Overlay
                        </button>
                        <button
                            type="button"
                            onClick={() => setViewMode('raw')}
                            className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                                viewMode === 'raw' ? 'bg-primary text-white shadow-sm' : 'text-ink-muted hover:text-ink'
                            }`}
                            title="Show original unannotated live camera frame"
                        >
                            Raw
                        </button>
                        <button
                            type="button"
                            onClick={() => setViewMode('split')}
                            className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                                viewMode === 'split' ? 'bg-primary text-white shadow-sm' : 'text-ink-muted hover:text-ink'
                            }`}
                            title="Side-by-side comparison"
                        >
                            Split
                        </button>
                    </div>

                    {/* Prev / Next Buttons */}
                    <div className="flex items-center gap-1">
                        <button
                            type="button"
                            onClick={() => setCurrentIndex((prev) => (prev > 0 ? prev - 1 : feedItems.length - 1))}
                            className="p-1 rounded bg-surface border border-line hover:bg-surface-sunken text-ink transition-colors"
                            title="Previous pipeline section"
                        >
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                            </svg>
                        </button>
                        <button
                            type="button"
                            onClick={() => setCurrentIndex((prev) => (prev < feedItems.length - 1 ? prev + 1 : 0))}
                            className="p-1 rounded bg-surface border border-line hover:bg-surface-sunken text-ink transition-colors"
                            title="Next pipeline section"
                        >
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                            </svg>
                        </button>
                    </div>
                </div>
            </div>

            {/* Quick Section Jump Strip */}
            <div className="flex items-center gap-1 overflow-x-auto pb-1 scrollbar-thin">
                {feedItems.map((item, idx) => {
                    const isSelected = idx === currentIndex;
                    const num = item.image_file.replace(/[^0-9]/g, '');
                    const hasDefect = item.defect_detected;
                    return (
                        <button
                            key={item.image_file}
                            type="button"
                            onClick={() => setCurrentIndex(idx)}
                            className={`px-2 py-1 text-[10px] font-mono rounded border transition-all shrink-0 flex items-center gap-1 ${
                                isSelected
                                    ? 'bg-primary text-white border-primary shadow-sm font-bold'
                                    : 'bg-surface border-line text-ink-muted hover:text-ink hover:border-line-hard'
                            }`}
                        >
                            <span>#{num}</span>
                            {hasDefect && (
                                <span
                                    className={`w-1.5 h-1.5 rounded-full ${
                                        isSelected ? 'bg-amber-300' : 'bg-red-500'
                                    }`}
                                />
                            )}
                        </button>
                    );
                })}
            </div>

            {/* Main Visual Display */}
            <div className="relative rounded-lg overflow-hidden border border-line bg-black shadow-inner">
                {viewMode === 'split' ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-px bg-line">
                        {/* Raw side */}
                        <div className="relative bg-black flex flex-col">
                            <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded bg-black/75 text-[10px] font-mono text-white border border-white/20">
                                RAW CAMERA FEED
                            </div>
                            <img
                                src={activeItem.raw_image_url}
                                alt={`Raw section ${sectionNum}`}
                                className="w-full h-auto object-contain max-h-[260px]"
                                loading="lazy"
                            />
                        </div>
                        {/* Annotated side */}
                        <div className="relative bg-black flex flex-col">
                            <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded bg-black/75 text-[10px] font-mono text-emerald-300 border border-emerald-500/30">
                                YOLOv8-SEG PREDICTION
                            </div>
                            <img
                                src={activeItem.annotated_image_url}
                                alt={`AI annotated section ${sectionNum}`}
                                className="w-full h-auto object-contain max-h-[260px]"
                                loading="lazy"
                            />
                        </div>
                    </div>
                ) : (
                    <div className="relative flex items-center justify-center bg-black">
                        {/* Badge Overlay */}
                        <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
                            <span className="px-2 py-0.5 rounded bg-black/75 text-[10px] font-mono text-white border border-white/20">
                                {viewMode === 'annotated' ? 'YOLOv8-SEG AI PREDICTION' : 'RAW LIVE FEED'}
                            </span>
                            {activeItem.defect_detected ? (
                                <span className="px-2 py-0.5 rounded bg-red-600/90 text-white font-mono text-[10px] font-bold shadow">
                                    {activeItem.detections_count} DEFECTS FOUND
                                </span>
                            ) : (
                                <span className="px-2 py-0.5 rounded bg-emerald-600/90 text-white font-mono text-[10px] font-bold shadow">
                                    NOMINAL
                                </span>
                            )}
                        </div>

                        {/* Image */}
                        <img
                            src={viewMode === 'annotated' ? activeItem.annotated_image_url : activeItem.raw_image_url}
                            alt={`Section ${sectionNum} - ${viewMode}`}
                            className="w-full h-auto object-contain max-h-[300px]"
                            loading="lazy"
                        />
                    </div>
                )}
            </div>

            {/* Detections & AI Metrics Card */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <div className="inner-card p-2.5">
                    <div className="text-[10px] text-ink-muted">Detected Regions</div>
                    <div className="text-xl font-bold font-mono text-critical mt-0.5">
                        {activeItem.detections_count}
                    </div>
                </div>

                <div className="inner-card p-2.5">
                    <div className="text-[10px] text-ink-muted">Peak Confidence</div>
                    <div className="text-xl font-bold font-mono text-ink mt-0.5">
                        {peakDetection ? `${(peakDetection.confidence * 100).toFixed(1)}%` : '0%'}
                    </div>
                </div>

                <div className="inner-card p-2.5">
                    <div className="text-[10px] text-ink-muted">Primary Defect</div>
                    <div className="text-sm font-bold font-mono text-amber-600 mt-1 truncate uppercase">
                        {peakDetection ? peakDetection.class_name.replace('_', ' ') : 'None'}
                    </div>
                </div>

                <div className="inner-card p-2.5">
                    <div className="text-[10px] text-ink-muted">Model Architecture</div>
                    <div className="text-xs font-semibold text-ink-soft mt-1 truncate">
                        YOLOv8n-Seg (40-img)
                    </div>
                </div>
            </div>

            {/* Detailed Detections List */}
            {activeItem.detections && activeItem.detections.length > 0 && !compact && (
                <div className="space-y-1.5 mt-1">
                    <div className="flex items-center justify-between text-xs font-semibold text-ink px-1">
                        <span>Defect Segmentation Breakdown</span>
                        <span className="text-[11px] text-ink-muted">{activeItem.detections.length} total</span>
                    </div>

                    <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1">
                        {activeItem.detections.map((det) => {
                            const colors = CLASS_COLORS[det.class_name] || CLASS_COLORS.none;
                            const sevColors = SEVERITY_COLORS[det.severity] || SEVERITY_COLORS.LOW;

                            return (
                                <div
                                    key={det.detection_id}
                                    className="p-2 rounded-lg bg-surface border border-line flex items-center justify-between text-xs hover:border-line-hard transition-colors"
                                >
                                    <div className="flex items-center gap-2">
                                        <span
                                            className="px-1.5 py-0.5 rounded text-[10px] font-bold font-mono uppercase border"
                                            style={{
                                                backgroundColor: colors.bg,
                                                color: colors.text,
                                                borderColor: colors.border,
                                            }}
                                        >
                                            {det.class_name.replace('_', ' ')}
                                        </span>
                                        <span className="text-[11px] text-ink-muted font-mono">
                                            #{det.detection_id}
                                        </span>
                                    </div>

                                    <div className="flex items-center gap-2">
                                        <span
                                            className="px-1.5 py-0.5 rounded text-[9px] font-semibold font-mono"
                                            style={{ backgroundColor: sevColors.bg, color: sevColors.text }}
                                        >
                                            {det.severity}
                                        </span>
                                        <div className="w-16 bg-surface-sunken h-2 rounded-full overflow-hidden border border-line">
                                            <div
                                                className="bg-primary h-full rounded-full"
                                                style={{ width: `${det.confidence * 100}%` }}
                                            />
                                        </div>
                                        <span className="font-mono font-semibold text-xs text-ink w-11 text-right">
                                            {(det.confidence * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Model Summary Footnote */}
            {summary && (
                <div className="p-2.5 rounded-lg bg-surface-sunken border border-line text-[11px] text-ink-muted flex items-center justify-between">
                    <span>
                        System Fleet: <strong>{summary.total_images_processed}</strong> sections scanned ·{' '}
                        <strong>{summary.total_detections}</strong> total defects found
                    </span>
                    <span className="font-mono text-emerald-600 font-semibold">
                        {summary.detection_rate_pct}% Fleet Coverage
                    </span>
                </div>
            )}
        </div>
    );
}

