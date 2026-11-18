import { useState, useEffect } from 'react';
import type { LiveFeedItem, LiveFeedSummary } from '../../types';
import LiveFeedViewer from './LiveFeedViewer';

interface LiveFeedGalleryModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSelectSegment?: (segmentId: string) => void;
}

export default function LiveFeedGalleryModal({ isOpen, onClose }: LiveFeedGalleryModalProps) {
    const [items, setItems] = useState<LiveFeedItem[]>([]);
    const [summary, setSummary] = useState<LiveFeedSummary | null>(null);
    const [selectedImage, setSelectedImage] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<'gallery' | 'inspector'>('gallery');

    useEffect(() => {
        if (!isOpen) return;

        async function load() {
            try {
                let data = null;
                try {
                    const res = await fetch('http://localhost:3001/api/cv/live-feed');
                    if (res.ok) data = await res.json();
                } catch {
                    // Fall back
                }

                if (!data) {
                    const [resDet, resSum] = await Promise.all([
                        fetch('/data/detections.json'),
                        fetch('/data/summary.json'),
                    ]);
                    const detJson = await resDet.json();
                    const sumJson = resSum.ok ? await resSum.json() : null;
                    data = {
                        items: detJson.map((d: LiveFeedItem) => ({
                            ...d,
                            raw_image_url: `/data/live_feed/${d.image_file}`,
                            annotated_image_url: `/data/annotated_feed/annotated_${d.image_file}`,
                        })),
                        summary: sumJson,
                    };
                }

                if (data?.items) {
                    setItems(data.items);
                    setSummary(data.summary || null);
                    if (!selectedImage && data.items.length > 0) {
                        setSelectedImage(data.items[0].image_file);
                    }
                }
            } catch (e) {
                console.error('Failed to load live feed gallery:', e);
            }
        }

        load();
    }, [isOpen, selectedImage]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/60 backdrop-blur-sm animate-fade-in">
            <div className="bg-surface rounded-2xl border border-line shadow-2xl w-full max-w-6xl max-h-[90vh] flex flex-col overflow-hidden">
                {/* Header */}
                <div className="px-6 py-4 border-b border-line flex items-center justify-between shrink-0 bg-surface">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-primary/10 text-primary">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                        </div>
                        <div>
                            <div className="flex items-center gap-2">
                                <h2 className="text-base font-bold text-ink tracking-tight">
                                    Pipeline Live Optical Feed &amp; YOLOv8 Defect Predictions
                                </h2>
                                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
                                    20 SECTIONS ACTIVE
                                </span>
                            </div>
                            <p className="text-xs text-ink-muted mt-0.5">
                                Real-time camera feeds processed through fine-tuned YOLOv8n-Seg (40-image training dataset)
                            </p>
                        </div>
                    </div>

                    <div className="flex items-center gap-2">
                        {/* Tab toggle */}
                        <div className="flex rounded-lg border border-line bg-surface-sunken p-0.5 text-xs">
                            <button
                                type="button"
                                onClick={() => setActiveTab('gallery')}
                                className={`px-3 py-1 rounded-md text-xs font-semibold transition-colors ${
                                    activeTab === 'gallery' ? 'bg-primary text-white shadow-sm' : 'text-ink-muted hover:text-ink'
                                }`}
                            >
                                All 20 Sections
                            </button>
                            <button
                                type="button"
                                onClick={() => setActiveTab('inspector')}
                                className={`px-3 py-1 rounded-md text-xs font-semibold transition-colors ${
                                    activeTab === 'inspector' ? 'bg-primary text-white shadow-sm' : 'text-ink-muted hover:text-ink'
                                }`}
                            >
                                Detailed Inspector
                            </button>
                        </div>

                        <button
                            type="button"
                            onClick={onClose}
                            className="p-1.5 rounded-lg text-ink-muted hover:text-ink hover:bg-surface-sunken transition-colors ml-2"
                            title="Close"
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                </div>

                {/* Subheader metrics banner */}
                {summary && (
                    <div className="px-6 py-2.5 bg-surface-sunken border-b border-line flex flex-wrap items-center justify-between text-xs text-ink-muted shrink-0 gap-4">
                        <div className="flex items-center gap-6">
                            <div>
                                <span className="text-ink font-bold font-mono text-sm">{summary.total_images_processed}</span> Sections Inspected
                            </div>
                            <div className="w-px h-4 bg-line" />
                            <div>
                                <span className="text-critical font-bold font-mono text-sm">{summary.total_detections}</span> Total Defect Instances
                            </div>
                            <div className="w-px h-4 bg-line" />
                            <div>
                                <span className="text-amber-600 font-bold font-mono text-sm">{(summary.highest_confidence_score * 100).toFixed(1)}%</span> Peak AI Confidence
                            </div>
                            <div className="w-px h-4 bg-line" />
                            <div>
                                <span className="text-emerald-600 font-bold font-mono text-sm">{summary.detection_rate_pct}%</span> Pipeline Anomaly Coverage
                            </div>
                        </div>
                        <div className="text-[11px] font-mono text-ink-muted">
                            Model: YOLOv8n-Seg · Device: CPU-Optimized
                        </div>
                    </div>
                )}

                {/* Content Body */}
                <div className="flex-1 overflow-y-auto p-6 bg-canvas">
                    {activeTab === 'gallery' ? (
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                            {items.map((item) => {
                                const secNum = item.image_file.replace(/[^0-9]/g, '');
                                const peakDet = item.detections && item.detections.length > 0
                                    ? [...item.detections].sort((a, b) => b.confidence - a.confidence)[0]
                                    : null;

                                return (
                                    <div
                                        key={item.image_file}
                                        onClick={() => {
                                            setSelectedImage(item.image_file);
                                            setActiveTab('inspector');
                                        }}
                                        className="group rounded-xl border border-line bg-surface overflow-hidden shadow-sm hover:shadow-md hover:border-primary/50 transition-all cursor-pointer flex flex-col"
                                    >
                                        {/* Image Frame with overlay */}
                                        <div className="relative aspect-[16/10] bg-black overflow-hidden">
                                            <img
                                                src={item.annotated_image_url}
                                                alt={`Section ${secNum}`}
                                                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                                loading="lazy"
                                            />
                                            {/* Tag */}
                                            <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/80 backdrop-blur-sm text-[10px] font-mono text-white border border-white/20">
                                                CAM #{secNum}
                                            </div>

                                            {item.defect_detected ? (
                                                <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-red-600/90 text-white font-mono text-[10px] font-bold">
                                                    {item.detections_count} DEFECTS
                                                </div>
                                            ) : (
                                                <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-emerald-600/90 text-white font-mono text-[10px] font-bold">
                                                    NOMINAL
                                                </div>
                                            )}
                                        </div>

                                        {/* Meta footer */}
                                        <div className="p-3 flex items-center justify-between text-xs">
                                            <div>
                                                <div className="font-semibold text-ink font-mono">
                                                    Section {secNum}
                                                </div>
                                                <div className="text-[11px] text-ink-muted capitalize">
                                                    {peakDet ? `${peakDet.class_name.replace('_', ' ')}` : 'Clean Pipe'}
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <div className="font-mono font-bold text-xs text-ink">
                                                    {peakDet ? `${(peakDet.confidence * 100).toFixed(0)}%` : '—'}
                                                </div>
                                                <div className="text-[10px] text-ink-muted">confidence</div>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    ) : (
                        <div className="max-w-4xl mx-auto">
                            <LiveFeedViewer initialImageFile={selectedImage || undefined} />
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
