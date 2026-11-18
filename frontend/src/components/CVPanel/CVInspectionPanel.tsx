import { useState } from 'react';
import { usePipelineStore } from '../../store/usePipelineStore';
import ExternalSurfaceFeed from './ExternalSurfaceFeed';
import LiveFeedViewer from './LiveFeedViewer';
import FeedbackControls from './FeedbackControls';

const CLASS_LABELS: Record<string, string> = {
    rust: 'Surface Rust / Oxidation',
    corrosion_zone: 'Active Corrosion Zone',
    BX: 'Box Defect',
    CJ: 'Corrosion Joint',
    CK: 'Crack',
    OBB: 'Object/Blockage',
    PL: 'Peeling',
    SG: 'Surface Gouging',
    ZW: 'Zone Wear',
    none: 'No Defect',
};

const CLASS_COLORS: Record<string, string> = {
    rust: '#ef4444',
    corrosion_zone: '#0d9488',
    BX: '#f59e0b',
    CJ: '#ef4444',
    CK: '#dc2626',
    OBB: '#8b5cf6',
    PL: '#f97316',
    SG: '#ec4899',
    ZW: '#06b6d4',
    none: '#6b7280',
};

/**
 * CV Inspection Panel - External pipeline surface corrosion detection via YOLOv8-Seg
 * Showcases live camera feeds + fine-tuned YOLO predictions alongside cross-section scanner.
 */
export default function CVInspectionPanel() {
    const selectedSegmentId = usePipelineStore(state => state.selectedSegmentId);
    const segments = usePipelineStore(state => state.segments);

    const selectedSegment = selectedSegmentId ? segments.get(selectedSegmentId) : null;
    const cvData = selectedSegment?.cv;

    // View selector: 'camera' | 'scanner'
    const [viewTab, setViewTab] = useState<'camera' | 'scanner'>('camera');

    // Compute corresponding live feed image file for this segment
    let segmentImageFile = cvData?.image_file;
    if (!segmentImageFile && selectedSegmentId) {
        const segNum = parseInt(selectedSegmentId.replace(/[^0-9]/g, '') || '1', 10);
        const imgNum = ((segNum - 1) % 20) + 1;
        segmentImageFile = `pipeline_section_${String(imgNum).padStart(2, '0')}.png`;
    }

    const isYolo = cvData?.is_yolo_result === true;
    const className = cvData?.class_name ?? 'none';
    const classColor = CLASS_COLORS[className] ?? '#6b7280';
    const classLabel = CLASS_LABELS[className] ?? className;

    return (
        <div className="h-full flex flex-col">
            {/* Panel Header */}
            <div className="px-4 py-3 border-b border-line bg-surface shrink-0">
                <div className="flex items-center justify-between">
                    <div>
                        <div className="flex items-center gap-2">
                            <span className="text-sm font-semibold text-ink">External Surface Inspection</span>
                            {isYolo && (
                                <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-emerald-50 text-emerald-700 border-emerald-200">
                                    YOLOv8-SEG LIVE
                                </span>
                            )}
                        </div>
                        <div className="text-[11px] text-ink-muted">
                            Optical live camera feed + AI anomaly detection
                        </div>
                    </div>

                    {/* View Switcher: Live Optical Camera vs SVG Scanner */}
                    <div className="flex rounded-lg border border-line bg-surface-sunken p-0.5 text-xs">
                        <button
                            type="button"
                            onClick={() => setViewTab('camera')}
                            className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-colors ${
                                viewTab === 'camera' ? 'bg-primary text-white shadow-sm' : 'text-ink-muted hover:text-ink'
                            }`}
                        >
                            Live Feed &amp; AI
                        </button>
                        <button
                            type="button"
                            onClick={() => setViewTab('scanner')}
                            className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-colors ${
                                viewTab === 'scanner' ? 'bg-primary text-white shadow-sm' : 'text-ink-muted hover:text-ink'
                            }`}
                        >
                            Wall Scanner
                        </button>
                    </div>
                </div>
            </div>

            {/* Panel Body */}
            <div className="flex-1 flex flex-col overflow-hidden">
                {!selectedSegmentId ? (
                    <div className="flex-1 flex items-center justify-center text-ink-muted">
                        <div className="text-center px-6">
                            <svg className="w-14 h-14 mx-auto mb-3 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                            <p className="text-sm">Select a pipeline segment to view external surface inspection</p>
                        </div>
                    </div>
                ) : (
                    <div className="p-4 overflow-auto space-y-3">
                        {viewTab === 'camera' ? (
                            /* Live Feed & Predictions View */
                            <div className="space-y-3">
                                <LiveFeedViewer initialImageFile={segmentImageFile} singleSection />
                            </div>
                        ) : (
                            /* Scanner View */
                            <div className="space-y-3">
                                <div style={{ height: '170px' }} className="bg-[#050505] rounded-lg overflow-hidden border border-line w-full">
                                    <ExternalSurfaceFeed segmentId={selectedSegmentId} />
                                </div>
                            </div>
                        )}

                        {/* Segment Detection Metadata Cards */}
                        {cvData && (
                            <div className="grid grid-cols-2 gap-2.5">
                                <div className="inner-card p-3">
                                    <div className="text-[11px] text-ink-muted mb-1">Surface Corrosion Est.</div>
                                    <div className="text-2xl font-bold font-mono text-critical">
                                        {cvData.corrosion_surface_pct.toFixed(1)}%
                                    </div>
                                    <div className="text-[10px] text-ink-muted mt-0.5">
                                        {cvData.detections_count ?? 1} defect zone(s)
                                    </div>
                                </div>

                                <div className="inner-card p-3">
                                    <div className="text-[11px] text-ink-muted mb-1">Peak AI Confidence</div>
                                    <div className="text-2xl font-bold font-mono text-ink">
                                        {(cvData.confidence * 100).toFixed(0)}%
                                    </div>
                                    <div className="text-[10px] text-ink-muted mt-0.5 font-mono">
                                        {cvData.image_file || segmentImageFile}
                                    </div>
                                </div>

                                {isYolo && (
                                    <>
                                        <div className="inner-card p-3">
                                            <div className="text-[11px] text-ink-muted mb-1">Defect Classification</div>
                                            <div className="flex items-center gap-2">
                                                <span className="text-base font-bold font-mono uppercase" style={{ color: classColor }}>
                                                    {className.replace('_', ' ')}
                                                </span>
                                            </div>
                                            <div className="text-[10px] text-ink-muted mt-0.5 truncate">{classLabel}</div>
                                        </div>

                                        <div className="inner-card p-3">
                                            <div className="text-[11px] text-ink-muted mb-1">Model Precision</div>
                                            <div className="text-2xl font-bold font-mono text-emerald-600">
                                                66.8%
                                            </div>
                                            <div className="text-[10px] text-ink-muted mt-0.5">
                                                mAP50: 0.634 (fine-tuned)
                                            </div>
                                        </div>
                                    </>
                                )}

                                <div className="inner-card col-span-2 p-3">
                                    <div className="text-[11px] text-ink-muted mb-1">
                                        Inference &amp; Frame Timestamp
                                    </div>
                                    <div className="text-xs font-mono text-ink-soft">
                                        {new Date(cvData.frame_timestamp).toLocaleString()}
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Human-in-the-loop feedback */}
                        {cvData && <FeedbackControls segmentId={selectedSegmentId} cvData={cvData} />}
                    </div>
                )}
            </div>
        </div>
    );
}
