import { usePipelineStore } from '../../store/usePipelineStore';
import ExternalSurfaceFeed from './ExternalSurfaceFeed';
import FeedbackControls from './FeedbackControls';

const CLASS_LABELS: Record<string, string> = {
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
 * Includes real YOLO inference trigger + human-in-the-loop feedback controls
 */
export default function CVInspectionPanel() {
    const selectedSegmentId = usePipelineStore(state => state.selectedSegmentId);
    const segments = usePipelineStore(state => state.segments);

    const selectedSegment = selectedSegmentId ? segments.get(selectedSegmentId) : null;
    const cvData = selectedSegment?.cv;

    // YOLOv8 inference is not wired up yet — the button below is a "coming soon" placeholder.
    // The original handler POSTed to /api/cv/analyze; restore it when the YOLO pipeline is deployable.

    const isYolo = cvData?.is_yolo_result === true;
    const className = cvData?.class_name ?? 'none';
    const classColor = CLASS_COLORS[className] ?? '#6b7280';
    const classLabel = CLASS_LABELS[className] ?? className;

    return (
        <div className="h-full flex flex-col">
            <div className="px-4 py-3 border-b border-line bg-surface shrink-0">
                <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-ink">External Damage &amp; Surface Inspection</span>
                    <div className="flex items-center gap-2">
                        {!isYolo && (
                            <span className="text-[10px] font-bold px-2 py-0.5 rounded border"
                                style={{ color: '#8A6100', borderColor: '#EBD9A8', background: '#FBF2D9' }}>
                                YOLO: COMING SOON
                            </span>
                        )}
                        {isYolo && (
                            <span className="text-[10px] font-bold px-2 py-0.5 rounded border"
                                style={{ color: '#3C7C61', borderColor: '#BFE0D8', background: '#E2F1EF' }}>
                                YOLO LIVE
                            </span>
                        )}
                    </div>
                </div>
            </div>

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
                        {/* External Pipe Wall SVG Inspector — kept as a dark "scan monitor" for contrast */}
                        <div style={{ height: '170px' }} className="bg-[#050505] rounded-lg overflow-hidden border border-line w-full">
                            <ExternalSurfaceFeed segmentId={selectedSegmentId} />
                        </div>

                        {/* YOLO analysis — coming soon */}
                        <button
                            disabled
                            title="YOLOv8 segmentation analysis is under development"
                            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-semibold text-sm"
                            style={{
                                background: '#8B8E9B',
                                color: '#fff',
                                border: 'none',
                                opacity: 0.7,
                                cursor: 'not-allowed',
                            }}
                        >
                            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
                            </svg>
                            YOLOv8 Segmentation Analysis — Coming Soon
                        </button>
                        <p className="text-[11px] text-ink-muted text-center -mt-1">
                            Live defect segmentation is under development. Readings below are simulated optical-scan data.
                        </p>

                        {/* Detection metadata */}
                        {cvData && (
                            <div className="grid grid-cols-2 gap-3">
                                {/* Corrosion Area */}
                                <div className="inner-card p-3">
                                    <div className="text-xs text-ink-muted mb-1">External Corrosion Area</div>
                                    <div className="text-2xl font-bold text-critical">
                                        {cvData.corrosion_surface_pct.toFixed(1)}%
                                    </div>
                                </div>

                                {/* Detection Confidence */}
                                <div className="inner-card p-3">
                                    <div className="text-xs text-ink-muted mb-1">Detection Confidence</div>
                                    <div className="text-2xl font-bold text-ink">
                                        {(cvData.confidence * 100).toFixed(0)}%
                                    </div>
                                </div>

                                {/* Class Name — only shown for real YOLO results */}
                                {isYolo && (
                                    <>
                                        <div className="inner-card p-3">
                                            <div className="text-xs text-ink-muted mb-1">Defect Class</div>
                                            <div className="flex items-center gap-2">
                                                <span className="text-lg font-bold font-mono" style={{ color: classColor }}>
                                                    {className.toUpperCase()}
                                                </span>
                                                <span className="text-xs text-ink-muted">{classLabel}</span>
                                            </div>
                                        </div>

                                        <div className="inner-card p-3">
                                            <div className="text-xs text-ink-muted mb-1">Detection Rate</div>
                                            <div className="text-2xl font-bold text-ink">
                                                {((cvData.detection_rate ?? 0) * 100).toFixed(0)}%
                                            </div>
                                            <div className="text-[10px] text-ink-muted mt-0.5">
                                                {cvData.frames_with_detections ?? 0}/{cvData.total_frames ?? 0} frames
                                            </div>
                                        </div>
                                    </>
                                )}

                                {/* Timestamp */}
                                <div className="inner-card col-span-2 p-3">
                                    <div className="text-xs text-ink-muted mb-1">
                                        {isYolo ? 'YOLO Inference Timestamp' : 'Optical Scan Timestamp'}
                                    </div>
                                    <div className="text-sm font-mono text-ink-soft">
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
