import { usePipelineStore } from '../../store/usePipelineStore';
import ExternalSurfaceFeed from './ExternalSurfaceFeed';
import FeedbackControls from './FeedbackControls';

/**
 * MOCK DESIGN ONLY — YOLOv8-Seg is NOT integrated.
 *
 * This panel is a UI mock-up of the planned external surface inspection feature.
 * No model runs, no detections are produced, and nothing shown here is a real measurement.
 * The original inference trigger (POST /api/cv/analyze) was removed; restore it from git history
 * once the YOLO pipeline is actually deployable.
 */
export default function CVInspectionPanel() {
    const selectedSegmentId = usePipelineStore(state => state.selectedSegmentId);

    return (
        <div className="h-full flex flex-col">
            <div className="px-4 py-3 border-b border-line bg-surface shrink-0">
                <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-semibold text-ink">External Damage &amp; Surface Inspection</span>
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded border whitespace-nowrap"
                        style={{ color: '#B33529', borderColor: '#EFC1BB', background: '#FBE4E1' }}>
                        MOCK DESIGN
                    </span>
                </div>
            </div>

            <div className="flex-1 flex flex-col overflow-hidden">
                {!selectedSegmentId ? (
                    <div className="flex-1 flex items-center justify-center text-ink-muted">
                        <div className="text-center px-6">
                            <svg className="w-14 h-14 mx-auto mb-3 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                            <p className="text-sm">Select a pipeline segment to preview the (mock) surface inspection layout</p>
                        </div>
                    </div>
                ) : (
                    <div className="p-4 overflow-auto space-y-3">
                        {/* Unmissable disclaimer */}
                        <div className="rounded-lg px-3 py-2.5"
                            style={{ background: '#FBE4E1', border: '2px dashed #D6473C', color: '#B33529' }}>
                            <div className="text-xs font-extrabold tracking-wider">
                                ⚠ MOCK DESIGN — NOT A WORKING MODEL
                            </div>
                            <div className="text-[11px] mt-1 leading-snug">
                                YOLOv8 is <b>not integrated and not ready</b>. No inference is running.
                                The graphic below is a static illustration and every value is a placeholder —
                                none of it is a real detection or measurement.
                            </div>
                        </div>

                        {/* Static illustration — clearly watermarked */}
                        <div style={{ height: '170px' }} className="bg-[#050505] rounded-lg overflow-hidden border border-line w-full">
                            <ExternalSurfaceFeed segmentId={selectedSegmentId} />
                        </div>

                        <button
                            disabled
                            title="YOLOv8 segmentation analysis is not implemented"
                            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-semibold text-sm"
                            style={{ background: '#8B8E9B', color: '#fff', border: 'none', opacity: 0.6, cursor: 'not-allowed' }}
                        >
                            YOLOv8 Segmentation Analysis — NOT AVAILABLE (Coming Soon)
                        </button>

                        {/* Placeholder metrics — intentionally contain no values */}
                        <div className="relative">
                            <div className="grid grid-cols-2 gap-3 opacity-50 select-none">
                                {[
                                    'External Corrosion Area',
                                    'Detection Confidence',
                                    'Defect Class',
                                    'Detection Rate',
                                ].map(label => (
                                    <div key={label} className="inner-card p-3">
                                        <div className="text-xs text-ink-muted mb-1">{label}</div>
                                        <div className="text-2xl font-bold text-ink-muted">—</div>
                                        <div className="text-[10px] font-bold mt-0.5" style={{ color: '#B33529' }}>
                                            MOCK · NO MODEL OUTPUT
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Human-in-the-loop mock — disabled */}
                        <div className="opacity-50 pointer-events-none select-none" aria-disabled="true">
                            <FeedbackControls segmentId={selectedSegmentId} />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
