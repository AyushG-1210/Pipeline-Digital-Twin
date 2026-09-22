import { usePipelineStore } from '../../store/usePipelineStore';
import SegmentRadar from './SegmentRadar';
import EnvironmentProfileChart from './EnvironmentProfileChart';
import WaterfallChart from './WaterfallChart';
import ExplanationText from './ExplanationText';
import EnsembleUncertainty from './EnsembleUncertainty';

/**
 * XAI (Explainable AI) Panel — real, per-segment model introspection.
 *
 * "Model confidence" is gone: a deterministic PINN forward pass has no
 * softmax to read a confidence from. In its place, ensemble uncertainty
 * across the 5 independently-trained checkpoints (seeds 42-46).
 *
 * The feature-contribution chart is gone too — it used to be a random
 * per-segment list of features the model has no such inputs for at all
 * ("Humidity Exposure", "Temperature Cycles"), sitting next to one static
 * global attribution number reused for every segment. In its place: a
 * fingerprint radar of 5 quantities that genuinely vary per segment, the
 * real environmental scenario behind the prediction, and an exact
 * per-segment waterfall of the model's own internal computation.
 */
export default function XAIPanel() {
    const selectedSegmentId = usePipelineStore(state => state.selectedSegmentId);
    const segments          = usePipelineStore(state => state.segments);

    const selectedSegment = selectedSegmentId ? segments.get(selectedSegmentId) : null;
    const xaiData = selectedSegment?.xai;

    return (
        <div className="h-full flex flex-col">
            <div className="px-4 py-3 border-b border-line bg-surface shrink-0">
                <span className="text-sm font-semibold text-ink">Explainable AI Diagnostics</span>
            </div>

            <div className="flex-1 p-4 overflow-auto">
                {!selectedSegmentId || !xaiData ? (
                    <div className="h-full flex items-center justify-center text-ink-muted">
                        <div className="text-center">
                            <svg className="w-16 h-16 mx-auto mb-4 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                            </svg>
                            <p>Select a segment to view explanations</p>
                        </div>
                    </div>
                ) : (
                    <div className="space-y-4">
                        {/* Segment Fingerprint — the headline "why this one's different" view */}
                        <div className="inner-card p-4">
                            <h3 className="text-sm font-semibold text-ink mb-1">Segment Fingerprint</h3>
                            <p className="text-xs text-ink-muted mb-1">
                                5 independent measurements, normalised against all 750 scenarios in the benchmark set.
                                Further out = healthier / more trustworthy.
                            </p>
                            <SegmentRadar fingerprint={xaiData.fingerprint} />
                        </div>

                        {/* Real environmental scenario */}
                        <div className="inner-card p-4">
                            <h3 className="text-sm font-semibold text-ink mb-1">Environmental Scenario</h3>
                            <p className="text-xs text-ink-muted mb-3">
                                The real soil resistivity and bulk O₂ curves this segment's prediction was computed from —
                                the literal cause, not a derived explanation. A different curve for every segment.
                            </p>
                            <EnvironmentProfileChart environment={xaiData.environment} />
                        </div>

                        {/* Ensemble Uncertainty (replaces "Model Confidence") */}
                        <div className="inner-card p-4">
                            <h3 className="text-sm font-semibold text-ink mb-1">Ensemble Uncertainty</h3>
                            <p className="text-xs text-ink-muted mb-3">
                                Agreement across 5 independently-trained checkpoints on this segment
                            </p>
                            <EnsembleUncertainty ensemble={xaiData.ensemble} />
                        </div>

                        {/* Waterfall Decomposition */}
                        <div className="inner-card p-4">
                            <div className="flex items-center justify-between mb-1">
                                <h3 className="text-sm font-semibold text-ink">Wall Concentration Breakdown</h3>
                                <span className="text-xs font-mono text-ink-muted">sf_wbc1_s42</span>
                            </div>
                            <p className="text-xs text-ink-muted mb-3">
                                Exact decomposition of the predicted wall value for this segment — not an approximation.
                                The meta bar (pipe geometry + defect score) is the same across every segment in this set:
                                there's no per-segment wall thickness in this benchmark, only the environment varies.
                            </p>
                            <WaterfallChart waterfall={xaiData.waterfall} />
                        </div>

                        {/* Natural Language Explanation */}
                        <div className="inner-card p-4">
                            <h3 className="text-sm font-semibold text-ink mb-3">Engineering Explanation</h3>
                            <ExplanationText waterfall={xaiData.waterfall} ensemble={xaiData.ensemble} />
                        </div>

                        {/* Trust & Transparency Note */}
                        <div className="inner-card p-4">
                            <div className="flex items-start gap-3">
                                <svg className="w-5 h-5 text-secondary flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                        d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                                </svg>
                                <div>
                                    <div className="text-sm font-semibold text-ink mb-1">Explainability &amp; Trust</div>
                                    <div className="text-xs text-ink-muted">
                                        This system never operates as a black box. Every prediction is backed by physics-based models
                                        and transparent feature contributions, allowing engineers to verify and validate the AI&apos;s reasoning.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
