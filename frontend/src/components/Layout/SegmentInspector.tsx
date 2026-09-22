import { useState } from 'react';
import { usePipelineStore } from '../../store/usePipelineStore';
import { getIntegrityColor } from '../../utils/colors';
import AnalyticsPanel from '../Analytics/AnalyticsPanel';
import CVInspectionPanel from '../CVPanel/CVInspectionPanel';
import XAIPanel from '../XAI/XAIPanel';

type Tab = 'overview' | 'surface' | 'explain';

const TABS: { key: Tab; label: string }[] = [
    { key: 'overview', label: 'Forecast' },
    { key: 'surface', label: 'Surface Scan' },
    { key: 'explain', label: 'Explainability' },
];

/**
 * Segment Inspector — slide-over drawer opened by selecting a segment
 * (from the 3D twin or the sidebar list). PINN's degradation forecast
 * is the default/hero tab since it's the model the rest of the system
 * is built around; CV and XAI are supporting evidence, one tab shift away.
 */
export default function SegmentInspector() {
    const selectedSegmentId = usePipelineStore(state => state.selectedSegmentId);
    const segments = usePipelineStore(state => state.segments);
    const selectSegment = usePipelineStore(state => state.selectSegment);

    const [tab, setTab] = useState<Tab>('overview');

    const segment = selectedSegmentId ? segments.get(selectedSegmentId) : null;
    const isOpen = !!segment;

    if (!isOpen || !segment) return null;

    const statusColor = getIntegrityColor(segment.integrity);

    return (
        <>
            {/* Backdrop */}
            <div
                className="fixed inset-0 z-40 bg-ink/10"
                onClick={() => selectSegment(null)}
            />

            {/* Drawer */}
            <div className="fixed top-0 right-0 z-50 h-full w-full sm:w-[560px] lg:w-[680px] bg-surface border-l border-line shadow-2xl flex flex-col">
                {/* Header */}
                <div className="px-5 py-4 border-b border-line flex items-center justify-between shrink-0">
                    <div className="flex items-center gap-3">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ background: statusColor }} />
                        <div>
                            <div className="text-sm font-mono font-semibold text-ink">{segment.segment_id}</div>
                            <div className="text-[11px] text-ink-muted">
                                {(segment.integrity * 100).toFixed(0)}% integrity
                                {segment.pinn && ` · ${segment.pinn.remaining_useful_life_days}d remaining`}
                            </div>
                        </div>
                    </div>
                    <button
                        onClick={() => selectSegment(null)}
                        className="p-1.5 rounded-lg text-ink-muted hover:text-ink hover:bg-surface-sunken transition-colors"
                        title="Close"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex px-5 border-b border-line shrink-0">
                    {TABS.map(t => (
                        <button
                            key={t.key}
                            onClick={() => setTab(t.key)}
                            className="relative px-3 py-2.5 text-xs font-medium transition-colors"
                            style={{ color: tab === t.key ? '#132257' : '#8B8E9B' }}
                        >
                            {t.label}
                            {tab === t.key && (
                                <span
                                    className="absolute left-0 right-0 -bottom-px h-[2px] rounded-full"
                                    style={{ background: '#FE4C40' }}
                                />
                            )}
                        </button>
                    ))}
                </div>

                {/* Tab content */}
                <div className="flex-1 min-h-0 bg-canvas">
                    {tab === 'overview' && <AnalyticsPanel />}
                    {tab === 'surface' && <CVInspectionPanel />}
                    {tab === 'explain' && <XAIPanel />}
                </div>
            </div>
        </>
    );
}
