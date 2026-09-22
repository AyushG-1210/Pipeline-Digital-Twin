import { useMemo, useState } from 'react';
import { usePipelineStore } from '../../store/usePipelineStore';
import { getIntegrityColor } from '../../utils/colors';

type Filter = 'all' | 'healthy' | 'warning' | 'critical';

function tierOf(integrity: number): Exclude<Filter, 'all'> {
    if (integrity < 0.3) return 'critical';
    if (integrity < 0.6) return 'warning';
    return 'healthy';
}

/**
 * Segment browser — the primary way to find and open a segment.
 * Clicking the 3D twin still works, but nothing here requires it:
 * every segment is listed, searchable, and filterable by benchmark tier.
 */
export default function Sidebar() {
    const [collapsed, setCollapsed] = useState(false);
    const [query, setQuery] = useState('');
    const [filter, setFilter] = useState<Filter>('all');

    const segments = usePipelineStore(state => state.segments);
    const selectedSegmentId = usePipelineStore(state => state.selectedSegmentId);
    const selectSegment = usePipelineStore(state => state.selectSegment);

    const segmentArray = useMemo(
        () => Array.from(segments.values()).sort((a, b) => a.segment_id.localeCompare(b.segment_id)),
        [segments]
    );

    const counts = useMemo(() => {
        const c = { healthy: 0, warning: 0, critical: 0 };
        segmentArray.forEach(s => c[tierOf(s.integrity)]++);
        return c;
    }, [segmentArray]);

    const filtered = segmentArray.filter(s => {
        if (filter !== 'all' && tierOf(s.integrity) !== filter) return false;
        if (query && !s.segment_id.toLowerCase().includes(query.toLowerCase())) return false;
        return true;
    });

    if (collapsed) {
        return (
            <div className="bg-surface border-r border-line p-2 flex flex-col items-center">
                <button
                    onClick={() => setCollapsed(false)}
                    className="p-2 hover:bg-surface-sunken rounded-lg transition-colors"
                    title="Expand segment list"
                >
                    <svg className="w-4 h-4 text-ink-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                    </svg>
                </button>
            </div>
        );
    }

    const filterPills: { key: Filter; label: string; color: string; count: number }[] = [
        { key: 'all', label: 'All', color: 'rgb(var(--color-primary))', count: segmentArray.length },
        { key: 'healthy', label: 'Healthy', color: '#4C9A78', count: counts.healthy },
        { key: 'warning', label: 'Warning', color: '#D98E2B', count: counts.warning },
        { key: 'critical', label: 'Critical', color: '#D6473C', count: counts.critical },
    ];

    return (
        <div className="bg-surface border-r border-line w-72 flex flex-col">
            <div className="p-3 border-b border-line flex items-center justify-between">
                <h2 className="text-sm font-semibold text-ink">Segments</h2>
                <button
                    onClick={() => setCollapsed(true)}
                    className="p-1 hover:bg-surface-sunken rounded transition-colors"
                    title="Collapse"
                >
                    <svg className="w-4 h-4 text-ink-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                    </svg>
                </button>
            </div>

            {/* Search */}
            <div className="p-3 pb-2">
                <input
                    type="text"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    placeholder="Search segment ID…"
                    className="w-full text-xs px-3 py-1.5 rounded-md bg-canvas border border-line text-ink placeholder:text-ink-muted outline-none focus:border-accent transition-colors"
                />
            </div>

            {/* Filter pills */}
            <div className="px-3 pb-3 flex flex-wrap gap-1.5">
                {filterPills.map(p => (
                    <button
                        key={p.key}
                        onClick={() => setFilter(p.key)}
                        className="flex items-center gap-1.5 px-2 py-1 rounded-full text-[11px] font-medium transition-colors border"
                        style={
                            filter === p.key
                                ? { background: 'rgb(var(--color-primary))', borderColor: 'rgb(var(--color-primary))', color: '#fff' }
                                : { background: 'transparent', borderColor: 'rgb(var(--color-line))', color: 'rgb(var(--color-ink-soft))' }
                        }
                    >
                        <span className="w-1.5 h-1.5 rounded-full" style={{ background: filter === p.key ? '#fff' : p.color }} />
                        {p.label}
                        <span className="opacity-70">{p.count}</span>
                    </button>
                ))}
            </div>

            {/* List */}
            <div className="flex-1 overflow-auto px-2 pb-3 space-y-1">
                {filtered.length === 0 && (
                    <div className="text-xs text-ink-muted text-center py-8">No segments match.</div>
                )}
                {filtered.map(seg => {
                    const isSelected = seg.segment_id === selectedSegmentId;
                    const color = getIntegrityColor(seg.integrity);
                    return (
                        <button
                            key={seg.segment_id}
                            onClick={() => selectSegment(seg.segment_id)}
                            className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-left transition-colors"
                            style={{
                                background: isSelected ? '#EAE8E1' : 'transparent',
                                borderLeft: isSelected ? '2px solid #FE4C40' : '2px solid transparent',
                            }}
                        >
                            <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
                            <span className="flex-1 min-w-0">
                                <span className="block text-xs font-mono font-semibold text-ink truncate">
                                    {seg.segment_id}
                                </span>
                                <span className="block text-[10px] text-ink-muted">
                                    {(seg.integrity * 100).toFixed(0)}% integrity
                                    {seg.pinn ? ` · ${seg.pinn.remaining_useful_life_days}d RUL` : ''}
                                </span>
                            </span>
                            {seg.cv?.is_yolo_result && (
                                <span className="w-1.5 h-1.5 rounded-full bg-secondary shrink-0" title="Real YOLO result" />
                            )}
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
