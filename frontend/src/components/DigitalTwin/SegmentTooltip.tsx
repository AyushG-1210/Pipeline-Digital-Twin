import { Html } from '@react-three/drei';
import { usePipelineStore } from '../../store/usePipelineStore';
import { formatPercentage } from '../../utils/colors';

/**
 * Tooltip that appears when hovering over pipeline segments
 * Shows segment ID, integrity percentage, and RUL
 * Currently disabled - will be implemented with proper hover state management
 */
export default function SegmentTooltip() {
    const segments = usePipelineStore(state => state.segments);

    // Tooltip functionality is currently disabled
    // In production, this would show on hover with proper event handling
    const hoveredSegment: string | null = null;

    if (!hoveredSegment) return null;

    const segment = segments.get(hoveredSegment);
    if (!segment) return null;

    return (
        <Html position={segment.position}>
            <div className="bg-industrial-950 border border-industrial-700/80 rounded-lg px-3 py-2 shadow-xl pointer-events-none">
                <div className="text-xs font-semibold text-emerald-400 mb-1">{segment.segment_id}</div>
                <div className="text-xs text-zinc-300">
                    Integrity: {formatPercentage(segment.integrity)}
                </div>
                {segment.pinn && (
                    <div className="text-xs text-zinc-300">
                        RUL: {segment.pinn.remaining_useful_life_days} days
                    </div>
                )}
            </div>
        </Html>
    );
}
