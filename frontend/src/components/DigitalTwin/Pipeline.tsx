import { usePipelineStore } from '../../store/usePipelineStore';
import PipelineSegment from './PipelineSegment';

/**
 * Main pipeline 3D object composed of multiple segments
 * Each segment is positioned end-to-end along the X-axis
 */
export default function Pipeline() {
    const segments = usePipelineStore(state => state.segments);

    // Convert Map to array for rendering
    const segmentArray = Array.from(segments.values());

    return (
        <group>
            {segmentArray.map((segment) => (
                <PipelineSegment
                    key={segment.segment_id}
                    segmentId={segment.segment_id}
                    position={segment.position}
                    direction={segment.direction}
                    integrity={segment.integrity}
                />
            ))}
        </group>
    );
}
