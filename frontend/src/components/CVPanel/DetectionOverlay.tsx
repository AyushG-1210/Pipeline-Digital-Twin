import type { CVOutput } from '../../types';

interface DetectionOverlayProps {
    cvData: CVOutput;
}

// YOLO frame size (inference runs at 640x640)
const YOLO_FRAME_SIZE = 640;

/**
 * SVG overlay showing detected corrosion regions as polygon masks.
 * Handles both:
 *   - Normalized mock coordinates (polygon_mask, 0-1 range)
 *   - Real YOLO pixel coordinates (yolo_mask_px, 0-640 range)
 */
export default function DetectionOverlay({ cvData }: DetectionOverlayProps) {
    const isYolo = cvData.is_yolo_result === true;
    const detected = cvData.corrosion_detected !== false; // treat undefined (mock) as detected

    // Build polygon points string for SVG viewBox 0 0 200 200
    let polygonPoints = '';

    if (isYolo && cvData.yolo_mask_px && cvData.yolo_mask_px.length > 0) {
        // Real YOLO pixel coords: normalize from 640x640 ? 200x200 viewBox
        polygonPoints = cvData.yolo_mask_px
            .map(([x, y]) => `${(x / YOLO_FRAME_SIZE) * 200},${(y / YOLO_FRAME_SIZE) * 200}`)
            .join(' ');
    } else if (cvData.polygon_mask && cvData.polygon_mask.length > 0) {
        // Mock normalized coords: 0-1 ? 200x200 viewBox
        polygonPoints = cvData.polygon_mask
            .map(([x, y]) => `${x * 200},${y * 200}`)
            .join(' ');
    }

    const className = cvData.class_name ?? '';
    const hasClass = className && className !== 'none';
    const classColor = hasClass ? getClassColor(className) : '#ef4444';

    // Calculate bounding box center for label placement
    let labelX = 10, labelY = 20;
    if (isYolo && cvData.yolo_mask_px && cvData.yolo_mask_px.length > 0) {
        const xs = cvData.yolo_mask_px.map(p => (p[0] / YOLO_FRAME_SIZE) * 200);
        const ys = cvData.yolo_mask_px.map(p => (p[1] / YOLO_FRAME_SIZE) * 200);
        labelX = Math.min(...xs) + 2;
        labelY = Math.max(10, Math.min(...ys) - 4);
    }

    return (
        <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox="0 0 200 200"
            preserveAspectRatio="xMidYMid slice"
        >
            <defs>
                <filter id="yolo-glow">
                    <feGaussianBlur stdDeviation="1.5" result="coloredBlur" />
                    <feMerge>
                        <feMergeNode in="coloredBlur" />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
            </defs>

            {/* Detected corrosion region polygon */}
            {polygonPoints && detected && (
                <polygon
                    points={polygonPoints}
                    fill={`${classColor}40`}
                    stroke={classColor}
                    strokeWidth={isYolo ? 1.5 : 2}
                    filter={isYolo ? 'url(#yolo-glow)' : undefined}
                    className="animate-pulse-slow"
                />
            )}

            {/* Class name label (only for real YOLO results) */}
            {isYolo && hasClass && detected && (
                <g>
                    <rect
                        x={labelX - 1}
                        y={labelY - 7}
                        width={className.length * 5.5 + 4}
                        height={9}
                        rx={1}
                        fill="#000000aa"
                    />
                    <text
                        x={labelX + 1}
                        y={labelY}
                        fill={classColor}
                        fontSize="7"
                        fontFamily="monospace"
                        fontWeight="bold"
                    >
                        [{className}]
                    </text>
                </g>
            )}

            {/* Confidence indicator */}
            <text x="10" y="192" fill="#22d3ee" fontSize="7" fontFamily="monospace" fontWeight="bold">
                CONF: {(cvData.confidence * 100).toFixed(0)}%
            </text>

            {/* Corrosion percentage */}
            <text x="10" y="182" fill={classColor} fontSize="7" fontFamily="monospace" fontWeight="bold">
                {isYolo ? `YOLO: ${cvData.severity_score ? (cvData.severity_score * 100).toFixed(0) : 0}%` : `CORR: ${cvData.corrosion_surface_pct.toFixed(1)}%`}
            </text>

            {/* YOLO badge top-right */}
            {isYolo && (
                <g>
                    <rect x="149" y="3" width="48" height="11" rx="2" fill="#0c4a6ecc" />
                    <text x="152" y="11" fill="#38bdf8" fontSize="7" fontFamily="monospace" fontWeight="bold">
                        YOLOv8-SEG
                    </text>
                </g>
            )}
        </svg>
    );
}

function getClassColor(cls: string): string {
    const map: Record<string, string> = {
        BX: '#f59e0b',
        CJ: '#ef4444',
        CK: '#dc2626',
        OBB: '#8b5cf6',
        PL: '#f97316',
        SG: '#ec4899',
        ZW: '#06b6d4',
    };
    return map[cls] ?? '#ef4444';
}
