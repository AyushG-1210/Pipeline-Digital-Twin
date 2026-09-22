/**
 * Mock video feed component
 * In production, this would stream from actual inspection cameras
 */
export default function VideoFeed({ segmentId }: { segmentId: string }) {
    // Mock: Display a static frame representing pipeline inspection
    // In production, this would be a live video stream or recorded footage

    return (
        <div className="w-full h-full bg-gradient-to-br from-zinc-900 via-industrial-950 to-black flex items-center justify-center relative">
            {/* Mock pipeline cross-section visualization */}
            <svg className="w-3/4 h-3/4" viewBox="0 0 200 200">
                {/* Outer pipe wall */}
                <circle cx="100" cy="100" r="80" fill="#38383c" stroke="#505050" strokeWidth="2" />

                {/* Inner pipe wall */}
                <circle cx="100" cy="100" r="65" fill="#0f0f10" stroke="#2a2a2d" strokeWidth="2" />

                {/* Mock corrosion regions (varies by segment) */}
                {segmentId.includes('3') || segmentId.includes('7') ? (
                    <>
                        <path d="M 100 20 Q 120 25 130 40" fill="none" stroke="#ef4444" strokeWidth="4" opacity="0.8" />
                        <path d="M 140 60 Q 145 70 145 85" fill="none" stroke="#f59e0b" strokeWidth="3" opacity="0.7" />
                    </>
                ) : null}

                {/* Scan line animation */}
                <line x1="100" y1="0" x2="100" y2="200" stroke="#10b981" strokeWidth="1" opacity="0.4">
                    <animateTransform
                        attributeName="transform"
                        type="rotate"
                        from="0 100 100"
                        to="360 100 100"
                        dur="4s"
                        repeatCount="indefinite"
                    />
                </line>
            </svg>

            {/* Live indicator */}
            <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/50 px-2 py-1 rounded">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                <span className="text-xs text-white font-medium">LIVE</span>
            </div>

            {/* Segment ID overlay */}
            <div className="absolute bottom-3 left-3 bg-black/50 px-2 py-1 rounded">
                <span className="text-xs text-white font-mono">{segmentId}</span>
            </div>
        </div>
    );
}
