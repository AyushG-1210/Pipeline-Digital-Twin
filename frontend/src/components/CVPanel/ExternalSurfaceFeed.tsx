// SVG viewBox for the pipe graphic
const SVG_W = 480;
const SVG_H = 200;

/**
 * MOCK ILLUSTRATION — static pipe graphic used to preview the planned external surface inspection UI.
 * It is NOT a camera feed and NOT model output: no YOLO inference runs, nothing is animated or "live",
 * and the "defect" markers are placeholder art keyed off the segment ID.
 */
export default function ExternalSurfaceFeed({ segmentId }: { segmentId: string }) {
    const isCorroded =
        segmentId.includes('3') ||
        segmentId.includes('7') ||
        segmentId.includes('1') ||
        segmentId.includes('5');

    return (
        <div className="relative w-full h-full">
            <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="pipeGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#64748b" />
                        <stop offset="30%" stopColor="#475569" />
                        <stop offset="70%" stopColor="#334155" />
                        <stop offset="100%" stopColor="#1e293b" />
                    </linearGradient>
                    <linearGradient id="flangeGrad" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="#475569" />
                        <stop offset="100%" stopColor="#334155" />
                    </linearGradient>
                    <radialGradient id="corrGrad" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="#92400e" stopOpacity="0.85" />
                        <stop offset="100%" stopColor="#92400e" stopOpacity="0" />
                    </radialGradient>
                    <radialGradient id="rustGrad" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="#78350f" stopOpacity="0.7" />
                        <stop offset="100%" stopColor="#92400e" stopOpacity="0" />
                    </radialGradient>
                    <linearGradient id="laserGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#c8ccd2" stopOpacity="0" />
                        <stop offset="30%" stopColor="#c8ccd2" stopOpacity="0.9" />
                        <stop offset="70%" stopColor="#e2e8f0" stopOpacity="0.9" />
                        <stop offset="100%" stopColor="#c8ccd2" stopOpacity="0" />
                    </linearGradient>
                    <clipPath id="pipeClip">
                        <rect x="30" y="38" width="420" height="124" rx="2" />
                    </clipPath>
                    <filter id="yolo-glow-pipe">
                        <feGaussianBlur stdDeviation="2" result="blur" />
                        <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                    </filter>
                </defs>

                {/* Background */}
                <rect width={SVG_W} height={SVG_H} fill="#050505" />

                {/* Reference grid */}
                <g clipPath="url(#pipeClip)" stroke="#ffffff" strokeOpacity="0.06" strokeWidth="0.5">
                    {[80, 160, 240, 320, 400].map(x => <line key={`vg-${x}`} x1={x} y1="38" x2={x} y2="162" />)}
                    {[75, 100, 125, 138].map(y => <line key={`hg-${y}`} x1="30" y1={y} x2="450" y2={y} />)}
                </g>

                {/* Main Pipe Body */}
                <rect x="30" y="38" width="420" height="124" rx="2" fill="url(#pipeGrad)" />
                <rect x="30" y="38" width="420" height="14" rx="2" fill="white" fillOpacity="0.12" />
                <rect x="30" y="148" width="420" height="14" rx="2" fill="black" fillOpacity="0.4" />
                <line x1="30" y1="100" x2="450" y2="100" stroke="#475569" strokeWidth="2.5" strokeDasharray="12 5" />

                {/* Left Flange */}
                <rect x="8" y="30" width="28" height="140" rx="3" fill="url(#flangeGrad)" />
                <rect x="8" y="30" width="4" height="140" rx="2" fill="white" fillOpacity="0.2" />
                {[50, 80, 120, 150].map(y => <circle key={`lb-${y}`} cx="22" cy={y} r="4" fill="#0f172a" stroke="#64748b" strokeWidth="1" />)}

                {/* Right Flange */}
                <rect x="444" y="30" width="28" height="140" rx="3" fill="url(#flangeGrad)" />
                <rect x="468" y="30" width="4" height="140" rx="2" fill="white" fillOpacity="0.2" />
                {[50, 80, 120, 150].map(y => <circle key={`rb-${y}`} cx="458" cy={y} r="4" fill="#0f172a" stroke="#64748b" strokeWidth="1" />)}

                {/* Bore holes */}
                <ellipse cx="30" cy="100" rx="4" ry="62" fill="#0f172a" stroke="#334155" strokeWidth="1" />
                <ellipse cx="450" cy="100" rx="4" ry="62" fill="#0f172a" stroke="#334155" strokeWidth="1" />

                {/* === Placeholder corrosion art (mock — not a detection) === */}
                {isCorroded && (
                    <>
                        <ellipse cx="190" cy="55" rx="48" ry="22" fill="url(#corrGrad)" />
                        <rect x="140" y="38" width="100" height="44" fill="none" stroke="#ef4444" strokeWidth="1.2" strokeDasharray="5 3" />
                        <line x1="140" y1="38" x2="150" y2="38" stroke="#ef4444" strokeWidth="1.5" />
                        <line x1="140" y1="38" x2="140" y2="48" stroke="#ef4444" strokeWidth="1.5" />
                        <line x1="240" y1="38" x2="230" y2="38" stroke="#ef4444" strokeWidth="1.5" />
                        <line x1="240" y1="38" x2="240" y2="48" stroke="#ef4444" strokeWidth="1.5" />
                        <line x1="140" y1="82" x2="150" y2="82" stroke="#ef4444" strokeWidth="1.5" />
                        <line x1="140" y1="82" x2="140" y2="72" stroke="#ef4444" strokeWidth="1.5" />
                        <line x1="240" y1="82" x2="230" y2="82" stroke="#ef4444" strokeWidth="1.5" />
                        <line x1="240" y1="82" x2="240" y2="72" stroke="#ef4444" strokeWidth="1.5" />
                        <text x="143" y="35" fill="#ef4444" fontSize="7" fontFamily="monospace" fontWeight="bold">
                            [MOCK] PLACEHOLDER MARKER
                        </text>
                        <ellipse cx="330" cy="148" rx="36" ry="14" fill="url(#rustGrad)" />
                        <rect x="292" y="135" width="76" height="27" fill="none" stroke="#f59e0b" strokeWidth="0.8" strokeDasharray="4 3" strokeOpacity="0.8" />
                        <text x="296" y="131" fill="#f59e0b" fontSize="6.5" fontFamily="monospace">[MOCK] PLACEHOLDER MARKER</text>
                    </>
                )}

            </svg>

            {/* Diagonal watermark */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-20 overflow-hidden">
                <span
                    className="font-extrabold tracking-widest whitespace-nowrap select-none"
                    style={{
                        fontSize: '34px',
                        color: '#ef4444',
                        opacity: 0.4,
                        transform: 'rotate(-12deg)',
                        textShadow: '0 0 4px #000',
                    }}
                >
                    MOCK DESIGN · NOT LIVE
                </span>
            </div>

            {/* HUD overlays */}
            <div className="absolute top-2 left-3 flex items-center gap-2 bg-black/80 border border-industrial-700/80 px-2.5 py-1 rounded text-[10px] pointer-events-none z-10">
                <div className="w-2 h-2 bg-amber-500 rounded-full" />
                <span className="text-amber-300 font-semibold tracking-wider">MOCK ILLUSTRATION</span>
                <span className="text-zinc-500 font-mono">| NO CAMERA · NO MODEL</span>
            </div>

            <div className="absolute bottom-2 left-3 bg-black/80 border border-industrial-700/80 px-2 py-0.5 rounded text-[10px] font-mono text-zinc-400 pointer-events-none z-10">
                SEG: <span className="text-white font-bold">{segmentId}</span>
                <span className="text-zinc-600 ml-1">· OUTER WALL</span>
            </div>

            <div className="absolute bottom-2 right-3 bg-black/80 border border-industrial-700/80 px-2 py-0.5 rounded text-[10px] font-mono pointer-events-none z-10">
                <span className="text-zinc-500">SAMPLE ART — NOT A RESULT</span>
            </div>
        </div>
    );
}
