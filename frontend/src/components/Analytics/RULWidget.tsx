import { getUrgencyLevel, getUrgencyColor } from '../../utils/colors';

interface RULWidgetProps {
    rul: number; // Remaining Useful Life in days
    isPhysicsValidated?: boolean; // true when data comes from real PINN inference
}

/**
 * Remaining Useful Life widget with color-coded urgency
 */
export default function RULWidget({ rul, isPhysicsValidated }: RULWidgetProps) {
    const urgency = getUrgencyLevel(rul);
    const color = getUrgencyColor(urgency);

    const urgencyLabels = {
        low: 'Normal',
        medium: 'Monitor',
        high: 'Attention Required',
        critical: 'CRITICAL'
    };

    return (
        <div
            className="inner-card p-6 relative overflow-hidden"
            style={{ borderLeft: `3px solid ${color}` }}
        >
            <div className="relative z-10">
                <div className="text-sm text-ink-muted mb-2">Remaining Useful Life</div>

                <div className="flex items-baseline gap-3 mb-3">
                    <div
                        className="text-5xl font-bold"
                        style={{ color }}
                    >
                        {rul}
                    </div>
                    <div className="text-2xl text-ink-muted">days</div>
                </div>

                <div className="flex items-center gap-2 mb-4">
                    <span
                        className="status-badge text-xs border-0"
                        style={{
                            backgroundColor: `${color}1A`,
                            color: color
                        }}
                    >
                        {urgencyLabels[urgency]}
                    </span>
                </div>

                <div className="text-xs text-ink-muted italic">
                    <svg className="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    {isPhysicsValidated
                        ? "Computed from FactoredMIONet (sf_wbc1_s42) wall-flux using Butler\u2013Volmer kinetics"
                        : "Calculated using Fick\u2019s 2nd Law diffusion model with observed corrosion rates"}
                </div>
                {isPhysicsValidated && (
                    <div
                        className="mt-2 inline-flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full font-semibold"
                        style={{ background: "#E2F1EF", color: "#3C7C61" }}
                    >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        Physics-Validated
                    </div>
                )}
            </div>

            {/* Background decoration */}
            <div
                className="absolute right-0 top-0 w-32 h-32 opacity-[0.06]"
                style={{ color }}
            >
                <svg viewBox="0 0 100 100" fill="currentColor">
                    <path d="M50 10 L90 90 L10 90 Z" />
                </svg>
            </div>
        </div>
    );
}
