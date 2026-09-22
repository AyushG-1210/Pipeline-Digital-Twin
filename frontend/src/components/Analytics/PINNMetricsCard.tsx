import type { PINNOutput } from "../../types";

interface PINNMetricsCardProps {
    pinnData: PINNOutput;
}

function MetricRow({
    label, value, unit = "", good, warn,
    invert = false,
    description,
}: {
    label: string;
    value: number | undefined;
    unit?: string;
    good: number;
    warn: number;
    invert?: boolean;
    description?: string;
}) {
    if (value === undefined || value === null) return null;

    let color: string;
    if (!invert) {
        // Higher is better (e.g. op_corr)
        color = value >= good ? "#4C9A78" : value >= warn ? "#D98E2B" : "#D6473C";
    } else {
        // Lower is better (e.g. C_l2, flux)
        color = value <= good ? "#4C9A78" : value <= warn ? "#D98E2B" : "#D6473C";
    }

    const pct = !invert
        ? Math.min(100, (value / good) * 100)
        : Math.min(100, Math.max(0, (1 - (value - good) / (warn - good + 1e-9)) * 100));

    return (
        <div className="space-y-1">
            <div className="flex justify-between items-baseline text-xs">
                <span className="text-ink-muted">{label}</span>
                <span className="font-mono font-semibold" style={{ color }}>
                    {value.toFixed(4)}{unit}
                </span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-line overflow-hidden">
                <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${pct}%`, backgroundColor: color }}
                />
            </div>
            {description && (
                <div className="text-xs text-ink-muted italic">{description}</div>
            )}
        </div>
    );
}

/**
 * Compact card displaying PINN model quality metrics.
 * Shows C_l2, op_corr, and flux with colour-coded quality bars.
 */
export default function PINNMetricsCard({ pinnData }: PINNMetricsCardProps) {
    const isReal = pinnData.is_pinn_result;

    return (
        <div className="inner-card p-4 space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-ink">
                    Physics Model Quality
                </h3>
                <div className="flex items-center gap-2">
                    {isReal ? (
                        <span
                            className="status-badge text-xs px-2 py-0.5 border-0"
                            style={{ background: "#E2F1EF", color: "#3C7C61", borderRadius: 6 }}
                        >
                            ⚡ Real Inference
                        </span>
                    ) : (
                        <span
                            className="status-badge text-xs px-2 py-0.5 border-0"
                            style={{ background: "rgb(var(--color-surface-sunken))", color: "rgb(var(--color-ink-muted))", borderRadius: 6 }}
                        >
                            Mock Data
                        </span>
                    )}
                </div>
            </div>

            {/* Model tag */}
            {pinnData.model_tag && (
                <div className="flex items-center gap-2 text-xs">
                    <span className="text-ink-muted">Checkpoint:</span>
                    <span className="font-mono text-primary">{pinnData.model_tag}</span>
                </div>
            )}

            {/* Inputs */}
            {(pinnData.wt_in || pinnData.od_in) && (
                <div className="flex gap-4 text-xs text-ink-muted border-t border-line pt-3">
                    {pinnData.wt_in && <span>WT: <b className="text-ink-soft">{pinnData.wt_in.toFixed(3)}&quot;</b></span>}
                    {pinnData.od_in && <span>OD: <b className="text-ink-soft">{pinnData.od_in.toFixed(2)}&quot;</b></span>}
                    {pinnData.yolo_score !== undefined && (
                        <span>YOLO: <b className="text-ink-soft">{(pinnData.yolo_score * 100).toFixed(1)}%</b></span>
                    )}
                    {pinnData.inference_time_s !== undefined && (
                        <span className="ml-auto text-ink-muted">{pinnData.inference_time_s.toFixed(2)}s</span>
                    )}
                </div>
            )}

            {/* Metrics */}
            <div className="space-y-3 border-t border-line pt-3">
                <MetricRow
                    label="C_l2 — Concentration L2 error"
                    value={pinnData.C_l2}
                    good={0.16}
                    warn={0.34}
                    invert={true}
                    description="Lower is better. Computed against this scenario's real FDM ground truth (25th/75th pct. across the benchmark set: 0.16 / 0.34)."
                />
                <MetricRow
                    label="op_corr — Operational correlation"
                    value={pinnData.op_corr}
                    good={0.50}
                    warn={0.25}
                    invert={false}
                    description="Real correlation between predicted and true wall concentration across the 50-sample holdout set — a genuine weak point on reconstructed real-world inputs, not the paper's clean-eval number."
                />
                <MetricRow
                    label="flux — Surface corrosion flux"
                    value={pinnData.flux}
                    unit=""
                    good={0.38}
                    warn={0.52}
                    invert={true}
                    description="Mean |dC/dd| at outer wall. Lower = less aggressive corrosion. Range across the benchmark set: 0.18–0.67."
                />
                {pinnData.phi_l2 !== undefined && (
                    <MetricRow
                        label="φ_l2 — Potential L2 error"
                        value={pinnData.phi_l2}
                        good={0.24}
                        warn={0.42}
                        invert={true}
                    />
                )}
            </div>

            {/* Physics params summary */}
            {pinnData.physics_params && (
                <div className="border-t border-line pt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                    <span className="text-ink-muted">D<sub>global</sub></span>
                    <span className="font-mono text-ink-soft">{pinnData.physics_params.D_global.toExponential(2)} m²/s</span>
                    <span className="text-ink-muted">k<sub>rate</sub></span>
                    <span className="font-mono text-ink-soft">{pinnData.physics_params.k_rate.toExponential(2)} m/s</span>
                    <span className="text-ink-muted">α (Butler-Volmer)</span>
                    <span className="font-mono text-ink-soft">{pinnData.physics_params.alpha}</span>
                    <span className="text-ink-muted">E<sub>eq</sub></span>
                    <span className="font-mono text-ink-soft">{pinnData.physics_params.E_eq} V</span>
                </div>
            )}
        </div>
    );
}
