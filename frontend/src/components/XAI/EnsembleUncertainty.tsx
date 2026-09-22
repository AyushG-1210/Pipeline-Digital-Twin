import type { EnsembleUncertainty as EnsembleUncertaintyData } from "../../types";

interface EnsembleUncertaintyProps {
    ensemble: EnsembleUncertaintyData;
}

/**
 * Stands in for "model confidence" — a deterministic PINN forward pass has
 * no softmax to read a confidence from. What's real instead: 5
 * independently-trained checkpoints (seeds 42-46) run on this exact same
 * segment. How much they agree is a genuine uncertainty signal — narrow
 * spread means the result doesn't depend on which training run you happened
 * to get, wide spread means it does.
 */
export default function EnsembleUncertainty({ ensemble }: EnsembleUncertaintyProps) {
    const { rul_mean, rul_std, rul_min, rul_max, seeds } = ensemble;

    // Agreement: how small is the spread relative to the mean. Not a
    // probability — a relative-consistency score across the ensemble.
    const agreement = rul_mean > 0
        ? Math.max(0, Math.min(1, 1 - rul_std / rul_mean))
        : 1;
    const pct = agreement * 100;

    const color = pct >= 85 ? "#4C9A78" : pct >= 65 ? "#D98E2B" : "#D6473C";

    const circumference = 2 * Math.PI * 45;
    const offset = circumference - (pct / 100) * circumference;

    return (
        <div className="flex items-center gap-5">
            <div className="relative w-24 h-24 shrink-0">
                <svg className="w-full h-full transform -rotate-90">
                    <circle cx="48" cy="48" r="45" stroke="rgb(var(--color-line))" strokeWidth="8" fill="none" />
                    <circle
                        cx="48" cy="48" r="45" stroke={color} strokeWidth="8" fill="none"
                        strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round"
                        className="transition-all duration-500"
                    />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                        <div className="text-2xl font-bold" style={{ color }}>{pct.toFixed(0)}</div>
                        <div className="text-[10px] text-ink-muted">% agree</div>
                    </div>
                </div>
            </div>

            <div className="flex-1 space-y-1.5">
                <div className="text-sm text-ink-soft">
                    <span className="font-mono font-semibold text-ink">{rul_mean.toFixed(0)}</span>
                    <span className="text-ink-muted"> ± {rul_std.toFixed(0)} days</span>
                </div>
                <div className="text-xs text-ink-muted">
                    Range across {seeds.length} seeds: {rul_min}–{rul_max} days
                </div>
                <div className="text-xs text-ink-muted italic">
                    {seeds.length} independently-trained checkpoints (seeds {seeds.join(", ")}), same scenario.
                </div>
            </div>
        </div>
    );
}
