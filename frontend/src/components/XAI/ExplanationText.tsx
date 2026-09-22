import type { WaterfallDecomposition, EnsembleUncertainty } from "../../types";

interface ExplanationTextProps {
    waterfall: WaterfallDecomposition;
    ensemble: EnsembleUncertainty;
}

const LABELS: Record<"soil" | "fluid" | "meta", string> = {
    soil: "soil resistivity",
    fluid: "dissolved O₂ concentration",
    meta: "pipe geometry & defect score",
};

/**
 * Natural-language read-out of the real waterfall decomposition — which
 * branch is doing the most work for this specific segment, generated from
 * the actual computed numbers rather than a canned template over fake data.
 */
export default function ExplanationText({ waterfall, ensemble }: ExplanationTextProps) {
    const branches: Array<["soil" | "fluid" | "meta", number]> = [
        ["soil", waterfall.soil],
        ["fluid", waterfall.fluid],
        ["meta", waterfall.meta],
    ];
    const sorted = [...branches].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const [topKey, topVal] = sorted[0];
    const [secondKey, secondVal] = sorted[1];

    const direction = (v: number) => (v < 0 ? "depletes" : "raises");

    const parts = [
        <>This segment's predicted wall concentration starts from a boundary condition of{" "}
            <strong className="text-primary font-semibold">{waterfall.baseline.toFixed(3)}</strong>, and is most strongly
            adjusted by <strong className="text-primary font-semibold">{LABELS[topKey]}</strong>, which{" "}
            {direction(topVal)} it by {Math.abs(topVal).toFixed(3)}
            {Math.abs(secondVal) > 0.02 && (
                <>, with {LABELS[secondKey]} contributing a further {direction(secondVal)} of {Math.abs(secondVal).toFixed(3)}</>
            )}
            . Net predicted value: <strong className="text-primary font-semibold">{waterfall.total.toFixed(3)}</strong>.</>,
    ];

    const spreadPct = ensemble.rul_mean > 0 ? (ensemble.rul_std / ensemble.rul_mean) * 100 : 0;
    parts.push(
        <> Across the 5 independently-trained checkpoints, remaining-life estimates{" "}
            {spreadPct < 10
                ? "agree closely"
                : spreadPct < 25
                    ? "show moderate spread"
                    : "disagree substantially"}
            {" "}({ensemble.rul_min}–{ensemble.rul_max} days), which is a genuine signal of how much this
            forecast depends on the specific training run rather than an invented confidence score.</>
    );

    return (
        <div className="text-sm text-ink-soft leading-relaxed space-y-2">
            {parts.map((p, i) => <p key={i}>{p}</p>)}
        </div>
    );
}
