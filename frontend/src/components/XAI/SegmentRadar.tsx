import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from "recharts";
import type { SegmentFingerprint } from "../../types";

interface SegmentRadarProps {
    fingerprint: SegmentFingerprint;
}

const AXES: Array<{ key: keyof SegmentFingerprint; label: string }> = [
    { key: "environmental_integrity", label: "Environment" },
    { key: "soil_resistivity",        label: "Soil Resistivity" },
    { key: "oxygen_scarcity",         label: "O₂ Scarcity" },
    { key: "model_accuracy",          label: "Model Accuracy" },
    { key: "ensemble_agreement",      label: "Ensemble Agreement" },
];

/**
 * Five real, independently-varying quantities normalised against the fleet
 * (all 750 scenarios) — every axis genuinely differs segment to segment,
 * so the shape here is a real fingerprint, not a repeated template. All
 * axes point the same way: further out = healthier / more trustworthy.
 */
export default function SegmentRadar({ fingerprint }: SegmentRadarProps) {
    const data = AXES.map(({ key, label }) => ({
        axis: label,
        value: fingerprint[key] * 100,
    }));

    const CustomTooltip = ({ active, payload }: any) => {
        if (!active || !payload?.length) return null;
        const p = payload[0];
        return (
            <div style={{ background: "rgb(var(--color-surface))", border: "1px solid rgb(var(--color-line))", borderRadius: 8, padding: "6px 10px", fontSize: 12, color: "rgb(var(--color-ink))" }}>
                <div className="font-semibold">{p.payload.axis}</div>
                <div className="font-mono">{p.value.toFixed(0)}%</div>
            </div>
        );
    };

    return (
        <div style={{ background: "rgb(var(--color-surface))", borderRadius: "0.5rem" }}>
            <ResponsiveContainer width="100%" height={240}>
                <RadarChart data={data} outerRadius="70%">
                    <PolarGrid stroke="rgb(var(--color-line))" />
                    <PolarAngleAxis dataKey="axis" style={{ fontSize: 10 }} tick={{ fill: "rgb(var(--color-ink-soft))" }} />
                    <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Radar
                        dataKey="value"
                        stroke="rgb(var(--color-primary))"
                        strokeWidth={2}
                        fill="rgb(var(--color-secondary))"
                        fillOpacity={0.35}
                        isAnimationActive={false}
                    />
                </RadarChart>
            </ResponsiveContainer>
        </div>
    );
}
