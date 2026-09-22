import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";
import type { WaterfallDecomposition } from "../../types";

interface WaterfallChartProps {
    waterfall: WaterfallDecomposition;
}

const NAVY = "rgb(var(--color-primary))";
const CRITICAL = "#D6473C";
const HEALTHY = "#4C9A78";

/**
 * Exact waterfall decomposition of the predicted wall concentration:
 * boundary condition -> soil correction -> fluid correction -> meta
 * correction -> predicted value. Every bar is a real, computed number for
 * THIS segment (not an approximation, not a fixed global split) — see
 * build_benchmarks.py's wall_waterfall().
 */
export default function WaterfallChart({ waterfall }: WaterfallChartProps) {
    let cum = 0;
    const rows: Array<{ name: string; base: number; value: number; delta: number; isAnchor: boolean }> = [];

    rows.push({ name: "Boundary\ncondition", base: 0, value: waterfall.baseline, delta: waterfall.baseline, isAnchor: true });
    cum = waterfall.baseline;

    const steps: Array<[string, number]> = [
        ["Soil\nbranch", waterfall.soil],
        ["Fluid\nbranch", waterfall.fluid],
        ["Meta\nbranch", waterfall.meta],
    ];
    for (const [name, delta] of steps) {
        const start = cum;
        const end = cum + delta;
        rows.push({ name, base: Math.min(start, end), value: Math.abs(delta), delta, isAnchor: false });
        cum = end;
    }

    rows.push({ name: "Predicted\nC(wall)", base: 0, value: waterfall.total, delta: waterfall.total, isAnchor: true });

    const CustomTooltip = ({ active, payload }: any) => {
        if (!active || !payload?.length) return null;
        const row = payload[0].payload;
        return (
            <div style={{ background: "rgb(var(--color-surface))", border: "1px solid rgb(var(--color-line))", borderRadius: 8, padding: "8px 12px", fontSize: 12, color: "rgb(var(--color-ink))" }}>
                <div className="font-semibold mb-0.5">{row.name.replace("\n", " ")}</div>
                <div className="font-mono">{row.isAnchor ? row.delta.toFixed(4) : `${row.delta >= 0 ? "+" : ""}${row.delta.toFixed(4)}`}</div>
            </div>
        );
    };

    return (
        <div style={{ background: "rgb(var(--color-surface))", borderRadius: "0.5rem" }}>
            <ResponsiveContainer width="100%" height={220}>
                <BarChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }} barCategoryGap="20%">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--color-primary))" strokeOpacity={0.08} vertical={false} />
                    <XAxis dataKey="name" stroke="rgb(var(--color-ink-muted))" style={{ fontSize: 10 }} interval={0} />
                    <YAxis stroke="rgb(var(--color-ink-muted))" style={{ fontSize: 10 }} tickFormatter={v => v.toFixed(2)} />
                    <ReferenceLine y={0} stroke="rgb(var(--color-line))" />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgb(var(--color-primary))", fillOpacity: 0.04 }} />
                    {/* Invisible spacer bar to float the visible bar at the right height */}
                    <Bar dataKey="base" stackId="wf" fill="transparent" isAnimationActive={false} />
                    <Bar dataKey="value" stackId="wf" radius={[3, 3, 3, 3]} isAnimationActive={false}>
                        {rows.map((row, i) => (
                            <Cell key={i} fill={row.isAnchor ? NAVY : row.delta >= 0 ? HEALTHY : CRITICAL} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}
