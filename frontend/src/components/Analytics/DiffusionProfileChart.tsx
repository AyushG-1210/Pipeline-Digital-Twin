import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Legend
} from "recharts";
import type { PINNOutput } from "../../types";

interface DiffusionProfileChartProps {
    pinnData: PINNOutput;
    /** "wall" = axial wall profile (100pts), "depth" = depth profile at midpoint (50pts) */
    mode?: "wall" | "depth";
}

/**
 * Spatial diffusion profile chart.
 *
 * Wall mode:  x-axis = normalised axial position (0–1) along pipe.
 *             Shows C(x) and φ(x) at the outer wall surface.
 * Depth mode: x-axis = normalised depth (0 = outer wall, 1 = bulk interior).
 *             Shows how concentration and potential vary through the wall thickness.
 */
export default function DiffusionProfileChart({
    pinnData,
    mode = "depth",
}: DiffusionProfileChartProps) {
    const isDepth = mode === "depth";
    const C_data   = isDepth ? pinnData.C_depth_profile   : pinnData.C_profile;
    const phi_data = isDepth ? pinnData.phi_depth_profile : pinnData.phi_profile;

    if (!C_data || !phi_data) return null;

    const N = C_data.length;
    const chartData = Array.from({ length: N }, (_, i) => {
        const x = parseFloat((i / (N - 1)).toFixed(3));
        return {
            x,
            C:   parseFloat(C_data[i].toFixed(4)),
            phi: parseFloat(phi_data[i].toFixed(4)),
        };
    });

    const xLabel = isDepth
        ? "Depth (0 = outer wall, 1 = bulk)"
        : "Axial position (0–1)";

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (!active || !payload?.length) return null;
        return (
            <div
                style={{
                    background: "rgb(var(--color-surface))",
                    border: "1px solid rgb(var(--color-line))",
                    borderRadius: 8,
                    padding: "8px 12px",
                    fontSize: 12,
                    color: "rgb(var(--color-ink))",
                }}
            >
                <div className="font-mono text-ink-muted mb-1">
                    {isDepth ? "depth" : "x"} = {label}
                </div>
                {payload.map((p: any) => (
                    <div key={p.name} style={{ color: p.color }}>
                        {p.name === "C" ? "C (conc.)" : "φ (potential)"} ={" "}
                        {p.value.toFixed(4)}
                    </div>
                ))}
            </div>
        );
    };

    // Normalise phi to the same 0–1 range as C for overlay, keep tick labels accurate
    const phiMin = Math.min(...phi_data);
    const phiMax = Math.max(...phi_data);
    const phiRange = phiMax - phiMin || 1e-6;
    const chartDataNorm = chartData.map(d => ({
        ...d,
        phi_norm: (d.phi - phiMin) / phiRange,
    }));

    return (
        <div style={{ background: "rgb(var(--color-surface))", borderRadius: "0.5rem" }}>
            <ResponsiveContainer width="100%" height={220}>
                <AreaChart
                    data={chartDataNorm}
                    margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
                >
                    <defs>
                        <linearGradient id="gradC" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor="rgb(var(--color-primary))" stopOpacity={0.35} />
                            <stop offset="95%" stopColor="rgb(var(--color-primary))" stopOpacity={0.03} />
                        </linearGradient>
                        <linearGradient id="gradPhi" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor="#68B0AB" stopOpacity={0.4} />
                            <stop offset="95%" stopColor="#68B0AB" stopOpacity={0.05} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--color-primary))" strokeOpacity={0.08} />
                    <XAxis
                        dataKey="x"
                        stroke="rgb(var(--color-ink-muted))"
                        style={{ fontSize: 10 }}
                        label={{ value: xLabel, position: "insideBottom", offset: -2, fill: "rgb(var(--color-ink-muted))", fontSize: 10 }}
                        tickFormatter={v => v.toFixed(2)}
                    />
                    <YAxis
                        stroke="rgb(var(--color-ink-muted))"
                        domain={[0, 1]}
                        style={{ fontSize: 10 }}
                        tickFormatter={v => v.toFixed(1)}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                        wrapperStyle={{ paddingTop: 8, fontSize: 11 }}
                        formatter={(val) => val === "C" ? "C(x) — Concentration" : "φ(x) — Potential (norm.)"}
                    />
                    <Area
                        type="monotone"
                        dataKey="C"
                        stroke="rgb(var(--color-primary))"
                        strokeWidth={2}
                        fill="url(#gradC)"
                        dot={false}
                        name="C"
                    />
                    <Area
                        type="monotone"
                        dataKey="phi_norm"
                        stroke="#68B0AB"
                        strokeWidth={2}
                        fill="url(#gradPhi)"
                        dot={false}
                        name="phi"
                    />
                </AreaChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-4 text-xs text-ink-muted pb-2">
                <span>φ range: [{phiMin.toFixed(3)}, {phiMax.toFixed(3)}]</span>
                <span className="text-line-strong">|</span>
                <span>C range: [{Math.min(...C_data).toFixed(3)}, {Math.max(...C_data).toFixed(3)}]</span>
            </div>
        </div>
    );
}
