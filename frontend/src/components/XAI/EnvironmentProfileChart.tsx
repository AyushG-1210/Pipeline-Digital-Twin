import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Legend
} from "recharts";
import type { EnvironmentProfile } from "../../types";

interface EnvironmentProfileChartProps {
    environment: EnvironmentProfile;
}

/**
 * The real environmental scenario behind this segment's prediction — the
 * actual soil resistivity and bulk O2 concentration curves recovered from
 * the FDM dataset. This is the literal cause, not a derived explanation:
 * every segment's inputs come from a distinct real scenario, so this chart
 * is never the same shape twice.
 */
export default function EnvironmentProfileChart({ environment }: EnvironmentProfileChartProps) {
    const { x, soil_resistivity, fluid_concentration } = environment;

    const rhoMin = Math.min(...soil_resistivity);
    const rhoMax = Math.max(...soil_resistivity);
    const rhoRange = rhoMax - rhoMin || 1e-6;

    const data = x.map((xi, i) => ({
        x: parseFloat(xi.toFixed(3)),
        rho: soil_resistivity[i],
        rho_norm: (soil_resistivity[i] - rhoMin) / rhoRange,
        c_bulk: fluid_concentration[i],
    }));

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (!active || !payload?.length) return null;
        return (
            <div style={{ background: "rgb(var(--color-surface))", border: "1px solid rgb(var(--color-line))", borderRadius: 8, padding: "8px 12px", fontSize: 12, color: "rgb(var(--color-ink))" }}>
                <div className="font-mono text-ink-muted mb-1">x = {label}</div>
                {payload.map((p: any) => (
                    <div key={p.name} style={{ color: p.color }}>
                        {p.name === "rho_norm"
                            ? `ρ (soil) = ${p.payload.rho.toFixed(0)} Ω·m`
                            : `C_bulk (O₂) = ${p.payload.c_bulk.toFixed(3)}`}
                    </div>
                ))}
            </div>
        );
    };

    return (
        <div style={{ background: "rgb(var(--color-surface))", borderRadius: "0.5rem" }}>
            <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                    <defs>
                        <linearGradient id="gradRho" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#D98E2B" stopOpacity={0.35} />
                            <stop offset="95%" stopColor="#D98E2B" stopOpacity={0.03} />
                        </linearGradient>
                        <linearGradient id="gradCBulk" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="rgb(var(--color-primary))" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="rgb(var(--color-primary))" stopOpacity={0.03} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--color-primary))" strokeOpacity={0.08} />
                    <XAxis
                        dataKey="x" stroke="rgb(var(--color-ink-muted))" style={{ fontSize: 10 }}
                        label={{ value: "Axial position (0–1)", position: "insideBottom", offset: -2, fill: "rgb(var(--color-ink-muted))", fontSize: 10 }}
                        tickFormatter={v => v.toFixed(1)}
                    />
                    <YAxis domain={[0, 1]} stroke="rgb(var(--color-ink-muted))" style={{ fontSize: 10 }} tickFormatter={v => v.toFixed(1)} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                        wrapperStyle={{ paddingTop: 8, fontSize: 11 }}
                        formatter={(val) => val === "rho_norm" ? "Soil resistivity ρ(x)" : "Bulk O₂ concentration"}
                    />
                    <Area type="monotone" dataKey="rho_norm" stroke="#D98E2B" strokeWidth={2} fill="url(#gradRho)" dot={false} name="rho_norm" />
                    <Area type="monotone" dataKey="c_bulk" stroke="rgb(var(--color-primary))" strokeWidth={2} fill="url(#gradCBulk)" dot={false} name="c_bulk" />
                </AreaChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-4 text-xs text-ink-muted pb-2">
                <span>ρ range: [{rhoMin.toFixed(0)}, {rhoMax.toFixed(0)}] Ω·m</span>
                <span className="text-line-strong">|</span>
                <span>C_bulk range: [{Math.min(...fluid_concentration).toFixed(3)}, {Math.max(...fluid_concentration).toFixed(3)}]</span>
            </div>
        </div>
    );
}
