import { useState } from "react";
import { usePipelineStore } from "../../store/usePipelineStore";
import type { PINNOutput } from "../../types";
import IntegrityChart from "./IntegrityChart";
import RULWidget from "./RULWidget";
import DiffusionProfileChart from "./DiffusionProfileChart";
import PINNMetricsCard from "./PINNMetricsCard";

/**
 * Analytics Panel — the PINN degradation forecast for the selected segment.
 * This is the hero tab of the Segment Inspector: everything else (surface
 * scan, explainability) exists to support what's shown here.
 *
 * There's no "Run PINN Analysis" button here anymore. Every segment's PINN
 * output is the real sf_wbc1_s42 checkpoint run once, offline, over a real
 * scenario recovered from the FDM training/holdout dataset (see
 * PINN/build_benchmarks.py) — PINN is the model the project is built
 * around, not an on-demand side feature a user triggers per segment.
 */
export default function AnalyticsPanel() {
    const selectedSegmentId = usePipelineStore(state => state.selectedSegmentId);
    const segments          = usePipelineStore(state => state.segments);

    const selectedSegment = selectedSegmentId ? segments.get(selectedSegmentId) : null;
    const pinnData: PINNOutput | undefined = selectedSegment?.pinn;

    const [profileMode, setProfileMode] = useState<"depth" | "wall">("depth");

    return (
        <div className="h-full flex flex-col">
            {/* Header */}
            <div className="px-4 py-3 border-b border-line bg-surface flex items-center gap-2 shrink-0">
                <span className="text-sm font-semibold text-ink">Degradation Forecast</span>
                {pinnData && (
                    <span className="text-xs text-ink-muted font-mono">
                        {pinnData.governing_equation}
                    </span>
                )}
            </div>

            <div className="flex-1 p-4 overflow-auto">
                {!selectedSegmentId || !pinnData ? (
                    <div className="h-full flex items-center justify-center text-ink-muted">
                        <div className="text-center">
                            <svg className="w-16 h-16 mx-auto mb-4 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                            <p>Select a segment to view its forecast</p>
                        </div>
                    </div>
                ) : (
                    <div className="space-y-4">
                        {/* ── RUL Widget ── */}
                        <RULWidget rul={pinnData.remaining_useful_life_days} isPhysicsValidated={pinnData.is_pinn_result} />

                        {/* ── Integrity Time Series Chart ── */}
                        <div className="inner-card p-4">
                            <h3 className="text-sm font-semibold text-ink mb-4">Integrity Forecast</h3>
                            <IntegrityChart pinnData={pinnData} />
                            <div className="mt-4 flex items-start gap-4 text-xs">
                                <div className="flex items-center gap-2">
                                    <div className="w-8 h-0.5 bg-ink-muted" />
                                    <span className="text-ink-muted">Historical Sensor Integrity</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-8 h-0.5" style={{ background: "#68B0AB" }} />
                                    <span className="text-ink-muted">Physics-Constrained Prediction</span>
                                </div>
                            </div>
                        </div>

                        {/* ── Diffusion Profile Chart ── */}
                        {(pinnData.C_depth_profile || pinnData.C_profile) && (
                            <div className="inner-card p-4">
                                <div className="flex items-center justify-between mb-3">
                                    <h3 className="text-sm font-semibold text-ink">
                                        Diffusion Profile
                                        <span className="ml-2 text-xs font-normal" style={{ color: "#68B0AB" }}>
                                            FactoredMIONet output
                                        </span>
                                    </h3>
                                    {/* Mode toggle */}
                                    <div className="flex rounded-md overflow-hidden text-xs border border-line">
                                        {(["depth", "wall"] as const).map(m => (
                                            <button
                                                key={m}
                                                onClick={() => setProfileMode(m)}
                                                className="px-2.5 py-1 transition-colors"
                                                style={{
                                                    background: profileMode === m ? "#132257" : "transparent",
                                                    color:      profileMode === m ? "#fff"    : "#8B8E9B",
                                                }}
                                            >
                                                {m === "depth" ? "Depth" : "Axial"}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <DiffusionProfileChart pinnData={pinnData} mode={profileMode} />
                                <p className="mt-2 text-xs text-ink-muted italic">
                                    {profileMode === "depth"
                                        ? "Concentration C(d) and potential φ(d) through the pipe wall. d=0 = outer surface, d=1 = bulk interior."
                                        : "C(x) and φ(x) along the pipe axis at the outer wall surface."}
                                </p>
                            </div>
                        )}

                        {/* ── PINN Metrics Card ── */}
                        {(pinnData.C_l2 !== undefined || pinnData.op_corr !== undefined) && (
                            <PINNMetricsCard pinnData={pinnData} />
                        )}

                        {/* ── Governing Physics ── */}
                        <div className="inner-card p-4">
                            <h3 className="text-sm font-semibold text-ink mb-2">Governing Physics</h3>
                            <div className="text-sm text-ink-soft space-y-2">
                                <p>
                                    <span className="font-mono text-ink font-semibold">
                                        {pinnData.governing_equation}
                                    </span>
                                    {" "}— Diffusion-based corrosion propagation model
                                    {pinnData.governing_equation === "Fick_2nd_Law_strong" && (
                                        <span className="ml-1 text-xs" style={{ color: "#68B0AB" }}>(strong form, pointwise collocation)</span>
                                    )}
                                </p>
                                <p className="text-xs text-ink-muted italic">
                                    Forecast produced by the trained FactoredMIONet (sf_wbc1_s42). The model solves the
                                    coupled PDE for O₂ concentration C and electric potential φ, enforcing Fick’s 2nd Law
                                    and Butler–Volmer boundary conditions at the outer wall.
                                </p>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
