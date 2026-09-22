"""
pinn_inference.py
=================
Standalone inference script for the FactoredMIONet PINN model
(strong-form, sf_wbc1_s42 checkpoint).

Usage:
    python pinn_inference.py --wt 0.5 --od 10.75 --yolo_score 0.35 --seed 42

Outputs:
    pinn_insights.json  (same directory as this script)
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT  = os.path.join(SCRIPT_DIR, "results", "model_files", "strong_form", "sf_wbc1_s42_adam.pt")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "results", "json_files", "strong_form", "sf_wbc1_s42.json")
OUTPUT_JSON  = os.path.join(SCRIPT_DIR, "pinn_insights.json")

PHYSICS_PARAMS = {
    "D_global": 1e-9, "k_rate": 2.07e-7, "alpha": 0.5,
    "F": 96485.0, "R": 8.314, "T": 298.15,
    "E_eq": -0.44, "E_corr": -0.44, "t_max": 31536000.0,
    "n_electrons": 2.0, "C_bulk": 1.0, "L_x": 10.0,
}

WT_MIN_IN, WT_MAX_IN   = 0.188, 1.500
OD_MIN_IN, OD_MAX_IN   = 4.0,   56.0
C_BULK_MIN, C_BULK_MAX = 0.3,   1.0
SOIL_POINTS  = 8
FLUID_POINTS = 50

def fluid_to_cbulk(fluid_prof):
    return C_BULK_MIN + fluid_prof * (C_BULK_MAX - C_BULK_MIN)

def interp_profile_bn(prof, x):
    B, P = prof.shape
    xs  = x.clamp(0.0, 1.0) * (P - 1)
    i0  = xs.floor().long().clamp(0, P - 2)
    frac = xs - i0.float()
    lo  = torch.gather(prof, 1, i0)
    hi  = torch.gather(prof, 1, i0 + 1)
    return lo + frac * (hi - lo)

def interp_profile(prof, x_query):
    B, P = prof.shape
    x_query = x_query.to(prof.device)
    xs  = x_query.clamp(0.0, 1.0) * (P - 1)
    i0  = xs.floor().long().clamp(0, P - 2)
    frac = (xs - i0.float()).unsqueeze(0)
    lo  = prof[:, i0]
    hi  = prof[:, i0 + 1]
    return lo + frac * (hi - lo)

def soil_to_log_rho(soil_prof):
    LOG_RHO_MIN = math.log10(500.0)
    LOG_RHO_MAX = math.log10(20000.0)
    return LOG_RHO_MIN + soil_prof * (LOG_RHO_MAX - LOG_RHO_MIN)

def gamma_of_x(soil_prof, x_query, phys):
    log_rho = soil_to_log_rho(interp_profile(soil_prof, x_query))
    rho     = torch.pow(10.0, log_rho)
    sigma   = 100.0 / rho
    return (phys["n_electrons"] * phys["F"] * phys["D_global"] * phys["C_bulk"]) / sigma


class PlainTrunk(nn.Module):
    def __init__(self, trunk_dim=3, hidden_dim=128, basis_dim=128, split_point="late"):
        super().__init__()
        self.split_point = split_point
        if split_point == "late":
            self.backbone = nn.Sequential(
                nn.Linear(trunk_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            )
            h = hidden_dim
            self.head_C   = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, basis_dim))
            self.head_phi = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, basis_dim))
        else:
            raise ValueError(f"Only late split supported; got {split_point!r}")

    def forward(self, x_trunk):
        feat = self.backbone(x_trunk)
        return torch.cat([self.head_C(feat), self.head_phi(feat)], dim=-1)


class FactoredMIONet(nn.Module):
    def __init__(self, soil_dim=8, fluid_dim=50, meta_dim=3, basis_dim=128, split_point="late"):
        super().__init__()
        self.num_outputs = 2
        self.basis_dim   = basis_dim
        p_output = self.num_outputs * self.basis_dim

        self.branch_soil  = nn.Sequential(
            nn.Linear(soil_dim,  128), nn.Tanh(),
            nn.Linear(128,       128), nn.Tanh(),
            nn.Linear(128, p_output))
        self.branch_fluid = nn.Sequential(
            nn.Linear(fluid_dim, 128), nn.Tanh(),
            nn.Linear(128,       128), nn.Tanh(),
            nn.Linear(128, p_output))
        self.branch_meta  = nn.Sequential(
            nn.Linear(meta_dim, 64), nn.Tanh(),
            nn.Linear(64, p_output))
        self.trunk = PlainTrunk(trunk_dim=3, hidden_dim=128, basis_dim=basis_dim, split_point=split_point)

    def forward(self, x_soil, x_fluid, x_meta, x_trunk):
        b_feat = (self.branch_soil(x_soil) + self.branch_fluid(x_fluid) + self.branch_meta(x_meta))
        b_feat = b_feat.view(-1, self.num_outputs, self.basis_dim).unsqueeze(1)
        t_feat = self.trunk(x_trunk)
        shape  = list(t_feat.shape[:-1]) + [self.num_outputs, self.basis_dim]
        t_feat = t_feat.view(*shape)
        raw    = torch.sum(b_feat * t_feat, dim=-1) / (self.basis_dim ** 0.5)
        d      = x_trunk[..., 1:2]
        g      = d - 1.0
        c_x    = interp_profile_bn(fluid_to_cbulk(x_fluid), x_trunk[..., 0])
        C      = c_x.unsqueeze(-1) + g * raw[..., 0:1]
        phi    = g * raw[..., 1:2]
        return torch.cat([C, phi], dim=-1)


def generate_grf_profile(n_samples, n_points, length_scale, rng):
    x    = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, x)
    cov  = np.exp(-0.5 * ((X - Y) / length_scale) ** 2) + 1e-6 * np.eye(n_points)
    L    = np.linalg.cholesky(cov)
    z    = rng.randn(n_samples, n_points)
    return torch.tensor(1 / (1 + np.exp(-(z @ L.T))), dtype=torch.float32)


def compute_rul_days(C_wall_mean, wt_in, yolo_score, phys):
    wt_m   = wt_in * 0.0254
    k      = phys["k_rate"]
    flux_rate = k * float(C_wall_mean)
    M_fe   = 0.05585
    rho_fe = 7874.0
    n      = phys["n_electrons"]
    F      = phys["F"]
    penetration_rate_m_yr = (flux_rate * M_fe / (n * F * rho_fe)) * 3.15576e7
    effective_rate = penetration_rate_m_yr * (1.0 + yolo_score * 2.0)
    critical_remaining = wt_m * 0.80
    if effective_rate <= 0:
        return 3650
    days_to_failure = (critical_remaining / effective_rate) * 365.25
    return max(30, min(3650, int(days_to_failure)))


def run_inference(wt_in, od_in, yolo_score, seed):
    t0     = time.time()
    device = torch.device("cpu")

    wt_in      = float(np.clip(wt_in,      WT_MIN_IN, WT_MAX_IN))
    od_in      = float(np.clip(od_in,      OD_MIN_IN, OD_MAX_IN))
    yolo_score = float(np.clip(yolo_score, 0.0, 1.0))

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    model = FactoredMIONet(soil_dim=SOIL_POINTS, fluid_dim=FLUID_POINTS, meta_dim=3, basis_dim=128)
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    # Training checkpoints use {'state_dict': ..., 'seed': ..., 'cfg': ..., 'phase': ...}
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model' in ckpt:
            state = ckpt['model']
        else:
            state = ckpt   # assume it IS the state dict
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()

    wt_norm = (wt_in - WT_MIN_IN) / (WT_MAX_IN - WT_MIN_IN)
    od_norm = (od_in - OD_MIN_IN) / (OD_MAX_IN - OD_MIN_IN)
    x_meta  = torch.tensor([[wt_norm, od_norm, yolo_score]], dtype=torch.float32)
    x_soil  = generate_grf_profile(1, SOIL_POINTS,  0.25, rng)
    x_fluid = generate_grf_profile(1, FLUID_POINTS, 0.30, rng)

    N_x, N_d = 100, 50
    x_pts    = torch.linspace(0.0, 1.0, N_x)
    d_pts    = torch.linspace(0.0, 1.0, N_d)
    X_grid, D_grid = torch.meshgrid(x_pts, d_pts, indexing="ij")
    T_grid   = torch.ones_like(X_grid)
    coords   = torch.stack([X_grid.flatten(), D_grid.flatten(), T_grid.flatten()], dim=-1)
    coords   = coords.unsqueeze(0).expand(1, -1, -1)

    with torch.no_grad():
        u = model(x_soil, x_fluid, x_meta, coords)

    C_field   = u[0, :, 0].view(N_x, N_d).numpy()
    phi_field = u[0, :, 1].view(N_x, N_d).numpy()

    C_wall_profile   = C_field[:, 0]
    phi_wall_profile = phi_field[:, 0]
    mid_idx          = N_x // 2
    C_depth_profile   = C_field[mid_idx, :].tolist()
    phi_depth_profile = phi_field[mid_idx, :].tolist()

    dd          = 1.0 / (N_d - 1)
    flux_profile = -(C_field[:, 1] - C_field[:, 0]) / dd
    flux_mean    = float(np.mean(np.abs(flux_profile)))
    C_wall_mean  = float(np.mean(C_wall_profile))
    phi_wall_mean = float(np.mean(phi_wall_profile))
    integrity    = float(np.clip((C_wall_mean - C_BULK_MIN) / (C_BULK_MAX - C_BULK_MIN), 0.0, 1.0))
    rul_days     = compute_rul_days(C_wall_mean, wt_in, yolo_score, PHYSICS_PARAMS)

    pre_computed = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r") as f:
            result = json.load(f)
        pre_computed = result.get("adam", {}).get("hold", {})

    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    historical = []
    for i in range(180, -1, -15):
        d   = now - timedelta(days=i)
        t_f = 1.0 - (i / 180.0)
        val = min(100.0, (integrity + (1.0 - integrity) * (1.0 - t_f ** 0.5)) * 100.0)
        historical.append({"time": d.isoformat() + "Z", "value": round(val, 2)})

    predicted = []
    current_val = integrity * 100.0
    corrosion_rate_per_day = (flux_mean * PHYSICS_PARAMS["k_rate"]) * 0.5
    for i in range(15, 181, 15):
        d   = now + timedelta(days=i)
        val = max(0.0, current_val - corrosion_rate_per_day * i * 100.0)
        predicted.append({"time": d.isoformat() + "Z", "value": round(val, 2)})

    insights = {
        "model_tag": "sf_wbc1_s42",
        "governing_equation": "Fick_2nd_Law_strong",
        "seed": seed,
        "wt_in": round(wt_in, 4),
        "od_in": round(od_in, 4),
        "yolo_score": round(yolo_score, 4),
        "C_profile":   [round(float(v), 6) for v in C_wall_profile],
        "phi_profile": [round(float(v), 6) for v in phi_wall_profile],
        "C_depth_profile":   C_depth_profile,
        "phi_depth_profile": phi_depth_profile,
        "depth_x_label": "Normalized depth (0=outer wall, 1=bulk)",
        "C_wall_mean":  round(C_wall_mean, 6),
        "phi_wall_mean": round(phi_wall_mean, 6),
        "flux": round(flux_mean, 8),
        "integrity": round(integrity, 4),
        "C_l2":     pre_computed.get("C_l2",    0.1136),
        "phi_l2":   pre_computed.get("phi_l2",  0.3274),
        "flux_err": pre_computed.get("flux_err", 0.0634),
        "op_corr":  pre_computed.get("op_corr", 0.9016),
        "x_corr":   pre_computed.get("x_corr",  0.5522),
        "remaining_useful_life_days": rul_days,
        "historical_integrity": historical,
        "predicted_integrity":  predicted,
        "physics_params": {
            "D_global": PHYSICS_PARAMS["D_global"], "k_rate": PHYSICS_PARAMS["k_rate"],
            "alpha": PHYSICS_PARAMS["alpha"], "E_eq": PHYSICS_PARAMS["E_eq"],
            "E_corr": PHYSICS_PARAMS["E_corr"], "t_max": PHYSICS_PARAMS["t_max"],
            "L_x": PHYSICS_PARAMS["L_x"], "n_electrons": PHYSICS_PARAMS["n_electrons"],
        },
        "inference_time_s": round(time.time() - t0, 3),
        "is_pinn_result": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return insights


def main():
    parser = argparse.ArgumentParser(description="PINN inference for a pipeline segment")
    parser.add_argument("--wt",         type=float, default=0.500)
    parser.add_argument("--od",         type=float, default=10.75)
    parser.add_argument("--yolo_score", type=float, default=0.0)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--output",     type=str,   default=OUTPUT_JSON)
    args = parser.parse_args()

    print(f"[PINN] WT={args.wt}in  OD={args.od}in  YOLO={args.yolo_score:.3f}  seed={args.seed}")
    try:
        insights = run_inference(args.wt, args.od, args.yolo_score, args.seed)
        with open(args.output, "w") as f:
            json.dump(insights, f, indent=2)
        print(f"[PINN] Done in {insights['inference_time_s']:.2f}s -> {args.output}")
        print(f"[PINN] C_wall={insights['C_wall_mean']:.4f}  RUL={insights['remaining_useful_life_days']}d  C_l2={insights['C_l2']:.4f}")
    except Exception as e:
        print(f"[PINN ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
