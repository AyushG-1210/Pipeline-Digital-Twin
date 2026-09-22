"""
Figures 1-3 for the paper, regenerated from the real checkpoints / JSON records.

Inputs (read-only):
  main.ipynb                       function + class definitions only (cells 3, 11, 13),
                                   pulled by AST so no cell side effects run
  data/dataset_v2.pt               soil / fluid / meta / raw tensors (rows 950-999 = held-out)
  data/fdm_hold_950_1000.npz       FDM reference on the held-out rows
  results/model_files/...          Adam checkpoints, seed 42, step-matched arms:
        strong_form/sa_wbc1_noanchor_s42_adam.pt   strong, W_ANCHOR=0   (7,216 steps)
        weak_form/dn_plain_anchored_s42_adam.pt    weak,   W_ANCHOR=1   (7,205 steps)
        weak_form/dn_plain_noanchor_s42_adam.pt    weak,   W_ANCHOR=0   (7,205 steps)
  results/json_files/weak_form/m2_bs64_matched_s4X.json  (7,205 steps)  } Figure 2
  results/json_files/weak_form/m2_bs64_full_s4X.json     (13,200 steps) }

Outputs: results/figures/fig{1,2,3}_*.{png,pdf} and fig_data.json (the plotted numbers).

Run from PINN/:  python results/figures/make_figures.py
"""
import ast, json, os, statistics as st
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PINN = os.path.abspath(os.path.join(HERE, "..", ".."))
NB = os.path.join(PINN, "main.ipynb")
DATA = os.path.join(PINN, "data")
JSON_W = os.path.join(PINN, "results", "json_files", "weak_form")
CKPT = os.path.join(PINN, "results", "model_files")
SEEDS = [42, 43, 44, 45, 46]

# ---- palette (dataviz reference instance, light mode) ----------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
})


# =============================================================================
# Pull definitions from the notebook without executing cell bodies
# =============================================================================
def notebook_defs(cell_idx, names=None):
    """Return source of top-level def/class (and simple assigns) from a cell.
    names=None -> everything except bare expression statements / calls."""
    nb = json.load(open(NB, encoding="utf-8"))
    src = "".join(nb["cells"][cell_idx]["source"])
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nm = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name):
                nm = t.id
            elif isinstance(t, ast.Tuple) and all(isinstance(e, ast.Name) for e in t.elts):
                nm = t.elts[0].id                      # e.g. C_BULK_MIN, C_BULK_MAX = 0.3, 1.0
            else:
                continue
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(ast.get_source_segment(src, node)); continue
        else:
            continue
        if names is None or nm in names:
            keep.append(ast.get_source_segment(src, node))
    return "\n".join(keep)


ns = {"execution_device": "cpu", "torch": torch, "np": np}
# cell 3: only the two helpers the model's forward() needs (+ their constants)
exec(notebook_defs(3, {"C_BULK_MIN", "C_BULK_MAX", "fluid_to_cbulk", "interp_profile_bn"}), ns)
# cell 11: quadrature + test-function builders and the module-level tensors.
# Executed whole: it only builds tensors, asserts and prints (no file writes),
# and its tuple-unpacking assigns are what build PTS_SPACE / D_EDGES / quad_tensors.
_nb = json.load(open(NB, encoding="utf-8"))
exec("".join(_nb["cells"][11]["source"]), ns)
# cell 13: PlainTrunk / FactoredMIONet (+ SPLIT_POINT and the width helpers)
exec(notebook_defs(13), ns)
FactoredMIONet = ns["FactoredMIONet"]

# =============================================================================
# Data + checkpoints
# =============================================================================
cache = torch.load(os.path.join(DATA, "dataset_v2.pt"), map_location="cpu", weights_only=False)
soil, fluid, meta, raw = cache["tensors"]
HOLD = slice(950, 1000)
b_soil, b_fluid, b_meta, raw_meta = soil[HOLD], fluid[HOLD], meta[HOLD], raw[HOLD]

ho = np.load(os.path.join(DATA, "fdm_hold_950_1000.npz"))
fdm_x = torch.tensor(ho["x"], dtype=torch.float32)
fdm_y = torch.tensor(ho["y"], dtype=torch.float32)
C_true = torch.tensor(ho["C"], dtype=torch.float32)          # [50, Nx, Ny]
cw_true = C_true[:, :, -1]                                   # wall = last y index (d = 0)
N_HOLD = C_true.shape[0]


def load_model(rel):
    m = FactoredMIONet(soil_dim=8, fluid_dim=50, meta_dim=3, split_point="late")
    sd = torch.load(os.path.join(CKPT, rel), map_location="cpu", weights_only=False)["state_dict"]
    m.load_state_dict(sd); m.eval()
    return m


ARMS = [  # label, checkpoint, colour
    ("strong, unanchored",   "strong_form/sa_wbc1_noanchor_s42_adam.pt", BLUE),
    ("weak, anchored",       "weak_form/dn_plain_anchored_s42_adam.pt",  AQUA),
    ("weak, unanchored",     "weak_form/dn_plain_noanchor_s42_adam.pt",  ORANGE),
]


@torch.no_grad()
def wall_profile(m, x_grid):
    """C_wall(x) at t = 1 for every held-out environment: coords (x, d=0, t=1)."""
    coords = torch.stack([x_grid, torch.zeros_like(x_grid), torch.ones_like(x_grid)], -1)
    coords = coords.unsqueeze(0).expand(N_HOLD, -1, -1)
    return m(b_soil, b_fluid, b_meta, coords)[..., 0]        # [50, Nx]


@torch.no_grad()
def full_field_wall(m):
    """Same evaluation validate_pinn_vs_fdm uses (meshgrid, d = 1 - y), wall slice."""
    X, Y = torch.meshgrid(fdm_x, fdm_y, indexing="ij")
    coords = torch.stack([X.flatten(), 1.0 - Y.flatten(), torch.ones_like(X.flatten())], -1)
    coords = coords.unsqueeze(0).expand(N_HOLD, -1, -1)
    return m(b_soil, b_fluid, b_meta, coords)[..., 0].view(N_HOLD, len(fdm_x), len(fdm_y))[:, :, -1]


fig_data = {}

# =============================================================================
# FIGURE 1 -- C_wall(x) for four held-out environments, four curves per panel
# =============================================================================
x_fine = torch.linspace(0, 1, 201)
models = {lab: load_model(p) for lab, p, _ in ARMS}
prof_fine = {lab: wall_profile(m, x_fine) for lab, m in models.items()}
prof_grid = {lab: full_field_wall(m) for lab, m in models.items()}

# consistency check against the recorded metric (x_corr is the per-env mean corr on the FDM grid)
rec_xcorr = {
    "strong, unanchored": json.load(open(os.path.join(PINN, "results/json_files/strong_form/sa_wbc1_noanchor_s42.json")))["hold"]["x_corr"],
    "weak, anchored":     json.load(open(os.path.join(JSON_W, "dn_plain_anchored_s42.json")))["hold"]["x_corr"],
    "weak, unanchored":   json.load(open(os.path.join(JSON_W, "dn_plain_noanchor_s42.json")))["hold"]["x_corr"],
}
env_xcorr = {}
for lab in models:
    cs = [np.corrcoef(prof_grid[lab][i].numpy(), cw_true[i].numpy())[0, 1] for i in range(N_HOLD)]
    env_xcorr[lab] = cs
    got = float(np.mean(cs))
    print(f"[fig1 check] {lab:<20} x_corr recomputed {got:+.4f}  recorded {rec_xcorr[lab]:+.4f}")
    assert abs(got - rec_xcorr[lab]) < 2e-3, "checkpoint / metric mismatch"

# four environments spanning wall thickness: thinnest, two interior quantiles, thickest
order = torch.argsort(raw_meta[:, 0]).tolist()
ENVS = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]]

fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3), sharey=False)
for ax, i in zip(axes, ENVS):
    ax.plot(fdm_x, cw_true[i], color=INK, lw=2.0, marker="o", ms=3.2, label="FDM reference")
    for lab, _, col in ARMS:
        ax.plot(x_fine, prof_fine[lab][i], color=col, lw=1.8, ls="--", label=lab)
    r = {lab: env_xcorr[lab][i] for lab, _, _ in ARMS}
    ax.set_title(f"held-out row {950 + i},  WT = {raw_meta[i, 0]:.2f} in\n"
                 f"$r_x$ = {r['strong, unanchored']:+.2f} / {r['weak, anchored']:+.2f} / "
                 f"{r['weak, unanchored']:+.2f}",
                 fontsize=7.8, color=INK)
    ax.set_xlabel("x (dimensionless pipe axis)")
    ax.grid(axis="y")
    ax.set_xlim(0, 1)
axes[0].set_ylabel("$C_{\\mathrm{wall}}(x)$ at $t = 1$")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=8.5,
           title="per-panel $r_x$ = corr(C_wall pred, C_wall FDM) on the FDM grid, listed strong / weak-anch / weak-unanch",
           title_fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig1_cwall_profiles.{ext}"), bbox_inches="tight")
plt.close(fig)

fig_data["fig1"] = {
    "seed": 42, "checkpoints": {lab: p for lab, p, _ in ARMS},
    "envs": [{"held_out_row": 950 + i, "WT_in": float(raw_meta[i, 0]),
              "per_env_xcorr": {lab: float(env_xcorr[lab][i]) for lab in models}} for i in ENVS],
    "recorded_hold_xcorr": rec_xcorr,
}

# =============================================================================
# FIGURE 2 -- objective vs accuracy at the two measured step counts (Adam phase)
# =============================================================================
STEPS = {"m2_bs64_matched": 7205, "m2_bs64_full": 13200}
recs = {arm: {s: json.load(open(os.path.join(JSON_W, f"{arm}_s{s}.json"))) for s in SEEDS} for arm in STEPS}
for arm, n in STEPS.items():
    for s in SEEDS:
        assert recs[arm][s]["cfg"]["grad_steps"] == n and recs[arm][s]["cfg"]["batch_size"] == 64

PANELS = [("weak_total", "weak", "weak-form objective  (train batch, Adam)", "falls"),
          ("flux_err",   "hold", "wall-flux rel. error  (held-out)",         "rises"),
          ("C_l2",       "hold", "C rel. L2  (held-out)",                     "rises")]
xs = [STEPS["m2_bs64_matched"], STEPS["m2_bs64_full"]]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
f2 = {}
for ax, (k, blk, title, _) in zip(axes, PANELS):
    per_seed = {}
    for s in SEEDS:
        ys = [recs["m2_bs64_matched"][s]["adam"][blk][k], recs["m2_bs64_full"][s]["adam"][blk][k]]
        per_seed[s] = ys
        ax.plot(xs, ys, color=BLUE, alpha=0.35, lw=1.0, marker="o", ms=3.5, mfc="white")
    med = [st.median([per_seed[s][j] for s in SEEDS]) for j in range(2)]
    ax.plot(xs, med, color=BLUE, lw=2.2, marker="o", ms=6, label="median of 5 seeds")
    for xj, mj in zip(xs, med):
        ax.annotate(f"{mj:.3g}", (xj, mj), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:,}" for x in xs])
    ax.set_xlim(xs[0] - 1500, xs[1] + 1500)
    ax.set_xlabel("Adam gradient steps (bs = 64, modes = 2)")
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.grid(axis="y")
    f2[k] = {"per_seed": {str(s): per_seed[s] for s in SEEDS}, "median": med}
axes[0].legend(loc="upper right", fontsize=8)
fig.suptitle("Two measured step counts, five seeds each. Lines only pair the same seed; nothing is measured between the points.",
             fontsize=8, color=INK2, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig2_objective_vs_accuracy.{ext}"), bbox_inches="tight")
plt.close(fig)
fig_data["fig2"] = {"steps": xs, "arms": list(STEPS), "phase": "adam", **f2}

# =============================================================================
# FIGURE 3 -- sorted eigenvalue spectrum of the H1-seminorm Gram matrix (cell 51)
# =============================================================================
gen = ns["generate_true_hp_test_functions"]
V_S2, GRAD_V_S2 = gen(ns["PTS_SPACE"], ns["D_EDGES"], num_modes_x=2)
QW = ns["quad_tensors"][1].to(GRAD_V_S2.dtype)
G = torch.einsum("ipd,p,jpd->ij", GRAD_V_S2, QW, GRAD_V_S2)
G = 0.5 * (G + G.T)
ev = torch.linalg.eigvalsh(G.double()).numpy()
assert G.shape[0] == 36 and (ev > 0).all()
cond = ev.max() / ev.min()
print(f"[fig3] K={G.shape[0]}  eig min {ev.min():.4g}  max {ev.max():.4g}  cond {cond:.4g}")

fig, ax = plt.subplots(figsize=(4.6, 3.2))
idx = np.arange(1, len(ev) + 1)
ax.vlines(idx, ev.min() / 3, ev, color=BLUE, lw=1.0, alpha=0.35)
ax.plot(idx, ev, color=BLUE, lw=0, marker="o", ms=4.5)
ax.set_yscale("log")
ax.set_ylim(ev.min() / 3, ev.max() * 3)
ax.set_xlim(0, len(ev) + 1)
ax.set_xlabel("eigenvalue index (sorted ascending)")
ax.set_ylabel("$\\lambda_i(G)$,  $G_{ij}=\\int \\nabla v_i\\cdot\\nabla v_j$")
ax.grid(axis="y")
for v, s in ((ev.min(), f"$\\lambda_{{\\min}}$ = {ev.min():.3g}"), (ev.max(), f"$\\lambda_{{\\max}}$ = {ev.max():.4g}")):
    ax.axhline(v, color=INK2, lw=0.6, ls=":")
    ax.text(0.6 if v > 1 else 9.0, v * (1.35 if v < 1 else 1 / 1.35), s, fontsize=7.5, color=INK2,
            va="bottom" if v < 1 else "top", ha="left")
ax.set_title(f"K = {len(ev)} test functions (modes = 2 x 12 hats), "
             f"$\\lambda_{{\\max}}/\\lambda_{{\\min}}$ = {cond:,.0f}", fontsize=8.5, color=INK)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig3_gram_spectrum.{ext}"), bbox_inches="tight")
plt.close(fig)
fig_data["fig3"] = {"K": int(G.shape[0]), "eigenvalues_sorted": ev.tolist(),
                    "eig_min": float(ev.min()), "eig_max": float(ev.max()), "condition": float(cond)}

with open(os.path.join(HERE, "fig_data.json"), "w") as f:
    json.dump(fig_data, f, indent=1)
print("saved fig1/fig2/fig3 (.png, .pdf) and fig_data.json in", os.path.relpath(HERE, PINN))
