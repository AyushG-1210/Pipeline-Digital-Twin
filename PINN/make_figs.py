"""
make_paper_figs.py -- every figure, table and per-seed CSV, built from results you already logged.
No GPU, no torch: it only reads JSON. Needs numpy and matplotlib; scikit-learn for the Gaussian-process figure.

USAGE (from the PINN folder):
    pip install numpy matplotlib scikit-learn
    python make_paper_figs.py                       # writes to results/
    python make_paper_figs.py --root . --out results

WRITES (paper_figs inside --out; the other three next to the data they describe)
    paper_figs/       the paper's figures, under the names the tex uses:
                        fig3_heat     warm-start error over time, heat 1D/2D, matching vs clashing IC
                        fig6_sweep    Figure 1: early/late error against IC weight, 1D and 2D (one row, two panels)
                        fig7_gp       Gaussian-process fit of each sweep's gap, location of the minimum
                        fig8_spread   seed spread at t = 1 against the 0.05 match threshold
                      paper_tables.md / paper_tables.tex   every table the paper's numbers come from (check against the tex)
                      per_seed.csv                          every heat, advection-diffusion and ramp run, one row per
                                                            seed and metric, plus every measured k
    heat_runs/heat_figs/        research figures for the heat study (descriptive names, for the project report)
    heat_runs/advdiff_figs/     research figures for the advection-diffusion replication
    electrochem/results/electrochem_figs/  the corrosion figures and table (repo history only; nothing from here enters the paper),
                      electrochem_tables.md / .tex, electrochem_per_seed.csv
Anything missing is reported at the end instead of crashing; a figure that fails is skipped with its reason.

READS  (every heat folder below lives under heat_runs/)
    electrochem/results/results.json    electrochem keys (early_time_3seeds, rework_summary, mechanism_probe, ...)
    heat_results/, gm_heat_results/      heat defaults, IC off, band, dual norm, supervised starts
    heat_matched/                        IC sweeps (1D, 2D), annealing, matched_measure.json (ten-draw k)
    heat_mesh/                           weak form on 4/8/32 elements, strong at the ten-draw k, matched_measure_e*.json
    heat_final/                          normalization and test-function order (weak defaults, strong at ten-draw k)
    heat_k100/                           k with 100 draws (matched_measure.json, _e4/_e8/_e32, _p2/_p4/_p10)
    heat_k100_runs/                      strong form at the 100-draw k (1D sweep points, mesh, order, normalization)
    heat_adv/, heat_adv_k100/            advection-diffusion runs and measured k
    heat_ramp001/, heat_ramp005/         ramped boundary, tau = 0.01 and 0.05 (each with its matched_measure.json)

CHANGES (Oct 15)
    Split into four output folders; electrochem kept for the repo only, with its own CSV and tables.
    fig6_sweep: right column dropped, plain axis labels, weak runs placed at multiplier x k, dashed line at k.
    fig7_gp, fig8_spread computed from the JSON (they were typed numbers in make_gap_figs.py, which this replaces).
    Match threshold 0.05.
    per_seed.csv is now written centrally: every run in every non-electrochem folder, every metric, every measured k.
    New research figures in heat_figs/ and advdiff_figs/.
"""
import os, json, glob, csv, argparse, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

# Okabe-Ito (colour-blind safe)
C = dict(start="#7F7F7F", weak_on="#D55E00", weak_off="#56B4E9", strong="#009E73", strong_off="#7FD3B8",
         dual="#E69F00", warm="#0072B2", cold="#CC79A7", early="#E69F00", late="#0072B2", k="#000000",
         ic="#CC79A7", phys="#0072B2", up="#D55E00", down="#009E73")
SEEDS = (42, 43, 44)
HEAT_T = ("t=0.01", "t=0.05", "t=0.1", "t=0.3", "t=1.0")
TV = np.array([0.01, 0.05, 0.1, 0.3, 1.0])
THRESH = 0.05            # match threshold used in the paper
A_DIFF = 0.1
K_FALLBACK = {"heat1": 2200.0, "heat2": 1.2e5, "adv": 2000.0}
K100_SWEEP = (490, 940, 1300, 1600, 2200, 3400, 7100)  # heat_k100_runs multipliers that join the 1D strong sweep
MESHES, ORDERS = (4, 8, 16, 32), (2, 4, 6, 10)
DEF_NEL, DEF_P = {1: 16, 2: 8}, {1: 6, 2: 4}
# electrochem (repo only)
TAGS = ("0.25/Fo", "1/Fo", "t=1")
TAG_LABEL = {"0.25/Fo": r"$t=0.25/\mathrm{Fo}$", "1/Fo": r"$t=1/\mathrm{Fo}$", "t=1": r"$t=1$"}
STEADY_GUESS = 0.326
LINEAR_MAX_STEP = 3e-4
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 200})

MISSING, PAPER_ROWS, EC_ROWS, EC_LOGGED = [], [], [], set()
GP_RESULTS = {}

def note(msg):
    MISSING.append(msg); print("  [missing] " + msg)

def safe(fn):
    """Run a figure or table builder; on any error, report it and carry on."""
    def wrapped(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            note(f"{fn.__name__} skipped: {type(e).__name__}: {e}")
    wrapped.__name__ = fn.__name__
    return wrapped

def stats(v):
    v = np.asarray(v, float)
    return v.mean(), v.min(), v.max(), len(v)

def cell(v, prec=3):
    if v is None or len(v) == 0:
        return "--"
    m, lo, hi, n = stats(v)
    g = lambda x: f"{x:#.{prec}g}".rstrip(".")
    return f"{g(m)} [{g(lo)}-{g(hi)}]" if n > 1 else f"{g(m)} (n=1)"

def f3(x):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"

def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig); print(f"  wrote {os.path.basename(out)}/{name}.pdf/.png")

# ----------------------------------------------------------------------------- loading
FOLDERS = ("heat_results", "gm_heat_results", "heat_matched", "heat_mesh", "heat_final", "heat_k100_runs",
           "heat_adv", "heat_ramp001", "heat_ramp005")

def load_dir(root, d):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "heat_runs", d, "results", "*.json"))):
        try:
            r = json.load(open(f))
        except Exception as e:
            note(f"unreadable {f}: {e}"); continue
        dim = r.get("dim", 1)
        for k, v in (("band", 0.0), ("dual", False), ("anneal", False), ("norm", False), ("adv", 0.0), ("ramp", 0.0),
                     ("n_el", DEF_NEL.get(dim, 16)), ("p_deg", DEF_P.get(dim, 6))):
            if r.get(k) is None:
                r[k] = v
        r["_src"] = d
        rows.append(r)
    if not rows:
        note(f"no results in {d}/results")
    return rows

def load_measure(root, d, fn="matched_measure.json"):
    p = os.path.join(root, "heat_runs", d, fn)
    if not os.path.exists(p):
        note(f"{d}/{fn} not found"); return {}
    return json.load(open(p))

def kval(M, key="d1_warm", field="k_rounded"):
    v = M.get(key, {}).get(field)
    return float(v) if v is not None else None

def load_electrochem(root):
    p = os.path.join(root, "electrochem", "results", "results.json")
    if not os.path.exists(p):
        note(f"{p} not found (electrochem figures skipped)"); return {}
    return json.load(open(p))

def sel(R, **kw):
    out = []
    for r in R:
        ok = True
        for k, v in kw.items():
            x = r.get(k)
            if isinstance(v, float) and isinstance(x, (int, float)) and not isinstance(x, bool):
                ok = math.isclose(float(x), v, rel_tol=1e-6, abs_tol=1e-12)
            else:
                ok = (x == v)
            if not ok:
                break
        if ok:
            out.append(r)
    return out

def dedupe(rows, key):
    seen, out = set(), []
    for r in rows:
        k = key(r)
        if k not in seen:
            seen.add(k); out.append(r)
    return out

def E(rs):
    return np.array([[r["err"][t] for t in HEAT_T] for r in rs])

def gap(rs, target):
    return float(np.abs(E(rs).mean(0) - E(target).mean(0)).max())

def change_from(rs, start):
    return gap(rs, start)

def by_mult(rs):
    out = {}
    for r in rs:
        out.setdefault(float(r["ic"]), []).append(r)
    return dict(sorted(out.items()))

def nearest(groups, target, tol=0.05):
    """The multiplier group closest to target, if within a relative tol."""
    groups = {g: v for g, v in (groups or {}).items() if g > 0}
    if not groups or not target:
        return None, []
    m = min(groups, key=lambda x: abs(math.log(x / target)))
    return (m, groups[m]) if abs(m / target - 1) <= tol else (None, [])

# ----------------------------------------------------------------------------- data model
class Data:
    def __init__(self, root):
        self.D = {d: load_dir(root, d) for d in FOLDERS}
        self.H = self.D["heat_results"] + self.D["gm_heat_results"]
        heat_all = [r for d in ("heat_results", "gm_heat_results", "heat_matched", "heat_mesh", "heat_final", "heat_k100_runs")
                    for r in self.D[d]]
        self.heat_all = heat_all
        # measured k
        self.M = {
            "m10": load_measure(root, "heat_matched"), "m10_compat": load_measure(root, "heat_matched", "matched_measure_compat.json"),
            "m100": load_measure(root, "heat_k100"),
            "adv10": load_measure(root, "heat_adv"), "adv10_compat": load_measure(root, "heat_adv", "matched_measure_compat.json"),
            "adv100": load_measure(root, "heat_adv_k100"),
            "ramp001": load_measure(root, "heat_ramp001"), "ramp005": load_measure(root, "heat_ramp005"),
        }
        for n in (4, 8, 32):
            self.M[f"mesh10_e{n}"] = load_measure(root, "heat_mesh", f"matched_measure_e{n}.json")
            self.M[f"k100_e{n}"] = load_measure(root, "heat_k100", f"matched_measure_e{n}.json")
        for p in (2, 4, 10):
            self.M[f"final10_p{p}"] = load_measure(root, "heat_final", f"matched_measure_p{p}.json")
            self.M[f"k100_p{p}"] = load_measure(root, "heat_k100", f"matched_measure_p{p}.json")
        for n in MESHES:
            fn = "matched_measure_norm.json" if n == 16 else f"matched_measure_e{n}_norm.json"
            self.M[f"final10_norm_e{n}"] = load_measure(root, "heat_final", fn)
        self.k = {"heat1": kval(self.M["m100"]) or K_FALLBACK["heat1"],
                  "heat2": kval(self.M["m10"], "d2_warm") or K_FALLBACK["heat2"],
                  "adv": kval(self.M["adv100"]) or K_FALLBACK["adv"]}
        # strong runs pooled across heat folders (the strong form does not depend on the weak mesh or order)
        pool = [r for r in heat_all if r["dim"] == 1 and r["case"] == "clash" and r["init"] == "warm" and r["form"] == "strong"
                and not r["anneal"] and r["band"] == 0 and not r.get("dual") and float(r["ic"]) > 0]
        self.strong1 = by_mult(dedupe(pool, lambda r: (f"{float(r['ic']):.6g}", r["seed"])))

    def k_mesh(self, n, draws):
        if draws == 10:
            return kval(self.M["m10"]) if n == 16 else kval(self.M[f"mesh10_e{n}"])
        return kval(self.M["m100"]) if n == 16 else kval(self.M[f"k100_e{n}"])

    def k_order(self, p, draws):
        if p == 6:
            return self.k_mesh(16, draws)
        return kval(self.M[f"final10_p{p}"]) if draws == 10 else kval(self.M[f"k100_p{p}"])

    def k_norm(self, n, draws):
        if draws == 10:
            return kval(self.M[f"final10_norm_e{n}"])
        M = self.M["m100"] if n == 16 else self.M[f"k100_e{n}"]
        v = M.get("d1_warm", {}).get("k_elem_normalized")
        return float(v) if v is not None else None

    def weak1(self, n=16, p=6, norm=False):
        rs = [r for r in self.heat_all if r["dim"] == 1 and r["case"] == "clash" and r["init"] == "warm" and r["form"] == "weak"
              and not r["anneal"] and r["band"] == 0 and not r.get("dual") and float(r["ic"]) == 1.0
              and r["n_el"] == n and r["p_deg"] == p and bool(r["norm"]) == norm]
        return dedupe(rs, lambda r: r["seed"])

    def defaults(self, dim, form, case="clash", init="warm", src=None):
        R = self.H if src is None else self.D[src]
        return sel(R, dim=dim, case=case, init=init, form=form, ic=1.0, band=0.0, dual=False)

    def sweeps(self):
        """{key: dict(label, form, k, mults{m: runs} incl. x1 default, own_default, target)} for the six sweeps."""
        out = {}
        for dim in (1, 2):
            sdef, wdef = self.defaults(dim, "strong"), self.defaults(dim, "weak")
            base = [r for r in self.D["heat_matched"] if r["dim"] == dim and r["case"] == "clash" and r["init"] == "warm"
                    and not r["anneal"] and r["band"] == 0 and float(r["ic"]) not in (0.0, 1.0)]
            extra = [r for r in self.D["heat_k100_runs"] if dim == 1 and r["form"] == "strong" and r["dim"] == 1
                     and any(math.isclose(float(r["ic"]), m) for m in K100_SWEEP)]
            k = self.k["heat1" if dim == 1 else "heat2"]
            for form, own, target in (("strong", sdef, wdef), ("weak", wdef, sdef)):
                rs = dedupe([r for r in base + extra if r["form"] == form], lambda r: (f"{float(r['ic']):.6g}", r["seed"]))
                M = by_mult(rs)
                if own:
                    M = dict(sorted({**M, 1.0: own}.items()))
                out[f"heat{dim}_{form}"] = dict(label=f"{dim}D heat, {form} {'raised' if form == 'strong' else 'lowered'}",
                                                form=form, k=k, mults=M, own=own, target=target, dim=dim)
        A = self.D["heat_adv"]
        sdef, wdef = sel(A, case="clash", init="warm", form="strong", ic=1.0), sel(A, case="clash", init="warm", form="weak", ic=1.0)
        for form, own, target in (("strong", sdef, wdef), ("weak", wdef, sdef)):
            rs = [r for r in A if r["case"] == "clash" and r["init"] == "warm" and r["form"] == form and not r["anneal"]
                  and float(r["ic"]) != 1.0]
            M = by_mult(dedupe(rs, lambda r: (f"{float(r['ic']):.6g}", r["seed"])))
            if own:
                M = dict(sorted({**M, 1.0: own}.items()))
            out[f"adv_{form}"] = dict(label=f"adv.-diff., {form} {'raised' if form == 'strong' else 'lowered'}",
                                      form=form, k=self.k["adv"], mults=M, own=own, target=target, dim=1)
        return out

def xk(form, m, k):
    """Effective weight in units of k: strong m/k, weak m*k."""
    return m / k if form == "strong" else m * k

# ----------------------------------------------------------------------------- per-seed CSV (paper)
def log_all(data):
    for d, rows in data.D.items():
        for r in rows:
            arm = r.get("name", "?").rsplit("_s", 1)[0]
            for t in HEAT_T:
                if t in r.get("err", {}):
                    PAPER_ROWS.append((d, arm, r.get("seed"), f"err {t}", r["err"][t]))
                if t in (r.get("err_abs") or {}):
                    PAPER_ROWS.append((d, arm, r.get("seed"), f"err_abs {t}", r["err_abs"][t]))
            for k, v in (r.get("terms") or {}).items():
                PAPER_ROWS.append((d, arm, r.get("seed"), f"term {k}", v))
            if r.get("ic_weight_final") is not None:
                PAPER_ROWS.append((d, arm, r.get("seed"), "ic_weight_final", r["ic_weight_final"]))
    for name, M in data.M.items():
        for key, blk in M.items():
            for s, row in (blk.get("per_seed") or {}).items():
                for k, v in row.items():
                    PAPER_ROWS.append(("measure:" + name, key, s, k, v))
            for k in ("k", "k_rounded", "k_elem_normalized"):
                if k in blk:
                    PAPER_ROWS.append(("measure:" + name, key, "", k, blk[k]))

# ----------------------------------------------------------------------------- paper figures
@safe
def fig_heat(data, out):
    H = data.H
    if not H:
        return
    lines = [("Weak, IC on", dict(form="weak", ic=1.0, dual=False), C["weak_on"], "-"),
             ("Weak (dual norm), IC on", dict(form="weak", ic=1.0, dual=True), C["dual"], "--"),
             ("Strong, IC on", dict(form="strong", ic=1.0, dual=False), C["strong"], "-"),
             ("Weak, IC off", dict(form="weak", ic=0.0, dual=False), C["weak_off"], ":")]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4), sharex=True)
    for i, dim in enumerate((1, 2)):
        for j, case in enumerate(("compat", "clash")):
            ax = axes[i, j]
            for lab, s, col, ls in lines:
                rs = sel(H, dim=dim, case=case, init="warm", band=0.0, **s)
                if not rs:
                    continue
                Ev = np.maximum(E(rs), 1e-4)
                ax.plot(TV, Ev.mean(0), ls, color=col, lw=1.4, label=lab)
                ax.fill_between(TV, Ev.min(0), Ev.max(0), color=col, alpha=0.18, lw=0)
            st = sel(H, dim=dim, case=case, init="pre")
            if st:
                ax.plot(TV, np.maximum(E(st), 1e-4).mean(0), color=C["start"], lw=1, label="Supervised start")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 2)
            ax.set_title(f"{dim}D, {'IC matches boundary' if case == 'compat' else 'IC clashes with boundary'}")
            if j == 0: ax.set_ylabel("rel. error of the transient")
            if i == 1: ax.set_xlabel("time $t$")
    h, l = axes[1, 1].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.text(0.99, 0.0, "warm start; lines: mean of 3 seeds; bands: min-max", ha="right", fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, out, "fig3_heat")

@safe
def fig_sweep(data, out, SW):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for ax, dim in zip(axes, (1, 2)):
        k = data.k["heat1" if dim == 1 else "heat2"]
        drew = False
        for form in ("strong", "weak"):
            s = SW.get(f"heat{dim}_{form}")
            if not s or len(s["mults"]) < 2:
                continue
            ms = np.array(list(s["mults"])); Em = {m: E(rs) for m, rs in s["mults"].items()}
            x = ms * (k if form == "weak" else 1.0)
            o = np.argsort(x)
            ls, mfc = ("-", None) if form == "strong" else ("--", "white")
            xs = x[o]
            # break the line where no runs lie in between (gaps over two decades, e.g. 2D strong: x1 then x1e4)
            cuts = [0] + [i + 1 for i in range(len(xs) - 1) if xs[i + 1] / xs[i] > 100] + [len(xs)]
            for t, col in (("t=0.01", C["early"]), ("t=1.0", C["late"])):
                j = HEAT_T.index(t)
                mean = np.array([Em[m][:, j].mean() for m in ms])[o]
                lo = np.array([Em[m][:, j].min() for m in ms])[o]; hi = np.array([Em[m][:, j].max() for m in ms])[o]
                for a0, a1 in zip(cuts[:-1], cuts[1:]):
                    seg = slice(a0, a1)
                    ax.plot(xs[seg], mean[seg], ls, marker="o", ms=3.2, lw=1.2, color=col, mfc=mfc or col)
                    if a1 - a0 > 1:
                        ax.fill_between(xs[seg], lo[seg], hi[seg], color=col, alpha=0.15, lw=0)
            drew = True
        if not drew:
            note(f"fig6_sweep: no sweep runs for {dim}D")
        ax.axvline(k, color="0.3", ls="--", lw=0.8)
        ax.text(k, 0.97, "$k$", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        ax.set(xscale="log", yscale="log", title=f"{dim}D heat",
               xlabel="IC weight relative to the strong form's default\n(weak-form runs placed at multiplier $\\times\\,k$)")
        if dim == 1:
            ax.set_ylabel("relative error of the transient")
    handles = [Line2D([], [], color=C["early"], lw=1.4, label="early ($t=0.01$)"),
               Line2D([], [], color=C["late"], lw=1.4, label="late ($t=1$)"),
               Line2D([], [], color="black", marker="o", ms=4, lw=1.2, label="strong form, weight raised"),
               Line2D([], [], color="black", marker="o", ms=4, lw=1.2, ls="--", mfc="white", label="weak form, weight lowered")]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.07), fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig6_sweep")

def gp_fit(xlog, ylog, grid, rng):
    from sklearn.gaussian_process import GaussianProcessRegressor as GPR
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel
    import warnings
    warnings.filterwarnings("ignore")
    gp = GPR(Ck(1.0, (0.1, 10)) * RBF(0.8, (0.3, 3.0)) + WhiteKernel(0.04, (0.01, 0.25)),
             normalize_y=True, n_restarts_optimizer=8, random_state=0).fit(xlog[:, None], ylog)
    mu, cov = gp.predict(grid[:, None], return_cov=True)
    cov = (cov + cov.T) / 2 + 1e-8 * np.eye(len(grid))
    f = rng.multivariate_normal(mu, cov, size=3000)
    return mu, np.sqrt(np.diag(cov)), grid[f.argmin(1)]

def sweep_gaps(s):
    """(multipliers, gaps) excluding the x1 default; gap to the other form's default."""
    if not s["target"]:
        return np.array([]), np.array([])
    ms = [m for m in s["mults"] if m != 1.0]
    return np.array(ms), np.array([gap(s["mults"][m], s["target"]) for m in ms])

@safe
def fig_gp(data, out, SW):
    rng = np.random.default_rng(0)
    grid = np.linspace(-1.6, 1.6, 641)
    order = ["heat1_strong", "heat1_weak", "heat2_strong", "heat2_weak", "adv_strong", "adv_weak"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharey=True); axes = axes.ravel()
    for ax, key in zip(axes, order):
        s = SW.get(key)
        ms, g = sweep_gaps(s) if s else (np.array([]), np.array([]))
        if len(ms) < 3:
            ax.set_visible(False); note(f"fig7_gp: too few points for {key}"); continue
        xall = np.log10(xk(s["form"], ms, s["k"]))
        keep = (xall >= np.log10(0.03)) & (xall <= np.log10(40))
        x, gg = xall[keep], g[keep]
        mu, sd, xm = gp_fit(x, np.log10(gg), grid, rng)
        lo, med, hi = np.percentile(10 ** xm, [5, 50, 95])
        d0 = gap(s["own"], s["target"]) if s["own"] else float("nan")
        GP_RESULTS[key] = dict(label=s["label"], median=med, lo=lo, hi=hi, p3=float(np.mean(np.abs(xm) <= np.log10(3))),
                               ratio=d0 / g.min(), n=int(keep.sum()), k=s["k"])
        ax.fill_between(10 ** grid, 10 ** (mu - 1.64 * sd), 10 ** (mu + 1.64 * sd), color=C["late"], alpha=.2, lw=0)
        ax.plot(10 ** grid, 10 ** mu, color=C["late"], lw=1.3)
        ax.plot(10 ** xall, g, "o", color="k", ms=4, mfc="none"); ax.plot(10 ** x, gg, "o", color="k", ms=4)
        ax.axvline(1, color=".5", lw=.7); ax.axhline(THRESH, color="k", ls=":", lw=.9)
        ax.axvspan(lo, hi, ymin=0, ymax=.06, color=C["up"], alpha=.9)
        ax.set(xscale="log", yscale="log", ylim=(.003, 1), xlim=(.02, 50), title=s["label"], xlabel="effective weight / $k$")
    for ax in axes[::3]:
        ax.set_ylabel("gap to the other form's default")
    fig.tight_layout()
    save(fig, out, "fig7_gp")
    for key, r in GP_RESULTS.items():
        for k, v in r.items():
            if k != "label":
                PAPER_ROWS.append(("gp_fit", key, "", k, v))

@safe
def fig_spread(data, out):
    rows = []
    for dim in (1, 2):
        for init in ("warm", "cold"):
            for lab, s in (("weak", dict(form="weak", dual=False)), ("dual norm", dict(form="weak", dual=True)),
                           ("strong", dict(form="strong", dual=False))):
                rs = sel(data.H, dim=dim, case="clash", init=init, ic=1.0, band=0.0, **s)
                if rs:
                    v = [r["err"]["t=1.0"] for r in rs]
                    rows.append((f"{dim}D {init}, {lab}", max(v) - min(v)))
    if not rows:
        note("fig8_spread: no heat default runs"); return
    fig, ax = plt.subplots(figsize=(6.4, 3.8)); y = np.arange(len(rows))
    rg = np.array([r[1] for r in rows])
    ax.barh(y + .2, rg, .38, color="#888", label="seed range (max $-$ min), three seeds")
    ax.barh(y - .2, rg / 2.93, .38, color=C["late"], label="estimated standard error of a three-seed mean")
    ax.axvline(THRESH, color="k", lw=.9); ax.text(THRESH * 1.08, len(rows) - .6, f"{THRESH:g}", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7); ax.invert_yaxis()
    ax.set(xscale="log", xlabel="error at $t=1$")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    save(fig, out, "fig8_spread")

# ----------------------------------------------------------------------------- paper tables
class Tables:
    def __init__(self):
        self.md, self.tex = [], []
    def emit(self, title, header, rows):
        if not rows:
            note(f"table skipped (no rows): {title}"); return
        self.md.append(f"\n### {title}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
        self.md.extend("| " + " | ".join(map(str, r)) + " |" for r in rows)
        self.tex.append(f"% {title}\n\\begin{{tabular}}{{l" + "c" * (len(header) - 1) + "}\n\\toprule\n"
                        + " & ".join(header) + " \\\\\n\\midrule")
        self.tex.extend(" & ".join(str(x).replace("[", "{\\scriptsize[").replace("]", "]}") for x in r) + " \\\\" for r in rows)
        self.tex.append("\\bottomrule\n\\end{tabular}\n")
    def write(self, out, stem):
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, f"{stem}.md"), "w").write("\n".join(self.md) + "\n")
        open(os.path.join(out, f"{stem}.tex"), "w").write("\n".join(self.tex) + "\n")
        print(f"  wrote {os.path.basename(out)}/{stem}.md / .tex")

def transient_rms(dim, case, t):
    """RMS of u - 1 on the evaluation grid of heat_drift.report (201 points in 1D, 41 x 41 in 2D)."""
    g = np.linspace(0, 1, 201 if dim == 1 else 41)
    if case == "compat":
        s = np.sin(np.pi * g); prof = s if dim == 1 else np.outer(s, s)
        v = prof * np.exp(-dim * A_DIFF * np.pi ** 2 * t)
    else:
        n = np.arange(1, 800, 2)[:, None]
        f = (4 / (np.pi * n) * np.sin(np.pi * n * g[None, :]) * np.exp(-A_DIFF * np.pi ** 2 * n ** 2 * t)).sum(0)
        v = f if dim == 1 else np.outer(f, f)
    return float(np.sqrt(np.mean(v ** 2)))

@safe
def t_dual(data, T):
    rows = []
    for dim in (1, 2):
        for init in ("warm", "cold"):
            r = [f"{dim}D, {init}"]
            for s in (dict(form="weak", dual=False), dict(form="weak", dual=True), dict(form="strong", dual=False)):
                rs = sel(data.H, dim=dim, case="clash", init=init, ic=1.0, band=0.0, **s)
                r.append(cell([x["err"]["t=1.0"] for x in rs]) if rs else "--")
            rows.append(r)
    T.emit("Heat, clashing IC: error at t=1 (mean [min-max], 3 seeds)", ["Problem, start", "Weak (plain)", "Weak (dual norm)", "Strong"], rows)

@safe
def t_giveup(data, T):
    rows = []
    for lab, s in (("Supervised start", dict(init="pre")), ("Weak", dict(init="warm", form="weak", ic=1.0, band=0.0, dual=False)),
                   ("Weak (dual norm)", dict(init="warm", form="weak", ic=1.0, band=0.0, dual=True)),
                   ("Strong", dict(init="warm", form="strong", ic=1.0, band=0.0, dual=False))):
        for dim in (1, 2):
            rs = sel(data.H, dim=dim, case="clash", **s)
            if rs:
                rows.append([f"{dim}D, {lab}", cell([x["terms"]["ic"] for x in rs]), cell([x["terms"]["strong"] for x in rs]),
                             cell([x["err"]["t=0.01"] for x in rs]), cell([x["err"]["t=1.0"] for x in rs])])
    T.emit("Heat, clashing IC, warm start: what each form gives up", ["Model", "IC loss", "pointwise PDE residual", "error t=0.01", "error t=1"], rows)

@safe
def t_sweeps(data, T, SW):
    for key, s in SW.items():
        if not s["target"] or len(s["mults"]) < 2:
            continue
        rows = [[f"{'Weak' if s['form'] == 'strong' else 'Strong'}, default (target)"] + [cell([r["err"][t] for r in s["target"]]) for t in HEAT_T] + ["--"] * 3]
        for m, rs in s["mults"].items():
            rows.append([f"{s['form'].capitalize()}, x{m:g}"] + [cell([r["err"][t] for r in rs]) for t in HEAT_T]
                        + [cell([r["terms"]["ic"] for r in rs]), cell([r["terms"]["strong"] for r in rs]), f3(gap(rs, s["target"]))])
        T.emit(f"{s['label']}: IC weight sweep, clashing IC, warm start (k for the axis = {s['k']:g})",
               ["Arm"] + [f"error {t}" for t in HEAT_T] + ["IC loss", "pointwise PDE residual", "gap to other default"], rows)

@safe
def t_anneal(data, T):
    AN = [r for r in data.D["heat_matched"] if r["anneal"]]
    for init in ("warm", "cold"):
        rows = []
        for dim in (1, 2):
            for case in ("compat", "clash"):
                for form in ("weak", "strong"):
                    rs = [r for r in AN if r["dim"] == dim and r["case"] == case and r["init"] == init and r["form"] == form]
                    if not rs:
                        continue
                    d = sel(data.H, dim=dim, case=case, init=init, form=form, ic=1.0, band=0.0, dual=False)
                    rows.append([f"{dim}D, {'matching' if case == 'compat' else 'clashing'}, {form}"]
                                + [cell([r["err"][t] for r in rs]) for t in HEAT_T]
                                + [cell([r["ic_weight_final"] for r in rs]), f3(change_from(rs, d)) if d else "--"])
        T.emit(f"Heat, annealed IC weight (Wang et al. 2021), {init} start", ["Problem, IC, form"] + [f"error {t}" for t in HEAT_T]
               + ["final IC weight", "max change vs default"], rows)

def match_row(data, label, k10, k100, weak):
    """[label, k10, gap10, k100, gap100] for the weak runs against the strong form at each k."""
    out = [label]
    for k in (k10, k100):
        m, st = nearest(data.strong1, k)
        out += [f"{k:.3g}" if k else "--", f3(gap(st, weak)) if (st and weak) else "--"]
    return out

@safe
def t_mesh_order_norm(data, T):
    rows, spread = [], {}
    for n in MESHES:
        w = data.weak1(n=n)
        rows.append(match_row(data, f"{n} elements", data.k_mesh(n, 10), data.k_mesh(n, 100), w))
    T.emit("Heat 1D, mesh check: weak default on N elements vs strong at x k(N)", ["Mesh", "k, 10 draws", "gap", "k, 100 draws", "gap"], rows)
    rows = []
    for p in ORDERS:
        w = data.weak1(p=p)
        rows.append(match_row(data, f"p = {p}", data.k_order(p, 10), data.k_order(p, 100), w))
    T.emit("Heat 1D, test-function order (16 elements): weak default vs strong at x k(p)", ["Order", "k, 10 draws", "gap", "k, 100 draws", "gap"], rows)
    rows = []
    for n in MESHES:
        w = data.weak1(n=n, norm=True)
        rows.append(match_row(data, f"{n} elements", data.k_norm(n, 10), data.k_norm(n, 100), w))
    T.emit("Heat 1D, normalized weak residual: weak default vs strong at x k_norm", ["Mesh", "k_norm, 10 draws", "gap", "k_norm, 100 draws", "gap"], rows)
    for norm in (False, True):
        Ms = [E(data.weak1(n=n, norm=norm)).mean(0) for n in MESHES if data.weak1(n=n, norm=norm)]
        if len(Ms) == 4:
            spread["normalized" if norm else "plain"] = float((np.max(Ms, 0) - np.min(Ms, 0)).max())
    T.emit("Heat 1D: largest spread of the weak default over the four meshes at any time",
           ["Weak residual", "spread"], [[k, f3(v)] for k, v in spread.items()])
    rows = []
    for n in MESHES:
        k10, k100 = data.k_mesh(n, 10), data.k_mesh(n, 100)
        rows.append([f"{n} elements, p = 6", f"{k10:.3g}" if k10 else "--", f"{k100:.3g}" if k100 else "--",
                     f"{k100 / k10:.2f}" if (k10 and k100) else "--",
                     f"{data.k_norm(n, 10):.3g}" if data.k_norm(n, 10) else "--", f"{data.k_norm(n, 100):.3g}" if data.k_norm(n, 100) else "--"])
    for p in (2, 4, 10):
        k10, k100 = data.k_order(p, 10), data.k_order(p, 100)
        rows.append([f"16 elements, p = {p}", f"{k10:.3g}" if k10 else "--", f"{k100:.3g}" if k100 else "--",
                     f"{k100 / k10:.2f}" if (k10 and k100) else "--", "--", "--"])
    T.emit("Heat 1D: k with ten and with 100 draws", ["Setting", "k, 10", "k, 100", "ratio", "k_norm, 10", "k_norm, 100"], rows)

@safe
def t_adv(data, T, SW):
    A = data.D["heat_adv"]
    rows = []
    for case in ("clash", "compat"):
        start = sel(A, case=case, init="pre")
        for lab, rs in (("start", start), ("weak, default", sel(A, case=case, init="warm", form="weak", ic=1.0)),
                        ("strong, default", sel(A, case=case, init="warm", form="strong", ic=1.0))):
            if rs:
                rows.append([f"{case}, {lab}"] + [cell([r["err"][t] for r in rs]) for t in HEAT_T]
                            + [f3(change_from(rs, start)) if (start and lab != "start") else "--"])
        if case == "compat":
            for m, rs in by_mult(sel(A, case="compat", init="warm", form="strong")).items():
                if m != 1.0:
                    rows.append([f"compat, strong x{m:g}"] + [cell([r["err"][t] for r in rs]) for t in HEAT_T] + [f3(change_from(rs, start)) if start else "--"])
    T.emit("Advection-diffusion: defaults and change from the supervised start", ["Arm"] + [f"error {t}" for t in HEAT_T] + ["change from start"], rows)
    ks = [["clash, 10 draws", f"{kval(data.M['adv10']) or float('nan'):.4g}"], ["compat, 10 draws", f"{kval(data.M['adv10_compat']) or float('nan'):.4g}"],
          ["clash, 100 draws", f"{kval(data.M['adv100']) or float('nan'):.4g}"]]
    T.emit("Advection-diffusion: measured k", ["Setting", "k"], ks)

@safe
def t_ramp(data, T):
    rows = []
    for tau, d, mkey in ((0.05, "heat_ramp005", "ramp005"), (0.01, "heat_ramp001", "ramp001")):
        R = data.D[d]; start = sel(R, init="pre")
        k = kval(data.M[mkey])
        arms = [("supervised start", start), ("weak, default", sel(R, init="warm", form="weak", ic=1.0)),
                ("strong, default", sel(R, init="warm", form="strong", ic=1.0))]
        if k:
            arms.append((f"strong, x{k:g}", sel(R, init="warm", form="strong", ic=k)))
        for lab, rs in arms:
            if rs:
                rows.append([f"tau = {tau:g}, {lab}"] + [cell([r["err"][t] for r in rs]) for t in HEAT_T]
                            + [cell([r["terms"]["ic"] for r in rs]), f3(change_from(rs, start)) if (start and lab != "supervised start") else "--"])
    T.emit("Ramped boundary, 1D heat, warm start", ["Arm"] + [f"error {t}" for t in HEAT_T] + ["IC loss", "change from start"], rows)

@safe
def t_gp(T):
    rows = [[r["label"], f"{r['median']:.2f} [{r['lo']:.2f}-{r['hi']:.2f}]", f"{r['p3']:.2f}", f"{r['ratio']:.1f}", r["n"]]
            for r in GP_RESULTS.values()]
    T.emit("Location of the minimum gap (Gaussian-process fit, Figure fig7_gp)",
           ["Sweep", "minimum at (x k), median [90%]", "P(within 3x of k)", "default gap / best observed", "points fitted"], rows)

@safe
def t_abs(data, T):
    rows = []
    for dim in (1, 2):
        for case in ("clash", "compat"):
            rms = [transient_rms(dim, case, t) for t in TV]
            rows.append([f"{dim}D {case}: RMS of u - 1"] + [f"{v:.4f}" for v in rms])
            for form in ("weak", "strong"):
                rs = sel(data.H, dim=dim, case=case, init="warm", form=form, ic=1.0, band=0.0, dual=False)
                if rs:
                    rows.append([f"{dim}D {case}, {form}, warm"] + [f"{E(rs).mean(0)[j] * rms[j]:.4f}" for j in range(5)])
    T.emit("Heat, absolute (RMS) error = relative error x RMS of u - 1, mean over seeds", ["Arm"] + list(HEAT_T), rows)

# ----------------------------------------------------------------------------- heat research figures
def _err_band(ax, rs, col, ls="-", lab=None, lw=1.3):
    Ev = np.maximum(E(rs), 1e-4)
    ax.plot(TV, Ev.mean(0), ls, color=col, lw=lw, label=lab)
    ax.fill_between(TV, Ev.min(0), Ev.max(0), color=col, alpha=.15, lw=0)

@safe
def heat_error_over_time_all_arms(data, out):
    arms = [("weak", dict(form="weak", ic=1.0, dual=False), C["weak_on"], "-"), ("weak, dual norm", dict(form="weak", ic=1.0, dual=True), C["dual"], "--"),
            ("strong", dict(form="strong", ic=1.0, dual=False), C["strong"], "-"), ("weak, IC off", dict(form="weak", ic=0.0, dual=False), C["weak_off"], ":"),
            ("strong, IC off", dict(form="strong", ic=0.0, dual=False), C["strong_off"], ":")]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True, sharey=True)
    for i, dim in enumerate((1, 2)):
        for j, (case, init) in enumerate((("compat", "warm"), ("compat", "cold"), ("clash", "warm"), ("clash", "cold"))):
            ax = axes[i, j]
            for lab, s, col, ls in arms:
                rs = sel(data.H, dim=dim, case=case, init=init, band=0.0, **s)
                if rs: _err_band(ax, rs, col, ls, lab)
            st = sel(data.H, dim=dim, case=case, init="pre")
            if st: ax.plot(TV, np.maximum(E(st), 1e-4).mean(0), color=C["start"], lw=1, label="supervised start")
            ax.set(xscale="log", yscale="log", ylim=(1e-4, 3), title=f"{dim}D, {case} IC, {init} start")
            if i == 1: ax.set_xlabel("time $t$")
            if j == 0: ax.set_ylabel("relative error of the transient")
    h, l = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l: h.append(hh); l.append(ll)
    fig.legend(h, l, ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "heat_error_over_time_every_default_arm")

@safe
def heat_per_seed_boxplots(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for i, dim in enumerate((1, 2)):
        for j, t in enumerate(("t=0.01", "t=1.0")):
            ax = axes[i, j]; labs, vals = [], []
            for init in ("warm", "cold"):
                for lab, s in (("weak", dict(form="weak", dual=False, ic=1.0)), ("dual", dict(form="weak", dual=True, ic=1.0)),
                               ("strong", dict(form="strong", dual=False, ic=1.0)), ("weak IC off", dict(form="weak", dual=False, ic=0.0))):
                    rs = sel(data.H, dim=dim, case="clash", init=init, band=0.0, **s)
                    if rs: labs.append(f"{init}\n{lab}"); vals.append([r["err"][t] for r in rs])
            if not vals: continue
            ax.boxplot(vals, widths=.5, showfliers=False)
            for k, v in enumerate(vals):
                ax.plot(np.full(len(v), k + 1) + np.linspace(-.08, .08, len(v)), v, "o", ms=4, color="k", alpha=.7)
            ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs, fontsize=7)
            ax.set(yscale="log", title=f"{dim}D heat, clashing IC, error at {t}", ylabel="relative error")
            ax.axhline(THRESH, color="k", ls=":", lw=.8)
    fig.tight_layout()
    save(fig, out, "heat_per_seed_error_boxplots")

def _waterfall(ax, start_terms, end_terms, title):
    """Start objective -> change in IC term -> change in physics term -> end objective."""
    s, e = start_terms, end_terms
    steps = [("start", s[0] + s[1], None), ("IC term", e[0] - s[0], "ic"), ("physics term", e[1] - s[1], "phys"), ("end", e[0] + e[1], None)]
    level = 0.0
    for i, (lab, v, kind) in enumerate(steps):
        if kind is None:
            ax.bar(i, v, color="0.6", width=.6); level = v
        else:
            ax.bar(i, v, bottom=level, color=C["up"] if v > 0 else C["down"], width=.6); level += v
        ax.text(i, max(level, v if kind is None else level) * 1.02, f"{v:+.3g}" if kind else f"{v:.3g}", ha="center", va="bottom", fontsize=7)
    top = max(s[0] + s[1], e[0] + e[1], s[0] + s[1] + max(e[0] - s[0], 0), level)
    ax.set_ylim(0, top * 1.18)
    ax.set_xticks(range(4)); ax.set_xticklabels([x[0] for x in steps], fontsize=7)
    ax.set_title(title, fontsize=8); ax.set_ylabel("weighted loss term")

@safe
def heat_objective_waterfalls(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for i, dim in enumerate((1, 2)):
        st = sel(data.H, dim=dim, case="clash", init="pre")
        for j, form in enumerate(("weak", "strong")):
            rs = sel(data.H, dim=dim, case="clash", init="warm", form=form, ic=1.0, band=0.0, dual=False)
            if not (st and rs):
                axes[i, j].set_visible(False); continue
            mt = lambda R, k: float(np.mean([r["terms"][k] for r in R]))
            _waterfall(axes[i, j], (mt(st, "ic"), mt(st, form)), (mt(rs, "ic"), mt(rs, form)),
                       f"{dim}D heat, {form} form, clashing IC: its own objective from start to end")
    fig.tight_layout()
    save(fig, out, "heat_objective_waterfall_start_to_end")

def _tug(ax, M, keys, title):
    labs, rows = [], []
    for lab, key in keys:
        ps = (M.get(key) or {}).get("per_seed") or {}
        if ps:
            labs.append(lab); rows.append(np.array([[v["ic"], v["phys_weak"], v["phys_strong"]] for v in ps.values()]))
    if not rows:
        ax.set_visible(False); return False
    x = np.arange(len(labs)); w = .26
    for j, (name, col) in enumerate((("IC term", C["ic"]), ("physics, weak", C["weak_on"]), ("physics, strong", C["strong"]))):
        means = [np.exp(np.log(r[:, j]).mean()) for r in rows]
        ax.bar(x + (j - 1) * w, means, w, color=col, label=name)
        for i, r in enumerate(rows):
            ax.plot(np.full(len(r), x[i] + (j - 1) * w), r[:, j], "k.", ms=3)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7); ax.set_yscale("log")
    ax.set_ylabel("gradient norm (weighted term)"); ax.set_title(title, fontsize=8); ax.legend(fontsize=7, frameon=False)
    return True

@safe
def heat_gradient_tug_of_war(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _tug(axes[0], {**{f"10 {k}": v for k, v in data.M["m10"].items()}, **{f"100 {k}": v for k, v in data.M["m100"].items()},
                   **{f"compat {k}": v for k, v in data.M["m10_compat"].items()}},
         [("1D clash\n10 draws", "10 d1_warm"), ("1D clash\n100 draws", "100 d1_warm"), ("1D matching\n10 draws", "compat d1_warm"),
          ("2D clash\n10 draws", "10 d2_warm"), ("2D matching\n10 draws", "compat d2_warm")],
         "Heat, supervised start: IC pull against the physics pull (bars: geometric mean; dots: seeds)")
    keys = [(f"{n} el.", "d1_warm") for n in MESHES]
    Ms = [data.M["k100_e4"], data.M["k100_e8"], data.M["m100"], data.M["k100_e32"]]
    merged = {f"{n} el.": M.get("d1_warm") for n, M in zip(MESHES, Ms) if M.get("d1_warm")}
    _tug(axes[1], merged, [(k, k) for k in merged], "Heat 1D, 100 draws: the weak physics pull shrinks as elements shrink")
    fig.tight_layout()
    save(fig, out, "heat_gradient_tug_of_war_ic_vs_physics")

@safe
def heat_k_vs_element_size_and_order(data, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for draws, mk, col in ((10, "o", C["weak_off"]), (100, "s", C["late"])):
        ns = [n for n in MESHES if data.k_mesh(n, draws)]
        ks = [data.k_mesh(n, draws) for n in ns]
        if len(ns) >= 2:
            h = 1 / np.array(ns, float); slope = np.polyfit(np.log(h), np.log(ks), 1)[0]
            a.plot(h, ks, mk + "-", color=col, label=f"{draws} draws: k ~ h^{slope:.2f}")
        ps = [p for p in ORDERS if data.k_order(p, draws)]
        if ps:
            b.plot(ps, [data.k_order(p, draws) for p in ps], mk + "-", color=col, label=f"{draws} draws")
    a.set(xscale="log", yscale="log", xlabel="element width $h$", ylabel="$k$", title="k against element size (p = 6)"); a.legend(frameon=False, fontsize=7)
    b.set(yscale="log", xlabel="test-function order $p$", ylabel="$k$", title="k against test-function order (16 elements)"); b.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, out, "heat_k_against_element_size_and_order")

def _gap_heatmap(ax, s):
    if not s["target"]:
        ax.set_visible(False); return
    ms = list(s["mults"]); tgt = E(s["target"]).mean(0)
    G = np.array([np.abs(E(s["mults"][m]).mean(0) - tgt) for m in ms])
    im = ax.imshow(G, aspect="auto", cmap="viridis", vmin=0, vmax=max(G.max(), THRESH))
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            ax.text(j, i, f"{G[i, j]:.3f}", ha="center", va="center", fontsize=6, color="w" if G[i, j] < G.max() * .6 else "k")
    ax.set_xticks(range(5)); ax.set_xticklabels([t[2:] for t in HEAT_T]); ax.set_xlabel("time $t$")
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([f"x{m:g}" for m in ms], fontsize=7)
    ax.set_title(f"{s['label']}: |error - other form's default|", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=.04)

@safe
def heat_sweep_gap_heatmaps(data, out, SW):
    keys = [k for k in ("heat1_strong", "heat1_weak", "heat2_strong", "heat2_weak") if k in SW]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 5.5))
    for ax, k in zip(np.atleast_1d(axes), keys):
        _gap_heatmap(ax, SW[k])
    fig.tight_layout()
    save(fig, out, "heat_sweep_gap_heatmap_by_multiplier_and_time")

def _gap_vs_mult(ax, s, title):
    ms, g = sweep_gaps(s)
    if len(ms) == 0:
        ax.set_visible(False); return
    ax.plot(ms, g, "o-" if s["form"] == "strong" else "s--", color="k", mfc="k" if s["form"] == "strong" else "w", label=s["label"])
    ax.axhline(THRESH, color=".4", ls=":"); ax.axvline(s["k"] if s["form"] == "strong" else 1 / s["k"], color=".3", ls="--", lw=.8)
    ax.set(xscale="log", yscale="log", xlabel="IC weight / own default", ylabel="worst gap to the other form's default", title=title)

@safe
def heat_sweep_gap_vs_multiplier(data, out, SW):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, k in zip(axes.ravel(), ("heat1_strong", "heat1_weak", "heat2_strong", "heat2_weak")):
        if k in SW: _gap_vs_mult(ax, SW[k], SW[k]["label"] + " (dashed: k or 1/k; dotted: 0.05)")
    fig.tight_layout()
    save(fig, out, "heat_sweep_gap_against_multiplier")

@safe
def heat_annealed_weight_trajectories(data, out):
    AN = [r for r in data.D["heat_matched"] if r["anneal"] and r.get("ic_weight_hist")]
    if not AN:
        note("no annealing histories"); return
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for i, dim in enumerate((1, 2)):
        for j, case in enumerate(("compat", "clash")):
            ax = axes[i, j]
            for r in AN:
                if r["dim"] != dim or r["case"] != case: continue
                h = np.array(r["ic_weight_hist"], float)
                ax.plot(h[:, 0], h[:, 1] / 3.0, "-" if r["init"] == "warm" else "--", color=C["weak_on"] if r["form"] == "weak" else C["strong"], lw=1, alpha=.8)
            ax.set(yscale="log", title=f"{dim}D heat, {'matching' if case == 'compat' else 'clashing'} IC", ylabel="annealed IC weight / default")
            if i == 1: ax.set_xlabel("training step")
    fig.legend(handles=[Line2D([], [], color=C["weak_on"], label="weak"), Line2D([], [], color=C["strong"], label="strong"),
                        Line2D([], [], color="k", label="warm"), Line2D([], [], color="k", ls="--", label="cold")],
               ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, out, "heat_annealed_ic_weight_over_training")

@safe
def heat_mesh_order_norm_matches(data, out):
    sets = [("mesh", [(f"{n} elements", data.weak1(n=n), data.k_mesh(n, 100)) for n in MESHES]),
            ("order", [(f"p = {p}", data.weak1(p=p), data.k_order(p, 100)) for p in ORDERS]),
            ("normalized", [(f"{n} el., normalized", data.weak1(n=n, norm=True), data.k_norm(n, 100)) for n in MESHES])]
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), sharey=True)
    for i, (_, items) in enumerate(sets):
        for j, (lab, w, k) in enumerate(items):
            ax = axes[i, j]; m, st = nearest(data.strong1, k)
            if not w or not st:
                ax.set_visible(False); continue
            _err_band(ax, w, C["weak_on"], "-", "weak, default"); _err_band(ax, st, C["strong"], "--", f"strong, x{m:g}")
            ax.set(xscale="log", yscale="log", title=f"{lab}: gap {gap(st, w):.3f}")
            ax.legend(fontsize=6, frameon=False)
            if j == 0: ax.set_ylabel("relative error")
    fig.suptitle("Weak default against the strong form at the 100-draw k, 1D heat, clashing IC, warm start", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .97))
    save(fig, out, "heat_weak_default_vs_strong_at_k_mesh_order_normalization")

@safe
def heat_ramp_error_over_time(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (tau, d, mkey) in zip(axes, ((0.05, "heat_ramp005", "ramp005"), (0.01, "heat_ramp001", "ramp001"))):
        R = data.D[d]; k = kval(data.M[mkey])
        arms = [("supervised start", sel(R, init="pre"), C["start"], "-"), ("weak, default", sel(R, init="warm", form="weak", ic=1.0), C["weak_on"], "-"),
                ("strong, default", sel(R, init="warm", form="strong", ic=1.0), C["strong"], "-")]
        if k: arms.append((f"strong, x{k:g}", sel(R, init="warm", form="strong", ic=k), C["strong"], "--"))
        for lab, rs, col, ls in arms:
            if rs: _err_band(ax, rs, col, ls, lab)
        ax.set(xscale="log", yscale="log", title=f"Ramped boundary, tau = {tau:g}", xlabel="time $t$")
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("relative error of the transient")
    fig.tight_layout()
    save(fig, out, "heat_ramped_boundary_error_over_time")

@safe
def heat_band_mask(data, out):
    B = [r for r in data.H if r["band"] > 0 and float(r["ic"]) == 1.0]
    if not B:
        note("no band-masked runs"); return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, dim in zip(axes, (1, 2)):
        for form, col in (("weak", C["weak_on"]), ("strong", C["strong"])):
            for init, ls in (("warm", "-"), ("cold", "--")):
                bands = sorted({0.0} | {r["band"] for r in B if r["dim"] == dim})
                ys = []
                for b in bands:
                    rs = sel(data.H, dim=dim, case="clash", init=init, form=form, ic=1.0, band=b, dual=False)
                    ys.append(np.mean([r["err"]["t=1.0"] for r in rs]) if rs else np.nan)
                ax.plot(bands, ys, ls, marker="o", color=col, label=f"{form}, {init}")
        ax.set(yscale="log", title=f"{dim}D heat: masking IC points near the boundary", xlabel="band width $b$")
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("error at $t=1$")
    fig.tight_layout()
    save(fig, out, "heat_masked_ic_band_effect_at_t1")

@safe
def heat_absolute_error(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, dim in zip(axes, (1, 2)):
        rms = np.array([transient_rms(dim, "clash", t) for t in TV])
        ax.plot(TV, rms, color="k", lw=1, label="RMS of u - 1 (size of the transient)")
        for form, col in (("weak", C["weak_on"]), ("strong", C["strong"])):
            rs = sel(data.H, dim=dim, case="clash", init="warm", form=form, ic=1.0, band=0.0, dual=False)
            if rs:
                A = E(rs) * rms
                ax.plot(TV, A.mean(0), color=col, lw=1.3, label=f"{form}, absolute error")
                ax.fill_between(TV, A.min(0), A.max(0), color=col, alpha=.15, lw=0)
        ax.set(xscale="log", yscale="log", title=f"{dim}D heat, clashing IC, warm start", xlabel="time $t$")
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("RMS error")
    fig.tight_layout()
    save(fig, out, "heat_absolute_error_against_size_of_transient")

@safe
def heat_seed_spread_all_arms(data, out, SW):
    rows = []
    for key, s in SW.items():
        if not key.startswith("heat"): continue
        for m, rs in s["mults"].items():
            if len(rs) >= 2:
                v = E(rs); rows.append((f"{s['label']} x{m:g}", (v.max(0) - v.min(0)).max()))
    if not rows: return
    fig, ax = plt.subplots(figsize=(7, max(3, .16 * len(rows))))
    y = np.arange(len(rows)); rg = np.array([r[1] for r in rows])
    ax.barh(y, rg, color="#888"); ax.barh(y, rg / 2.93, color=C["late"], height=.4)
    ax.axvline(THRESH, color="k", lw=.9)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.5); ax.invert_yaxis()
    ax.set(xscale="log", xlabel="largest seed range over the five times (grey) and estimated SE of the mean (blue)",
           title="Seed spread of every sweep arm against the 0.05 threshold")
    fig.tight_layout()
    save(fig, out, "heat_seed_spread_every_sweep_arm")

# ----------------------------------------------------------------------------- advection-diffusion research figures
@safe
def adv_error_over_time(data, out):
    A = data.D["heat_adv"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, case in zip(axes, ("clash", "compat")):
        st = sel(A, case=case, init="pre")
        if st: ax.plot(TV, np.maximum(E(st), 1e-4).mean(0), color=C["start"], lw=1, label="supervised start")
        for form, col in (("weak", C["weak_on"]), ("strong", C["strong"])):
            rs = sel(A, case=case, init="warm", form=form, ic=1.0)
            if rs: _err_band(ax, rs, col, "-", f"{form}, default")
        if case == "compat":
            for m, rs in by_mult(sel(A, case="compat", init="warm", form="strong")).items():
                if m != 1.0: _err_band(ax, rs, C["strong"], "--", f"strong, x{m:g}")
        ax.set(xscale="log", yscale="log", title=f"Advection-diffusion, {'clashing' if case == 'clash' else 'matching'} IC", xlabel="time $t$")
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("relative error of the transient")
    fig.tight_layout()
    save(fig, out, "advdiff_error_over_time_defaults_and_controls")

@safe
def adv_sweep_error_vs_weight(data, out, SW):
    fig, ax = plt.subplots(figsize=(5, 3.4)); k = data.k["adv"]
    for form in ("strong", "weak"):
        s = SW.get(f"adv_{form}")
        if not s: continue
        ms = np.array(list(s["mults"])); x = ms * (k if form == "weak" else 1.0); o = np.argsort(x)
        for t, col in (("t=0.01", C["early"]), ("t=1.0", C["late"])):
            j = HEAT_T.index(t); y = np.array([E(s["mults"][m])[:, j].mean() for m in ms])[o]
            ax.plot(x[o], y, "-" if form == "strong" else "--", marker="o", ms=3.2, color=col, mfc=col if form == "strong" else "w")
    ax.axvline(k, color=".3", ls="--", lw=.8)
    ax.set(xscale="log", yscale="log", xlabel="IC weight relative to the strong form's default\n(weak-form runs at multiplier x k)",
           ylabel="relative error", title="Advection-diffusion: early (orange) and late (blue) error")
    fig.tight_layout()
    save(fig, out, "advdiff_early_late_error_against_ic_weight")

@safe
def adv_gap_figs(data, out, SW):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, k in zip(axes, ("adv_strong", "adv_weak")):
        if k in SW: _gap_vs_mult(ax, SW[k], SW[k]["label"])
    fig.tight_layout(); save(fig, out, "advdiff_sweep_gap_against_multiplier")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, k in zip(axes, ("adv_strong", "adv_weak")):
        if k in SW: _gap_heatmap(ax, SW[k])
    fig.tight_layout(); save(fig, out, "advdiff_sweep_gap_heatmap_by_multiplier_and_time")

@safe
def adv_tug_and_waterfall(data, out):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    _tug(ax, {"c10": data.M["adv10"].get("d1_warm"), "m10": data.M["adv10_compat"].get("d1_warm"), "c100": data.M["adv100"].get("d1_warm")},
         [("clash, 10 draws", "c10"), ("matching, 10 draws", "m10"), ("clash, 100 draws", "c100")],
         "Advection-diffusion, supervised start: IC pull against physics pull")
    fig.tight_layout(); save(fig, out, "advdiff_gradient_tug_of_war_ic_vs_physics")
    A = data.D["heat_adv"]; st = sel(A, case="clash", init="pre")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, form in zip(axes, ("weak", "strong")):
        rs = sel(A, case="clash", init="warm", form=form, ic=1.0)
        if not (st and rs): ax.set_visible(False); continue
        mt = lambda R, k: float(np.mean([r["terms"][k] for r in R]))
        _waterfall(ax, (mt(st, "ic"), mt(st, form)), (mt(rs, "ic"), mt(rs, form)), f"Advection-diffusion, {form} form, clashing IC")
    fig.tight_layout(); save(fig, out, "advdiff_objective_waterfall_start_to_end")

@safe
def adv_per_seed_boxplots(data, out, SW):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, k in zip(axes, ("adv_strong", "adv_weak")):
        s = SW.get(k)
        if not s: ax.set_visible(False); continue
        ms = list(s["mults"]); vals = [[r["err"]["t=1.0"] for r in s["mults"][m]] for m in ms]
        ax.boxplot(vals, widths=.5, showfliers=False)
        for i, v in enumerate(vals):
            ax.plot(np.full(len(v), i + 1), v, "k.", ms=5)
        ax.set_xticks(range(1, len(ms) + 1)); ax.set_xticklabels([f"x{m:g}" for m in ms], fontsize=7)
        ax.set(yscale="log", title=f"{s['label']}: error at t = 1 per seed", ylabel="relative error")
    fig.tight_layout(); save(fig, out, "advdiff_per_seed_error_boxplots_t1")

# ----------------------------------------------------------------------------- electrochem (repo only)
def ec_early(R, arm):
    E_ = R.get("early_time_3seeds", {})
    vals = {t: [E_[f"{arm}_s{s}"][t] for s in SEEDS if f"{arm}_s{s}" in E_] for t in TAGS}
    if not vals["t=1"]:
        note(f"early_time_3seeds has no '{arm}'")
    if ("early", arm) not in EC_LOGGED:
        EC_LOGGED.add(("early", arm))
        for t in TAGS:
            for s, v in zip([s for s in SEEDS if f"{arm}_s{s}" in E_], vals[t]):
                EC_ROWS.append(("electrochem", arm, s, t, v))
    return vals

def ec_summary(R, pat):
    T = R.get("rework_summary", {}).get("test", {})
    if pat not in T:
        note(f"rework_summary has no '{pat}'"); return None
    if ("summary", pat) not in EC_LOGGED:
        EC_LOGGED.add(("summary", pat))
        for t in TAGS:
            for i, v in enumerate(T[pat][t]):
                EC_ROWS.append(("electrochem", pat.replace("_s{s}", ""), f"seed#{i}", t, v))
    return T[pat]

EC_ARMS = [("warmstart", "Supervised start", C["start"], None), ("warm_weak", "Weak, IC on (warm)", C["weak_on"], None),
           ("cold_weak", "Weak, IC on (cold)", C["weak_on"], "//"), ("warm_weak_noIC", "Weak, IC off (warm)", C["weak_off"], None),
           ("cold_weak_noIC", "Weak, IC off (cold)", C["weak_off"], "//"), ("warm_strong", "Strong, IC on (warm)", C["strong"], None),
           ("warm_strong_noIC", "Strong, IC off (warm)", C["strong_off"], None)]

@safe
def ec_drift(R, out):
    data = [(lab, col, hat, ec_early(R, a)) for a, lab, col, hat in EC_ARMS]
    data = [d for d in data if d[3]["t=1"]]
    if not data: return
    fig, ax = plt.subplots(figsize=(6.8, 2.6)); w = 0.8 / len(data)
    for j, (lab, col, hat, v) in enumerate(data):
        for i, t in enumerate(TAGS):
            m, lo, hi, n = stats(v[t]); x = i + (j - (len(data) - 1) / 2) * w
            ax.bar(x, m, w * 0.92, color=col, hatch=hat, edgecolor="white" if not hat else "black", linewidth=0.4, label=lab if i == 0 else None)
            ax.errorbar(x, m, yerr=[[max(m - lo, 0)], [max(hi - m, 0)]], color="black", lw=0.7, capsize=1.5)
    ax.plot([-0.45, 0.45], [STEADY_GUESS] * 2, "k:", lw=1); ax.text(-0.45, STEADY_GUESS + 0.02, "steady-state guess", va="bottom", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels([TAG_LABEL[t] for t in TAGS]); ax.set_ylabel(r"held-out rel. $L^2$ error in $C$")
    ax.legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    save(fig, out, "electrochem_drift_by_arm")

@safe
def ec_probe(R, out):
    P = R.get("mechanism_probe", {}).get("probe", {})
    if not P:
        note("mechanism_probe not found"); return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.4)); marks = ["o", "s", "^"]
    for k, (s, rows) in enumerate(sorted(P.items())):
        st = [r["step"] for r in rows]; x = np.arange(len(st))
        a1.plot(x, [r["weak_obj"] - rows[0]["weak_obj"] for r in rows], marker=marks[k % 3], color=C["weak_on"], lw=1)
        a1.plot(x, [r["strong_obj"] - rows[0]["strong_obj"] for r in rows], marker=marks[k % 3], color=C["strong"], lw=1)
        a2.plot(x, [r["err_t1"] for r in rows], marker=marks[k % 3], color="0.3", lw=1, label=f"seed {s}")
        for r in rows:
            EC_ROWS.append(("probe", f"step={r['step']:g}", s, "d_weak|d_strong|err_t1",
                            f"{r['weak_obj'] - rows[0]['weak_obj']:.4e}|{r['strong_obj'] - rows[0]['strong_obj']:.4e}|{r['err_t1']:.4f}"))
    labels = [f"{v:g}" for v in st]; n_lin = sum(1 for v in st if v <= LINEAR_MAX_STEP)
    for ax in (a1, a2):
        ax.set_xlim(-0.4, len(st) - 0.5); ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel(r"step along the IC term's pull ($\times\|\theta\|$)")
        if n_lin < len(st): ax.axvspan(n_lin - 0.5, len(st) - 0.5, color="0.92", lw=0, zorder=0)
    a1.axhline(0, color="black", lw=0.6); a1.set_yscale("symlog", linthresh=1e-3); a1.set_ylabel("change in training objective")
    a2.set_ylabel(r"held-out error at $t=1$"); a2.legend(frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, out, "electrochem_ic_pull_probe")

@safe
def ec_icweight(R, out):
    WS = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3); choice = R.get("rework_summary", {}).get("val_choice", {})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for ax, tag, title in zip(axes, ("0.25/Fo", "t=1"), ("early: $t=0.25/\\mathrm{Fo}$", "late: $t=1$")):
        for init in ("warm", "cold"):
            xs, ms = [], []
            for w in WS + (1.0,):
                v = ec_summary(R, f"{init}_weak_s{{s}}" if w == 1.0 else f"icw{w:g}_{init}_weak_s{{s}}")
                if v is not None: xs.append(w); ms.append(np.mean(v[tag]))
            if xs: ax.plot(xs, ms, "o-", color=C[init], lw=1.2, ms=4, label=f"{init} start")
            ch = choice.get(init, {}).get("chosen", "")
            if "w=" in ch:
                wc = float(ch.split("w=")[1].split()[0]); v = ec_summary(R, f"icw{wc:g}_{init}_weak_s{{s}}")
                if v: ax.plot(wc, np.mean(v[tag]), "*", color="black", ms=9, zorder=5)
        strong = ec_summary(R, "warm_strong_s{s}")
        if strong: ax.axhline(np.mean(strong[tag]), color=C["strong"], ls="--", lw=1, label="strong form")
        ax.set(xscale="log", xlabel="IC weight $w$", title=title, ylabel="held-out rel. error")
        if tag == "t=1": ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); save(fig, out, "electrochem_ic_weight_tradeoff")

@safe
def ec_per_seed_strip(R, out):
    data = [(lab, ec_early(R, a)) for a, lab, _, _ in EC_ARMS]
    data = [d for d in data if d[1]["t=1"]]
    if not data: return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, t in zip(axes, TAGS):
        for i, (lab, v) in enumerate(data):
            ax.plot(np.full(len(v[t]), i), v[t], "o", color="k", ms=4, alpha=.7)
            ax.plot([i - .25, i + .25], [np.mean(v[t])] * 2, color=C["up"], lw=2)
        ax.set_xticks(range(len(data))); ax.set_xticklabels([d[0] for d in data], rotation=45, ha="right", fontsize=7)
        ax.set_title(TAG_LABEL[t])
    axes[0].set_ylabel("held-out rel. error (dots: seeds; bar: mean)")
    fig.tight_layout(); save(fig, out, "electrochem_per_seed_error_strip")

@safe
def ec_tables(R, out):
    T = Tables(); rows = []
    def add(label, v):
        if v is not None: rows.append([label] + [cell(v[t]) for t in TAGS])
    add("Weak, IC weight 1 (default)", ec_summary(R, "warm_weak_s{s}"))
    for lab, pat in (("  + 72 test functions", "warm_weak_ts72_s{s}"), ("  + 144 test functions", "warm_weak_ts144_s{s}"),
                     ("  no Phase A (curriculum)", "warm_weak_noA_s{s}"), ("  per-time residual", "warm_weak_pertime_s{s}")):
        add(lab, ec_summary(R, pat))
    B = R.get("band_and_icweight_electrochem", {})
    for b in (0.05, 0.1, 0.2):
        k = f"band{b:g}_warm_weak_s42"
        if k in B: rows.append([f"  IC band {b:g} (seed 42)"] + [cell([B[k]['err'][t]]) for t in TAGS])
    F = R.get("ramp_fix_3seeds", {})
    for tau in (0.01, 0.05):
        ks = [f"ramp{tau:g}_warm_weak_s{s}" for s in SEEDS if f"ramp{tau:g}_warm_weak_s{s}" in F]
        if ks: rows.append([f"  ramped boundary, tau={tau:g}"] + [cell([F[k]["err"][t] for k in ks]) for t in TAGS])
    for lab, pat in (("  IC weight 0 (off)", "warm_weak_noIC_s{s}"), ("  IC weight 0.1", "icw0.1_warm_weak_s{s}"),
                     ("  IC weight 0.05 (validation pick)", "icw0.05_warm_weak_s{s}"), ("Strong, IC weight 1 (reference)", "warm_strong_s{s}")):
        add(lab, ec_summary(R, pat))
    T.emit("Electrochem, warm start: what does and doesn't remove the drift (held-out rel. error, mean [min-max])", ["Variant"] + list(TAGS), rows)
    T.write(out, "electrochem_tables")

# ----------------------------------------------------------------------------- main
def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "arm", "seed", "metric", "value"]); w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    base = os.path.join(a.root, a.out) if not os.path.isabs(a.out) else a.out
    P = os.path.join(base, "paper_figs")                                  # shared by heat and electrochem
    HF, AF = (os.path.join(a.root, "heat_runs", d) for d in ("heat_figs", "advdiff_figs"))
    EF = os.path.join(a.root, "electrochem", "results", "electrochem_figs")
    data = Data(a.root)
    SW = data.sweeps()
    print(f"k used for the axes: 1D heat {data.k['heat1']:g}, 2D heat {data.k['heat2']:g}, adv.-diff. {data.k['adv']:g}")
    log_all(data)
    # paper
    fig_heat(data, P); fig_sweep(data, P, SW); fig_gp(data, P, SW); fig_spread(data, P)
    T = Tables()
    t_dual(data, T); t_giveup(data, T); t_sweeps(data, T, SW); t_anneal(data, T); t_mesh_order_norm(data, T)
    t_adv(data, T, SW); t_ramp(data, T); t_gp(T); t_abs(data, T)
    T.write(P, "paper_tables")
    write_csv(os.path.join(P, "per_seed.csv"), PAPER_ROWS)
    # heat research
    for fn in (heat_error_over_time_all_arms, heat_per_seed_boxplots, heat_objective_waterfalls, heat_gradient_tug_of_war,
               heat_k_vs_element_size_and_order, heat_annealed_weight_trajectories, heat_mesh_order_norm_matches,
               heat_ramp_error_over_time, heat_band_mask, heat_absolute_error):
        fn(data, HF)
    heat_sweep_gap_heatmaps(data, HF, SW); heat_sweep_gap_vs_multiplier(data, HF, SW); heat_seed_spread_all_arms(data, HF, SW)
    # advection-diffusion research
    adv_error_over_time(data, AF); adv_sweep_error_vs_weight(data, AF, SW); adv_gap_figs(data, AF, SW)
    adv_tug_and_waterfall(data, AF); adv_per_seed_boxplots(data, AF, SW)
    # electrochem (repo only)
    R = load_electrochem(a.root)
    if R:
        ec_drift(R, EF); ec_probe(R, EF); ec_icweight(R, EF); ec_per_seed_strip(R, EF); ec_tables(R, EF)
        write_csv(os.path.join(EF, "electrochem_per_seed.csv"), EC_ROWS)
    if MISSING:
        print(f"\n{len(MISSING)} items missing or skipped (see above) -- everything else was written.")