"""
make_paper_figs.py -- build the paper's figures, tables and a per-seed CSV from results you already logged.
No GPU, no torch: it only reads JSON.

USAGE (from the amorflux folder):
    pip install numpy matplotlib
    python make_paper_figs.py                       # writes to results/paper_figs/
    python make_paper_figs.py --root . --out results/paper_figs

READS
    results/results.json            electrochem keys: early_time_3seeds, rework_summary, mechanism_probe,
                                    ramp_fix_3seeds, band_and_icweight_electrochem
    heat_results/results/*.json     heat runs (plain weak / strong, band)
    gm_heat_results/results/*.json  heat runs (dual-norm weak)
WRITES (in --out)
    fig1_drift.pdf/.png         the failure: error at 3 times, weak IC on/off vs strong, warm/cold, 3 seeds
    fig2_probe.pdf/.png         mechanism: change in each training objective along the IC term's pull
    fig3_heat.pdf/.png          generality: heat 1D/2D, matching vs clashing IC, error over time
    fig4_icweight.pdf/.png      mitigation: early vs late error as the IC weight varies; validation pick starred
    tables.md / tables.tex      robustness (what doesn't fix it), heat dual-norm, heat "what each form gives up"
    per_seed.csv                every number behind the figures, one row per seed
Anything missing is reported at the end instead of crashing.

CHANGES (Oct 5)
    fig1: strong arms labelled "(warm)" like the others.
    fig2: steps beyond the near-linear range (> 3e-4) shaded and labelled; left legend uses plain lines
          (markers encode seed, as in the right panel).
    fig4: late panel on a log y-axis so strong / IC-off / validation pick are separable.
    csv:  each arm is logged once (arms used by several figures/tables were duplicated before).
"""
import os, json, glob, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

# Okabe-Ito (colour-blind safe)
C = dict(start="#7F7F7F", weak_on="#D55E00", weak_off="#56B4E9", strong="#009E73", strong_off="#7FD3B8",
         dual="#E69F00", warm="#0072B2", cold="#CC79A7")
TAGS = ("0.25/Fo", "1/Fo", "t=1")
TAG_LABEL = {"0.25/Fo": r"$t=0.25/\mathrm{Fo}$", "1/Fo": r"$t=1/\mathrm{Fo}$", "t=1": r"$t=1$"}
SEEDS = (42, 43, 44)
STEADY_GUESS = 0.326   # error at 0.25/Fo of a model that outputs only the t=1 steady state (measured in-notebook)
LINEAR_MAX_STEP = 3e-4  # probe steps above this are outside the near-linear range
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 200})

MISSING, CSV_ROWS, LOGGED = [], [], set()

def note(msg):
    MISSING.append(msg); print("  [missing] " + msg)

def stats(v):
    v = np.asarray(v, float)
    return v.mean(), v.min(), v.max(), len(v)

def cell(v, prec=3):
    """Mean [min-max] with 3 significant figures throughout, so small values (0.004) and large ones (40.9)
    both read correctly and the range never rounds to something below/above the mean."""
    if v is None or len(v) == 0:
        return "--"
    m, lo, hi, n = stats(v)
    g = lambda x: f"{x:#.{prec}g}".rstrip(".")      # keeps trailing zeros: 0.540, not 0.54
    return f"{g(m)} [{g(lo)}-{g(hi)}]" if n > 1 else f"{g(m)} (n=1)"

def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig); print(f"  wrote {name}.pdf/.png")

def first_time(key):
    """True the first time an arm is logged to the CSV, so arms reused across figures appear once."""
    if key in LOGGED:
        return False
    LOGGED.add(key); return True

# ----------------------------------------------------------------------------- loading
def load_electrochem(root):
    p = os.path.join(root, "results", "results.json")
    if not os.path.exists(p):
        note(f"{p} not found"); return {}
    return json.load(open(p))

def load_heat(root):
    rows = []
    for d in ("heat_results", "gm_heat_results"):
        for f in glob.glob(os.path.join(root, d, "results", "*.json")):
            r = json.load(open(f)); r.setdefault("band", 0.0); r.setdefault("dual", False); rows.append(r)
    if not rows:
        note("no heat result JSONs found")
    return rows

def early(R, arm):
    """Per-seed errors {tag: [..]} for an electrochem arm from the early-time table (3 seeds)."""
    E = R.get("early_time_3seeds", {})
    vals = {t: [E[f"{arm}_s{s}"][t] for s in SEEDS if f"{arm}_s{s}" in E] for t in TAGS}
    if not vals["t=1"]:
        note(f"early_time_3seeds has no '{arm}'")
    if first_time(("early", arm)):
        for t in TAGS:
            for s, v in zip([s for s in SEEDS if f"{arm}_s{s}" in E], vals[t]):
                CSV_ROWS.append(("electrochem", arm, s, t, v))
    return vals

def summary_arm(R, pat):
    """Per-seed errors {tag: [..]} from rework_summary['test'] (pattern like 'icw0.1_warm_weak_s{s}')."""
    T = R.get("rework_summary", {}).get("test", {})
    if pat not in T:
        note(f"rework_summary has no '{pat}'"); return None
    if first_time(("summary", pat)):
        for t in TAGS:
            for i, v in enumerate(T[pat][t]):
                CSV_ROWS.append(("electrochem", pat.replace("_s{s}", ""), f"seed#{i}", t, v))
    return T[pat]

def heat_sel(H, **kw):
    return [r for r in H if all(r.get(k) == v for k, v in kw.items())]

# ----------------------------------------------------------------------------- figure 1: the failure
def fig_drift(R, out):
    arms = [("warmstart", "Supervised start", C["start"], None),
            ("warm_weak", "Weak, IC on (warm)", C["weak_on"], None),
            ("cold_weak", "Weak, IC on (cold)", C["weak_on"], "//"),
            ("warm_weak_noIC", "Weak, IC off (warm)", C["weak_off"], None),
            ("cold_weak_noIC", "Weak, IC off (cold)", C["weak_off"], "//"),
            ("warm_strong", "Strong, IC on (warm)", C["strong"], None),
            ("warm_strong_noIC", "Strong, IC off (warm)", C["strong_off"], None)]
    data = [(lab, col, hat, early(R, a)) for a, lab, col, hat in arms]
    data = [d for d in data if d[3]["t=1"]]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    w = 0.8 / len(data)
    for j, (lab, col, hat, v) in enumerate(data):
        for i, t in enumerate(TAGS):
            m, lo, hi, n = stats(v[t])
            x = i + (j - (len(data) - 1) / 2) * w
            ax.bar(x, m, w * 0.92, color=col, hatch=hat, edgecolor="white" if not hat else "black",
                   linewidth=0.4, label=lab if i == 0 else None)
            ax.errorbar(x, m, yerr=[[max(m - lo, 0)], [max(hi - m, 0)]], color="black", lw=0.7, capsize=1.5)
    ax.plot([-0.45, 0.45], [STEADY_GUESS] * 2, "k:", lw=1)
    ax.text(-0.45, STEADY_GUESS + 0.02, "steady-state guess", va="bottom", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels([TAG_LABEL[t] for t in TAGS])
    ax.set_ylabel(r"held-out rel. $L^2$ error in $C$")
    ax.legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.text(0.99, -0.03, "bars: mean of 3 seeds; whiskers: min-max", ha="right", fontsize=7, color="0.4")
    save(fig, out, "fig1_drift")

# ----------------------------------------------------------------------------- figure 2: mechanism probe
def fig_probe(R, out):
    P = R.get("mechanism_probe", {}).get("probe", {})
    if not P:
        note("mechanism_probe not found"); return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.4))
    marks = ["o", "s", "^"]
    for k, (s, rows) in enumerate(sorted(P.items())):
        st = [r["step"] for r in rows]; x = np.arange(len(st))
        dw = [r["weak_obj"] - rows[0]["weak_obj"] for r in rows]
        ds = [r["strong_obj"] - rows[0]["strong_obj"] for r in rows]
        a1.plot(x, dw, marker=marks[k % 3], color=C["weak_on"], lw=1)
        a1.plot(x, ds, marker=marks[k % 3], color=C["strong"], lw=1)
        a2.plot(x, [r["err_t1"] for r in rows], marker=marks[k % 3], color="0.3", lw=1, label=f"seed {s}")
        for r in rows:
            CSV_ROWS.append(("probe", f"step={r['step']:g}", s, "d_weak|d_strong|err_t1",
                             f"{r['weak_obj'] - rows[0]['weak_obj']:.4e}|{r['strong_obj'] - rows[0]['strong_obj']:.4e}|{r['err_t1']:.4f}"))
    labels = [f"{v:g}" for v in st]
    # shade the steps outside the near-linear range (first-order argument only holds for small steps)
    n_lin = sum(1 for v in st if v <= LINEAR_MAX_STEP)
    for ax in (a1, a2):
        ax.set_xlim(-0.4, len(st) - 0.5)
        if n_lin < len(st):
            ax.axvspan(n_lin - 0.5, len(st) - 0.5, color="0.92", lw=0, zorder=0)
            ax.text((n_lin - 0.5 + len(st) - 0.5) / 2, 0.04, "beyond\nlinear range", ha="center", va="bottom",
                    fontsize=6.5, color="0.35", transform=ax.get_xaxis_transform())
    a1.axhline(0, color="black", lw=0.6)
    a1.set_yscale("symlog", linthresh=1e-3)
    a1.set_xticks(x); a1.set_xticklabels(labels)
    a1.set_xlabel(r"step along the IC term's pull ($\times\|\theta\|$)")
    a1.set_ylabel("change in training objective")
    a1.legend(handles=[Line2D([], [], color=C["weak_on"], lw=1.4, label="weak objective"),
                       Line2D([], [], color=C["strong"], lw=1.4, label="strong objective")],
              frameon=False, loc="upper left")
    a1.set_title("below 0: following the pull helps that objective", fontsize=7, color="0.4")
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_xlabel(r"step along the IC term's pull ($\times\|\theta\|$)")
    a2.set_ylabel(r"held-out error at $t=1$")
    a2.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.text(0.99, -0.03, "markers: seed (both panels)", ha="right", fontsize=7, color="0.4")
    save(fig, out, "fig2_probe")

# ----------------------------------------------------------------------------- figure 3: heat generality
def fig_heat(H, out):
    if not H:
        return
    T_EVAL = ["t=0.01", "t=0.05", "t=0.1", "t=0.3", "t=1.0"]
    tv = [float(t[2:]) for t in T_EVAL]
    lines = [("Weak, IC on", dict(form="weak", ic=1, dual=False), C["weak_on"], "-"),
             ("Weak (dual norm), IC on", dict(form="weak", ic=1, dual=True), C["dual"], "--"),
             ("Strong, IC on", dict(form="strong", ic=1, dual=False), C["strong"], "-"),
             ("Weak, IC off", dict(form="weak", ic=0, dual=False), C["weak_off"], ":")]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4), sharex=True)
    for i, dim in enumerate((1, 2)):
        for j, case in enumerate(("compat", "clash")):
            ax = axes[i, j]
            for lab, sel, col, ls in lines:
                rs = heat_sel(H, dim=dim, case=case, init="warm", band=0.0, **sel)
                if not rs:
                    continue
                E = np.array([[max(r["err"][t], 1e-4) for t in T_EVAL] for r in rs])
                ax.plot(tv, E.mean(0), ls, color=col, lw=1.4, label=lab)
                ax.fill_between(tv, E.min(0), E.max(0), color=col, alpha=0.18, lw=0)
                for r in rs:
                    for t in T_EVAL:
                        CSV_ROWS.append(("heat", f"d{dim}_{case}_warm_{sel['form']}_ic{sel['ic']}"
                                         + ("_dual" if sel["dual"] else ""), r["seed"], t, r["err"][t]))
            st = heat_sel(H, dim=dim, case=case, init="pre")
            if st:
                E = np.array([[max(r["err"][t], 1e-4) for t in T_EVAL] for r in st])
                ax.plot(tv, E.mean(0), color=C["start"], lw=1, label="Supervised start")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 2)
            ax.set_title(f"{dim}D, {'IC matches boundary' if case == 'compat' else 'IC clashes with boundary'}")
            if j == 0:
                ax.set_ylabel("rel. error of the transient")
            if i == 1:
                ax.set_xlabel("time $t$")
    h, l = axes[1, 1].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.text(0.99, 0.0, "warm start; lines: mean of 3 seeds; bands: min-max", ha="right", fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, out, "fig3_heat")

# ----------------------------------------------------------------------------- figure 4: IC-weight trade-off
def fig_icweight(R, out):
    WS = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3)
    choice = R.get("rework_summary", {}).get("val_choice", {})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=False)
    for ax, tag, title in zip(axes, ("0.25/Fo", "t=1"), ("early: $t=0.25/\\mathrm{Fo}$", "late: $t=1$")):
        for init in ("warm", "cold"):
            xs, ms, los, his, ns = [], [], [], [], []
            for w in WS + (1.0,):
                pat = f"{init}_weak_s{{s}}" if w == 1.0 else f"icw{w:g}_{init}_weak_s{{s}}"
                v = summary_arm(R, pat)
                if v is None:
                    continue
                m, lo, hi, n = stats(v[tag]); xs.append(w); ms.append(m); los.append(lo); his.append(hi); ns.append(n)
            if not xs:
                continue
            col = C[init]
            ax.plot(xs, ms, "-", color=col, lw=1.2, label=f"{init} start")
            for x, m, lo, hi, n in zip(xs, ms, los, his, ns):
                ax.errorbar(x, m, yerr=[[max(m - lo, 0)], [max(hi - m, 0)]], fmt="o", color=col, ms=4, capsize=1.5, lw=0.7,
                            mfc=col if n >= 3 else "white")
            off = summary_arm(R, f"{init}_weak_noIC_s{{s}}")
            if off:
                ax.axhline(np.mean(off[tag]), color=col, ls=":", lw=1)
            ch = choice.get(init, {}).get("chosen", "")
            if "w=" in ch:
                wc = float(ch.split("w=")[1].split()[0]); v = summary_arm(R, f"icw{wc:g}_{init}_weak_s{{s}}")
                if v:
                    ax.plot(wc, np.mean(v[tag]), "*", color="black", ms=9, zorder=5,
                            label="validation pick" if init == "warm" else None)
        strong = summary_arm(R, "warm_strong_s{s}")
        if strong:
            ax.axhline(np.mean(strong[tag]), color=C["strong"], ls="--", lw=1, label="strong form")
        if tag == "0.25/Fo":
            ax.axhline(STEADY_GUESS, color="black", ls=":", lw=0.8, label="steady-state guess")
        else:
            # late errors span 0.1-1.1; a log axis separates strong (~0.105), IC off (~0.12) and the pick (~0.14)
            ax.set_yscale("log")
            ticks = [0.1, 0.2, 0.3, 0.5, 1.0]
            ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks])
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xscale("log"); ax.set_xlabel("IC weight $w$"); ax.set_title(title)
        ax.set_ylabel("held-out rel. error")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l: h.append(hh); l.append(ll)
    fig.legend(h, l, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    fig.text(0.99, -0.02, "dotted coloured lines: IC off (w=0); open markers: 1 seed, filled: 3 seeds",
             ha="right", fontsize=7, color="0.4")
    fig.tight_layout()
    save(fig, out, "fig4_icweight")

# ----------------------------------------------------------------------------- tables
def tables(R, H, out):
    md, tex = [], []

    def emit(title, header, rows):
        md.append(f"\n### {title}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
        md.extend("| " + " | ".join(r) + " |" for r in rows)
        tex.append(f"% {title}\n\\begin{{tabular}}{{l" + "c" * (len(header) - 1) + "}\n\\toprule\n"
                   + " & ".join(header) + " \\\\\n\\midrule")
        tex.extend(" & ".join(x.replace("[", "{\\scriptsize[").replace("]", "]}") for x in r) + " \\\\" for r in rows)
        tex.append("\\bottomrule\n\\end{tabular}\n")

    # Table 1: what does / doesn't remove the drift (electrochem, warm weak)
    rows = []
    def add(label, v, n=None):
        if v is None:
            return
        rows.append([label] + [cell(v[t]) for t in TAGS])
    add("Weak, IC weight 1 (default)", summary_arm(R, "warm_weak_s{s}"))
    add("  + 72 test functions", summary_arm(R, "warm_weak_ts72_s{s}"))
    add("  + 144 test functions", summary_arm(R, "warm_weak_ts144_s{s}"))
    add("  no Phase A (curriculum)", summary_arm(R, "warm_weak_noA_s{s}"))
    add("  per-time residual", summary_arm(R, "warm_weak_pertime_s{s}"))
    B = R.get("band_and_icweight_electrochem", {})
    for b in (0.05, 0.1, 0.2):
        k = f"band{b:g}_warm_weak_s42"
        if k in B:
            rows.append([f"  IC band {b:g} (seed 42)"] + [cell([B[k]['err'][t]]) for t in TAGS])
        else:
            note(f"band_and_icweight_electrochem has no '{k}'")
    F = R.get("ramp_fix_3seeds", {})
    for tau in (0.01, 0.05):
        ks = [f"ramp{tau:g}_warm_weak_s{s}" for s in SEEDS if f"ramp{tau:g}_warm_weak_s{s}" in F]
        if ks:
            rows.append([f"  ramped boundary, tau={tau:g}"] + [cell([F[k]["err"][t] for k in ks]) for t in TAGS])
        else:
            note(f"ramp_fix_3seeds has no ramp{tau:g}_warm_weak")
    add("  IC weight 0 (off)", summary_arm(R, "warm_weak_noIC_s{s}"))
    add("  IC weight 0.1", summary_arm(R, "icw0.1_warm_weak_s{s}"))
    add("  IC weight 0.05 (validation pick)", summary_arm(R, "icw0.05_warm_weak_s{s}"))
    add("Strong, IC weight 1 (reference)", summary_arm(R, "warm_strong_s{s}"))
    emit("Electrochem, warm start: what does and doesn't remove the drift (held-out rel. error, mean [min-max])",
         ["Variant"] + list(TAGS), rows)

    # Table 2: heat, plain vs dual-norm weak loss under the clash
    rows = []
    for dim in (1, 2):
        for init in ("warm", "cold"):
            r = [f"{dim}D, {init}"]
            for sel in (dict(form="weak", dual=False), dict(form="weak", dual=True), dict(form="strong", dual=False)):
                rs = heat_sel(H, dim=dim, case="clash", init=init, ic=1, band=0.0, **sel)
                r.append(cell([x["err"]["t=1.0"] for x in rs]) if rs else "--")
            rows.append(r)
    emit("Heat, clashing IC: error at t=1 (mean [min-max], 3 seeds)",
         ["Problem, start", "Weak (plain)", "Weak (dual norm)", "Strong"], rows)

    # Table 3: heat, what each form gives up (2D clash, warm, IC on)
    rows = []
    for lab, sel in (("Supervised start", dict(init="pre")),
                     ("Weak", dict(init="warm", form="weak", ic=1, band=0.0, dual=False)),
                     ("Weak (dual norm)", dict(init="warm", form="weak", ic=1, band=0.0, dual=True)),
                     ("Strong", dict(init="warm", form="strong", ic=1, band=0.0, dual=False))):
        rs = heat_sel(H, dim=2, case="clash", **sel)
        if not rs:
            continue
        rows.append([lab, cell([x["terms"]["ic"] for x in rs]), cell([x["terms"]["strong"] for x in rs]),
                     cell([x["err"]["t=0.01"] for x in rs]), cell([x["err"]["t=1.0"] for x in rs])])
    emit("Heat 2D, clashing IC, warm start: what each form gives up",
         ["Model", "IC loss", "pointwise PDE residual", "error t=0.01", "error t=1"], rows)

    open(os.path.join(out, "tables.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(out, "tables.tex"), "w").write("\n".join(tex) + "\n")
    print("  wrote tables.md / tables.tex")

# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=os.path.join("results", "paper_figs"))
    a = ap.parse_args()
    out = os.path.join(a.root, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(out, exist_ok=True)
    R, H = load_electrochem(a.root), load_heat(a.root)
    print(f"loaded {len(R)} electrochem keys, {len(H)} heat runs")
    fig_drift(R, out); fig_probe(R, out); fig_heat(H, out); fig_icweight(R, out); tables(R, H, out)
    with open(os.path.join(out, "per_seed.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "arm", "seed", "metric", "value"]); w.writerows(CSV_ROWS)
    print(f"  wrote per_seed.csv ({len(CSV_ROWS)} rows)")
    if MISSING:
        print(f"\n{len(MISSING)} items missing (see above) -- figures/tables use whatever exists.")