"""
pull_heat_tables.py -- writes the four appendix tables that paper.tex reads with \\inputif. CPU only, reads JSON.

    python pull_heat_tables.py                    # reads heat_runs/heat_results and heat_runs/gm_heat_results, writes results/paper_figs
    python pull_heat_tables.py --root . --out path/next/to/paper.tex

WRITES
    app_heat_1d.tex     Table "tab:app-heat-1d": 1D heat, every default arm, error over time (Appendix A.3)
    app_heat_2d.tex     Table "tab:app-heat-2d": the same in 2D
    app_heat_terms.tex  Table "tab:app-heat-terms": loss terms under the clash, 1D and cold starts (Appendix A.6)
    app_heat_band.tex   Table "tab:app-heat-band": masked initial condition, error against band width (Appendix A.7)
Put the four files next to paper.tex and compile twice.
"""
import argparse, glob, json, os
import numpy as np

T = ("t=0.01", "t=0.05", "t=0.1", "t=0.3", "t=1.0")
HEAD = r"$t=0.01$ & $t=0.05$ & $t=0.1$ & $t=0.3$ & $t=1$"

def load(root):
    rows, seen = [], set()
    for d in ("heat_results", "gm_heat_results"):
        for f in sorted(glob.glob(os.path.join(root, "heat_runs", d, "results", "*.json"))):
            r = json.load(open(f))
            if r.get("name") in seen:
                continue
            seen.add(r.get("name"))
            r.setdefault("band", 0.0); r.setdefault("dual", False)
            r["band"] = float(r["band"] or 0.0); r["dual"] = bool(r["dual"])
            rows.append(r)
    if not rows:
        raise SystemExit(f"no results in {root}/heat_runs/heat_results/results or heat_runs/gm_heat_results/results")
    return rows

def sel(R, **kw):
    out = []
    for r in R:
        if all((abs(float(r.get(k, -1)) - v) < 1e-9) if isinstance(v, float) else r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out

def g(x):
    """3 significant figures, keeping trailing zeros (0.0310, 0.351, 52.9); very small or large values as
    $5.74 x 10^-5$ in LaTeX math and 1{,}390, never raw e-notation."""
    x = float(x)
    if x != 0 and abs(x) < 1e-3:
        m, e = f"{x:.2e}".split("e")
        return "$" + m + r"\times10^{" + str(int(e)) + "}$"
    if abs(x) >= 1e3:
        v = float(f"{x:.3g}")
        return f"{v:,.0f}".replace(",", "{,}")
    return f"{x:#.3g}".rstrip(".")

def mean_cell(v):
    return g(np.mean(v)) if len(v) else "--"

def range_cell(v):
    if len(v) == 0:
        return "--"
    return f"{g(np.mean(v))} {{\\scriptsize[{g(min(v))}--{g(max(v))}]}}" if len(v) > 1 else f"{g(v[0])} (n=1)"

def errs(rs):
    return [mean_cell([r["err"][t] for r in rs]) for t in T]

def t1_range(rs):
    v = [r["err"]["t=1.0"] for r in rs]
    return f"{g(min(v))}--{g(max(v))}" if len(v) > 1 else "--"

def write(path, caption, label, colspec, header, body):
    tex = [r"\begin{table}[H]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\small",
           r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{" + colspec + "}", r"\toprule", header + r" \\", r"\midrule"]
    tex += body
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(path, "w").write("\n".join(tex))
    print(f"  wrote {path} ({sum(1 for b in body if b.endswith(chr(92) * 2))} rows)")

ARMS = [("weak", dict(form="weak", ic=1.0, dual=False)), ("weak, dual norm", dict(form="weak", ic=1.0, dual=True)),
        ("weak, IC off", dict(form="weak", ic=0.0, dual=False)), ("strong", dict(form="strong", ic=1.0, dual=False)),
        ("strong, IC off", dict(form="strong", ic=0.0, dual=False))]

def table_over_time(R, dim, out):
    body = []
    for ci, case in enumerate(("compat", "clash")):
        if ci:
            body.append(r"\midrule")
        first = True
        lab = "matching" if case == "compat" else "clashing"
        st = sel(R, dim=dim, case=case, init="pre")
        if st:
            body.append(f"{lab} & supervised start & " + " & ".join(errs(st)) + f" & {t1_range(st)} \\\\"); first = False
        for init in ("warm", "cold"):
            for name, s in ARMS:
                rs = sel(R, dim=dim, case=case, init=init, band=0.0, **s)
                if rs:
                    body.append(f"{lab if first else ''} & {init}, {name} & " + " & ".join(errs(rs)) + f" & {t1_range(rs)} \\\\")
                    first = False
    write(os.path.join(out, f"app_heat_{dim}d.tex"),
          f"Heat {dim}D, default weights: relative error of the transient, mean over three seeds, and the range over seeds "
          "at $t=1$. Warm runs start from the supervised fit, cold runs from a random initialization; IC off trains on the "
          "physics term alone.",
          f"tab:app-heat-{dim}d", "llcccccc", f"IC & Arm & {HEAD} & range at $t=1$", body)

def table_terms(R, out):
    body = []
    groups = [(1, "warm"), (1, "cold"), (2, "cold")]
    for gi, (dim, init) in enumerate(groups):
        if gi:
            body.append(r"\addlinespace")
        if init == "warm":
            st = sel(R, dim=dim, case="clash", init="pre")
            if st:
                body.append(f"{dim}D, supervised start & " + " & ".join(
                    range_cell([r["terms"][k] for r in st]) for k in ("ic", "strong")) + " & " + " & ".join(
                    range_cell([r["err"][t] for r in st]) for t in ("t=0.01", "t=1.0")) + r" \\")
        for name, s in (("weak", dict(form="weak", dual=False)), ("weak, dual norm", dict(form="weak", dual=True)),
                        ("strong", dict(form="strong", dual=False))):
            rs = sel(R, dim=dim, case="clash", init=init, ic=1.0, band=0.0, **s)
            if rs:
                body.append(f"{dim}D, {init}, {name} & " + " & ".join(
                    range_cell([r["terms"][k] for r in rs]) for k in ("ic", "strong")) + " & " + " & ".join(
                    range_cell([r["err"][t] for r in rs]) for t in ("t=0.01", "t=1.0")) + r" \\")
    write(os.path.join(out, "app_heat_terms.tex"),
          "Loss terms under the clash, 1D (warm and cold starts) and 2D (cold start; the 2D warm start is Table~\\ref{tab:giveup}): "
          "mean [min--max] over three seeds. The pointwise PDE residual is $12$ times the mean of the squared residual on fixed "
          "random points ($1{,}280$ in 1D, $16{,}000$ in 2D).",
          "tab:app-heat-terms", "lcccc", r"Model & IC loss & pointwise PDE residual & error $t=0.01$ & error $t=1$", body)

def table_band(R, out):
    banded = [r for r in R if r["band"] > 0 and not r["dual"]]
    if not banded:
        print("  no band-masked runs found: app_heat_band.tex not written"); return
    widths = sorted({r["band"] for r in banded})
    body, first_block = [], True
    for dim in sorted({r["dim"] for r in banded}):
        for case in ("clash", "compat"):
            for init in ("warm", "cold"):
                for form in ("weak", "strong"):
                    if not [r for r in banded if r["dim"] == dim and r["case"] == case and r["init"] == init and r["form"] == form]:
                        continue
                    if not first_block:
                        body.append(r"\addlinespace")
                    first_block = False
                    lab = f"{dim}D {'clashing' if case == 'clash' else 'matching'}, {init}, {form}"
                    for i, b in enumerate([0.0] + widths):
                        rs = sel(R, dim=dim, case=case, init=init, form=form, ic=1.0, band=b, dual=False)
                        if rs:
                            body.append(f"{lab if i == 0 else ''} & {b:g} & " + " & ".join(errs(rs)) + f" & {t1_range(rs)} \\\\")
    write(os.path.join(out, "app_heat_band.tex"),
          "Masked initial condition: initial-condition points within a band $b$ of the boundary are dropped ($b=0$ is the "
          "default). Relative error of the transient, mean over three seeds, and the range over seeds at $t=1$.",
          "tab:app-heat-band", "lccccccc", f"Problem, start, form & $b$ & {HEAD} & range at $t=1$", body)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--out", default=os.path.join("results", "paper_figs"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = load(a.root)
    print(f"loaded {len(R)} heat runs")
    table_over_time(R, 1, a.out); table_over_time(R, 2, a.out); table_terms(R, a.out); table_band(R, a.out)