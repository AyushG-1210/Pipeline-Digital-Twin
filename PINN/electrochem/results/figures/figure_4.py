"""
Figure 4 -- where the gradient goes.

(a) Per-term gradient-norm share on the headline weak family (dn_plain_*),
    four configurations: unanchored / anchored x production / unit weights.
    Bar = median over n=5 seeds, whisker = [min, max]. D=50 draws per seed.
(b) Interface share per seed: weak Butler-Volmer flux path (transient and
    stiffness detached) vs strong bc_C penalty. Lines pair the same seed.

Inputs (both must exist; nothing is hardcoded):
    per_term_share.json   -- written by per_term_share.py --full
    interface_share.json  -- written in the notebook, see README note below

    python fig4_gradient_share.py [--pts per_term_share.json]
                                  [--iface interface_share.json]
                                  [--out fig4_gradient_share.pdf]
"""
import json, argparse, statistics as st
import numpy as np
import matplotlib.pyplot as plt

ARMS = [  # (json key, legend label, colour, hatch)
    ("dn_plain_noanchor",              "Unanchored",               "#4C72B0", ""),
    ("dn_plain_noanchor_unitweights",  "Unanchored, unit weights", "#4C72B0", "///"),
    ("dn_plain_anchored",              "Anchored",                 "#DD8452", ""),
    ("dn_plain_anchored_unitweights",  "Anchored, unit weights",   "#DD8452", "///"),
]
TERMS = ["res_C", "res_phi", "bounds", "ic_C", "ends_C", "smooth", "ends_phi", "ANCHOR"]
TERM_LABEL = {"res_C": r"res$_C$", "res_phi": r"res$_\phi$", "bounds": "bounds",
              "ic_C": r"IC$_C$", "ends_C": r"ends$_C$", "smooth": "smooth",
              "ends_phi": r"ends$_\phi$", "ANCHOR": "anchor"}


def panel_a(ax, pts):
    x = np.arange(len(TERMS))
    w = 0.2
    for i, (key, lab, col, hat) in enumerate(ARMS):
        arm = pts["arms"][key]
        assert arm["n"] == 5, f"{key}: n={arm['n']}, expected 5"
        sp = arm["share_pct"]
        for j, t in enumerate(TERMS):
            if t not in sp:          # anchor excluded from unanchored denominators
                continue
            med, lo, hi = sp[t]["median"], sp[t]["min"], sp[t]["max"]
            xpos = x[j] + (i - 1.5) * w
            ax.bar(xpos, med, w, color=col if not hat else "white",
                   edgecolor=col, hatch=hat, linewidth=1.0,
                   label=lab if j == 0 else None)
            ax.plot([xpos, xpos], [lo, hi], color="0.2", lw=0.8)
    ax.set_xticks(x, [TERM_LABEL[t] for t in TERMS])
    ax.set_ylabel("Share of total gradient norm (%)")
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.72, 0.97, "no anchor bar for unanchored arms:\nterm excluded from their denominator",
            transform=ax.transAxes, ha="center", va="top", fontsize=7, color="0.35")
    ax.set_title("(a) Per-term share, weak form", loc="left", fontsize=10)


def panel_b(ax, iface):
    weak, strong = iface["weak_flux_path"], iface["strong_bc_C"]
    seeds = sorted(weak, key=int)
    assert sorted(strong, key=int) == seeds and len(seeds) == 5, "seed sets differ"
    jit = np.linspace(-0.08, 0.08, len(seeds))   # small fixed offset so tight points don't stack
    for k, s in enumerate(seeds):
        ax.plot([0 + jit[k], 1 + jit[k]], [strong[s], weak[s]], color="0.75", lw=0.8, zorder=1)
    ax.scatter(0 + jit, [strong[s] for s in seeds], color="#55A868", marker="s",
               zorder=2, label=r"Strong: $w_{\mathrm{BC}}\,\mathrm{bc}_C$")
    ax.scatter(1 + jit, [weak[s] for s in seeds], color="#4C72B0", marker="o",
               zorder=2, label="Weak: flux path of res$_C$")
    for xi, d in ((0, strong), (1, weak)):
        m = st.median(d.values())
        ax.plot([xi - 0.18, xi + 0.18], [m, m], color="k", lw=1.6, zorder=3)
    ax.set_xticks([0, 1], ["Strong", "Weak"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 50)
    ax.set_ylabel("Interface share of gradient norm (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    ax.set_title("(b) Interface share, unanchored", loc="left", fontsize=10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pts", default="per_term_share.json")
    ap.add_argument("--iface", default="interface_share.json")
    ap.add_argument("--out", default="fig4_gradient_share.pdf")
    a = ap.parse_args()

    pts = json.load(open(a.pts))
    iface = json.load(open(a.iface))
    assert pts["draws"] == 50 and iface.get("draws") == 50, "expected D=50 on both inputs"

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10, 3.6),
                                   gridspec_kw={"width_ratios": [3, 1]})
    panel_a(axa, pts)
    panel_b(axb, iface)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")

    # print what was plotted, per seed, so the figure is checkable against its inputs
    print(f"saved {a.out}")
    for s in sorted(iface["weak_flux_path"], key=int):
        print(f"  seed {s}: strong {iface['strong_bc_C'][s]:5.2f}%  "
              f"weak {iface['weak_flux_path'][s]:5.2f}%")