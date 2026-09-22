"""
Figure 5 -- held-out C rel-L2, every seed of every arm.

One column per arm, one dot per seed, black tick at the median. Dashed line at
1.0: the error of predicting C = 0 everywhere.

Every value is read from a per-run results JSON; nothing is typed in.
Fill ARMS below with the real paths. A listed path that does not exist is an
error, never a silent skip. An arm with an EMPTY list is left out with a
warning.

Da=0 is deliberately not one of the ARMS: it is scored on a different problem
(Da=0 held-out set) at a different eval time than the t=1 reacting arms below,
so it does not belong on this axis. It stays in the text/Limitations instead.

Each per-run JSON is expected to hold the metric at KEY_PATH, i.e.
r["adam"]["hold"]["C_l2"]. The Da=0 run must be evaluated against the Da=0
held-out references (fdm_hold_da0.npz) and saved in the same format.

    python fig5_perseed_crel.py [--out fig5_perseed_crel.pdf]

Also writes fig5_values.json: the exact per-seed numbers plotted, with the
file each came from.
"""
import json, os, argparse, statistics as st
import numpy as np
import matplotlib.pyplot as plt

KEY_PATH = ("hold", "C_l2")
SEEDS = range(42, 47)

# TODO: set each directory to where that family's per-run JSONs actually live.
SF_DIR = "results/json_files/strong_form"     # strong arms
RUN_DIR = "results/json_files/weak_form"      # plain weak arms
DN_DIR = "results/json_files/weak_form"   # dual-norm arms

ARMS = [  # (label, [per-seed json paths], colour, marker)
    ("Strong\nunanchored",  [f"{SF_DIR}/sa_wbc1_noanchor_s{s}.json" for s in SEEDS], "#55A868", "s"),
    # Anchored strong-form, w_bc=1: sf_wbc1_s{seed}.json (confirmed by value against
    # Table 3: s42=0.1136, s43=0.1407, s44=0.1352, s45=0.1251, s46=0.1224).
    ("Strong\nanchored",    [f"{SF_DIR}/sf_wbc1_s{s}.json" for s in SEEDS], "#55A868", "D"),
    ("Weak\nanchored",      [f"{RUN_DIR}/dn_plain_anchored_s{s}.json" for s in SEEDS], "#DD8452", "D"),
    ("Weak\nunanchored",    [f"{RUN_DIR}/dn_plain_noanchor_s{s}.json" for s in SEEDS], "#4C72B0", "o"),
    ("Dual-norm\nanchored", [f"{DN_DIR}/dn_dual_anchored_s{s}.json" for s in SEEDS], "#DD8452", "^"),
    ("Dual-norm\nunanchored", [f"{DN_DIR}/dn_dual_noanchor_s{s}.json" for s in SEEDS], "#4C72B0", "^"),
]


def read_metric(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} -- fix the path in ARMS; not skipping silently")
    r = json.load(open(path))
    # Schema drift in results/json_files/strong_form/sa_wbc1_noanchor_*: seeds 42-43 keep
    # "hold" at the top level, seeds 44-46 nest it under "adam" (cfg.lbfgs is false in
    # both, so it's the same Adam-stage metric either way). Unwrap "adam" if present so
    # KEY_PATH resolves the same value regardless of which shape a given file uses.
    if "adam" in r and KEY_PATH[0] != "adam":
        r = r["adam"]
    for k in KEY_PATH:
        if k not in r:
            raise KeyError(f"{path}: missing key {k!r} in {KEY_PATH}")
        r = r[k]
    return float(r)


def main(out):
    plotted, record = [], {}
    for label, paths, col, mk in ARMS:
        if not paths:
            print(f"[skip] {label.replace(chr(10), ' ')}: no paths listed")
            continue
        vals = [read_metric(p) for p in paths]
        assert len(vals) == 5, f"{label}: {len(vals)} seeds, expected 5"
        plotted.append((label, vals, col, mk))
        record[label.replace("\n", " ")] = {
            "n": len(vals), "median": st.median(vals), "min": min(vals), "max": max(vals),
            "per_seed": {str(s): {"C_l2": v, "file": p}
                         for s, v, p in zip(SEEDS, vals, paths)}}

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8, 3.8))
    jit = np.linspace(-0.12, 0.12, 5)
    for i, (label, vals, col, mk) in enumerate(plotted):
        ax.scatter(i + jit, vals, color=col, marker=mk, s=36, zorder=2,
                   edgecolor="white", linewidth=0.5)
        m = st.median(vals)
        ax.plot([i - 0.25, i + 0.25], [m, m], color="k", lw=1.6, zorder=3)
    ax.axhline(1.0, color="0.3", ls="--", lw=1.0, zorder=1)
    ax.text(len(plotted) - 0.5, 1.0, "predicting $C=0$ everywhere",
            ha="right", va="bottom", fontsize=8, color="0.3")
    ax.set_xticks(range(len(plotted)), [p[0] for p in plotted], fontsize=8)
    ax.set_xlim(-0.6, len(plotted) - 0.4)
    ax.set_ylim(0, max(1.3, max(max(p[1]) for p in plotted) * 1.08))
    ax.set_ylabel(r"Held-out $C$ rel. $L^2$ at $t=1$")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")

    vj = out.replace(".pdf", "_values.json")
    json.dump(record, open(vj, "w"), indent=2)
    print(f"saved {out} and {vj}\n")
    print(f"{'arm':<26}" + "".join(f"{'s'+str(s):>9}" for s in SEEDS) + f"{'median':>9}")
    for label, d in record.items():
        print(f"{label:<26}" + "".join(f"{d['per_seed'][str(s)]['C_l2']:>9.4f}" for s in SEEDS)
              + f"{d['median']:>9.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig5_perseed_crel.pdf")
    main(ap.parse_args().out)