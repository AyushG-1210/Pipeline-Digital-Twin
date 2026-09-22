"""
Task 1: medians and ranges for the modes-5 arm (n=5, seeds 42-46) and the
test-space-size comparison against the modes-2 arm at n=5.

Sources (all under results/json_files/weak_form/):
  s4{2..6}a.json              modes-5 base records: "adam" (200+600 ep, bs 64 = 8,800
                              steps) and "lbfgs" (250 L-BFGS steps from that Adam point)
  m5_s4{2..6}a_lbfgs100.json  modes-5 repolished: 100 L-BFGS steps from the SAME Adam
                              checkpoint (m5_s4Xa_adam.pt -- naming: JSON s42a.json
                              <-> checkpoint m5_s42a_adam.pt)
  m2_bs64_full_s4{2..6}.json  modes-2, n=5: 300+900 ep, bs 64 = 13,200 Adam steps,
                              "adam" and "lbfgs" (= 100 L-BFGS steps). This is the
                              modes-2 arm to compare against. m2_s42a.json is the
                              orphaned single-seed run at the same config; its Adam
                              checkpoint IS m2_bs64_full_s42 (cell 29 copies it), so
                              it adds nothing and is not used here.

Comparison (the only one the data supports): per metric, do the two n=5 seed
ranges overlap, and does each arm's median fall inside the other arm's range?
Remaining confound: Adam budget, 13,200 (modes-2) vs 8,800 (modes-5) steps.

Run from PINN/:  python results/figures/pull_m5_stats.py
"""
import json, os, statistics as st

ROOT = os.path.join(os.path.dirname(__file__), "..", "json_files", "weak_form")
SEEDS = [42, 43, 44, 45, 46]
HOLD_KEYS = ["C_l2", "phi_l2", "flux_err", "op_corr", "x_corr", "x_var_ratio"]
WEAK_KEYS = ["res_C", "res_phi", "bounds", "smooth", "weak_total"]
SHARE_KEYS = ["ANCHOR", "weak"]
LOWER_BETTER = {"C_l2", "phi_l2", "flux_err", "res_C", "res_phi", "bounds", "smooth", "weak_total"}
HIGHER_BETTER = {"op_corr", "x_corr"}


def load(name):
    with open(os.path.join(ROOT, name)) as f:
        return json.load(f)


base = {s: load(f"s{s}a.json") for s in SEEDS}
repo = {s: load(f"m5_s{s}a_lbfgs100.json") for s in SEEDS}
m2 = {s: load(f"m2_bs64_full_s{s}.json") for s in SEEDS}

for s in SEEDS:
    assert repo[s]["n_modes"] == 5 and repo[s]["seed"] == s and repo[s]["tag"] == f"m5_s{s}a"
    assert base[s]["cfg"]["n_modes"] == 5 and base[s]["cfg"]["lbfgs_steps"] == 250
    assert m2[s]["cfg"]["n_modes"] == 2 and m2[s]["cfg"]["grad_steps"] == 13200 and m2[s]["cfg"]["lbfgs_steps"] == 100

# phase -> split -> seed -> block
M5 = {"adam":     {sp: {s: base[s]["adam"][sp]  for s in SEEDS} for sp in ("hold", "train", "weak", "share")},
      "lbfgs250": {sp: {s: base[s]["lbfgs"][sp] for s in SEEDS} for sp in ("hold", "train", "weak", "share")},
      "lbfgs100": {sp: {s: repo[s][sp]          for s in SEEDS} for sp in ("hold", "train", "weak", "share")}}
M2 = {"adam":     {sp: {s: m2[s]["adam"][sp]  for s in SEEDS} for sp in ("hold", "train", "weak", "share")},
      "lbfgs100": {sp: {s: m2[s]["lbfgs"][sp] for s in SEEDS} for sp in ("hold", "train", "weak", "share")}}


def stats(vals):
    return {"median": st.median(vals), "min": min(vals), "max": max(vals), "range": max(vals) - min(vals)}


def keys_for(split):
    return HOLD_KEYS if split in ("hold", "train") else WEAK_KEYS if split == "weak" else SHARE_KEYS


def fmt(v, k):
    if k in SHARE_KEYS:
        return f"{v:7.1f}"
    if k in WEAK_KEYS:
        return f"{v:9.3e}"
    return f"{v:8.4f}"


def summarize(arm):
    out = {}
    for phase, splits in arm.items():
        out[phase] = {}
        for sp, per_seed in splits.items():
            out[phase][sp] = {}
            for k in keys_for(sp):
                vals = [per_seed[s][k] for s in SEEDS]
                d = {**stats(vals), "per_seed": dict(zip(SEEDS, vals))}
                # skew flag: one seed sits outside the range of the other four by more
                # than TWICE that four-seed span (pulls min or max far from the pack)
                for s in SEEDS:
                    rest = [per_seed[t][k] for t in SEEDS if t != s]
                    span = max(rest) - min(rest)
                    if span > 0 and (per_seed[s][k] < min(rest) - 2 * span or per_seed[s][k] > max(rest) + 2 * span):
                        d["outlier_seed"] = s
                        d["others"] = stats(rest)
                out[phase][sp][k] = d
    return out


S5, S2 = summarize(M5), summarize(M2)


def print_phase(S, phase, title):
    print("\n" + "=" * 104 + f"\n {title}\n" + "=" * 104)
    for sp in ("hold", "train", "weak", "share"):
        print(f"\n  [{sp}]")
        print(f"  {'metric':<12}{'median':>11}{'min':>11}{'max':>11}{'range':>11}   per seed (42..46)")
        for k in keys_for(sp):
            d = S[phase][sp][k]
            ps = "  ".join(fmt(d["per_seed"][s], k).strip() for s in SEEDS)
            flag = ""
            if "outlier_seed" in d:
                o = d["others"]
                flag = (f"\n  {'':<12}  ^ SKEWED: seed {d['outlier_seed']} = {fmt(d['per_seed'][d['outlier_seed']], k).strip()}; "
                        f"other four: median {fmt(o['median'], k).strip()} [{fmt(o['min'], k).strip()}-{fmt(o['max'], k).strip()}]")
            print(f"  {k:<12}{fmt(d['median'], k):>11}{fmt(d['min'], k):>11}{fmt(d['max'], k):>11}{fmt(d['range'], k):>11}   {ps}{flag}")


print_phase(S5, "adam",     "MODES=5, n=5 -- ADAM (8,800 steps)                source: s4Xa.json['adam']")
print_phase(S5, "lbfgs250", "MODES=5, n=5 -- ADAM + L-BFGS 250 (base record)   source: s4Xa.json['lbfgs']")
print_phase(S5, "lbfgs100", "MODES=5, n=5 -- ADAM + L-BFGS 100 (repolished)    source: m5_s4Xa_lbfgs100.json")
print_phase(S2, "adam",     "MODES=2, n=5 -- ADAM (13,200 steps)               source: m2_bs64_full_s4X.json['adam']")
print_phase(S2, "lbfgs100", "MODES=2, n=5 -- ADAM + L-BFGS 100                 source: m2_bs64_full_s4X.json['lbfgs']")

# ------------------------------------------------------------- the comparison
PAIRINGS = [("adam", "adam", "ADAM: modes-2 13,200 steps vs modes-5 8,800 steps"),
            ("lbfgs100", "lbfgs100", "ADAM + L-BFGS 100 (matched polish budget): modes-2 vs modes-5 repolished")]
check = {}
for p2, p5, title in PAIRINGS:
    print("\n" + "=" * 104 + f"\n DOES TEST-SPACE SIZE SHOW UP?  n=5 vs n=5  --  {title}\n" + "=" * 104)
    check[p2] = {}
    for sp in ("hold", "train", "weak", "share"):
        print(f"\n  [{sp}]")
        print(f"  {'metric':<12}{'m2 median [min-max]':>28}{'m5 median [min-max]':>28}   verdict")
        check[p2][sp] = {}
        for k in keys_for(sp):
            a, b = S2[p2][sp][k], S5[p5][sp][k]
            overlap = not (a["max"] < b["min"] or a["min"] > b["max"])
            m2_in_m5 = b["min"] <= a["median"] <= b["max"]
            m5_in_m2 = a["min"] <= b["median"] <= a["max"]
            if not overlap:
                side = "lower" if a["max"] < b["min"] else "higher"
                if k in LOWER_BETTER:
                    dirn = "modes-2 BETTER" if side == "lower" else "modes-2 WORSE"
                elif k in HIGHER_BETTER:
                    dirn = "modes-2 BETTER" if side == "higher" else "modes-2 WORSE"
                else:
                    dirn = f"modes-2 {side}"
                verdict = f"DISJOINT ranges -> distinguishes; {dirn}"
            elif m2_in_m5 and m5_in_m2:
                verdict = "ranges overlap, each median inside the other's range -> does not distinguish"
            else:
                which = "m2 median inside m5 range" if m2_in_m5 else "m5 median inside m2 range" if m5_in_m2 else "neither median inside the other's range"
                side = "lower" if a["median"] < b["median"] else "higher"
                verdict = f"ranges overlap; {which} -> weak evidence at most (modes-2 median {side})"
            check[p2][sp][k] = {"m2": {kk: a[kk] for kk in ("median", "min", "max")},
                                "m5": {kk: b[kk] for kk in ("median", "min", "max")},
                                "ranges_overlap": overlap, "m2_median_in_m5_range": m2_in_m5,
                                "m5_median_in_m2_range": m5_in_m2, "verdict": verdict}
            print(f"  {k:<12}{fmt(a['median'], k).strip():>9} [{fmt(a['min'], k).strip()}-{fmt(a['max'], k).strip()}]"
                  f"{'':>4}{fmt(b['median'], k).strip():>9} [{fmt(b['min'], k).strip()}-{fmt(b['max'], k).strip()}]   {verdict}")

out = {"sources": {"modes5_base": "s4{2..6}a.json", "modes5_repolished": "m5_s4{2..6}a_lbfgs100.json",
                   "modes2": "m2_bs64_full_s4{2..6}.json"},
       "confound": "Adam budget 13,200 (modes-2) vs 8,800 (modes-5) steps; L-BFGS budget matched at 100 in the polished pairing",
       "modes5_summary": S5, "modes2_summary": S2, "comparison": check}
dst = os.path.join(os.path.dirname(__file__), "m5_vs_m2_comparison.json")
with open(dst, "w") as f:
    json.dump(out, f, indent=1)
print(f"\nsaved {os.path.relpath(dst)}")
