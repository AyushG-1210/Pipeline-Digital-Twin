"""anneal_weights.py -- print where the annealed IC weight ended up, per run. CPU only, reads JSON.
    python anneal_weights.py --out heat_runs/heat_matched"""
import argparse, glob, json, os
p = argparse.ArgumentParser(); p.add_argument("--out", default="heat_runs/heat_matched"); a = p.parse_args()
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.out, "results", "*_anneal_*.json")))]
print(f"{'run':<40}{'final IC weight':>16}{'x default (3)':>15}   history (step: weight)")
for r in rows:
    h = r.get("ic_weight_hist", [])
    pick = h[:: max(1, len(h) // 5)] + ([h[-1]] if h else [])
    print(f"{r['name']:<40}{r['ic_weight_final']:>16.3e}{r['ic_weight_final'] / 3:>15.3g}   "
          + "  ".join(f"{s}:{w:.2g}" for s, w in pick))