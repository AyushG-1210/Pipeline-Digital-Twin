"""
run_final.py -- the last three checks, as one session. Run from the repo root (GPU recommended; 1D only, so a T4 works).

    python run_final.py                       # all three stages, 24 runs at a time
    python run_final.py --stages A B          # skip advection-diffusion
    python run_final.py --jobs 12             # fewer parallel runs (if RAM is tight)

STAGE A  normalization.  1D heat, clashing IC, warm start. For each weak-form mesh N in {4, 8, 16, 32}:
         measure k with the weak residual divided by the element measure (k_norm), train the normalized weak form at its
         default weight, and train the strong form at x k_norm(N).
         Rules fixed before running:
           (A1) on every mesh, the normalized weak form is within 0.02 of the strong form at x k_norm(N), at every time;
           (A2) the normalized weak form moves less with the mesh than the unnormalized one: over the four meshes, the
                largest range across meshes of the mean error at any time is below 0.089 (the unnormalized value).
STAGE B  test-space order at fixed mesh (16 elements). For p in {2, 4, 10}: measure k(p), train the weak form at its
         default weight, and the strong form at x k(p). The paper's default is p = 6.
         Rule: k varies by less than 2x across p if k is set by element size and not by test-space richness; the same
         0.02 match rule as above is applied at each p.
STAGE C  advection-diffusion, u_t + u_x = 0.1 u_xx, 1D, exact reference. Supervised starts, defaults for the clashing and
         the matching IC, k, strong form at k x {0.1, 0.3, 1, 3, 10}, weak form at (1/k) x {0.1, 0.3, 1, 3, 10}, and the
         matching-IC control (strong form at x k). Rules fixed before running:
           (C1) matching IC: both defaults, and the strong form at x k, stay within 0.02 of their supervised start;
           (C2) clashing IC: the strong default is worse than the weak default at t = 0.01, and the weak default is worse
                than the strong default at t = 1;
           (C3) some strong multiplier and some weak multiplier land within 0.02 of the other form's default at every
                time, within a factor of 3 of k and of 1/k respectively.
Every stage is restartable: finished runs are skipped. Results go to heat_runs/heat_final/ (A, B) and heat_runs/heat_adv/ (C).
"""
import argparse, json, os, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument("--jobs", type=int, default=24, help="parallel runs (split across the invocations of a stage)")
p.add_argument("--stages", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
p.add_argument("--out", default="heat_runs/heat_final")
p.add_argument("--out-adv", default="heat_runs/heat_adv")
p.add_argument("--starts", default=os.path.join("heat_runs", "heat_results", "ckpt"), help="existing 1D heat starts (stages A and B)")
p.add_argument("--meshes", type=int, nargs="+", default=[4, 8, 16, 32])
p.add_argument("--orders", type=int, nargs="+", default=[2, 4, 10])
p.add_argument("--adv", type=float, default=1.0)
p.add_argument("--D", type=int, default=10)
p.add_argument("--smoke", action="store_true")
a = p.parse_args()

HD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heat_drift.py")
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
BASE = [sys.executable, "-u", HD] + (["--smoke"] if a.smoke else [])
os.makedirs(a.out, exist_ok=True)

def run(args, log, out=None):
    out = out or a.out
    print("  " + " ".join(args[len(BASE):]), flush=True)
    with open(os.path.join(out, log), "w") as fh:
        return subprocess.run(BASE + args, env=ENV, stdout=fh, stderr=subprocess.STDOUT).returncode

def par(items):
    """Run (args, log) items at the same time and wait for all of them."""
    procs = []
    for args, log in items:
        print("  start: " + " ".join(args), flush=True)
        procs.append((log, subprocess.Popen(BASE + args, env=ENV, stdout=open(os.path.join(a.out, log), "w"),
                                            stderr=subprocess.STDOUT)))
    for log, pr in procs:
        pr.wait()
        print(f"  finished {log}" + ("" if pr.returncode == 0 else f"  (exit {pr.returncode}, see {a.out}/{log})"), flush=True)

def measure(tag, extra):
    """k at the supervised starts for one setting; returns k_rounded."""
    run(["--measure", "--dims", "1", "--inits", "warm", "--out", a.out, "--starts", a.starts, "--D", str(a.D)] + extra,
        f"measure{tag}.log")
    f = os.path.join(a.out, f"matched_measure{tag}.json")
    if not os.path.exists(f):
        sys.exit(f"measurement {tag} failed, see {a.out}/measure{tag}.log")
    return json.load(open(f))["d1_warm"]["k_rounded"]

common = ["--dims", "1", "--out", a.out, "--starts", a.starts, "--retries", "1"]

if "A" in a.stages:
    print("\n=== STAGE A: normalization ===", flush=True)
    kn = {n: measure(f"_e{n}_norm", ["--n-el", str(n), "--normalize"]) for n in a.meshes}
    print("k_norm per mesh: " + ", ".join(f"{n}: {k:g}" for n, k in kn.items()), flush=True)
    per_weak = max(1, (a.jobs // 2) // len(a.meshes))
    items = [(["--sweep", "--sweep-form", "weak", "--mults", "1", "--n-el", str(n), "--normalize",
               "--jobs", str(per_weak)] + common, f"A_weak_e{n}.log") for n in a.meshes]
    items.append((["--sweep", "--sweep-form", "strong", "--mults"] + [f"{kn[n]:g}" for n in a.meshes]
                  + ["--jobs", str(max(1, a.jobs - per_weak * len(a.meshes)))] + common, "A_strong.log"))
    par(items)

if "B" in a.stages:
    print("\n=== STAGE B: test-space order at fixed mesh ===", flush=True)
    kp = {q: measure(f"_p{q}", ["--p-deg", str(q)]) for q in a.orders}
    print("k per order: " + ", ".join(f"p={q}: {k:g}" for q, k in kp.items()) + "   (p=6 default: see the mesh check)", flush=True)
    per_weak = max(1, (a.jobs // 2) // len(a.orders))
    items = [(["--sweep", "--sweep-form", "weak", "--mults", "1", "--p-deg", str(q), "--jobs", str(per_weak)] + common,
              f"B_weak_p{q}.log") for q in a.orders]
    items.append((["--sweep", "--sweep-form", "strong", "--mults"] + [f"{kp[q]:g}" for q in a.orders]
                  + ["--jobs", str(max(1, a.jobs - per_weak * len(a.orders)))] + common, "B_strong.log"))
    par(items)

if a.stages and {"A", "B"} & set(a.stages):
    with open(os.path.join(a.out, "summary.txt"), "w") as fh:
        subprocess.run(BASE + ["--summary", "--out", a.out], env=ENV, stdout=fh)
    print(f"\nA/B summary: {a.out}/summary.txt", flush=True)

if "C" in a.stages:
    print("\n=== STAGE C: advection-diffusion ===", flush=True)
    os.makedirs(a.out_adv, exist_ok=True)
    with open(os.path.join(a.out_adv, "batch.log"), "w") as fh:
        r = subprocess.run(BASE + ["--adv-batch", "--adv", str(a.adv), "--jobs", str(a.jobs), "--retries", "2",
                                   "--D", str(a.D), "--out", a.out_adv], env=ENV, stdout=fh, stderr=subprocess.STDOUT)
    print(f"  advection-diffusion batch finished (exit {r}); log and summary: {a.out_adv}/batch.log", flush=True)

print("\nDone. Send me: heat_runs/heat_final/summary.txt, the 'k_norm per mesh' and 'k per order' lines above, and the end of "
      "heat_runs/heat_adv/batch.log (it ends with the summary table).", flush=True)