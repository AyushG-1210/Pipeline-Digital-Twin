"""
run_mesh.py -- does the measured k predict where the weak form fails when only its mesh changes? (1D heat, clash)

    python run_mesh.py                       # from the repo root; uses all CPU cores
    python run_mesh.py --jobs 9              # parallel runs per launcher (strong and each weak mesh run at once)
    python run_mesh.py --starts heat_runs/heat_results/ckpt --out heat_runs/heat_mesh

For each weak-form mesh N (elements in 1D; the paper uses 16):
  1. measure k(N) at the supervised starts (no training)
  2. train the weak form at its DEFAULT weight on mesh N, 3 seeds
  3. train the strong form (unchanged, default mesh) at x k(N), 3 seeds
Rule fixed before running: on every mesh, the weak form is within 0.02 of the strong form at x k(N) at every time.
Finished runs are skipped, so the script is safe to restart. Works on Windows, Linux and Colab; CPU is fine.
On one thread a weak run takes about 3 minutes and a strong run about 10.
"""
import argparse, json, os, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument("--jobs", type=int, default=os.cpu_count() or 2, help="parallel runs per launcher (default: CPU cores)")
p.add_argument("--out", default="heat_runs/heat_mesh")
p.add_argument("--starts", default=os.path.join("heat_runs", "heat_results", "ckpt"), help="folder with the d1_clash_start_s*.pt files")
p.add_argument("--meshes", type=int, nargs="+", default=[4, 8, 32], help="weak-form meshes to train on (16 is the paper's)")
p.add_argument("--D", type=int, default=10, help="draws per gradient measurement")
p.add_argument("--smoke", action="store_true", help="tiny check run")
a = p.parse_args()

HD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heat_drift.py")
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")        # one thread per run; parallelism is across runs
base = [sys.executable, "-u", HD, "--dims", "1", "--out", a.out, "--starts", a.starts]
if a.smoke:
    base.append("--smoke")
os.makedirs(a.out, exist_ok=True)

def run(args, log):
    print(" ".join(args[3:]), flush=True)
    with open(os.path.join(a.out, log), "w") as fh:
        r = subprocess.run(args, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        print(f"  FAILED, see {os.path.join(a.out, log)}", flush=True)
    return r.returncode

# 1. k(N) for every mesh, including the paper's 16
ks = {}
for n in sorted(set(a.meshes) | {16}):
    run(base + ["--measure", "--inits", "warm", "--n-el", str(n), "--D", str(a.D)], f"measure_e{n}.log")
    f = os.path.join(a.out, f"matched_measure_e{n}.json")
    if not os.path.exists(f):
        sys.exit(f"measurement for mesh {n} failed")
    ks[n] = json.load(open(f))["d1_warm"]["k_rounded"]
print("k per mesh: " + ", ".join(f"{n}: {k:g}" for n, k in ks.items()), flush=True)

# 2. strong form at x k(N) and the weak form at its default weight on every mesh, all running at once
procs = []
def launch(args, log):
    fh = open(os.path.join(a.out, log), "w")
    print("start: " + " ".join(args[3:]), flush=True)
    procs.append((log, subprocess.Popen(args, env=ENV, stdout=fh, stderr=subprocess.STDOUT), fh))

launch(base + ["--sweep", "--sweep-form", "strong", "--mults"] + [f"{ks[n]:g}" for n in a.meshes]
       + ["--jobs", str(a.jobs), "--retries", "1"], "strong.log")
for n in a.meshes:
    launch(base + ["--sweep", "--sweep-form", "weak", "--mults", "1", "--n-el", str(n),
                   "--jobs", str(a.jobs), "--retries", "1"], f"weak_e{n}.log")
for log, pr, fh in procs:
    pr.wait(); fh.close()
    print(f"finished {log}" + ("" if pr.returncode == 0 else f" (exit {pr.returncode}, see {os.path.join(a.out, log)})"), flush=True)

# 4. summary
with open(os.path.join(a.out, "summary.txt"), "w") as fh:
    subprocess.run(base[:3] + ["--summary", "--out", a.out], env=ENV, stdout=fh)
print(open(os.path.join(a.out, "summary.txt")).read())
print("k per mesh: " + ", ".join(f"{n}: {k:g}" for n, k in ks.items()))