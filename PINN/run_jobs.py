"""
run_jobs.py -- run extra electrochem training jobs in parallel from a terminal,
so they can run ALONGSIDE the notebook (whose kernel is busy with CELL E).

USAGE (Colab terminal, from the amorflux folder):
    python run_jobs.py --par 15

Edit JOBS below. A job is skipped if its checkpoint exists, or if its log was written
in the last 20 minutes (i.e. it is probably still running from another launcher).
Logs: results/logs/<name>.log    Checkpoints: ckpt/<name>.pt (same names the notebook uses)
"""
import os, sys, json, time, argparse, subprocess

ROOT     = "/content/drive/MyDrive/amorflux"
NB_PATH  = "/content/drive/MyDrive/Colab Notebooks/amorflux.ipynb"   # <- same notebook as CELL E
CKPT_DIR = os.path.join(ROOT, "ckpt")
LOG_DIR  = os.path.join(ROOT, "results", "logs")

# (init, form, seed, band, w_ic, variant)
#   band = 0 -> IC on the full domain;  w_ic = 1 -> production IC weight
#   variant: ""        plain run
#            "noA"     skip Phase A (same total epochs)            -- rival explanation E5
#            "pertime" per-time weak residual (no cross-time sum)  -- rival explanation E7
#            "ts72"    test space 24 elements x 3 x-modes =  72 functions (default is 36)
#            "ts144"   test space 24 elements x 6 x-modes = 144 functions
# Listed in priority order: the queue runs top to bottom.
W3 = (0.01, 0.1, 0.3)
'''JOBS = [
    # 1) IC-weight sweep, seeds 43-44 (seed 42 done) -- 12 runs
    *[(i, "weak", s, 0.0, w, "") for w in W3 for i in ("warm", "cold") for s in (43, 44)],
    # 2) test-space size: is the drift just an under-resolved test space? warm weak, IC w=1 -- 6 runs
    *[("warm", "weak", s, 0.0, 1.0, v) for v in ("ts72", "ts144") for s in (42, 43, 44)],
    # 3) rival explanations, seeds 43-44 (seed 42 done): no Phase A, per-time residual -- 4 runs
    *[("warm", "weak", s, 0.0, 1.0, v) for v in ("noA", "pertime") for s in (43, 44)],
    # 4) finer IC-weight grid around 0.1, seed 42 only (lowest priority, runs last) -- 6 runs
    *[(i, "weak", 42, 0.0, w, "") for w in (0.03, 0.05, 0.2) for i in ("warm", "cold")],
]'''
JOBS = [
    ("warm", "strong", 42, 0.0, 71,  ""),
    ("warm", "strong", 43, 0.0, 71,  ""),
    ("warm", "strong", 44, 0.0, 71,  ""),
    ("warm", "strong", 44, 0.0, 210, ""),
]

# (num_elements_d, num_modes_x, x-quadrature points). Default notebook test space: (12, 2, 8) = 36 functions.
TS_VARIANTS = {"ts72": (24, 2, 8), "ts144": (24, 5, 12)}

def job_name(init, form, seed, band=0.0, w_ic=1.0, variant=""):
    tag = (f"band{band:g}_" if band > 0 else "") + (f"icw{w_ic:g}_" if w_ic != 1.0 else "")
    var = f"_{variant}" if variant else ""
    return f"{tag}{init}_{form}{var}_s{seed}"     # matches the notebook's names (warm_weak_noA_s42, band0.1_..., ...)

# Same IC penalty as the notebook, except IC points have d in [0, 1 - BAND]. BAND = 0 reproduces the original.
PENALTY_SRC = r'''
def compute_essential_penalties(model, data_tensors, N_samples=50, device=execution_device):
    b_soil, b_fluid, b_meta, _ = data_tensors
    B = b_soil.shape[0]
    pts_ic = torch.stack([
        torch.rand(N_samples, device=device),
        (1.0 - BAND) * torch.rand(N_samples, device=device),
        torch.zeros(N_samples, device=device)], dim=-1)
    coords_ic = pts_ic.unsqueeze(0).expand(B, -1, -1)
    loss_ic_C = torch.mean(model(b_soil, b_fluid, b_meta, coords_ic)[..., 0] ** 2)
    pts_ends = torch.stack([
        torch.randint(0, 2, (N_samples,), device=device, dtype=torch.float32),
        torch.rand(N_samples, device=device),
        torch.rand(N_samples, device=device)], dim=-1)
    coords_ends = pts_ends.unsqueeze(0).expand(B, -1, -1).clone().requires_grad_(True)
    u_ends = model(b_soil, b_fluid, b_meta, coords_ends)
    gC = torch.autograd.grad(u_ends[..., 0].sum(), coords_ends, create_graph=True)[0]
    gP = torch.autograd.grad(u_ends[..., 1].sum(), coords_ends, create_graph=True)[0]
    zero = torch.tensor(0.0, device=device)
    return zero, zero, loss_ic_C, torch.mean(gC[..., 0] ** 2), torch.mean(gP[..., 0] ** 2)
'''

TS_SRC = r'''
def rebuild_test_space(num_elements_d, num_modes_x, nx_quad):
    """Rebuild quadrature + test functions with a bigger test space; the trainers read these globals."""
    global quad_tensors, TE
    P, W, E = build_hp_spatial_quadrature_distance(Nx=nx_quad, num_elements_d=num_elements_d,
                                                   quad_per_elem=3, k=STRETCH_K)
    PW, WW = build_boundary_spatial_quadrature(Nx=nx_quad)
    V, GV = generate_true_hp_test_functions(P, E, num_modes_x=num_modes_x)
    VW, _ = generate_true_hp_test_functions(torch.cat([PW, torch.zeros_like(PW)], -1), E, num_modes_x=num_modes_x)
    nh = len(E) - 1
    mask = torch.zeros(V.shape[0], dtype=torch.bool)
    mask[[m * nh for m in range(num_modes_x + 1)]] = True
    assert VW[mask].abs().sum() > 0 and VW[~mask].abs().sum() == 0, "wall mask mismatch"
    quad_tensors = tuple(t.to(dev) for t in (P, W, PW, WW))
    TE = tuple(t.to(dev) for t in (V, GV, VW, mask))
    return V.shape[0]
'''

def worker(init, form, seed, band, w_ic, variant):
    """Exec the notebook's definition cells (setup .. train_strong), override the IC penalty, train, save."""
    g = {"__name__": "nbworker"}
    cells = ["".join(c["source"]) for c in json.load(open(NB_PATH))["cells"] if c["cell_type"] == "code"]
    stop = next(i for i, c in enumerate(cells) if "def train_strong(m_init" in c)
    for i, src in enumerate(cells[:stop + 1]):
        if i == 0:
            src = src.replace('drive.mount("/content/drive")', "pass")
        exec(compile(src, f"cell{i}", "exec"), g)
        if i == 0:
            g["RUN"] = {k: False for k in g["RUN"]}          # skip solver checks
    g["BAND"] = band
    exec(PENALTY_SRC, g)                                   # trainers look this up at call time
    torch, ckpt = g["torch"], g["ckpt"]
    if variant in TS_VARIANTS:
        exec(TS_SRC, g)
        print(f"test space: {g['rebuild_test_space'](*TS_VARIANTS[variant])} functions", flush=True)
    kw = {}
    if variant == "noA":
        ea, ec = g["EP"]["weak"]; kw["epochs"] = (0, ea + ec)
    elif variant == "pertime":
        kw["per_time"] = True
    assert not kw or form == "weak", "noA/pertime are weak-form variants"
    name = job_name(init, form, seed, band, w_ic, variant)
    torch.manual_seed(seed)
    m = g["new_net"]()
    if init == "warm":
        m.load_state_dict(torch.load(ckpt(f"warmstart_s{seed}"), map_location=g["dev"],
                                     weights_only=False)["state_dict"])
    fn = g["train_weak"] if form == "weak" else g["train_strong"]
    t0 = time.time()
    with g["weights"](W_ANCHOR=0.0, W_IC_C=w_ic):
        m = fn(m, seed, **kw)
    torch.save({"state_dict": m.state_dict(), "init": init, "form": form, "seed": seed,
                "band": band, "w_ic": w_ic, "variant": variant}, ckpt(name))
    print(f"saved {name} ({(time.time() - t0) / 60:.1f} min)", flush=True)

def launcher(par):
    os.makedirs(LOG_DIR, exist_ok=True)
    queue = []
    for j in JOBS:
        name = job_name(*j)
        log = os.path.join(LOG_DIR, name + ".log")
        if os.path.exists(os.path.join(CKPT_DIR, name + ".pt")):
            print(f"  skip {name} (done)")
        elif os.path.exists(log) and time.time() - os.path.getmtime(log) < 20 * 60:
            print(f"  skip {name} (log updated <20 min ago -- probably still running elsewhere)")
        else:
            queue.append(j)
    print(f"{len(queue)} to run, up to {par} at once", flush=True)
    live, t0 = [], time.time()
    while queue or live:
        while queue and len(live) < par:
            j = queue.pop(0); name = job_name(*j)
            fh = open(os.path.join(LOG_DIR, name + ".log"), "w")
            args = [*map(str, j[:5]), j[5] or "-"]
            p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--worker", *args],
                                 stdout=fh, stderr=subprocess.STDOUT)
            live.append((name, p, fh)); print(f"  start {name}", flush=True)
        time.sleep(15)
        for item in live[:]:
            name, p, fh = item
            if p.poll() is not None:
                fh.close(); live.remove(item)
                ok = p.returncode == 0
                print(f"  {'done  ' if ok else 'FAILED'} {name}  ({(time.time() - t0) / 60:.1f} min)"
                      + ("" if ok else f"  -> see results/logs/{name}.log"), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--par", type=int, default=15)
    ap.add_argument("--worker", nargs=6, metavar=("INIT", "FORM", "SEED", "BAND", "W_IC", "VARIANT"))
    a = ap.parse_args()
    if a.worker:
        i, f, s, b, w, v = a.worker
        worker(i, f, int(s), float(b), float(w), "" if v == "-" else v)
    else:
        launcher(a.par)