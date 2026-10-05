"""
val_refs.py -- FDM references for a VALIDATION slice (dataset rows 700-749).

Rows 700-949 are never used for training (0-699) or testing (950-999), so choosing the IC weight
(or anything else) on these rows keeps the 50 held-out test samples untouched.

Writes  <data>/fdm_val_700_750.npz  with C and phi at three times per sample:
    C_te025, phi_te025   at t = 0.25/Fo
    C_te,    phi_te      at t = 1/Fo
    C,       phi         at t = 1       (same Nt=200 setup as the training/test references)
plus t_e (= 1/Fo), wt (inches), rho, c_bulk, x, y, rows.

USAGE (local PC):
    python val_refs.py --data path/to/PINN/data --workers 8
NEEDS: numpy, scipy, torch (CPU is fine -- only used to regenerate the dataset with the same RNG),
       early_refs.py in the SAME folder as this script (its verified solver is reused),
       fdm_hold_950_1000.npz in --data (used for the safety checks; fdm_hold_da0_te.npz optional).

Before solving, it regenerates the dataset and checks (1) rows 950-999 reproduce the stored rho and
c_bulk, (2) wall thickness matches the stored t_e if fdm_hold_da0_te.npz is present, and (3) re-solving
held-out sample 0 to t=1 reproduces the stored C. Any mismatch stops it.
"""
import os, time, argparse
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from fdm_700 import solve_one, PHYSICS_PARAMS, NX, NY

ROWS = np.arange(700, 750)

# ---- dataset generator: copied from main.ipynb section 2 (same seed, same draw order) ----
DATA_SEED = 1234
N_ENVS, SOIL_POINTS, FLUID_POINTS = 1000, 8, 50
SOIL_LS, FLUID_LS = 0.25, 0.3
WT_RANGE, OD_RANGE = (0.25, 1.0), (4.0, 48.0)
LOG_RHO_MIN, LOG_RHO_MAX = np.log10(500.0), np.log10(20000.0)
C_BULK_MIN, C_BULK_MAX = 0.3, 1.0

def grf_profiles(n, points, length_scale):
    x = np.linspace(0, 1, points)
    X, Y = np.meshgrid(x, x)
    cov = np.exp(-0.5 * ((X - Y) / length_scale) ** 2) + 1e-6 * np.eye(points)
    L = np.linalg.cholesky(cov)
    z = np.random.randn(n, points)
    return torch.tensor(1 / (1 + np.exp(-(z @ L.T))), dtype=torch.float32)

def make_dataset():
    torch.manual_seed(DATA_SEED); np.random.seed(DATA_SEED)
    soil = grf_profiles(N_ENVS, SOIL_POINTS, SOIL_LS)
    fluid = grf_profiles(N_ENVS, FLUID_POINTS, FLUID_LS)
    wt = torch.FloatTensor(N_ENVS).uniform_(*WT_RANGE)
    od = torch.FloatTensor(N_ENVS).uniform_(*OD_RANGE)            # drawn only to keep the RNG order
    defect = torch.FloatTensor(N_ENVS).uniform_(0.0, 1.0)         # same
    return soil.numpy(), fluid.numpy(), wt.numpy()

def inputs(rows, soil, fluid):
    """rho(x), c_bulk(x) on the Nx FDM nodes, exactly as generate_fdm_dataset built them."""
    xn = np.linspace(0, 1, NX)
    xs, xf = np.linspace(0, 1, soil.shape[1]), np.linspace(0, 1, fluid.shape[1])
    rho = np.stack([10 ** (LOG_RHO_MIN + np.interp(xn, xs, soil[r]) * (LOG_RHO_MAX - LOG_RHO_MIN)) for r in rows])
    cb = np.stack([C_BULK_MIN + np.interp(xn, xf, fluid[r]) * (C_BULK_MAX - C_BULK_MIN) for r in rows])
    return rho, cb

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    soil, fluid, wt = make_dataset()
    D, tmax = PHYSICS_PARAMS["D_global"], PHYSICS_PARAMS["t_max"]

    # ---- safety checks against the stored held-out references ----
    hold = np.load(os.path.join(a.data, "fdm_hold_950_1000.npz"))
    rho_h, cb_h = inputs(range(950, 1000), soil, fluid)
    ok = np.allclose(rho_h, hold["rho"], rtol=1e-4) and np.allclose(cb_h, hold["c_bulk"], rtol=1e-4, atol=1e-6)
    print(f"check 1: regenerated rows 950-999 match stored rho/c_bulk: {ok}")
    assert ok, "dataset regeneration does not match -- stopping"
    te_file = os.path.join(a.data, "fdm_hold_da0_te.npz")
    if os.path.exists(te_file):
        wt_ref = np.sqrt(np.load(te_file)["t_e"] * D * tmax) / 0.0254
        rel = np.abs(wt[950:1000] / wt_ref - 1).max()
        print(f"check 2: wall thickness vs stored t_e, max rel diff {rel:.1e}")
        assert rel < 1e-4, "wall thickness mismatch -- stopping"
    else:
        print("check 2: skipped (fdm_hold_da0_te.npz not found)")
    t0 = time.time()
    _, _, C0, _ = solve_one((0, 1.0, float(wt[950]) * 0.0254, rho_h[0], cb_h[0]))
    rel = np.linalg.norm(C0 - hold["C"][0]) / np.linalg.norm(hold["C"][0])
    print(f"check 3: held-out sample 0 re-solved to t=1, rel diff {rel:.1e} ({time.time() - t0:.0f}s)")
    assert rel < 1e-4, "solver does not reproduce the stored reference -- stopping"

    # ---- validation solves: 50 samples x 3 times ----
    rho_v, cb_v = inputs(ROWS, soil, fluid)
    L = wt[ROWS].astype(np.float64) * 0.0254
    te = L ** 2 / (D * tmax)                                   # = 1/Fo
    times = {"te025": 0.25 * te, "te": te, "t1": np.ones_like(te)}
    jobs = [(i, float(tt[i]), L[i], rho_v[i], cb_v[i]) for tt in times.values() for i in range(len(ROWS))]
    lookup = {(i, float(tt[i])): k for k, tt in times.items() for i in range(len(ROWS))}
    out = {k: (np.zeros((len(ROWS), NX, NY), np.float32), np.zeros((len(ROWS), NX, NY), np.float32)) for k in times}

    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, t_end, C, phi in ex.map(solve_one, jobs):
            k = lookup[(i, t_end)]
            out[k][0][i], out[k][1][i] = C, phi
            done += 1
            if done % 15 == 0:
                print(f"  {done}/{len(jobs)} solves ({time.time() - t0:.0f}s)", flush=True)

    path = os.path.join(a.data, "fdm_val_700_750.npz")
    np.savez_compressed(path,
                        C_te025=out["te025"][0], phi_te025=out["te025"][1],
                        C_te=out["te"][0], phi_te=out["te"][1],
                        C=out["t1"][0], phi=out["t1"][1],
                        t_e=te, wt=wt[ROWS], rho=rho_v, c_bulk=cb_v,
                        y=hold["y"], x=hold["x"], rows=ROWS)
    print(f"saved {path}  | C_wall at t=1: {out['t1'][0][:, :, -1].min():.4f}..{out['t1'][0][:, :, -1].max():.4f}")