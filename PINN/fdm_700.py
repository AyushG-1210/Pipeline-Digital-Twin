"""
early_refs.py -- early-time FDM references for the 50 held-out samples (reacting problem, Da > 0).

Writes, next to your other data files:
    fdm_hold_te025.npz   C, phi at t = 0.25/Fo   (per sample)
    fdm_hold_te.npz      C, phi at t = 1/Fo
Same keys as the Da=0 versions (C, t_e, rho, c_bulk, y, x, n) plus phi.

No torch needed. Inputs are read from files you already have:
    fdm_hold_950_1000.npz  -> rho(x), c_bulk(x) per sample, and the t=1 C used for the sanity check
    fdm_hold_da0_te.npz    -> t_e = 1/Fo per sample, which gives the wall thickness L = sqrt(D * t_max * t_e)

USAGE
    python fdm_700.py                    # all 100 solves, parallel; paths resolve from PINN/ automatically
    python fdm_700.py --workers 8        # (--data / --da0 override the folders)
Before solving, it re-solves sample 0 to t = 1 and checks it against the stored reference,
so a wrong parameter or file stops it immediately.
"""
import os, time, argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from concurrent.futures import ProcessPoolExecutor

PHYSICS_PARAMS = {
    "D_global": 1e-9, "k_rate": 2.07e-7, "alpha": 0.5, "F": 96485.0, "R": 8.314, "T": 298.15,
    "E_eq": -0.44, "E_corr": -0.44, "t_max": 31536000.0, "n_electrons": 2.0, "C_bulk": 1.0, "L_x": 10.0,
}
NX, NY, NT = 21, 61, 200

# local layout: this file lives in PINN/rework/
PINN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PINN_DIR, "data")                            # fdm_hold_950_1000.npz in, fdm_hold_te*.npz out
DA0_DIR  = os.path.join(PINN_DIR, "electrochem", "results", "positive_control")     # fdm_hold_da0_te.npz

# =============================================================================
# FDM solver -- copied unchanged from main.ipynb (Section 3)
# =============================================================================
def build_index(Nx, Ny):
    return lambda i, j: i * Ny + j

def stretched_y_grid(Ny, k=6.0):
    xi = np.linspace(0.0, 1.0, Ny)
    y = np.tanh(k * xi) / np.tanh(k)
    y[0] = 0.0; y[-1] = 1.0
    return y

def y_second_deriv_coeffs(y):
    Ny = len(y); coeffs = np.zeros((Ny, 3))
    for j in range(1, Ny - 1):
        h1 = y[j] - y[j - 1]; h2 = y[j + 1] - y[j]
        coeffs[j, 0] = 2.0 / (h1 * (h1 + h2))
        coeffs[j, 1] = -2.0 / (h1 * h2)
        coeffs[j, 2] = 2.0 / (h2 * (h1 + h2))
    return coeffs

def y_boundary_deriv_coeffs(y):
    Ny = len(y)
    h1 = y[Ny - 1] - y[Ny - 2]; h2 = y[Ny - 2] - y[Ny - 3]; H = h1 + h2
    return (H + h1) / (h1 * H), -H / (h1 * h2), h1 / (H * h2)

def build_phi_matrix(Nx, Ny, y, dx, A2=1.0):
    N = Nx * Ny; idx = build_index(Nx, Ny)
    A = sp.lil_matrix((N, N))
    y2c = y_second_deriv_coeffs(y)
    b_j, b_jm1, b_jm2 = y_boundary_deriv_coeffs(y)
    for i in range(Nx):
        for j in range(Ny):
            k_ = idx(i, j)
            if j == 0:
                A[k_, k_] = 1.0
            elif j == Ny - 1:
                A[k_, k_] = b_j; A[k_, idx(i, j - 1)] = b_jm1; A[k_, idx(i, j - 2)] = b_jm2
            elif i == 0:
                A[k_, k_] = 3.0; A[k_, idx(1, j)] = -4.0; A[k_, idx(2, j)] = 1.0
            elif i == Nx - 1:
                A[k_, k_] = 3.0; A[k_, idx(Nx - 2, j)] = -4.0; A[k_, idx(Nx - 3, j)] = 1.0
            else:
                cL, cC, cR = y2c[j]
                A[k_, k_] = -2.0 * A2 / dx**2 + cC
                A[k_, idx(i - 1, j)] = A2 / dx**2; A[k_, idx(i + 1, j)] = A2 / dx**2
                A[k_, idx(i, j - 1)] = cL; A[k_, idx(i, j + 1)] = cR
    return A.tocsr()

def bv_reaction_rate_and_deriv(C, phi, Da, params):
    if Da == 0.0:
        return np.zeros_like(C), np.zeros_like(C)
    if "E_corr" not in params:
        raise KeyError("params must define E_corr explicitly")
    alpha, F, R, T = params["alpha"], params["F"], params["R"], params["T"]
    E_eq, E_corr = params["E_eq"], params["E_corr"]
    k_const = alpha * F / (R * T)
    eta = -phi - (E_eq - E_corr)
    raw_exponent = k_const * eta
    exponent = np.clip(raw_exponent, -10.0, 10.0)
    saturated = (raw_exponent < -10.0) | (raw_exponent > 10.0)
    exp_term = np.exp(exponent)
    rate = Da * C * exp_term
    d_rate_dC = np.where(~saturated, Da * exp_term, 0.0)
    return rate, d_rate_dC

def solve_fdm(Nx, Ny, Nt, dt, L_meters, params, rho_x, c_x=None, y_stretch_k=6.0, Da_override=None):
    dx = 1.0 / (Nx - 1)
    y = stretched_y_grid(Ny, k=y_stretch_k)
    idx = build_index(Nx, Ny); N = Nx * Ny
    Fo = params["D_global"] * params["t_max"] / (L_meters ** 2)
    Da = params["k_rate"] * L_meters / params["D_global"] if Da_override is None else Da_override
    A2 = (L_meters / params["L_x"]) ** 2
    y2c = y_second_deriv_coeffs(y)
    b_j, b_jm1, b_jm2 = y_boundary_deriv_coeffs(y)
    A_phi_csr = build_phi_matrix(Nx, Ny, y, dx, A2=A2)
    phi_flat = np.zeros(N)
    rho_x = np.asarray(rho_x, dtype=np.float64)
    sigma_x = 100.0 / rho_x
    gamma_x = (params["n_electrons"] * params["F"] * params["D_global"] * params.get("C_bulk", 1.0)) / sigma_x
    c_x = np.ones(Nx) if c_x is None else np.asarray(c_x, dtype=np.float64)
    C_flat_prev = np.zeros(N)

    for n in range(1, Nt):
        C_flat = C_flat_prev.copy()
        for it in range(30):
            b_phi = np.zeros(N)
            for i in range(Nx):
                k_wall = idx(i, Ny - 1)
                r, _ = bv_reaction_rate_and_deriv(np.array([C_flat[k_wall]]), np.array([phi_flat[k_wall]]), Da, params)
                b_phi[k_wall] = -gamma_x[i] * r[0]
            phi_flat = spsolve(A_phi_csr, b_phi)

            A = sp.lil_matrix((N, N)); R_vec = np.zeros(N)
            for i in range(Nx):
                for j in range(Ny):
                    k_ = idx(i, j)
                    if j == 0:
                        A[k_, k_] = 1.0; R_vec[k_] = C_flat[k_] - c_x[i]; continue
                    if j == Ny - 1:
                        r, dr = bv_reaction_rate_and_deriv(np.array([C_flat[k_]]), np.array([phi_flat[k_]]), Da, params)
                        r, dr = r[0], dr[0]
                        dCdy = b_j * C_flat[k_] + b_jm1 * C_flat[idx(i, j - 1)] + b_jm2 * C_flat[idx(i, j - 2)]
                        R_vec[k_] = dCdy + r
                        A[k_, k_] = b_j + dr; A[k_, idx(i, j - 1)] = b_jm1; A[k_, idx(i, j - 2)] = b_jm2
                        continue
                    if i == 0:
                        A[k_, k_] = 3.0; A[k_, idx(1, j)] = -4.0; A[k_, idx(2, j)] = 1.0
                        R_vec[k_] = 3 * C_flat[k_] - 4 * C_flat[idx(1, j)] + C_flat[idx(2, j)]; continue
                    if i == Nx - 1:
                        A[k_, k_] = 3.0; A[k_, idx(Nx - 2, j)] = -4.0; A[k_, idx(Nx - 3, j)] = 1.0
                        R_vec[k_] = 3 * C_flat[k_] - 4 * C_flat[idx(Nx - 2, j)] + C_flat[idx(Nx - 3, j)]; continue
                    cL, cC, cR = y2c[j]
                    coef = 1.0 / dt + 2 * Fo * A2 / dx**2 - Fo * cC
                    A[k_, k_] = coef
                    A[k_, idx(i - 1, j)] = -Fo * A2 / dx**2; A[k_, idx(i + 1, j)] = -Fo * A2 / dx**2
                    A[k_, idx(i, j - 1)] = -Fo * cL; A[k_, idx(i, j + 1)] = -Fo * cR
                    R_vec[k_] = (coef * C_flat[k_] - Fo * A2 / dx**2 * C_flat[idx(i - 1, j)]
                                 - Fo * A2 / dx**2 * C_flat[idx(i + 1, j)] - Fo * cL * C_flat[idx(i, j - 1)]
                                 - Fo * cR * C_flat[idx(i, j + 1)] - C_flat_prev[k_] / dt)
            delta = spsolve(A.tocsr(), -R_vec)
            C_flat = C_flat + delta
            if np.max(np.abs(delta)) < 1e-10:
                break
        C_flat_prev = C_flat.copy()
    return C_flat.reshape(Nx, Ny), phi_flat.reshape(Nx, Ny), y

# =============================================================================
# driver
# =============================================================================
def solve_one(job):
    """job = (sample index, end time t_end, L, rho_x, c_x). Nt=200 steps over [0, t_end]."""
    i, t_end, L, rho_x, c_x = job
    C, phi, _ = solve_fdm(NX, NY, NT, t_end / (NT - 1), L, PHYSICS_PARAMS, rho_x=rho_x, c_x=c_x)
    return i, t_end, C.astype(np.float32), phi.astype(np.float32)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA_DIR, help="folder with fdm_hold_950_1000.npz (and where outputs are written)")
    p.add_argument("--da0", default=DA0_DIR, help="folder with fdm_hold_da0_te.npz")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    hold = np.load(os.path.join(a.data, "fdm_hold_950_1000.npz"))
    te1 = np.load(os.path.join(a.da0, "fdm_hold_da0_te.npz"))["t_e"].astype(np.float64)    # 1/Fo
    rho, cb = hold["rho"].astype(np.float64), hold["c_bulk"].astype(np.float64)
    n = rho.shape[0]
    L = np.sqrt(PHYSICS_PARAMS["D_global"] * PHYSICS_PARAMS["t_max"] * te1)                 # Fo = D t_max / L^2
    print(f"{n} samples | WT {L.min() / 0.0254:.3f}..{L.max() / 0.0254:.3f} in | 1/Fo {te1.min():.4f}..{te1.max():.4f}")

    # sanity check: sample 0 solved to t = 1 must reproduce the stored reference
    t0 = time.time()
    _, _, C0, _ = solve_one((0, 1.0, L[0], rho[0], cb[0]))
    rel = np.linalg.norm(C0 - hold["C"][0]) / np.linalg.norm(hold["C"][0])
    print(f"check: sample 0 at t=1 vs stored reference, rel diff {rel:.1e} ({time.time() - t0:.0f}s)")
    assert rel < 1e-4, "does not reproduce fdm_hold_950_1000.npz -- wrong data folder or parameters; stopping"

    targets = {"fdm_hold_te025.npz": 0.25 * te1, "fdm_hold_te.npz": te1}
    jobs = [(i, float(t_e[i]), L[i], rho[i], cb[i]) for t_e in targets.values() for i in range(n)]
    out = {k: (np.zeros((n, NX, NY), np.float32), np.zeros((n, NX, NY), np.float32)) for k in targets}
    lookup = {(i, float(t_e[i])): k for k, t_e in targets.items() for i in range(n)}

    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, t_end, C, phi in ex.map(solve_one, jobs):
            k = lookup[(i, t_end)]
            out[k][0][i], out[k][1][i] = C, phi
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)} solves ({time.time() - t0:.0f}s)", flush=True)

    for k, t_e in targets.items():
        path = os.path.join(a.data, k)
        np.savez_compressed(path, C=out[k][0], phi=out[k][1], t_e=t_e, rho=hold["rho"], c_bulk=hold["c_bulk"],
                            y=hold["y"], x=hold["x"], n=n)
        cw = out[k][0][:, :, -1]
        print(f"saved {path} | C_wall {cw.min():.4f}..{cw.max():.4f}")