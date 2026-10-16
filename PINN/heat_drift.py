"""
heat_drift.py -- does an incompatible initial condition pull weak-form training off the solution?

Second-problem test for the paper. Transient heat equation u_t = a * laplacian(u) on [0,1]^dim,
dim = 1 or 2, with the SAME structure as the electrochemical model:

    hard Dirichlet boundary  u = 1 on the boundary, built into the network:  u = 1 + B(x) * NN(x, t)
    soft initial condition   IC loss penalizes u(x, 0) - u0(x)

Two initial conditions:
    compat : u0 = 1 - prod sin(pi x_i)   -> equals 1 on the boundary, no clash (positive control)
    clash  : u0 = 0                      -> clashes with the boundary value 1 (the electrochem setup)
Both have exact solutions (separable Fourier series), so no FDM is needed.

Weak loss  = your FastVPINN loss (Legendre-difference test functions, per-time residual), BC term dropped.
Strong loss = your strong-form PINN loss (pointwise u_t - a*lap u), BC term dropped.
Weights are your solvers' (physics 12, IC 3). Schedule is cosine (as in the main notebook). No gradient clipping.

Arms, per dim x case x seed:
    start               supervised fit to the exact solution (the warm-start probe)
    warm/cold x weak/strong x IC weight on/off

USAGE
    python heat_drift.py --all --jobs 8                # everything, in parallel processes
    python heat_drift.py --summary                     # table from finished runs
    python heat_drift.py --band-sweep --jobs 8         # band-masked IC fix (2D by default; --dims 1 2 for both)
    python heat_drift.py --dim 2 --case clash --init warm --form weak --ic 1 --seed 42   # one run
    add --smoke for a tiny check run, --out DIR to change the output folder (default ./heat_runs/heat_results)

SCALE-MATCHED CONTROL (new)
    python heat_drift.py --measure --dims 1 2 --out heat_runs/heat_matched --starts DIR
        gradient norms of the IC term and the physics term, weak vs strong, clashing IC, at the point each
        run starts from (warm: the supervised start; cold: the seeded random init). Writes
        <out>/matched_measure.json and prints the strong-form IC multiplier k that makes the IC term as loud,
        relative to the physics gradient, as in the weak form at the default weight.
        Add --case compat (and --starts pointing at the compat starts) to measure the matching-IC problem; that
        writes <out>/matched_measure_compat.json instead.
    python heat_drift.py --matched --dims 1 2 --jobs 4 --out heat_runs/heat_matched --starts DIR
        trains the strong form on the clashing IC at that multiplier: dims x {warm, cold} x seeds 42-44.
    DIR is the folder holding the EXISTING supervised starts (d{1,2}_clash_start_s{42,43,44}.pt), so warm runs
    start from the same fits as every other heat run. Both modes stop if a start is missing.

IC-WEIGHT SWEEP (dose-response for the matched control)
    python heat_drift.py --sweep --dims 1 --jobs 3 --out heat_runs/heat_matched --starts DIR
        strong form, clashing IC, warm start, seeds 42-44, at each multiplier in --mults
        (default 10 30 100 300 3000 10000). Finished runs are skipped, so it can be restarted.
        Results land next to the x970 matched runs, and --summary shows one ICx<mult> arm per multiplier.
    python heat_drift.py --sweep --sweep-form weak --dims 1 --jobs 24 --out heat_runs/heat_matched --starts DIR \
        --mults 0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001
        same, but the WEAK form with its IC weight turned DOWN (the other direction of the dial).
        A weak IC multiplier of about 1/k gives the weak form the strong form's default effective weight.

ONE POOL FOR A SESSION
    python heat_drift.py --batch --jobs 24 --retries 2 --out heat_runs/heat_matched --starts DIR
        2D weak-sweep gaps, 2D and 1D annealing, then the 2D strong sweep (drop it with --no-bonus), as one queue.
        Jobs without a result are retried twice, at 12 and then 6 at once.

NORMALIZATION, POLYNOMIAL ORDER, ADVECTION-DIFFUSION (the final checks)
    --normalize      divide the weak residual by the element measure (the loss is multiplied by the number of elements
                     squared); names end in _norm, and --measure reports the normalized k
    --p-deg P        test-function order of the weak form (default 6 in 1D, 4 in 2D; the quadrature is raised if needed);
                     names end in _pP. With --n-el this separates element size from test-space richness.
    --adv C          advection-diffusion u_t + C u_x = a u_xx (1D only) instead of the heat equation, with the same hard
                     boundary u = 1 and the same clashing / matching initial conditions. Exact solution: the
                     Cole-Hopf-style change of variables u = 1 - exp(C x/(2a) - C^2 t/(4a)) w, w_t = a w_xx.
                     Names start d1adv<C>_ so they never collide with heat runs. Use a separate --out folder.
    python heat_drift.py --adv-batch --adv 1 --jobs 24 --retries 2 --out heat_runs/heat_adv
        the whole advection-diffusion replication: supervised starts, defaults (clash and matching IC), k, sweeps
        around k and 1/k, and the matching-IC control at k times the default strong weight.
    run_final.py runs normalization, polynomial order and advection-diffusion as one session.

RAMPED BOUNDARY (removes the clash by ramping the boundary up instead of switching it on)
    --ramp TAU       1D heat, u0 = 0, boundary g(t) = 1 - exp(-t/TAU) hard-coded as u = g(t) + B(x) * NN(x, t).
                     g(0) = 0 = u0, so the IC can be met exactly. Exact solution: w = u - g solves
                     w_t = a w_xx - g'(t) with w = 0 at both ends and w(x, 0) = 0, a sine series in closed form
                     (checked against Crank-Nicolson to ~3e-6 at t = 0.01). Only the clashing IC (u0 = 0) is allowed,
                     since the matching IC would clash with g(0) = 0. Names start d1ramp<TAU>_. Use one --out folder
                     per TAU, because --summary does not separate them.
    python heat_drift.py --ramp-batch --ramp 0.01 --jobs 12 --retries 2 --out heat_runs/heat_ramp001
    python heat_drift.py --ramp-batch --ramp 0.05 --jobs 12 --retries 2 --out heat_runs/heat_ramp005
        supervised starts, weak and strong at default weights, k measured at the starts, then the strong form at
        k times its default. Warm starts, seeds 42-44.
        Rule fixed before running: every arm stays within 0.02 of its supervised start at every time.
    Every new run also stores err_abs, the RMS absolute error on the evaluation grid, next to the relative error.

MESH CHECK (does the measured k predict where the weak form fails on another mesh?)
    add --n-el N to --measure / --sweep: the weak form uses N elements per dimension, names end in _eN, the strong
    form is unchanged (its point budget stays at the default mesh). See run_mesh.sh.

DYNAMIC LOSS BALANCING (Wang, Teng & Perdikaris 2021, learning-rate annealing)
    python heat_drift.py --anneal-sweep --dims 1 2 --jobs 12 --out heat_runs/heat_matched --starts DIR
        both forms x {warm, cold} x seeds 42-44, clashing IC. The IC weight starts at the default (3) and every
        --anneal-every steps is moved toward  max|grad(physics term)| / mean|grad(IC loss)|  with factor
        --anneal-alpha (paper default 0.9). Run names end in _anneal; the JSON stores the weight history.
"""
import os, sys, json, math, time, argparse, subprocess, glob
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn
from scipy.special import eval_legendre, legendre as legendre_poly

torch.set_default_dtype(torch.float64)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------- settings
A_DIFF = 0.1            # diffusivity. With a = 1 (old setup) the solution decays ~1e-9 by t = 1,
                        # so late-time errors are meaningless; a = 0.1 keeps a transient across [0, 1]
W_PHYS, W_IC = 12.0, 3.0
T_EVAL = (0.01, 0.05, 0.1, 0.3, 1.0)
N_SERIES = 400          # odd Fourier modes; accurate for t >= 1e-3
T_MIN_FIT = 1e-3        # the supervised start fits t in [1e-3, 1] (the clashing solution jumps at t = 0)
SEEDS = (42, 43, 44)
CFG = {  # weak-form discretization per dim (2D = your FastVPINN settings)
    1: dict(n_el=16, n_q=8, p_deg=6, n_t=10, n_ic=500),
    2: dict(n_el=8,  n_q=5, p_deg=4, n_t=10, n_ic=500),
}
DEFAULT_NEL = {d: c["n_el"] for d, c in CFG.items()}   # the paper's weak-form meshes (16 in 1D, 8x8 in 2D)
DEFAULT_NQ = {d: c["n_q"] for d, c in CFG.items()}
DEFAULT_P = {d: c["p_deg"] for d, c in CFG.items()}
MESH_TAG = ""           # run tag from --n-el / --p-deg / --normalize: names end in _e<n>, _p<P>, _norm (in that order)
NORMALIZE = False       # --normalize: weak residual divided by the element measure
ADV_C = 0.0             # --adv C: advection speed (0 = heat equation)
RAMP_TAU = 0.0          # --ramp TAU: boundary g(t) = 1 - exp(-t/TAU) instead of a sudden 1 (0 = sudden)
PDE_TAG = ""            # "adv<C>" or "ramp<TAU>"; goes into every run name
ITERS = dict(warm=10000, cold=15000, pre=5000)
LR = dict(warm=1e-4, cold=1e-3, pre=1e-3)

# ----------------------------------------------------------------------------- exact solutions
ODD = torch.arange(1, 2 * N_SERIES, 2, dtype=torch.float64, device=DEV)

def f_series(x, t):
    """1D: solution of v_t = a v_xx, v = 0 at x = 0,1, v(x,0) = 1.  x, t: [P] -> [P]."""
    s = torch.sin(math.pi * x[:, None] * ODD) * torch.exp(-A_DIFF * math.pi ** 2 * ODD ** 2 * t[:, None])
    return (4.0 / (math.pi * ODD) * s).sum(1)

ALL_N = torch.arange(1, 2 * N_SERIES, dtype=torch.float64, device=DEV)    # modes 1..799 (advection breaks odd symmetry)

def _cosint(m, beta):
    """int_0^1 exp(-beta x) cos(m pi x) dx for integer m >= 0 (tensor)."""
    mp = math.pi * m
    sg = torch.where(torch.remainder(m, 2) == 0, 1.0, -1.0)
    return beta * (1.0 - sg * math.exp(-beta)) / (beta ** 2 + mp ** 2)

def _adv_coeffs(case, beta):
    """Sine coefficients of exp(-beta x) v0(x), v0 = 1 (clash) or sin(pi x) (matching)."""
    n = ALL_N
    if case == "clash":
        sg = torch.where(torch.remainder(n, 2) == 0, 1.0, -1.0)
        k = math.pi * n
        return 2.0 * k * (1.0 - sg * math.exp(-beta)) / (beta ** 2 + k ** 2)
    return _cosint(n - 1.0, beta) - _cosint(n + 1.0, beta)

def exact_adv(X, case):
    """u_t + c u_x = a u_xx on [0,1], u = 1 at x = 0, 1.  u = 1 - v, v_t + c v_x = a v_xx, v = 0 on the boundary.
    With v = exp(beta x - c^2 t/(4a)) w, beta = c/(2a): w_t = a w_xx, w(x,0) = exp(-beta x) v0(x), a sine series."""
    c, a = ADV_C, A_DIFF
    beta = c / (2.0 * a)
    x, t = X[:, 0], X[:, -1]
    b = _adv_coeffs(case, beta)
    w = (b * torch.exp(-a * (math.pi * ALL_N) ** 2 * t[:, None]) * torch.sin(math.pi * x[:, None] * ALL_N)).sum(1)
    return 1.0 - torch.exp(beta * x - c * c * t / (4.0 * a)) * w

def ramp_g(t):
    """Ramped boundary value g(t) = 1 - exp(-t/TAU)."""
    return 1.0 - torch.exp(-t / RAMP_TAU)

def exact_ramp(X):
    """u_t = a u_xx on [0,1], u = g(t) at x = 0, 1, u(x,0) = 0. w = u - g: w_t = a w_xx - g'(t), w = 0 at the ends,
    w(x,0) = 0. With -g'(t) = -r exp(-r t), r = 1/TAU, and 1 = sum c_n sin(n pi x) over odd n, c_n = 4/(n pi):
    b_n(t) = -c_n r (exp(-r t) - exp(-lam_n t)) / (lam_n - r), lam_n = a (n pi)^2 (limit t exp(-lam t) at lam = r)."""
    x, t = X[:, 0], X[:, -1]
    r = 1.0 / RAMP_TAU
    lam = A_DIFF * (math.pi * ODD) ** 2
    c = 4.0 / (math.pi * ODD)
    tt = t[:, None]
    d = lam - r
    near = d.abs() < 1e-9
    frac = torch.where(near, tt * torch.exp(-lam * tt),
                       (torch.exp(-r * tt) - torch.exp(-lam * tt)) / torch.where(near, torch.ones_like(d), d))
    b = -c * r * frac
    return ramp_g(t) + (b * torch.sin(math.pi * x[:, None] * ODD)).sum(1)

def exact(X, case, dim):
    """X: [P, dim+1] with t in the last column -> u: [P]."""
    if ADV_C:
        assert dim == 1, "--adv is 1D only"
        return exact_adv(X, case)
    if RAMP_TAU:
        assert dim == 1 and case == "clash", "--ramp is 1D with u0 = 0 only"
        return exact_ramp(X)
    t = X[:, -1]
    if case == "compat":
        S = torch.ones_like(t)
        for i in range(dim):
            S = S * torch.sin(math.pi * X[:, i])
        return 1.0 - S * torch.exp(-dim * A_DIFF * math.pi ** 2 * t)
    P = torch.ones_like(t)
    for i in range(dim):
        P = P * f_series(X[:, i], t)           # separable: 2D = product of 1D solutions
    return 1.0 - P

def u0(xs, case):
    if case == "compat":
        return 1.0 - torch.prod(torch.sin(math.pi * xs), dim=1)
    return torch.zeros(xs.shape[0], device=xs.device)

# ----------------------------------------------------------------------------- model
class FCN(nn.Module):                          # your FCN, unchanged
    def __init__(self, input_dim=3, output_dim=1, hidden=64, layers=4):
        super().__init__()
        self.pinn = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            *[nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU()) for _ in range(layers)],
            nn.Linear(hidden, output_dim))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.pinn(x)

class HardBC(nn.Module):
    """u = g(t) + B(x) * NN(x, t). B = prod 4 x_i (1 - x_i) is 0 on the boundary, so u = g there for all t.
    g = 1 (sudden boundary, the default) or 1 - exp(-t/TAU) with --ramp."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = FCN(input_dim=dim + 1)

    def forward(self, X):
        B = torch.ones(X.shape[0], 1, device=X.device)
        for i in range(self.dim):
            B = B * 4.0 * X[:, i:i + 1] * (1.0 - X[:, i:i + 1])
        base = ramp_g(X[:, -1:]) if RAMP_TAU else 1.0
        return base + B * self.net(X)

# ----------------------------------------------------------------------------- weak form (your FastVPINN, any dim)
def build_weak(dim, n_el, n_q, p_deg, **_):
    r, w = np.polynomial.legendre.leggauss(n_q)
    modes = np.arange(1, p_deg + 2)
    v1 = np.array([eval_legendre(m + 1, r) - eval_legendre(m - 1, r) for m in modes])
    dv1 = np.array([legendre_poly(m + 1).deriv()(r) - legendre_poly(m - 1).deriv()(r) for m in modes])
    h = 1.0 / n_el
    centers = (np.arange(n_el) + 0.5) * h
    if dim == 1:
        V, dVs, Wq = v1, [dv1 * (2 / h)], w * (h / 2)
        pts = (centers[:, None] + 0.5 * h * r[None, :])[..., None]                 # [E, Q, 1]
    else:
        M, Q = len(modes), n_q
        V = np.einsum("aq,br->abqr", v1, v1).reshape(M * M, Q * Q)                   # k = m*M + n, q = i*Q + j
        dVx = np.einsum("aq,br->abqr", dv1, v1).reshape(M * M, Q * Q) * (2 / h)
        dVy = np.einsum("aq,br->abqr", v1, dv1).reshape(M * M, Q * Q) * (2 / h)
        dVs, Wq = [dVx, dVy], np.outer(w, w).reshape(-1) * (h / 2) ** 2
        xi, eta = np.meshgrid(r, r, indexing="ij")
        pts = np.array([[cx + 0.5 * h * xi.reshape(-1), cy + 0.5 * h * eta.reshape(-1)]
                        for cx in centers for cy in centers]).transpose(0, 2, 1)       # [E, Q, 2]
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float64, device=DEV)
    return T(pts), T(V), [T(d) for d in dVs], T(Wq)

def weak_residual(model, W, n_t, gen=None):
    pts, V, dVs, Wq = W
    E, Q, dim = pts.shape
    t = torch.rand(n_t, generator=gen, device=DEV)
    X = torch.cat([pts.unsqueeze(0).expand(n_t, E, Q, dim),
                   t.view(n_t, 1, 1, 1).expand(n_t, E, Q, 1)], -1).reshape(-1, dim + 1)
    X.requires_grad_(True)
    u = model(X)
    g = torch.autograd.grad(u.sum(), X, create_graph=True)[0].reshape(n_t, E, Q, dim + 1)
    R = torch.einsum("teq,kq,q->tek", g[..., -1], V, Wq)                             # mass
    for i in range(dim):
        R = R + A_DIFF * torch.einsum("teq,kq,q->tek", g[..., i], dVs[i], Wq)        # stiffness
    if ADV_C and dim == 1:
        R = R + ADV_C * torch.einsum("teq,kq,q->tek", g[..., 0], V, Wq)              # advection
    out = (R ** 2).mean()
    return out * float(E) ** 2 if NORMALIZE else out        # R / h^n squared: E = number of elements = h^-n

# ----------------------------------------------------------------------------- strong form (your PINN, any dim)
def strong_residual(model, dim, n_f, gen=None):
    X = torch.rand(n_f, dim + 1, generator=gen, device=DEV).requires_grad_(True)
    u = model(X)
    g = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
    lap = 0.0
    for i in range(dim):
        lap = lap + torch.autograd.grad(g[:, i].sum(), X, create_graph=True)[0][:, i]
    adv = ADV_C * g[:, 0] if (ADV_C and dim == 1) else 0.0
    return ((g[:, -1] + adv - A_DIFF * lap) ** 2).mean()

def ic_loss(model, dim, case, n_ic, gen=None, band=0.0):
    """band > 0: IC points only in [band, 1-band]^dim, i.e. skip a strip of width `band` next to the
    boundary. With the clashing IC this makes the IC loss reachable (u = 1 + B*NN, B > 0 inside the strip)."""
    xs = band + (1.0 - 2.0 * band) * torch.rand(n_ic, dim, generator=gen, device=DEV)
    X = torch.cat([xs, torch.zeros(n_ic, 1, device=DEV)], 1)
    return ((model(X)[:, 0] - u0(xs, case)) ** 2).mean()

def n_strong(dim, W):
    """Same point budget as the weak form on the DEFAULT mesh, so --n-el never changes the strong form."""
    return CFG[dim]["n_t"] * DEFAULT_NEL[dim] ** dim * DEFAULT_NQ[dim] ** dim

# ----------------------------------------------------------------------------- training
def pretrain(dim, case, seed, steps, bs=4096):
    torch.manual_seed(seed)
    m = HardBC(dim).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR["pre"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    lo = math.log10(T_MIN_FIT)
    for k in range(steps):
        xs = torch.rand(bs, dim, device=DEV)
        t = torch.cat([T_MIN_FIT + (1 - T_MIN_FIT) * torch.rand(bs // 2, device=DEV),
                       10 ** (lo * torch.rand(bs - bs // 2, device=DEV))])         # half uniform, half log
        X = torch.cat([xs, t[:, None]], 1)
        with torch.no_grad():
            tgt = exact(X, case, dim)
        loss = ((m(X)[:, 0] - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if k % 1000 == 0 or k == steps - 1:
            print(f"  pretrain {k}: {loss.item():.2e}", flush=True)
    return m

def train(model, dim, case, form, w_ic, iters, lr, W, band=0.0, anneal=None):
    """anneal: None, or dict(every=N, alpha=a). Then w_ic is only the starting IC weight; every N steps it is moved
    toward lam_hat = max|grad(W_PHYS*phys)| / mean|grad(ic)| (Wang et al. 2021, Algorithm 1). Returns the history."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-6)
    nf, cfg = n_strong(dim, W), CFG[dim]
    params = [p for p in model.parameters() if p.requires_grad]
    hist = []
    for i in range(iters):
        phys = weak_residual(model, W, cfg["n_t"]) if form == "weak" else strong_residual(model, dim, nf)
        ic = ic_loss(model, dim, case, cfg["n_ic"], band=band) if w_ic > 0 else None
        if anneal and ic is not None and i % anneal["every"] == 0:
            gp = torch.autograd.grad(W_PHYS * phys, params, retain_graph=True, allow_unused=True)
            gi = torch.autograd.grad(ic, params, retain_graph=True, allow_unused=True)
            mx = max(g.abs().max().item() for g in gp if g is not None)
            mn = torch.cat([g.abs().reshape(-1) for g in gi if g is not None]).mean().item()
            lam_hat = mx / max(mn, 1e-30)
            w_ic = (1 - anneal["alpha"]) * w_ic + anneal["alpha"] * lam_hat
            if i % 500 == 0:
                hist.append((i, w_ic))
        loss = W_PHYS * phys
        if ic is not None:
            loss = loss + w_ic * ic
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if i % 2000 == 0 or i == iters - 1:
            print(f"  iter {i}: loss {loss.item():.3e} | phys {phys.item():.3e}"
                  + (f" | IC weight {w_ic:.3e}" if anneal else ""), flush=True)
    if anneal:
        hist.append((iters, w_ic))
    model.anneal_hist = hist
    return model

# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def report(model, dim, case):
    """Rel. error of the TRANSIENT: ||u_pred - u|| / ||u - 1||. u -> 1 everywhere, so a plain
    rel. L2 on u would look tiny for any model that just outputs 1.
    Also returns the RMS absolute error on the same grid (rel. error times the RMS of u - 1)."""
    g = torch.linspace(0, 1, 201 if dim == 1 else 41, device=DEV)
    xs = g[:, None] if dim == 1 else torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    out, out_abs = {}, {}
    for t in T_EVAL:
        X = torch.cat([xs, torch.full((xs.shape[0], 1), t, device=DEV)], 1)
        ue, up = exact(X, case, dim), model(X)[:, 0]
        out[f"t={t}"] = ((up - ue).norm() / (ue - 1).norm()).item()
        out_abs[f"t={t}"] = ((up - ue).norm() / math.sqrt(xs.shape[0])).item()
    return out, out_abs

def terms(model, dim, case, W, band=0.0):
    """Weighted loss terms on FIXED draws, so start and end are compared on the same points."""
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    ic = ic_loss(model, dim, case, 2000, gen).item()
    wk = weak_residual(model, W, CFG[dim]["n_t"], gen).item()
    st = strong_residual(model, dim, n_strong(dim, W), gen).item()
    icb = ic_loss(model, dim, case, 2000, gen, band=band).item() if band > 0 else ic
    return dict(ic=W_IC * ic, ic_band=W_IC * icb, weak=W_PHYS * wk, strong=W_PHYS * st)

# ----------------------------------------------------------------------------- one run
def names(dim, case, init, form, ic, seed, band=0.0, anneal=False):
    if init == "pre":
        return f"d{dim}{PDE_TAG}_{case}_start_s{seed}"
    if anneal:
        return f"d{dim}{PDE_TAG}_{case}_{init}_{form}_anneal" + (f"_b{band:g}" if band > 0 else "") + MESH_TAG + f"_s{seed}"
    # ic:g keeps the old names (ic1, ic0) and gives e.g. ic450 for a matched multiplier
    return f"d{dim}{PDE_TAG}_{case}_{init}_{form}_ic{ic:g}" + (f"_b{band:g}" if band > 0 else "") + MESH_TAG + f"_s{seed}"

def start_path(a, dim, case, seed):
    """Where the supervised start lives: --starts if given (must exist there), else <out>/ckpt."""
    fn = names(dim, case, "pre", "", 0, seed) + ".pt"
    if getattr(a, "starts", None):
        p = os.path.join(a.starts, fn)
        if not os.path.exists(p):
            sys.exit(f"missing supervised start {p} -- point --starts at the folder with the existing starts")
        return p
    return os.path.join(a.out, "ckpt", fn)

def run_one(a):
    ck, rs = os.path.join(a.out, "ckpt"), os.path.join(a.out, "results")
    os.makedirs(ck, exist_ok=True); os.makedirs(rs, exist_ok=True)
    if a.smoke:
        for k in ITERS: ITERS[k] = 30
    dim, case, seed = a.dim, a.case, a.seed
    W = build_weak(dim, **CFG[dim])
    start_ck = start_path(a, dim, case, seed)

    def get_start():
        m = HardBC(dim).to(DEV)
        if os.path.exists(start_ck):
            m.load_state_dict(torch.load(start_ck, map_location=DEV)); return m
        m = pretrain(dim, case, seed, ITERS["pre"])
        torch.save(m.state_dict(), start_ck)
        return m

    name = names(dim, case, a.init, a.form, a.ic, seed, a.band, a.anneal)
    ann = dict(every=a.anneal_every, alpha=a.anneal_alpha) if a.anneal else None
    t0 = time.time()
    if a.init == "pre":
        m = get_start()
    else:
        if a.init == "warm":
            m = get_start()
        else:
            torch.manual_seed(seed); m = HardBC(dim).to(DEV)
        torch.manual_seed(seed)
        m = train(m, dim, case, a.form, W_IC * a.ic, ITERS[a.init], LR[a.init], W, band=a.band, anneal=ann)
        torch.save(m.state_dict(), os.path.join(ck, name + ".pt"))
    hist = getattr(m, "anneal_hist", [])
    err, err_abs = report(m, dim, case)
    res = dict(name=name, dim=dim, case=case, init=a.init, form=a.form, ic=a.ic, seed=seed, band=a.band,
               anneal=bool(a.anneal), n_el=CFG[dim]["n_el"], p_deg=CFG[dim]["p_deg"], norm=NORMALIZE, adv=ADV_C,
               ramp=RAMP_TAU, ic_weight_hist=hist, ic_weight_final=(hist[-1][1] if hist else None),
               err=err, err_abs=err_abs, terms=terms(m, dim, case, W, a.band), minutes=(time.time() - t0) / 60)
    json.dump(res, open(os.path.join(rs, name + ".json"), "w"), indent=2)
    print(name, " ".join(f"{k}:{v:.3f}" for k, v in res["err"].items()), flush=True)

# ----------------------------------------------------------------------------- scale-matched control (new)
def _grad_vec(model, loss):
    params = list(model.parameters())
    gs = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)]).detach()

def measure(a):
    """Gradient norms of W_IC*IC and W_PHYS*physics for both forms, clashing IC, averaged over D fixed draws.
    The IC draws use the same seeds for both forms, so the IC gradient is identical between them.
    k = ||g_phys_strong|| / ||g_phys_weak||: a strong-form IC multiplier of k matches the weak form at 1."""
    os.makedirs(a.out, exist_ok=True)
    case, out = (a.case or "clash"), {}
    for dim in (int(d) for d in a.dims):
        W = build_weak(dim, **CFG[dim]); cfg = CFG[dim]; nf = n_strong(dim, W)
        for init in a.inits:
            rows = {}
            for seed in SEEDS:
                if init == "warm":
                    m = HardBC(dim).to(DEV); m.load_state_dict(torch.load(start_path(a, dim, case, seed), map_location=DEV))
                else:
                    torch.manual_seed(seed); m = HardBC(dim).to(DEV)          # same init as a cold run
                acc = {}
                for d in range(a.D):
                    gi = torch.Generator(device=DEV); gi.manual_seed(2000 + d)
                    gw = torch.Generator(device=DEV); gw.manual_seed(1000 + d)
                    gs = torch.Generator(device=DEV); gs.manual_seed(1000 + d)
                    parts = dict(ic=W_IC * ic_loss(m, dim, case, cfg["n_ic"], gi),
                                 weak=W_PHYS * weak_residual(m, W, cfg["n_t"], gw),
                                 strong=W_PHYS * strong_residual(m, dim, nf, gs))
                    for k, L in parts.items():
                        acc[k] = acc.get(k, 0) + _grad_vec(m, L) / a.D
                n = {k: v.norm().item() for k, v in acc.items()}
                rows[seed] = dict(ic=n["ic"], phys_weak=n["weak"], phys_strong=n["strong"],
                                  ratio_weak=n["ic"] / n["weak"], ratio_strong=n["ic"] / n["strong"],
                                  k=n["strong"] / n["weak"])
            ks = [r["k"] for r in rows.values()]
            kg = math.exp(sum(map(math.log, ks)) / len(ks))
            out[f"d{dim}_{init}"] = dict(per_seed=rows, k=kg, k_rounded=float(f"{kg:.2g}"))
            print(f"\n=== {dim}D {case}, {init} start ===")
            print(f"{'seed':>6}{'||g_IC||':>12}{'||g_phys|| weak':>17}{'||g_phys|| strong':>19}"
                  f"{'IC/phys weak':>14}{'IC/phys strong':>16}{'k':>10}")
            for s, r in rows.items():
                print(f"{s:>6}{r['ic']:>12.3e}{r['phys_weak']:>17.3e}{r['phys_strong']:>19.3e}"
                      f"{r['ratio_weak']:>14.3g}{r['ratio_strong']:>16.3g}{r['k']:>10.3g}")
            print(f"  k (geometric mean) = {kg:.3g}  -> matched strong-form IC multiplier {float(f'{kg:.2g}'):g}"
                  f"  (IC weight {W_IC * float(f'{kg:.2g}'):g})")
            if not NORMALIZE:
                hd = (1.0 / CFG[dim]["n_el"]) ** dim                           # element measure
                print(f"  k with the weak residual divided by the element measure ({hd:.3g}): {kg * hd ** 2:.3g}"
                      f"  (a constant factor, so this equals turning the weak IC weight down by {1 / hd ** 2:.3g})")
                out[f"d{dim}_{init}"]["k_elem_normalized"] = kg * hd ** 2
    # the clash file is what --matched reads; other cases get their own file so nothing is overwritten
    path = os.path.join(a.out, ("matched_measure" if case == "clash" else f"matched_measure_{case}") + MESH_TAG + ".json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nsaved {path}")

def launch_matched(a):
    """Strong form, clashing IC, at the matched multiplier for each dim and start: dims x inits x seeds."""
    path = os.path.join(a.out, "matched_measure.json")
    if not os.path.exists(path):
        sys.exit(f"run --measure first ({path} not found)")
    M = json.load(open(path))
    jobs = []
    for dim in (int(d) for d in a.dims):
        for init in a.inits:
            key = f"d{dim}_{init}"
            if key not in M:
                sys.exit(f"{key} missing from {path}: rerun --measure with the same --dims/--inits")
            for s in SEEDS:
                if init == "warm":
                    start_path(a, dim, "clash", s)                            # stops early if a start is missing
                jobs.append((dim, "clash", init, "strong", M[key]["k_rounded"], s, 0.0))
    print("matched strong-form multipliers: " + ", ".join(f"{k}: x{v['k_rounded']:g}" for k, v in M.items()))
    run_waves(a, (("matched strong-form runs", jobs),))

def launch_sweep(a):
    """Dose-response: one form (--sweep-form, default strong), clashing IC, warm start, at each multiplier in
    --mults: dims x mults x seeds. Uses the same supervised starts as every other heat run; nothing is measured,
    the multipliers are fixed in advance."""
    jobs = []
    for dim in (int(d) for d in a.dims):
        for s in SEEDS:
            start_path(a, dim, "clash", s)                                    # stops early if a start is missing
        jobs += [(dim, "clash", "warm", a.sweep_form, float(m), s, 0.0) for m in a.mults for s in SEEDS]
    print(f"{a.sweep_form}-form IC multipliers: " + ", ".join(f"x{m:g}" for m in a.mults))
    run_waves(a, ((f"{a.sweep_form}-form IC sweep", jobs),))

def launch_anneal(a):
    """Dynamic balancing: dims x forms x inits x seeds, IC weight annealed from the default. Clashing IC unless
    --case compat (the matching-IC control: does balancing behave when the IC term can reach zero?)."""
    jobs, case = [], (a.case or "clash")                                     # --case compat: the no-clash control
    for dim in (int(d) for d in a.dims):
        for s in SEEDS:
            if "warm" in a.inits:
                start_path(a, dim, case, s)                                   # stops early if a start is missing
        jobs += [(dim, case, i, f, 1.0, s, 0.0, True) for f in a.forms for i in a.inits for s in SEEDS]
    run_waves(a, (("annealed IC weight (Wang et al. 2021)", jobs),))

def launch_batch(a):
    """Everything for the A100 session as ONE pool, in priority order (finished runs are skipped):
    the 2D weak-sweep runs still missing, 2D then 1D annealing (both forms, warm and cold), then the 2D strong sweep
    unless --no-bonus. Long 2D runs are queued first so they are not left to the end."""
    for dim in (1, 2):
        for s in SEEDS:
            start_path(a, dim, "clash", s)                                    # stops early if a start is missing
    jobs = [(2, "clash", "warm", "weak", m, s, 0.0) for m in (0.1, 0.01, 0.001, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
            for s in SEEDS]
    jobs += [(2, "clash", i, f, 1.0, s, 0.0, True) for f in ("strong", "weak") for i in ("warm", "cold") for s in SEEDS]
    jobs += [(1, "clash", i, f, 1.0, s, 0.0, True) for f in ("strong", "weak") for i in ("warm", "cold") for s in SEEDS]
    if not a.no_bonus:
        jobs += [(2, "clash", "warm", "strong", m, s, 0.0) for m in (1e4, 3e4, 1.2e5, 3e5, 1e6) for s in SEEDS]
    run_waves(a, (("A100 batch", jobs),))

def launch_adv(a):
    """Advection-diffusion replication (1D): starts, defaults for both ICs, k, sweeps around k and 1/k, and the
    matching-IC control (strong form at k times its default weight). Run with a separate --out folder."""
    a.dims, a.inits, a.starts = ["1"], ["warm"], None          # starts are built in <out>/ckpt
    pre = [(1, c, "pre", "-", 1, s, 0.0) for c in ("clash", "compat") for s in SEEDS]
    defaults = [(1, c, "warm", f, 1.0, s, 0.0) for c in ("clash", "compat") for f in ("weak", "strong") for s in SEEDS]
    run_waves(a, (("supervised starts", pre), ("default weights", defaults)))
    for case in ("clash", "compat"):
        a.case = case
        measure(a)
    a.case = None
    mfile = os.path.join(a.out, "matched_measure" + MESH_TAG + ".json")
    k = json.load(open(mfile))["d1_warm"]["k_rounded"]
    r2 = lambda v: float(f"{v:.2g}")
    fac = (0.1, 0.3, 1.0, 3.0, 10.0)
    jobs = [(1, "clash", "warm", "strong", r2(k * f), s, 0.0) for f in fac for s in SEEDS]
    jobs += [(1, "clash", "warm", "weak", r2(f / k), s, 0.0) for f in fac for s in SEEDS]
    jobs += [(1, "compat", "warm", "strong", r2(k), s, 0.0) for s in SEEDS]               # harmless without a clash
    print(f"k = {k:g}: strong x" + ", x".join(f"{r2(k * f):g}" for f in fac)
          + "; weak x" + ", x".join(f"{r2(f / k):g}" for f in fac) + "; compat strong x" + f"{r2(k):g}", flush=True)
    run_waves(a, (("sweeps around k and 1/k", jobs),))

def launch_ramp(a):
    """Ramped boundary (1D heat, u0 = 0, one TAU per call): supervised starts, weak and strong at default weights,
    k at the starts, then the strong form at k times its default. Rule fixed before running: every arm stays within
    0.02 of its supervised start at every time. Run with a separate --out folder per TAU."""
    a.dims, a.inits, a.starts, a.case = ["1"], ["warm"], None, "clash"     # starts are built in <out>/ckpt
    pre = [(1, "clash", "pre", "-", 1, s, 0.0) for s in SEEDS]
    defaults = [(1, "clash", "warm", f, 1.0, s, 0.0) for f in ("weak", "strong") for s in SEEDS]
    run_waves(a, (("supervised starts", pre), ("default weights", defaults)))
    measure(a)
    k = json.load(open(os.path.join(a.out, "matched_measure" + MESH_TAG + ".json")))["d1_warm"]["k_rounded"]
    print(f"ramp tau = {RAMP_TAU:g}: k = {k:g}, strong form at x{k:g}", flush=True)
    run_waves(a, (("strong form at k", [(1, "clash", "warm", "strong", k, s, 0.0) for s in SEEDS]),))

# ----------------------------------------------------------------------------- parallel launcher
def launch_all(a):
    pre = [(d, c, "pre", "-", 1, s, 0.0) for d in (1, 2) for c in ("compat", "clash") for s in SEEDS]
    main = [(d, c, i, f, ic, s, 0.0) for d in (1, 2) for c in ("compat", "clash") for i in ("warm", "cold")
            for f in ("weak", "strong") for ic in (1, 0) for s in SEEDS]
    run_waves(a, (("supervised starts", pre), ("training runs", main)))

def launch_band(a):
    """Band-masked IC, IC term ON. Clash: warm/cold x weak/strong. Compat: warm weak (control)."""
    dims = tuple(int(d) for d in a.dims)
    pre = [(d, c, "pre", "-", 1, s, 0.0) for d in dims for c in ("compat", "clash") for s in SEEDS]
    main = ([(d, "clash", i, f, 1, s, b) for b in a.bands for d in dims for i in ("warm", "cold")
             for f in ("weak", "strong") for s in SEEDS]
            + [(d, "compat", "warm", "weak", 1, s, b) for b in a.bands for d in dims for s in SEEDS])
    run_waves(a, (("supervised starts", pre), ("band runs", main)))

def run_waves(a, waves):
    """Run each wave's jobs as parallel subprocesses, at most --jobs at a time. A job counts as done only when its
    result JSON exists. After a wave, jobs without a JSON are retried up to --retries times, each retry at half the
    previous concurrency (a run killed for lack of RAM usually finishes with fewer neighbours)."""
    rs, logs = os.path.join(a.out, "results"), os.path.join(a.out, "logs")
    os.makedirs(rs, exist_ok=True); os.makedirs(logs, exist_ok=True)
    done = lambda j: os.path.exists(os.path.join(rs, names(*j) + ".json"))
    for wave_name, jobs in waves:
        n_jobs = a.jobs
        for attempt in range(getattr(a, "retries", 0) + 1):
            queue = [j for j in jobs if not done(j)]
            if not queue:
                break
            label = wave_name if attempt == 0 else f"{wave_name}, retry {attempt}"
            print(f"{label}: {len(queue)} to run, {len(jobs) - len(queue)} done, up to {n_jobs} at once", flush=True)
            live, t0 = [], time.time()
            while queue or live:
                while queue and len(live) < n_jobs:
                    d, c, i, f, ic, s, b, *rest = queue.pop(0); an = bool(rest and rest[0])
                    tag = names(d, c, i, f, ic, s, b, an)
                    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--dim", str(d), "--case", c,
                           "--init", i, "--form", f, "--ic", str(ic), "--seed", str(s), "--band", str(b),
                           "--out", a.out]
                    if getattr(a, "starts", None): cmd += ["--starts", a.starts]
                    if getattr(a, "n_el", None): cmd += ["--n-el", str(a.n_el)]
                    if getattr(a, "p_deg", None): cmd += ["--p-deg", str(a.p_deg)]
                    if getattr(a, "normalize", False): cmd.append("--normalize")
                    if getattr(a, "adv", 0): cmd += ["--adv", str(a.adv)]
                    if getattr(a, "ramp", 0): cmd += ["--ramp", str(a.ramp)]
                    if an: cmd += ["--anneal", "--anneal-every", str(a.anneal_every), "--anneal-alpha", str(a.anneal_alpha)]
                    if a.smoke: cmd.append("--smoke")
                    fh = open(os.path.join(logs, tag + ".log"), "w")
                    live.append((tag, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh))
                time.sleep(5)
                for item in live[:]:
                    tag, p, fh = item
                    if p.poll() is not None:
                        fh.close(); live.remove(item)
                        ok = p.returncode == 0
                        print(f"  {'done  ' if ok else 'FAILED'} {tag} ({(time.time() - t0) / 60:.1f} min)"
                              + ("" if ok else f" -> logs/{tag}.log"), flush=True)
            n_jobs = max(1, n_jobs // 2)
        left = [names(*j) for j in jobs if not done(j)]
        if left:
            print(f"{wave_name}: {len(left)} still without results after retries: " + ", ".join(left), flush=True)
    summary(a)

# ----------------------------------------------------------------------------- summary
def _ic_label(ic):
    return "ICoff" if ic == 0 else ("ICon" if ic == 1 else f"ICx{ic:g}")     # matched runs get their own arm

def _ic_order(arm):
    """Sort ICx<mult> arms by multiplier so the sweep reads as a curve; other arms keep name order."""
    tail = arm.rsplit("ICx", 1)
    try:
        return (tail[0], float(tail[1]))
    except (IndexError, ValueError):
        return (arm, 0.0)

def summary(a):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(a.out, "results", "*.json"))]
    if not rows:
        print("no results yet"); return
    groups = {}
    for r in rows:
        b = r.get("band", 0.0)
        arm = "start" if r["init"] == "pre" else (f"{r['init']} {r['form']} "
                                                  + ("anneal" if r.get("anneal") else _ic_label(r["ic"]))
                                                  + (f" band{b:g}" if b > 0 else "")
                                                  + (f" mesh{r['n_el']}" if r.get("n_el", DEFAULT_NEL[r["dim"]])
                                                     != DEFAULT_NEL[r["dim"]] else "")
                                                  + (f" p{r['p_deg']}" if r.get("p_deg", DEFAULT_P[r["dim"]])
                                                     != DEFAULT_P[r["dim"]] else "")
                                                  + (" norm" if r.get("norm") else ""))
        groups.setdefault((r["dim"], r["case"], arm), []).append(r)
    order = ["start"] + [f"{i} {f} IC{x}" for i in ("warm", "cold") for f in ("weak", "strong") for x in ("on", "off")]
    order += sorted({k[2] for k in groups} - set(order), key=_ic_order)   # band, matched and sweep arms
    for d in (1, 2):
        for c in ("compat", "clash"):
            print(f"\n=== {d}D, {c} IC  (transient rel. error, mean over seeds [max]) ===")
            print(f"{'arm':<30}" + "".join(f"{f't={t}':>16}" for t in T_EVAL) + f"{'ic':>10}{'ic_band':>10}{'weak':>10}{'strong':>10}  n")
            for arm in order:
                g = groups.get((d, c, arm))
                if not g: continue
                errs = "".join(f"{np.mean([r['err'][f't={t}'] for r in g]):>8.3f} [{max(r['err'][f't={t}'] for r in g):.2f}]"
                               for t in T_EVAL)
                tm = {k: np.mean([r["terms"].get(k, r["terms"]["ic"]) for r in g]) for k in ("ic", "ic_band", "weak", "strong")}
                print(f"{arm:<30}{errs}{tm['ic']:>10.2e}{tm['ic_band']:>10.2e}{tm['weak']:>10.2e}{tm['strong']:>10.2e}  {len(g)}")

# ----------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true"); p.add_argument("--summary", action="store_true")
    p.add_argument("--jobs", type=int, default=6); p.add_argument("--out", default="heat_runs/heat_results")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dim", type=int, choices=(1, 2)); p.add_argument("--case", choices=("compat", "clash"))
    p.add_argument("--init", choices=("pre", "warm", "cold")); p.add_argument("--form", default="weak")
    p.add_argument("--ic", type=float, default=1.0, help="IC weight multiplier (0 = off, 1 = default W_IC)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--band", type=float, default=0.0, help="skip IC points within this distance of the boundary")
    p.add_argument("--band-sweep", action="store_true", help="run the band-masked IC arms (see launch_band)")
    p.add_argument("--bands", type=float, nargs="+", default=[0.05, 0.1])
    p.add_argument("--dims", nargs="+", default=["2"], help="dims for --band-sweep/--measure/--matched/--sweep, e.g. --dims 1 2")
    p.add_argument("--measure", action="store_true", help="gradient balance at the start, weak vs strong (clash)")
    p.add_argument("--matched", action="store_true", help="strong form at the measured matched IC multiplier")
    p.add_argument("--sweep", action="store_true", help="strong form, warm, clash, at each IC multiplier in --mults")
    p.add_argument("--mults", type=float, nargs="+", default=[10, 30, 100, 300, 3000, 10000],
                   help="IC multipliers for --sweep")
    p.add_argument("--anneal", action="store_true", help="anneal the IC weight (Wang et al. 2021) in a single run")
    p.add_argument("--anneal-sweep", action="store_true", help="annealed runs: dims x forms x inits x seeds")
    p.add_argument("--anneal-every", type=int, default=10, help="steps between IC-weight updates")
    p.add_argument("--anneal-alpha", type=float, default=0.9, help="update factor (Wang et al. 2021 use 0.9)")
    p.add_argument("--forms", nargs="+", default=["weak", "strong"], help="forms for --anneal-sweep")
    p.add_argument("--n-el", type=int, default=None, help="weak-form elements per dimension (default 16 in 1D, 8 in 2D)")
    p.add_argument("--normalize", action="store_true", help="divide the weak residual by the element measure")
    p.add_argument("--p-deg", type=int, default=None, help="weak-form test-function order (default 6 in 1D, 4 in 2D)")
    p.add_argument("--adv", type=float, default=0.0, help="advection speed: u_t + c u_x = a u_xx (1D); 0 = heat")
    p.add_argument("--adv-batch", action="store_true", help="advection-diffusion replication (implies --adv 1 if unset)")
    p.add_argument("--ramp", type=float, default=0.0, help="ramped boundary g(t) = 1 - exp(-t/TAU) (1D heat, u0 = 0)")
    p.add_argument("--ramp-batch", action="store_true", help="the ramped-boundary arms for one --ramp TAU")
    p.add_argument("--batch", action="store_true", help="the whole A100 session as one pool (see launch_batch)")
    p.add_argument("--no-bonus", action="store_true", help="--batch without the 2D strong sweep")
    p.add_argument("--retries", type=int, default=0, help="retry jobs without results, at half concurrency each time")
    p.add_argument("--sweep-form", choices=("strong", "weak"), default="strong",
                   help="which form --sweep trains (weak: pass multipliers below 1 in --mults)")
    p.add_argument("--inits", nargs="+", default=["warm", "cold"], help="starts for --measure/--matched")
    p.add_argument("--D", type=int, default=10, help="draws averaged per gradient in --measure")
    p.add_argument("--starts", default=None, help="folder with the existing supervised starts (*_start_s*.pt)")
    a = p.parse_args()
    if a.adv_batch and not a.adv:
        a.adv = 1.0
    if a.ramp_batch and not a.ramp:
        sys.exit("--ramp-batch needs --ramp TAU")
    if a.adv and a.ramp:
        sys.exit("--adv and --ramp cannot be combined")
    if a.adv:
        ADV_C, PDE_TAG = a.adv, f"adv{a.adv:g}"
        dims_used = [a.dim or 1] if not (a.measure or a.sweep) else a.dims     # --adv-batch sets its own dims
        if not a.adv_batch and any(str(d) != "1" for d in dims_used):
            sys.exit("--adv is 1D only: pass --dims 1")
    if a.ramp:
        RAMP_TAU, PDE_TAG = a.ramp, f"ramp{a.ramp:g}"
        if not a.ramp_batch:
            dims_used = [a.dim or 1] if not (a.measure or a.sweep) else a.dims
            if any(str(d) != "1" for d in dims_used):
                sys.exit("--ramp is 1D only: pass --dims 1")
            if a.case == "compat":
                sys.exit("--ramp works with the clashing IC only (u0 = 0 matches g(0) = 0)")
    tag = ""
    if a.n_el:
        for d in CFG:
            CFG[d]["n_el"] = a.n_el
        tag += f"_e{a.n_el}"
    if a.p_deg:
        for d in CFG:
            CFG[d]["p_deg"] = a.p_deg
            CFG[d]["n_q"] = max(CFG[d]["n_q"], a.p_deg + 2)          # keep the quadrature exact for the higher order
        tag += f"_p{a.p_deg}"
    if a.normalize:
        NORMALIZE = True
        tag += "_norm"
    MESH_TAG = tag
    if a.all: launch_all(a)
    elif a.band_sweep: launch_band(a)
    elif a.measure: measure(a)
    elif a.matched: launch_matched(a)
    elif a.sweep: launch_sweep(a)
    elif a.anneal_sweep: launch_anneal(a)
    elif a.batch: launch_batch(a)
    elif a.adv_batch: launch_adv(a)
    elif a.ramp_batch: launch_ramp(a)
    elif a.summary: summary(a)
    else: run_one(a)