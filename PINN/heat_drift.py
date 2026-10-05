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
Weights are your solvers' (physics 12, IC 3). Schedule is cosine (as in the main notebook).

Arms, per dim x case x seed:
    start               supervised fit to the exact solution (the warm-start probe)
    warm/cold x weak/strong x IC weight on/off

USAGE
    python heat_drift.py --all --jobs 8                # everything, in parallel processes
    python heat_drift.py --summary                     # table from finished runs
    python heat_drift.py --band-sweep --jobs 8         # band-masked IC fix (2D by default; --dims 1 2 for both)
    python heat_drift.py --dim 2 --case clash --init warm --form weak --ic 1 --seed 42   # one run
    add --smoke for a tiny check run, --out DIR to change the output folder (default ./heat_results)
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
ITERS = dict(warm=10000, cold=15000, pre=5000)
LR = dict(warm=1e-4, cold=1e-3, pre=1e-3)

# ----------------------------------------------------------------------------- exact solutions
ODD = torch.arange(1, 2 * N_SERIES, 2, dtype=torch.float64, device=DEV)

def f_series(x, t):
    """1D: solution of v_t = a v_xx, v = 0 at x = 0,1, v(x,0) = 1.  x, t: [P] -> [P]."""
    s = torch.sin(math.pi * x[:, None] * ODD) * torch.exp(-A_DIFF * math.pi ** 2 * ODD ** 2 * t[:, None])
    return (4.0 / (math.pi * ODD) * s).sum(1)

def exact(X, case, dim):
    """X: [P, dim+1] with t in the last column -> u: [P]."""
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
    """u = 1 + B(x) * NN(x, t). B = prod 4 x_i (1 - x_i) is 0 on the boundary, so u = 1 there for all t."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = FCN(input_dim=dim + 1)

    def forward(self, X):
        B = torch.ones(X.shape[0], 1, device=X.device)
        for i in range(self.dim):
            B = B * 4.0 * X[:, i:i + 1] * (1.0 - X[:, i:i + 1])
        return 1.0 + B * self.net(X)

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
    return (R ** 2).mean()

# ----------------------------------------------------------------------------- strong form (your PINN, any dim)
def strong_residual(model, dim, n_f, gen=None):
    X = torch.rand(n_f, dim + 1, generator=gen, device=DEV).requires_grad_(True)
    u = model(X)
    g = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
    lap = 0.0
    for i in range(dim):
        lap = lap + torch.autograd.grad(g[:, i].sum(), X, create_graph=True)[0][:, i]
    return ((g[:, -1] - A_DIFF * lap) ** 2).mean()

def ic_loss(model, dim, case, n_ic, gen=None, band=0.0):
    """band > 0: IC points only in [band, 1-band]^dim, i.e. skip a strip of width `band` next to the
    boundary. With the clashing IC this makes the IC loss reachable (u = 1 + B*NN, B > 0 inside the strip)."""
    xs = band + (1.0 - 2.0 * band) * torch.rand(n_ic, dim, generator=gen, device=DEV)
    X = torch.cat([xs, torch.zeros(n_ic, 1, device=DEV)], 1)
    return ((model(X)[:, 0] - u0(xs, case)) ** 2).mean()

def n_strong(dim, W):
    return CFG[dim]["n_t"] * W[0].shape[0] * W[0].shape[1]       # same point budget as the weak form

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

def train(model, dim, case, form, w_ic, iters, lr, W, band=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-6)
    nf, cfg = n_strong(dim, W), CFG[dim]
    for i in range(iters):
        phys = weak_residual(model, W, cfg["n_t"]) if form == "weak" else strong_residual(model, dim, nf)
        loss = W_PHYS * phys
        if w_ic > 0:
            loss = loss + w_ic * ic_loss(model, dim, case, cfg["n_ic"], band=band)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if i % 2000 == 0 or i == iters - 1:
            print(f"  iter {i}: loss {loss.item():.3e} | phys {phys.item():.3e}", flush=True)
    return model

# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def report(model, dim, case):
    """Rel. error of the TRANSIENT: ||u_pred - u|| / ||u - 1||. u -> 1 everywhere, so a plain
    rel. L2 on u would look tiny for any model that just outputs 1."""
    g = torch.linspace(0, 1, 201 if dim == 1 else 41, device=DEV)
    xs = g[:, None] if dim == 1 else torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    out = {}
    for t in T_EVAL:
        X = torch.cat([xs, torch.full((xs.shape[0], 1), t, device=DEV)], 1)
        ue, up = exact(X, case, dim), model(X)[:, 0]
        out[f"t={t}"] = ((up - ue).norm() / (ue - 1).norm()).item()
    return out

def terms(model, dim, case, W, band=0.0):
    """Weighted loss terms on FIXED draws, so start and end are compared on the same points."""
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    ic = ic_loss(model, dim, case, 2000, gen).item()
    wk = weak_residual(model, W, CFG[dim]["n_t"], gen).item()
    st = strong_residual(model, dim, n_strong(dim, W), gen).item()
    icb = ic_loss(model, dim, case, 2000, gen, band=band).item() if band > 0 else ic
    return dict(ic=W_IC * ic, ic_band=W_IC * icb, weak=W_PHYS * wk, strong=W_PHYS * st)

# ----------------------------------------------------------------------------- one run
def names(dim, case, init, form, ic, seed, band=0.0):
    if init == "pre":
        return f"d{dim}_{case}_start_s{seed}"
    return f"d{dim}_{case}_{init}_{form}_ic{ic}" + (f"_b{band:g}" if band > 0 else "") + f"_s{seed}"

def run_one(a):
    ck, rs = os.path.join(a.out, "ckpt"), os.path.join(a.out, "results")
    os.makedirs(ck, exist_ok=True); os.makedirs(rs, exist_ok=True)
    if a.smoke:
        for k in ITERS: ITERS[k] = 30
    dim, case, seed = a.dim, a.case, a.seed
    W = build_weak(dim, **CFG[dim])
    start_ck = os.path.join(ck, names(dim, case, "pre", "", 0, seed) + ".pt")

    def get_start():
        m = HardBC(dim).to(DEV)
        if os.path.exists(start_ck):
            m.load_state_dict(torch.load(start_ck, map_location=DEV)); return m
        m = pretrain(dim, case, seed, ITERS["pre"])
        torch.save(m.state_dict(), start_ck)
        return m

    name = names(dim, case, a.init, a.form, a.ic, seed, a.band)
    t0 = time.time()
    if a.init == "pre":
        m = get_start()
    else:
        if a.init == "warm":
            m = get_start()
        else:
            torch.manual_seed(seed); m = HardBC(dim).to(DEV)
        torch.manual_seed(seed)
        m = train(m, dim, case, a.form, W_IC * a.ic, ITERS[a.init], LR[a.init], W, band=a.band)
        torch.save(m.state_dict(), os.path.join(ck, name + ".pt"))
    res = dict(name=name, dim=dim, case=case, init=a.init, form=a.form, ic=a.ic, seed=seed, band=a.band,
               err=report(m, dim, case), terms=terms(m, dim, case, W, a.band), minutes=(time.time() - t0) / 60)
    json.dump(res, open(os.path.join(rs, name + ".json"), "w"), indent=2)
    print(name, " ".join(f"{k}:{v:.3f}" for k, v in res["err"].items()), flush=True)

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
    rs, logs = os.path.join(a.out, "results"), os.path.join(a.out, "logs")
    os.makedirs(rs, exist_ok=True); os.makedirs(logs, exist_ok=True)
    for wave_name, jobs in waves:
        queue = [j for j in jobs if not os.path.exists(os.path.join(rs, names(*j) + ".json"))]
        print(f"{wave_name}: {len(queue)} to run, {len(jobs) - len(queue)} done", flush=True)
        live, t0 = [], time.time()
        while queue or live:
            while queue and len(live) < a.jobs:
                d, c, i, f, ic, s, b = queue.pop(0); tag = names(d, c, i, f, ic, s, b)
                cmd = [sys.executable, "-u", os.path.abspath(__file__), "--dim", str(d), "--case", c,
                       "--init", i, "--form", f, "--ic", str(ic), "--seed", str(s), "--band", str(b),
                       "--out", a.out]
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
    summary(a)

# ----------------------------------------------------------------------------- summary
def summary(a):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(a.out, "results", "*.json"))]
    if not rows:
        print("no results yet"); return
    groups = {}
    for r in rows:
        b = r.get("band", 0.0)
        arm = "start" if r["init"] == "pre" else (f"{r['init']} {r['form']} IC{'on' if r['ic'] else 'off'}"
                                                  + (f" band{b:g}" if b > 0 else ""))
        groups.setdefault((r["dim"], r["case"], arm), []).append(r)
    order = ["start"] + [f"{i} {f} IC{x}" for i in ("warm", "cold") for f in ("weak", "strong") for x in ("on", "off")]
    order += sorted({k[2] for k in groups} - set(order))           # band arms
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
    p.add_argument("--jobs", type=int, default=6); p.add_argument("--out", default="heat_results")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dim", type=int, choices=(1, 2)); p.add_argument("--case", choices=("compat", "clash"))
    p.add_argument("--init", choices=("pre", "warm", "cold")); p.add_argument("--form", default="weak")
    p.add_argument("--ic", type=int, default=1); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--band", type=float, default=0.0, help="skip IC points within this distance of the boundary")
    p.add_argument("--band-sweep", action="store_true", help="run the band-masked IC arms (see launch_band)")
    p.add_argument("--bands", type=float, nargs="+", default=[0.05, 0.1])
    p.add_argument("--dims", nargs="+", default=["2"], help="dims for --band-sweep, e.g. --dims 1 2")
    a = p.parse_args()
    if a.all: launch_all(a)
    elif a.band_sweep: launch_band(a)
    elif a.summary: summary(a)
    else: run_one(a)