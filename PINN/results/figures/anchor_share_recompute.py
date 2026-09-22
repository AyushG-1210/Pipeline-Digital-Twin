"""
Anchor gradient share on m2_bs64_matched (seeds 42-46), recomputed with every
input that could make two runs disagree printed alongside the result:

  * draw count                    (D = 5 and D = 50, both reported)
  * W_ANCHOR / W_RES_C / W_RES_PHI and the other static weights (cell 17)
  * the full per-term gradient-NORM dictionary, all eight terms
  * the denominator: share_k = ||grad L_k|| / sum_j ||grad L_j||  over those
    eight terms (sum of norms, NOT norm of the summed gradient)
  * batch (first b_eval=64 TRAINING rows, idx = arange) and test space (modes=2, K=36)

All function definitions are pulled from main.ipynb by AST (cells 3, 5, 11, 13,
15, 17, 23) -- nothing is re-implemented here. `anchor_share` (cell 23) is called
unchanged; a thin copy of its inner loop is used only to capture the per-term
norms it computes but does not return.

Run from PINN/:  python results/figures/anchor_share_recompute.py [--draws 50]
"""
import ast, json, os, sys, time, statistics as st
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PINN = os.path.abspath(os.path.join(HERE, "..", ".."))
NB = os.path.join(PINN, "main.ipynb")
DATA = os.path.join(PINN, "data")
CKPT = os.path.join(PINN, "results", "model_files", "weak_form")
JSON_W = os.path.join(PINN, "results", "json_files", "weak_form")
SEEDS = [42, 43, 44, 45, 46]
DRAWS = [5, 50]
if "--draws" in sys.argv:
    DRAWS = [int(sys.argv[sys.argv.index("--draws") + 1])]

torch.set_num_threads(max(1, os.cpu_count() - 1))
nb = json.load(open(NB, encoding="utf-8"))


def cell_defs(i, names=None):
    src = "".join(nb["cells"][i]["source"])
    keep = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nm = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name):
                nm = t.id
            elif isinstance(t, ast.Tuple) and all(isinstance(e, ast.Name) for e in t.elts):
                nm = t.elts[0].id
            else:
                continue
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(ast.get_source_segment(src, node)); continue
        else:
            continue
        if names is None or nm in names:
            keep.append(ast.get_source_segment(src, node))
    return "\n".join(keep)


ns = {"execution_device": "cpu", "torch": torch, "np": np}
exec(cell_defs(3, {"RHO_MIN", "LOG_RHO_MIN", "soil_to_log_rho", "interp_profile", "gamma_of_x",
                   "C_BULK_MIN", "fluid_to_cbulk", "interp_profile_bn"}), ns)
exec(cell_defs(5, {"PHYSICS_PARAMS"}), ns)                 # the L_x-bearing version
exec("".join(nb["cells"][11]["source"]), ns)               # quadrature (prints only)
exec(cell_defs(13), ns)                                    # model
exec(cell_defs(15), ns)                                    # weak loss, penalties, anchor loss
exec(cell_defs(17, {"W_RES_C", "W_RES_PHI", "W_ANCHOR", "W_BOUNDS", "W_SMOOTH",
                    "W_IC_C", "W_ENDS_C", "W_ENDS_PHI"}), ns)
exec(cell_defs(23, {"CFG", "anchor_share"}), ns)           # the measurement function itself
ns["flush"] = lambda: None
assert ns["PHYSICS_PARAMS"].get("L_x") == 10.0, "cell-5 PHYSICS_PARAMS (with L_x) expected"

# ---- data the function reads from module scope --------------------------
cache = torch.load(os.path.join(DATA, "dataset_v2.pt"), map_location="cpu", weights_only=False)
soil, fluid, meta, raw = cache["tensors"]
ns["dataset_tensors"] = [t[0:700] for t in (soil, fluid, meta, raw)]
tr = np.load(os.path.join(DATA, "fdm_train_0_700.npz"))
f32 = lambda a: torch.tensor(a, dtype=torch.float32)
ns["fdm_C"], ns["fdm_phi"], ns["fdm_x"], ns["fdm_y"] = f32(tr["C"]), f32(tr["phi"]), f32(tr["x"]), f32(tr["y"])
assert ns["fdm_C"].shape[0] == 700

# ---- modes=2 test space, exactly as cell 47 rebuilds it -------------------
gen = ns["generate_true_hp_test_functions"]
V_S2, GRAD_V_S2 = gen(ns["PTS_SPACE"], ns["D_EDGES"], num_modes_x=2)
V_W2, _ = gen(torch.cat([ns["PTS_WALL_X"], torch.zeros_like(ns["PTS_WALL_X"])], dim=-1), ns["D_EDGES"], num_modes_x=2)
MASK2 = torch.zeros(V_S2.shape[0], dtype=torch.bool)
MASK2[[m * ns["NUM_HATS"] for m in range(3)]] = True
assert V_S2.shape[0] == 36
ns["TE"] = (V_S2, GRAD_V_S2, V_W2, MASK2)

W = {k: float(ns[k]) for k in ("W_RES_C", "W_RES_PHI", "W_ANCHOR", "W_BOUNDS", "W_SMOOTH", "W_IC_C", "W_ENDS_C", "W_ENDS_PHI")}
B_EVAL = ns["CFG"]["b_eval"]
print("static weights (cell 17):", W)
print(f"b_eval={B_EVAL}  batch = training rows 0..{B_EVAL-1}, idx=arange  test space K={V_S2.shape[0]} (modes=2)")
print("share_k = ||grad L_k|| / sum over 8 terms of ||grad L_j||   (cell 23 anchor_share)\n")

TERMS = ["res_C", "res_phi", "bounds", "smooth", "ic_C", "ends_C", "ends_phi", "ANCHOR"]


def per_term_norms(mdl, B, draws):
    """Copy of the loop body of cell-23 anchor_share, returning the raw per-term
    norms as well as the two shares. Same batch, same weights, same reduction."""
    g = ns
    idx = torch.arange(B)
    batch = [t[:B] for t in g["dataset_tensors"]]
    P = list(mdl.parameters())
    rows = []
    for _ in range(draws):
        rC, rP, bd, sm = g["compute_weak_loss_split"](mdl, batch, g["quad_tensors"], g["TE"], g["PHYSICS_PARAMS"], N_t=5, device="cpu")
        _, _, ic, eC, eP = g["compute_essential_penalties"](mdl, batch, device="cpu")
        an = g["compute_anchor_loss_fdm"](mdl, idx, batch, g["fdm_C"], g["fdm_phi"], g["fdm_x"], g["fdm_y"], g["PHYSICS_PARAMS"], n_pts=4, device="cpu")
        terms = {"res_C": W["W_RES_C"] * rC, "res_phi": W["W_RES_PHI"] * rP, "bounds": W["W_BOUNDS"] * bd,
                 "smooth": W["W_SMOOTH"] * sm, "ic_C": W["W_IC_C"] * ic, "ends_C": W["W_ENDS_C"] * eC,
                 "ends_phi": W["W_ENDS_PHI"] * eP, "ANCHOR": W["W_ANCHOR"] * an}
        n = {}
        for k, v in terms.items():
            gr = torch.autograd.grad(v, P, retain_graph=True, allow_unused=True)
            n[k] = float(sum((x ** 2).sum() for x in gr if x is not None).sqrt().item())
        tot = sum(n.values())
        rows.append({"norms": n, "ANCHOR": 100 * n["ANCHOR"] / tot, "weak": 100 * (n["res_C"] + n["res_phi"]) / tot})
        mdl.zero_grad(set_to_none=True)
    med = lambda k: st.median(r[k] for r in rows)
    return {"ANCHOR": med("ANCHOR"), "weak": med("weak"),
            "norms_median": {t: st.median(r["norms"][t] for r in rows) for t in TERMS},
            "share_median_all_terms": {t: st.median(100 * r["norms"][t] / sum(r["norms"].values()) for r in rows) for t in TERMS}}


out = {"arm": "m2_bs64_matched", "phase": "adam", "weights": W, "b_eval": B_EVAL, "n_test_functions": 36,
       "denominator": "sum of the 8 per-term gradient norms", "torch_seed": 0, "runs": {}}
for D in DRAWS:
    print("=" * 100 + f"\n draws = {D}\n" + "=" * 100)
    res = {}
    for s in SEEDS:
        m = ns["FactoredMIONet"](soil_dim=8, fluid_dim=50, meta_dim=3, split_point="late")
        m.load_state_dict(torch.load(os.path.join(CKPT, f"m2_bs64_matched_s{s}_adam.pt"), map_location="cpu", weights_only=False)["state_dict"])
        m.eval()
        t0 = time.time()
        torch.manual_seed(0)                                   # draws are random (t, penalty pts, anchor pts)
        r = per_term_norms(m, B_EVAL, D)
        if D <= 5:   # cross-check: the unmodified cell-23 function on the same seed stream
            torch.manual_seed(0)
            ref = ns["anchor_share"](m, B_EVAL, draws=D)
            assert abs(ref["ANCHOR"] - r["ANCHOR"]) < 1e-6 and abs(ref["weak"] - r["weak"]) < 1e-6, "loop copy diverged from anchor_share"
        res[s] = r
        nm = r["norms_median"]
        print(f"  s{s}: ANCHOR {r['ANCHOR']:5.1f}%  weak {r['weak']:5.1f}%   ({time.time()-t0:.0f}s)")
        print("        median ||grad|| per term: " + "  ".join(f"{t}={nm[t]:.3e}" for t in TERMS))
        print("        median share per term (%): " + "  ".join(f"{t}={r['share_median_all_terms'][t]:.1f}" for t in TERMS))
    a = [res[s]["ANCHOR"] for s in SEEDS]; w = [res[s]["weak"] for s in SEEDS]
    print(f"\n  n=5, draws={D}:  ANCHOR median {st.median(a):.1f}% [{min(a):.1f}-{max(a):.1f}]   "
          f"weak (res_C+res_phi) median {st.median(w):.1f}% [{min(w):.1f}-{max(w):.1f}]")
    out["runs"][str(D)] = {"per_seed": {str(s): res[s] for s in SEEDS},
                           "ANCHOR": {"median": st.median(a), "min": min(a), "max": max(a)},
                           "weak": {"median": st.median(w), "min": min(w), "max": max(w)}}

# ---- the stored numbers this has to be reconciled against --------------------
print("\n" + "=" * 100 + "\n STORED VALUES FOR THE SAME ARM\n" + "=" * 100)
comp = json.load(open(os.path.join(PINN, "results/json_files/compiled_results/anchor_share_m2.json")))
print(f"  compiled/anchor_share_m2.json  (file says draws={comp['draws']}, W_ANCHOR={comp['W_ANCHOR']}):  "
      f"ANCHOR {comp['anchor']['median']:.1f} [{comp['anchor']['min']:.1f}-{comp['anchor']['max']:.1f}]   "
      f"weak {comp['weak']['median']:.1f} [{comp['weak']['min']:.1f}-{comp['weak']['max']:.1f}]")
a = []; w = []
for s in SEEDS:
    sh = json.load(open(os.path.join(JSON_W, f"m2_bs64_matched_s{s}.json")))["adam"]["share"]
    a.append(sh["ANCHOR"]); w.append(sh["weak"])
print(f"  m2_bs64_matched_s4X.json['adam']['share']  (evaluate(), draws=5):        "
      f"ANCHOR {st.median(a):.1f} [{min(a):.1f}-{max(a):.1f}]   weak {st.median(w):.1f} [{min(w):.1f}-{max(w):.1f}]")
print("  paper text:                                                              ANCHOR 56 [47-73]   weak 14 [7-16]")
out["stored"] = {"compiled_anchor_share_m2": comp, "evaluate_blocks": {"ANCHOR": a, "weak": w}}
json.dump(out, open(os.path.join(HERE, "anchor_share_recompute.json"), "w"), indent=1)
print("\nsaved results/figures/anchor_share_recompute.json")
