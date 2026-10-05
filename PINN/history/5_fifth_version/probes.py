"""
Per-term gradient-norm share on the weak-form arms.

Answers two things at once:

  (a) the reviewer's objection -- the existing diagnostic is measured only on
      ANCHORED checkpoints at W_ANCHOR=1.0, while the UNANCHORED arm is the
      load-bearing experiment. This measures both.

  (b) the provenance of the "92.9% res_phi / 7.0% res_C" figure in Section 5.2,
      which currently exists only as a notebook comment and is not in any
      results file. `split_C_pct` below is that quantity, computed properly.

Run from PINN/ with main.ipynb's cells 0-8 already executed (needs
compute_weak_loss_split, compute_essential_penalties, compute_anchor_loss_fdm,
dataset_tensors, quad_tensors, fdm_*, and the W_* globals).

    python per_term_share.py --probe          # 5 draws, 1 seed, ~5 min CPU
    python per_term_share.py --full           # 50 draws, 10 ckpts, GPU

CRITICAL: TE must be the modes-2 space (36 functions). Asserted below.
"""

import os, json, glob, argparse, statistics as st
import torch

KEYS = ["res_C", "res_phi", "bounds", "smooth", "ic_C", "ends_C", "ends_phi", "ANCHOR"]


def per_term_share(mdl, TE_local, B=32, draws=50, w_anchor=1.0, unit_weights=False):
    """Median gradient-norm share of every weighted objective term.

    w_anchor      : pass 0.0 for unanchored arms. Their anchor share is zero by
                    construction, so it is excluded from the denominator rather
                    than reported as a finding.
    unit_weights  : set every W_* to 1.0 before differentiating. Distinguishes a
                    dimensional imbalance from one the weights are creating.

    Returns (share_pct, raw_norms, split_C_pct) where split_C_pct is
    res_C / (res_C + res_phi), the two-term ratio Section 5.2 refers to.
    """
    assert TE_local[0].shape[0] == 36, (
        f"TE has {TE_local[0].shape[0]} test functions, expected 36 for modes=2")

    w = {k: 1.0 for k in KEYS} if unit_weights else {
        "res_C": float(W_RES_C), "res_phi": float(W_RES_PHI),
        "bounds": float(W_BOUNDS), "smooth": float(W_SMOOTH),
        "ic_C": float(W_IC_C), "ends_C": float(W_ENDS_C),
        "ends_phi": float(W_ENDS_PHI), "ANCHOR": float(w_anchor)}
    if unit_weights:
        w["ANCHOR"] = 1.0 if w_anchor > 0 else 0.0

    active = [k for k in KEYS if w[k] > 0]        # drop ANCHOR when w_anchor=0
    idx   = torch.arange(B, device=execution_device)
    batch = [t[:B].to(execution_device) for t in dataset_tensors]
    q     = [t.to(execution_device) for t in quad_tensors]
    t_    = [t.to(execution_device) for t in TE_local]
    P     = list(mdl.parameters())

    acc   = {k: [] for k in active}
    raw   = {k: [] for k in active}
    split = []

    for _ in range(draws):
        rC, rP, bd, sm = compute_weak_loss_split(
            mdl, batch, q, t_, PHYSICS_PARAMS, N_t=5, device=execution_device)
        _, _, ic, eC, eP = compute_essential_penalties(
            mdl, batch, device=execution_device)
        an = compute_anchor_loss_fdm(
            mdl, idx, batch, fdm_C, fdm_phi, fdm_x, fdm_y,
            PHYSICS_PARAMS, n_pts=4, device=execution_device)

        terms = {"res_C": w["res_C"]*rC, "res_phi": w["res_phi"]*rP,
                 "bounds": w["bounds"]*bd, "smooth": w["smooth"]*sm,
                 "ic_C": w["ic_C"]*ic, "ends_C": w["ends_C"]*eC,
                 "ends_phi": w["ends_phi"]*eP, "ANCHOR": w["ANCHOR"]*an}

        n = {}
        for k in active:
            g = torch.autograd.grad(terms[k], P, retain_graph=True, allow_unused=True)
            n[k] = float(sum((x**2).sum() for x in g if x is not None).sqrt().item())
            del g
        tot = sum(n.values())
        for k in active:
            raw[k].append(n[k])
            acc[k].append(100.0 * n[k] / tot)
        denom = n["res_C"] + n["res_phi"]
        split.append(100.0 * n["res_C"] / denom if denom > 0 else float("nan"))

        del terms, rC, rP, bd, sm, ic, eC, eP, an
        mdl.zero_grad(set_to_none=True)
        flush()

    del batch, q, t_, P
    flush()
    return ({k: st.median(v) for k, v in acc.items()},
            {k: st.median(v) for k, v in raw.items()},
            st.median(split))


def build_te_m2():
    """modes-2 test space, built explicitly. Do not inherit a global."""
    vs, gvs = generate_true_hp_test_functions(PTS_SPACE, D_EDGES, num_modes_x=2)
    vw, _ = generate_true_hp_test_functions(
        torch.cat([PTS_WALL_X, torch.zeros_like(PTS_WALL_X)], dim=-1),
        D_EDGES, num_modes_x=2)
    mk = torch.zeros(vs.shape[0], dtype=torch.bool)
    mk[[m * NUM_HATS for m in range(3)]] = True
    assert vw[mk].abs().sum() > 0 and vw[~mk].abs().sum() == 0
    return (vs, gvs, vw, mk)


def load(path):
    m = FactoredMIONet(soil_dim=8, fluid_dim=50, meta_dim=3, split_point="late")
    m.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    return m.to(execution_device).eval()


CAND = ["weak_dualnorm", "results/model_files/weak_form"]

def ckpt(tag):
    for d in CAND:
        p = os.path.join(d, f"{tag}_adam.pt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{tag}_adam.pt not in {CAND}")


# ---------------------------------------------------------------- probe -----
def probe(B=32, draws=5):
    """One unanchored seed, few draws. Tells you which way the objection goes."""
    TE_M2 = build_te_m2()
    m = load(ckpt("dn_plain_noanchor_s42"))
    sh, raw, split = per_term_share(m, TE_M2, B=B, draws=draws, w_anchor=0.0)
    print("\ndn_plain_noanchor_s42  (unanchored, 7 terms, W_ANCHOR=0)")
    for k, v in sorted(sh.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<10} {v:6.2f}%   (norm {raw[k]:.3e})")
    print(f"\n   res_C : res_phi  =  {split:.1f} : {100-split:.1f}")
    print("\n   Section 5.2 claims 7.0 : 92.9 at a trained point.")
    print("   If res_C here is in the tens of percent, the 'concentration")
    print("   equation is not being trained' reading does not hold for the")
    print("   arm the paper's central claim rests on.")
    del m; flush()
    return sh, raw, split


# ----------------------------------------------------------------- full -----
def full(B=32, draws=50):
    TE_M2 = build_te_m2()
    out = {"draws": draws, "b_eval": B,
           "weights": {"W_RES_C": float(W_RES_C), "W_RES_PHI": float(W_RES_PHI),
                       "W_BOUNDS": float(W_BOUNDS), "W_SMOOTH": float(W_SMOOTH),
                       "W_IC_C": float(W_IC_C), "W_ENDS_C": float(W_ENDS_C),
                       "W_ENDS_PHI": float(W_ENDS_PHI)},
           "note": "unanchored arms run at w_anchor=0.0; their anchor term is "
                   "excluded from the denominator rather than reported as 0%",
           "arms": {}}

    for arm, wa in [("dn_plain_noanchor", 0.0), ("dn_plain_anchored", 1.0)]:
        for unit in (False, True):
            key = f"{arm}{'_unitweights' if unit else ''}"
            rows = []
            for s in range(42, 47):
                tag = f"{arm}_s{s}"
                m = load(ckpt(tag))
                sh, raw, split = per_term_share(m, TE_M2, B=B, draws=draws,
                                                w_anchor=wa, unit_weights=unit)
                rows.append({"seed": s, "share_pct": sh, "raw_norms": raw,
                             "split_C_pct": split})
                print(f"{tag:<28}{'unit' if unit else 'prod':<6}"
                      f"res_C {sh['res_C']:5.1f}%  res_phi {sh['res_phi']:5.1f}%"
                      + (f"  ANCHOR {sh['ANCHOR']:5.1f}%" if wa > 0 else "")
                      + f"   C:phi {split:5.1f}:{100-split:.1f}")
                del m; flush()

            agg = {}
            for k in rows[0]["share_pct"]:
                v = [r["share_pct"][k] for r in rows]
                agg[k] = {"median": st.median(v), "min": min(v), "max": max(v)}
            sp = [r["split_C_pct"] for r in rows]
            out["arms"][key] = {
                "n": len(rows), "w_anchor": wa, "unit_weights": unit,
                "share_pct": agg,
                "split_C_pct": {"median": st.median(sp), "min": min(sp), "max": max(sp)},
                "per_seed": rows}

    d = ("results/json_files/compiled_results"
         if os.path.isdir("results/json_files/compiled_results")
         else "multiseed_m5/compiled")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "per_term_share.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nsaved {p}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--draws", type=int, default=None)
    a = ap.parse_args()
    if a.probe:
        probe(B=a.B, draws=a.draws or 5)
    elif a.full:
        full(B=a.B, draws=a.draws or 50)
    else:
        ap.error("pass --probe or --full")