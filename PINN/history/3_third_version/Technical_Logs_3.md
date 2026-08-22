# AmorFlux — Technical Audit Log v4
**Purpose:** chronological record of failures, root causes, and what has been
verified vs. merely proposed. Supersedes Technical_Logs v1–v3.1, both of which
contain claims that are now known to be stale (see §0).

**Status legend:**
- ✅ **VERIFIED** — implemented AND confirmed by a run/audit/validation check
- 🔶 **PROPOSED** — designed but not implemented or not run
- ⚠️ **UNRESOLVED** — mechanism understood, fix incomplete or unvalidated
- ❌ **REFUTED** — tested and shown false (kept deliberately; these are results)

---

## 0. CORRECTIONS TO EARLIER LOGS — read before citing anything

Several claims in v1–v3.1 are now known to be wrong. They are listed here rather
than deleted, because the wrong turns are themselves the paper's evidence.

| Old claim | Correction |
|---|---|
| "ultra-thin reactive boundary layer ~2.7e-6" | **Artifact.** Product of `k_rate=1e-5` (50× above literature) plus a spurious 440 mV overpotential. No boundary layer exists in the corrected regime. |
| "FDM ground truth `dC/dy ≈ -1.000`" | Regime-dependent. Now ≈ `-0.78`, and varies with WT and rho. |
| "the `50.0` clamp distorts kinetics over ~the entire C range" | Now **inactive** — max rate ≈ 0.73 in the corrected regime. |
| v3.1 item 1: "~0.015 s/epoch, 300× speedup" | Measured **1.06 s/epoch**. The same entry's own "Nuance Correction" contradicts the headline. |
| v3.1 item 3: m=0 marked "LIVE BUG" | Fixed and verified (§2.1). |
| v3.1 item 4: `η = -φ - E_eq` marked ✅ | Superseded — that convention *was* the 440 mV artifact (§1.2). |
| `[cite: 2]`, `[cite: 2,3]` markers throughout | LLM citation leakage. Same class as the `[cite: 1]` fragments once found inside `MIONet.forward()`. Strip all. |

**README:** still reports L2 errors and inference timings for runs that were
never executed, and describes ChromaDB/Neo4j/YOLOv8/Three.js components absent
from the codebase. It contradicts these logs directly. Resolve before any
submission.

---

## 1. PHYSICS — the root cause of ~2 months of misdirected work

### 1.1 `k_rate` was 50× above the literature ceiling
**Status: ✅ VERIFIED**

`k_rate = 1e-5` was never derived from anything. Converting literature exchange
current densities for API 5L in soil (`i_0 ∈ [1e-3, 1e-1] A/m²`) via
`k = i_0/(nFC_bulk)`, `n=2`, `C_bulk ≈ 0.25 mol/m³`:

| i_0 (A/m²) | k (m/s) | Da = kL/D |
|---|---|---|
| 1e-3 | 2.07e-08 | 0.263 |
| 1e-2 | 2.07e-07 | 2.633 |
| 1e-1 | 2.07e-06 | 26.3 |

The old value corresponds to `Da = 127`. Corrected to `k_rate = 2.07e-7`.

### 1.2 The overpotential reference injected a spurious 440 mV
**Status: ✅ VERIFIED**

Grounding the steel at `φ=0` while `E_eq = -0.44 V` gives `η = 0.44 V` —
`exp(kη) = 5236`. Combined with 1.1, effective `A = Da·exp(kη) ≈ 6.9e5`.
Fixed by adding `E_corr` (free-corrosion reference), `E_corr = E_eq ⇒ η = -φ`.

**Trap found later:** `params.get("E_corr", 0.0)` silently restored the broken
convention for any params dict omitting the key — `run_validation()` was hitting
this. Now raises `KeyError`.

### 1.3 Consequence: the problem was degenerate
**Status: ✅ VERIFIED**

| Regime | slope `s` range | variance across all inputs |
|---|---|---|
| Original (`A ≈ 6.9e5`) | -0.999999 .. -0.999997 | **0.00022%** |
| Corrected (`A ≈ 1.4–5.5`) | -0.844 .. -0.573 | **32%** |

At `A ≫ 1` the wall is a perfect sink and `C = 1 − y` for *every* input. There
was no operator to learn, and "80% accuracy" was trivially achievable against a
constant.

**The methodological point:** v1 listed "independent check that `k_rate`,
`alpha` are realistic" as open item #3 and deferred it, while months of effort
went into samplers, test functions, coordinate warping and encodings — all built
to resolve a boundary layer that the deferred check would have shown did not
exist.

---

## 2. WEAK-FORM / VARIATIONAL FAILURES

### 2.1 The m=0 test-function blind spot
**Status: ✅ VERIFIED — numerically, on the production quadrature**

Test functions `v(x,y) = cos(mπx)·sin(nπy/2)` with `m ≥ 1` are **exactly
orthogonal to any x-uniform error**, since `∫₀¹cos(mπx)dx = 0` for integer `m≥1`.

Projection of a constant residual:

| basis | max \|projection\| |
|---|---|
| `m ∈ {1..4}` (original) | **1.77e-16** |
| `m ∈ {0..4}` (corrected) | **0.636** |

Empirical signature before the fix: `phi` flat across x (0.2776 / 0.2774 /
0.2761 at x = 0.2/0.5/0.8) at 145× the true value, with `res_phi` at 1.3e-6.

**Most transferable finding in the project.** It is a property of the test
space, not of this problem. Demonstrating it on a second PDE with uniform BCs
would upgrade it from "a bug we found" to "a class of bug in VPINNs."

### 2.2 Residual–accuracy anti-correlation
**Status: ✅ VERIFIED — controlled sweep, 2500 epochs, both architecture arms**

`W_RES_PHI` sweep, all else fixed:

| W_RES_PHI | C rel L2 | phi rel L2 | phi_wall | res_phi (final) |
|---|---|---|---|---|
| 1.0 | 0.1235 | 0.5973 | -5.995e-04 | 1.835e-03 |
| 5.0 | 0.2772 | 0.7239 | -4.086e-04 | 7.774e-04 |
| 10.0 | 0.3917 | 0.8244 | -2.609e-04 | 4.007e-04 |
| FDM | — | — | **-1.408e-03** | — |

Objective improves **4.6×** while `phi_wall` error grows and `C` degrades 3.2×.
Monotonic, reproduced in both arms.

**Corroborating instances:** `res_C` rose `1.47e-2 → 1.68e-2` while `C` accuracy
improved 2×; a run with `res_C = 1.09e-2` / `res_phi = 1.65e-3` (both essentially
satisfied) had `C_wall` 2.7× wrong at high WT and no operator behaviour at all.

### 2.3 Test functions must vanish on Dirichlet boundaries
**Status: ✅ VERIFIED**

The final hat function equalled 1 at `d=1`, where the ansatz hard-enforces the
Dirichlet condition. Green's identity drops the boundary term there, which is
valid only if `v=0` — so the formulation silently imposed a spurious zero-flux
condition on top of the Dirichlet, over-determining that boundary. The offending
test function spanned **32% of the domain**. Fixed by dropping the last hat
(13 → 12; `V_SPACE` [39,·] → [36,·]).

Worth stating in the paper: hard-enforced BC ansatzes make this easy to miss.

### 2.4 Aggregation dilutes boundary constraints
**Status: ✅ VERIFIED**

Only the boundary half-hat has support at the wall: 3 of 39 test functions. A
uniform `mean()` gives the Butler-Volmer condition ~7.7% of the residual it
should carry. Predicted signature `1/39 = 0.0256` vs measured `res_phi = 0.0155`.

Hand-weighting the wall rows failed at every value (`W=5` → 29% flux error with
`C>1`; `W=1` → 87% with a flat field). FastVPINNs per-element aggregation (mean
within element, sum across elements) removes the knob.

---

## 3. NORMALIZATION AND WEIGHTING

### 3.1 Residual normalization is a precondition, not a tuning knob
**Status: ✅ VERIFIED**

Loss-term gradient traces spanned **1.7e19**:

| term | Tr(K) before | after normalization |
|---|---|---|
| res_C | 3.1258e+12 | 2.7711e+01 |
| res_phi | 4.5935e-04 | 6.8475e-06 |
| bounds | 1.8443e-07 | 3.4519e-08 |

`res_C` held 99.99999% of the gradient. Two dimensional causes:

- **`Fo` inside the mass residual**, varying 16× across a batch (`48.9` at
  WT=1.0in to `782.1` at WT=0.25in) → a **256× trace spread between samples of
  the same batch**. Fixed by `res_C /= Fo`.
- **`gamma` scaling phi**: `phi = O(gamma)` at the solution, so the squared
  residual is `O(gamma²)` and the trace `O(gamma⁴)`. Fixed by `res_phi /= gamma`.

Ratio `res_C/res_phi`: **3.75e9 → 0.0475**. Equal weighting then follows on
dimensional grounds rather than by tuning.

**Caution, learned later:** the `/gamma` normalization *overshot*. A later trace
measurement at a trained point gave `res_phi` 92.9% share vs `res_C` 7.0% —
`res_phi` now dominates by 13.2×. The fix flipped which term owns the gradient
rather than balancing them.

### 3.2 Adaptive NTK weighting fails in both available forms
**Status: ❌ REFUTED (a caveat on Wang/Yu/Perdikaris Algorithm 1)**

- **Unrenormalized:** converged terms' traces shrink → weights inflate without
  bound. `λ_res_phi` reached **3.5e5** while `res_phi` itself sat at **1e-8** —
  the scheme chases terms it has already solved.
- **Renormalized to a fixed budget:** the squeeze starves terms that still
  matter. `λ_res_C` 0.616→0.139, `λ_bounds` 0.658→0.064, `C > 1` returned.

Both are feedback loops driven by the run's own state. Static, dimensionally
normalized weights beat both.

**Also:** traces are seed-unstable — `res_C` varied 11.8× and `bounds` 45.4×
between two untrained inits, giving a 64× swing in derived weights. Any scheme
keying off a single measurement inherits that noise.

### 3.3 Reference-state trace measurement does not work
**Status: ❌ REFUTED**

Measuring `Tr(K_i)` at a solution-matched network to set static weights fails
because *every* term is satisfied there by definition — all traces collapse
toward zero and their ratios measure fit-quality noise, not gradient scale.
`res_C/res_phi` came out `3.75e9`, which is meaningless.

Additional finding: `ic_C` had the **largest** trace at that state, revealing
that the reference itself violates the initial condition (the isolation test
fits the steady state at all `t`, including `t=0` where `C` should be 0).

### 3.4 The `Fo`-on-flux dimensional bug
**Status: ✅ VERIFIED**

Green's identity on `∂C/∂t = Fo∇²C` gives
`∫(∂C/∂t)v + Fo∫∇C·∇v − Fo∮(∂C/∂n)v = 0` — **`Fo` multiplies the boundary term
too.** The code had `Fo` on stiffness but not flux, so the network satisfied the
residual with a wall gradient of `rate/Fo`:

| WT (in) | Fo | gradient understated by |
|---|---|---|
| 0.25 | 782.1 | 782× |
| 0.50 | 195.5 | 196× |
| 1.00 | 48.9 | 49× |

Does **not** cancel at steady state (only the correct form has `Fo` cancel),
which is why it survived earlier checks.

---

## 4. ARCHITECTURE

### 4.1 Capacity was never the limit
**Status: ✅ VERIFIED — supervised isolation test**

Fitting the identical architecture to the verified solution under plain MSE
reaches **0.016–0.05% wall-flux error**, `~3e-4` bulk L2, repeatedly, across
every physics regime tested. Capacity, ansatz conditioning and optimizer are all
eliminated as causes of the weak-form gap.

**This protocol is itself a contribution:** verified analytic/numerical reference
→ supervised isolation fit → weak-form comparison. It separates *cannot
represent* / *cannot optimize* / *objective does not measure*.

### 4.2 Trunk split point: null result
**Status: ✅ VERIFIED (negative)**

Hypothesis: shared trunk layers cause negative transfer between `C` and `phi`.
Parameter-matched (83,072 vs 82,792, 0.34% apart):

| W_RES_PHI | C L2 late | C L2 early | difference |
|---|---|---|---|
| 1.0 | 0.12353 | 0.12139 | 1.7% |
| 5.0 | 0.27718 | 0.28268 | 2.0% |
| 10.0 | 0.39171 | 0.38331 | 2.1% |

Null at every weight. **Methodological note:** an unmatched version (early arm
32% smaller) appeared to show a 57% advantage for late-split. Capacity confounds
are easy to mistake for architectural effects.

### 4.3 Coordinate encoding: added capacity was harmful
**Status: ✅ VERIFIED (negative)**

Fourier features + `tanh`/log coordinate warping were added to resolve a
boundary layer that (per §1.3) did not exist. A plain MLP trunk on raw
coordinates matched or beat them, and the warped trunk produced a **wrong-signed
wall derivative** (`+12.5` vs true `-0.764`) despite an accurate profile —
disqualifying when the deliverable *is* a boundary derivative.

Also found: the frequency band was `2**linspace(0,4,16)` → max frequency **16**,
not the intended 1024. Combined with a 10× warp, the encoding was ~6,250× short
of the feature scale it was built for. It had been contributing nothing.

### 4.4 Hard-enforced Dirichlet closes one loophole, not all of them
**Status: ✅ VERIFIED**

`C = 1 + (d−1)·NN`, `phi = (d−1)·NN` enforces the Dirichlet condition identically
(`dir_C`, `dir_phi` traces exactly `0.0`), closing a failure mode where the
network satisfied a 50-point penalty with an arbitrarily sharp spike.

**Critical caveat:** a zero gradient trace can be *structural* (ansatz enforces
it) or merely *inactive* (a `clamp` flat because the field is currently in
range). `loss_bounds` measured zero and was removed on that basis — `C` then ran
to **3.19**, climbing *toward* the wall. Restoring it at weight 100 fixed both
the overshoot and a U-shaped profile artifact.

---

## 5. OPTIMIZATION AND METRICS

### 5.1 Scheduler-induced silent training death
**Status: ✅ VERIFIED**

`ReduceLROnPlateau(patience=100)` on a loss swinging 5.1× on batch noise fired
repeatedly on that noise: LR fell **1024×** to `9.77e-08` and `res_C` sat flat
for the final 1000 epochs while training had effectively stopped. The loss log
looks like convergence. `CosineAnnealingLR` with `eta_min` cannot collapse this
way.

### 5.2 Gradient clipping placed after `optimizer.step()`
**Status: ✅ VERIFIED**

A no-op. Correcting the order improved a supervised fit by **2564×**
(`2.07e-04 → 8.07e-08`). Implementation slip, not a finding — listed only so it
is not rediscovered.

### 5.3 Conditioning of the reported quantity
**Status: ✅ VERIFIED**

`C_wall = 1 + s` with `s ≈ -1` is a catastrophic cancellation — ~6 digits
destroyed by construction. Wall *flux* `dC/dy = s` is `O(1)`.

Demonstration: a **0.21% error in the raw network output** produced a **1040×
error in `C_wall`** while the flux from the same model was accurate to 0.16%.

Report wall flux (which sets corrosion current density) as the headline. Note
also the visualization trap: **a linear-axis plot cannot distinguish 1e-2 from
1e-6**, which caused a 52× error to be misread as a perfect fit.

### 5.4 Operator fidelity needs its own metric
**Status: ✅ VERIFIED**

Pointwise metrics (`C L2`, flux error) can look good while the model predicts
the batch average and ranks environments randomly. Two metrics added:

- `pred_spread` — predicted `C_wall` range as % of mean, vs true
- `corr(pred, true)` across samples

At `W_RES_C=1`: flux error 10.6% but `corr = +0.353`, spread 6.6% vs true 132%.
Raising to `W_RES_C=8–10` gave `corr = +0.946`, spread 110% — a far better model
that scored *worse* on flux. Judging on `C L2` alone would have discarded it.

**Guard:** normalize `pred_spread` by the *true* mean. Using the predicted mean
gave a bogus 3305% when predictions collapsed toward zero.

### 5.5 Dataset regeneration invalidated cross-run comparison
**Status: ✅ VERIFIED**

The generator was unseeded and re-executed between runs. Validation targets
shifted **26–28% at high rho** between two runs whose model predictions were
bit-identical. Every comparison spanning a regeneration was invalid. Fixed by
seeding + caching to disk.

---

## 6. CURRENT STATE (v3 → v4 transition)

### 6.1 Best verified weak-form result
`W_RES_C=10`, `W_RES_PHI=1`, `W_ANCHOR=10`, anchor at `d=0`, 700 epochs,
`split="late"`, `N_t=5`, dropped hat, per-sample rho:

```
C rel L2 0.1276 | flux 64.0% | pred spread 110.2% (true 135.2%) | corr +0.946
```

**Not reproducible after the server loss.** Post-loss runs on the reseeded
dataset land at `C L2 ≈ 0.42–0.74` with `corr` ranging **-0.555 to +0.892**
across nominally similar configs. At n=1 per config the variance exceeds the
config differences — several earlier "best config" conclusions may have been
noise. **Multi-seed runs are required before any config claim goes in the paper.**

### 6.2 The anchor dominates the loss
**Status: ⚠️ UNRESOLVED**

Measured share at init, `W_ANCHOR=10`:

| term | weight | raw | share |
|---|---|---|---|
| ANCHOR | 10.0 | 1.277e+02 | **99.48%** |
| res_phi | 1.0 | 4.553e+00 | 0.35% |
| res_C | 10.0 | 1.128e-01 | 0.09% |

The anchor's phi half divides by per-sample `gamma`, which reaches `1.016e-03`;
`(0.1/1.016e-3)² ≈ 9700` for a single low-rho sample. The entire weak form is
0.52% of the gradient — this is supervised regression on one point per sample
with the PDE as a rounding error.

Normalizing by `gamma.mean()` fixes the imbalance but was measured only at 2500
epochs; at 700 epochs the *unnormalized* version produced the best run in the
project. Unresolved which is correct.

### 6.3 The operator is degenerate in its inputs
**Status: ✅ VERIFIED — this is the central open problem**

| input | dims | enters the physics? |
|---|---|---|
| soil GRF | 4 | no |
| fluid GRF | 50 | no |
| meta[0] WT | 1 | **yes** → Fo, Da |
| meta[1] OD | 1 | no |
| meta[2] defect | 1 | no |
| meta[3] rho | 1 | **yes** → gamma |

**2 of 58 dimensions affect the solution.** A correct model must assign zero
attribution to the other 56 — XAI would be reporting the truth, and there is
nothing to explain. No loss tuning changes this; it is a property of the
governing equations as written.

### 6.4 sigma(x) was implemented and is too weak a lever
**Status: ✅ VERIFIED (partial success, wrong variable)**

`sigma(x)` from an 8-node soil GRF, FDM extended to x-varying coefficients,
regression test passing (`C_wall` and `phi_wall` to `1.2e-11`, x-uniformity
`3.4e-12`). Ground truth: 200 samples, `Nt=200`, ~20 min.

But the sensitivity ranking was not checked first:

| input | swing | effect on C_wall |
|---|---|---|
| rho | 40× | 39% |
| WT | 4× | 102% |
| **C_bulk** | **3×** | **96%** |

`rho` is the weakest of the three. Realized x-variation in the FDM data is
`0.00357`, while across-sample variation is 2.4× — a **68× stronger** signal.
Result: `x_corr = +0.073`, predicted x-variation 36× *larger* than truth, i.e.
pure noise.

**Next: `C_bulk(x)` from the fluid GRF** — it enters as the Dirichlet condition
`C(x, y=0) = C_bulk(x)`, so the solution scales with it almost directly, and it
uses the 50-node profile that is currently entirely decorative.

### 6.5 Batch-size / dataset-size interaction
**Status: ✅ VERIFIED**

Cutting the dataset to the 200 FDM rows at `batch_size=256` gave 1 batch/epoch
instead of 4 — a 3.5× reduction in gradient steps (2800 → 800) that presents as
a 36× wall-clock speedup and looks like "not learning." Use `batch_size=64`.

---

## 7. VERIFIED INFRASTRUCTURE (Methods material)

### 7.1 FDM solver
Independent analytic steady state from the self-consistent Butler-Volmer balance
`s = −Da(1+s)exp(kη)`:

| quantity | analytic | FDM | rel error |
|---|---|---|---|
| C_wall | 1.4485103710e-06 | 1.4485103700e-06 | 6.8e-10 |
| phi_wall | -1.9296972048e-03 | -1.9296972000e-03 | 2.5e-09 |
| rate | 0.9999985515 | 0.9999985515 | 1.5e-14 |

Convergence to **1e-11** across `Ny ∈ {31,61,121}`, `Nx ∈ {11,21}`,
`Nt ∈ {200,400}`, stretch `k ∈ {4,6,8}`. Plus `Da=0` vs the analytic slab
solution: `3.9e-08` relative L2.

This verifies the **solver**, not the physical model — the kinetic constants
carry their own separate assumptions (§8).

### 7.2 Regression test for the x-varying path
Constant `rho` must collapse to the verified 1D case. Checks both `C_wall`
against the analytic balance and x-uniformity of the solution. Run after any
FDM edit; it catches spurious x-structure introduced by the per-x code path.

---

## 8. OPEN RESEARCH ITEMS

1. **`C_bulk(x)`** — the strong x-lever, uses the 50-node fluid profile (§6.4).
2. **Anchor normalization** — per-sample vs mean `gamma`, unresolved (§6.2).
3. **Multi-seed** — every config claim is currently n=1 against variance that
   exceeds the effects being claimed (§6.1).
4. **`corr(rho_x, C_wall) = -0.86`** — measured negative; the intuitive argument
   (high rho → low sigma → larger |phi| → lower eta → slower rate → higher
   C_wall) predicts positive. One step is wrong. This is the sign of the physical
   dependence XAI will attribute, so it must be derived, not assumed.
5. **`C_bulk = 0.25 mol/m³` and `E_corr = E_eq`** are estimates, not
   measurements. Cite or sensitivity-test before they anchor results.
6. **Trunk factorization** — branches are factored; the trunk still evaluates
   `B×N` points where `N` are unique (256× redundant). v3.1 item 1 claims this is
   done; it is not.
7. **Temporal sampling** — with `Fo ≈ 196` the transient occupies ~2.6% of the
   domain. Stratified sampling was added; its effect has not been isolated.

---

## 9. WHAT CARRIES THE PAPER

1. **Residual–accuracy anti-correlation** (§2.2) — a designed experiment, not an
   observation.
2. **The m=0 blind spot** (§2.1) — the only finding that transfers beyond this
   problem.
3. **Parameter validation before architectural work** (§1) — with a dated record
   of the deferral and its cost.

Supporting: the isolation protocol (§4.1), metric conditioning (§5.3),
normalization as precondition (§3.1), and the two architecture negatives (§4.2,
§4.3). The NTK-weighting refutation (§3.2) is a citable caveat on a published
method.

**Closest related work:** Farooqi et al., *PINNACLE*, AI4Mat-NeurIPS-2025
(poster) — same domain, same Butler-Volmer kinetics, four failure modes that map
onto §1–§3. They are **strong-form**; the entire variational apparatus here
(§2.1, §2.3, §2.4) is outside their scope. They report NTK weighting working;
§3.2 reports it failing — a direct, reportable disagreement. Their headline
(one FEM data point: 2412% → 0.32% error) is the anchor experiment done properly
and should be cited when framing §6.2.
