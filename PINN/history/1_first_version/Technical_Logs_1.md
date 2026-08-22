# AmorFlux — Technical Audit Log & Paper Source Material
**Purpose:** chronological record of failures, root causes, and what has actually been
verified vs. merely proposed. This is the primary source record for the paper's
Diagnostic / Failure Analysis section, in place of git history.

**Status legend** (used throughout — be strict about this distinction, it matters for
what you can honestly claim in a paper):
- ✅ **VERIFIED** — implemented AND confirmed with a run/audit/validation check
- 🔶 **PROPOSED** — discussed and designed, but not yet implemented or run
- 🔬 **PENDING RESEARCH** — requires literature/domain input before it can be implemented
- ⚠️ **PARTIALLY RESOLVED** — mechanism understood, fix incomplete or unverified

---

## 1. Boundary Layer Sampling Blind Spot

- **Symptom:** Low aggregate training loss, but every audited checkpoint under-predicted
  the true boundary reactive flux (PINN: `dC/dy ≈ -0.037` to `-0.38` across runs; FDM
  ground truth: `dC/dy ≈ -1.000`, at WT=0.5in).
- **Root cause:** At WT=0.5in, `Fo ≈ 196` (Fourier number is thickness-dependent — this
  is not a fixed constant across the trained WT range of 0.25–1.0in, it should be
  recomputed per-thickness for the paper). This drives an ultra-thin reactive boundary
  layer measured at `~2.7×10⁻⁶` in dimensionless y (order-of-magnitude estimate from the
  finest resolved FDM mesh; not yet confirmed mesh-converged to more than 1 sig fig).
- **Why it failed:** Uniform `torch.rand` sampling has probability `~2.7×10⁻⁶` of
  landing inside this layer per draw. Over ~4.5 million total interior samples across
  training, the expected number of samples landing inside the true layer is on the
  order of 10.
- **Status: 🔶 PROPOSED, NOT YET IMPLEMENTED.** A blended sampler (mixing uniform
  samples with `tanh(k·ξ)`-stretched samples, k=6.0) was designed and numerically
  verified to increase in-layer sampling probability by ~6,000× *for a fully-stretched
  (100%) sampler*. A blend ratio (discussed: ~70/30 uniform/stretched) was proposed to
  avoid halving bulk-domain coverage, but **no blended sampler code has been written or
  trained**. Next action: implement, run a short (~200-500 epoch) validation pass, check
  loss stability before committing to a full training run.

---

## 2. The Poisoned Gradient Annealing (EMA) Bug

- **Symptom:** By epoch 200 of an adaptive-weighting run, `w_steel` had grown to
  `~74,753×` while `w_bulk` sat at `~5.7×` — the auto-tuner assigned the *smallest*
  weight to the constraint the physics needed most.
- **Root cause:** At epoch 1 (random init), `loss_pde`'s second-order derivatives
  (Fick + Laplace hessians) are chaotic and enormous on an untrained network, while
  first-order boundary-loss gradients are comparatively small. The measured
  `grad_pde/grad_x` ratio on epoch 1 reflects random-initialization noise, not physical
  priority — and the slow EMA (`α=0.1`) baked this single spurious measurement into the
  weighting scheme for hundreds of subsequent epochs.
- **Additional confirmed measurement bug:** `get_final_layer_grad_mean` averaged
  gradients across the full 256-dim shared output layer, even for losses (like
  `loss_ic`) that only touch the 128-dim concentration sub-block — diluting that
  loss's measured gradient by roughly 2× regardless of the EMA issue above.
- **Status: 🔶 PROPOSED, NOT YET RE-RUN.** Dynamic gradient annealing was abandoned.
  Static weights (discussed target: `W_PDE=1.0, W_STEEL≈5.0, W_BULK=1.0`, exact values
  not finalized) were proposed as the replacement, but **no training run has been
  executed with corrected static weights since this bug was found** — the project
  pivoted to FDM ground-truth validation instead. Next action: pick concrete static
  weights, run, and audit before this can be called resolved.

---

## 3. The Optimizer Shortcut & Domain Explosion

- **Symptom:** Under the poisoned EMA weights above, concentration at y=0 blew past its
  Dirichlet target, climbing to `12.14 → 17.50` across the audited x/t range, while
  potential collapsed to `-17V → -31V`.
- **Root cause:** The extreme (poisoned) `w_steel` weight rewarded any trick that
  minimized the Butler-Volmer residual over correctly resolving transport physics.
  Driving `phi` very negative sends the exponential overpotential term toward zero,
  killing the reaction-rate term in `loss_steel` regardless of `C`'s value — while
  `loss_bulk`, carrying the smallest weight, was essentially undefended, and
  `loss_ceiling` (weight=10) was not strong enough to stop `C` from drifting upward in
  compensation.
- **Status: ✅ VERIFIED (as a diagnosis) / ⚠️ PARTIALLY RESOLVED (as a fix).** The
  mechanism is confirmed directly from the logged weight/loss trajectory. The fix
  (static weights + corrected sampling) is the same one described in items 1 and 2 —
  i.e., still proposed, not yet re-run end-to-end.

---

## 4. The Unconstrained Potential (phi) Field

- **Symptom:** Every PINN audit showed `phi` varying with both `x` and `t`, despite
  nothing in the loss function that should drive time-dependence.
- **Root cause:** `loss_steel` (Butler-Volmer) only produces a residual on `C`; there is
  no residual anywhere constraining `phi` or `dphi/dy` at y=1, x=0, or x=1. The only
  direct constraint on `phi` is `loss_bulk_phi` (phi=0 at y=0) plus the interior Laplace
  penalty.
- **Why it failed:** A harmonic field (`∇²phi=0`) with Dirichlet=0 on one boundary and
  zero-flux (Neumann) on the remaining three has exactly one solution: the constant
  field `phi≡0`, confirmed both analytically (max principle) and numerically (FDM
  solve). Any x/t-structure the PINN's `phi` shows is not required by, and is not
  explained by, the governing equations as currently specified.
- **Status: 🔬 PENDING RESEARCH.** Requires deriving the real charge-conservation
  boundary condition (`-σ·dphi/dy ∝ reaction rate`, with a literature- or
  standards-sourced conductivity `σ`) before it can be added to `butler_volmer_bc`
  and the FDM solver.

---

## 5. The Artificial Reaction-Rate Ceiling (`max=50.0`)

- **Symptom:** FDM ground truth (using the PINN's exact clamped formula) showed `C`
  collapsing to `~1.5×10⁻⁶` within a boundary layer `~2.7×10⁻⁶` wide.
- **Root cause:** `butler_volmer_bc` hard-clamps `Da·C·exp(exponent)` at `50.0`. This
  clamp value was chosen during training-stability debugging to keep this term within
  roughly the same order of magnitude as the other loss terms — **it was never derived
  from or checked against any physical reference.**
- **Precise mechanism (more specific than "forced collapse"):** at WT=0.5in with
  `phi≈0` baseline, `Da≈127` and `exp(η)≈5,218`, giving an unclamped rate of
  `~662,686·C`. This exceeds `50.0` for any `C` above `~7.5×10⁻⁵`. So for nearly the
  *entire* concentration range (`C` from 1 down to `7.5×10⁻⁵`), the boundary condition
  is actually a constant, artificial flux `dC/dy = -50` — not the designed exponential
  Butler-Volmer relationship. Only in the final, razor-thin sliver where `C<7.5×10⁻⁵`
  does the true exponential kinetics take over.
- **Open question, not yet resolved either way:** it is not yet established whether an
  *unclamped*, physically-derived reaction rate would produce a thicker or thinner
  boundary layer than what's currently measured — the unclamped coefficient (~662,686)
  is itself enormous, so a thin layer may be the genuinely correct physical answer
  *if* `k_rate=1e-5` and the other kinetic constants are realistic. That has not been
  checked against literature values either.
- **Status: 🔬 PENDING RESEARCH.** Requires a literature-sourced mass-transport-limited
  saturation function (or a validated `k_rate`) to replace the arbitrary clamp — and,
  separately, independent validation that `k_rate`, `alpha`, `F/RT` are realistic for
  this system before treating any resulting profile as physically meaningful.

---

## 6. The Bulk Shape Reassessment (with a self-correction preserved)

- **Symptom:** An early checkpoint with a sharp, 27× boundary-layer curvature was
  treated as the "good" reference; later, flatter checkpoints were initially assumed
  to be optimization failures.
- **Root cause:** `Fo≈196` (at WT=0.5in) means diffusion equilibrates almost instantly
  relative to the 1-year time domain, so the true steady profile is nearly linear
  across the bulk of the domain.
- **First-pass conclusion (later corrected — keep this in the paper as shown, it's a
  real methodological point):** initial framing was "the flatter checkpoints were
  mathematically correct all along." This overstated the finding.
- **Corrected, verified finding:** the FDM ground truth confirms the *bulk shape* is
  indeed close to linear (✅ verified) — but a direct flux comparison showed **every
  PINN checkpoint audited so far, flat or curved, under-predicted the true wall flux
  (`-1.000`) by a factor of roughly 3× to 75×.** Getting the bulk shape approximately
  right did not mean the physically important quantity (wall flux / corrosion current
  density) was resolved correctly by any checkpoint. This distinction — coarse shape
  vs. the specific quantity the model exists to predict — is worth keeping as an
  explicit point in the paper, not smoothing over.
- **Status: ✅ VERIFIED** (bulk shape claim and the flux under-prediction finding are
  both directly measured, not inferred).

---

## Summary Matrix (corrected)

| Component | Symptom | Status | Next concrete action |
|---|---|---|---|
| Collocation sampler | Uniform sampling misses the `~10⁻⁶`-scale reactive layer | 🔶 Proposed (blend designed, not run) | Implement blended sampler, short validation run |
| Loss weighting | EMA poisoned at epoch 1 by random-init noise | 🔶 Proposed (reverted conceptually, not re-run) | Pick static weights, run, re-audit |
| Optimizer shortcut / domain explosion | Confirmed consequence of the above | ⚠️ Diagnosed, fix pending on the two items above | — |
| Potential (phi) boundary condition | System under-determined; phi≡0 is the only self-consistent solution under current BCs | 🔬 Pending research | Derive charge-conservation BC with real σ |
| Butler-Volmer reaction-rate clamp | Arbitrary `50.0` cap distorts kinetics over ~entire C range | 🔬 Pending research | Literature-sourced saturation function / validate k_rate |
| FDM numerical ground truth | Previously none; relied on subjective loss-log reading | ✅ Verified as a *numerical method* (validated against analytic Da=0 case, though early-time error still ~6% at current Nt) / ⚠️ NOT yet physically complete — inherits the same open gaps as items 4 & 5 | Refine Nt for early-time accuracy; do not treat as final ground truth until phi-BC and reaction-rate items are resolved |

---

## Open research items (blocking full resolution)

1. Real charge-conservation boundary condition for `phi` at y=1 (needs conductivity σ)
2. Literature-validated saturation/mass-transport-limited reaction rate, replacing the
   arbitrary `50.0` clamp
3. Independent check that `k_rate`, `alpha`, and related kinetic constants are
   realistic for this specific corrosion system (currently unvalidated placeholders)
4. Once 1–3 are resolved: rebuild the FDM solver with the corrected (coupled,
   two-variable) boundary system, re-derive the Newton Jacobian accordingly
5. Re-run PINN training with: corrected static loss weights, blended collocation
   sampler, and (once available) the corrected physical boundary conditions
6. Only then compute the final, citable relative L2 error against the corrected FDM
   baseline — this is the number the paper's results section should report

## Suggested mapping to paper sections

- **Methods:** MIONet architecture, FDM baseline derivation (stencils, Newton scheme,
  analytic validation) — items verified in this log
- **Failure Analysis / Diagnostics section:** items 1–3 and 6 (sampling blind spot, EMA
  poisoning, optimizer shortcuts, bulk-shape vs. flux distinction) — this is the paper's
  actual novel contribution: a catalogued set of PINN failure modes specific to
  extreme-scale-separation electrochemical boundary layers, each with a verified
  root cause rather than a guessed one
- **Limitations / Future Work:** items 4 and 5 (phi BC, reaction-rate clamp) — frame
  honestly as identified-but-unresolved, not as solved
