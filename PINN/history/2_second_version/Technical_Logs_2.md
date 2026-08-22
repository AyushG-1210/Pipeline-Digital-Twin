# AmorFlux — Technical Audit Log & Paper Source Material (Update v3.2)
**Purpose:** A chronological record of failures, root causes, and what has actually been verified vs. merely proposed. This serves as the primary source record for the paper's Diagnostic / Failure Analysis section.

**Status legend:**
- ✅ **VERIFIED** — implemented AND confirmed with a run/audit/validation check
- 🔶 **PROPOSED** — discussed and designed, but not yet implemented or run
- 🔬 **PENDING RESEARCH** — requires literature/domain input before it can be implemented
- ⚠️ **LIVE BUG / PARTIALLY RESOLVED** — mechanism understood, code-level gap or unverified fix

---

## 1. MIONet Tensor Factorization & Memory Structure
- **Symptom:** VRAM out-of-memory (OOM) errors on 40GB A100s at batch sizes as low as 16.
- **Root cause:** Branch tensors were physically expanded to match the 3D grid before evaluation, forcing millions of redundant autograd graph allocations.
- **Status: ✅ VERIFIED.** The Branch pass evaluates $[B]$ environments without spatial expansion, eliminating the memory bottleneck. *Nuance Correction:* The Trunk pass evaluates $[B, N, 3]$ coordinate inputs; because Trunk passes are lightweight MLPs without environmental parameters, this maintains high throughput (~0.015s/epoch) without OOMs.

## 2. Activation Saturation & Scaled Dot-Product
- **Symptom:** Bounded outputs ($C \in [0, 1]$, $\phi \in [-2, 0]$) with zero gradients, halting learning.
- **Root cause:** Hard `sigmoid`/`tanh` activations caused saturation dead-zones.
- **Status: ✅ VERIFIED.** Outputs converted to raw linear values with soft physical bounds (`loss_bounds`). Scaled Dot-Product (`/ sqrt(128)`) prevents initial logit explosion.

## 3. "Ghost Physics" & Test Function Mode Omission ($m=0$)
- **Symptom:** Mass transport and potential residuals dropped to near-zero while predicting an uncoupled, flat potential field ($\phi_{\text{wall}} \approx +0.277\text{ V}$ vs FDM $-0.0019\text{ V}$).
- **Root cause:** Test functions used `range(1, num_modes_x + 1)` ($m \in \{1, 2, 3, 4\}$)[cite: 2]. Because $\int_0^1 \cos(m \pi x) dx = 0$ for $m \ge 1$, the test space was mathematically orthogonal to $x$-uniform (DC) errors. The weak-form residual was blind to 1D mean field errors.
- **Status: ⚠️ LIVE BUG / IN PROGRESS.** Identified as an active code gap in `new_2.md` line 144[cite: 2]. Corrected to `range(0, num_modes_x + 1)` to include $m=0$ ($\cos(0)=1$).

## 4. Overpotential ($\eta$) Sign Convention
- **Symptom:** 40x magnitude discrepancy in wall potential vs FDM.
- **Root cause:** PINN used $\eta = \phi - E_{\text{eq}}$, while FDM used $\eta = -\phi - E_{\text{eq}}$.
- **Status: ✅ VERIFIED.** PINN aligned to $\eta = -\phi - E_{\text{eq}}$ in loss engine.

## 5. L-BFGS Stationarity & Float32 Precision
- **Symptom:** Premature L-BFGS exit citing "precision loss".
- **Root cause:** Non-stationary loss landscape from stochastic time sampling; tolerance below float32 machine epsilon ($1.19 \times 10^{-7}$).
- **Status: ✅ VERIFIED.** `torch.manual_seed(42)` inside closure; `tolerance_change` set to `1e-6`. Confirmed 250 closure evals completed.

## 6. Dynamic Charge Coupling ($\gamma$) Placeholder
- **Symptom:** Conductivity $\sigma$ relies on global scalar `soil_resistivity_ohm_cm`[cite: 2, 3].
- **Root cause:** `x_soil` (4D GRF) lacks mapped physical semantics.
- **Status: 🔬 PENDING RESEARCH (Option A).** Explicitly retained as a single representative $1000\,\Omega\cdot\text{cm}$ soil placeholder until NACE pipeline datasets are integrated.

## 7. Weak-Form Loss Illusion vs. Spectral Resolution (FDM Ground-Truth Audit)
- **Symptom:** Low weak loss (`res_C: 0.049`, `res_phi: 1.3e-6`) despite high relative error (**$C$ $L_2$ error = 160.6%**, **$\phi$ $L_2$ error = 13,817%**).
- **Root cause:** Dual failure mode: (1) $m=0$ mode omission (Item 3), and (2) low-order global Fourier modes ($n \le 4$) lacking spectral resolution to penalize errors inside the ultra-thin $\sim 10^{-6}$ boundary layer.
- **Status: ⚠️ PARTIALLY RESOLVED.** $m=0$ bug fix deployed; re-evaluation required to isolate boundary-layer spectral resolution error from the $m=0$ blind spot.

---

## Summary Matrix

| Component | Symptom | Status | Next Concrete Action |
|---|---|---|---|
| MIONet Tensor Factorization | OOM errors, 7-min epochs | ✅ Verified | Proceed with training runs |
| Output Activations | Saturated dead-zones | ✅ Verified | Scaled Dot-Product verified |
| Test Functions ($m=0$ Mode) | Weak loss blind to DC error | ⚠️ Live Bug Fix | Re-run with `range(0, ...)` |
| Overpotential Sign ($\eta$) | Flipped kinetics sign | ✅ Verified | Aligned to FDM ground truth |
| L-BFGS Precision | Premature termination | ✅ Verified | Seed locked, 250 steps run |
| Charge Coupling ($\gamma$) | Global scalar placeholder | 🔬 Pending Research | Await real soil pipeline data |
| Test-Function Resolution | 160% / 13,817% $L_2$ error | ⚠️ Unisolated | Evaluate re-run after $m=0$ fix |