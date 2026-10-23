# TODO resolution: modes-5 statistics, Figures 1–3, and the unreconciled gradient number

Everything here is derived only from `PINN/main.ipynb` (code) and `PINN/electrochem/results/` (records and
checkpoints), plus the two read-only inputs Figure 1 cannot do without (`data/dataset_v2.pt`,
`data/fdm_hold_950_1000.npz`). Reproduce with, from `PINN/`:

```
python electrochem/results/figures/pull_m5_stats.py     # -> m5_vs_m2_comparison.json + tables below
python electrochem/results/figures/make_figures.py      # -> fig1/fig2/fig3 .png/.pdf + fig_data.json
```

---

## 1. Modes-5 medians and ranges (n = 5, seeds 42–46)

Sources: `results/json_files/weak_form/s4{2..6}a.json` (base records; `adam` = 200+600 epochs at
bs 64 = 8,800 steps, `lbfgs` = +250 L-BFGS steps) and `m5_s4{2..6}a_lbfgs100.json` (repolished:
+100 L-BFGS steps from the *same* Adam checkpoint `m5_s4Xa_adam.pt`). Naming trap confirmed:
the JSON is `s42a.json`, the checkpoint is `m5_s42a_adam.pt`, the repolish record is
`m5_s42a_lbfgs100.json`.

Held-out, median [min – max]:

| metric | Adam (8,800) | + L-BFGS 250 (base) | + L-BFGS 100 (repolished) |
|---|---|---|---|
| C rel-L2 | 0.219 [0.115 – 0.229] | 0.334 [0.274 – 0.345] | 0.233 [0.122 – 0.237] |
| φ rel-L2 | 0.293 [0.291 – 0.301] | 0.379 [0.361 – 0.406] | 0.360 [0.355 – 0.368] |
| flux_err | 1.148 [0.284 – 1.231] | 1.833 [1.410 – 1.881] | 1.250 [0.366 – 1.293] |
| op_corr | 0.965 [0.960 – 0.967] | 0.925 [0.904 – 0.939] | 0.953 [0.942 – 0.958] |
| x_corr | 0.237 [0.128 – 0.314] | 0.118 [0.073 – 0.168] | 0.211 [0.157 – 0.277] |
| x_var_ratio | 0.782 [0.745 – 0.931] | 0.819 [0.756 – 1.036] | 0.796 [0.772 – 0.830] |
| weak_total | 2.22e-2 [1.15e-2 – 3.60e-2] | 2.48e-2 [2.25e-2 – 2.54e-2] | 2.46e-2 [2.35e-2 – 2.54e-2] |
| anchor share (%) | 52.3 [45.0 – 73.0] | 54.7 [50.3 – 60.7] | 69.0 [60.8 – 77.6] |

Train-split, per-seed values, and the `res_C / res_phi / bounds / smooth` components are in
`m5_vs_m2_comparison.json` and in the script's stdout. **C rel-L2 is skewed, not spread**: seed 42
sits at 0.115 while the other four cluster at 0.201–0.229 (median of those four alone: 0.220);
median-and-range (0.219 [0.115–0.229]) reads as a symmetric spread but is one low outlier pulling
a tight four-seed cluster down. The same seed drives the flux_err range (0.284 vs. 1.03–1.23 for
the rest). Do not read the Table-1 modes-5 median of 0.219 against the modes-2 (matched-budget)
median of 0.174 in Figure 2 as a clean 0.045 gap — the four-of-five modes-5 number (0.220) is the
more representative one, and against that the gap is 0.046, essentially the same conclusion but
not resting on an outlier seed.

LaTeX-ready:

```latex
\begin{tabular}{lccc}
\toprule
metric & Adam (8{,}800 steps) & +L-BFGS 250 (base) & +L-BFGS 100 (repolished) \\
\midrule
$C$ rel.\ $L^2$   & 0.219 [0.115--0.229] & 0.334 [0.274--0.345] & 0.233 [0.122--0.237] \\
$\phi$ rel.\ $L^2$& 0.293 [0.291--0.301] & 0.379 [0.361--0.406] & 0.360 [0.355--0.368] \\
flux err          & 1.148 [0.284--1.231] & 1.833 [1.410--1.881] & 1.250 [0.366--1.293] \\
op.\ corr         & 0.965 [0.960--0.967] & 0.925 [0.904--0.939] & 0.953 [0.942--0.958] \\
$x$ corr          & 0.237 [0.128--0.314] & 0.118 [0.073--0.168] & 0.211 [0.157--0.277] \\
$x$ var.\ ratio   & 0.782 [0.745--0.931] & 0.819 [0.756--1.036] & 0.796 [0.772--0.830] \\
weak total        & $2.22\times10^{-2}$ & $2.48\times10^{-2}$ & $2.46\times10^{-2}$ \\
\bottomrule
\end{tabular}
```

### Does the modes-2 result fall inside the modes-5 seed spread?

The modes-2 record used here is `m2_bs64_full_s4{2..6}.json` (n = 5, modes = 2, 300+900 epochs =
**13,200** Adam steps, +100 L-BFGS). `m2_s42a.json` — a single seed-42 run at the same config —
is **not** used: its Adam checkpoint is byte-identical to `m2_bs64_full_s42` (cell 29 copies it),
so it is a redundant single draw from the same 5-seed set already below, and comparing 1 seed
against the modes-5 5-seed spread was throwing away the seed-42–46 comparison this data already
supports, forcing a spurious "seed 42 is the modes-5 arm's best seed" confound into the readout.
Reconciled against the correct n = 5 vs n = 5 comparison, only one confound remains: the Adam
budget differs (13,200 modes-2 vs 8,800 modes-5 steps).

**Adam phase, n = 5 vs n = 5, held-out** (m2 Adam 13,200 vs m5 Adam 8,800):

| metric | modes-2 [min – max] | m5 [min – max] | verdict |
|---|---|---|---|
| C rel-L2 | 0.2858 [0.2485 – 0.2966] | 0.2190 [0.1153 – 0.2289] | DISJOINT → modes-2 **worse** |
| φ rel-L2 | 0.2837 [0.2822 – 0.2876] | 0.2930 [0.2906 – 0.3009] | DISJOINT → modes-2 **better** |
| flux_err | 1.3584 [0.8057 – 1.3992] | 1.1478 [0.2839 – 1.2310] | overlap; m5 median inside m2 range → weak evidence (modes-2 higher) |
| op_corr | 0.9639 [0.9595 – 0.9680] | 0.9647 [0.9597 – 0.9674] | overlap, each median inside the other's range → does not distinguish |
| x_corr | 0.2511 [0.2025 – 0.3489] | 0.2373 [0.1277 – 0.3144] | overlap, each median inside the other's range → does not distinguish |
| x_var_ratio | 0.6780 [0.6384 – 0.7129] | 0.7817 [0.7452 – 0.9312] | DISJOINT → modes-2 lower (flatter along x) |
| weak_total | 1.16e-2 [1.01e-2 – 1.45e-2] | 2.22e-2 [1.15e-2 – 3.60e-2] | overlap; m2 median inside m5 range → weak evidence (modes-2 lower) |

**Matched L-BFGS budget, n = 5 vs n = 5, held-out** (m2 +100 vs m5 repolished +100 — the cleaner
polished pairing):

| metric | modes-2 [min – max] | m5 [min – max] | verdict |
|---|---|---|---|
| C rel-L2 | 0.2963 [0.2777 – 0.3106] | 0.2329 [0.1224 – 0.2369] | DISJOINT → modes-2 **worse** |
| φ rel-L2 | 0.3454 [0.3390 – 0.3620] | 0.3603 [0.3550 – 0.3679] | overlap; m5 median inside m2 range → weak evidence (modes-2 lower) |
| flux_err | 1.4831 [1.0428 – 1.6012] | 1.2502 [0.3661 – 1.2930] | overlap; m5 median inside m2 range → weak evidence (modes-2 higher) |
| op_corr | 0.9432 [0.9261 – 0.9538] | 0.9533 [0.9421 – 0.9580] | overlap, each median inside the other's range → does not distinguish |
| x_corr | 0.1864 [0.1428 – 0.2233] | 0.2107 [0.1572 – 0.2770] | overlap, each median inside the other's range → does not distinguish |
| x_var_ratio | 0.7135 [0.6954 – 0.8101] | 0.7955 [0.7715 – 0.8299] | overlap; m5 median inside m2 range → weak evidence (modes-2 lower) |
| weak_total | 2.90e-2 [2.70e-2 – 3.01e-2] | 2.46e-2 [2.35e-2 – 2.54e-2] | DISJOINT → modes-2 **worse** (higher objective) |

(The third pairing, m2 +100 vs the base m5 +250, is in `m5_vs_m2_comparison.json`; it is not a
like-for-like budget and reverses several verdicts because L-BFGS 250 degrades every seed — the
`paired_lbfgs.json` result, 18/18 comparisons favour 100.)

**What the data supports saying.** In both like-for-like pairings, `flux_err`, `op_corr`, and
`x_corr` put the modes-2 result's seed spread *overlapping* the modes-5 spread (medians each
inside or near the other's range), so none of the three cleanly distinguishes test-space size
here. `C rel-L2` is the one accuracy metric with disjoint ranges in both pairings, and the
direction is that modes-2 is *worse* (0.286 vs ≤ 0.229 Adam; 0.296 vs ≤ 0.237 polished) — note the
modes-5 side of that gap is the seed-42-skewed number flagged above, though even against the
four-seed cluster (median 0.220) the gap holds. `x_var_ratio` is consistently lower for modes-2
(flatter along x). `weak_total` disagrees by phase (modes-2 lower on Adam, higher on the matched
L-BFGS polish) and should not be read as a consistent direction. The one remaining confound — Adam
budget, 13,200 (modes-2) vs 8,800 (modes-5) — runs *against* modes-5 (it had fewer Adam steps), so
the C rel-L2 direction is not an artefact of that budget gap; but the Figure 2 result shows more
Adam steps *hurt* the modes-2 arm's own C rel-L2, which is the other way the confound could cut.
State the confound; this is n = 5 vs n = 5, not n = 1, but the step-count gap is not equalized.

---

## 2. Figures

### Figure 1 — `fig1_cwall_profiles.{png,pdf}`  (Section 6.3, next to Table 2)

Wall concentration $C_{\mathrm{wall}}(x)$ at $t=1$ for four held-out environments (rows 950,
991, 970, 956 — the thinnest, two interior WT quantiles, and the thickest wall). FDM reference
solid with markers on its 21-node grid; the three PINN arms dashed on a 201-point grid. All
three arms are the **seed-42, step-matched** Adam checkpoints of the Table-2 experiment:
`sa_wbc1_noanchor_s42` (strong, $W_{\mathrm{ANCHOR}}=0$, 7,216 steps), `dn_plain_anchored_s42`
(weak, $W_{\mathrm{ANCHOR}}=1$, 7,205 steps), `dn_plain_noanchor_s42` (weak,
$W_{\mathrm{ANCHOR}}=0$, 7,205 steps). Per-panel $r_x$ is the per-environment
$\mathrm{corr}(C_{\mathrm{wall}}^{\mathrm{pred}}, C_{\mathrm{wall}}^{\mathrm{FDM}})$ on the
FDM grid; averaged over all 50 held-out rows these reproduce the recorded `x_corr`
(+0.492 / +0.280 / −0.079) to four decimals (asserted in the script).

What it shows: the unanchored weak arm sits at $C_{\mathrm{wall}}\approx 0.36$–$0.47$ for
every environment while the reference lies at $0.09$–$0.33$, and its along-$x$ shape does not
track the reference (per-panel $r_x$ = −0.83, −0.49, +0.04, 0.00). That is the "non-solution":
a nearly environment-independent, offset field with the wrong $x$-structure. The strong
unanchored arm follows the shape ($r_x$ = +0.63…+0.73) with an amplitude/offset error in the
thin-wall panels; the anchored weak arm recovers the level and most of the shape but is the
least consistent of the two solutions ($r_x$ from +0.78 down to −0.43 on the thickest wall).

Suggested caption: *Wall concentration profiles for four held-out environments spanning the
WT range. Solid: FDM reference. Dashed: strong unanchored, weak anchored, weak unanchored
(seed 42, ~7.2k Adam steps each). The unanchored weak arm converges to an offset,
near-flat field whose along-x shape is uncorrelated or anticorrelated with the reference —
the negative held-out x_corr in Table 2 made visible.*

### Figure 2 — `fig2_objective_vs_accuracy.{png,pdf}`  (Section 6.5)

Three panels sharing the step axis at the two measured budgets, 7,205 (`m2_bs64_matched`)
and 13,200 (`m2_bs64_full`) Adam steps, modes = 2, bs = 64, Adam phase. Left: `weak_total`
(train batch). Middle / right: held-out `flux_err` and `C rel-L2`. Light lines = the five seeds,
heavy line = median. Medians: `weak_total` 2.07e-2 → 1.16e-2 (falls for 5/5 seeds);
`flux_err` 0.541 → 1.358 (rises for 5/5); `C rel-L2` 0.174 → 0.286 (rises for 5/5).

Layout note: the spec asked for two y-axes on one panel; I used three side-by-side panels
instead (dual y-axes let the reader infer a crossing that the scales do not define). The
reading is the same — left falls, right rises — and it is a one-line change to revert.

Suggested caption: *Weak-form objective and held-out accuracy at the two measured Adam
budgets (n = 5 seeds each; light lines pair the same seed, heavy line is the median). The
objective falls for every seed while both accuracy metrics rise for every seed. Two step
counts only — the connecting segments indicate pairing, not a trajectory; nothing was
measured between 7,205 and 13,200 steps.*

### Figure 3 — `fig3_gram_spectrum.{png,pdf}`  (Appendix H)

Sorted eigenvalues of the H¹-seminorm Gram matrix $G_{ij}=\int\nabla v_i\cdot\nabla v_j$ on
the modes-2 test space, rebuilt from the notebook's cell-11 quadrature and cell-51 recipe:
$K=36$, $\lambda_{\min}=0.1033$, $\lambda_{\max}=3770$, $\lambda_{\max}/\lambda_{\min}=36{,}503$
(matches the recorded 3.65e4 / 0.103 / 3770 in `all_dualnorm.json`). The spectrum climbs
in visible triplets — the three $x$-modes ($m=0,1,2$) sharing each hat function — with the
$d$-direction hat width setting the ~10⁴ spread. Full eigenvalue list in `fig_data.json`.

Suggested caption: *Eigenvalue spectrum of the test-space Gram matrix (log scale). The plain
mean-of-squares reduction weights residual directions by these eigenvalues; the
$3.65\times10^{4}$ span is what the dual norm $R^{\mathsf T}G^{-1}R$ removes.*

### Fourth candidate
`diagnostics/xai_compare.png` exists and needs no new compute; it was outside the scope of
this pass and is unchanged.

---

## 3. The number that does not reconcile (Section 8) — resolved: drop it

Searched `main.ipynb` (all 70 cells, code and comments) and every JSON under `results/` for
each candidate figure: `3.75e9`, `0.0475`, `3.1258e12`, `4.5935e-4`, `1.7e19`, and the implied
`7e15`. **None of them appears anywhere.** No run in `results/` stores per-term gradient
traces at all; the only gradient-decomposition quantity that is recorded is the
gradient-*norm share* of the anchor vs. the two residual terms (`share.ANCHOR`, `share.weak`
in `s4Xa.json`, and `compiled_results/anchor_share_m2.json`).

What the notebook does contain, as code comments (cell 15 and cell 17), is qualitative and
mutually consistent:

- "Without this [/γ normalisation], res_phi's trace sits ~1e9x below res_C's for purely
  dimensional reasons" — an order-of-magnitude dimensional estimate, not a measurement.
- "A later trace at a trained point gave res_phi 92.9% share vs res_C 7.0% — it flipped which
  term owns the gradient" — a share, with no raw values recorded.
- "after /gamma normalization the two residual traces sit within ~20x."
- NTK-weighting history: "res_phi hit 3.5e5 while its own loss was at 1e-8".

Conclusion, per the rule in the TODO: since no candidate can be traced to a file, the
qualitative statement in Section 5 stands on its own and no specific raw ratio should be
restored.

**Reconciling the anchor-share number.** Four independent evaluations of the same quantity
(same arm `m2_bs64_matched`, same weights, same `W_ANCHOR=1.0`, same denominator — sum of the
8 per-term gradient norms — confirmed identical by printing all of them, see
`anchor_share_recompute.py`/`.json`) disagreed on the median: `anchor_share_m2.json` says
50.9 % [44.5–62.6]; `m2_bs64_matched_s4X.json['adam']['share']` says 57.7 % [50.3–76.2]; the
paper text says 56 % [47–73]. **This is not a weight, denominator, or code discrepancy — it is
draw-count noise.** All three stored numbers use only `draws=5`: each seed's share is the
median of 5 stochastic draws (random `t`, penalty points, anchor points), so a 5-draw median
has real run-to-run scatter, and `anchor_share_m2.json`'s docstring claim of "50 draws" is
simply wrong — the file's own `draws` field says 5. Re-running at `draws=50` (low-variance
reference, `anchor_share_recompute.py --draws 50`) gives ANCHOR median **53.7 % [45.5–68.3]**
and weak (res_C+res_phi) median **13.4 % [7.6–17.2]** — inside all three stored ranges,
including the paper's [47–73] / [7–16]. A `draws=5` rerun here landed at 54.5 % [45.5–70.3] /
15.0 % [6.5–19.7], also consistent. **Use the draws=50 number (53.7 % / 13.4 %) as the cited
figure** wherever this appears (verdict table, gradient-imbalance section) — it is the
low-noise estimate, and it does not change the qualitative claim (anchor term dominates the
gradient budget at roughly half-to-two-thirds share). Full per-term norm dictionaries for
every seed at both draw counts are in `anchor_share_recompute.json`.

On the modes-5 Adam checkpoints (table above): 52.3 % / 13.9 %, `draws=5` per seed as recorded
— not separately re-run at higher draw count here.

---

## Side note found while doing this

The committed `data/dataset_v2.pt` carries the older 6-field fingerprint
`(1234, 1000, 8, 50, 0.25, 0.3)`, while cell 3 now expects the 11-field `DATA_FINGERPRINT`.
Cell 3 will therefore log "cache is stale" and regenerate on its next run. That is harmless —
same seed, and I verified the regenerated tensors reproduce the FDM files' stored `rho(x)` and
`c_bulk(x)` to float32 rounding (max rel. diff 9e-8) — but it will rewrite the tracked file.
(I triggered this once while exploring and restored the file from git; the figure script reads
the tensors directly and does not execute cell 3.)
