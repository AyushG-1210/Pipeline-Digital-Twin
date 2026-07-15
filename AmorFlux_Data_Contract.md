# AmorFlux — Data Contract for Branch Tensor Delivery
**From:** Ayush (network owner)
**To:** Swaraag, Aditya
**Purpose:** exact shapes, dtypes, and semantics required so your outputs can be loaded directly into `MIONet.forward()` and `interior_pde`/`butler_volmer_bc` with zero glue code.

---

## 0. Rules that apply to both of you (read first)

These aren't style preferences — every one of them maps to a bug we already hit and fixed while building the network.

1. **dtype: `torch.float32` everywhere.** No float64, no int tensors passed as features. Cast explicitly before handoff.
2. **No NaNs, ever.** Swaraag's Phase-4 zero-fill imputation on the Neo4j export is exactly right — replicate that pattern for any missing value in your pipeline, and **document what you filled and why** (zero, mean, forward-fill) so it's auditable later.
3. **One row = one physical instance, and every branch must agree on what that instance is.** Right now the network expects: row `i` of the soil tensor, row `i` of the fluid tensor, and row `i` of the metadata tensor all describe *the same physical thing* (currently: one pipe segment / inspection instance). If your data has a different natural grain (per-incident vs per-segment vs per-sensor-reading), say so now — don't silently pick one.
4. **Carry a join key.** Every row you deliver needs a stable ID column (e.g. `segment_id`) alongside the tensor — not baked into the tensor itself, but delivered next to it (see §4, delivery format). This is non-negotiable: we've already had multiple bugs in this project caused by two related tensors silently drifting out of row-alignment after an independent reshuffle. A join key is how we catch that before it becomes a silent bug instead of a loud error.
5. **Deliver raw AND normalized values for any physical quantity that goes through `PHYSICS_PARAMS`.** The network's physics equations (Butler-Volmer, Fick's law) need real units (inches, meters, volts). The branch network needs normalized `[0,1]` values. Give me both, same column order, in the same file — not two files I have to hope stay in sync.
6. **Enforce range bounds before handoff, don't just report them.** If a value falls outside the physical range the model is trained on, the model will extrapolate silently and confidently wrong (PINNs don't know when they're out of distribution). Clip or flag out-of-range rows rather than passing them through.
7. **Bundle everything as one artifact per delivery**, not loose CSVs — see exact format in §4.

---

## 1. Swaraag — Neo4j / GEE / GIS pipeline

### 1a. Branch 3 (Metadata) — Wall Thickness & Outer Diameter

This is the most well-defined piece of your job — it maps directly onto what you already have from T-4 permits / RRC GIS data.

| Field | Shape | Dtype | Units | Valid range | Notes |
|---|---|---|---|---|---|
| `wt_raw` | `[N, 1]` | float32 | inches | `[0.25, 1.0]` *(confirm against Ayush's API-5L standards docs — this range was a placeholder)* | real wall thickness, unnormalized |
| `wt_norm` | `[N, 1]` | float32 | dimensionless | `[0, 1]` | `(wt_raw - wt_min) / (wt_max - wt_min)` using the **same** `wt_min`/`wt_max` as above — don't compute this with a different min/max than what ships with the model config |
| `od_raw` | `[N, 1]` | float32 | inches | `[4.0, 48.0]` *(same — confirm against real API-5L data)* | |
| `od_norm` | `[N, 1]` | float32 | dimensionless | `[0, 1]` | same normalization rule as `wt_norm` |

**Where this comes from:** T-4 permit data / pipe standards docs should give nominal WT/OD per segment. If a segment has a range (e.g. wall thickness varies along its length) rather than a single value, tell me — right now the model assumes one scalar WT per row.

### 1b. Branch 1 (Soil) — ⚠️ blocked on architecture decision

Your Phase 4 GEE/Neo4j export (`input_tensor.npy`, shape `(5831, 4)`) does **not** match what the current model expects (`[Batch, 50]`, a spatial *profile*, not 4 scalar features). Don't build more of this pipeline until this is resolved:

- **If we go with your 4-feature format (recommended):** tell me exactly what the 4 columns are (you mentioned soil moisture + surface temperature — what are the other two?), their units, and their valid ranges. I'll shrink `branch_soil`'s input dimension from 50 to 4 and retrain. This keeps your pipeline as real, physically-grounded data instead of GRF-smoothed synthetic filler.
- **If we keep `[Batch, 50]`:** this means sampling your environmental variable(s) at 50 fixed normalized positions along each pipe segment's length (a true spatial profile per segment), not one point per incident node. This is a materially bigger extraction job — confirm with me before starting.

**My default assumption, pending your confirmation:** go with your existing 4-feature format and I'll adjust the model. Let me know if that's wrong.

### 1c. Branch 2 (Fluid) — currently unowned

Nothing in five phases of logs assigns internal pressure / fluid flow profile data to anyone. This branch is 100% synthetic right now. Two questions for you (since you already own the GIS/permit data layer, this may be closest to your existing pipeline):

- Does the RRC GIS/permit data include operating pressure or flow data per segment, or does that require a different data source (SCADA feed, operator-reported data)?
- If it's genuinely out of reach for this phase, say so explicitly — I'd rather know it's synthetic-only for now than assume it's real when it isn't.

---

## 2. Aditya — ColPali/GraphRAG + YOLOv8

### 2a. Branch 3 (Metadata) — Defect Mask Score

This is your clean, well-scoped deliverable — matches the README's Branch 3 diagram exactly.

| Field | Shape | Dtype | Units | Valid range | Notes |
|---|---|---|---|---|---|
| `defect_score` | `[N, 1]` | float32 | dimensionless | `[0.0, 1.0]` | ratio of corrupted pixels to total structural surface area, per the README's definition (0 = flawless, 1 = complete breach) |

**Requirements:**
- One `defect_score` per physical instance (same join key / grain as Swaraag's `segment_id`), not per raw image. If a segment has multiple inspection images/frames, decide and document your aggregation rule (max severity? mean? most recent inspection?) — don't leave it implicit.
- Segments with **no detected defect** should get `defect_score = 0.0` explicitly, not a missing/null row. A missing row breaks the join; an explicit zero doesn't.
- No `defect_score` needs a "raw" counterpart — it's already a normalized ratio by construction, unlike WT/OD.

### 2b. GraphRAG / ColPali — clarify scope, this does NOT feed a tensor directly

Based on the logs, the RAG pipeline's job is answering questions against standards docs (NACE, API-5L, PHMSA) — it's a knowledge/QA layer, not a tensor source. **Confirm this is still the intent.** If instead you want RAG-extracted values (e.g., a standard's specified `k_rate` bound, or a regulatory pressure limit) to feed into `PHYSICS_PARAMS` or a validation/range-check step, that's a different, much more precise task: extracting a specific structured field from a specific document with a citation, not open-ended retrieval. Tell me if that's actually needed and I'll spec it separately — don't let it stay ambiguous, since "RAG output" as a tensor input is not something the current architecture has any place to receive.

---

## 3. Delivery format (both of you)

One file per delivery, not scattered CSVs/npy files. Recommended:

```python
torch.save({
    "segment_id":  segment_id_list,      # list[str] or [N] tensor, the join key — same order as every array below
    "soil":        soil_tensor,          # [N, 50] or [N, 4] — pending §1b decision
    "fluid":       fluid_tensor,         # [N, ?] — pending §1c
    "wt_raw":      wt_raw,               # [N, 1]
    "wt_norm":     wt_norm,              # [N, 1]
    "od_raw":      od_raw,               # [N, 1]
    "od_norm":     od_norm,              # [N, 1]
    "defect_score": defect_score,        # [N, 1]
    "meta_source_notes": {...}           # free text: imputation rules, aggregation rules, date range, etc.
}, "amorflux_branch_data_v1.pt")
```

If you can't merge into one file on your end, at minimum guarantee row order is identical across files and include `segment_id` in every one of them so misalignment is a loud KeyError, not a silent wrong number.

---

## 4. Open questions blocking finalization

1. Swaraag: what are the actual 4 columns in your GEE export, their units, and valid ranges?
2. Swaraag/Ayush: confirm real WT/OD ranges from the API-5L standards docs — current `[0.25,1.0]"` / `[4,48]"` are placeholders from synthetic data generation.
3. Swaraag: does RRC/permit data include real pressure/flow data for Branch 2, or is that out of scope this phase?
4. Aditya: confirm RAG's role is QA-only, not a tensor source — or specify what structured field(s) it needs to extract if it is.
5. Both: what's the natural grain of your data — per pipe segment, per incident, or something else? Needs to be one consistent answer across both pipelines.
