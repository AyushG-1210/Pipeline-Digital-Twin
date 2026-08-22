# AmorFlux: Multi-Branch Physics-Informed Neural Operators with GraphRAG for Amortized Pipeline Corrosion Prognostics

> Ayush Gouda, Aditya Prakash, Swaraag Hebbar N  
> Department of Computer Science & Engineering, RV Institute of Technology and Management

---

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-5.19-008CC1?style=flat-square&logo=neo4j&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

**AmorFlux** is a multi-modal, physics-informed Scientific Machine Learning (SciML) framework designed to simulate real-time localized electrochemical corrosion dynamics and predict the Remaining Useful Life (RUL) of midstream oil and gas pipelines.

Conventional physics-informed neural networks (PINNs) act as isolated numerical solvers, requiring computationally expensive retraining from scratch every time an environmental profile, pipe geometry, or fluid property changes. **AmorFlux** circumvents this boundary-value bottleneck by utilizing a Multi-Input Neural Operator (**MIONet**) architecture. By training on continuous parameter spaces, the framework functions as an *amortized solver*—executing instantaneous forward-pass inferences across arbitrary pipeline networks without re-optimization.

The neural operator component is the subject of an in-preparation paper on objective–accuracy behaviour in weak-form operator training. Measured results are reported below.

### System Architecture Pipeline

The framework orchestrates three deeply integrated data and deep learning layers:
1. **Multi-Modal GraphRAG Ingestion:** Unstructured compliance documentation from the Texas Railroad Commission (RRC), PHMSA, NACE, and API-5L specifications are ingested via a `ColPali` vision-language retrieval model. Extracted structural attributes and regulatory standards are mapped into a `Neo4j` topological knowledge graph.
2. **Spatially Correlated Parameterization:** Automated connectors hook into Google Earth Engine (GEE) to extract localized soil moisture and surface temperatures. These features are smoothed along the longitudinal pipe coordinate using a 1D Gaussian Random Field (GRF) kernel to feed the model with realistic environmental patches.
3. **Multi-Branch Neural Operator Evaluation:** The parameter tensors and continuous space-time coordinates are split across parallel Branch and Trunk neural networks, continuously constrained by the underlying electrochemical partial differential equations.

---

## Working of Components

### Hybrid Vector-Graph RAG Pipeline
To provide robust contextual reasoning across both structured data (like PHMSA pipeline incident logs) and unstructured text (like API-5L standards), AmorFlux uses a hybrid GraphRAG architecture augmented with live web search capabilities:

1. **Semantic Vector Search (ChromaDB)**
   Unstructured documents are parsed, chunked, and embedded into a persistent local ChromaDB using the `BAAI/bge-large-en` model. When an operator asks a question, this layer identifies the top semantically relevant chunks.

2. **Graph Traversal (Neo4j AuraDB)**
   During document ingestion, a Gemini LLM extracts specialized `Operator`, `Pipeline`, `Incident`, and `Location` entities from the text chunks, pushing them into a Neo4j knowledge graph. When vector chunks are retrieved by ChromaDB, the system fetches 1-hop relationship facts (e.g., *Operator owns Pipeline*) from the graph and seamlessly injects this structured context alongside the raw text.

3. **Web Scraping Fallback**
   For temporal queries (e.g., "latest news", "2026") or highly specific edge-cases where the local vector confidence falls below a strict threshold (L2 Distance > 1.2), the system dynamically routes the query to a DuckDuckGo web search. It scrapes live HTML, filters out boilerplate, and extracts semantic snippets via an ephemeral vector search before returning it to the generation LLM.

---

## Core Mathematical Framework

AmorFlux maps the physical degradation of the pipeline wall boundary layer by continuously solving the mass-transport equations coupled with non-linear electrochemical reaction kinetics.

### 1. Interior Domain Governing Equation (Fick's Second Law)
The migration of aggressive corrosive species (e.g., oxygen, moisture, chlorides) through the soil or electrolyte matrix toward the steel pipe wall over time is governed by a 2D spatial, 1D temporal diffusion equation:

$$\frac{\partial C}{\partial t} = D \left( \frac{\partial^2 C}{\partial x^2} + \frac{\partial^2 C}{\partial y^2} \right)$$

Where $C(x, y, t)$ represents the localized concentration profile of the corrosive species, and $D$ is the global diffusion coefficient parameterized by the soil branch.

### 2. Interface Condition (Butler-Volmer Kinetics)
At the extreme upper edge of the boundary layer corresponding to the steel pipe interface ($y = 1$), mass flux is coupled directly to the rate of chemical consumption via non-linear electrochemical kinetics:

$$\left. D \frac{\partial C}{\partial y} \right|_{y=1} + k_{\text{rate}} C \cdot \exp\left( \frac{\alpha F}{R T} (\phi - E_{\text{eq}}) \right) = 0$$

Where $\phi$ is the localized electrical potential predicted by the network, $E_{\text{eq}}$ is the equilibrium potential, $F$ is Faraday's constant, $R$ is the universal gas constant, and $T$ is the interface temperature profile. The model solves this boundary system to isolate the **corrosion current density ($I_{\text{corr}}$)**, which is integrated over time to generate RUL prognostic arrays.

### 3. Loss Formulations

Two formulations are implemented against these equations, and both are evaluated in the study below.

The **weak form** integrates the residual against hp-refined test functions, obtaining the Butler–Volmer interface condition through Green's identity without an explicit penalty term. The **strong form** enforces the residual pointwise and requires an explicit Neumann penalty carrying a free weight `W_BC`. This structural asymmetry — the weak form gets the interface condition for free, the strong form pays for it with a tunable weight — is central to interpreting the comparison.

### 4. Visual Defect Segmentation & YOLOv8 Vision Pipeline

To anchor the continuous neural operator to real-world structural degradation, the framework integrates a real-time computer vision pipeline engineered to detect and segment localized macro-defects from physical inspection feeds (e.g., drone imagery, robotic crawler videos).

#### Implementation & Setup Details

##### 1. Directory Structure & Setup
Created a self-contained folder structure inside `./YOLO_Pipeline/`:

- `download_dataset.py`: Programmatically downloads the pipeline dataset from Roboflow.
- `train.py`: Fine-tunes the `yolov8n-seg.pt` model with specialized augmentations.
- `inference.py`: Processes video frame-by-frame and exports visual insights.
- `generate_test_video.py`: Helper script generating a dark synthetic test video.
- `venv/`: Isolated Python virtual environment.
- `dataset/`: Contains Roboflow images and annotations (split into `train` and `val`).
- `weights/best.pt`: Best fine-tuned model weights.
- `visual_insights.json`: Structured outputs file.

##### 2. Environment & Dependency Isolation
Created `venv` and successfully upgraded `pip`.
Installed `ultralytics`, `opencv-python`, `numpy`, and `roboflow`.
Installed FFmpeg on the system using `winget install Gyan.FFmpeg`.

##### 3. Roboflow Dataset Ingestion & Alignment
Executed `download_dataset.py` to connect via Roboflow API.
Handled edge case where the project has raw images but no existing dataset version by programmatically generating version 1 on the fly via `generate_version()`.
Renamed `valid/` directory to `val/` to align with folder layout.
Updated `data.yaml` to enforce absolute control:
```yaml
path: ./YOLO_Pipeline/dataset
train: train/images
val: val/images
```

##### 4. Model Fine-Tuning & Custom Augmentations
To generalize against Non-RGB, Thermal, and IR feeds, we injected heavy color-space augmentations inside `train.py`:
- `hsv_v=0.4` (simulate extreme luminance/exposure changes).
- `grayscale=0.5` (50% chance to drop color to simulate thermal grayscale feeds).
- `hsv_s=0.0` (zero saturation variance to match IR/thermal sensors).
- Note: Since grayscale is not natively validated by the Ultralytics configuration parser, we dynamically monkeypatched `ultralytics.cfg.check_dict_alignment` to intercept and allow the parameter to be passed without raising a validation syntax error.
- Trained the model for 1 epoch at `imgsz=160` to optimize for CPU performance, successfully outputting `weights/best.pt`.

##### 5. Standalone Inference Pipeline (`inference.py`)
Developed a production-grade inference script resolving these edge cases:
- Media Fallback: Checks if OpenCV can natively decode the input video. If not, it uses a custom resolver `find_ffmpeg()` to locate the binary inside the WinGet packages directory and runs a subprocess to transcode it to a standard H.264 encoded `.mp4` container.
- Low-Light CLAHE Preprocessing: Converts frames to LAB color space, extracts the L-channel (lightness), applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance defects in dark regions without color distortion, and merges it back.
- JSON Output Contract Preservation: Evaluates and exports the single frame that yielded the absolute highest severity/confidence score into `./YOLO_Pipeline/visual_insights.json`

##### 6. End-to-End Verification Results
- Generated synthetic low-light video: `./YOLO_Pipeline/test_video.mp4`
- Executed (powershell):
  ```powershell
  .\YOLO_Pipeline\venv\Scripts\python.exe ./YOLO_Pipeline/inference.py ./YOLO_Pipeline/test_video.mp4
  ```
- Verified the output schema in `visual_insights.json`:
  ```json
  {
    "frame_id": 0,
    "corrosion_detected": false,
    "severity_score": 0.0,
    "mask_coordinates": []
  }
  ```
- Tested the FFmpeg transcode fallback by inputting a corrupted dummy file:
  - Program successfully detected OpenCV decoding failure.
  - Dynamically resolved FFmpeg binary path.
  - Spawned transcoding subprocess.

The vision layer utilizes a specialized `YOLOv8-seg` instance segmentation architecture pre-trained on high-resolution industrial surface defect datasets. The model tracks five distinct classes of structural anomalies: external pitting, line cracks, coating degradation, structural gouges, and localized anomalies.

    ┌──────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
    │ Raw Drone/Crawler Feed   │ ──► │  YOLOv8-seg Model      │ ──► │ Defect Severity Mask   │
    │ (Surface Inspection)     │     │  (Instance Segmentation│     │ (Pixel Area Compute)   │
    └──────────────────────────┘     └────────────────────────┘     └────────────────────────┘
    │
    ▼
    ┌──────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
    │ Branch 3 Input Tensor    │ ◄── │ Min-Max Normalization  │ ◄── │ Normalized Score [0,1] │
    │ [WT, Resistivity, Defect]│     │(0: Intact, 1: Critical)│     │ (Feature Extraction)   │
    └──────────────────────────┘     └────────────────────────┘     └────────────────────────┘

When a surface defect is isolated, the model extracts the binary mask coordinates and computes the ratio of corrupted pixels to total structural surface area. This spatial ratio is converted into a normalized **Defect Mask Score** bounded strictly between `[0.0, 1.0]`, where `0.0` represents flawless surface integrity and `1.0` denotes complete localized wall breach. This scalar value acts as the third feature vector within Branch Net 3.

---

## Tensor Contract & Architecture Design

To ensure strict dimensional alignment and prevent gradient shape compilation failures within PyTorch, the system architecture separates discrete physical parameter spaces from continuous space-time tracking fields:

              ┌──────────────────────────────┐
              │   Branch 1: Soil (GRF)       │ ──► [Batch, 8]   ┐
              └──────────────────────────────┘                  │
              ┌──────────────────────────────┐                  │    Late-split
              │   Branch 2: Fluid (GRF)      │ ──► [Batch, 50]  ┼──► factored merge
              └──────────────────────────────┘                  │
              ┌──────────────────────────────┐                  │         │
              │   Branch 3: Meta (API-5L)    │ ──► [Batch, 3]   ┘         │
              └──────────────────────────────┘                            ▼
                                                                    Dot Product Mapping
                                                                    ───────┬───────────
                                                                           ▼
              ┌──────────────────────────────┐                        Multi-Variable Output
              │   Trunk Net: Space-Time Grid │ ──► [B, N, 3] ──────►  [Concentration, Potential]
              └──────────────────────────────┘

- **Branch Net 1 (Soil Input):** Formulates environmental patches via a 1D GRF kernel with a localized spatial correlation length scale ($\ell = 0.15$), inputting a tensor shape of `[Batch, 8]`. The profile enters the physics through a spatially varying conductivity $\sigma(x)$.
- **Branch Net 2 (Fluid Input):** Parameterizes inner pipeline dynamics via a smooth GRF kernel ($\ell = 0.30$), inputting a tensor shape of `[Batch, 50]`. The profile enters the physics through a spatially varying bulk concentration $C_{\text{bulk}}(x)$.
- **Branch Net 3 (Structural Metadata):** Processes structural constraints derived from the API-5L scraping routines and the localized visual defect severity scores output by the `YOLOv8` pipeline, inputting a tensor shape of `[Batch, 3]` mapping `[Wall Thickness, Mean Resistivity, Defect Mask Score]`.
- **Trunk Net (Spacetime Grid):** Tracks continuous non-dimensional coordinates $(\tilde{x}, \tilde{d}, \tilde{t})$ where $\tilde{d} = 1 - \tilde{y}$, inputting a tensor shape of `[Batch, N, 3]`.

Both spatially varying couplings were introduced deliberately: an earlier revision of the operator was sensitive to only two of its input dimensions, leaving the remaining branches decorative.

---

## Measured Operator Performance

Accuracy is evaluated against a 2D Finite Difference Method (FDM) numerical baseline on held-out samples.

All figures are **median [min–max] across seeds**, with $n$ stated. Single-run numbers are not reported: a same-seed, same-config replicate differed as much as different seeds did, so run-to-run variance exceeds most configuration differences in this setup.

`flux_err` is relative error on wall flux — the quantity the boundary term directly penalizes, and the closest available proxy for corrosion current density. `op_corr` is the correlation of mean wall concentration across environments, i.e. whether the operator ranks environments in the correct order.

| Arm | n | C rel. $L_2$ | flux_err | op_corr |
| :--- | :---: | :---: | :---: | :---: |
| Strong form, `W_BC=1` | 2 | 0.127 | 0.069 | +0.897 |
| Strong form, anchor removed | 2 | 0.109 | 0.065 | — |
| Weak form, bs=64 matched | 5 | 0.174 | 0.359–0.834 | +0.958 |
| Weak form, anchor removed | 2 | 0.99 | 2.23 | — |

### Ablation Findings

Three results, all multi-seed:

**Objective–accuracy misalignment.** Training the weak form for longer lowers the weak-form objective while raising the error. From 7,205 to 13,200 gradient steps at bs=64, `weak_total` fell 2.08e-2 → 1.16e-2 while flux error rose 0.54 → 1.36 and C rel. $L_2$ rose 0.174 → 0.286. All five seeds, monotone, and it survives the L-BFGS refinement phase.

**The weak form does not identify the solution without supervision.** With the semi-supervised FDM anchor removed, the weak form reaches C rel. $L_2 \approx 0.99$ with negative lateral correlation — a non-solution rather than a degraded solution. The strong form under the identical ablation reaches 0.109.

**The strong-form control outperforms the weak form** on every metric except `op_corr`, where the weak form leads (+0.958 vs +0.897).

L-BFGS refinement degraded results in every instance tested and was dropped from the final configuration.

---

## Interpretability

Attribution of wall flux to the three input branches, cross-checked with input×gradient and SHAP expected gradients:

| Branch | Share |
| :--- | :---: |
| Fluid chemistry | 82–91% |
| Soil resistivity | 8–11% |
| Structural scalars | 8–10% |

These shares reflect how much each input **varies across this dataset**, not a universal ranking of physical importance. Bulk concentration sets the boundary condition the solution scales from, and it varies widely here by construction.

**Per-section attribution was tested and does not hold.** Correlation between attribution centroid and queried section appeared strong (+0.97 strong form, +0.55 weak), but correlation is scale-free. The regression slope is ≈0.2 for every model across both loss forms — a section's attribution centroid moves a fifth as far as the section does — and attribution spread (0.306–0.323) exceeds a uniform distribution over the domain (0.289). The operator learned a global response to the $C_{\text{bulk}}$ profile rather than a spatially resolved one. Retained in `diagnostics/` as a negative result; no per-section attribution view should be built on it.

---

## Verification & Deployment Strategy

The trained operator serves as the backend engine for an interactive web dashboard. Pipeline asset operators adjust structural configurations or environmental parameters via a GUI, triggering a forward pass that updates localized degradation heatmaps without placing computational load on cluster hardware.

Recommended display constraints: wall-adjacent quantities on a log axis (a linear axis cannot separate 1e-2 from 1e-6), whole-pipe attribution only, and no confidence bands — seed spread is available but is not calibrated uncertainty.

---

## Current Limitations

Four constraints bound the results above and are not scheduled for resolution within this project's scope:

1. **The FDM reference is not fully validated.** The potential boundary condition and reaction-rate kinetics carry unresolved assumptions, and a sign discrepancy in the `rho → C_wall` mapping is open. Predicted values should be treated as research output rather than calibrated corrosion current densities.
2. **The soil input has no validated mapping to field measurements.** It is currently a synthetic Gaussian random field; connecting it to real resistivity survey data is outstanding work.
3. **Sample sizes are uneven.** Strong-form arms are $n = 2$; weak-form arms are $n = 5$. Batch size is unmatched between them (32 vs 64) even though gradient steps are matched.
4. **A known weak-form remedy was not applied.** Dual-norm reweighting for test-space null spaces (Rojas et al.) addresses a pathology present in this construction and was not tested. The identifiability result therefore scopes to the weak form *as standardly constructed*, not to weak formulations in general.

---

## Repository Notes

Model checkpoints are tracked under `results/model_files/`. All five weak-form seeds are retained — they are the evidence base for the seed-variance claim, and regenerating them requires retraining.

Evaluation JSON under `results/json_files/` is sufficient to regenerate every reported figure without the checkpoints.

`papers/` (third-party PDFs) and `ignore/` are gitignored and local-only.

---

## Citation

If you utilize the AmorFlux architectural framework or system design pipelines in your research, please use the following citation format:

```bibtex
@misc{Gouda2026:AmorFlux,
  author       = {Ayush Gouda and Aditya Prakash and Swaraag Hebbar N.},
  title        = {{AmorFlux: Multi-Branch Physics-Informed Neural Operators with GraphRAG for Amortized Pipeline Corrosion Prognostics}},
  howpublished = {\url{https://github.com/AyushG-1210/Pipeline-Digital-Twin}},
  year         = {2026},
  month        = {August}
}
```