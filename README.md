# AmorFlux: Multi-Branch Physics-Informed Neural Operators with GraphRAG for Amortized Pipeline Corrosion Prognostics

> Ayush Gouda, Aditya Prakash, Swaraag Hebbar N  
> Department of Computer Science & Engineering, RV Institute of Technology and Management

---
## Project Status: Ongoing (Expected Completion: August 2026)

This repository contains the foundational system design, data ingestion pipelines, and neural operator code architecture for **AmorFlux**. The framework is currently being optimized on a cloud-hosted NVIDIA A100 cluster environment via Kubeflow.

---

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![DeepXDE](https://img.shields.io/badge/DeepXDE_SciML-1.12.0-00B4D8?style=flat-square&logo=scipy&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-5.19-008CC1?style=flat-square&logo=neo4j&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

**AmorFlux** is a multi-modal, physics-informed Scientific Machine Learning (SciML) framework designed to simulate real-time localized electrochemical corrosion dynamics and predict the Remaining Useful Life (RUL) of midstream oil and gas pipelines. 

Conventional physics-informed neural networks (PINNs) act as isolated numerical solvers, requiring computationally expensive retraining from scratch every time an environmental profile, pipe geometry, or fluid property changes. **AmorFlux** circumvents this boundary-value bottleneck by utilizing a Multi-Input Neural Operator (**MIONet**) architecture. By training on continuous parameter spaces, the framework functions as an *amortized solver*—executing instantaneous, millisecond-level forward-pass inferences across arbitrary pipeline networks without re-optimization.

### System Architecture Pipeline

The framework orchestrates three deeply integrated data and deep learning layers:
1. **Multi-Modal GraphRAG Ingestion:** Unstructured compliance documentation from the Texas Railroad Commission (RRC), PHMSA, NACE, and API-5L specifications are ingested via a `ColPali` vision-language retrieval model. Extracted structural attributes and regulatory standards are mapped into a `Neo4j` topological knowledge graph.
2. **Spatially Correlated Parameterization:** Automated connectors hook into Google Earth Engine (GEE) to extract localized soil moisture and surface temperatures. These features are smoothed along the longitudinal pipe coordinate using a 1D Gaussian Random Field (GRF) kernel to feed the model with realistic environmental patches.
3. **Multi-Branch Neural Operator Evaluation:** The parameter tensors and continuous space-time coordinates are split across parallel Branch and Trunk neural networks, continuously constrained by the underlying electrochemical partial differential equations.

---

## Core Mathematical Framework

AmorFlux maps the physical degradation of the pipeline wall boundary layer box by continuously solving the mass-transport equations coupled with non-linear electrochemical reaction kinetics.

### 1. Interior Domain Governing Equation (Fick's Second Law)
The migration of aggressive corrosive species (e.g., oxygen, moisture, chlorides) through the soil or electrolyte matrix toward the steel pipe wall over time is governed by a 2D spatial, 1D temporal diffusion equation:

$$\frac{\partial C}{\partial t} = D \left( \frac{\partial^2 C}{\partial x^2} + \frac{\partial^2 C}{\partial y^2} \right)$$

Where $C(x, y, t)$ represents the localized concentration profile of the corrosive species, and $D$ is the global diffusion coefficient parameterized by the soil branch.

### 2. Interface Interface Condition (Butler-Volmer Kinetics)
At the extreme upper edge of the boundary layer corresponding to the steel pipe interface ($y = 1$), mass flux is coupled directly to the rate of chemical consumption via non-linear electrochemical kinetics:

$$\left. D \frac{\partial C}{\partial y} \right|_{y=1} + k_{\text{rate}} C \cdot \exp\left( \frac{\alpha F}{R T} (\phi - E_{\text{eq}}) \right) = 0$$

Where $\phi$ is the localized electrical potential predicted by the network, $E_{\text{eq}}$ is the equilibrium potential, $F$ is Faraday's constant, $R$ is the universal gas constant, and $T$ is the interface temperature profile. The model solves this boundary system to isolate the true **corrosion current density ($I_{\text{corr}}$)**, which is integrated over time to generate RUL prognostic arrays.

### 3. Visual Defect Segmentation & YOLOv8 Vision Pipeline

To anchor the continuous neural operator to real-world structural degradation, the framework integrates a real-time computer vision pipeline engineered to detect and segment localized macro-defects from physical inspection feeds (e.g., drone imagery, robotic crawler videos). 

The vision layer utilizes a specialized `YOLOv8-seg` instance segmentation architecture pre-trained on high-resolution industrial surface defect datasets. The model tracks five distinct classes of structural anomalies: external pitting, line cracks, coating degradation, structural gouges, and localized anomalies.

    ┌──────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
    │ Raw Drone/Crawler Feed   │ ──► │  YOLOv8-seg Model      │ ──► │ Defect Severity Mask   │
    │ (Surface Inspection)     │     │  (Instance Segmentation│     │ (Pixel Area Compute)   │
    └──────────────────────────┘     └────────────────────────┘     └────────────────────────┘
    │
    ▼
    ┌──────────────────────────┐     ┌────────────────────────┐     ┌────────────────────────┐
    │ Branch 3 Input Tensor    │ ◄── │ Min-Max Normalization  │ ◄── │ Normalized Score [0,1] │
    │ [WT, OD, Defect_Score]   │     │(0: Intact, 1: Critical)│     │ (Feature Extraction)   │
    └──────────────────────────┘     └────────────────────────┘     └────────────────────────┘

When a surface defect is isolated, the model extracts the binary mask coordinates and computes the ratio of corrupted pixels to total structural surface area. This spatial ratio is converted into a normalized **Defect Mask Score** bounded strictly between `[0.0, 1.0]`, where `0.0` represents flawless surface integrity and `1.0` denotes complete localized wall breach. This scalar value acts as the crucial third feature vector within Branch Net 3, ensuring the downstream neural operator heavily penalizes localized structural resistance and projects accelerated corrosion current densities at pre-damaged physical coordinates.

---

## Tensor Contract & Architecture Design

To ensure strict dimensional alignment and prevent gradient shape compilation failures within PyTorch, the system architecture separates discrete physical parameter spaces from continuous space-time tracking fields:

              ┌──────────────────────────────┐
              │   Branch 1: Soil (GRF)       │ ──► [Batch, 50]  ┐
              └──────────────────────────────┘                  │
              ┌──────────────────────────────┐                  │   Element-wise
              │   Branch 2: Fluid (GRF)      │ ──► [Batch, 50]  ┼──► Multiplication
              └──────────────────────────────┘                  │    [Batch, 128]
              ┌──────────────────────────────┐                  │         │
              │   Branch 3: Meta (API-5L)    │ ──► [Batch, 3]   ┘         │
              └──────────────────────────────┘                            ▼
                                                                    Dot Product Mapping
                                                                    ───────┬───────────
                                                                           ▼
              ┌──────────────────────────────┐                       Multi-Variable Output
              │   Trunk Net: Space-Time Grid │ ──► [M, 3] ─────────►  [Concentration, Potential]
              └──────────────────────────────┘

- **Branch Net 1 (Soil Input):** Formulates environmental patches via a 1D GRF kernel with a localized spatial correlation length scale ($\ell = 0.15$), inputting a tensor shape of `[Batch, 50]`.
- **Branch Net 2 (Fluid Input):** Parameterizes inner pipeline dynamics (internal pressure, fluid flow profiles) via a smooth GRF kernel ($\ell = 0.30$), inputting a tensor shape of `[Batch, 50]`.
- **Branch Net 3 (Structural Metadata):** Processes structural constraints derived from the API-5L scraping routines and the localized visual defect severity scores output by the team's `YOLOv8` computer vision pipeline, inputting a tensor shape of `[Batch, 3]` mapping `[Wall Thickness, Outer Diameter, Defect Mask Score]`.
- **Trunk Net (Spacetime Grid):** Tracks continuous non-dimensional coordinates $(\tilde{x}, \tilde{y}, \tilde{t})$ using Space-Filling Latin Hypercube Sampling across the normalized `[0, 1]` domain box, inputting a tensor shape of `[M, 3]`.

---

## Expected Results & Ablation Goals

The model evaluates prediction accuracy using the $L_2$ relative error metric compared against a high-fidelity 2D Finite Difference Method (FDM) classical numerical baseline.

### Targeted Operator Performance Metrics

| Simulation Profile Scenario | Target L2 Relative Error | Expected Inference Velocity | Convergence Stability Epochs |
| :--- | :---: | :---: | :---: |
| **Homogeneous Soil (Baseline)** | < 0.35% | 1.24 ms | ~2,500 Epochs |
| **Spatially Correlated Soil (GRF)** | < 0.82% | 1.26 ms | ~4,500 Epochs |
| **Severe Localized Pitting (YOLO Influenced)** | < 1.45% | 1.31 ms | ~7,000 Epochs |

### Anti-Cheat Pathological Ablation Summary

To prevent the neural network from falling into trivial local minima (shortcut pathologies where the network outputs a flat, blanket response to minimize the diffusion derivatives), a custom **Physics Pathology Tracker** is integrated into the PyTorch backward loop to monitor decoupled loss trajectories.

| Optimization Methodology | PDE Residual Convergence | Boundary Interface Resolution | Shortcut Vulnerability Rate |
| :--- | :---: | :---: | :---: |
| **Standard MSE Baseline** | 1.2e-5 (Cheated) | 4.8e-1 (Unresolved) | 88.4% (Collapses to Flat Constant) |
| **Decoupled Loss Balancing** | 4.3e-4 (Valid) | 6.1e-3 (Resolved) | 12.1% (Stable Initialization) |
| **Dynamic Learning Rate Annealing (Ours)** | **1.1e-4 (Valid)** | **8.5e-4 (Resolved)** | **< 0.5% (Robust Convergence)** |

---

## Verification & Deployment Strategy

Once the multi-stage training sequence (Adam navigation followed by L-BFGS refinement) converges on the Kubeflow cluster, the final weights are frozen and compiled. 

The resulting amortized neural operator serves as the backend engine for a real-time `Three.js` interactive web dashboard. Pipeline asset operators can adjust structural configurations or environmental parameters via a GUI, triggering an immediate model forward-pass that instantly updates 3D localized degradation heatmaps without placing any computational load on cluster hardware.

---

## Citation

If you utilize the AmorFlux architectural framework or system design pipelines in your research, please use the following citation format:

```bibtex
@misc{Gouda2026:AmorFlux,
  author       = {Ayush Gouda and Aditya Prakash and Swaraag Hebbar N.},
  title        = {{AmorFlux: Multi-Branch Physics-Informed Neural Operators with GraphRAG for Amortized Pipeline Corrosion Prognostics}},
  howpublished = {\url{https://github.com/AyushG-1210/AmorFlux}},
  year         = {2026},
  month        = {August}
}
```