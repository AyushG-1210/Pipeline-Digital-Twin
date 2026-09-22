/**
 * Data Contracts for Digital Twin Pipeline Integrity System
 * These interfaces mirror the outputs from ML models (CV, PINN, XAI)
 * and must match the production API specifications exactly.
 */

/**
 * Computer Vision Output - YOLOv8-Seg corrosion detection
 */
export interface CVOutput {
    segment_id: string;
    corrosion_surface_pct: number; // 0-100: percentage of surface area affected
    confidence: number; // 0-1: model confidence in detection
    polygon_mask: number[][]; // normalized [x, y] coordinates for corrosion region
    frame_timestamp: string; // ISO 8601 date string

    // Real YOLOv8 inference fields (populated when Run YOLO Analysis is triggered)
    corrosion_detected?: boolean;
    class_id?: number;           // -1 if no detection
    class_name?: string;         // 'BX'|'CJ'|'CK'|'OBB'|'PL'|'SG'|'ZW'|'none'
    severity_score?: number;     // raw model confidence of best detection (0-1)
    detection_rate?: number;     // frames_with_detections / total_frames
    total_frames?: number;
    frames_with_detections?: number;
    yolo_mask_px?: number[][];   // raw pixel coords [x,y] from YOLO (640x640 space)
    is_yolo_result?: boolean;    // true when populated by real inference
}

/**
 * Physics constants used by the PINN model (from MANIFEST.json)
 */
export interface PhysicsParams {
    D_global: number;    // Diffusion coefficient (m²/s), 1e-9
    k_rate: number;      // Reaction rate constant (m/s), 2.07e-7
    alpha: number;       // Butler-Volmer transfer coefficient, 0.5
    E_eq: number;        // Equilibrium potential (V), -0.44
    E_corr: number;      // Free-corrosion potential (V), -0.44
    t_max: number;       // Max time (s), 31536000 = 1 year
    L_x: number;         // Axial domain length (m), 10.0
    n_electrons: number; // Electrons per reaction, 2.0
}

/**
 * Physics-Informed Neural Network Output - Subsurface degradation forecast
 * Mirrors the output of pinn_inference.py (FactoredMIONet, sf_wbc1_s42).
 */
export interface PINNOutput {
    segment_id: string;
    historical_integrity: Array<{ time: string; value: number }>; // Past sensor readings
    predicted_integrity:  Array<{ time: string; value: number }>; // Future projections
    remaining_useful_life_days: number; // Days until critical threshold
    governing_equation: 'Fick_2nd_Law' | 'Fick_2nd_Law_strong';  // Physics model used

    // --- Real PINN inference fields (null/false on mock data) ---
    /** 100-point concentration profile at the outer wall, C(x) ∈ [0, 1] */
    C_profile?: number[];
    /** 100-point electric potential profile at the outer wall, φ(x) */
    phi_profile?: number[];
    /** 50-point concentration depth profile at pipe midpoint, C(d) d∈[0=wall,1=bulk] */
    C_depth_profile?: number[];
    /** 50-point potential depth profile at pipe midpoint */
    phi_depth_profile?: number[];
    /** Mean concentration at outer wall */
    C_wall_mean?: number;
    /** Mean absolute surface corrosion flux */
    flux?: number;
    /** Hold-set relative L2 error for concentration (lower = better) */
    C_l2?: number;
    /** Hold-set relative L2 error for electric potential */
    phi_l2?: number;
    /** Operational correlation — across-sample discriminability (higher = better) */
    op_corr?: number;
    /** Model checkpoint tag, e.g. 'sf_wbc1_s42' */
    model_tag?: string | null;
    /** Physics constants used during inference */
    physics_params?: PhysicsParams;
    /** true when data comes from real PyTorch inference; false for mock */
    is_pinn_result?: boolean;
    /** Wall thickness input in inches */
    wt_in?: number;
    /** Outer diameter input in inches */
    od_in?: number;
    /** YOLO defect score used as model input */
    yolo_score?: number;
    /** Time taken for inference in seconds */
    inference_time_s?: number;
}

/**
 * Exact decomposition of the predicted wall concentration into the
 * prescribed boundary condition plus each branch net's learned correction.
 * Because the architecture sums branch features linearly before the trunk
 * dot product, these four numbers sum EXACTLY to the true predicted value
 * — this isn't an approximation (SHAP/IG), it's the real computation this
 * specific segment's prediction is built from.
 */
export interface WaterfallDecomposition {
    baseline: number; // prescribed O2 boundary condition, from the fluid input directly
    soil: number;     // learned correction from the soil-resistivity branch
    fluid: number;    // learned correction from the fluid/O2-concentration branch
    meta: number;     // learned correction from geometry + defect score (WT, OD, YOLO)
    total: number;    // baseline + soil + fluid + meta == predicted C(wall)
}

/**
 * Ensemble uncertainty across the 5 independently-trained checkpoints
 * (seeds 42-46). A deterministic PINN forward pass has no softmax to read
 * a "confidence" from — this is the real substitute: how much do 5
 * separately-trained models agree on this specific segment.
 */
export interface EnsembleUncertainty {
    seeds: number[];
    integrity_mean: number;
    integrity_std: number;
    rul_mean: number;
    rul_std: number;
    rul_min: number;
    rul_max: number;
    flux_std: number;
}

/**
 * Five real, independently-varying per-segment quantities, each normalised
 * against the full 750-scenario population and oriented so higher = better
 * (healthier / more trustworthy). Unlike the waterfall's meta branch (held
 * constant across this benchmark set — no per-segment pipe geometry in the
 * FDM dataset), every one of these axes genuinely differs segment to
 * segment; this is what explains why one segment looks different from the
 * next. See PINN/build_benchmarks.py's compute_fingerprint().
 */
export interface SegmentFingerprint {
    environmental_integrity: number; // inverse of corrosion flux
    soil_resistivity: number;        // higher = less conductive soil = healthier
    oxygen_scarcity: number;         // inverse of bulk O2 concentration
    model_accuracy: number;          // inverse of this scenario's C_l2 vs FDM ground truth
    ensemble_agreement: number;      // agreement across the 5-seed ensemble
}

/**
 * The real environmental scenario this segment's prediction is based on —
 * the actual 21-point soil resistivity and bulk O2 concentration curves
 * recovered from the FDM dataset, along the scenario's normalised axial
 * position. This is the literal cause of the prediction, not a derived
 * explanation of it.
 */
export interface EnvironmentProfile {
    x: number[];
    soil_resistivity: number[];   // Ohm*m
    fluid_concentration: number[]; // fraction, 0-1
}

/**
 * Explainable AI Output — real, per-segment model introspection.
 */
export interface XAIOutput {
    segment_id: string;
    waterfall: WaterfallDecomposition;
    ensemble: EnsembleUncertainty;
    fingerprint: SegmentFingerprint;
    environment: EnvironmentProfile;
}

/**
 * Unified segment data combining all outputs
 */
export interface SegmentData {
    segment_id: string;
    position: [number, number, number]; // 3D position in scene
    direction?: [number, number, number]; // 3D direction vector
    integrity: number; // 0-1: current integrity (1 = healthy, 0 = critical)
    cv?: CVOutput;
    pinn?: PINNOutput;
    xai?: XAIOutput;
    lastUpdated: string; // ISO timestamp
}

/**
 * Human-in-the-loop feedback for CV detections
 */
export interface FeedbackEntry {
    segment_id: string;
    detection_timestamp: string;
    feedback_type: 'confirm' | 'false_positive';
    user_id?: string;
    timestamp: string;
}

/**
 * WebSocket message types
 */
export interface WSMessage {
    type: 'segment_update' | 'cv_detection' | 'pinn_forecast' | 'xai_explanation' | 'initial_data';
    data: SegmentData | CVOutput | PINNOutput | XAIOutput | SegmentData[];
    timestamp: string;
}

/**
 * Urgency levels for RUL (Remaining Useful Life)
 */
export type UrgencyLevel = 'low' | 'medium' | 'high' | 'critical';

/**
 * Chart data point for time series
 */
export interface ChartDataPoint {
    time: string;
    value: number;
    label?: string;
}

/**
 * YOLO Analysis API response
 */
export interface YOLOAnalysisResult {
    segment_id: string;
    cv: CVOutput;
    success: boolean;
    error?: string;
}
