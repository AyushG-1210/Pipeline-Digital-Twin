/**
 * Segment data generators.
 *
 * PINN and XAI output are no longer mocked: both are loaded from
 * pinn_model/benchmarks.json, built offline by pinn_model/build_benchmarks.py
 * by running the production checkpoint (sf_wbc1_s42, plus the 4 other
 * trained seeds for the XAI ensemble) over real scenarios recovered from the
 * FDM training/holdout dataset. Each of the SEGMENT_COUNT segments gets one
 * real, distinct benchmark assigned at startup.
 *
 * (pinn_model/ holds the deployed checkpoints + generated artifacts for
 * this app — the actual PINN research project, training data, and full
 * experiment history live in the root project's PINN/ folder.)
 *
 * CV is still mock — that's the last pass (YOLO wiring).
 */

import { readFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BENCHMARKS_PATH = path.join(__dirname, '..', 'pinn_model', 'benchmarks.json');

const SEGMENT_COUNT = 60; // Increased length for a more impressive pipeline
const SEGMENT_LENGTH = 2; // meters

function loadBenchmarks() {
    const raw = readFileSync(BENCHMARKS_PATH, 'utf-8');
    const benchmarks = JSON.parse(raw);
    if (benchmarks.length < SEGMENT_COUNT) {
        throw new Error(
            `pinn_model/benchmarks.json has ${benchmarks.length} entries, need >= ${SEGMENT_COUNT}. ` +
            `Regenerate with: python pinn_model/build_benchmarks.py --n_segments ${SEGMENT_COUNT}`
        );
    }
    return benchmarks;
}

/**
 * Generate initial segment data
 */
function generateSegments() {
    const segments = [];

    // We trace the curve using small steps and place a segment every SEGMENT_LENGTH distance.
    const amplitudeY = 3;
    const frequencyY = 0.08;
    const amplitudeZ = 8;
    const frequencyZ = 0.12;

    // Helper to get point on curve
    const getPoint = (t) => {
        return [
            t,
            Math.sin(t * frequencyY) * amplitudeY,
            Math.sin(t * frequencyZ) * amplitudeZ
        ];
    };
    
    // Helper to get derivative (tangent) on curve
    const getTangent = (t) => {
        return [
            1,
            Math.cos(t * frequencyY) * frequencyY * amplitudeY,
            Math.cos(t * frequencyZ) * frequencyZ * amplitudeZ
        ];
    };

    let currentT = -(SEGMENT_COUNT * SEGMENT_LENGTH) / 2;
    const benchmarks = loadBenchmarks();

    for (let i = 0; i < SEGMENT_COUNT; i++) {
        const segmentId = `SEG-${String(i + 1).padStart(3, '0')}`;
        // Each segment gets one real, distinct PINN benchmark — the model's
        // actual computed integrity for that scenario, not a random number.
        // xai (waterfall decomposition + ensemble uncertainty) travels with
        // it, computed by the same script — also real, also per-segment.
        const { xai, ...pinnFields } = benchmarks[i];
        const integrity = pinnFields.integrity;

        const position = getPoint(currentT);
        const direction = getTangent(currentT);

        segments.push({
            segment_id: segmentId,
            position: position,
            direction: direction,
            integrity: integrity,
            cv: generateCVOutput(segmentId, integrity),
            pinn: { ...pinnFields, segment_id: segmentId },
            xai: { ...xai, segment_id: segmentId },
            lastUpdated: new Date().toISOString()
        });

        // Advance currentT by approximately SEGMENT_LENGTH
        // dt = ds / ||tangent||
        const tangent = getTangent(currentT);
        const tangentLength = Math.sqrt(tangent[0]*tangent[0] + tangent[1]*tangent[1] + tangent[2]*tangent[2]);
        currentT += SEGMENT_LENGTH / tangentLength;
    }

    return segments;
}

/**
 * Generate CV (Computer Vision) output
 */
function generateCVOutput(segmentId, integrity) {
    const corrosionPct = (1 - integrity) * 100 * (0.8 + Math.random() * 0.4);
    const confidence = 0.7 + Math.random() * 0.25;

    // Generate polygon mask (normalized coordinates)
    const polygonMask = [
        [0.4 + Math.random() * 0.1, 0.3 + Math.random() * 0.1],
        [0.5 + Math.random() * 0.1, 0.3 + Math.random() * 0.1],
        [0.55 + Math.random() * 0.1, 0.4 + Math.random() * 0.1],
        [0.5 + Math.random() * 0.1, 0.45 + Math.random() * 0.1],
        [0.4 + Math.random() * 0.1, 0.4 + Math.random() * 0.1]
    ];

    return {
        segment_id: segmentId,
        corrosion_surface_pct: Math.max(0, Math.min(100, corrosionPct)),
        confidence: confidence,
        polygon_mask: polygonMask,
        frame_timestamp: new Date().toISOString()
    };
}

/**
 * Refresh a random segment's still-mock CV reading on the periodic "live"
 * tick. PINN, xai, and integrity are intentionally left untouched here —
 * they're the real benchmark result assigned in generateSegments(), and a
 * per-tick Math.random() nudge would just reintroduce the fake churn this
 * was built to get rid of. Real-time degradation modelling is future work.
 */
function updateRandomSegment(segments) {
    const index = Math.floor(Math.random() * segments.length);
    const segment = segments[index];

    segment.cv = generateCVOutput(segment.segment_id, segment.integrity);
    segment.lastUpdated = new Date().toISOString();

    return segment;
}

export {
    generateSegments,
    updateRandomSegment
};
