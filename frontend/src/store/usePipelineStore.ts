import { create } from 'zustand';
import type { SegmentData, FeedbackEntry, CVOutput, PINNOutput, XAIOutput } from '../types/index.js';

interface PipelineStore {
    // State
    segments: Map<string, SegmentData>;
    selectedSegmentId: string | null;
    feedbackQueue: FeedbackEntry[];
    wsConnected: boolean;
    lastUpdate: string | null;

    // Actions
    setSegments: (segments: SegmentData[]) => void;
    updateSegment: (segmentId: string, data: Partial<SegmentData>) => void;
    selectSegment: (segmentId: string | null) => void;
    addFeedback: (feedback: FeedbackEntry) => void;
    setWSConnected: (connected: boolean) => void;
    updateCV: (cv: CVOutput) => void;
    updatePINN: (pinn: PINNOutput) => void;
    updateXAI: (xai: XAIOutput) => void;
}

export const usePipelineStore = create<PipelineStore>((set, get) => ({
    // Initial state
    segments: new Map(),
    selectedSegmentId: null,
    feedbackQueue: [],
    wsConnected: false,
    lastUpdate: null,

    // Actions
    setSegments: (segments) => {
        const segmentMap = new Map<string, SegmentData>();
        segments.forEach(seg => segmentMap.set(seg.segment_id, seg));
        set({ segments: segmentMap, lastUpdate: new Date().toISOString() });
    },

    updateSegment: (segmentId, data) => {
        const segments = new Map(get().segments);
        const existing = segments.get(segmentId);
        if (existing) {
            segments.set(segmentId, { ...existing, ...data, lastUpdated: new Date().toISOString() });
            set({ segments, lastUpdate: new Date().toISOString() });
        }
    },

    selectSegment: (segmentId) => {
        set({ selectedSegmentId: segmentId });
    },

    addFeedback: (feedback) => {
        set(state => ({
            feedbackQueue: [...state.feedbackQueue, feedback]
        }));
    },

    setWSConnected: (connected) => {
        set({ wsConnected: connected });
    },

    updateCV: (cv) => {
        const segments = new Map(get().segments);
        const existing = segments.get(cv.segment_id);
        if (existing) {
            segments.set(cv.segment_id, {
                ...existing,
                cv,
                lastUpdated: new Date().toISOString()
            });
            set({ segments, lastUpdate: new Date().toISOString() });
        }
    },

    updatePINN: (pinn) => {
        const segments = new Map(get().segments);
        const existing = segments.get(pinn.segment_id);
        if (existing) {
            segments.set(pinn.segment_id, {
                ...existing,
                pinn,
                lastUpdated: new Date().toISOString()
            });
            set({ segments, lastUpdate: new Date().toISOString() });
        }
    },

    updateXAI: (xai) => {
        const segments = new Map(get().segments);
        const existing = segments.get(xai.segment_id);
        if (existing) {
            segments.set(xai.segment_id, {
                ...existing,
                xai,
                lastUpdated: new Date().toISOString()
            });
            set({ segments, lastUpdate: new Date().toISOString() });
        }
    },
}));
