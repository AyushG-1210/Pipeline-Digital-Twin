import { useState } from 'react';
import { usePipelineStore } from '../../store/usePipelineStore';
import type { CVOutput, FeedbackEntry } from '../../types';

interface FeedbackControlsProps {
    segmentId: string;
    cvData: CVOutput;
}

/**
 * Human-in-the-loop feedback controls for CV detections
 * Allows operators to confirm or reject detections
 */
export default function FeedbackControls({ segmentId, cvData }: FeedbackControlsProps) {
    const [feedbackGiven, setFeedbackGiven] = useState(false);
    const [feedbackType, setFeedbackType] = useState<'confirm' | 'false_positive' | null>(null);
    const addFeedback = usePipelineStore(state => state.addFeedback);
    const feedbackQueue = usePipelineStore(state => state.feedbackQueue);

    const handleFeedback = (type: 'confirm' | 'false_positive') => {
        const feedback: FeedbackEntry = {
            segment_id: segmentId,
            detection_timestamp: cvData.frame_timestamp,
            feedback_type: type,
            timestamp: new Date().toISOString()
        };

        addFeedback(feedback);
        setFeedbackGiven(true);
        setFeedbackType(type);

        setTimeout(() => {
            setFeedbackGiven(false);
            setFeedbackType(null);
        }, 2000);
    };

    return (
        <div className="inner-card p-4">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <svg className="w-4 h-4 text-secondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                    <h3 className="text-sm font-medium text-ink">Human Verification</h3>
                </div>
                {feedbackQueue.length > 0 && (
                    <span className="flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-semibold rounded-full"
                        style={{ background: '#FBEEDA', color: '#B5721B' }}>
                        {feedbackQueue.length} queued
                    </span>
                )}
            </div>

            <div className="flex gap-3">
                <button
                    onClick={() => handleFeedback('confirm')}
                    disabled={feedbackGiven}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors"
                    style={{
                        background: feedbackType === 'confirm' ? '#E4F2EA' : '#FFFFFF',
                        border: `1px solid ${feedbackType === 'confirm' ? '#4C9A78' : '#E1DED3'}`,
                        color: feedbackType === 'confirm' ? '#3C7C61' : '#454C67',
                        opacity: feedbackGiven && feedbackType !== 'confirm' ? 0.4 : 1,
                    }}
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                    </svg>
                    Confirm
                </button>

                <button
                    onClick={() => handleFeedback('false_positive')}
                    disabled={feedbackGiven}
                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors"
                    style={{
                        background: feedbackType === 'false_positive' ? '#FBE4E1' : '#FFFFFF',
                        border: `1px solid ${feedbackType === 'false_positive' ? '#D6473C' : '#E1DED3'}`,
                        color: feedbackType === 'false_positive' ? '#B33529' : '#454C67',
                        opacity: feedbackGiven && feedbackType !== 'false_positive' ? 0.4 : 1,
                    }}
                >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    Reject
                </button>
            </div>

            <div className={`mt-3 overflow-hidden transition-all duration-500 ${feedbackGiven ? 'h-5 opacity-100' : 'h-0 opacity-0'}`}>
                <div className="flex items-center justify-center gap-2 text-xs font-medium"
                    style={{ color: feedbackType === 'confirm' ? '#3C7C61' : '#B33529' }}>
                    Feedback recorded
                </div>
            </div>

            {!feedbackGiven && (
                <div className="mt-3 text-[11px] text-center text-ink-muted">
                    Refines YOLOv8-Seg accuracy over time
                </div>
            )}
        </div>
    );
}
