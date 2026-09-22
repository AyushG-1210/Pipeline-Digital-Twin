import { usePipelineStore } from '../store/usePipelineStore';
import type { WSMessage } from '../types';

let ws: WebSocket | null = null;
let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
const RECONNECT_DELAY = 3000;

/**
 * Initialize WebSocket connection to backend
 */
export function initWebSocket(url: string = 'ws://localhost:3001') {
    if (ws && ws.readyState === WebSocket.OPEN) {
        console.log('WebSocket already connected');
        return;
    }

    try {
        ws = new WebSocket(url);

        ws.onopen = () => {
            console.log('WebSocket connected');
            usePipelineStore.getState().setWSConnected(true);

            // Clear any pending reconnect
            if (reconnectTimeout) {
                clearTimeout(reconnectTimeout);
                reconnectTimeout = null;
            }
        };

        ws.onmessage = (event) => {
            try {
                const message: WSMessage = JSON.parse(event.data);
                handleWSMessage(message);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected');
            usePipelineStore.getState().setWSConnected(false);
            ws = null;

            // Attempt to reconnect
            reconnectTimeout = setTimeout(() => {
                console.log('Attempting to reconnect...');
                initWebSocket(url);
            }, RECONNECT_DELAY);
        };
    } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        usePipelineStore.getState().setWSConnected(false);
    }
}

/**
 * Handle incoming WebSocket messages
 */
function handleWSMessage(message: WSMessage) {
    const store = usePipelineStore.getState();

    switch (message.type) {
        case 'segment_update':
            if ('segment_id' in message.data && 'integrity' in message.data) {
                store.updateSegment(message.data.segment_id, message.data);
            }
            break;

        case 'cv_detection':
            if ('segment_id' in message.data && 'corrosion_surface_pct' in message.data) {
                store.updateCV(message.data);
            }
            break;

        case 'pinn_forecast':
            if ('segment_id' in message.data && 'remaining_useful_life_days' in message.data) {
                store.updatePINN(message.data);
            }
            break;

        case 'xai_explanation':
            if ('segment_id' in message.data && 'waterfall' in message.data) {
                store.updateXAI(message.data);
            }
            break;

        default:
            console.warn('Unknown message type:', message.type);
    }
}

/**
 * Send message through WebSocket
 */
export function sendWSMessage(message: any) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
    } else {
        console.warn('WebSocket not connected, cannot send message');
    }
}

/**
 * Close WebSocket connection
 */
export function closeWebSocket() {
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }

    if (ws) {
        ws.close();
        ws = null;
    }
}
