// Backend location. Set VITE_API_URL (and optionally VITE_WS_URL) at build time for deployment;
// both fall back to the local dev server.
const DEFAULT_API_URL = 'http://localhost:3001';

export const API_URL: string = (import.meta.env.VITE_API_URL ?? DEFAULT_API_URL).replace(/\/+$/, '');

// If VITE_WS_URL isn't set, derive it from the API URL (http -> ws, https -> wss).
export const WS_URL: string = (import.meta.env.VITE_WS_URL ?? API_URL.replace(/^http/, 'ws')).replace(/\/+$/, '');
