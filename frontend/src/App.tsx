import { useEffect } from 'react';
import DashboardLayout from './components/Layout/DashboardLayout';
import { initWebSocket, closeWebSocket } from './utils/websocket';
import { usePipelineStore } from './store/usePipelineStore';
import { API_URL, WS_URL } from './config';
import './index.css';

function App() {
  useEffect(() => {
    // Initialize WebSocket connection
    initWebSocket(WS_URL);

    // Cleanup on unmount
    return () => {
      closeWebSocket();
    };
  }, []);

  // Load initial segments over HTTP. Free-tier hosts put the API to sleep, so the first
  // request(s) can fail while it wakes up — retry until it succeeds. (The WebSocket
  // 'initial_data' message is a second path that fills the same store.)
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const load = async (attempt: number) => {
      try {
        const response = await fetch(`${API_URL}/api/segments`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!cancelled) usePipelineStore.getState().setSegments(data);
      } catch (error) {
        if (cancelled) return;
        console.error(`Failed to fetch initial segment data (attempt ${attempt}):`, error);
        // Stop retrying once the WebSocket has delivered the data.
        if (usePipelineStore.getState().segments.size > 0) return;
        timer = setTimeout(() => load(attempt + 1), Math.min(2000 * attempt, 10000));
      }
    };

    load(1);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  return <DashboardLayout />;
}

export default App;
