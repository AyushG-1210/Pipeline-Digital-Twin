import { useEffect } from 'react';
import DashboardLayout from './components/Layout/DashboardLayout';
import { initWebSocket, closeWebSocket } from './utils/websocket';
import { usePipelineStore } from './store/usePipelineStore';
import './index.css';

function App() {
  useEffect(() => {
    // Initialize WebSocket connection
    initWebSocket('ws://localhost:3001');

    // Cleanup on unmount
    return () => {
      closeWebSocket();
    };
  }, []);

  // Initialize mock data on mount (will be replaced by WebSocket data)
  useEffect(() => {
    const initializeMockData = async () => {
      try {
        const response = await fetch('http://localhost:3001/api/segments');
        const data = await response.json();
        usePipelineStore.getState().setSegments(data);
      } catch (error) {
        console.error('Failed to fetch initial segment data:', error);
      }
    };

    initializeMockData();
  }, []);

  return <DashboardLayout />;
}

export default App;
