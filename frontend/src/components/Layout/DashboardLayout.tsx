import Header from './Header';
import Sidebar from './Sidebar';
import SegmentInspector from './SegmentInspector';
import PipelineScene from '../DigitalTwin/PipelineScene';
import FloatingChatbot from '../Chatbot/FloatingChatbot';

/**
 * Main dashboard layout.
 * The 3D digital twin is the single main view — selecting a segment (from
 * the twin or the sidebar list) opens the Segment Inspector drawer with the
 * PINN forecast, surface scan, and explainability as tabs rather than
 * competing panels crammed onto the same screen.
 */
export default function DashboardLayout() {
    return (
        <div className="h-screen flex flex-col bg-canvas relative">
            <Header />

            <div className="flex-1 flex overflow-hidden">
                <Sidebar />

                <div className="flex-1 p-4 overflow-hidden">
                    <div className="w-full h-full rounded-xl overflow-hidden border border-line shadow-sm">
                        <PipelineScene />
                    </div>
                </div>
            </div>

            <SegmentInspector />
            <FloatingChatbot />
        </div>
    );
}
