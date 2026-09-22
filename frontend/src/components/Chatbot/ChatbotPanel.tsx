import { useState, useRef, useEffect } from 'react';
import { usePipelineStore } from '../../store/usePipelineStore';
import { formatPercentage } from '../../utils/colors';

interface Message {
    id: string;
    sender: 'user' | 'assistant';
    text: string;
    timestamp: string;
    segmentId?: string;
}

/**
 * AI Assistant Chatbot Panel Component
 * Provides interactive natural language diagnostics, telemetry querying, and physics insights
 */
export default function ChatbotPanel() {
    const segments = usePipelineStore((state) => state.segments);
    const selectedSegmentId = usePipelineStore((state) => state.selectedSegmentId);
    const selectSegment = usePipelineStore((state) => state.selectSegment);

    const [inputQuery, setInputQuery] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const segmentArray = Array.from(segments.values());
    const criticalSegments = segmentArray.filter((s) => s.integrity < 0.3);
    const warningSegments = segmentArray.filter((s) => s.integrity >= 0.3 && s.integrity < 0.6);

    // Initial conversation history
    const [messages, setMessages] = useState<Message[]>([
        {
            id: 'init-1',
            sender: 'assistant',
            text: `Hello! I am your Pipeline Digital Twin AI Assistant. I have live access to telemetry from ${segments.size || 10} pipeline segments, YOLOv8-Seg external surface corrosion data, and PINN physics forecasts. How can I help you today?`,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
    ]);

    // Auto-scroll chat to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isTyping]);

    // Generate intelligent AI response based on query keywords and live store data
    const generateAIResponse = (query: string): string => {
        const q = query.toLowerCase();

        // 1. Direct segment lookup (e.g. SEG-003 or SEG-001)
        const matchedSeg = segmentArray.find(s => q.includes(s.segment_id.toLowerCase()));
        if (matchedSeg) {
            const extCorrosion = matchedSeg.cv ? `${matchedSeg.cv.corrosion_surface_pct.toFixed(1)}%` : 'N/A';
            const rulDays = matchedSeg.pinn ? `${matchedSeg.pinn.remaining_useful_life_days} days` : 'N/A';
            const statusLabel = matchedSeg.integrity < 0.3 ? 'CRITICAL 🚨' : matchedSeg.integrity < 0.6 ? 'WARNING ⚠️' : 'HEALTHY ✅';

            return `**${matchedSeg.segment_id} Telemetry Report:**\n` +
                `• **Status:** ${statusLabel}\n` +
                `• **Integrity Score:** ${formatPercentage(matchedSeg.integrity)}\n` +
                `• **External Surface Corrosion:** ${extCorrosion}\n` +
                `• **Remaining Useful Life (RUL):** ${rulDays}\n` +
                (matchedSeg.cv?.confidence ? `• **YOLOv8 Detection Confidence:** ${(matchedSeg.cv.confidence * 100).toFixed(0)}%\n` : '') +
                `Would you like me to highlight ${matchedSeg.segment_id} on the 3D twin?`;
        }

        // 2. Critical or high-risk segments inquiry
        if (q.includes('critical') || q.includes('defect') || q.includes('risk') || q.includes('damage') || q.includes('worst')) {
            if (criticalSegments.length === 0) {
                return `All ${segments.size} active pipeline segments are currently above the critical threshold. There are ${warningSegments.length} segments in the Warning range that require routine monitoring.`;
            }
            const listStr = criticalSegments.map(s => `**${s.segment_id}** (${formatPercentage(s.integrity)} integrity, ${s.pinn?.remaining_useful_life_days || 0} days RUL)`).join('\n• ');
            return `I identified **${criticalSegments.length} Critical Segment(s)** requiring urgent attention:\n• ${listStr}\n\nRecommended Action: Schedule immediate external ultrasonic thickness gauging and surface coating repair.`;
        }

        // 3. Physics / RUL governing model inquiry
        if (q.includes('pinn') || q.includes('fick') || q.includes('physics') || q.includes('equation') || q.includes('model') || q.includes('rul')) {
            return `The subsurface degradation forecast uses **Fick's 2nd Law of Diffusion**:\n\n` +
                `$$\\frac{\\partial C}{\\partial t} = D \\frac{\\partial^2 C}{\\partial x^2}$$\n\n` +
                `Our Physics-Informed Neural Network (PINN) embeds these mass transport physics into loss functions, ensuring predictions obey physical conservation laws rather than extrapolating blindly from noisy sensor data.`;
        }

        // 4. Maintenance recommendation inquiry
        if (q.includes('maintenance') || q.includes('schedule') || q.includes('repair') || q.includes('recommend')) {
            const sortedByRul = [...segmentArray].sort((a, b) => (a.pinn?.remaining_useful_life_days || 999) - (b.pinn?.remaining_useful_life_days || 999));
            const topUrgent = sortedByRul[0];

            return `**Priority Maintenance Schedule:**\n` +
                `1. **${topUrgent?.segment_id || 'SEG-003'}**: Priority 1 - ${topUrgent?.pinn?.remaining_useful_life_days || 30} Days RUL Remaining (External pitting detected).\n` +
                `2. **General External Coating Touchup**: Recommended for segments with external corrosion surface area > 10%.\n\n` +
                `Would you like me to flag these segments in the feedback queue?`;
        }

        // Default response
        return `I evaluated live telemetry across all active pipeline segments. Overall pipeline health is monitored via optical surface vision and PINN physics models. Try asking:\n` +
            `• *"What is the status of SEG-003?"*\n` +
            `• *"List all critical external surface defects"*\n` +
            `• *"Explain Fick's 2nd Law PINN forecast"*\n` +
            `• *"Recommend maintenance priorities"*`;
    };

    // Submit handler
    const handleSend = (textToSend?: string) => {
        const query = (textToSend || inputQuery).trim();
        if (!query) return;

        const userMsg: Message = {
            id: `usr-${Date.now()}`,
            sender: 'user',
            text: query,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };

        setMessages((prev) => [...prev, userMsg]);
        if (!textToSend) setInputQuery('');
        setIsTyping(true);

        // Check if query selects a segment
        const matchedSeg = segmentArray.find(s => query.toLowerCase().includes(s.segment_id.toLowerCase()));
        if (matchedSeg) {
            selectSegment(matchedSeg.segment_id);
        }

        // Simulate AI thinking delay
        setTimeout(() => {
            const aiResponseText = generateAIResponse(query);
            const aiMsg: Message = {
                id: `ai-${Date.now()}`,
                sender: 'assistant',
                text: aiResponseText,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            };
            setMessages((prev) => [...prev, aiMsg]);
            setIsTyping(false);
        }, 600);
    };

    const promptPills = [
        'Inspect SEG-003 external defect',
        'List critical pipeline segments',
        'Explain Fick\'s 2nd Law PINN model',
        'Recommend maintenance schedule',
    ];

    return (
        <div className="panel h-full flex flex-col bg-industrial-950 border border-industrial-800 rounded-lg overflow-hidden">
            {/* Panel Header */}
            <div className="panel-header flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span className="text-xs font-semibold uppercase tracking-wider text-zinc-100">
                        Pipeline Integrity AI Assistant
                    </span>
                </div>
                <button
                    onClick={() => setMessages([messages[0]])}
                    className="text-[10px] text-zinc-400 hover:text-zinc-200 border border-industrial-700 hover:border-industrial-600 px-2 py-0.5 rounded transition-colors"
                    title="Clear conversation"
                >
                    Reset Chat
                </button>
            </div>

            {/* Chat Messages Body */}
            <div className="flex-1 p-3 overflow-y-auto space-y-3 font-sans">
                {messages.map((msg) => (
                    <div
                        key={msg.id}
                        className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
                    >
                        <div className="flex items-center gap-1.5 mb-1">
                            <span className="text-[10px] text-zinc-500 font-mono">
                                {msg.sender === 'user' ? 'Operator' : 'AI Assistant'} • {msg.timestamp}
                            </span>
                        </div>
                        <div
                            className={`max-w-[90%] px-3.5 py-2.5 rounded-lg text-xs leading-relaxed shadow-md ${
                                msg.sender === 'user'
                                    ? 'bg-industrial-800 text-zinc-100 border border-industrial-700 rounded-br-none'
                                    : 'bg-industrial-900 text-zinc-200 border border-industrial-800 rounded-bl-none'
                            }`}
                        >
                            {msg.text.split('\n').map((line, idx) => (
                                <p key={idx} className={idx > 0 ? 'mt-1' : ''}>
                                    {line}
                                </p>
                            ))}
                        </div>
                    </div>
                ))}

                {/* AI Typing Indicator */}
                {isTyping && (
                    <div className="flex items-center gap-2 text-zinc-400 text-xs italic pl-1">
                        <div className="flex gap-1">
                            <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-bounce"></span>
                            <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                            <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                        </div>
                        <span>Evaluating telemetry models...</span>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Quick Suggestion Pills */}
            <div className="px-3 py-1.5 bg-industrial-900/60 border-t border-industrial-800 flex gap-1.5 overflow-x-auto scrollbar-none">
                {promptPills.map((pill, i) => (
                    <button
                        key={i}
                        onClick={() => handleSend(pill)}
                        className="shrink-0 text-[10px] text-zinc-300 bg-industrial-850 hover:bg-industrial-800 border border-industrial-750 hover:border-industrial-600 px-2 py-1 rounded-full transition-colors truncate max-w-[200px]"
                    >
                        ⚡ {pill}
                    </button>
                ))}
            </div>

            {/* Input Bar */}
            <form
                onSubmit={(e) => {
                    e.preventDefault();
                    handleSend();
                }}
                className="p-2.5 bg-industrial-900 border-t border-industrial-800 flex gap-2 items-center"
            >
                <input
                    type="text"
                    value={inputQuery}
                    onChange={(e) => setInputQuery(e.target.value)}
                    placeholder="Ask AI assistant about segment status, RUL forecasts..."
                    className="flex-1 bg-industrial-950 border border-industrial-750 focus:border-emerald-500/60 text-zinc-100 placeholder-zinc-500 text-xs rounded-md px-3 py-2 outline-none transition-colors"
                />
                <button
                    type="submit"
                    disabled={!inputQuery.trim()}
                    className="px-3.5 py-2 bg-industrial-800 hover:bg-industrial-750 disabled:opacity-40 text-emerald-400 border border-industrial-700 font-semibold text-xs rounded-md transition-colors flex items-center gap-1"
                >
                    <span>Send</span>
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                </button>
            </form>
        </div>
    );
}
