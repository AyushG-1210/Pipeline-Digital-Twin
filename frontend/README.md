# Pipeline Integrity Digital Twin Dashboard

Production-grade MVP for Oil & Gas Pipeline Integrity Management using AI-driven monitoring with Computer Vision, Physics-Informed Neural Networks, and Explainable AI.

## 🚀 Quick Start

### Installation & Running

1. **Install dependencies:**
```bash
npm install
```

2. **Start the backend server (Terminal 1):**
```bash
npm run server
```

3. **Start the frontend (Terminal 2):**
```bash
npm run dev
```

4. **Open browser:**
Navigate to `http://localhost:5173`

## 🏗️ Architecture

### System Components

- **3D Digital Twin**: Interactive pipeline visualization with custom GLSL shaders
- **CV Inspection Panel**: YOLOv8-Seg corrosion detection with human-in-the-loop feedback
- **Analytics Panel**: PINN-based integrity forecasts using Fick's 2nd Law
- **XAI Diagnostics**: SHAP-style feature contributions and natural language explanations

### Technology Stack

**Frontend**: React 19 + TypeScript + React-Three-Fiber + Zustand + Recharts + Tailwind CSS

**Backend**: Express + WebSocket (mock server for demo)

## 📊 Features

1. **3D Digital Twin Engine** - Custom shaders, color-coded integrity heatmap, interactive segments
2. **Computer Vision Panel** - Mock video feed, polygon overlays, corrosion metrics, feedback controls
3. **Physics-Informed Analytics** - Historical vs predicted charts, RUL widget, physics-constrained forecasts
4. **Explainable AI** - Feature contributions, confidence meter, natural language explanations

## 🎯 Usage

1. Click any pipeline segment in the 3D view
2. View CV inspection data in the right panel
3. Check Analytics tab for integrity forecasts
4. Review XAI Diagnostics for prediction explanations
5. Use ✅/❌ buttons to validate CV detections

## 🔮 Future Integration

- Replace mock data with actual ML inference endpoints
- Connect to real YOLOv8-Seg, PINN, and SHAP services
- Add MQTT/Kafka for real sensor data
- Implement authentication and RBAC

---

Built for industrial digital twin applications with engineering trust and explainability at the core.
