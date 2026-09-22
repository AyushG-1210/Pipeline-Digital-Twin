import express from 'express';
import { WebSocketServer } from 'ws';
import cors from 'cors';
import { spawn } from 'child_process';
import { readFile } from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { generateSegments, updateRandomSegment } from './mockData.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
const PORT = 3001;

// Paths to the YOLO pipeline (sibling repo)
const YOLO_DIR = path.resolve(__dirname, '../../Pipeline-Digital-Twin/YOLO_Pipeline');
const PYTHON_EXE = path.join(YOLO_DIR, 'venv', 'Scripts', 'python.exe');
const INFERENCE_SCRIPT = path.join(YOLO_DIR, 'inference.py');
const TEST_VIDEO = path.join(YOLO_DIR, 'test_video.mp4');
const INSIGHTS_JSON = path.join(YOLO_DIR, 'visual_insights.json');

// Middleware
app.use(cors());
app.use(express.json());

// Initialize segment data
let segments = generateSegments();

// Cache for latest real YOLO result
let latestYoloResult = null;

/**
 * REST API Endpoints
 */

// Get all segments
app.get('/api/segments', (req, res) => {
    res.json(segments);
});

// Get specific segment
app.get('/api/segments/:id', (req, res) => {
    const segment = segments.find(s => s.segment_id === req.params.id);
    if (segment) {
        res.json(segment);
    } else {
        res.status(404).json({ error: 'Segment not found' });
    }
});

/**
 * POST /api/cv/analyze
 * Spawns inference.py against test_video.mp4 (or a provided video_path),
 * reads visual_insights.json, maps it to CVOutput, patches the segment,
 * and broadcasts cv_detection over WebSocket.
 */
app.post('/api/cv/analyze', async (req, res) => {
    const { segment_id, video_path } = req.body;

    if (!segment_id) {
        return res.status(400).json({ error: 'segment_id is required' });
    }

    const videoToUse = video_path || TEST_VIDEO;
    console.log(`\n?? YOLO Analysis requested for ${segment_id}`);
    console.log(`   Video : ${videoToUse}`);
    console.log(`   Script: ${INFERENCE_SCRIPT}`);

    try {
        // Run inference.py as a child process
        await new Promise((resolve, reject) => {
            const proc = spawn(PYTHON_EXE, [INFERENCE_SCRIPT, videoToUse], {
                cwd: YOLO_DIR,
            });

            let stderr = '';
            proc.stderr.on('data', (d) => { stderr += d.toString(); });
            proc.stdout.on('data', (d) => { process.stdout.write('[YOLO] ' + d.toString()); });

            proc.on('close', (code) => {
                if (code === 0) resolve();
                else reject(new Error(`inference.py exited with code ${code}\n${stderr}`));
            });

            proc.on('error', reject);
        });

        // Read visual_insights.json output
        const raw = await readFile(INSIGHTS_JSON, 'utf-8');
        const insights = JSON.parse(raw);

        // Map YOLO output ? CVOutput schema
        const cvOutput = {
            segment_id,
            corrosion_surface_pct: insights.corrosion_detected
                ? Math.round(insights.severity_score * 100)
                : 0,
            confidence: insights.severity_score || 0,
            polygon_mask: insights.mask_coordinates.length > 0
                ? insights.mask_coordinates.map(([x, y]) => [x / 640, y / 640]) // normalize to 0-1
                : [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
            frame_timestamp: new Date().toISOString(),

            // Real YOLO fields
            corrosion_detected: insights.corrosion_detected,
            class_id: insights.class_id,
            class_name: insights.class_name,
            severity_score: insights.severity_score,
            detection_rate: insights.stats?.detection_rate ?? 0,
            total_frames: insights.stats?.total_frames ?? 0,
            frames_with_detections: insights.stats?.frames_with_detections ?? 0,
            yolo_mask_px: insights.mask_coordinates,
            is_yolo_result: true,
        };

        // Patch the in-memory segment
        const seg = segments.find(s => s.segment_id === segment_id);
        if (seg) {
            seg.cv = cvOutput;
            seg.lastUpdated = new Date().toISOString();
            // Adjust integrity based on detection severity
            if (insights.corrosion_detected) {
                seg.integrity = Math.max(0.1, seg.integrity - insights.severity_score * 0.15);
            }
        }

        // Cache result
        latestYoloResult = cvOutput;

        // Broadcast to all WebSocket clients
        const wsMessage = JSON.stringify({
            type: 'cv_detection',
            data: cvOutput,
            timestamp: new Date().toISOString(),
        });
        wss.clients.forEach((client) => {
            if (client.readyState === 1) client.send(wsMessage);
        });

        console.log(`? YOLO result for ${segment_id}: detected=${insights.corrosion_detected}, class=${insights.class_name}, score=${insights.severity_score}`);
        res.json({ success: true, segment_id, cv: cvOutput });

    } catch (err) {
        console.error('? YOLO inference error:', err.message);
        res.status(500).json({ success: false, error: err.message });
    }
});

/**
 * GET /api/cv/latest
 * Returns the most recent YOLO visual_insights.json result.
 */
app.get('/api/cv/latest', async (_req, res) => {
    try {
        const raw = await readFile(INSIGHTS_JSON, 'utf-8');
        res.json(JSON.parse(raw));
    } catch {
        res.status(404).json({ error: 'No YOLO results available yet' });
    }
});

// Start HTTP server
const server = app.listen(PORT, () => {
    console.log(`\n?? Pipeline Digital Twin Backend`);
    console.log(`   HTTP Server : http://localhost:${PORT}`);
    console.log(`   WebSocket   : ws://localhost:${PORT}`);
    console.log(`\n?? Mock Data Initialized:`);
    console.log(`   - ${segments.length} pipeline segments`);
    console.log(`\n?? YOLO Pipeline:`);
    console.log(`   - Python  : ${PYTHON_EXE}`);
    console.log(`   - Script  : ${INFERENCE_SCRIPT}`);
    console.log(`   - Video   : ${TEST_VIDEO}`);
    console.log(`\n?  Real-time mock updates every 3 seconds\n`);
});

/**
 * WebSocket Server
 */
const wss = new WebSocketServer({ server });

wss.on('connection', (ws) => {
    console.log('?? WebSocket client connected');

    // Send initial data
    ws.send(JSON.stringify({
        type: 'initial_data',
        data: segments,
        timestamp: new Date().toISOString()
    }));

    ws.on('close', () => { console.log('?? WebSocket client disconnected'); });
    ws.on('error', (error) => { console.error('WebSocket error:', error); });
});

/**
 * Simulate real-time mock updates every 3 seconds
 */
setInterval(() => {
    const updatedSegment = updateRandomSegment(segments);

    const message = {
        type: 'segment_update',
        data: updatedSegment,
        timestamp: new Date().toISOString()
    };

    wss.clients.forEach((client) => {
        if (client.readyState === 1) client.send(JSON.stringify(message));
    });
}, 3000);

/**
 * Graceful shutdown
 */
process.on('SIGINT', () => {
    console.log('\n\n?? Shutting down server...');
    wss.clients.forEach((client) => client.close());
    server.close(() => { process.exit(0); });
});
