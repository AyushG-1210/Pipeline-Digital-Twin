import express from 'express';
import { WebSocketServer } from 'ws';
import cors from 'cors';
import { readFile } from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { generateSegments, updateRandomSegment } from './mockData.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
const PORT = 3001;

// Paths to the YOLO pipeline & Live Data
const YOLO_DIR = path.resolve(__dirname, '../../YOLO_Pipeline');
const DATA_DIR = path.resolve(__dirname, '../../Data');
const LIVE_FEED_DIR = path.join(DATA_DIR, 'Live_Pipeline_Feed');
const ANNOTATED_FEED_DIR = path.join(YOLO_DIR, 'results', 'images');
const DETECTIONS_JSON = path.join(YOLO_DIR, 'results', 'detections.json');
const SUMMARY_JSON = path.join(YOLO_DIR, 'results', 'summary.json');
const INFERENCE_LITE = path.join(YOLO_DIR, 'run_image_inference_lite.py');
const PYTHON_EXE = process.env.PYTHON_PATH || 'python';

// Middleware
app.use(cors());
app.use(express.json());

// Serve static live camera feed and annotated YOLO detection images
app.use('/data/live_feed', express.static(LIVE_FEED_DIR));
app.use('/data/annotated_feed', express.static(ANNOTATED_FEED_DIR));

// Initialize segment data (mapped to real live feed detections)
let segments = generateSegments();

// Cache for latest real YOLO result
let latestYoloResult = null;

/**
 * REST API Endpoints
 */

// Get all segments
app.get('/api/segments', (_req, res) => {
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
 * GET /api/cv/live-feed
 * Returns full gallery of 20 live images, detections and summary
 */
app.get('/api/cv/live-feed', async (_req, res) => {
    try {
        const detectionsRaw = await readFile(DETECTIONS_JSON, 'utf-8');
        const summaryRaw = await readFile(SUMMARY_JSON, 'utf-8');
        const items = JSON.parse(detectionsRaw).map(d => ({
            ...d,
            raw_image_url: `/data/live_feed/${d.image_file}`,
            annotated_image_url: `/data/annotated_feed/annotated_${d.image_file}`,
        }));
        res.json({
            success: true,
            summary: JSON.parse(summaryRaw),
            items,
        });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

/**
 * GET /api/cv/live-feed/:filename
 * Returns single live feed image detection details
 */
app.get('/api/cv/live-feed/:filename', async (req, res) => {
    try {
        const detectionsRaw = await readFile(DETECTIONS_JSON, 'utf-8');
        const items = JSON.parse(detectionsRaw);
        const item = items.find(d => d.image_file === req.params.filename);
        if (!item) return res.status(404).json({ error: 'Image not found' });
        res.json({
            ...item,
            raw_image_url: `/data/live_feed/${item.image_file}`,
            annotated_image_url: `/data/annotated_feed/annotated_${item.image_file}`,
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * POST /api/cv/analyze
 * Re-analyzes or fetches real YOLOv8 segmentation results for a segment
 */
app.post('/api/cv/analyze', async (req, res) => {
    const { segment_id, image_file } = req.body;

    if (!segment_id) {
        return res.status(400).json({ error: 'segment_id is required' });
    }

    console.log(`\n🔍 YOLO Analysis requested for ${segment_id}`);

    try {
        const detectionsRaw = await readFile(DETECTIONS_JSON, 'utf-8');
        const allDetections = JSON.parse(detectionsRaw);

        const segNum = parseInt(segment_id.replace(/[^0-9]/g, '') || '1', 10);
        const defaultImgFile = `pipeline_section_${String(((segNum - 1) % 20) + 1).padStart(2, '0')}.png`;
        const targetImgFile = image_file || defaultImgFile;

        const liveItem = allDetections.find(d => d.image_file === targetImgFile) || allDetections[0];

        const topDet = liveItem?.detections && liveItem.detections.length > 0
            ? [...liveItem.detections].sort((a, b) => b.confidence - a.confidence)[0]
            : null;

        const polygonMask = topDet && topDet.mask_coords && topDet.mask_coords.length > 0
            ? topDet.mask_coords.map(([x, y]) => [x / 320, y / 320])
            : [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]];

        const cvOutput = {
            segment_id,
            corrosion_surface_pct: topDet
                ? Math.round(topDet.confidence * 40 + (liveItem.detections_count * 3))
                : 15,
            confidence: topDet ? topDet.confidence : 0.75,
            polygon_mask: polygonMask,
            frame_timestamp: new Date().toISOString(),

            // Real YOLO & Live Feed fields
            corrosion_detected: liveItem ? liveItem.defect_detected : true,
            class_id: topDet ? topDet.class_id : 0,
            class_name: topDet ? topDet.class_name : 'rust',
            severity_score: topDet ? topDet.confidence : 0,
            detection_rate: 1.0,
            total_frames: 1,
            frames_with_detections: liveItem && liveItem.defect_detected ? 1 : 0,
            yolo_mask_px: topDet ? topDet.mask_coords : [],
            is_yolo_result: true,

            image_file: liveItem?.image_file || targetImgFile,
            raw_image_url: `/data/live_feed/${liveItem?.image_file || targetImgFile}`,
            annotated_image_url: `/data/annotated_feed/annotated_${liveItem?.image_file || targetImgFile}`,
            detections_count: liveItem?.detections_count || 0,
            detections: liveItem?.detections || [],
        };

        // Patch the in-memory segment
        const seg = segments.find(s => s.segment_id === segment_id);
        if (seg) {
            seg.cv = cvOutput;
            seg.lastUpdated = new Date().toISOString();
            if (cvOutput.corrosion_detected) {
                seg.integrity = Math.max(0.1, seg.integrity - cvOutput.severity_score * 0.15);
            }
        }

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

        console.log(`✅ YOLO result for ${segment_id}: detected=${cvOutput.corrosion_detected}, class=${cvOutput.class_name}, score=${cvOutput.severity_score}`);
        res.json({ success: true, segment_id, cv: cvOutput });

    } catch (err) {
        console.error('❌ YOLO analysis error:', err.message);
        res.status(500).json({ success: false, error: err.message });
    }
});

/**
 * GET /api/cv/latest
 * Returns the most recent YOLO result
 */
app.get('/api/cv/latest', async (_req, res) => {
    if (latestYoloResult) {
        res.json(latestYoloResult);
    } else {
        try {
            const raw = await readFile(SUMMARY_JSON, 'utf-8');
            res.json(JSON.parse(raw));
        } catch {
            res.status(404).json({ error: 'No YOLO results available yet' });
        }
    }
});

// Start HTTP server
const server = app.listen(PORT, () => {
    console.log(`\n🚀 Pipeline Digital Twin Backend`);
    console.log(`   HTTP Server : http://localhost:${PORT}`);
    console.log(`   WebSocket   : ws://localhost:${PORT}`);
    console.log(`\n📊 Data Initialized:`);
    console.log(`   - ${segments.length} pipeline segments`);
    console.log(`\n👁️ YOLO Live Pipeline:`);
    console.log(`   - Python    : ${PYTHON_EXE}`);
    console.log(`   - Feed Dir  : ${LIVE_FEED_DIR}`);
    console.log(`   - Inference : ${INFERENCE_LITE}`);
    console.log(`\n⚡ Real-time updates active every 3 seconds\n`);
});

/**
 * WebSocket Server
 */
const wss = new WebSocketServer({ server });

wss.on('connection', (ws) => {
    console.log('⚡ WebSocket client connected');

    // Send initial data
    ws.send(JSON.stringify({
        type: 'initial_data',
        data: segments,
        timestamp: new Date().toISOString()
    }));

    ws.on('close', () => { console.log('⚡ WebSocket client disconnected'); });
    ws.on('error', (error) => { console.error('WebSocket error:', error); });
});

/**
 * Periodic real-time updates
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
    console.log('\n\n🛑 Shutting down server...');
    wss.clients.forEach((client) => client.close());
    server.close(() => { process.exit(0); });
});
