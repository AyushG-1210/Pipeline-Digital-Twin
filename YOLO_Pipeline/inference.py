import os
import sys
import json
import subprocess
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

def check_video_decodable(video_path):
    """Check if OpenCV can successfully decode the video path."""
    if not os.path.exists(video_path):
        return False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    cap.release()
    return ret

def find_ffmpeg():
    """Locate FFmpeg executable in system PATH or common winget paths."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return "ffmpeg"
    except FileNotFoundError:
        pass

    winget_dir = r"C:\Users\adipr\AppData\Local\Microsoft\WinGet\Packages"
    if os.path.exists(winget_dir):
        for root, dirs, files in os.walk(winget_dir):
            if "ffmpeg.exe" in files:
                ffmpeg_path = os.path.join(root, "ffmpeg.exe")
                return ffmpeg_path
    return "ffmpeg"

def transcode_video(input_path):
    """Transcode the video to a standard h.264 mp4 container using FFmpeg."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    transcoded_path = os.path.join(script_dir, "transcoded_temp.mp4")
    print(f"Video format/codec is incompatible with OpenCV. Transcoding to H.264 MP4: {transcoded_path}")
    
    ffmpeg_bin = find_ffmpeg()
    print(f"Using FFmpeg binary at: {ffmpeg_bin}")
    
    # Spawn FFmpeg subprocess to transcode
    cmd = [
        ffmpeg_bin, "-y", "-i", input_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an", # Drop audio to speed up and keep it simple
        transcoded_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("Transcoding completed successfully.")
        return transcoded_path, True
    except Exception as e:
        print(f"Warning: FFmpeg transcoding failed ({e}). Attempting fallback decoding.")
        return input_path, False

def apply_clahe_lab(frame):
    """Apply CLAHE on the Lightness channel of the LAB color space to enhance low-light defects."""
    # Convert BGR to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Create CLAHE object and apply it to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge back and convert to BGR
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_frame

def run_inference(video_path, model_path):
    print(f"Loading fine-tuned YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    
    # Determine if video is decodable by OpenCV natively
    decodable = check_video_decodable(video_path)
    video_to_process = video_path
    transcoded = False
    
    if not decodable:
        video_to_process, transcoded = transcode_video(video_path)
        
    print(f"Opening video feed: {video_to_process}")
    cap = cv2.VideoCapture(video_to_process)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_to_process}")
        sys.exit(1)
        
    best_frame_id = 0
    best_severity_score = 0.0
    best_mask_coords = []
    corrosion_detected = False
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Apply Low-Light Preprocessing (CLAHE)
        processed_frame = apply_clahe_lab(frame)
        
        # 2. Run model inference
        results = model(processed_frame, verbose=False)[0]
        
        # 3. Track instances
        if results.boxes is not None and len(results.boxes) > 0 and results.masks is not None:
            confs = results.boxes.conf.cpu().numpy()
            masks_xy = results.masks.xy
            
            for i, conf in enumerate(confs):
                if conf > best_severity_score:
                    best_severity_score = float(conf)
                    best_frame_id = frame_idx
                    corrosion_detected = True
                    # Convert float mask coordinates to list of list of integers
                    mask_pts = masks_xy[i]
                    best_mask_coords = [[int(pt[0]), int(pt[1])] for pt in mask_pts]
                    
        frame_idx += 1
        
    cap.release()
    
    # Clean up temporary transcoded file
    if transcoded and os.path.exists(video_to_process):
        try:
            os.remove(video_to_process)
            print("Cleaned up temporary transcoded video file.")
        except Exception as e:
            print(f"Warning: Could not remove temporary transcoded file: {e}")
            
    # Prepare JSON result
    insights = {
        "frame_id": best_frame_id,
        "corrosion_detected": corrosion_detected,
        "severity_score": best_severity_score,
        "mask_coordinates": best_mask_coords
    }
    
    # Save visual_insights.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_json_path = os.path.join(script_dir, "visual_insights.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)
        
    print(f"Saved visual insights JSON to: {output_json_path}")
    print(json.dumps(insights, indent=2))

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_video = os.path.join(script_dir, "test_video.mp4")
    default_model = os.path.join(script_dir, "weights", "best.pt")
    
    input_video = sys.argv[1] if len(sys.argv) > 1 else default_video
    input_model = sys.argv[2] if len(sys.argv) > 2 else default_model
    
    run_inference(input_video, input_model)
