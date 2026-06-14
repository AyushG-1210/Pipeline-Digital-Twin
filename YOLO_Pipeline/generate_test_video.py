import cv2
import numpy as np
import os

def create_synthetic_video():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "test_video.mp4")
    
    # Write a 30-frame low-light video (640x640 resolution, 10 fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 640))
    
    for frame_idx in range(30):
        # Dark background (simulating pipeline interior)
        frame = np.zeros((640, 640, 3), dtype=np.uint8) + 15
        
        # Add a faint rust/corrosion defect in the middle frames
        if 10 <= frame_idx <= 20:
            # Faint brownish polygon
            pts = np.array([[250, 250], [450, 260], [420, 450], [270, 420]], np.int32)
            cv2.fillPoly(frame, [pts], (20, 30, 80))  # BGR format
            # Faint text annotation
            cv2.putText(frame, "corrosion defect test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 40, 90), 2)
            
        out.write(frame)
        
    out.release()
    print(f"Synthetic test video created at: {video_path}")

if __name__ == "__main__":
    create_synthetic_video()
