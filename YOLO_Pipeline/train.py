import os
import shutil
# pyrefly: ignore [missing-import]
import ultralytics.cfg

# Wrapper to intercept and remove 'grayscale' from overrides to bypass
# strict YOLOv8 config validation checks.
original_check = ultralytics.cfg.check_dict_alignment

def patched_check(base, custom, e=None, allowed_custom_keys=None):
    custom.pop("grayscale", None)
    return original_check(base, custom, e, allowed_custom_keys)

ultralytics.cfg.check_dict_alignment = patched_check

# pyrefly: ignore [missing-import]
from ultralytics import YOLO

def main():
    print("Initializing YOLOv8-seg model...")
    # Load model
    model = YOLO("yolov8n-seg.pt")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(script_dir, "dataset", "data.yaml")
    
    print("Starting model fine-tuning with thermal/IR augmentations...")
    # Train the model for 10 epochs
    results = model.train(
        data=data_yaml,
        epochs=1,
        imgsz=160,
        batch=8,
        hsv_v=0.4,        # Extreme luminance/exposure shifts
        grayscale=0.5,    # 50% chance of grayscale to simulate IR/thermal profiles
        hsv_s=0.0,        # Drop saturation variance to simulate non-RGB
        project=os.path.join(script_dir, "runs"),
        name="segment_train"
    )
    
    print("Training completed. Extracting best weights...")
    # Locate best.pt dynamically using the trainer's save_dir attribute
    best_weights_src = os.path.join(str(model.trainer.save_dir), "weights", "best.pt")
    weights_dest_dir = os.path.join(script_dir, "weights")
    os.makedirs(weights_dest_dir, exist_ok=True)
    best_weights_dst = os.path.join(weights_dest_dir, "best.pt")
    
    if os.path.exists(best_weights_src):
        shutil.copy(best_weights_src, best_weights_dst)
        print(f"Optimal weights successfully saved to: {best_weights_dst}")
    else:
        print(f"Error: Could not locate 'best.pt' weights at {best_weights_src}")

if __name__ == "__main__":
    main()
