import os
import sys
# pyrefly: ignore [missing-import]
from roboflow import Roboflow

def main():
    print("Initializing Roboflow client...")
    rf = Roboflow(api_key="")
    
    print("Accessing project...")
    workspace = rf.workspace("adityas-workspace-6hmkd")
    project = workspace.project("pipeline-ehswt-0wt5o")
    
    versions = []
    try:
        versions = project.versions()
        print(f"Retrieved versions: {versions}")
    except Exception as e:
        print(f"Error listing versions: {e}")
        
    if not versions:
        print("No versions found. Programmatically generating version 1...")
        try:
            # Added format: "Stretch to" inside resize preprocessing
            version_id = project.generate_version(settings={
                "preprocessing": {
                    "auto-orient": True,
                    "resize": {
                        "width": 640,
                        "height": 640,
                        "format": "Stretch to",
                        "enabled": True
                    }
                },
                "augmentation": {}
            })
            print(f"Successfully generated version: {version_id}")
            versions = project.versions()
        except Exception as e:
            print(f"Failed to generate version programmatically: {e}")
            sys.exit(1)
            
    if versions:
        latest_version = versions[0].version
        print(f"Using version {latest_version}")
    else:
        print("Error: Could not identify or generate a dataset version.")
        sys.exit(1)
        
    print(f"Downloading version {latest_version} in yolov8 format to ./YOLO_Pipeline/dataset...")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset"))
    
    # Download dataset
    dataset = project.version(latest_version).download("yolov8", location=dataset_path)
    print("Download completed successfully!")

if __name__ == "__main__":
    main()
