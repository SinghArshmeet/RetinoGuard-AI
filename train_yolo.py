"""
YOLOv8 Retinoblastoma Tumor Bounding-Box Detection Pipeline
===========================================================
Trains YOLOv8 spatial object detector to locate and delineate
Retinoblastoma tumor nodules, tumor borders, and retinal landmarks.

Dataset: 197 clinical images with annotated bounding boxes
Labels: 0 = 'Retina', 1 = 'Retino Blastoma'
"""

import os
import shutil
from ultralytics import YOLO
import pandas as pd

def train_retinoblastoma_yolo(
    data_yaml: str = r"Dataset\02_Tumor_Detection_YOLO_COCO\Retinoblastoma_v2_YOLOv8\data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 20,
    imgsz: int = 416,
    batch: int = 16,
    project_dir: str = "runs/detect",
    name: str = "rb_tumor_detector",
    checkpoints_dir: str = "checkpoints"
):
    print("\n" + "="*70)
    print("  STARTING YOLOV8 RETINOBLASTOMA TUMOR DETECTION TRAINING")
    print(f"  Data Config: {data_yaml}")
    print(f"  Base Model: {model_name} | Epochs: {epochs} | Img Size: {imgsz} | Batch: {batch}")
    print("="*70 + "\n")
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Initialize YOLOv8 model
    model = YOLO(model_name)
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=0,
        project=project_dir,
        name=name,
        exist_ok=True,
        plots=True,
        verbose=True
    )
    
    # Search for best weights in nested or direct project directory
    possible_paths = [
        os.path.join(project_dir, name, "weights", "best.pt"),
        os.path.join(project_dir, project_dir, name, "weights", "best.pt"),
        os.path.join("runs", "detect", name, "weights", "best.pt"),
        os.path.join("runs", "detect", "runs", "detect", name, "weights", "best.pt")
    ]
    best_weights_src = None
    for p in possible_paths:
        if os.path.exists(p):
            best_weights_src = p
            break

    dest_ckpt = os.path.join(checkpoints_dir, "rb_yolov8_best.pt")
    
    if best_weights_src and os.path.exists(best_weights_src):
        shutil.copy2(best_weights_src, dest_ckpt)
        print(f"\n[OK] Copied best weights -> {dest_ckpt}")
        
    # Validate
    val_model = YOLO(dest_ckpt if os.path.exists(dest_ckpt) else best_weights_src)
    metrics = val_model.val(data=data_yaml, imgsz=imgsz, batch=batch, workers=0)
    
    print("\n" + "="*70)
    print("YOLOV8 TUMOR DETECTION VALIDATION METRICS:")
    print("="*70)
    print(f"  Overall mAP@50:    {metrics.box.map50*100:.2f}%")
    print(f"  Overall mAP@50-95: {metrics.box.map*100:.2f}%")
    print(f"  Overall Precision: {metrics.box.mp*100:.2f}%")
    print(f"  Overall Recall:    {metrics.box.mr*100:.2f}%")
    print("="*70 + "\n")
    
    return metrics

if __name__ == "__main__":
    train_retinoblastoma_yolo()
