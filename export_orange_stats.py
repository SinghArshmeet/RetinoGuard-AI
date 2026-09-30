"""
RetinoGuard AI - Orange Data Mining Batch Statistics & Predictions Exporter
===========================================================================
Runs multi-modal deep learning inference across a directory of fundus images
and generates Orange Data Mining compatible (.tab and .csv) datasets.

Usage:
  python export_orange_stats.py
  python export_orange_stats.py --input-dir demo_samples
"""

import os
import glob
import argparse
import cv2
import pandas as pd
from predict import ClinicalDiagnosticSuite

def main():
    parser = argparse.ArgumentParser(description="Export RetinoGuard AI Predictions for Orange Data Mining")
    parser.add_argument("--input-dir", type=str, default="demo_samples", help="Directory of test images")
    parser.add_argument("--output-dir", type=str, default="results/orange", help="Output directory for Orange datasets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    suite = ClinicalDiagnosticSuite()
    
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"[!] No images found in {args.input_dir}")
        return

    print(f"[*] Found {len(image_paths)} images in {args.input_dir}. Running clinical inference suite...")
    
    records = []
    for idx, path in enumerate(image_paths, 1):
        filename = os.path.basename(path)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        # Inferred ground truth heuristic from filename for demo cases
        fname_lower = filename.lower()
        if "retinoblastoma" in fname_lower or "rb" in fname_lower:
            actual = "Retinoblastoma"
        elif "cataract" in fname_lower:
            actual = "Pediatric Cataract"
        elif "rop" in fname_lower:
            actual = "Pediatric ROP"
        elif "normal" in fname_lower:
            actual = "Normal Control"
        else:
            actual = "Unknown / Clinical Test"

        # 1. Primary ResNet-CBAM Ensemble
        p_res = suite.run_primary_detection(img_bgr)
        # 2. YOLOv8
        y_dets = suite.run_yolo_detection(img_bgr)
        # 3. Attention U-Net
        s_res = suite.run_segmentation(img_bgr)
        # 4. ICRB Staging
        stage_res = suite.staging_engine.evaluate_stage(
            tumor_detections=y_dets,
            has_vitreous_seeds=s_res.get("has_vitreous_seeds", False),
            is_confirmed_malignant=p_res["is_retinoblastoma"]
        )

        p_rb = round(p_res["rb_confidence"] / 100.0, 4)
        p_diff = round(p_res["top_differential_prob"] / 100.0, 4)
        pred_condition = p_res["top_differential"].split(" (")[0]
        verdict = "POSITIVE" if p_res["is_retinoblastoma"] else "NEGATIVE"
        icrb_group = stage_res.group if (p_res["is_retinoblastoma"] and stage_res) else "Clear (Non-Neoplasm)"
        risk = stage_res.risk_level if (p_res["is_retinoblastoma"] and stage_res) else "Benign / Normal"

        records.append({
            "Image": filename,
            "Actual_Class": actual,
            "Predicted_Class": pred_condition,
            "Malignancy_Verdict": verdict,
            "P_Retinoblastoma": p_rb,
            "P_Differential": p_diff,
            "Certainty_Score": p_res["uncertainty"]["reliability_score"],
            "Entropy": p_res["uncertainty"]["entropy"],
            "Tumor_Count": len(y_dets),
            "Tumor_Area_Pct": round(s_res.get("tumor_pixel_pct", 0.0), 2),
            "ICRB_Stage": icrb_group,
            "Risk_Category": risk
        })

    df = pd.DataFrame(records)
    
    # Save standard CSV
    csv_path = os.path.join(args.output_dir, "model_predictions_eval.csv")
    df.to_csv(csv_path, index=False)
    
    # Save native Orange 3-row header TAB file
    tab_path = os.path.join(args.output_dir, "model_predictions_eval.tab")
    with open(tab_path, "w", encoding="utf-8") as f:
        # Header 1: Column Names
        f.write("\t".join(df.columns) + "\n")
        # Header 2: Data Types
        types = [
            "string", "discrete", "discrete", "discrete",
            "continuous", "continuous", "continuous", "continuous",
            "continuous", "continuous", "discrete", "discrete"
        ]
        f.write("\t".join(types) + "\n")
        # Header 3: Orange Roles (meta, class/target, feature)
        roles = [
            "meta", "class", "feature", "feature",
            "feature", "feature", "feature", "feature",
            "feature", "feature", "feature", "feature"
        ]
        f.write("\t".join(roles) + "\n")
        # Data rows
        for _, row in df.iterrows():
            f.write("\t".join(str(val) for val in row.values) + "\n")

    print("\n" + "="*70)
    print("  ORANGE DATASETS SUCCESSFULLY EXPORTED")
    print("="*70)
    print(f" -> CSV file : {csv_path}")
    print(f" -> TAB file : {tab_path}")
    print(f" Total Samples Processed: {len(records)}")
    print(df[["Image", "Actual_Class", "Predicted_Class", "P_Retinoblastoma", "ICRB_Stage"]].to_string())

if __name__ == "__main__":
    main()
