"""
Clinical Inference & Automated Multi-Modal Retinoblastoma Diagnostic Suite
==========================================================================
Integrates all clinical deep learning modules for end-to-end diagnosis:
1. Primary Retinoblastoma Detector Ensemble (ResNet-CBAM across 5-folds)
2. 8-Class Pediatric Ophthalmic Differential Diagnosis
3. YOLOv8 Spatial Tumor Bounding-Box Detection & Localization
4. Attention U-Net Pixel-Level Boundary Segmentation & Seeding Analysis
5. Automated ICRB Clinical Staging Engine (Murphree Groups A through E)
6. Multi-Panel Medical Diagnostic Report & Visual Overlay Generation

Usage:
  python predict.py --image path/to/fundus.jpg
  python predict.py --image path/to/fundus.jpg --visualize --save-report
"""

import argparse
import os
import glob
import cv2
import torch
import numpy as np
from typing import Dict, List, Optional

from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.retinoblastoma_detector import RetinoblastomaPrimaryDetector
from src.models.attention_unet import AttentionUNet
from src.models.icrb_staging import ICRBStagingEngine, ICRBStageResult

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

CLASS_NAMES = [
    "Retinoblastoma (Intraocular Malignancy)",
    "Pediatric Cataract (Leukocoria Differential)",
    "Pediatric ROP (Retinopathy of Prematurity)",
    "Pediatric Normal RetCam (Healthy Retina)",
    "Retinal Capillary Hemangioma (RCH)",
    "Uveal Melanoma (UM)",
    "Choroidal Osteoma (CO)",
    "Choroidal Hemangioma (CH)"
]

class ClinicalDiagnosticSuite:
    """
    Unified multi-modal inference pipeline for Retinoblastoma detection,
    differential diagnosis, lesion localization, segmentation, and ICRB staging.
    """
    def __init__(
        self,
        checkpoints_dir: str = "checkpoints",
        backbone_name: str = "resnet_cbam",
        device: Optional[str] = None
    ):
        self.checkpoints_dir = checkpoints_dir
        self.backbone_name = backbone_name
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pipeline = PreprocessingPipeline(target_size=256)
        self.staging_engine = ICRBStagingEngine()
        
        # Discover primary detector checkpoints
        ckpt_pattern = os.path.join(checkpoints_dir, f"rb_detector_{backbone_name}_best_fold_*.pt")
        self.detector_ckpts = sorted(glob.glob(ckpt_pattern))
        if not self.detector_ckpts:
            default_ckpt = os.path.join(checkpoints_dir, f"rb_detector_{backbone_name}_best_fold_0.pt")
            if os.path.exists(default_ckpt):
                self.detector_ckpts = [default_ckpt]
                
        # Discover YOLOv8 tumor detector checkpoint
        self.yolo_ckpt = os.path.join(checkpoints_dir, "rb_yolov8_best.pt")
        if not os.path.exists(self.yolo_ckpt):
            # Fallback to runs/detect
            fallback_yolo = os.path.join("runs", "detect", "rb_tumor_detector", "weights", "best.pt")
            if os.path.exists(fallback_yolo):
                self.yolo_ckpt = fallback_yolo

        # Discover Attention U-Net checkpoint
        self.unet_ckpt = os.path.join(checkpoints_dir, "rb_attention_unet_best.pt")

    def run_primary_detection(self, img_bgr: np.ndarray) -> Dict:
        """Runs ensemble inference across trained fold models for primary detection."""
        if not self.detector_ckpts:
            raise FileNotFoundError("No trained primary detector checkpoints found in checkpoints directory.")

        tensor = self.pipeline(img_bgr).unsqueeze(0).to(self.device)
        
        all_rb_probs = []
        all_diff_probs = []

        for ckpt in self.detector_ckpts:
            model = RetinoblastomaPrimaryDetector(backbone_name=self.backbone_name, num_differential_classes=8, pretrained=False)
            try:
                state_dict = torch.load(ckpt, map_location=self.device, weights_only=True)
            except Exception:
                state_dict = torch.load(ckpt, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            with torch.no_grad():
                out = model(tensor)
                rb_p = torch.softmax(out["rb_logits"], dim=1)[0, 1].item()
                diff_p = torch.softmax(out["diff_logits"], dim=1)[0].cpu().numpy()
                all_rb_probs.append(rb_p)
                all_diff_probs.append(diff_p)

        mean_rb_prob = float(np.mean(all_rb_probs))
        mean_diff_probs = np.mean(all_diff_probs, axis=0)
        pred_diff_idx = int(np.argmax(mean_diff_probs))

        is_rb = mean_rb_prob >= 0.45  # Clinical high-sensitivity threshold

        return {
            "is_retinoblastoma": is_rb,
            "rb_confidence": mean_rb_prob * 100.0,
            "num_models_ensembled": len(self.detector_ckpts),
            "top_differential": CLASS_NAMES[pred_diff_idx],
            "top_differential_prob": float(mean_diff_probs[pred_diff_idx]) * 100.0,
            "all_differential_probs": mean_diff_probs.tolist()
        }

    def run_yolo_detection(self, img_bgr: np.ndarray, conf_thresh: float = 0.10) -> List[Dict]:
        """Runs YOLOv8 bounding-box tumor localization."""
        if not YOLO_AVAILABLE or not os.path.exists(self.yolo_ckpt):
            return []

        yolo = YOLO(self.yolo_ckpt)
        results = yolo(img_bgr, conf=conf_thresh, verbose=False)[0]
        
        detections = []
        h, w = img_bgr.shape[:2]
        
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cname = results.names.get(cls_id, str(cls_id))
            
            # Normalized coordinates [x1, y1, x2, y2]
            norm_box = (float(coords[0]/w), float(coords[1]/h), float(coords[2]/w), float(coords[3]/h))
            area_pct = float((abs(norm_box[2] - norm_box[0]) * abs(norm_box[3] - norm_box[1])) * 100.0)
            
            detections.append({
                "box": norm_box,
                "box_pixels": [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                "confidence": float(conf),
                "class_id": int(cls_id),
                "class_name": str(cname),
                "area_pct": float(area_pct)
            })
            
        return detections

    def run_segmentation(self, img_bgr: np.ndarray) -> Dict:
        """Runs Attention U-Net pixel-level boundary segmentation."""
        if not os.path.exists(self.unet_ckpt):
            return {"mask": None, "tumor_pixel_pct": 0.0, "has_vitreous_seeds": False}

        unet = AttentionUNet(in_channels=3, num_classes=1).to(self.device)
        try:
            state_dict = torch.load(self.unet_ckpt, map_location=self.device, weights_only=True)
        except Exception:
            state_dict = torch.load(self.unet_ckpt, map_location=self.device)
        unet.load_state_dict(state_dict)
        unet.eval()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]
        img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            logits = unet(img_tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        bin_mask_256 = (probs > 0.5).astype(np.uint8)
        tumor_pixel_pct = float(np.mean(bin_mask_256) * 100.0)

        # Detect satellite seeds via connected components
        num_labels, labels_im = cv2.connectedComponents(bin_mask_256)
        # Multiple small disconnected components indicate vitreous or subretinal seeds
        has_seeds = (num_labels - 1) >= 3

        bin_mask_orig = cv2.resize(bin_mask_256, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        return {
            "mask": bin_mask_orig,
            "mask_probs": probs,
            "tumor_pixel_pct": tumor_pixel_pct,
            "connected_lesions": num_labels - 1,
            "has_vitreous_seeds": has_seeds
        }

    def generate_diagnostic_overlay(
        self,
        img_bgr: np.ndarray,
        primary_res: Dict,
        yolo_dets: List[Dict],
        seg_res: Dict,
        stage_res: ICRBStageResult
    ) -> np.ndarray:
        """Draws an integrated clinical diagnostic panel on the fundus image."""
        vis = img_bgr.copy()
        h, w = vis.shape[:2]

        # 1. Overlay Attention U-Net segmentation mask (Crimson / Magenta translucent overlay)
        if seg_res.get("mask") is not None:
            mask = seg_res["mask"]
            color_mask = np.zeros_like(vis)
            color_mask[mask > 0] = [0, 0, 230]  # Red overlay
            vis = cv2.addWeighted(vis, 0.75, color_mask, 0.25, 0)
            
            # Draw contour around tumor boundaries
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)  # Yellow contour

        # 2. Draw YOLOv8 tumor bounding boxes
        for det in yolo_dets:
            x1, y1, x2, y2 = det["box_pixels"]
            cname = det["class_name"]
            conf = det["confidence"]
            color = (0, 0, 255) if "blast" in cname.lower() or det.get("class_id") == 1 else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{cname}: {conf*100:.1f}%"
            cv2.putText(vis, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 3. Create top banner for Primary Verdict and ICRB Stage
        banner_h = 100
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        if primary_res["is_retinoblastoma"]:
            banner[:] = (15, 15, 120)  # Dark Red
            cv2.putText(banner, f"PRIMARY VERDICT: RETINOBLASTOMA DETECTED ({primary_res['rb_confidence']:.1f}% CONF)",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(banner, f"ICRB STAGE: {stage_res.group} | Risk: {stage_res.risk_level} | Est Diam: {stage_res.estimated_diameter_mm}mm",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(banner, f"Diff: {primary_res['top_differential']} ({primary_res['top_differential_prob']:.1f}%) | Mask Area: {seg_res.get('tumor_pixel_pct', 0.0):.1f}%",
                        (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        else:
            banner[:] = (20, 100, 20)  # Dark Green
            cv2.putText(banner, f"PRIMARY VERDICT: NEGATIVE / NON-MALIGNANT ({100-primary_res['rb_confidence']:.1f}% CONF)",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(banner, f"Top Differential: {primary_res['top_differential']} ({primary_res['top_differential_prob']:.1f}%)",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

        output_panel = np.vstack([banner, vis])
        return output_panel

    def diagnose_image(self, image_path: str, save_output: bool = True, output_dir: str = "results/predictions") -> Dict:
        """Full end-to-end multi-modal diagnostic pipeline."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Failed to read image at: {image_path}")

        # 1. Primary Classification & Differential
        primary_res = self.run_primary_detection(img_bgr)

        # 2. YOLOv8 Tumor Localization
        yolo_dets = self.run_yolo_detection(img_bgr)

        # 3. Attention U-Net Boundary Segmentation
        seg_res = self.run_segmentation(img_bgr)

        # 4. Automated ICRB Clinical Staging
        stage_res = self.staging_engine.evaluate_stage(
            tumor_detections=yolo_dets,
            has_vitreous_seeds=seg_res.get("has_vitreous_seeds", False)
        )

        # 5. Output Diagnostic Summary to Console
        print("\n" + "="*75)
        print("          PEDIATRIC RETINOBLASTOMA CLINICAL AI DIAGNOSTIC REPORT")
        print("="*75)
        print(f"Target Image: {os.path.basename(image_path)}")
        print(f"Ensemble Models Loaded: {primary_res['num_models_ensembled']} fold(s)")
        print("-" * 75)
        
        if primary_res["is_retinoblastoma"]:
            print(f">> [POSITIVE] RETINOBLASTOMA DETECTED")
            print(f"   Malignancy Confidence : {primary_res['rb_confidence']:.2f}%")
            print(f"   ICRB Clinical Stage   : {stage_res.group}")
            print(f"   Risk Category         : {stage_res.risk_level}")
            print(f"   Est. Max Diameter     : {stage_res.estimated_diameter_mm} mm")
            print(f"   Tumor Area (Mask)     : {seg_res.get('tumor_pixel_pct', 0.0):.2f}% of retina")
            print(f"   Vitreous Seeds Sign   : {'YES (Satellite clusters detected)' if stage_res.seeding_detected else 'No significant seeding'}")
            print(f"   Detected Tumors (YOLO): {len(yolo_dets)} lesion(s)")
            print(f"\n>> CLINICAL RATIONALE:\n   {stage_res.clinical_rationale}")
            print(f"\n>> ONCOLOGIC TREATMENT RECOMMENDATIONS:")
            for rec in stage_res.treatment_recommendations:
                print(f"   - {rec}")
        else:
            print(f">> [NEGATIVE] NO RETINOBLASTOMA DETECTED")
            print(f"   Non-Malignant Probability: {100.0 - primary_res['rb_confidence']:.2f}%")
            print(f"   Top Differential Match   : {primary_res['top_differential']} ({primary_res['top_differential_prob']:.2f}%)")
            print("   Clinical Priority        : Standard Differential Management")

        print("-" * 75)
        print("DIFFERENTIAL DIAGNOSIS SPECTRUM:")
        for idx, (cname, prob) in enumerate(zip(CLASS_NAMES, primary_res["all_differential_probs"])):
            indicator = " <-- [PRIMARY MATCH]" if cname == primary_res["top_differential"] else ""
            bar = "#" * int(prob * 30)
            print(f"  {prob*100:5.1f}% | {bar:<30} | {cname}{indicator}")
        print("=" * 75 + "\n")

        # 6. Save visualization if requested
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            overlay = self.generate_diagnostic_overlay(img_bgr, primary_res, yolo_dets, seg_res, stage_res)
            out_name = f"diagnosis_{os.path.splitext(os.path.basename(image_path))[0]}.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, overlay)
            print(f"[OK] Saved Clinical Diagnostic Overlay -> {out_path}\n")

        return {
            "primary": primary_res,
            "yolo_detections": yolo_dets,
            "segmentation": {
                "tumor_pixel_pct": seg_res.get("tumor_pixel_pct", 0.0),
                "has_vitreous_seeds": seg_res.get("has_vitreous_seeds", False)
            },
            "icrb_stage": {
                "group": stage_res.group,
                "risk_level": stage_res.risk_level,
                "diameter_mm": stage_res.estimated_diameter_mm,
                "rationale": stage_res.clinical_rationale,
                "recommendations": stage_res.treatment_recommendations
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Retinoblastoma Diagnostic Suite")
    parser.add_argument("--image", type=str, required=True, help="Path to input clinical fundus / RetCam image")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Path to directory containing checkpoints")
    parser.add_argument("--backbone", type=str, default="resnet_cbam", help="Model backbone name")
    parser.add_argument("--save-report", action="store_true", default=True, help="Save diagnostic visualization")
    parser.add_argument("--output-dir", type=str, default="results/predictions", help="Directory to save visual report")
    
    args = parser.parse_args()
    suite = ClinicalDiagnosticSuite(checkpoints_dir=args.checkpoints_dir, backbone_name=args.backbone)
    suite.diagnose_image(args.image, save_output=args.save_report, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
