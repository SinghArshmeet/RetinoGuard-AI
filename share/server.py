"""
RetinoGuard AI - Clinical FastAPI Backend Service
=================================================
Connects the Stitch / Web UI to the trained Retinoblastoma deep learning suite:
- 5-Fold ResNet-CBAM Ensemble
- 8-Class Differential Diagnosis
- YOLOv8 Tumor Localization
- Attention U-Net Boundary Segmentation
- Automated ICRB Clinical Staging Engine

Endpoints:
  GET  /api/health        - Verifies loaded models and system status
  GET  /api/demo-samples  - Returns preloaded clinical demo test cases
  POST /api/diagnose      - Analyzes uploaded fundus/RetCam image and returns JSON + Base64 visual overlay
"""

import os
import glob
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional

from predict import ClinicalDiagnosticSuite

app = FastAPI(
    title="RetinoGuard AI - Clinical Diagnostic Suite API",
    description="Multi-modal deep learning API for pediatric Retinoblastoma detection, differential diagnosis, lesion localization, and ICRB staging.",
    version="1.0.0"
)

# Enable CORS for local or remote web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the clinical suite once at startup
print("\n[INIT] Initializing Clinical Multi-Modal AI Diagnostic Suite...", flush=True)
suite = ClinicalDiagnosticSuite(checkpoints_dir="checkpoints", backbone_name="resnet_cbam")
print(f"[INIT] Loaded {len(suite.detector_ckpts)} Primary Detector Folds.", flush=True)
print(f"[INIT] YOLOv8 Checkpoint: {suite.yolo_ckpt} (Exists: {os.path.exists(suite.yolo_ckpt)})", flush=True)
print(f"[INIT] Attention U-Net Checkpoint: {suite.unet_ckpt} (Exists: {os.path.exists(suite.unet_ckpt)})", flush=True)

# Curate 4 preloaded clinical demo test cases
DEMO_CASES = [
    {
        "id": "case_rb_positive",
        "name": "Case 1: Retinoblastoma (Patient RB-42)",
        "condition": "Retinoblastoma (Malignant)",
        "expected_verdict": "POSITIVE",
        "path": r"Dataset\01_Fundus_Classification\Augmented_Retinoblastoma_File_1\aug_10_1050_RB42_png.rf.af2de43328269bebae2dbfa2c96d3ebf.jpg",
        "description": "Pediatric patient presenting with leukocoria. Confirmed intraocular retinoblastoma."
    },
    {
        "id": "case_cataract_mimicker",
        "name": "Case 2: Pediatric Cataract (Leukocoria Mimicker)",
        "condition": "Pediatric Cataract",
        "expected_verdict": "NEGATIVE",
        "path": r"Dataset\01_Fundus_Classification\Pediatric_Cataract\5621050615_85cc77061a_o.jpg",
        "description": "Benign white pupillary reflex (leukocoria) secondary to pediatric cataract without tumor."
    },
    {
        "id": "case_rop_control",
        "name": "Case 3: Retinopathy of Prematurity (ROP)",
        "condition": "Pediatric ROP",
        "expected_verdict": "NEGATIVE",
        "path": r"Dataset\01_Fundus_Classification\Pediatric_ROP\031_M_GA25_BW910_PA32_DG2_PF0_D1_S01_1.jpg",
        "description": "Preterm infant retinal examination showing peripheral vascular avascularity characteristic of ROP."
    },
    {
        "id": "case_normal_control",
        "name": "Case 4: Healthy Infant RetCam Control",
        "condition": "Pediatric Normal RetCam",
        "expected_verdict": "NEGATIVE",
        "path": r"Dataset\01_Fundus_Classification\Pediatric_Normal_RetCam\00569c080bf51c7f182cbe4c76f1823a.0.jpg",
        "description": "Normal, healthy pediatric retina with crisp optic disc and clear macula."
    }
]

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "ensemble_folds_loaded": len(suite.detector_ckpts),
        "yolo_detector_active": os.path.exists(suite.yolo_ckpt),
        "attention_unet_active": os.path.exists(suite.unet_ckpt),
        "icrb_engine_active": True,
        "device": str(suite.device)
    }

@app.get("/api/demo-samples")
def get_demo_samples():
    """Returns available preloaded test cases for quick 1-click testing in UI."""
    available_cases = []
    for c in DEMO_CASES:
        if os.path.exists(c["path"]):
            available_cases.append({
                "id": c["id"],
                "name": c["name"],
                "condition": c["condition"],
                "expected_verdict": c["expected_verdict"],
                "description": c["description"]
            })
    return {"demo_cases": available_cases}

@app.post("/api/diagnose")
async def diagnose(
    file: Optional[UploadFile] = File(None),
    demo_case_id: Optional[str] = Form(None)
):
    """
    Analyzes an uploaded image or preselected demo case.
    Returns complete structured clinical findings and a Base64-encoded visual overlay.
    """
    img_bgr = None
    filename = "upload.jpg"

    if demo_case_id:
        target_case = next((c for c in DEMO_CASES if c["id"] == demo_case_id), None)
        if not target_case or not os.path.exists(target_case["path"]):
            raise HTTPException(status_code=404, detail="Demo case not found on disk")
        img_bgr = cv2.imread(target_case["path"])
        filename = os.path.basename(target_case["path"])
    elif file and file.filename:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        filename = file.filename
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded image file or a demo_case_id")

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # 1. Primary classification & differential
    primary_res = suite.run_primary_detection(img_bgr)

    # 2. YOLOv8 tumor localization
    yolo_dets = suite.run_yolo_detection(img_bgr)

    # 3. Attention U-Net segmentation
    seg_res = suite.run_segmentation(img_bgr)

    # 4. Automated ICRB Staging (Only applicable for verified malignant Retinoblastoma)
    if primary_res["is_retinoblastoma"]:
        stage_res = suite.staging_engine.evaluate_stage(
            tumor_detections=yolo_dets,
            has_vitreous_seeds=seg_res.get("has_vitreous_seeds", False)
        )
    else:
        stage_res = suite.staging_engine.evaluate_stage([])

    # 5. Generate Visual Diagnostic Overlay
    overlay = suite.generate_diagnostic_overlay(img_bgr, primary_res, yolo_dets, seg_res, stage_res)
    _, buffer = cv2.imencode(".png", overlay)
    overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    # Encode original image in base64 for side-by-side comparison in UI
    _, orig_buf = cv2.imencode(".jpg", img_bgr)
    orig_base64 = base64.b64encode(orig_buf).decode("utf-8")

    # Format clinical action & badge
    if primary_res["is_retinoblastoma"]:
        verdict_badge = "POSITIVE"
        verdict_color = "red"
        clinical_action = (
            "CRITICAL ONCOLOGIC PRIORITY: Immediate pediatric ophthalmic oncology referral recommended within 48–72 hours. "
            "Contraindication: DO NOT perform intraocular aspiration or biopsy (high risk of tumor seeding)."
        )
    else:
        verdict_badge = "NEGATIVE"
        verdict_color = "green"
        clinical_action = (
            "NON-MALIGNANT CLINICAL MANAGEMENT: No retinoblastoma detected. Follow standard ophthalmic protocol "
            f"for differential diagnosis ({primary_res['top_differential']})."
        )

    return {
        "filename": filename,
        "verdict": {
            "badge": verdict_badge,
            "color": verdict_color,
            "is_retinoblastoma": primary_res["is_retinoblastoma"],
            "rb_confidence": round(primary_res["rb_confidence"], 2),
            "non_malignant_confidence": round(100.0 - primary_res["rb_confidence"], 2),
            "clinical_action": clinical_action
        },
        "differential": {
            "top_match": primary_res["top_differential"],
            "top_match_prob": round(primary_res["top_differential_prob"], 2),
            "all_classes": [
                {"name": name, "prob": round(p * 100.0, 2)}
                for name, p in zip([
                    "Retinoblastoma (Intraocular Malignancy)",
                    "Pediatric Cataract (Leukocoria Differential)",
                    "Pediatric ROP (Retinopathy of Prematurity)",
                    "Pediatric Normal RetCam (Healthy Retina)",
                    "Retinal Capillary Hemangioma (RCH)",
                    "Uveal Melanoma (UM)",
                    "Choroidal Osteoma (CO)",
                    "Choroidal Hemangioma (CH)"
                ], primary_res["all_differential_probs"])
            ]
        },
        "tumor_localization": {
            "lesion_count": len(yolo_dets),
            "detections": yolo_dets
        },
        "tumor_segmentation": {
            "retinal_area_pct": round(seg_res.get("tumor_pixel_pct", 0.0), 2),
            "vitreous_seeds_detected": seg_res.get("has_vitreous_seeds", False)
        },
        "icrb_staging": {
            "group": stage_res.group,
            "risk_level": stage_res.risk_level,
            "estimated_diameter_mm": stage_res.estimated_diameter_mm,
            "clinical_rationale": stage_res.clinical_rationale,
            "treatment_recommendations": stage_res.treatment_recommendations
        },
        "images": {
            "original_base64": f"data:image/jpeg;base64,{orig_base64}",
            "overlay_base64": f"data:image/png;base64,{overlay_base64}"
        }
    }

# Mount static frontend directory if it exists
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
elif os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("  RETINOGUARD AI CLINICAL BACKEND SERVICE")
    print("  Access API documentation at: http://localhost:8000/docs")
    print("="*70 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
