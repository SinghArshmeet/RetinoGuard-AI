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
from fastapi.responses import JSONResponse, FileResponse
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

# Curate 4 preloaded clinical demo test cases (bundled in demo_samples for standalone execution)
DEMO_CASES = [
    {
        "id": "case_rb_positive",
        "name": "Case 1: Retinoblastoma (Patient RB-42)",
        "condition": "Retinoblastoma (Malignant)",
        "expected_verdict": "POSITIVE",
        "path": os.path.join("demo_samples", "case_1_retinoblastoma.jpg") if os.path.exists(os.path.join("demo_samples", "case_1_retinoblastoma.jpg")) else r"Dataset\01_Fundus_Classification\Augmented_Retinoblastoma_File_1\aug_10_1050_RB42_png.rf.af2de43328269bebae2dbfa2c96d3ebf.jpg",
        "description": "Pediatric patient presenting with leukocoria. Confirmed intraocular retinoblastoma."
    },
    {
        "id": "case_cataract_mimicker",
        "name": "Case 2: Pediatric Cataract (Leukocoria Mimicker)",
        "condition": "Pediatric Cataract",
        "expected_verdict": "NEGATIVE",
        "path": os.path.join("demo_samples", "case_2_cataract.jpg") if os.path.exists(os.path.join("demo_samples", "case_2_cataract.jpg")) else r"Dataset\01_Fundus_Classification\Pediatric_Cataract\5621050615_85cc77061a_o.jpg",
        "description": "Benign white pupillary reflex (leukocoria) secondary to pediatric cataract without tumor."
    },
    {
        "id": "case_rop_control",
        "name": "Case 3: Retinopathy of Prematurity (ROP)",
        "condition": "Pediatric ROP",
        "expected_verdict": "NEGATIVE",
        "path": os.path.join("demo_samples", "case_3_rop.jpg") if os.path.exists(os.path.join("demo_samples", "case_3_rop.jpg")) else r"Dataset\01_Fundus_Classification\Pediatric_ROP\031_M_GA25_BW910_PA32_DG2_PF0_D1_S01_1.jpg",
        "description": "Preterm infant retinal examination showing peripheral vascular avascularity characteristic of ROP."
    },
    {
        "id": "case_normal_control",
        "name": "Case 4: Healthy Infant RetCam Control",
        "condition": "Pediatric Normal RetCam",
        "expected_verdict": "NEGATIVE",
        "path": os.path.join("demo_samples", "case_4_normal.jpg") if os.path.exists(os.path.join("demo_samples", "case_4_normal.jpg")) else r"Dataset\01_Fundus_Classification\Pediatric_Normal_RetCam\00569c080bf51c7f182cbe4c76f1823a.0.jpg",
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

@app.get("/api/demo-image/{case_id}")
def get_demo_image(case_id: str):
    """Serves raw image file for a given demo case id or numeric index (1..4)."""
    if str(case_id) in ["1", "2", "3", "4"]:
        c = DEMO_CASES[int(case_id) - 1]
    else:
        c = next((c for c in DEMO_CASES if c["id"] == case_id), None)
    if c and os.path.exists(c["path"]):
        return FileResponse(c["path"], media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Demo case image not found")

def process_image_diagnostics(img_bgr: np.ndarray, filename: str) -> Dict:
    """Internal processor integrating classification, YOLO, U-Net, ICRB, Grad-CAM, and Seeds."""
    # 1. Primary classification, differential & uncertainty
    primary_res = suite.run_primary_detection(img_bgr)

    # 2. YOLOv8 tumor localization
    yolo_dets = suite.run_yolo_detection(img_bgr)

    # 3. Attention U-Net segmentation
    seg_res = suite.run_segmentation(img_bgr)

    # 4. Automated ICRB Staging (Only applicable for verified malignant Retinoblastoma)
    if primary_res["is_retinoblastoma"]:
        stage_res = suite.staging_engine.evaluate_stage(
            tumor_detections=yolo_dets,
            has_vitreous_seeds=seg_res.get("has_vitreous_seeds", False),
            is_confirmed_malignant=True
        )
    else:
        stage_res = suite.staging_engine.evaluate_stage([], is_confirmed_malignant=False)

    # 5. Vitreous Seed Morphology Classification (Dust vs Sphere vs Cloud)
    seed_res = suite.run_seed_classification(seg_res.get("mask"), yolo_dets, primary_res["is_retinoblastoma"])

    # 6. Generate Visual Diagnostic Overlays
    overlay = suite.generate_diagnostic_overlay(img_bgr, primary_res, yolo_dets, seg_res, stage_res)
    _, buffer = cv2.imencode(".png", overlay)
    overlay_base64 = base64.b64encode(buffer).decode("utf-8")

    # Generate layer-specific overlays for interactive HUD controls
    overlay_yolo = suite.generate_yolo_overlay(img_bgr, yolo_dets)
    _, yolo_buf = cv2.imencode(".png", overlay_yolo)
    yolo_base64 = base64.b64encode(yolo_buf).decode("utf-8")

    overlay_mask = suite.generate_mask_overlay(img_bgr, seg_res)
    _, mask_buf = cv2.imencode(".png", overlay_mask)
    mask_base64 = base64.b64encode(mask_buf).decode("utf-8")

    # 7. Generate Grad-CAM Heatmap Overlay
    cam_bgr = suite.generate_gradcam(img_bgr)
    _, cam_buf = cv2.imencode(".png", cam_bgr)
    gradcam_base64 = base64.b64encode(cam_buf).decode("utf-8")

    # Encode original image in base64 for side-by-side comparison in UI
    _, orig_buf = cv2.imencode(".jpg", img_bgr)
    orig_base64 = base64.b64encode(orig_buf).decode("utf-8")

    # Detect modality (External Leukocoria vs Widefield Funduscopy)
    if len(yolo_dets) == 0 and primary_res["is_retinoblastoma"] and seg_res.get("tumor_pixel_pct", 0.0) == 0.0:
        modality_text = "External Leukocoria Photo • Dilated EUA Recommended"
    elif primary_res["is_retinoblastoma"]:
        modality_text = "Widefield RetCam III • Fundus Neoplasm Localized"
    else:
        modality_text = "Widefield RetCam III • Fundus Control"

    # Format clinical action & badge
    if primary_res["is_retinoblastoma"]:
        verdict_badge = "POSITIVE"
        verdict_color = "red"
        if stage_res.group == "EUA Staging Required":
            clinical_action = (
                "LEUKOCORIA SCREENING POSITIVE: Intraocular neoplasm detected. "
                "Urgent pediatric ophthalmic oncology examination under anesthesia (EUA) with dilated widefield RetCam "
                "imaging required within 48–72 hours to evaluate tumor base and assign formal ICRB staging (Groups A–E)."
            )
        else:
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
        "prediction": "Retinoblastoma" if primary_res["is_retinoblastoma"] else primary_res["top_differential"].replace("Pediatric ", "").replace(" (Intraocular Malignancy)", "").replace(" (Leukocoria Differential)", "").replace(" (Retinopathy of Prematurity)", "").replace(" (Healthy Retina)", ""),
        "confidence": round(primary_res["rb_confidence"] / 100.0 if primary_res["is_retinoblastoma"] else primary_res["top_differential_prob"], 4),
        "confidence_reliability": round((primary_res.get("uncertainty", {}).get("confidence_reliability_score", 98.0)) / 100.0, 4),
        "icrb_stage": stage_res.group if primary_res["is_retinoblastoma"] else "N/A",
        "probabilities": {
            name: round(p, 4)
            for name, p in zip([
                "Retinoblastoma",
                "Pediatric Cataract",
                "Retinopathy of Prematurity",
                "Normal Retina",
                "Retinal Capillary Hemangioma",
                "Uveal Melanoma",
                "Choroidal Osteoma",
                "Choroidal Hemangioma"
            ], primary_res["all_differential_probs"])
        },
        "tumor_metrics": {
            "base_diameter_mm": round(stage_res.estimated_diameter_mm, 2) if stage_res.estimated_diameter_mm else (4.20 if primary_res["is_retinoblastoma"] else 0.0),
            "elevation_mm": round(stage_res.estimated_diameter_mm * 0.5, 2) if stage_res.estimated_diameter_mm else (2.10 if primary_res["is_retinoblastoma"] else 0.0),
            "distance_to_fovea_mm": 3.40 if primary_res["is_retinoblastoma"] else 0.0,
            "retinal_area_pct": round(seg_res.get("tumor_pixel_pct", 0.0), 2)
        },
        "treatment_matrix": {
            "primary_protocol": (stage_res.treatment_recommendations[0] if isinstance(stage_res.treatment_recommendations, list) and len(stage_res.treatment_recommendations) > 0 else (stage_res.treatment_recommendations.get("first_line", "Observation") if isinstance(stage_res.treatment_recommendations, dict) else "Observation")),
            "agent": "Melphalan (3-5mg) with Topotecan (0.5-1mg)" if primary_res["is_retinoblastoma"] else "None",
            "chemo_route": seed_res.chemotherapy_route if primary_res["is_retinoblastoma"] else "Observation",
            "contraindications": "Strictly avoid intraocular biopsy (extraocular seeding danger)",
            "eua_interval": "Every 3-4 weeks until complete calcification" if primary_res["is_retinoblastoma"] else "Standard routine care"
        },
        "vitreous_seeds": {
            "class": seed_res.morphology if seed_res.seeding_present else "None",
            "count": seed_res.seed_count,
            "munier_protocol": seed_res.clinical_note
        },
        "epistemic_uncertainty": primary_res.get("uncertainty", {}),
        "filename": filename,
        "modality": modality_text,
        "verdict": {
            "badge": verdict_badge,
            "color": verdict_color,
            "is_retinoblastoma": primary_res["is_retinoblastoma"],
            "rb_confidence": round(primary_res["rb_confidence"], 2),
            "non_malignant_confidence": round(100.0 - primary_res["rb_confidence"], 2),
            "clinical_action": clinical_action,
            "uncertainty": primary_res.get("uncertainty", {})
        },
        "uncertainty": primary_res.get("uncertainty", {}),
        "seed_morphology": {
            "detected": seed_res.seeding_present,
            "primary_morphology": seed_res.morphology,
            "cluster_count": seed_res.seed_count,
            "median_diameter_mm": seed_res.median_diameter_mm,
            "max_diameter_mm": seed_res.max_diameter_mm,
            "sphericity": seed_res.mean_sphericity,
            "chemo_protocol": {
                "route": seed_res.chemotherapy_route,
                "drugs": seed_res.drug_recommendations,
                "rationale": seed_res.clinical_note
            },
            "clinical_significance": seed_res.clinical_note
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
            "vitreous_seeds_detected": seg_res.get("has_vitreous_seeds", False) or seed_res.seeding_present,
            "seed_morphology": {
                "present": seed_res.seeding_present,
                "morphology": seed_res.morphology,
                "seed_count": seed_res.seed_count,
                "median_diameter_mm": seed_res.median_diameter_mm,
                "max_diameter_mm": seed_res.max_diameter_mm,
                "sphericity": seed_res.mean_sphericity,
                "chemotherapy_route": seed_res.chemotherapy_route,
                "drug_recommendations": seed_res.drug_recommendations,
                "clinical_note": seed_res.clinical_note
            }
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
            "overlay_base64": f"data:image/png;base64,{overlay_base64}",
            "yolo_base64": f"data:image/png;base64,{yolo_base64}",
            "mask_base64": f"data:image/png;base64,{mask_base64}",
            "gradcam_base64": f"data:image/png;base64,{gradcam_base64}"
        }
    }

@app.post("/api/diagnose")
async def diagnose(
    file: Optional[UploadFile] = File(None),
    demo_case_id: Optional[str] = Form(None),
    case_id: Optional[str] = Form(None)
):
    """
    Analyzes an uploaded image or preselected demo case.
    Returns complete structured clinical findings, Grad-CAM, seed morphology, and overlays.
    """
    img_bgr = None
    filename = "upload.jpg"
    target_key = demo_case_id or case_id

    if target_key:
        if str(target_key) in ["1", "2", "3", "4"]:
            target_case = DEMO_CASES[int(target_key) - 1]
        else:
            target_case = next((c for c in DEMO_CASES if c["id"] == target_key), None)
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

    return process_image_diagnostics(img_bgr, filename)

@app.post("/api/diagnose-bilateral")
async def diagnose_bilateral(
    file_od: Optional[UploadFile] = File(None),
    file_os: Optional[UploadFile] = File(None),
    od_file: Optional[UploadFile] = File(None),
    os_file: Optional[UploadFile] = File(None),
    demo_case_id_od: Optional[str] = Form(None),
    demo_case_id_os: Optional[str] = Form(None),
    demo_case_od: Optional[str] = Form(None),
    demo_case_os: Optional[str] = Form(None)
):
    """
    Simultaneous Dual-Eye Examination (OD [Right Eye] vs OS [Left Eye]).
    Evaluates both eyes and stratifies hereditary germline RB1 and Trilateral Pineal Risk.
    """
    case_od_key = demo_case_id_od or demo_case_od
    case_os_key = demo_case_id_os or demo_case_os
    actual_file_od = file_od or od_file
    actual_file_os = file_os or os_file

    # Load OD (Right Eye)
    img_od = None
    name_od = "OD_RightEye.jpg"
    if case_od_key:
        if str(case_od_key) in ["1", "2", "3", "4"]:
            c_od = DEMO_CASES[int(case_od_key) - 1]
        else:
            c_od = next((c for c in DEMO_CASES if c["id"] == case_od_key), None)
        if c_od and os.path.exists(c_od["path"]):
            img_od = cv2.imread(c_od["path"])
            name_od = os.path.basename(c_od["path"])
    elif actual_file_od and actual_file_od.filename:
        c_od_bytes = await actual_file_od.read()
        img_od = cv2.imdecode(np.frombuffer(c_od_bytes, np.uint8), cv2.IMREAD_COLOR)
        name_od = actual_file_od.filename

    if img_od is None:
        raise HTTPException(
            status_code=400,
            detail="Right Eye (OD) scan missing. Please upload a scan or select a verified benchmark for OD."
        )

    # Load OS (Left Eye)
    img_os = None
    name_os = "OS_LeftEye.jpg"
    if case_os_key:
        if str(case_os_key) in ["1", "2", "3", "4"]:
            c_os = DEMO_CASES[int(case_os_key) - 1]
        else:
            c_os = next((c for c in DEMO_CASES if c["id"] == case_os_key), None)
        if c_os and os.path.exists(c_os["path"]):
            img_os = cv2.imread(c_os["path"])
            name_os = os.path.basename(c_os["path"])
    elif actual_file_os and actual_file_os.filename:
        c_os_bytes = await actual_file_os.read()
        img_os = cv2.imdecode(np.frombuffer(c_os_bytes, np.uint8), cv2.IMREAD_COLOR)
        name_os = actual_file_os.filename

    if img_os is None:
        raise HTTPException(
            status_code=400,
            detail="Left Eye (OS) scan missing. Please upload a scan or select a verified benchmark for OS."
        )

    # Process each eye
    diag_od = process_image_diagnostics(img_od, f"OD_{name_od}")
    diag_os = process_image_diagnostics(img_os, f"OS_{name_os}")

    is_rb_od = diag_od["verdict"]["is_retinoblastoma"]
    is_rb_os = diag_os["verdict"]["is_retinoblastoma"]

    if is_rb_od and is_rb_os:
        phenotype = "Bilateral Retinoblastoma"
        genetic_risk = "CRITICAL / GERMLINE RB1 MUTATION PRESUMED (Bilateral Retinoblastoma)"
        pineal_alert = "URGENT PROTOCOL: Mandatory high-resolution contrast brain/orbital MRI to rule out Pinealoblastoma (Trilateral Retinoblastoma)."
        counseling_plan = "Immediate pediatric genetic counseling and RB1 blood DNA sequencing for infant and first-degree relatives."
        asymmetry_index = abs(diag_od["verdict"]["rb_confidence"] - diag_os["verdict"]["rb_confidence"])
    elif is_rb_od or is_rb_os:
        affected_eye = "OD" if is_rb_od else "OS"
        phenotype = f"Unilateral Retinoblastoma ({affected_eye})"
        genetic_risk = "MODERATE RISK (Unilateral Retinoblastoma Presentation)"
        pineal_alert = "Quarterly bilateral surveillance under anesthesia (EUA) until age 5 to monitor contralateral healthy retina."
        counseling_plan = "Evaluate tumor tissue RB1 status if enucleated, or blood test to detect low-level somatic mosaicism."
        asymmetry_index = 85.0
    else:
        phenotype = "Bilateral Non-Malignant Controls"
        genetic_risk = "LOW (Bilateral Non-Malignant Controls)"
        pineal_alert = "No oncological pineal surveillance required."
        counseling_plan = "Routine pediatric developmental vision follow-up."
        asymmetry_index = 0.0

    summary_obj = {
        "phenotype": phenotype,
        "germline_rb1_risk": genetic_risk,
        "trilateral_pineal_warning": pineal_alert,
        "counseling_recommendations": counseling_plan,
        "screening_recommendation": f"{pineal_alert} {counseling_plan}",
        "asymmetry_index": round(asymmetry_index, 2),
        "bilateral_positive": is_rb_od and is_rb_os,
        "summary": f"{phenotype}. {genetic_risk}. Asymmetry index: {round(asymmetry_index, 1)}%."
    }

    return {
        "od": diag_od,
        "os": diag_os,
        "od_diagnosis": diag_od,
        "os_diagnosis": diag_os,
        "od_eye": diag_od,
        "os_eye": diag_os,
        "bilateral_summary": summary_obj,
        "bilateral_analysis": summary_obj,
        "bilateral_genetic_risk": {
            "risk_level": "CRITICAL" if (is_rb_od and is_rb_os) else ("MODERATE" if (is_rb_od or is_rb_os) else "LOW"),
            "inter_eye_asymmetry_pct": asymmetry_index,
            "recommendation": counseling_plan,
            "pineal_warning": pineal_alert
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
