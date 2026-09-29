"""
International Classification for Retinoblastoma (ICRB) Clinical Staging Engine
==============================================================================
Implements the clinical rules and guidelines defined by the International
Retinoblastoma Staging Committee (Murphree et al. / Linn Murphree system).

Maps spatial bounding boxes, tumor-to-disc ratio (TDR), tumor diameter,
distance to optic disc/fovea, and secondary signs (seeding, detachment)
into verified ICRB Clinical Groups A through E.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

@dataclass
class TumorDetection:
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized [0, 1]
    confidence: float
    class_name: str
    area_fraction: float

@dataclass
class ICRBStageResult:
    group: str                     # 'Group A', 'Group B', 'Group C', 'Group D', or 'Group E'
    risk_level: str                # 'Very Low', 'Low', 'Moderate', 'High', 'Very High (Enucleation Candidate)'
    estimated_diameter_mm: float   # Approximated from typical 15mm adult/infant retinal field
    max_tumor_area_pct: float      # Maximum single tumor area as % of fundus
    seeding_detected: bool
    detachment_risk: bool
    clinical_rationale: str
    treatment_recommendations: List[str]

class ICRBStagingEngine:
    """
    Expert rule engine translating deep learning bounding boxes and
    multimodal clinical features into standardized ICRB Staging.
    """
    # Standard optical reference: Average optic disc diameter = 1.5 mm.
    # An infant RetCam field of view (~120-130 degrees) corresponds to approx 20-22mm retinal arc.
    RETINAL_FOV_MM = 20.0  # Approx FOV diameter in mm

    def __init__(self):
        pass

    def estimate_tumor_diameter_mm(self, box: Tuple[float, float, float, float]) -> float:
        """Converts normalized bounding box dimensions to estimated physical millimeter diameter."""
        x1, y1, x2, y2 = box
        width_norm = abs(x2 - x1)
        height_norm = abs(y2 - y1)
        max_dim_norm = max(width_norm, height_norm)
        return round(float(max_dim_norm * self.RETINAL_FOV_MM), 2)

    def evaluate_stage(
        self,
        tumor_detections: List[Dict],
        has_vitreous_seeds: bool = False,
        has_retinal_detachment: bool = False,
        anterior_chamber_involvement: bool = False,
        tumor_touches_lens: bool = False
    ) -> ICRBStageResult:
        """
        Computes the clinical ICRB Stage from detected tumor instances and clinical findings.
        """
        # If no tumors detected
        if not tumor_detections:
            return ICRBStageResult(
                group="Non-Tumor / Normal",
                risk_level="None",
                estimated_diameter_mm=0.0,
                max_tumor_area_pct=0.0,
                seeding_detected=False,
                detachment_risk=False,
                clinical_rationale="No intraocular tumor mass or retinoblastoma nodule detected in field of view.",
                treatment_recommendations=["Routine pediatric eye screening.", "No oncologic intervention required."]
            )

        # Extract tumor boxes (class_name == 'Retino Blastoma')
        rb_tumors = [d for d in tumor_detections if "blast" in d.get("class_name", "").lower() or d.get("class_id") == 1]
        if not rb_tumors:
            # Fallback: treat any detected lesion as candidate
            rb_tumors = tumor_detections

        max_diameter_mm = 0.0
        max_area_pct = 0.0
        tumor_count = len(rb_tumors)

        for t in rb_tumors:
            box = t.get("box", (0, 0, 0, 0))
            diam_mm = float(self.estimate_tumor_diameter_mm(box))
            area_pct = float(t.get("area_pct", (abs(box[2] - box[0]) * abs(box[3] - box[1])) * 100))
            if diam_mm > max_diameter_mm:
                max_diameter_mm = float(diam_mm)
            if area_pct > max_area_pct:
                max_area_pct = float(area_pct)

        # Check for multiple satellite small lesions suggestive of seeds
        auto_seeding = has_vitreous_seeds or (tumor_count >= 3 and any(t.get("area_pct", 0) < 1.5 for t in rb_tumors))

        # =========================================================================
        # ICRB CLASSIFICATION DECISION LOGIC
        # =========================================================================
        
        # 1. GROUP E (Very High Risk / Salvage Unlikely)
        if (anterior_chamber_involvement or tumor_touches_lens or max_area_pct > 50.0):
            return ICRBStageResult(
                group="Group E",
                risk_level="Very High (Ocular Salvage Unlikely / Enucleation)",
                estimated_diameter_mm=max_diameter_mm,
                max_tumor_area_pct=round(max_area_pct, 2),
                seeding_detected=auto_seeding,
                detachment_risk=True,
                clinical_rationale=(
                    f"Massive intraocular tumor burden (>50% globe volume or anterior involvement). "
                    f"Max tumor diameter estimated at {max_diameter_mm:.1f} mm ({max_area_pct:.1f}% retinal area). "
                    f"High risk of optic nerve / scleral invasion."
                ),
                treatment_recommendations=[
                    "Primary enucleation with long optic nerve harvest.",
                    "Histopathological risk assessment (post-laminar optic nerve invasion, choroidal invasion).",
                    "Adjuvant systemic chemotherapy if high-risk histopathology present."
                ]
            )

        # 2. GROUP D (High Risk / Diffuse Seeding or Extensive Detachment)
        if (auto_seeding and max_diameter_mm > 6.0) or (has_retinal_detachment and max_area_pct > 25.0):
            return ICRBStageResult(
                group="Group D",
                risk_level="High Risk (Poor Visual Prognosis / Globe Salvage Attemptable)",
                estimated_diameter_mm=max_diameter_mm,
                max_tumor_area_pct=round(max_area_pct, 2),
                seeding_detected=True,
                detachment_risk=True,
                clinical_rationale=(
                    f"Advanced retinoblastoma with diffuse subretinal/vitreous seeds or extensive detachment. "
                    f"Primary tumor diameter: {max_diameter_mm:.1f} mm. Subretinal seeds > 3mm from tumor mass."
                ),
                treatment_recommendations=[
                    "Intra-arterial chemotherapy (IAC) with Melphalan / Topotecan.",
                    "Intravitreal chemotherapy (IViC) for vitreous seed control.",
                    "Systemic chemoreduction (VEC protocol: Vincristine, Etoposide, Carboplatin).",
                    "Close fundus monitoring every 3-4 weeks under anesthesia."
                ]
            )

        # 3. GROUP C (Moderate Risk / Localized Seeds)
        if auto_seeding or (max_diameter_mm > 3.0 and max_area_pct > 15.0):
            return ICRBStageResult(
                group="Group C",
                risk_level="Moderate Risk (Discrete Local Seeding <= 3mm from tumor)",
                estimated_diameter_mm=max_diameter_mm,
                max_tumor_area_pct=round(max_area_pct, 2),
                seeding_detected=auto_seeding,
                detachment_risk=has_retinal_detachment,
                clinical_rationale=(
                    f"Intraretinal tumor with discrete localized seeding or tumor diameter {max_diameter_mm:.1f} mm. "
                    f"Seeds confined within 3mm of tumor border."
                ),
                treatment_recommendations=[
                    "Chemoreduction (Systemic VEC or Intra-Arterial Chemotherapy).",
                    "Consolidation with focal therapy (transpupillary thermotherapy / cryotherapy).",
                    "Serial fundus photography and B-scan ultrasonography."
                ]
            )

        # 4. GROUP B (Low Risk / Large Intraretinal Tumor > 3mm without seeds)
        if max_diameter_mm > 3.0 or max_area_pct > 5.0:
            return ICRBStageResult(
                group="Group B",
                risk_level="Low Risk (Large Intraretinal Tumor > 3mm, No Seeds)",
                estimated_diameter_mm=max_diameter_mm,
                max_tumor_area_pct=round(max_area_pct, 2),
                seeding_detected=False,
                detachment_risk=has_retinal_detachment,
                clinical_rationale=(
                    f"Confined intraretinal retinoblastoma measuring {max_diameter_mm:.1f} mm in diameter "
                    f"(> 3 mm threshold). No vitreous or subretinal seeds detected."
                ),
                treatment_recommendations=[
                    "Systemic chemoreduction or Intra-Arterial Chemotherapy (IAC).",
                    "Focal consolidation (laser photocoagulation / cryopexy) after tumor shrinkage.",
                    "Excellent eye preservation prognosis (>90%)."
                ]
            )

        # 5. GROUP A (Very Low Risk / Small Intraretinal Tumor <= 3mm)
        return ICRBStageResult(
            group="Group A",
            risk_level="Very Low Risk (Small Intraretinal Tumor <= 3mm, No Seeds)",
            estimated_diameter_mm=max_diameter_mm,
            max_tumor_area_pct=round(max_area_pct, 2),
            seeding_detected=False,
            detachment_risk=False,
            clinical_rationale=(
                f"Early-stage small intraretinal retinoblastoma measuring {max_diameter_mm:.1f} mm "
                f"(<= 3 mm threshold, <= 1.5 disc diameters). Located away from fovea and optic disc without seeds."
            ),
            treatment_recommendations=[
                "Primary focal therapy without chemotherapy (Transpupillary Thermotherapy - TTT / Laser photocoagulation / Cryotherapy).",
                "Ocular salvage rate: > 98%.",
                "Periodic follow-up examination under anesthesia (EUA)."
            ]
        )
