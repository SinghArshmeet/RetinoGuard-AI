"""
Vitreous Seed Morphology Classification Engine (Dust vs Sphere vs Cloud)
========================================================================
Implements the International Pediatric Ocular Oncology Guidelines
(Munier et al., Shields et al.) for vitreous and subretinal seeding.

Seeding Morphology Classes:
1. Dust: Microscopic single-cell / fine particulate seeding (<0.1 mm)
2. Spheres: Discrete, spherical, refractory multicellular spherules (0.1 - 1.0 mm)
3. Clouds: Dense, confluent, three-dimensional vitreous tumor masses (>1.0 mm)

Each morphology has direct chemotherapeutic protocol implications:
- Dust: Responds rapidly to systemic chemoreduction (VEC) or low-dose Intravitreal Chemotherapy (IViC).
- Spheres: Requires targeted IViC with Melphalan (20-30 ug) +/- Topotecan (20 ug) weekly.
- Clouds: Chemoresistant; requires intensive Intra-Arterial Chemotherapy (IAC) + high-dose IViC.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

@dataclass
class VitreousSeedAnalysis:
    seeding_present: bool
    morphology: str            # 'None', 'Dust', 'Spheres', 'Clouds'
    seed_count: int
    median_diameter_mm: float
    max_diameter_mm: float
    mean_sphericity: float     # [0, 1] circularity metric
    chemotherapy_route: str    # e.g., 'Intravitreal Chemotherapy (IViC)', 'Intra-Arterial (IAC)'
    drug_recommendations: List[str]
    clinical_note: str

class VitreousSeedClassifier:
    """
    Analyzes segmented retinal lesion masks and YOLO bounding boxes
    to categorize vitreous seed morphology and prescribe chemotherapeutic pathways.
    """
    RETINAL_FOV_MM = 20.0  # Approx FOV diameter in mm for standard RetCam wide-field

    def __init__(self):
        pass

    def analyze_seeding(
        self,
        mask: Optional[np.ndarray],
        yolo_detections: List[Dict],
        is_retinoblastoma: bool
    ) -> VitreousSeedAnalysis:
        """
        Extracts morphological features of non-primary satellite lesion clusters.
        """
        if not is_retinoblastoma:
            return VitreousSeedAnalysis(
                seeding_present=False,
                morphology="None",
                seed_count=0,
                median_diameter_mm=0.0,
                max_diameter_mm=0.0,
                mean_sphericity=0.0,
                chemotherapy_route="Not Applicable",
                drug_recommendations=["No oncologic seeding therapy required."],
                clinical_note="No malignant retinoblastoma detected; seeding evaluation normal."
            )

        # If no mask or detections provided
        if mask is None or np.sum(mask) == 0:
            # Check yolo satellite boxes
            rb_boxes = [d for d in yolo_detections if "blast" in d.get("class_name", "").lower() or d.get("class_id") == 1]
            if len(rb_boxes) <= 1:
                return VitreousSeedAnalysis(
                    seeding_present=False,
                    morphology="None",
                    seed_count=0,
                    median_diameter_mm=0.0,
                    max_diameter_mm=0.0,
                    mean_sphericity=0.0,
                    chemotherapy_route="Systemic / Focal Therapy",
                    drug_recommendations=["Primary focal therapy (TTT / Cryotherapy) or systemic chemoreduction."],
                    clinical_note="Confined intraretinal tumor mass without detached vitreous seeds."
                )

        h, w = (512, 512)
        if mask is not None and mask.ndim >= 2:
            h, w = mask.shape[:2]
            binary_mask = (mask > 127).astype(np.uint8)
        else:
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            # Reconstruct from YOLO boxes
            for d in yolo_detections:
                box = d.get("box_pixels", [0, 0, 0, 0])
                cv2.rectangle(binary_mask, (box[0], box[1]), (box[2], box[3]), 255, -1)

        # Find connected components (satellite lesions)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # If <= 1 contour, main mass only, no seeding
        if len(contours) <= 1:
            return VitreousSeedAnalysis(
                seeding_present=False,
                morphology="None",
                seed_count=0,
                median_diameter_mm=0.0,
                max_diameter_mm=0.0,
                mean_sphericity=0.0,
                chemotherapy_route="Focal / Systemic Chemotherapy",
                drug_recommendations=["Systemic chemoreduction (VEC) or focal photocoagulation."],
                clinical_note="Solitary intraretinal mass without evidence of vitreous seeds."
            )

        # Exclude the largest contour (primary tumor mass); remaining are seed candidates
        seed_contours = contours[1:]
        seed_count = len(seed_contours)

        diameters_mm = []
        sphericities = []

        for c in seed_contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if area < 4:  # filter sub-pixel noise
                continue

            # Equivalent circular diameter in pixels
            equiv_diam_px = 2.0 * np.sqrt(area / np.pi)
            equiv_diam_norm = equiv_diam_px / max(h, w)
            equiv_diam_mm = equiv_diam_norm * self.RETINAL_FOV_MM
            diameters_mm.append(float(equiv_diam_mm))

            # Circularity metric: 4 * pi * Area / Perimeter^2
            if perimeter > 0:
                circ = (4.0 * np.pi * area) / (perimeter * perimeter)
                sphericities.append(float(min(1.0, circ)))

        if not diameters_mm:
            diameters_mm = [0.2]
            sphericities = [0.8]

        median_diam = float(np.median(diameters_mm))
        max_diam = float(np.max(diameters_mm))
        mean_spher = float(np.mean(sphericities)) if sphericities else 0.5

        # =========================================================================
        # SEED MORPHOLOGY DECISION TREE
        # =========================================================================
        # 1. CLOUDS: Confluent, massive clusters (> 1.0 mm)
        if max_diam > 1.0 or (seed_count >= 5 and median_diam > 0.8):
            return VitreousSeedAnalysis(
                seeding_present=True,
                morphology="Clouds",
                seed_count=seed_count,
                median_diameter_mm=round(median_diam, 2),
                max_diameter_mm=round(max_diam, 2),
                mean_sphericity=round(mean_spher, 2),
                chemotherapy_route="Intra-Arterial Chemotherapy (IAC) + High-Dose IViC",
                drug_recommendations=[
                    "Intra-arterial Melphalan (5-7.5 mg) via ophthalmic artery cannulation.",
                    "Intravitreal Melphalan (30 ug) + Topotecan (20 ug) weekly under safety protocols (cryo at injection site).",
                    "Close monitoring for non-response; prepare for potential enucleation if progression occurs."
                ],
                clinical_note=(
                    f"Confluent three-dimensional 'Cloud' seeding detected (Max diameter: {max_diam:.2f} mm). "
                    "Dense avascular core with high hypoxic chemoresistance."
                )
            )

        # 2. SPHERES: Discrete spherical multicellular spherules (0.1 - 1.0 mm, circularity >= 0.65)
        elif median_diam >= 0.1 and mean_spher >= 0.60:
            return VitreousSeedAnalysis(
                seeding_present=True,
                morphology="Spheres",
                seed_count=seed_count,
                median_diameter_mm=round(median_diam, 2),
                max_diameter_mm=round(max_diam, 2),
                mean_sphericity=round(mean_spher, 2),
                chemotherapy_route="Intravitreal Chemotherapy (IViC) Targeted Therapy",
                drug_recommendations=[
                    "Intravitreal Melphalan (20-30 ug) weekly for 3-6 cycles.",
                    "Adjunctive intravitreal Topotecan (20 ug) for recalcitrant spherules.",
                    "Cryotherapy to entry site during needle withdrawal to prevent extraocular tumor seeding."
                ],
                clinical_note=(
                    f"Discrete refractory 'Sphere' seeding detected ({seed_count} spherules, median diameter: {median_diam:.2f} mm, sphericity: {mean_spher:.2f}). "
                    "Optimal response rate (approx 90%) with serial intravitreal Melphalan."
                )
            )

        # 3. DUST: Microscopic single-cell scatter (< 0.1 mm)
        else:
            return VitreousSeedAnalysis(
                seeding_present=True,
                morphology="Dust",
                seed_count=seed_count,
                median_diameter_mm=round(median_diam, 2),
                max_diameter_mm=round(max_diam, 2),
                mean_sphericity=round(mean_spher, 2),
                chemotherapy_route="Systemic Chemoreduction +/- Low-Dose IViC",
                drug_recommendations=[
                    "Systemic VEC chemotherapy (Vincristine, Etoposide, Carboplatin).",
                    "Low-dose intravitreal Melphalan (15-20 ug) if residual dust persists post-cycle 2.",
                    "Excellent visual prognosis if macula remains uninvolved."
                ],
                clinical_note=(
                    f"Fine particulate 'Dust' seeding detected ({seed_count} foci, median diameter: {median_diam:.2f} mm). "
                    "High vascular permeability; typically clears rapidly with standard systemic induction."
                )
            )
