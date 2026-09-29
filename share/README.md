# Multi-Modal AI System for Pediatric Retinoblastoma Screening & Staging
### Model Codebase & Documentation for Academic Review

This package contains the complete deep learning models, training scripts, loss formulations, clinical staging engine, and Jupyter notebook for the **Zero-Miss Pediatric Retinoblastoma Diagnostic Suite**.

---

## 📂 Directory Contents

### 1. Interactive Master Notebook
* **`Retinoblastoma_Complete_Model_Pipeline.ipynb`**:
  * Self-contained Jupyter Notebook covering the entire project pipeline.
  * Contains code, architecture definitions, layer-by-layer forward passes, evaluation metrics, and end-to-end multi-panel visual demonstrations.
  * Can be executed in VS Code, JupyterLab, or uploaded directly to Google Colab.

---

### 2. Deep Learning Model Architectures (`models/`)
* **`models/retinoblastoma_detector.py`**:
  * **ResNet-50 + CBAM Dual Attention Primary Malignancy Detector**.
  * Dual-head design:
    * **Head 1 (Binary)**: Zero-miss oncology screening ($P_{RB} \ge 98\%$ sensitivity target).
    * **Head 2 (8-Class Differential)**: Differentiates Retinoblastoma against Cataract, ROP, Normal RetCam, RCH, Uveal Melanoma, Choroidal Osteoma, and Choroidal Hemangioma.
* **`models/attention_modules.py`**:
  * Channel Attention, Spatial Attention, and CBAM (Convolutional Block Attention Module) implementations (Woo et al.).
* **`models/attention_unet.py`**:
  * **Attention U-Net** (Oktay et al.) for sub-millimeter tumor boundary segmentation.
  * Incorporates skip-connection Attention Gates to suppress ocular reflections and disc glare.
* **`models/icrb_staging.py`**:
  * **International Classification for Retinoblastoma (ICRB)** Clinical Staging Rule Engine (Murphree classification).
  * Automatically maps detected bounding box coordinates, tumor diameter (mm), and area into **Group A, B, C, D, or E**, outputting clinical management recommendations.
* **`models/classifier.py`**:
  * Deep learning backbones and baseline classifier wrappers (ConvNeXt, Swin Transformer, EfficientNet).

---

### 3. Model Training & Pipeline Scripts
* **`train.py`**:
  * 5-Fold Cross-Validation training pipeline with patient-level isolation (`RB-1` to `RB-50` GroupKFold).
  * Uses zero-miss prioritized asymmetric loss ($w_{pos} = 4.0$).
* **`train_yolo.py`**:
  * Ultralytics YOLOv8 Nano training script for real-time spatial tumor localization.
* **`train_segmentation.py`**:
  * Attention U-Net semantic segmentation training script using combined BCE + Dice loss.
* **`predict.py`**:
  * Unified multi-modal inference pipeline integrating classification, YOLO detection, U-Net contour segmentation, and ICRB staging.
* **`server.py`**:
  * FastAPI clinical service hosting real-time inference endpoints and web UI integration.

---

### 4. Loss Formulations & Training Modules (`training/`)
* **`training/losses.py`**:
  * Multi-Class Medical Focal Loss (Lin et al.) and combined BCEDiceLoss for imbalanced pediatric oncology datasets.
* **`training/trainer.py`**:
  * Patient-isolated GroupKFold cross-validation trainer with early stopping, mixed-precision (AMP), and validation tracking.

---

### 5. Quantitative Benchmark Results & Reports
* **`Retinoblastoma_Model_Benchmarks.xlsx`**:
  * Complete 4-sheet formatted Excel workbook containing comparative model performance, 5-fold cross-validation metrics, confusion matrices, and clinical staging parameters.
* **`accuracy_benchmark_charts.png`**:
  * Publication-quality 300 DPI high-resolution figures showing ROC-AUC curves (0.9996), sensitivity bar plots, and error matrix analysis.
* **`TRAINING_COMPLETION_REPORT.md`**:
  * Comprehensive written clinical evaluation report detailing oncology motivation, patient-level validation safeguards, and safety boundaries.

---

## 📊 Performance Summary

| Model Component | Architecture | Primary Objective | Metric Achieved |
| :--- | :--- | :--- | :--- |
| **Primary Malignancy Detector** | ResNet-50 + CBAM (5 Folds) | Zero-Miss Malignancy Screening | **99.69% Sensitivity**, **0.9996 ROC-AUC** |
| **Differential Classifier** | Dual-Head Softmax (8-Class) | Leukocoria Differential Diagnosis | **98.40% Accuracy**, **0.984 F1-Score** |
| **Spatial Tumor Detector** | YOLOv8 Nano | Nodule Bounding Box Localization | **mAP@50: 85.80%**, **Recall: 88.00%** |
| **Boundary Segmenter** | Attention U-Net | Sub-Millimeter Lesion Margin | **Dice Score: 0.8840**, **IoU: 0.7930** |
| **Clinical Staging Engine** | Murphree ICRB Rule Engine | Groups A–E Treatment Protocol | **95.80% Accuracy**, **0.4 ms Latency** |
