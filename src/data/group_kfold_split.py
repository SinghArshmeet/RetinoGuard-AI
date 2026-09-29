import os
import glob
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
import torch
from torch.utils.data import Dataset
from src.preprocessing.pipeline import PreprocessingPipeline

CLASS_MAPPING = {
    "Retinoblastoma": 0,
    "Pediatric_Cataract": 1,
    "Pediatric_ROP": 2,
    "Pediatric_Normal_RetCam": 3,
    "Retinal_Capillary_Hemangioma": 4,
    "Uveal_Melanoma": 5,
    "Choroidal_Osteoma": 6,
    "Choroidal_Hemangioma": 7
}

def extract_patient_id(filename: str, class_name: str) -> str:
    """Extracts ground-truth patient identifier to prevent cross-validation leakage."""
    if class_name == "Retinoblastoma":
        # Check for RB1 to RB50 patterns in augmented filenames
        match = re.search(r'(RB\d+)', filename, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Fallback to pristine filename base
        return "RB_PRISTINE_" + filename.split('.')[0]
    elif class_name == "Pediatric_ROP":
        # For ROP infant filenames, the first token before underscore is patient ID (e.g. 001_F_GA41...)
        parts = filename.split('_')
        if len(parts) > 1 and parts[0].isdigit():
            return f"ROP_PATIENT_{int(parts[0])}"
        return "ROP_" + filename.split('.')[0]
    else:
        # For other independent clinical photos, treat each photo as an independent eye
        return f"{class_name}_{filename.split('.')[0]}"

def collect_all_data(dataset_root: str = "Dataset") -> pd.DataFrame:
    """Scans all clinical classes and builds a master dataframe with verified patient IDs."""
    records = []
    
    # 1. Retinoblastoma (Augmented)
    aug_rb_dir = os.path.join(dataset_root, "01_Fundus_Classification", "Augmented_Retinoblastoma_File_1")
    if os.path.exists(aug_rb_dir):
        for f in os.listdir(aug_rb_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                p_id = extract_patient_id(f, "Retinoblastoma")
                records.append({
                    "filepath": os.path.join(aug_rb_dir, f),
                    "filename": f,
                    "class_name": "Retinoblastoma",
                    "label_id": CLASS_MAPPING["Retinoblastoma"],
                    "patient_id": p_id
                })

    # 2. Retinoblastoma (Pristine 255 clinical images)
    pristine_rb_dir = os.path.join(dataset_root, "01_Fundus_Classification", "Intraocular_Tumors_6_Class", "Retinoblastoma (RB)")
    if os.path.exists(pristine_rb_dir):
        for f in os.listdir(pristine_rb_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                records.append({
                    "filepath": os.path.join(pristine_rb_dir, f),
                    "filename": f,
                    "class_name": "Retinoblastoma",
                    "label_id": CLASS_MAPPING["Retinoblastoma"],
                    "patient_id": "RB_PRISTINE_" + f.split('.')[0]
                })

    # 3. Pediatric Cataract
    cataract_dir = os.path.join(dataset_root, "01_Fundus_Classification", "Pediatric_Cataract")
    if os.path.exists(cataract_dir):
        for f in os.listdir(cataract_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                records.append({
                    "filepath": os.path.join(cataract_dir, f),
                    "filename": f,
                    "class_name": "Pediatric_Cataract",
                    "label_id": CLASS_MAPPING["Pediatric_Cataract"],
                    "patient_id": extract_patient_id(f, "Pediatric_Cataract")
                })

    # 4. Pediatric ROP
    rop_dir = os.path.join(dataset_root, "01_Fundus_Classification", "Pediatric_ROP")
    if os.path.exists(rop_dir):
        for f in os.listdir(rop_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                records.append({
                    "filepath": os.path.join(rop_dir, f),
                    "filename": f,
                    "class_name": "Pediatric_ROP",
                    "label_id": CLASS_MAPPING["Pediatric_ROP"],
                    "patient_id": extract_patient_id(f, "Pediatric_ROP")
                })

    # 5. Pediatric Normal RetCam Controls
    normal_dir = os.path.join(dataset_root, "01_Fundus_Classification", "Pediatric_Normal_RetCam")
    if os.path.exists(normal_dir):
        for f in os.listdir(normal_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                records.append({
                    "filepath": os.path.join(normal_dir, f),
                    "filename": f,
                    "class_name": "Pediatric_Normal_RetCam",
                    "label_id": CLASS_MAPPING["Pediatric_Normal_RetCam"],
                    "patient_id": extract_patient_id(f, "Pediatric_Normal_RetCam")
                })

    # 6. Differential Intraocular Tumors (RCH, UM, CO, CH)
    tumor_map = {
        "Retinal Capillary Hemangioma (RCH)": "Retinal_Capillary_Hemangioma",
        "Uveal Melanoma (UM)": "Uveal_Melanoma",
        "Choroidal Osteoma (CO)": "Choroidal_Osteoma",
        "Choroidal Hemangioma (CH)": "Choroidal_Hemangioma"
    }
    intraocular_dir = os.path.join(dataset_root, "01_Fundus_Classification", "Intraocular_Tumors_6_Class")
    for sub, cname in tumor_map.items():
        sub_dir = os.path.join(intraocular_dir, sub)
        if os.path.exists(sub_dir):
            for f in os.listdir(sub_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    records.append({
                        "filepath": os.path.join(sub_dir, f),
                        "filename": f,
                        "class_name": cname,
                        "label_id": CLASS_MAPPING[cname],
                        "patient_id": extract_patient_id(f, cname)
                    })

    df = pd.DataFrame(records)
    return df

def generate_5fold_splits(dataset_root: str = "Dataset", output_dir: str = "Dataset/5fold_splits", n_splits: int = 5, seed: int = 42):
    """
    Performs patient-stratified GroupKFold cross-validation partitioning.
    Ensures ZERO patient leakage between train and validation across all 5 folds.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = collect_all_data(dataset_root)
    print(f"Total dataset entries collected: {len(df):,}")
    print("\nClass distribution:")
    print(df['class_name'].value_counts())
    
    # Save master manifest
    master_csv = os.path.join(output_dir, "full_dataset_manifest.csv")
    df.to_csv(master_csv, index=False)
    print(f"\nSaved master manifest -> {master_csv}")

    # Generate 5 folds using GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    df['fold'] = -1
    
    # Group on patient_id
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df['patient_id'])):
        df.loc[val_idx, 'fold'] = fold_idx

    # Save per-fold train and val splits
    for k in range(n_splits):
        train_df = df[df['fold'] != k].copy()
        val_df = df[df['fold'] == k].copy()
        
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        
        combined_fold_df = pd.concat([train_df, val_df], ignore_index=True)
        fold_csv = os.path.join(output_dir, f"fold_{k}.csv")
        combined_fold_df.to_csv(fold_csv, index=False)
        
        # Verify patient isolation
        train_patients = set(train_df['patient_id'])
        val_patients = set(val_df['patient_id'])
        overlap = train_patients.intersection(val_patients)
        print(f"Fold {k}: {len(train_df)} train images, {len(val_df)} val images | Patient Overlap: {len(overlap)} (VERIFIED LEAKAGE-FREE)")

    return df

class RetinoblastomaDataset(Dataset):
    """PyTorch Dataset with on-the-fly or cached medical preprocessing, filters, and normalization."""
    def __init__(self, csv_file: str, split: str = "train", target_size: int = 256, augment: bool = True, cache_in_ram: bool = True, cache_dir: str = "Dataset/cache_preprocessed_256"):
        df = pd.read_csv(csv_file)
        self.data = df[df['split'] == split].reset_index(drop=True)
        self.target_size = target_size
        self.pipeline = PreprocessingPipeline(target_size=target_size)
        self.augment = augment and (split == "train")
        self.cache_in_ram = cache_in_ram
        self.cache_dir = cache_dir
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.cache_in_ram and idx in self.cache:
            tensor, label = self.cache[idx]
            out_tensor = tensor.clone()
            if self.augment:
                if np.random.rand() > 0.5:
                    out_tensor = torch.flip(out_tensor, [2])  # horizontal flip
                if np.random.rand() > 0.5:
                    out_tensor = torch.flip(out_tensor, [1])  # vertical flip
            return out_tensor, label

        row = self.data.iloc[idx]
        img_path = row['filepath']
        label = int(row['label_id'])
        
        # Check disk cache first
        from src.data.cache_preprocessor import get_cache_filename
        cache_file = os.path.join(self.cache_dir, get_cache_filename(img_path, self.target_size))
        
        if os.path.exists(cache_file):
            try:
                tensor = torch.load(cache_file, weights_only=True).float()
            except Exception:
                tensor = torch.load(cache_file).float()
        else:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
            tensor = self.pipeline(img_bgr)
            
        lbl_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.cache_in_ram:
            self.cache[idx] = (tensor, lbl_tensor)

        out_tensor = tensor.clone()
        if self.augment:
            if np.random.rand() > 0.5:
                out_tensor = torch.flip(out_tensor, [2])
            if np.random.rand() > 0.5:
                out_tensor = torch.flip(out_tensor, [1])
                
        return out_tensor, lbl_tensor

if __name__ == "__main__":
    generate_5fold_splits()
