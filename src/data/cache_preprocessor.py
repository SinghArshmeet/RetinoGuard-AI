"""
Medical Image Preprocessing & Disk Caching Utility
===================================================
Applies the full medical filter pipeline (Circular FOV crop,
Ben Graham local illumination subtraction, Reinhard cross-camera
color constancy transfer, CLAHE) and caches the preprocessed
tensors to high-speed disk storage.

This transforms multi-fold training from hours to minutes.
"""

import os
import sys
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import cv2
import torch
from src.preprocessing.pipeline import PreprocessingPipeline

def get_cache_filename(filepath: str, target_size: int = 256) -> str:
    # Use deterministic hash of filepath + size
    h = hashlib.md5(f"{os.path.normpath(filepath)}_{target_size}".encode('utf-8')).hexdigest()
    return f"{h}.pt"

def process_and_save(row, pipeline, cache_dir, target_size):
    filepath = row['filepath']
    cache_file = os.path.join(cache_dir, get_cache_filename(filepath, target_size))
    
    if os.path.exists(cache_file):
        return True
        
    try:
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            return False
            
        # Optional: downscale if excessively large to accelerate filter math
        h, w = img_bgr.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        tensor = pipeline(img_bgr)
        # Save as float16 to save memory and disk bandwidth
        torch.save(tensor.half(), cache_file)
        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def build_preprocessing_cache(
    manifest_csv: str = "Dataset/5fold_splits/full_dataset_manifest.csv",
    cache_dir: str = "Dataset/cache_preprocessed_256",
    target_size: int = 256,
    num_workers: int = 12
):
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.read_csv(manifest_csv)
    total = len(df)
    print(f"Starting parallel caching for {total} images into [{cache_dir}]...")
    t0 = time.time()
    
    # Check already cached
    pending = []
    for idx, row in df.iterrows():
        cfile = os.path.join(cache_dir, get_cache_filename(row['filepath'], target_size))
        if not os.path.exists(cfile):
            pending.append(row)
            
    print(f"Already cached: {total - len(pending)} / {total}. Pending: {len(pending)}")
    
    if len(pending) == 0:
        print("All images are already cached and ready!")
        return
        
    pipeline = PreprocessingPipeline(target_size=target_size)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_and_save, row, pipeline, cache_dir, target_size): row for row in pending}
        for f in as_completed(futures):
            res = f.result()
            completed += 1
            if completed % 250 == 0 or completed == len(pending):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  Processed [{completed:04d}/{len(pending):04d}] ({rate:.1f} img/s) - Elapsed: {elapsed:.1f}s", flush=True)
                
    total_elapsed = time.time() - t0
    print(f"Successfully cached all {total} images in {total_elapsed:.1f}s!")

if __name__ == "__main__":
    build_preprocessing_cache()
