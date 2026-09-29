"""
Attention U-Net Retinoblastoma Tumor Segmentation Training Pipeline
===================================================================
Trains Attention U-Net with Oktay et al. Attention Gates to delineate
exact retinoblastoma tumor boundaries, satellite vitreous seeds,
and quantify tumor area percentages.
"""

import os
import glob
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.attention_unet import AttentionUNet

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice
        
        return 0.5 * bce_loss + 0.5 * dice_loss

class TumorSegmentationDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, target_size: int = 256, augment: bool = True):
        self.img_paths = sorted(glob.glob(os.path.join(images_dir, "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*")))
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.img_paths[idx])
        mask_gray = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if img_bgr is None or mask_gray is None:
            raise FileNotFoundError(f"Cannot load pair {self.img_paths[idx]} or {self.mask_paths[idx]}")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_gray, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        
        # Binarize mask: >128 is tumor
        mask_binary = (mask_resized > 128).astype(np.float32)
        
        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img_resized = np.fliplr(img_resized)
                mask_binary = np.fliplr(mask_binary)
            if np.random.rand() > 0.5:
                img_resized = np.flipud(img_resized)
                mask_binary = np.flipud(mask_binary)
                
        # Normalization
        img_tensor = torch.from_numpy(img_resized.copy()).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_binary.copy()).unsqueeze(0).float()
        
        return img_tensor, mask_tensor

def compute_dice_score(preds, targets, threshold=0.5):
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds_bin * targets).sum()
    dice = (2. * intersection + 1e-6) / (preds_bin.sum() + targets.sum() + 1e-6)
    return float(dice)

def train_segmentation(
    images_dir: str = r"Dataset\03_Tumor_Segmentation\retinoblastoma-tumor-segmentation-main\Sample Results of Supervised Technique\Original Images",
    masks_dir: str = r"Dataset\03_Tumor_Segmentation\retinoblastoma-tumor-segmentation-main\Sample Results of Supervised Technique\Segmentations by Model",
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 5e-4,
    target_size: int = 256,
    checkpoints_dir: str = "checkpoints"
):
    print("\n" + "="*70)
    print("  STARTING ATTENTION U-NET RETINOBLASTOMA TUMOR SEGMENTATION TRAINING")
    print(f"  Images: {images_dir}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | Size: {target_size}x{target_size}")
    print("="*70 + "\n", flush=True)
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = TumorSegmentationDataset(images_dir, masks_dir, target_size=target_size, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AttentionUNet(in_channels=3, num_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_dice = -1.0
    dest_ckpt = os.path.join(checkpoints_dir, "rb_attention_unet_best.pt")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_dice = 0.0
        t0 = time.time()
        
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            total_dice += compute_dice_score(logits, masks) * imgs.size(0)
            
        scheduler.step()
        epoch_loss = total_loss / len(dataset)
        epoch_dice = total_dice / len(dataset)
        elapsed = time.time() - t0
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Loss: {epoch_loss:.4f} | Dice Score: {epoch_dice*100:.2f}%", flush=True)
        
        if epoch_dice > best_dice:
            best_dice = epoch_dice
            torch.save(model.state_dict(), dest_ckpt)
            
    print("\n" + "="*70)
    print(f"ATTENTION U-NET TRAINING COMPLETE! Best Dice: {best_dice*100:.2f}%")
    print(f"Checkpoint saved -> {dest_ckpt}")
    print("="*70 + "\n", flush=True)

if __name__ == "__main__":
    train_segmentation()
