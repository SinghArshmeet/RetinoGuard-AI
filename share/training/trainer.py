import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from src.data.group_kfold_split import RetinoblastomaDataset
from src.models.retinoblastoma_detector import RetinoblastomaPrimaryDetector
from src.training.losses import FocalLoss
from src.evaluation.metrics import aggregate_5fold_results

class RetinoblastomaDedicatedTrainer:
    """
    Dedicated Retinoblastoma-First Trainer.
    Keeps Retinoblastoma as the primary detection target while providing
    full 8-class differential diagnosis across mimickers and other intraocular tumors.
    """
    def __init__(
        self,
        backbone_name: str = "resnet_cbam",
        batch_size: int = 16,
        epochs: int = 10,
        lr: float = 3e-4,
        rb_loss_weight: float = 2.5,
        target_size: int = 256,
        device: str = None,
        checkpoints_dir: str = "checkpoints",
        results_dir: str = "results"
    ):
        self.backbone_name = backbone_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.rb_loss_weight = rb_loss_weight
        self.target_size = target_size
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.checkpoints_dir = checkpoints_dir
        self.results_dir = results_dir
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    def train_one_epoch(self, model: nn.Module, loader: DataLoader, focal_rb: nn.Module, focal_diff: nn.Module, optimizer: torch.optim.Optimizer):
        model.train()
        total_loss = 0.0
        total_batches = len(loader)
        
        for b_idx, (images, labels) in enumerate(loader, 1):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Ground truth for primary RB detection: 1 if Retinoblastoma (label 0), else 0
            rb_targets = (labels == 0).long()
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss_rb = focal_rb(outputs["rb_logits"], rb_targets)
            loss_diff = focal_diff(outputs["diff_logits"], labels)
            
            loss = (self.rb_loss_weight * loss_rb) + loss_diff
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item() * images.size(0)
            total_loss += batch_loss
            
            if b_idx % 30 == 0 or b_idx == total_batches:
                avg_b_loss = total_loss / (b_idx * loader.batch_size)
                print(f"    Batch [{b_idx:03d}/{total_batches:03d}] Avg Loss: {avg_b_loss:.4f}", flush=True)
            
        return total_loss / len(loader.dataset)

    def evaluate(self, model: nn.Module, loader: DataLoader):
        model.eval()
        
        all_rb_preds = []
        all_rb_targets = []
        all_rb_probs = []
        
        all_diff_preds = []
        all_diff_targets = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = model(images)
                
                # Primary RB Detection evaluation
                rb_probs = torch.softmax(outputs["rb_logits"], dim=1)[:, 1]
                rb_preds = (rb_probs >= 0.45).long()  # Clinically sensitive threshold
                rb_targets = (labels == 0).long()
                
                all_rb_preds.extend(rb_preds.cpu().numpy())
                all_rb_targets.extend(rb_targets.numpy())
                all_rb_probs.extend(rb_probs.cpu().numpy())
                
                # Differential diagnosis evaluation
                diff_probs = torch.softmax(outputs["diff_logits"], dim=1)
                diff_preds = torch.argmax(diff_probs, dim=1)
                all_diff_preds.extend(diff_preds.cpu().numpy())
                all_diff_targets.extend(labels.numpy())

        y_true_rb = np.array(all_rb_targets)
        y_pred_rb = np.array(all_rb_preds)
        y_prob_rb = np.array(all_rb_probs)
        
        # Retinoblastoma primary detection metrics
        rb_sens = recall_score(y_true_rb, y_pred_rb, zero_division=0)
        rb_prec = precision_score(y_true_rb, y_pred_rb, zero_division=0)
        rb_f1 = f1_score(y_true_rb, y_pred_rb, zero_division=0)
        
        cm_rb = confusion_matrix(y_true_rb, y_pred_rb, labels=[0, 1])
        tn, fp, fn, tp = cm_rb.ravel() if cm_rb.size == 4 else (0, 0, 0, 0)
        rb_spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        
        try:
            rb_auc = roc_auc_score(y_true_rb, y_prob_rb)
        except Exception:
            rb_auc = np.nan
            
        # Overall 8-class differential accuracy and macro F1
        overall_acc = accuracy_score(all_diff_targets, all_diff_preds)
        macro_f1 = f1_score(all_diff_targets, all_diff_preds, average='macro', zero_division=0)

        return {
            "rb_sensitivity": round(float(rb_sens), 4),
            "rb_specificity": round(float(rb_spec), 4),
            "rb_precision": round(float(rb_prec), 4),
            "rb_f1": round(float(rb_f1), 4),
            "rb_auc": round(float(rb_auc), 4) if not np.isnan(rb_auc) else "N/A",
            "overall_acc": round(float(overall_acc), 4),
            "macro_f1": round(float(macro_f1), 4)
        }

    def run_fold(self, fold_idx: int, splits_dir: str = "Dataset/5fold_splits", skip_existing: bool = False) -> dict:
        fold_csv = os.path.join(splits_dir, f"fold_{fold_idx}.csv")
        print(f"\n{'='*25} RETINOBLASTOMA DETECTION: FOLD {fold_idx} / 5 {'='*25}", flush=True)
        
        train_ds = RetinoblastomaDataset(fold_csv, split="train", target_size=self.target_size, augment=True)
        val_ds = RetinoblastomaDataset(fold_csv, split="val", target_size=self.target_size, augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        model = RetinoblastomaPrimaryDetector(
            backbone_name=self.backbone_name,
            num_differential_classes=8,
            pretrained=True
        ).to(self.device)
        
        checkpoint_path = os.path.join(self.checkpoints_dir, f"rb_detector_{self.backbone_name}_best_fold_{fold_idx}.pt")
        
        if skip_existing and os.path.exists(checkpoint_path):
            print(f"[Fold {fold_idx}] Existing checkpoint detected: {checkpoint_path}", flush=True)
            print(f"[Fold {fold_idx}] Evaluating on validation cohort ({len(val_ds)} images)...", flush=True)
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
            except Exception:
                model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            val_m = self.evaluate(model, val_loader)
            val_m["fold"] = fold_idx
            print(f"[Fold {fold_idx} Evaluated Metrics] RB-Sens: {val_m['rb_sensitivity']*100:.1f}%, RB-Spec: {val_m['rb_specificity']*100:.1f}%, RB-F1: {val_m['rb_f1']:.4f} | Acc: {val_m['overall_acc']:.4f}", flush=True)
            return val_m
        
        focal_rb = FocalLoss(gamma=2.0)
        focal_diff = FocalLoss(gamma=2.0)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        best_score = -1.0
        best_metrics = {}
        
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            loss = self.train_one_epoch(model, train_loader, focal_rb, focal_diff, optimizer)
            val_m = self.evaluate(model, val_loader)
            scheduler.step()
            elapsed = time.time() - t0
            
            # Prioritize Retinoblastoma Sensitivity (Zero-Miss target) and RB F1
            score = (val_m["rb_sensitivity"] * 0.6) + (val_m["rb_f1"] * 0.4)
            print(f"Epoch [{epoch:02d}/{self.epochs:02d}] ({elapsed:.1f}s) | Loss: {loss:.4f} | RB-Sens: {val_m['rb_sensitivity']*100:.1f}% | RB-Spec: {val_m['rb_specificity']*100:.1f}% | RB-F1: {val_m['rb_f1']:.4f} | Acc: {val_m['overall_acc']:.4f}", flush=True)
            
            if score > best_score:
                best_score = score
                best_metrics = val_m.copy()
                best_metrics["fold"] = fold_idx
                torch.save(model.state_dict(), checkpoint_path)
                
        print(f"[Fold {fold_idx} Best Metrics] RB-Sens: {best_metrics['rb_sensitivity']*100:.1f}%, RB-Spec: {best_metrics['rb_specificity']*100:.1f}%, RB-F1: {best_metrics['rb_f1']:.4f}", flush=True)
        print(f"Saved Checkpoint -> {checkpoint_path}", flush=True)
        return best_metrics

    def run_all_5_folds(self, splits_dir: str = "Dataset/5fold_splits", skip_existing: bool = False) -> pd.DataFrame:
        print(f"\n======================================================================", flush=True)
        print(f"  LAUNCHING RETINOBLASTOMA PRIMARY DETECTION 5-FOLD CV PIPELINE", flush=True)
        print(f"  Backbone: [{self.backbone_name}] | Device: [{self.device}]", flush=True)
        print(f"======================================================================", flush=True)
        
        from src.data.cache_preprocessor import build_preprocessing_cache
        build_preprocessing_cache(target_size=self.target_size)
        
        fold_results = []
        for k in range(5):
            res = self.run_fold(fold_idx=k, splits_dir=splits_dir, skip_existing=skip_existing)
            fold_results.append(res)
            
        folds_df = pd.DataFrame(fold_results)
        summary_df = aggregate_5fold_results(fold_results)
        
        out_csv = os.path.join(self.results_dir, f"rb_detection_{self.backbone_name}_5fold_results.csv")
        folds_df.to_csv(out_csv, index=False)
        
        summary_csv = os.path.join(self.results_dir, f"rb_detection_{self.backbone_name}_5fold_summary.csv")
        summary_df.to_csv(summary_csv)
        
        print("\n" + "="*70, flush=True)
        print(f"FINAL 5-FOLD RETINOBLASTOMA DETECTION SUMMARY [{self.backbone_name}]:", flush=True)
        print("="*70, flush=True)
        print(summary_df.to_string(), flush=True)
        print(f"\nResults saved to: {out_csv} and {summary_csv}", flush=True)
        return summary_df
