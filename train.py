"""
Retinoblastoma Primary Detection - 5-Fold Training Entrypoint
============================================================
Keeps Retinoblastoma as the primary detection target across all 5,070 images.
Evaluates both primary cancer detection (Sensitivity, Specificity, F1, AUC)
and 8-class differential clinical diagnosis.

Usage:
  # Train Fold 0 with ResNet-18 + CBAM Attention:
  python train.py --model resnet_cbam --fold 0 --epochs 5 --batch-size 16

  # Train all 5 folds sequentially:
  python train.py --model resnet_cbam --all-folds --epochs 5

  # Train with Swin Transformer or ConvNeXt-CBAM:
  python train.py --model convnext_cbam --fold 0 --epochs 5
"""

import argparse
from src.training.trainer import RetinoblastomaDedicatedTrainer

def main():
    parser = argparse.ArgumentParser(description="Retinoblastoma Primary Detection 5-Fold Pipeline")
    parser.add_argument("--model", type=str, default="resnet_cbam", 
                        choices=["resnet_cbam", "convnext_cbam", "swin_t", "efficientnet_b0"],
                        help="Model architecture with attention/transformer backbones")
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Specific fold to train (0-4)")
    parser.add_argument("--all-folds", action="store_true",
                        help="Train across all 5 folds sequentially")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs per fold")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Preprocessed image input dimension (e.g. 256 or 512)")
    parser.add_argument("--rb-weight", type=float, default=2.5,
                        help="Loss multiplier weighting for Retinoblastoma primary detection")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip training folds that already have saved best checkpoints and evaluate them instead")
    
    args = parser.parse_args()
    
    trainer = RetinoblastomaDedicatedTrainer(
        backbone_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        rb_loss_weight=args.rb_weight,
        target_size=args.image_size
    )
    
    if args.all_folds:
        trainer.run_all_5_folds(skip_existing=args.skip_existing)
    else:
        trainer.run_fold(fold_idx=args.fold, skip_existing=args.skip_existing)

if __name__ == "__main__":
    main()
