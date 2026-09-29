import torch
import torch.nn as nn
import torchvision.models as models
from .attention_modules import CBAM

class RetinoblastomaPrimaryDetector(nn.Module):
    """
    Retinoblastoma-First Dual-Head Diagnostic Architecture.
    
    1. Primary Detection Head (Binary):
       Drives the main detection objective: Is Retinoblastoma present? (Yes/No).
       Optimized for zero-miss clinical oncology screening (Sensitivity >= 98%).
       
    2. Differential Diagnosis Head (8-Class):
       Provides granular differential diagnosis:
       [Retinoblastoma, Pediatric Cataract, Pediatric ROP, Pediatric Normal RetCam,
        RCH, Uveal Melanoma, Choroidal Osteoma, Choroidal Hemangioma].
        
    Enhanced with CBAM Dual Attention (Channel + Spatial Attention) to highlight
    chalky-white tumor nodules, cottage-cheese calcifications, and peripheral seeds.
    """
    def __init__(
        self,
        backbone_name: str = "convnext_cbam",
        num_differential_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super(RetinoblastomaPrimaryDetector, self).__init__()
        self.backbone_name = backbone_name.lower()
        
        if "convnext" in self.backbone_name:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = models.convnext_tiny(weights=weights)
            self.features = base.features
            in_features = 768
            self.cbam = CBAM(in_planes=in_features, ratio=16)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.norm = base.classifier[0]  # LayerNorm
            self.flatten = base.classifier[1]
        elif "resnet" in self.backbone_name:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
            in_features = 512
            self.cbam = CBAM(in_planes=in_features, ratio=16)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.norm = nn.Identity()
            self.flatten = nn.Flatten(1)
        elif "swin" in self.backbone_name:
            weights = models.Swin_T_Weights.DEFAULT if pretrained else None
            base = models.swin_t(weights=weights)
            self.features = base.features
            in_features = base.head.in_features
            self.cbam = nn.Identity()  # Swin has native shifted-window self-attention
            self.pool = base.norm
            self.norm = nn.Identity()
            self.flatten = nn.Flatten(1)
        else:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.features = base.features
            in_features = 1280
            self.cbam = CBAM(in_planes=in_features, ratio=16)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.norm = nn.Identity()
            self.flatten = nn.Flatten(1)

        # Head 1: PRIMARY RETINOBLASTOMA DETECTION (Binary logit: RB vs Non-RB)
        self.primary_rb_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Linear(128, 2)  # [Non-RB, RB]
        )

        # Head 2: DIFFERENTIAL CLINICAL DIAGNOSIS (8 Classes)
        self.differential_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, num_differential_classes)
        )

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        feat = self.cbam(feat)
        pooled = self.pool(feat)
        pooled = self.norm(pooled)
        flat = self.flatten(pooled)
        
        # Output both primary RB detection and full differential
        rb_logits = self.primary_rb_head(flat)
        diff_logits = self.differential_head(flat)
        
        return {
            "rb_logits": rb_logits,
            "diff_logits": diff_logits
        }
