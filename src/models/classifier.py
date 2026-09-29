import torch
import torch.nn as nn
import torchvision.models as models
from .attention_modules import CBAM

class SwinTransformerClassifier(nn.Module):
    """
    Vision Transformer Architecture: Swin-T (Shifted Window Attention).
    Models multi-scale hierarchical spatial and local vascular contexts.
    """
    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.3):
        super(SwinTransformerClassifier, self).__init__()
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        self.backbone = models.swin_t(weights=weights)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class ConvNeXtCBAMClassifier(nn.Module):
    """
    Modern Pure-Convolutional Architecture: ConvNeXt-Tiny + CBAM Attention.
    Combines ConvNeXt 7x7 inverted bottleneck depthwise convs with spatial & channel attention.
    """
    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.3):
        super(ConvNeXtCBAMClassifier, self).__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base = models.convnext_tiny(weights=weights)
        
        self.features = base.features
        # ConvNeXt-Tiny final feature dimension is 768
        self.cbam = CBAM(in_planes=768, ratio=16)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            base.classifier[0],  # LayerNorm
            base.classifier[1],  # Flatten
            nn.Dropout(p=dropout),
            nn.Linear(768, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.cbam(feat)
        out = self.avgpool(feat)
        out = self.classifier(out)
        return out

class ResNet50CBAMClassifier(nn.Module):
    """
    ResNet-50 Enhanced with CBAM Dual Attention (Channel + Spatial).
    """
    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.3):
        super(ResNet50CBAMClassifier, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        # ResNet-50 layer4 output channels = 2048
        self.cbam = CBAM(in_planes=2048, ratio=16)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class EfficientNetB4Classifier(nn.Module):
    """
    EfficientNet-B4 with Native Squeeze-and-Excitation (SE) Channel Recalibration.
    """
    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.4):
        super(EfficientNetB4Classifier, self).__init__()
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b4(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def build_model(model_name: str = "convnext_cbam", num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    """
    Model Factory Function.
    Supported model_name options:
      - 'swin_t' (Swin Vision Transformer)
      - 'convnext_cbam' (ConvNeXt + CBAM Attention)
      - 'resnet50_cbam' (ResNet50 + CBAM Attention)
      - 'efficientnet_b4' (EfficientNet-B4 with SE blocks)
    """
    name = model_name.lower()
    if name in ["swin", "swin_t", "transformer"]:
        return SwinTransformerClassifier(num_classes=num_classes, pretrained=pretrained)
    elif name in ["convnext", "convnext_cbam", "convnext_tiny"]:
        return ConvNeXtCBAMClassifier(num_classes=num_classes, pretrained=pretrained)
    elif name in ["resnet", "resnet50", "resnet50_cbam"]:
        return ResNet50CBAMClassifier(num_classes=num_classes, pretrained=pretrained)
    elif name in ["efficientnet", "efficientnet_b4"]:
        return EfficientNetB4Classifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from: swin_t, convnext_cbam, resnet50_cbam, efficientnet_b4")
