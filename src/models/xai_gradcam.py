"""
Explainable AI (XAI) Grad-CAM Module for Retinoblastoma Detection
==================================================================
Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
to highlight discriminative retinal lesion regions (nodular cores,
calcifications, peripheral seeding) guiding the ResNet-50 + CBAM decisions.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks"
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional

class RetinoblastomaGradCAM:
    """
    Grad-CAM implementation tailored for RetinoblastomaPrimaryDetector.
    Hooks into the final convolutional feature extraction stage (base.layer4)
    prior to CBAM dual attention pooling.
    """
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        
        # If target layer not specified, hook the last layer of features
        if target_layer is None:
            # For ResNet50 in features: layer4 is the final block
            if hasattr(self.model, "features") and len(self.model.features) > 7:
                self.target_layer = self.model.features[7]  # layer4 in nn.Sequential
            else:
                self.target_layer = self.model.features[-1]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int = 1  # 1 = Retinoblastoma
    ) -> np.ndarray:
        """
        Generates 2D normalized Grad-CAM heatmap [0, 1].
        
        Args:
            input_tensor: (1, 3, H, W) normalized image tensor
            target_class_idx: 1 for Retinoblastoma malignancy head
        """
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(input_tensor)
        rb_logits = outputs["rb_logits"]  # (1, 2) [Non-RB, RB]
        
        # Score for target class
        score = rb_logits[0, target_class_idx]
        
        # Backward pass
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling of gradients: weights alpha_k
        alpha_k = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of forward activation maps
        cam = torch.sum(alpha_k * self.activations, dim=1, keepdim=True)  # (1, 1, H', W')
        
        # Pass through ReLU (only features that positively correlate with class)
        cam = F.relu(cam)
        
        # Interpolate to input image dimensions
        _, _, H, W = input_tensor.shape
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-7:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

    def overlay_heatmap(
        self,
        image_bgr: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.55,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Blends 2D Grad-CAM heatmap with original BGR fundus photography.
        """
        h, w = image_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Colorize
        colored_cam = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Alpha blend
        blended = cv2.addWeighted(image_bgr, 1.0 - alpha, colored_cam, alpha, 0)
        return blended

    def remove_hooks(self):
        """Clean up hooks on deletion."""
        self.forward_handle.remove()
        self.backward_handle.remove()
