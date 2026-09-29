import cv2
import numpy as np
import torch
from .fov_crop import crop_fov_and_pad
from .filters import apply_ben_graham, apply_clahe
from .color_norm import ReinhardColorNormalizer

class PreprocessingPipeline:
    """
    Unified Medical Image Preprocessing Pipeline for Retinoblastoma AI.
    Executes:
    1. Circular FOV Detection & Aspect-preserving Padding
    2. Illumination Normalization (Ben Graham filter)
    3. Contrast Optimization (CLAHE)
    4. Cross-Camera Color Constancy Normalization (Reinhard)
    5. Tensor conversion with ImageNet Z-Score standardization
    """
    def __init__(
        self,
        target_size: int = 512,
        use_ben_graham: bool = True,
        use_clahe: bool = True,
        use_reinhard: bool = True,
        imagenet_normalize: bool = True
    ):
        self.target_size = target_size
        self.use_ben_graham = use_ben_graham
        self.use_clahe = use_clahe
        self.use_reinhard = use_reinhard
        self.imagenet_normalize = imagenet_normalize
        
        self.reinhard_normalizer = ReinhardColorNormalizer() if use_reinhard else None
        
        # Standard ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def process_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Processes raw BGR image and returns preprocessed BGR image."""
        # 1. FOV crop and aspect-pad to target size
        img = crop_fov_and_pad(image_bgr, target_size=self.target_size)
        
        # 2. Reinhard color transfer (domain alignment)
        if self.use_reinhard and self.reinhard_normalizer:
            img = self.reinhard_normalizer.normalize(img)
            
        # 3. Ben Graham illumination correction
        if self.use_ben_graham:
            img = apply_ben_graham(img, sigma=10.0)
            
        # 4. CLAHE contrast enhancement
        if self.use_clahe:
            img = apply_clahe(img, clip_limit=2.0)
            
        return img

    def to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        """
        Converts processed BGR image to normalized PyTorch tensor of shape (3, H, W).
        Pixel values normalized to [0, 1] and standardized with ImageNet mean/std.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        if self.imagenet_normalize:
            rgb = (rgb - self.mean) / self.std
            
        # HWC -> CHW
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1))
        return tensor

    def __call__(self, image_bgr: np.ndarray) -> torch.Tensor:
        processed_bgr = self.process_image(image_bgr)
        return self.to_tensor(processed_bgr)
