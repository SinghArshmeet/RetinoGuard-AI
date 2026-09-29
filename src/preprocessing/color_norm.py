import cv2
import numpy as np

class ReinhardColorNormalizer:
    """
    Reinhard Color Constancy Normalizer for Cross-Camera Domain Adaptation.
    Standardizes infant handheld RetCam images and adult tabletop fundus images
    into a common color reference distribution in LAB space.
    """
    def __init__(self, target_mean: np.ndarray = None, target_std: np.ndarray = None):
        # Default empirical target statistics (mean and std in LAB color space for high-quality fundus)
        # L in [0, 255], A in [0, 255], B in [0, 255]
        if target_mean is None:
            self.target_mean = np.array([125.0, 148.0, 155.0], dtype=np.float32)
        else:
            self.target_mean = np.array(target_mean, dtype=np.float32)
            
        if target_std is None:
            self.target_std = np.array([45.0, 14.0, 18.0], dtype=np.float32)
        else:
            self.target_std = np.array(target_std, dtype=np.float32)

    def fit_from_reference(self, reference_bgr: np.ndarray):
        """Fits target statistics from a reference pristine clinical fundus image."""
        lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Compute stats only on non-black pixels
        mask = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY) > 15
        if np.sum(mask) > 100:
            pixels = lab[mask]
            self.target_mean = np.mean(pixels, axis=0)
            self.target_std = np.std(pixels, axis=0) + 1e-6

    def normalize(self, source_bgr: np.ndarray) -> np.ndarray:
        """Transfers source image color distribution to reference distribution."""
        mask = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY) > 15
        if np.sum(mask) < 100:
            return source_bgr

        lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = lab[mask]
        
        src_mean = np.mean(pixels, axis=0)
        src_std = np.std(pixels, axis=0) + 1e-6
        
        # Scale and shift in LAB space
        norm_pixels = (pixels - src_mean) * (self.target_std / src_std) + self.target_mean
        norm_pixels = np.clip(norm_pixels, 0, 255)
        
        normalized_lab = np.zeros_like(lab)
        normalized_lab[mask] = norm_pixels
        normalized_bgr = cv2.cvtColor(normalized_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Ensure non-retina background stays pure black
        normalized_bgr[~mask] = 0
        return normalized_bgr
