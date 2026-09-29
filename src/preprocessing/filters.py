import cv2
import numpy as np

def apply_ben_graham(image_bgr: np.ndarray, sigma: float = 10.0, alpha: float = 4.0, beta: float = -4.0, gamma: float = 128.0) -> np.ndarray:
    """
    Applies Ben Graham's local average color subtraction filter.
    Normalizes lighting variation, removes vignetting, and standardizes exposure
    across disparate ophthalmic imaging systems.
    
    Formula: I_out = alpha * I + beta * GaussianBlur(I, sigma) + gamma
    """
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
    filtered = cv2.addWeighted(image_bgr, alpha, blurred, beta, gamma)
    
    # Mask out the outer background so black boundary remains pure black
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    filtered[mask == 0] = 0
    return filtered

def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the Luminance channel in LAB color space.
    Accentuates chalky-white retinoblastoma calcifications and peripheral vitreous seeds.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr

def apply_unsharp_mask(image_bgr: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    """
    Unsharp masking to emphasize vascular edges and differential tumor borders.
    """
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigma)
    sharpened = cv2.addWeighted(image_bgr, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
