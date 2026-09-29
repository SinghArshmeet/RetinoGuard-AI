import cv2
import numpy as np

def crop_fov_and_pad(image_bgr: np.ndarray, target_size: int = 512, margin_ratio: float = 0.02) -> np.ndarray:
    """
    Detects the circular fundus Field-of-View (FOV), crops the bounding box,
    and aspect-pads to a square image of size (target_size x target_size).
    
    Args:
        image_bgr: Input BGR image (OpenCV format)
        target_size: Desired square output dimension (default: 512)
        margin_ratio: Extra padding around the detected circle
        
    Returns:
        Square preprocessed BGR image of shape (target_size, target_size, 3)
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid or empty input image provided.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Threshold to locate the illuminated retina region
    # Use Otsu or simple thresholding with blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    
    # Clean up small noise/artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = image_bgr.shape[:2]
    if contours:
        # Find the largest contour which represents the retina disk
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest_cnt)
        
        # Verify the contour is large enough to be the retina
        if cw > w * 0.3 and ch > h * 0.3:
            margin_x = int(cw * margin_ratio)
            margin_y = int(ch * margin_ratio)
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + cw + margin_x)
            y2 = min(h, y + ch + margin_y)
            
            cropped = image_bgr[y1:y2, x1:x2]
        else:
            cropped = image_bgr
    else:
        cropped = image_bgr

    # 2. Aspect-preserving square padding
    ch, cw = cropped.shape[:2]
    max_side = max(ch, cw)
    
    padded = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    offset_y = (max_side - ch) // 2
    offset_x = (max_side - cw) // 2
    
    padded[offset_y:offset_y + ch, offset_x:offset_x + cw] = cropped
    
    # 3. Resize to target dimension
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized
