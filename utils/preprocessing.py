import cv2
import numpy as np
from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE, SATURATION_BOOST


def enhance_frame(frame):
    """Enhance frame contrast and color to make fire more visible.

    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
    lightness channel and boosts color saturation to improve fire detection
    in dark, smoky, or low-light conditions.

    Args:
        frame: OpenCV BGR image.

    Returns:
        Enhanced BGR image with improved contrast and saturation.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    l_enhanced = clahe.apply(l)

    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Boost saturation to make fire colors pop
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * SATURATION_BOOST, 0, 255).astype(np.uint8)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return enhanced
