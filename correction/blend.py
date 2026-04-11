"""
ATTONE AMD-01 | correction/blend.py
Feather-Blend Compositor — sole writer of final output pixels.

HARD LOCK: No cropped face ROI may ever be directly written back
into the final image. All edits projected through soft alpha masks.
"""
import numpy as np
import cv2


def feather_blend(
    global_corrected: np.ndarray,
    face_corrected: np.ndarray,
    face_influence_alpha: np.ndarray,
) -> np.ndarray:
    """
    Blend face_corrected over global_corrected using face_influence_alpha.
    Formula: output = global * (1 - mask) + face * mask
    Both corrections applied to full image first.
    Mask is the sole arbiter of the blend.
    """
    mask_3ch = np.stack([face_influence_alpha] * 3, axis=-1)
    blended = (
        global_corrected.astype(np.float32) * (1.0 - mask_3ch)
        + face_corrected.astype(np.float32) * mask_3ch
    )
    return np.clip(blended, 0, 255).astype(np.uint8)
