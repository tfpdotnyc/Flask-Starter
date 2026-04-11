"""
ATTONE AMD-01 | correction/skin_guard.py
Replaces: skin_tone_guard.py
Updated interface: receives skin_core_alpha mask (not bounding_box).
Skin tone guard rule unchanged: +/-8% L* luminance clamp.
Dark subjects not brightened. Fair subjects not darkened.
"""
import numpy as np
import cv2


def apply_skin_guard(
    original_img: np.ndarray,
    corrected_img: np.ndarray,
    skin_core_alpha: np.ndarray,
    clamp_pct: float = 0.08
) -> np.ndarray:
    """
    Enforce the PRD +/-8% L* luminance clamp within skin_core_alpha region.

    Args:
        original_img     - original RGB image (uint8)
        corrected_img    - globally corrected RGB image (uint8)
        skin_core_alpha  - float32 mask from face_masks.py (0.0-1.0)
        clamp_pct        - max luminance change in skin region (default 0.08)
    """
    orig_lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    corr_lab = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2LAB).astype(np.float32)

    orig_L = orig_lab[:, :, 0]
    corr_L = corr_lab[:, :, 0]

    delta_L     = corr_L - orig_L
    max_allowed = orig_L * clamp_pct
    clamped     = np.clip(delta_L, -max_allowed, max_allowed)

    guarded_L = (
        (orig_L + clamped)  * skin_core_alpha
        + corr_L            * (1.0 - skin_core_alpha)
    )

    result_lab        = corr_lab.copy()
    result_lab[:, :, 0] = np.clip(guarded_L, 0, 255)
    return cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
