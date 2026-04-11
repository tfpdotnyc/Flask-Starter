"""
ATTONE AMD-01 | correction/background_normalizer.py
Replaces: background_normalizer.py (root)
Updated interface: receives background_alpha mask (not face_bounding_box).
No correction bleeds into face or skin regions.
No access to face region — background only.
"""
import numpy as np
import cv2


def normalize_background(
    image: np.ndarray,
    control_bg_profile: dict | None,
    background_alpha: np.ndarray
) -> np.ndarray:
    """
    Apply background correction exclusively within background_alpha mask.

    Args:
        image              - RGB image (uint8)
        control_bg_profile - dict with target bg characteristics, or None
        background_alpha   - float32 mask from face_masks.py
    """
    if control_bg_profile is None:
        return image

    bg_mask_bool = background_alpha > 0.5
    if not bg_mask_bool.any():
        return image

    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

    current_bg = {
        "mean_L": float(img_lab[:, :, 0][bg_mask_bool].mean()),
        "mean_a": float(img_lab[:, :, 1][bg_mask_bool].mean()),
        "mean_b": float(img_lab[:, :, 2][bg_mask_bool].mean()),
    }

    delta_L = control_bg_profile.get("mean_L", current_bg["mean_L"]) - current_bg["mean_L"]
    delta_a = control_bg_profile.get("mean_a", current_bg["mean_a"]) - current_bg["mean_a"]
    delta_b = control_bg_profile.get("mean_b", current_bg["mean_b"]) - current_bg["mean_b"]

    mask_3d = background_alpha[:, :, np.newaxis]
    corrected = img_lab.copy()
    corrected[:, :, 0] = np.clip(img_lab[:, :, 0] + delta_L * mask_3d[:, :, 0], 0, 255)
    corrected[:, :, 1] = np.clip(img_lab[:, :, 1] + delta_a * mask_3d[:, :, 0], 0, 255)
    corrected[:, :, 2] = np.clip(img_lab[:, :, 2] + delta_b * mask_3d[:, :, 0], 0, 255)

    return cv2.cvtColor(corrected.astype(np.uint8), cv2.COLOR_LAB2RGB)
