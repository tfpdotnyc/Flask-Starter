"""
ATTONE — Correction Engine (Exposure + Saturation)
Adjusts an image so its luminance and saturation move toward a target profile.
Returns the corrected image as a Pillow Image object.
"""

import cv2
import numpy as np
from PIL import Image


def apply_correction(
    image: Image.Image,
    source_profile: dict,
    target_profile: dict,
) -> Image.Image:
    arr = np.array(image)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float64)

    lum_delta = target_profile["luminance_mean"] - source_profile["luminance_mean"]
    lab[:, :, 0] = np.clip(lab[:, :, 0] + lum_delta, 0, 255)

    src_sat = source_profile["saturation_mean"]
    tgt_sat = target_profile["saturation_mean"]
    if src_sat > 0:
        sat_ratio = tgt_sat / src_sat
    else:
        sat_ratio = 1.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_ratio, 0, 255)

    lab_corrected = lab.astype(np.uint8)
    bgr_from_lab = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    hsv_corrected = hsv.astype(np.uint8)
    bgr_from_hsv = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

    blended = cv2.addWeighted(bgr_from_lab, 0.5, bgr_from_hsv, 0.5, 0)

    rgb_out = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_out)
