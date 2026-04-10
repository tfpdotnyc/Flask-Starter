"""
ATTONE — Color Profile Extractor
Extracts a tonal profile dictionary from any image (JPG or decoded RAW).
This is the measurement core: control images and submission images both
run through this function. The delta between their outputs drives correction.
"""

import cv2
import numpy as np


def extract_profile(image_path: str) -> dict:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    l_channel = lab[:, :, 0].astype(np.float64)
    a_channel = lab[:, :, 1].astype(np.float64)
    b_channel = lab[:, :, 2].astype(np.float64)

    h_channel = hsv[:, :, 0].astype(np.float64)
    s_channel = hsv[:, :, 1].astype(np.float64)

    b_blue, g_green, r_red = (
        bgr[:, :, 0].astype(np.float64),
        bgr[:, :, 1].astype(np.float64),
        bgr[:, :, 2].astype(np.float64),
    )

    mean_r = float(np.mean(r_red))
    mean_b = float(np.mean(b_blue))
    rb_ratio = mean_r / mean_b if mean_b > 0 else 1.0
    est_temp_k = 6500.0 * rb_ratio

    hue_hist, _ = np.histogram(h_channel.ravel(), bins=180, range=(0, 180))
    dominant_hue_angle = float(np.argmax(hue_hist)) * 2.0

    profile = {
        "luminance_mean": round(float(np.mean(l_channel)), 2),
        "contrast_std": round(float(np.std(l_channel)), 2),
        "saturation_mean": round(float(np.mean(s_channel)), 2),
        "temperature_est_k": round(est_temp_k, 0),
        "dominant_hue_angle": round(dominant_hue_angle, 1),
        "shadow_5pct": round(float(np.percentile(l_channel, 5)), 2),
        "highlight_95pct": round(float(np.percentile(l_channel, 95)), 2),
        "blacks_1pct": round(float(np.percentile(l_channel, 1)), 2),
        "whites_99pct": round(float(np.percentile(l_channel, 99)), 2),
        "a_mean": round(float(np.mean(a_channel)), 2),
        "b_mean": round(float(np.mean(b_channel)), 2),
        "vibrance": round(float(np.std(s_channel)), 2),
    }

    return profile


if __name__ == "__main__":
    import json

    path = "test_images/sample_decoded.jpg"
    result = extract_profile(path)
    print(f"Profile for: {path}\n")
    print(json.dumps(result, indent=2))
