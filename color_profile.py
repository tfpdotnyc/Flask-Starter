"""
ATTONE — Color Profile Extractor
Extracts a tonal profile dictionary from any image (JPG, PNG, TIFF, or RAW).
This is the measurement core: control images and submission images both
run through this function. The delta between their outputs drives correction.
"""

import os
import cv2
import numpy as np

RAW_EXTENSIONS = {".cr2", ".cr3", ".arw", ".nef", ".nrw", ".dng", ".raf", ".orf", ".rw2"}


def _read_image_bgr(image_path: str) -> np.ndarray:
    ext = os.path.splitext(image_path)[1].lower()

    if ext in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError:
            raise RuntimeError(
                "RAW file detected (%s) but rawpy is not installed. "
                "Install rawpy to process RAW camera files." % ext.upper()
            )
        try:
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(
                "Failed to decode RAW file '%s' (%s): %s"
                % (os.path.basename(image_path), ext.upper(), str(e))
            )

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        supported = "JPG, PNG, TIFF, or RAW (%s)" % ", ".join(
            sorted(e.upper() for e in RAW_EXTENSIONS)
        )
        raise FileNotFoundError(
            "Could not read '%s'. The file may be corrupted or in an unsupported format. "
            "Supported formats: %s" % (os.path.basename(image_path), supported)
        )
    return bgr


def extract_profile(image_path: str) -> dict:
    bgr = _read_image_bgr(image_path)

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
