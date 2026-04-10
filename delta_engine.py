"""
ATTONE — Delta Engine
Computes signed correction recipes between source and target profiles,
then applies proportional corrections across all images in a cluster.
Translation layer between measurement (color_profile) and action (correction).
"""

import os
import time
import cv2
import numpy as np
from PIL import Image
from color_profile import extract_profile


CORRECTION_KEYS = [
    "luminance_mean",
    "contrast_std",
    "saturation_mean",
    "temperature_est_k",
    "shadow_5pct",
    "highlight_95pct",
    "blacks_1pct",
    "whites_99pct",
    "a_mean",
    "b_mean",
    "vibrance",
]

SCALE_FACTORS = {
    "luminance_mean": 1.0,
    "contrast_std": 0.6,
    "saturation_mean": 0.8,
    "temperature_est_k": 0.5,
    "shadow_5pct": 0.7,
    "highlight_95pct": 0.7,
    "blacks_1pct": 0.5,
    "whites_99pct": 0.5,
    "a_mean": 0.6,
    "b_mean": 0.6,
    "vibrance": 0.5,
}


def compute_delta(source_profile: dict, target_profile: dict) -> dict:
    delta = {}
    for key in CORRECTION_KEYS:
        src = source_profile.get(key, 0.0)
        tgt = target_profile.get(key, 0.0)
        raw_diff = tgt - src
        scale = SCALE_FACTORS.get(key, 1.0)
        delta[key] = round(raw_diff * scale, 3)
    return delta


def _apply_delta_to_image(image: Image.Image, delta: dict) -> Image.Image:
    arr = np.array(image)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float64)

    l_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    if "luminance_mean" in delta:
        l_ch += delta["luminance_mean"]

    if "contrast_std" in delta and delta["contrast_std"] != 0:
        l_mean = np.mean(l_ch)
        if delta["contrast_std"] > 0:
            factor = 1.0 + (delta["contrast_std"] / 100.0)
        else:
            factor = 1.0 + (delta["contrast_std"] / 100.0)
        l_ch = l_mean + (l_ch - l_mean) * factor

    if "shadow_5pct" in delta:
        shadow_mask = (l_ch < 50).astype(np.float64)
        l_ch += delta["shadow_5pct"] * shadow_mask

    if "highlight_95pct" in delta:
        highlight_mask = (l_ch > 200).astype(np.float64)
        l_ch += delta["highlight_95pct"] * highlight_mask

    if "blacks_1pct" in delta:
        blacks_mask = (l_ch < 20).astype(np.float64)
        l_ch += delta["blacks_1pct"] * blacks_mask

    if "whites_99pct" in delta:
        whites_mask = (l_ch > 230).astype(np.float64)
        l_ch += delta["whites_99pct"] * whites_mask

    if "a_mean" in delta:
        a_ch += delta["a_mean"]

    if "b_mean" in delta:
        b_ch += delta["b_mean"]

    if "saturation_mean" in delta and delta["saturation_mean"] != 0:
        s_mean = np.mean(s_ch)
        if s_mean > 0:
            sat_shift = delta["saturation_mean"]
            s_ch = s_ch + sat_shift
        else:
            s_ch = s_ch + delta["saturation_mean"]

    if "vibrance" in delta and delta["vibrance"] != 0:
        low_sat_mask = 1.0 - (s_ch / 255.0)
        s_ch += delta["vibrance"] * low_sat_mask

    if "temperature_est_k" in delta and delta["temperature_est_k"] != 0:
        temp_shift = delta["temperature_est_k"]
        warmth = temp_shift / 6500.0
        b_ch += warmth * 2.0
        a_ch -= warmth * 0.5

    lab[:, :, 0] = np.clip(l_ch, 0, 255)
    lab[:, :, 1] = np.clip(a_ch, 0, 255)
    lab[:, :, 2] = np.clip(b_ch, 0, 255)
    hsv[:, :, 1] = np.clip(s_ch, 0, 255)

    bgr_from_lab = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    hsv[:, :, 2] = np.clip(v_ch, 0, 255)
    bgr_from_hsv = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    blended = cv2.addWeighted(bgr_from_lab, 0.6, bgr_from_hsv, 0.4, 0)
    rgb_out = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_out)


def apply_delta_to_cluster(
    image_paths: list[str],
    delta: dict,
    output_dir: str = None,
    quality: int = 95,
) -> dict:
    results = {
        "total": len(image_paths),
        "corrected": 0,
        "failed": 0,
        "errors": [],
        "outputs": [],
    }

    for path in image_paths:
        try:
            t0 = time.time()

            img = Image.open(path).convert("RGB")
            corrected = _apply_delta_to_image(img, delta)

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(output_dir, base + ".jpg")
            else:
                d = os.path.dirname(path)
                toned = os.path.join(d, "TONED")
                os.makedirs(toned, exist_ok=True)
                base = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(toned, base + ".jpg")

            corrected.save(out_path, "JPEG", quality=quality)

            elapsed_ms = int((time.time() - t0) * 1000)
            results["corrected"] += 1
            results["outputs"].append({
                "source": path,
                "output": out_path,
                "time_ms": elapsed_ms,
            })

        except Exception as e:
            results["failed"] += 1
            results["errors"].append({"file": path, "error": str(e)})

    return results


if __name__ == "__main__":
    import json

    control_path = "test_images/control.jpg"
    target_profile = extract_profile(control_path)

    test_dir = "test_images/batch_input"
    test_files = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.isfile(os.path.join(test_dir, f))
    ])

    print("=== Delta Engine Test ===")
    print("Control: %s" % control_path)
    print("Target profile: %s" % json.dumps(target_profile, indent=2))
    print()

    src_profile = extract_profile(test_files[0])
    delta = compute_delta(src_profile, target_profile)
    print("Sample delta (image 0 → control):")
    print(json.dumps(delta, indent=2))
    print()

    print("Applying delta to first 3 images...")
    result = apply_delta_to_cluster(
        test_files[:3],
        delta,
        output_dir="test_images/delta_output",
    )
    print("Corrected: %d/%d" % (result["corrected"], result["total"]))
    if result["errors"]:
        print("Errors:", result["errors"])
    for o in result["outputs"]:
        print("  %s → %s (%dms)" % (
            os.path.basename(o["source"]),
            os.path.basename(o["output"]),
            o["time_ms"],
        ))
