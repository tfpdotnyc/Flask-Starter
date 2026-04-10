"""
ATTONE — Skin Tone Guard
Clamps luminance corrections inside the face bounding box to ±clamp_pct
of the original skin L* value. Prevents the correction engine from
brightening dark-skinned subjects or darkening fair-skinned subjects
beyond a safe perceptual threshold.

Outside the bounding box: corrected values pass through unchanged.
Inside the bounding box: L* delta is clamped per-pixel.
"""

import cv2
import numpy as np
from PIL import Image


def apply_with_skin_guard(
    original_img: Image.Image,
    corrected_img: Image.Image,
    bbox: dict,
    clamp_pct: float = 0.10,
) -> Image.Image:
    orig_rgb = np.array(original_img)
    corr_rgb = np.array(corrected_img)

    orig_lab = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    corr_lab = cv2.cvtColor(corr_rgb, cv2.COLOR_RGB2LAB).astype(np.float64)

    orig_l = orig_lab[:, :, 0]
    corr_l = corr_lab[:, :, 0]

    x = bbox["x"]
    y = bbox["y"]
    w = bbox["w"]
    h = bbox["h"]

    img_h, img_w = orig_l.shape
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    face_orig_l = orig_l[y1:y2, x1:x2]
    face_corr_l = corr_l[y1:y2, x1:x2]

    delta_l = face_corr_l - face_orig_l

    max_delta = face_orig_l * clamp_pct
    min_delta = -max_delta

    clamped_delta = np.clip(delta_l, min_delta, max_delta)

    corr_lab[y1:y2, x1:x2, 0] = np.clip(face_orig_l + clamped_delta, 0, 255)

    result_rgb = cv2.cvtColor(corr_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    return Image.fromarray(result_rgb)


if __name__ == "__main__":
    import os
    import json
    from face_guard import detect_face
    from color_profile import extract_profile
    from correction import apply_correction

    test_dir = "test_images/batch_input"
    control_path = "test_images/control.jpg"
    control_profile = extract_profile(control_path)

    paths = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.isfile(os.path.join(test_dir, f))
    ])[:3]

    print("=== Skin Tone Guard Test ===")
    print()

    for path in paths:
        name = os.path.basename(path)
        face = detect_face(path)
        if not face["detected"]:
            print("[SKIP] %s — no face" % name)
            continue

        bb = face["primary_face"]["bounding_box"]
        src_profile = extract_profile(path)
        original = Image.open(path).convert("RGB")
        corrected = apply_correction(original, src_profile, control_profile)

        orig_arr = np.array(original)
        corr_arr = np.array(corrected)
        orig_lab = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2LAB).astype(np.float64)
        corr_lab = cv2.cvtColor(corr_arr, cv2.COLOR_RGB2LAB).astype(np.float64)

        x1, y1 = bb["x"], bb["y"]
        x2, y2 = bb["x"] + bb["w"], bb["y"] + bb["h"]
        raw_delta = float(np.mean(np.abs(corr_lab[y1:y2, x1:x2, 0] - orig_lab[y1:y2, x1:x2, 0])))

        guarded = apply_with_skin_guard(original, corrected, bb, clamp_pct=0.10)

        guard_arr = np.array(guarded)
        guard_lab = cv2.cvtColor(guard_arr, cv2.COLOR_RGB2LAB).astype(np.float64)
        guarded_delta = float(np.mean(np.abs(guard_lab[y1:y2, x1:x2, 0] - orig_lab[y1:y2, x1:x2, 0])))

        orig_face_l = float(np.mean(orig_lab[y1:y2, x1:x2, 0]))
        max_allowed = orig_face_l * 0.10

        print("[OK] %s" % name)
        print("     Face L* mean: %.1f  |  Max allowed delta: ±%.1f" % (orig_face_l, max_allowed))
        print("     Raw correction delta: %.2f  →  Guarded delta: %.2f" % (raw_delta, guarded_delta))
        print("     Clamped: %s" % ("YES" if guarded_delta < raw_delta - 0.01 else "NO (within bounds)"))
        print()
