"""
ATTONE — Background Normalizer
Detects the non-subject region of a portrait and converges its color/tone
to match the control set's background. Ensures consistent backdrop appearance
across the entire yearbook so backgrounds never draw the viewer's eye.

Face bounding box + 15% padding is excluded from correction.
Everything outside that region gets LAB-corrected to the control background.
"""

import cv2
import numpy as np
from PIL import Image


def extract_bg_profile(image: Image.Image, face_bbox: dict, pad_pct: float = 0.15) -> dict:
    rgb = np.array(image)
    h, w = rgb.shape[:2]

    mask = _make_bg_mask(w, h, face_bbox, pad_pct)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)

    bg_pixels_lab = lab[mask]
    bg_pixels_hsv = hsv[mask]

    if len(bg_pixels_lab) == 0:
        return {}

    return {
        "bg_l_mean": round(float(np.mean(bg_pixels_lab[:, 0])), 2),
        "bg_a_mean": round(float(np.mean(bg_pixels_lab[:, 1])), 2),
        "bg_b_mean": round(float(np.mean(bg_pixels_lab[:, 2])), 2),
        "bg_l_std": round(float(np.std(bg_pixels_lab[:, 0])), 2),
        "bg_sat_mean": round(float(np.mean(bg_pixels_hsv[:, 1])), 2),
        "bg_pixel_count": int(np.sum(mask)),
    }


def normalize_background(
    image: Image.Image,
    control_bg_profile: dict,
    face_bbox: dict,
    pad_pct: float = 0.15,
    strength: float = 0.85,
) -> Image.Image:
    rgb = np.array(image)
    h, w = rgb.shape[:2]

    mask = _make_bg_mask(w, h, face_bbox, pad_pct)

    if np.sum(mask) == 0:
        return image

    src_bg_profile = extract_bg_profile(image, face_bbox, pad_pct)
    if not src_bg_profile:
        return image

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)

    delta_l = (control_bg_profile.get("bg_l_mean", src_bg_profile["bg_l_mean"]) - src_bg_profile["bg_l_mean"]) * strength
    delta_a = (control_bg_profile.get("bg_a_mean", src_bg_profile["bg_a_mean"]) - src_bg_profile["bg_a_mean"]) * strength
    delta_b = (control_bg_profile.get("bg_b_mean", src_bg_profile["bg_b_mean"]) - src_bg_profile["bg_b_mean"]) * strength

    lab[mask, 0] = np.clip(lab[mask, 0] + delta_l, 0, 255)
    lab[mask, 1] = np.clip(lab[mask, 1] + delta_a, 0, 255)
    lab[mask, 2] = np.clip(lab[mask, 2] + delta_b, 0, 255)

    edge_mask = _make_feather_mask(w, h, face_bbox, pad_pct)
    corrected_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    blend = edge_mask[:, :, np.newaxis]
    result = (corrected_rgb * blend + rgb * (1.0 - blend)).astype(np.uint8)

    return Image.fromarray(result)


def _make_bg_mask(w: int, h: int, bbox: dict, pad_pct: float) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)

    bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    px = int(bw * pad_pct)
    py = int(bh * pad_pct)

    x1 = max(0, bx - px)
    y1 = max(0, by - py)
    x2 = min(w, bx + bw + px)
    y2 = min(h, by + bh + py)

    mask[y1:y2, x1:x2] = False
    return mask


def _make_feather_mask(w: int, h: int, bbox: dict, pad_pct: float, feather_px: int = 20) -> np.ndarray:
    bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    px = int(bw * pad_pct)
    py = int(bh * pad_pct)

    x1 = max(0, bx - px)
    y1 = max(0, by - py)
    x2 = min(w, bx + bw + px)
    y2 = min(h, by + bh + py)

    mask = np.ones((h, w), dtype=np.float64)
    mask[y1:y2, x1:x2] = 0.0

    if feather_px > 0:
        kernel_size = feather_px * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), feather_px / 2)

    return mask


if __name__ == "__main__":
    import os
    import json
    from face_guard import detect_face
    from color_profile import extract_profile

    control_path = "test_images/control.jpg"
    control_img = Image.open(control_path).convert("RGB")
    control_face = detect_face(control_path)
    control_bb = control_face["primary_face"]["bounding_box"]
    control_bg = extract_bg_profile(control_img, control_bb)

    print("=== Background Normalizer Test ===")
    print("Control BG profile:")
    print(json.dumps(control_bg, indent=2))
    print()

    test_dir = "test_images/batch_input"
    paths = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.isfile(os.path.join(test_dir, f))
    ])[:5]

    os.makedirs("test_images/bg_normalized", exist_ok=True)

    for path in paths:
        name = os.path.basename(path)
        face = detect_face(path)
        if not face["detected"]:
            print("[SKIP] %s — no face" % name)
            continue

        bb = face["primary_face"]["bounding_box"]
        img = Image.open(path).convert("RGB")

        src_bg = extract_bg_profile(img, bb)
        delta_l = control_bg["bg_l_mean"] - src_bg["bg_l_mean"]
        delta_a = control_bg["bg_a_mean"] - src_bg["bg_a_mean"]
        delta_b = control_bg["bg_b_mean"] - src_bg["bg_b_mean"]

        result = normalize_background(img, control_bg, bb)
        out_path = os.path.join("test_images/bg_normalized", name.rsplit(".", 1)[0] + ".jpg")
        result.save(out_path, "JPEG", quality=95)

        result_bg = extract_bg_profile(result, bb)
        remaining_l = abs(control_bg["bg_l_mean"] - result_bg["bg_l_mean"])

        print("[OK] %s" % name)
        print("     BG delta L=%.1f  a=%.1f  b=%.1f  →  Remaining L gap: %.1f" % (
            delta_l, delta_a, delta_b, remaining_l
        ))
