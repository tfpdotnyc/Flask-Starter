"""
ATTONE — Master Correction Pipeline
Assembles all modules into a single function: one image in, one corrected image out.
decode → detect_face → apply_correction → skin_guard → normalize_background → return

Never raises exceptions. All errors are caught and returned as {status: "ERROR"}.
"""

import time
import cv2
import numpy as np
from PIL import Image
from color_profile import extract_profile
from correction import apply_correction
from face_guard import detect_face
from skin_tone_guard import apply_with_skin_guard
from background_normalizer import extract_bg_profile, normalize_background


def process_image(
    image_path: str,
    control_profile: dict,
    cluster_delta: dict = None,
    control_bg_profile: dict = None,
    clamp_pct: float = 0.10,
    bg_strength: float = 0.85,
) -> dict:
    t0 = time.time()

    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
    except Exception as e:
        return {
            "status": "ERROR",
            "message": "Decode failed: %s" % str(e),
            "image_path": image_path,
        }

    try:
        face_result = detect_face(image_path)
    except Exception as e:
        return {
            "status": "ERROR",
            "message": "Face detection failed: %s" % str(e),
            "image_path": image_path,
        }

    face_detected = face_result.get("detected", False)
    if not face_detected:
        return {
            "status": "FLAGGED",
            "message": "No face detected — skipping correction",
            "reason": face_result.get("reason", "no_face_found"),
            "image_path": image_path,
            "face_detected": False,
            "corrected_img": None,
            "processing_time_ms": int((time.time() - t0) * 1000),
        }

    bbox = face_result["primary_face"]["bounding_box"]

    try:
        src_profile = extract_profile(image_path)
    except Exception as e:
        return {
            "status": "ERROR",
            "message": "Profile extraction failed: %s" % str(e),
            "image_path": image_path,
        }

    try:
        corrected = apply_correction(img, src_profile, control_profile)
    except Exception as e:
        return {
            "status": "ERROR",
            "message": "Correction failed: %s" % str(e),
            "image_path": image_path,
        }

    skin_guard_applied = False
    try:
        corrected = apply_with_skin_guard(img, corrected, bbox, clamp_pct=clamp_pct)
        skin_guard_applied = True
    except Exception as e:
        pass

    bg_normalized = False
    if control_bg_profile:
        try:
            corrected = normalize_background(corrected, control_bg_profile, bbox, strength=bg_strength)
            bg_normalized = True
        except Exception as e:
            pass

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "status": "OK",
        "image_path": image_path,
        "corrected_img": corrected,
        "face_detected": True,
        "face_count": face_result.get("face_count", 1),
        "face_confidence": face_result["primary_face"]["confidence"],
        "face_bbox": bbox,
        "skin_guard_applied": skin_guard_applied,
        "bg_normalized": bg_normalized,
        "source_profile": src_profile,
        "processing_time_ms": elapsed_ms,
        "image_size": {"width": w, "height": h},
    }


if __name__ == "__main__":
    import os
    import json

    control_path = "test_images/control.jpg"
    control_profile = extract_profile(control_path)

    control_img = Image.open(control_path).convert("RGB")
    control_face = detect_face(control_path)
    control_bb = control_face["primary_face"]["bounding_box"]
    control_bg = extract_bg_profile(control_img, control_bb)

    test_dir = "test_images/batch_input"
    paths = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.isfile(os.path.join(test_dir, f))
    ])

    out_dir = "test_images/pipeline_output"
    os.makedirs(out_dir, exist_ok=True)

    print("=== Full Correction Pipeline — %d images ===" % len(paths))
    print()

    ok = 0
    flagged = 0
    errors = 0

    for path in paths:
        name = os.path.basename(path)
        result = process_image(
            path,
            control_profile,
            control_bg_profile=control_bg,
        )

        status = result["status"]
        ms = result.get("processing_time_ms", 0)

        if status == "OK":
            ok += 1
            out_path = os.path.join(out_dir, name.rsplit(".", 1)[0] + ".jpg")
            result["corrected_img"].save(out_path, "JPEG", quality=95)
            print("[OK]      %s  (%dms)  face=%.2f  skin_guard=%s  bg_norm=%s" % (
                name, ms,
                result["face_confidence"],
                result["skin_guard_applied"],
                result["bg_normalized"],
            ))
        elif status == "FLAGGED":
            flagged += 1
            print("[FLAGGED] %s  (%dms)  reason=%s" % (name, ms, result.get("reason")))
        else:
            errors += 1
            print("[ERROR]   %s  — %s" % (name, result.get("message")))

    print()
    print("=== Pipeline Complete ===")
    print("OK: %d  |  Flagged: %d  |  Errors: %d  |  Total: %d" % (ok, flagged, errors, len(paths)))

    out_files = sorted(os.listdir(out_dir))
    print("Output folder: %d file(s)" % len(out_files))
    for f in out_files:
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print("  %s  (%.0f KB)" % (f, size_kb))
