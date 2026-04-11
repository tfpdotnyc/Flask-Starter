"""
ATTONE — Face Guard (Identity Lock Foundation)
Detects faces using mediapipe FaceDetector (tasks API) to establish:
1. Skin region bounding boxes for the Skin Tone Guard clamp
2. Identity perimeter that no pixel manipulation may cross without explicit instruction

No face detected = image flagged, no correction applied.
Hijab/headscarf case: eyes = face. Detection still counts. Never crash the pipeline.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")
_BaseOptions = mp.tasks.BaseOptions
_FaceDetector = mp.tasks.vision.FaceDetector
_FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions


def detect_face(image_path: str) -> dict:
    try:
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return {
                "detected": False,
                "reason": "could_not_read_image",
                "face_count": 0,
            }

        h, w, _ = bgr.shape
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        options = _FaceDetectorOptions(
            base_options=_BaseOptions(model_asset_path=_MODEL_PATH),
            min_detection_confidence=0.3,
        )

        with _FaceDetector.create_from_options(options) as detector:
            result = detector.detect(mp_image)

        if not result.detections:
            return {
                "detected": False,
                "reason": "no_face_found",
                "face_count": 0,
                "image_size": {"width": w, "height": h},
            }

        faces = []
        for det in result.detections:
            bb = det.bounding_box
            confidence = det.categories[0].score if det.categories else 0.0

            bx = max(0, bb.origin_x)
            by = max(0, bb.origin_y)
            bw = min(bb.width, w - bx)
            bh = min(bb.height, h - by)

            if bw < 10 or bh < 10:
                continue

            faces.append({
                "bounding_box": {"x": bx, "y": by, "w": bw, "h": bh},
                "confidence": round(float(confidence), 4),
            })

        if not faces:
            return {
                "detected": False,
                "reason": "no_face_found",
                "face_count": 0,
                "image_size": {"width": w, "height": h},
            }

        faces.sort(key=lambda f: f["bounding_box"]["w"] * f["bounding_box"]["h"], reverse=True)
        primary = faces[0]

        return {
            "detected": True,
            "face_count": len(faces),
            "primary_face": primary,
            "all_faces": faces,
            "image_size": {"width": w, "height": h},
        }

    except Exception as e:
        return {
            "detected": False,
            "reason": "exception: %s" % str(e),
            "face_count": 0,
        }


def get_skin_region(image_path: str, face_result: dict = None) -> dict:
    if face_result is None:
        face_result = detect_face(image_path)

    if not face_result.get("detected", False):
        return {
            "has_skin_region": False,
            "reason": face_result.get("reason", "no_face_found"),
        }

    bb = face_result["primary_face"]["bounding_box"]
    img_w = face_result["image_size"]["width"]
    img_h = face_result["image_size"]["height"]

    pad_x = int(bb["w"] * 0.15)
    pad_y = int(bb["h"] * 0.15)
    x1 = max(0, bb["x"] - pad_x)
    y1 = max(0, bb["y"] - pad_y)
    x2 = min(img_w, bb["x"] + bb["w"] + pad_x)
    y2 = min(img_h, bb["y"] + bb["h"] + pad_y)

    return {
        "has_skin_region": True,
        "skin_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "face_box": bb,
        "confidence": face_result["primary_face"]["confidence"],
    }


if __name__ == "__main__":
    import json

    test_dir = "test_images/batch_input"
    paths = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.isfile(os.path.join(test_dir, f))
    ])

    print("=== Face Guard Test — %d images ===" % len(paths))
    print()

    detected = 0
    missed = 0
    for path in paths:
        result = detect_face(path)
        name = os.path.basename(path)
        if result["detected"]:
            detected += 1
            bb = result["primary_face"]["bounding_box"]
            conf = result["primary_face"]["confidence"]
            print("[OK]   %s — confidence=%.2f  box=(%d,%d,%d,%d)  faces=%d" % (
                name, conf, bb["x"], bb["y"], bb["w"], bb["h"], result["face_count"]
            ))
        else:
            missed += 1
            print("[MISS] %s — reason: %s" % (name, result.get("reason", "unknown")))

    print()
    print("Detected: %d/%d  |  Missed: %d" % (detected, len(paths), missed))

    print()
    print("=== Skin Region Test (first detected) ===")
    for path in paths:
        r = detect_face(path)
        if r["detected"]:
            skin = get_skin_region(path, r)
            print(json.dumps(skin, indent=2))
            break
