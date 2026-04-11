"""
ATTONE | correction_pipeline.py  (AMD-01 rewrite)
Master correction pipeline — orchestrates all processing stages.
AMD-01: Bbox masking removed. Now uses four-stage face pipeline
with soft, feathered, mesh-derived alpha masks.

HARD LOCK: No cropped face ROI may ever be directly written back
into the final image. All face-region edits projected through soft
mesh-derived alpha masks with feathered transitions.
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from vision.face_pipeline        import run_face_pipeline
from correction.blend             import feather_blend
from correction.skin_guard        import apply_skin_guard
from correction.background_normalizer import normalize_background
from storage.face_cache           import save_mask_cache, delete_cache


def decode_image_to_rgb(image_path: str) -> np.ndarray:
    ext = Path(image_path).suffix.lower()
    if ext in [".cr2", ".cr3", ".arw", ".nef", ".nrw"]:
        import rawpy
        with rawpy.imread(str(image_path)) as raw:
            return raw.postprocess(use_camera_wb=True, output_bps=8)
    else:
        return np.array(Image.open(image_path).convert("RGB"))


def apply_global_correction(
    img_rgb: np.ndarray,
    cluster_delta: dict
) -> np.ndarray:
    """
    Apply cluster-level color correction to the full image.
    Accepts both legacy delta keys (from compute_delta) and AMD-01 keys.
    No face-region logic here — global canvas only.
    """
    if not cluster_delta:
        return img_rgb

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    l_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]
    s_ch = hsv[:, :, 1]

    if "luminance_mean" in cluster_delta:
        l_ch += cluster_delta["luminance_mean"]
    elif "exposure" in cluster_delta:
        l_ch += cluster_delta["exposure"] * 10

    if "contrast_std" in cluster_delta and cluster_delta["contrast_std"] != 0:
        l_mean = np.mean(l_ch)
        factor = 1.0 + (cluster_delta["contrast_std"] / 100.0)
        l_ch[:] = l_mean + (l_ch - l_mean) * factor

    if "shadow_5pct" in cluster_delta:
        shadow_mask = (l_ch < 50).astype(np.float32)
        l_ch += cluster_delta["shadow_5pct"] * shadow_mask

    if "highlight_95pct" in cluster_delta:
        highlight_mask = (l_ch > 200).astype(np.float32)
        l_ch += cluster_delta["highlight_95pct"] * highlight_mask

    if "blacks_1pct" in cluster_delta:
        blacks_mask = (l_ch < 20).astype(np.float32)
        l_ch += cluster_delta["blacks_1pct"] * blacks_mask

    if "whites_99pct" in cluster_delta:
        whites_mask = (l_ch > 230).astype(np.float32)
        l_ch += cluster_delta["whites_99pct"] * whites_mask

    if "a_mean" in cluster_delta:
        a_ch += cluster_delta["a_mean"]

    if "b_mean" in cluster_delta:
        b_ch += cluster_delta["b_mean"]

    if "saturation_mean" in cluster_delta and cluster_delta["saturation_mean"] != 0:
        s_ch += cluster_delta["saturation_mean"]
    elif "saturation" in cluster_delta:
        scale = 1.0 + cluster_delta["saturation"] * 0.1
        lab[:, :, 1] *= scale
        lab[:, :, 2] *= scale

    if "vibrance" in cluster_delta and cluster_delta["vibrance"] != 0:
        low_sat_mask = 1.0 - (s_ch / 255.0)
        s_ch += cluster_delta["vibrance"] * low_sat_mask

    if "temperature_est_k" in cluster_delta and cluster_delta["temperature_est_k"] != 0:
        warmth = cluster_delta["temperature_est_k"] / 6500.0
        b_ch += warmth * 2.0
        a_ch -= warmth * 0.5
    elif "temperature" in cluster_delta:
        lab[:, :, 2] = np.clip(lab[:, :, 2] + cluster_delta["temperature"] * 5, 0, 255)

    if "tint" in cluster_delta:
        lab[:, :, 1] = np.clip(lab[:, :, 1] + cluster_delta["tint"] * 2, 0, 255)

    lab[:, :, 0] = np.clip(l_ch, 0, 255)
    lab[:, :, 1] = np.clip(a_ch, 0, 255)
    lab[:, :, 2] = np.clip(b_ch, 0, 255)
    hsv[:, :, 1] = np.clip(s_ch, 0, 255)

    bgr_from_lab = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    bgr_from_hsv = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    blended = cv2.addWeighted(bgr_from_lab, 0.6, bgr_from_hsv, 0.4, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def process_image(
    image_path: str,
    control_profile: dict,
    cluster_delta: dict,
    control_bg_profile: dict = None,
    session_id: str = None,
    image_id: str = None,
    debug_mode: bool = False
) -> dict:
    """
    AMD-01 corrected master processing function.

    Pipeline:
    1. Decode image to working RGB space
    2. Run AMD-01 four-stage face pipeline (detect -> mesh -> masks -> serialize)
    3. No face detected -> FLAGGED, export original as-is
    4. Apply global correction to full image canvas
    5. Apply face-region residual correction
    6. Feather-blend using face_influence_alpha (no hard boundaries)
    7. Apply skin luminance guard inside skin_core_alpha (+/-8% L* clamp)
    8. Normalize background inside background_alpha
    9. Cache masks if session context provided

    Returns:
        status          - CORRECTED | FLAGGED | ERROR
        corrected_img   - np.ndarray (RGB uint8)
        face_detected   - bool
        mesh_success    - bool
        confidence      - float
        bbox            - tuple or None
        face_zone_json  - str or None
        face_zone_version - str or None
        mask_cache_path - str or None
        error_message   - str or None
    """
    try:
        img_rgb = decode_image_to_rgb(image_path)

        face_result = run_face_pipeline(img_rgb)

        base_meta = {
            "face_detected":    face_result["face_detected"],
            "mesh_success":     face_result["mesh_success"],
            "confidence":       face_result["confidence"],
            "bbox":             face_result["bbox"],
            "face_zone_json":   face_result["face_zone_json"],
            "face_zone_version": "mediapipe_face_mesh_v1" if face_result["mesh_success"] else None,
            "mask_cache_path":  None,
        }

        if not face_result["face_detected"]:
            return {
                "status":         "FLAGGED",
                "corrected_img":  img_rgb,
                "error_message":  "No face detected — exported as-is",
                **base_meta,
            }

        global_corrected = apply_global_correction(img_rgb, cluster_delta)

        if face_result["mesh_success"] and face_result["masks"]:
            masks = face_result["masks"]

            face_delta = {k: v * 0.3 for k, v in cluster_delta.items()}
            face_corrected = apply_global_correction(img_rgb, face_delta)

            blended = feather_blend(
                global_corrected,
                face_corrected,
                masks["face_influence_alpha"]
            )

            guarded = apply_skin_guard(
                img_rgb, blended,
                masks["skin_core_alpha"],
                clamp_pct=0.08
            )

            bg_profile = None
            if control_bg_profile:
                bg_profile = control_bg_profile
            elif control_profile and "background" in control_profile:
                bg_profile = control_profile["background"]

            final = normalize_background(
                guarded, bg_profile, masks["background_alpha"]
            )

            cache_path = None
            if session_id and image_id:
                cache_path = save_mask_cache(
                    session_id, image_id,
                    masks["skin_core_alpha"],
                    masks["face_influence_alpha"],
                    masks["background_alpha"]
                )

            base_meta["mask_cache_path"] = cache_path

            return {
                "status":         "CORRECTED",
                "corrected_img":  final,
                "error_message":  None,
                **base_meta,
            }

        else:
            return {
                "status":         "FLAGGED",
                "corrected_img":  global_corrected,
                "error_message":  "Mesh failed — global correction applied, flagged for review",
                **base_meta,
            }

    except Exception as e:
        return {
            "status":         "ERROR",
            "corrected_img":  None,
            "face_detected":  False,
            "mesh_success":   False,
            "confidence":     0.0,
            "bbox":           None,
            "face_zone_json": None,
            "face_zone_version": None,
            "mask_cache_path": None,
            "error_message":  str(e),
        }
