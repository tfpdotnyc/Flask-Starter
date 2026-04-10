"""
ATTONE — Export Engine
Takes corrected image data from memory and writes properly quality-controlled
JPGs to a /TONED subfolder next to the source images. Embeds sRGB profile.
Never overwrites source files.
"""

import os
from PIL import Image, ImageCms

_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_SRGB_BYTES = ImageCms.ImageCmsProfile(_SRGB_PROFILE).tobytes()


def export_image(
    corrected_img: Image.Image,
    source_path: str,
    quality: int = 95,
    output_dir: str = None,
) -> dict:
    try:
        parent = os.path.dirname(os.path.abspath(source_path))
        toned_dir = output_dir or os.path.join(parent, "TONED")
        os.makedirs(toned_dir, exist_ok=True)

        stem = os.path.splitext(os.path.basename(source_path))[0]
        out_filename = stem + ".jpg"
        out_path = os.path.join(toned_dir, out_filename)

        if os.path.abspath(out_path) == os.path.abspath(source_path):
            return {
                "success": False,
                "output_path": None,
                "error_message": "Export would overwrite source file — blocked",
            }

        if corrected_img.mode != "RGB":
            corrected_img = corrected_img.convert("RGB")

        corrected_img.save(
            out_path,
            "JPEG",
            quality=quality,
            icc_profile=_SRGB_BYTES,
            subsampling=0,
        )

        file_size = os.path.getsize(out_path)

        return {
            "success": True,
            "output_path": out_path,
            "filename": out_filename,
            "file_size_bytes": file_size,
            "quality": quality,
            "color_space": "sRGB",
            "error_message": None,
        }

    except Exception as e:
        return {
            "success": False,
            "output_path": None,
            "error_message": str(e),
        }


if __name__ == "__main__":
    from color_profile import extract_profile
    from correction_pipeline import process_image
    from face_guard import detect_face
    from background_normalizer import extract_bg_profile

    control_path = "test_images/control.jpg"
    control_profile = extract_profile(control_path)
    control_face = detect_face(control_path)
    control_bb = control_face["primary_face"]["bounding_box"]
    control_img = Image.open(control_path).convert("RGB")
    control_bg = extract_bg_profile(control_img, control_bb)

    test_dir = "test_images/batch_input"
    paths = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.isfile(os.path.join(test_dir, f))
    ])

    print("=== Export Engine Test — %d images ===" % len(paths))
    print()

    exported = 0
    failed = 0

    for path in paths:
        name = os.path.basename(path)
        result = process_image(path, control_profile, control_bg_profile=control_bg)

        if result["status"] != "OK":
            print("[SKIP] %s — %s" % (name, result.get("message", result["status"])))
            continue

        exp = export_image(result["corrected_img"], path)
        if exp["success"]:
            exported += 1
            size_kb = exp["file_size_bytes"] / 1024
            print("[OK] %s → %s  (%.0f KB, %s)" % (
                name, exp["filename"], size_kb, exp["color_space"]
            ))
        else:
            failed += 1
            print("[FAIL] %s — %s" % (name, exp["error_message"]))

    print()
    print("=== Export Complete ===")
    print("Exported: %d  |  Failed: %d  |  Total: %d" % (exported, failed, len(paths)))

    toned_dir = os.path.join(test_dir, "TONED")
    if os.path.isdir(toned_dir):
        files = sorted(os.listdir(toned_dir))
        print("TONED folder: %d file(s)" % len(files))
