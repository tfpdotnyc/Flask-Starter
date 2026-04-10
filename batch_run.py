"""
ATTONE — Batch Processing Pipeline
Processes all supported images in an input folder against a control profile,
applies corrections, and exports to a /TONED subfolder.
Logs errors per image without stopping the loop.
"""

import os
import sys
import time
import json
from PIL import Image
from color_profile import extract_profile
from correction import apply_correction

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".cr2", ".cr3", ".arw", ".nef", ".nrw"}


def batch_process(input_dir: str, control_path: str) -> dict:
    toned_dir = os.path.join(input_dir, "TONED")
    os.makedirs(toned_dir, exist_ok=True)

    print("=== ATTONE Batch Run ===")
    print("Control:    %s" % control_path)
    print("Input dir:  %s" % input_dir)
    print("Output dir: %s" % toned_dir)
    print()

    print("Extracting control profile...")
    control_profile = extract_profile(control_path)
    print("Control profile OK")
    print()

    files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    total = len(files)
    print("Found %d image(s) to process" % total)
    print("-" * 50)

    results = {
        "total": total,
        "exported": 0,
        "failed": 0,
        "errors": [],
    }

    for i, filename in enumerate(files, 1):
        src_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(toned_dir, base_name + ".jpg")

        try:
            t0 = time.time()

            print("[%d/%d] %s" % (i, total, filename), end=" ... ")

            sub_profile = extract_profile(src_path)

            img = Image.open(src_path).convert("RGB")

            corrected = apply_correction(img, sub_profile, control_profile)

            corrected.save(out_path, "JPEG", quality=95)

            elapsed = time.time() - t0
            print("OK (%.2fs)" % elapsed)
            results["exported"] += 1

        except Exception as e:
            print("FAILED: %s" % str(e))
            results["failed"] += 1
            results["errors"].append({"file": filename, "error": str(e)})

    print("-" * 50)
    print()
    print("=== Batch Complete ===")
    print("Total:    %d" % results["total"])
    print("Exported: %d" % results["exported"])
    print("Failed:   %d" % results["failed"])

    if results["errors"]:
        print()
        print("Errors:")
        for err in results["errors"]:
            print("  %s — %s" % (err["file"], err["error"]))

    toned_files = os.listdir(toned_dir)
    print()
    print("TONED folder contains %d file(s):" % len(toned_files))
    for f in sorted(toned_files):
        fpath = os.path.join(toned_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print("  %s  (%.0f KB)" % (f, size_kb))

    return results


if __name__ == "__main__":
    control = sys.argv[1] if len(sys.argv) > 1 else "test_images/control.jpg"
    input_dir = sys.argv[2] if len(sys.argv) > 2 else "test_images/batch_input"

    batch_process(input_dir, control)
