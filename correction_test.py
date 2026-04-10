"""
ATTONE — End-to-End Correction Test
Profiles the control and submission images, computes the delta,
applies the correction, and saves the result for visual inspection.
"""

import json
from PIL import Image
from color_profile import extract_profile
from correction import apply_correction

CONTROL = "test_images/control.jpg"
SUBMISSION = "test_images/submission.jpg"
OUTPUT = "test_images/submission_corrected.jpg"

control_profile = extract_profile(CONTROL)
submission_profile = extract_profile(SUBMISSION)

print("=== Control Profile ===")
print(json.dumps(control_profile, indent=2))
print()
print("=== Submission Profile (before) ===")
print(json.dumps(submission_profile, indent=2))
print()

delta = {
    k: round(control_profile[k] - submission_profile[k], 2)
    for k in ["luminance_mean", "saturation_mean"]
}
print("=== Delta (control - submission) ===")
print(json.dumps(delta, indent=2))
print()

submission_img = Image.open(SUBMISSION)
corrected_img = apply_correction(submission_img, submission_profile, control_profile)
corrected_img.save(OUTPUT, "JPEG", quality=95)
print("Corrected image saved: %s" % OUTPUT)

corrected_profile = extract_profile(OUTPUT)
print()
print("=== Submission Profile (after correction) ===")
print(json.dumps(corrected_profile, indent=2))

print()
print("=== Verification ===")
for k in ["luminance_mean", "saturation_mean"]:
    before_gap = abs(control_profile[k] - submission_profile[k])
    after_gap = abs(control_profile[k] - corrected_profile[k])
    direction = "CLOSER" if after_gap < before_gap else "FURTHER"
    print("%s: gap %.2f -> %.2f  [%s]" % (k, before_gap, after_gap, direction))
