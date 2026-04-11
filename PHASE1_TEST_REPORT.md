# ATTONE Phase 1 Acceptance Test Report

**Date:** 2026-04-11
**Test Batch:** 12 images (10 WhatsApp portraits + 1 submission + 1 CR2 RAW)
**Control Set:** control_image_1775791412310.jpeg

---

## Phase 1 + AMD-01 Acceptance Checklist

| # | Criterion | Result |
|---|-----------|--------|
| 1 | All images exported as JPG to /TONED | PASS (12 files) |
| 2 | Source files unmodified | PASS (timestamps unchanged) |
| 3 | Filenames match originals exactly | PASS (12/12 match) |
| 4 | No rectangular bbox seam on any face | PASS (AMD-01 soft mesh masks only) |
| 5 | No face reconstruction visible | PASS |
| 6 | Dark skin not over-brightened or crushed to black | PASS (0% pure black pixels) |
| 6b | Fair skin not over-darkened or crushed to white | PASS (0% pure white pixels) |
| 7 | Backgrounds normalized | PASS (global + face-zone corrections applied) |
| 8 | Flagged images logged by type | PASS (1 no-face, 0 mesh-fail) |
| 9 | No pipeline crash on any image | PASS (0 errors) |
| 10 | face_zone_json stored in DB | PASS (oval_landmarks + bbox + version + confidence) |
| 11 | restore_face_mask() can rebuild mask without MediaPipe | PASS (face_zone_version=mediapipe_face_mesh_v1) |
| 12 | Visual inspection: no outlier portraits | PASS |

**Overall: PASS**

---

## Batch Results

- **Corrected:** 11/12 (91.7%)
- **Flagged no-face:** 1 (CR2 RAW file — not a portrait)
- **Flagged mesh-fail:** 0
- **Errors:** 0
- **Mesh confidence range:** 0.93 - 0.98

---

## Correction Effectiveness (Source vs Toned vs Control Target)

| Metric | Images Improved | Avg Gap Before | Avg Gap After | Reduction |
|--------|----------------|---------------|--------------|-----------|
| Luminance | 8/11 (73%) | 13.8 | 9.4 | -32% |
| Contrast | 5/11 (45%) | 4.7 | 4.5 | -3% |
| Saturation | 8/11 (73%) | 32.1 | 20.4 | -36% |
| Temperature (R/B) | 10/11 (91%) | 0.6 | 0.5 | -21% |
| Shadows (5th pct) | 8/11 (73%) | 5.8 | 3.6 | -38% |
| Highlights (95th pct) | 7/11 (64%) | 18.1 | 14.6 | -19% |

---

## Weakest Corrections (Largest Remaining Gap to Target)

1. **Saturation** — consistent undershoot; 5 images still >25 units from target
2. **Highlights** — 3 images still >22 units off
3. **Temperature** — PM5 has warm R/B already at 2.8, pushed further to 3.1 (target 1.5)

---

## Failure Mode Analysis

### Most Common Failure Mode: Saturation Under-Correction
The current 4-characteristic engine applies saturation correction globally, but the correction strength is conservative. Images starting far from the target (e.g., saturation 60 vs target 122) only get partially corrected. The engine moves them 30-40% of the way rather than the full distance.

### Second Mode: Temperature Overcorrection on Already-Warm Images
PM5 (warm outdoor lighting, R/B=2.8) was pushed warmer instead of cooled. The global correction adds the delta uniformly without detecting that the source is already beyond the target.

### Third Mode: Contrast Not Adjusted
Contrast correction is barely effective (+3% reduction). The engine adjusts luminance distribution but doesn't actively redistribute tonal range to match the control's contrast curve.

---

## Phase 2 Spec Input

### Priority Corrections to Add (ordered by impact)

1. **Saturation depth** — Current: single global sat shift. Need: HSL-aware saturation with per-channel control (orange/red skin tones, blue/cyan backgrounds separately)
2. **Tone curve / highlights-shadows** — Current: linear brightness shift. Need: proper curves adjustment (blacks, shadows, midtones, highlights, whites — 5-point curve)
3. **Color temperature + tint** — Current: rough R/B ratio. Need: proper white balance shift in Lab or a/b channels with separate tint control
4. **Contrast** — Current: minimal std adjustment. Need: S-curve contrast with midpoint control
5. **Vibrance vs Saturation** — Current: single sat metric. Need: separate vibrance (protect skin tones) and saturation (boost muted colors)
6. **HSL per-channel** — Hue, saturation, luminance adjustments per color range (reds, oranges, yellows, greens, aquas, blues, purples, magentas)
7. **Clarity / local contrast** — Unsharp mask on midtones for micro-contrast
8. **Blacks / Whites clipping** — Independent clip point control
9. **Dehaze** — Atmospheric correction
10. **Sharpening** — Output sharpening with radius/amount/masking
11. **Noise reduction** — Luminance + color noise reduction
12. **Split toning** — Shadows/highlights color cast

### Data Points for Phase 2 Calibration

- Control image luminance target: 76.2
- Control image saturation target: 122.4
- Control R/B temperature ratio: 1.508
- Control contrast (std): 63.9
- Control shadow floor (5th pct): 0.6
- Control highlight ceiling (95th pct): 165.7
