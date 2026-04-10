"""
ATTONE — Clustering Engine
Groups submission images by visual similarity using perceptual hashing
and color histogram distance. Replicates how human editors sort portraits
by shooting location/lighting before batch-correcting groups together.
"""

import numpy as np
import imagehash
import cv2
from PIL import Image
from sklearn.cluster import KMeans


def _compute_features(image_path: str) -> np.ndarray:
    pil_img = Image.open(image_path)
    phash = imagehash.phash(pil_img, hash_size=8)
    hash_bits = phash.hash.flatten().astype(np.float64)

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [64], [0, 256]).flatten()

    hist_h = hist_h / (hist_h.sum() + 1e-7)
    hist_s = hist_s / (hist_s.sum() + 1e-7)
    hist_v = hist_v / (hist_v.sum() + 1e-7)

    return np.concatenate([hash_bits, hist_h, hist_s, hist_v])


def _find_optimal_k(features: np.ndarray, max_k: int) -> int:
    n_samples = features.shape[0]
    max_k = min(max_k, n_samples)

    if max_k <= 1:
        return 1

    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(features)
        inertias.append(km.inertia_)

    if len(inertias) < 3:
        return 1

    if inertias[0] < 1e-6:
        return 1

    ratios = []
    for i in range(1, len(inertias)):
        ratios.append(inertias[i] / inertias[0])

    best_k = 1
    for i in range(1, len(ratios)):
        drop = ratios[i - 1] - ratios[i]
        if drop < 0.05:
            best_k = i + 1
            break
    else:
        best_k = max_k

    if ratios[0] > 0.85:
        return 1

    return best_k


def cluster_images(image_paths: list[str], max_clusters: int = 5) -> dict:
    if not image_paths:
        return {}

    if len(image_paths) == 1:
        return {"cluster_0": image_paths}

    features_list = []
    valid_paths = []
    errors = []

    for path in image_paths:
        try:
            feat = _compute_features(path)
            features_list.append(feat)
            valid_paths.append(path)
        except Exception as e:
            errors.append({"file": path, "error": str(e)})

    if not features_list:
        return {"errors": errors}

    features = np.array(features_list)

    optimal_k = _find_optimal_k(features, max_clusters)

    if optimal_k <= 1:
        result = {"cluster_0": valid_paths}
    else:
        km = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
        labels = km.fit_predict(features)

        result = {}
        for label_id in range(optimal_k):
            key = "cluster_%d" % label_id
            result[key] = [
                valid_paths[i] for i, l in enumerate(labels) if l == label_id
            ]
            if not result[key]:
                del result[key]

    if errors:
        result["_errors"] = errors

    return result


if __name__ == "__main__":
    import os
    import json

    batch_dir = "test_images/batch_input"
    paths = sorted([
        os.path.join(batch_dir, f)
        for f in os.listdir(batch_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not f.startswith(".")
        and os.path.isfile(os.path.join(batch_dir, f))
    ])

    print("Clustering %d images..." % len(paths))
    clusters = cluster_images(paths, max_clusters=5)

    for label, members in clusters.items():
        if label.startswith("_"):
            continue
        print("\n%s (%d images):" % (label, len(members)))
        for m in members:
            print("  %s" % os.path.basename(m))
