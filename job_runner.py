"""
ATTONE — Batch Job Runner with Real-Time Progress Tracking
Processes images asynchronously in background, streams status updates.
Pipeline: cluster → compute deltas → process each image → export → update DB.
"""

import os
import time
import threading
from datetime import datetime, timezone
from PIL import Image

from database import SessionLocal, SessionImage, Session as DBSession
from color_profile import extract_profile
from clustering_engine import cluster_images
from delta_engine import compute_delta
from correction_pipeline import process_image
from export_engine import export_image
from face_guard import detect_face
from background_normalizer import extract_bg_profile

_progress = {}
_lock = threading.Lock()


def _update_progress(session_id: int, **kwargs):
    with _lock:
        if session_id not in _progress:
            _progress[session_id] = {
                "total": 0,
                "processed": 0,
                "exported": 0,
                "flagged": 0,
                "failed": 0,
                "current_image": None,
                "phase": "initializing",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "errors": [],
            }
        _progress[session_id].update(kwargs)


def get_progress(session_id: int) -> dict | None:
    with _lock:
        return _progress.get(session_id, None)


def run_session(
    session_id: int,
    image_paths: list[str],
    control_profile: dict,
    control_bg_profile: dict = None,
    output_dir: str = None,
    quality: int = 95,
):
    total = len(image_paths)
    _update_progress(session_id, total=total, phase="clustering")

    db = SessionLocal()
    try:
        session = db.query(DBSession).filter(DBSession.id == session_id).first()
        if session:
            session.status = "processing"
            session.total_images = total
            db.commit()

        _update_progress(session_id, phase="clustering", current_image="analyzing batch")
        clusters = cluster_images(image_paths, max_clusters=5)

        cluster_deltas = {}
        cluster_count = 0
        for label, members in clusters.items():
            if label.startswith("_"):
                continue
            cluster_count += 1

            _update_progress(session_id, phase="computing deltas", current_image="cluster %s" % label)

            cluster_profiles = []
            for p in members:
                try:
                    cluster_profiles.append(extract_profile(p))
                except Exception:
                    pass

            if cluster_profiles:
                import numpy as np
                avg_profile = {}
                for key in cluster_profiles[0]:
                    avg_profile[key] = round(float(np.mean([cp[key] for cp in cluster_profiles])), 2)
                cluster_deltas[label] = compute_delta(avg_profile, control_profile)

        path_to_cluster = {}
        for label, members in clusters.items():
            if label.startswith("_"):
                continue
            for p in members:
                path_to_cluster[p] = label

        _update_progress(session_id, phase="processing images")
        processed = 0
        exported = 0
        flagged = 0
        failed = 0
        errors = []

        for i, path in enumerate(image_paths):
            name = os.path.basename(path)
            _update_progress(
                session_id,
                current_image=name,
                processed=processed,
                exported=exported,
                flagged=flagged,
                failed=failed,
            )

            cluster_label = path_to_cluster.get(path)
            delta = cluster_deltas.get(cluster_label) if cluster_label else None

            result = process_image(
                path,
                control_profile,
                cluster_delta=delta,
                control_bg_profile=control_bg_profile,
            )

            img_record = SessionImage(
                session_id=session_id,
                filename=name,
                original_path=path,
                status=result["status"].lower(),
            )

            if result["status"] == "OK":
                processed += 1
                img_record.width = result["image_size"]["width"]
                img_record.height = result["image_size"]["height"]
                img_record.profile_data = result.get("source_profile")
                img_record.skin_tone_clamped = result.get("skin_guard_applied", False)
                img_record.processing_time_ms = result.get("processing_time_ms")

                exp = export_image(
                    result["corrected_img"],
                    path,
                    quality=quality,
                    output_dir=output_dir,
                )

                if exp["success"]:
                    exported += 1
                    img_record.toned_path = exp["output_path"]
                    img_record.format = "JPEG"
                else:
                    failed += 1
                    img_record.status = "export_failed"
                    img_record.error_message = exp["error_message"]
                    errors.append({"file": name, "error": exp["error_message"]})

            elif result["status"] == "FLAGGED":
                flagged += 1
                img_record.error_message = result.get("message")

            else:
                failed += 1
                img_record.error_message = result.get("message")
                errors.append({"file": name, "error": result.get("message")})

            img_record.processed_at = datetime.now(timezone.utc)
            db.add(img_record)
            db.commit()

        if session:
            session.status = "completed"
            session.processed_images = exported
            session.failed_images = failed
            session.completed_at = datetime.now(timezone.utc)
            db.commit()

        _update_progress(
            session_id,
            phase="completed",
            processed=processed,
            exported=exported,
            flagged=flagged,
            failed=failed,
            current_image=None,
            errors=errors,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        _update_progress(
            session_id,
            phase="error",
            current_image=None,
            errors=[{"file": "session", "error": str(e)}],
        )
        if session:
            session.status = "error"
            session.completed_at = datetime.now(timezone.utc)
            db.commit()

    finally:
        db.close()


if __name__ == "__main__":
    import json
    from database import init_db

    init_db()

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

    db = SessionLocal()
    from database import Session as SessionModel, ControlSet
    cs = db.query(ControlSet).first()
    if not cs:
        from control_set_manager import ControlSetManager
        analysis = ControlSetManager.analyze([control_path])
        cs = ControlSetManager.save(db, "Test Control", analysis)

    sess = SessionModel(
        name="Batch Test Run",
        control_set_id=cs.id,
        input_dir=test_dir,
    )
    db.add(sess)
    db.commit()
    db.refresh(sess)
    session_id = sess.id
    db.close()

    print("=== Job Runner Test — Session %d ===" % session_id)
    print("Images: %d" % len(paths))
    print()

    t = threading.Thread(
        target=run_session,
        args=(session_id, paths, control_profile),
        kwargs={"control_bg_profile": control_bg},
    )
    t.start()

    while t.is_alive():
        prog = get_progress(session_id)
        if prog:
            print("[%s] %d/%d processed  %d exported  %d flagged  %d failed  current: %s" % (
                prog["phase"],
                prog["processed"],
                prog["total"],
                prog["exported"],
                prog["flagged"],
                prog["failed"],
                prog.get("current_image", "—"),
            ))
        time.sleep(1)

    t.join()
    final = get_progress(session_id)
    print()
    print("=== Final Status ===")
    print(json.dumps({k: v for k, v in final.items() if k != "errors"}, indent=2))
    if final.get("errors"):
        print("Errors:", final["errors"])
