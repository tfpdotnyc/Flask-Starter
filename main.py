"""
ATTONE — FastAPI Backend
Automated Portrait Toning Engine
"""

import os
import io
import rawpy
import numpy as np
from PIL import Image
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession
from fastapi import BackgroundTasks
from database import init_db, get_db, Session as SessionModel, ControlSet
from control_set_manager import ControlSetManager
from job_runner import run_session, get_progress
from face_guard import detect_face
from background_normalizer import extract_bg_profile


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="ATTONE", version="0.1.0", lifespan=lifespan)

RAW_EXTENSIONS = {".cr2", ".cr3", ".arw", ".nef", ".nrw", ".dng", ".raf", ".orf", ".rw2"}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/decode")
async def decode(file: UploadFile = File(...)):
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    data = await file.read()

    if ext in RAW_EXTENSIONS:
        try:
            raw = rawpy.imread(io.BytesIO(data))
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=8,
                no_auto_bright=True,
            )
            h, w = rgb.shape[:2]
            fmt = "RAW (%s)" % ext.upper().lstrip(".")
        except Exception as e:
            raise HTTPException(status_code=422, detail="RAW decode failed: %s" % str(e))

    elif ext in IMG_EXTENSIONS:
        try:
            img = Image.open(io.BytesIO(data))
            w, h = img.size
            fmt = img.format or ext.upper().lstrip(".")
        except Exception as e:
            raise HTTPException(status_code=422, detail="Image decode failed: %s" % str(e))

    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported format: %s" % ext,
        )

    return {
        "filename": filename,
        "width": w,
        "height": h,
        "format": fmt,
    }


class ControlSetCreate(BaseModel):
    name: str
    description: str | None = None
    image_paths: List[str]


@app.post("/control-sets")
def create_control_set(payload: ControlSetCreate, db: DBSession = Depends(get_db)):
    for p in payload.image_paths:
        if not os.path.isfile(p):
            raise HTTPException(status_code=400, detail="File not found: %s" % p)

    try:
        result = ControlSetManager.analyze(payload.image_paths)
    except Exception as e:
        raise HTTPException(status_code=422, detail="Analysis failed: %s" % str(e))

    cs = ControlSetManager.save(db, payload.name, result, description=payload.description)
    return ControlSetManager.load(db, cs_id=cs.id)


@app.get("/control-sets")
def list_control_sets(db: DBSession = Depends(get_db)):
    return ControlSetManager.list_all(db)


@app.get("/control-sets/{cs_id}")
def get_control_set(cs_id: int, db: DBSession = Depends(get_db)):
    data = ControlSetManager.load(db, cs_id=cs_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Control set not found")
    return data


class SessionCreate(BaseModel):
    name: str
    control_set_id: int
    image_paths: List[str]
    quality: int = 95


@app.post("/sessions")
def create_session(
    payload: SessionCreate,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
):
    cs = db.query(ControlSet).filter(ControlSet.id == payload.control_set_id).first()
    if not cs:
        raise HTTPException(status_code=404, detail="Control set not found")

    for p in payload.image_paths:
        if not os.path.isfile(p):
            raise HTTPException(status_code=400, detail="File not found: %s" % p)

    control_profile = cs.profile_data

    sess = SessionModel(
        name=payload.name,
        control_set_id=payload.control_set_id,
        total_images=len(payload.image_paths),
    )
    db.add(sess)
    db.commit()
    db.refresh(sess)

    control_bg = None
    if cs.source_dir:
        try:
            ctrl_img = Image.open(os.path.join(cs.source_dir, os.listdir(cs.source_dir)[0])).convert("RGB")
            ctrl_face = detect_face(os.path.join(cs.source_dir, os.listdir(cs.source_dir)[0]))
            if ctrl_face.get("detected"):
                control_bg = extract_bg_profile(ctrl_img, ctrl_face["primary_face"]["bounding_box"])
        except Exception:
            pass

    background_tasks.add_task(
        run_session,
        sess.id,
        payload.image_paths,
        control_profile,
        control_bg_profile=control_bg,
        quality=payload.quality,
    )

    return {
        "session_id": sess.id,
        "name": sess.name,
        "status": "queued",
        "total_images": len(payload.image_paths),
    }


@app.get("/sessions/{session_id}/progress")
def session_progress(session_id: int):
    prog = get_progress(session_id)
    if prog is None:
        return {
            "session_id": session_id,
            "phase": "not_started",
            "total": 0,
            "processed": 0,
            "exported": 0,
            "flagged": 0,
            "failed": 0,
            "current_image": None,
        }
    return {"session_id": session_id, **prog}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
