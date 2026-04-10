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
from database import init_db, get_db
from control_set_manager import ControlSetManager


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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
