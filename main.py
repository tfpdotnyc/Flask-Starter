"""
ATTONE — FastAPI Backend
Automated Portrait Toning Engine
"""

import os
import io
import rawpy
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="ATTONE", version="0.1.0")

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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
