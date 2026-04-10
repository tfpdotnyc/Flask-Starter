"""
ATTONE — RAW Decode Proof-of-Concept
Opens a CR2 or ARW file, converts to 8-bit sRGB, saves as JPG.
"""

import rawpy
import numpy as np
from PIL import Image

INPUT_FILE = "test_images/sample.CR2"
OUTPUT_FILE = "test_images/sample_decoded.jpg"

def decode_raw_to_jpg(input_path: str, output_path: str) -> None:
    with rawpy.imread(input_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=8,
            no_auto_bright=False,
        )

    img = Image.fromarray(rgb)
    img.save(output_path, "JPEG", quality=95)
    print(f"Decoded: {input_path}")
    print(f"Output:  {output_path}")
    print(f"Size:    {img.size[0]}x{img.size[1]}")
    print(f"Format:  8-bit sRGB JPG @ 95% quality")

if __name__ == "__main__":
    decode_raw_to_jpg(INPUT_FILE, OUTPUT_FILE)
