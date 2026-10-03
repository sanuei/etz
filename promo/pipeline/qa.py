"""QA: thumbnail contact sheets of the rendered picture every `step` seconds,
plus frame-exact strips around every cut. usage: qa.py picture.mkv out_dir"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

pic, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
W, H = 1920, 1080
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)


def frames(times, w=384):
    h = int(w * H / W)
    ims = []
    for t in times:
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.4f}", "-i", pic, "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        im = Image.fromarray(np.frombuffer(raw, np.uint8).reshape(H, W, 3)).resize((w, h), Image.BILINEAR)
        ImageDraw.Draw(im).text((4, 2), f"{t:.2f}", fill=(255, 255, 0), font=font)
        ims.append(im)
    return ims


def sheet(ims, cols, path):
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    s = Image.new("RGB", (cols * w, rows * h), (40, 40, 40))
    for i, im in enumerate(ims):
        s.paste(im, ((i % cols) * w, (i // cols) * h))
    s.save(path, quality=85)


step = 0.5
ts = np.arange(0, 60, step) + 0.5 / 24
for part in range(4):
    sel = ts[part * 30:(part + 1) * 30]
    sheet(frames(sel, 320), 6, os.path.join(out, f"overview_{part}.jpg"))
cuts = [5.25, 10.5, 15.0, 20.25, 24.0, 28.5, 31.5, 34.5, 37.5, 39.75, 42.0, 44.25, 45.75, 47.25, 49.5, 53.25]
strip = []
for c in cuts:
    strip += frames([c - 2 / 24 - 2e-3, c - 1 / 24 - 2e-3, c - 2e-3 + 1e-6, c + 1 / 24 - 2e-3], 320)
sheet(strip[:32], 4, os.path.join(out, "cuts_a.jpg"))
sheet(strip[32:], 4, os.path.join(out, "cuts_b.jpg"))
print("ok")
