"""Contact sheet of preview frames: sheet.py dir out.jpg [cols] [width]"""
import glob
import os
import sys

from PIL import Image, ImageDraw, ImageFont

d, out = sys.argv[1], sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 2
w = int(sys.argv[4]) if len(sys.argv) > 4 else 960
fs = sorted(glob.glob(os.path.join(d, "f_*.jpg")))
h = int(w * 1080 / 1920)
rows = (len(fs) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w + (cols - 1) * 6, rows * h + (rows - 1) * 6), (60, 60, 60))
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
for i, f in enumerate(fs):
    im = Image.open(f).resize((w, h), Image.LANCZOS)
    ImageDraw.Draw(im).text((8, 6), os.path.basename(f)[2:-4] + "s", fill=(255, 255, 0), font=font)
    sheet.paste(im, ((i % cols) * (w + 6), (i // cols) * (h + 6)))
sheet.save(out, quality=90)
print(out, sheet.size)
