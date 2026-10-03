"""Typography: tracked text rendered with PIL, composited in display space."""
import functools
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fx import ASSETS, BAR, FULL_H, FULL_W, H, W

FONT_FILES = {
    "en": "Montserrat[wght].ttf",
    "sans": "NotoSansSC[wght].ttf",
    "serif": "NotoSerifSC[wght].ttf",
}


@functools.lru_cache(maxsize=64)
def font(key, size, weight):
    f = ImageFont.truetype(os.path.join(ASSETS, "fonts", FONT_FILES[key]), size)
    f.set_variation_by_axes([weight])
    return f


@functools.lru_cache(maxsize=512)
def text_alpha(text, key="en", size=40, weight=300, tracking=0.0):
    """Return float32 alpha (h, w) of `text` with letter-spacing in em."""
    f = font(key, size, weight)
    asc, desc = f.getmetrics()
    pad = int(size * 0.6)
    adv = [f.getlength(ch) for ch in text]
    track = tracking * size
    width = int(sum(adv) + track * max(0, len(text) - 1)) + 2 * pad
    height = asc + desc + 2 * pad
    im = Image.new("L", (max(1, width), height), 0)
    d = ImageDraw.Draw(im)
    x = pad
    for ch, a in zip(text, adv):
        d.text((x, pad), ch, font=f, fill=255)
        x += a + track
    arr = np.asarray(im, np.float32) / 255.0
    ys, xs = np.nonzero(arr > 0.003)
    if len(xs) == 0:
        return arr[:1, :1] * 0
    x0, x1 = max(0, xs.min() - 4), min(arr.shape[1], xs.max() + 5)
    # keep a stable vertical box (baseline consistent between strings)
    return np.ascontiguousarray(arr[:, x0:x1])


def place(canvas, alpha, cx, cy, color=(1, 1, 1), opacity=1.0, glow=0.0, glow_color=None,
          blur=0.0, scale=1.0, anchor="center"):
    """Composite alpha mask onto display-space canvas (float RGB, any size)."""
    if opacity <= 0.002:
        return
    a = alpha
    if scale != 1.0:
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    gsig = max(2.0, alpha.shape[0] * scale * 0.08)
    pad = (int(blur * 3) + 2 if blur > 0.05 else 0) + (int(gsig * 3) + 2 if glow > 0 else 0)
    if pad:
        a = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    if blur > 0.05:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    h, w = a.shape
    if anchor == "center":
        x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    elif anchor == "left":
        x0, y0 = int(round(cx)), int(round(cy - h / 2))
    else:  # right
        x0, y0 = int(round(cx - w)), int(round(cy - h / 2))
    Hc, Wc = canvas.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(Wc, x0 + w), min(Hc, y0 + h)
    if x1c <= x0c or y1c <= y0c:
        return
    sub = a[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    region = canvas[y0c:y1c, x0c:x1c]
    col = np.asarray(color, np.float32)
    if glow > 0:
        g = cv2.GaussianBlur(sub, (0, 0), gsig)
        gc = np.asarray(glow_color if glow_color is not None else color, np.float32)
        region[:] = 1 - (1 - region) * (1 - np.clip(g * glow * opacity, 0, 1)[..., None] * gc)
    k = (sub * opacity)[..., None]
    region[:] = region * (1 - k) + col * k


def title_card(canvas, t, en, cn, start, end, cy=None, big=False, cx=None):
    """Trailer-style super: thin tracked caps that breathe open while fading."""
    if t < start - 0.01 or t > end + 0.01:
        return
    dur = end - start
    u = (t - start) / dur
    fade_in = min(0.45, dur * 0.3)
    fade_out = min(0.4, dur * 0.25)
    op = min(1.0, (t - start) / fade_in) if fade_in > 0 else 1.0
    op = min(op, (end - t) / fade_out) if fade_out > 0 else op
    op = max(0.0, op) ** 1.2
    if cy is None:
        cy = H / 2
    if cx is None:
        cx = W / 2
    # soft dark scrim behind the words keeps them legible over bright plates
    sw, sh = (620, 150) if big else (520, 100)
    x0, x1 = int(max(0, cx - sw * 1.6)), int(min(canvas.shape[1], cx + sw * 1.6))
    y0, y1 = int(max(0, cy - sh * 1.6)), int(min(canvas.shape[0], cy + sh * 1.6))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    scrim = np.exp(-0.5 * (((xx - cx) / (sw * 0.55)) ** 2 + ((yy - cy) / (sh * 0.55)) ** 2))
    canvas[y0:y1, x0:x1] *= (1 - (0.55 if big else 0.35) * op * scrim)[..., None]
    if big:
        tr = 0.62 + 0.10 * u
        a_en = text_alpha(en, "en", 58, 250, round(tr, 3))
        a_cn = text_alpha(cn, "serif", 30, 400, 0.9)
        blur = max(0.0, 1.0 - (t - start) / 0.25) * 4.0
        place(canvas, a_en, cx, cy - 16, opacity=op, glow=0.55, glow_color=(0.55, 1.0, 0.8), blur=blur)
        place(canvas, a_cn, cx, cy + 46, color=(0.82, 0.95, 0.9), opacity=op * 0.9)
    else:
        tr = 0.36 + 0.06 * u
        a_en = text_alpha(en, "en", 30, 300, round(tr, 3))
        a_cn = text_alpha(cn, "serif", 24, 400, 0.6)
        place(canvas, a_en, cx, cy - 12, opacity=op, glow=0.35, glow_color=(0.5, 1.0, 0.75))
        place(canvas, a_cn, cx, cy + 30, color=(0.80, 0.93, 0.88), opacity=op * 0.85)


def subtitle(frame_full, t, cn, en, start, end):
    """Bilingual subtitle in the lower letterbox bar of the 1920x1080 frame."""
    if t < start or t > end:
        return
    op = min(1.0, (t - start) / 0.18, (end - t) / 0.25)
    if op <= 0:
        return
    y_mid = BAR + H + (FULL_H - BAR - H) / 2
    a_cn = text_alpha(cn, "sans", 30, 350, 0.08)
    a_en = text_alpha(en, "en", 19, 300, 0.10)
    place(frame_full, a_cn, FULL_W / 2, y_mid - 17, color=(0.90, 0.90, 0.90), opacity=op)
    place(frame_full, a_en, FULL_W / 2, y_mid + 25, color=(0.62, 0.66, 0.66), opacity=op)
