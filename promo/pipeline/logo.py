"""ETZ logo geometry (from the official etz.com SVG) and an emissive glass render."""
import math
import re

import cv2
import numpy as np

from fx import H, W, add_glow

# Official lockup (etz.com footer logo, viewBox 0 0 129 33): mark pieces + "ETZ" wordmark.
MARK_D = [
    "M13.4745 22.2031L7.46094 25.8111L9.67378 27.0705L15.6873 23.4455L13.4745 22.2031Z",
    "M28.9848 16.3477L29.0017 23.6317L16.7382 31.0008L10.4375 27.495L28.9848 16.3477Z",
    "M22.5304 5.47183L4 16.5851V9.42019L16.3649 2L22.5304 5.47183Z",
    "M28.6448 8.92677L28.6617 12.4837L24.9286 14.7302L24.9962 14.3387L25.1144 13.5729L25.4185 11.5136"
    "L25.5198 10.9009L24.8104 10.6457L21.9219 9.64156L25.8239 7.29297L28.6448 8.92677Z",
    "M4.04941 19.8536L10.0798 16.2286L10.046 13.6758L4.01562 17.2838L4.04941 19.8536Z",
    "M4.0663 20.3807L4.01562 23.9036L6.9886 25.5034L21.6676 16.6877L22.2589 12.8584L18.6778 11.582L4.0663 20.3807Z",
]
WORD_D = (
    "M54.8535 4.07617H64.6094V10.7207H62.7812C59.6055 10.7207 57.4375 10.2227 56.2773 9.22656C55.3281 8.40625 "
    "54.8535 7.08203 54.8535 5.25391V4.07617ZM42.3555 4.07617H51.8125V13.709H58.8613V18.2969H51.8125V27.9648H42.3555"
    "V4.07617ZM54.8359 26.2949C54.8359 23.8691 56.2422 22.2227 59.0547 21.3555C60.1328 21.0273 61.5215 20.8633 "
    "63.2207 20.8633H64.6094V28H54.8359V26.2949ZM90.0977 4.07617H97.9727V14.0078H96.0039C94.7148 14.0078 93.7656 "
    "13.873 93.1562 13.6035C92.5469 13.3223 92.0664 13.0117 91.7148 12.6719C91.375 12.3203 91.082 11.8633 90.8359 "
    "11.3008C90.3438 10.1523 90.0977 8.55859 90.0977 6.51953V4.07617ZM78.7246 4.07617H88.1816V28H78.7246V4.07617Z"
    "M68.8809 4.07617H76.7559V6.51953C76.7559 9.53125 76.2109 11.582 75.1211 12.6719C74.2305 13.5625 72.8066 "
    "14.0078 70.8496 14.0078H68.8809V4.07617ZM101.699 23.377C101.699 22.3105 102.145 21.1738 103.035 19.9668"
    "L103.756 19.0176L116.447 4.07617H125.061V9.13867C125.061 10.416 124.352 11.9336 122.934 13.6914L109.996 28H101.699"
    "V23.377ZM101.699 7.16992C101.699 5.6582 101.887 4.62695 102.262 4.07617H113.02L103.984 13.6914H103.457C102.285 "
    "13.0117 101.699 10.8379 101.699 7.16992ZM125.061 24.8887C125.061 26.4238 124.885 27.4609 124.533 28H113.934"
    "L123.32 17.8398C124.293 18.3906 124.855 19.7031 125.008 21.7773C125.055 22.4219 125.078 23.1016 125.078 23.8164"
    "L125.061 24.8887Z"
)

_TOK = re.compile(r"[MLHVCZmlhvcz]|-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_path(d, steps=14):
    toks = _TOK.findall(d)
    i, cmd = 0, None
    cur = start = (0.0, 0.0)
    polys, pts = [], []

    def num():
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    while i < len(toks):
        tk = toks[i]
        if tk.isalpha():
            cmd = tk
            i += 1
            if cmd in "Zz":
                if pts:
                    polys.append(pts)
                pts = []
                cur = start
                continue
        if cmd == "M":
            if pts:
                polys.append(pts)
            cur = start = (num(), num())
            pts = [cur]
            cmd = "L"
        elif cmd == "L":
            cur = (num(), num())
            pts.append(cur)
        elif cmd == "H":
            cur = (num(), cur[1])
            pts.append(cur)
        elif cmd == "V":
            cur = (cur[0], num())
            pts.append(cur)
        elif cmd == "C":
            x1, y1, x2, y2, x, y = (num() for _ in range(6))
            x0, y0 = cur
            for k in range(1, steps + 1):
                s = k / steps
                a, b, c, e = (1 - s) ** 3, 3 * (1 - s) ** 2 * s, 3 * (1 - s) * s * s, s ** 3
                pts.append((a * x0 + b * x1 + c * x2 + e * x, a * y0 + b * y1 + c * y2 + e * y))
            cur = (x, y)
        else:
            raise ValueError(f"unsupported path command {cmd}")
    if pts:
        polys.append(pts)
    return [np.asarray(p, np.float64) for p in polys]


MARK = [parse_path(d)[0] for d in MARK_D]
WORD = parse_path(WORD_D)
MARK_CENTER = np.mean(np.concatenate(MARK), 0)
LOCKUP_BOX = (4.0, 2.0, 125.1, 31.0)


def raster(polys, scale, ox, oy, shape=(H, W)):
    """Anti-aliased coverage of polygons (logo units -> px: p*scale + offset)."""
    m = np.zeros(shape, np.uint8)
    pts = [np.round((p * scale + (ox, oy)) * 16).astype(np.int32) for p in polys]
    cv2.fillPoly(m, pts, 255, cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255.0


def transform_piece(poly, center, dx, dy, rot_deg, s):
    a = math.radians(rot_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (poly - center) @ R.T * s + center + (dx, dy)


def shade(mask, t, base=(0.10, 0.85, 0.52), light_dir=(-0.6, -0.8), sweep=None, intensity=1.0):
    """Emissive emerald-glass look for a coverage mask (linear RGB, HDR)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask > 0.01)
    out = np.zeros((h, w, 3), np.float32)
    if len(xs) == 0:
        return out
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = 24
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    m = mask[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    # diagonal gradient (bright top-left -> deep emerald bottom-right)
    g = ((xx - x0) * light_dir[0] + (yy - y0) * light_dir[1])
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    deep = np.array([0.006, 0.12, 0.07], np.float32)
    mint = np.array([0.42, 1.35, 0.92], np.float32)
    body = deep + (mint - deep) * (g ** 1.8)[..., None]
    # bright rim + inner fresnel glow from the coverage gradient
    er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2)
    rim = np.clip(m - er, 0, 1)
    rim = cv2.GaussianBlur(rim, (0, 0), 0.8)
    inner = cv2.GaussianBlur(rim, (0, 0), 5.0) * m
    col = (body * m[..., None] + rim[..., None] * np.array([1.8, 2.8, 2.3], np.float32)
           + inner[..., None] * np.array([0.5, 1.6, 1.1], np.float32))
    if sweep is not None:
        pos, ang, width, amp = sweep
        d = (xx * math.cos(ang) + yy * math.sin(ang)) - pos
        band = np.exp(-0.5 * (d / width) ** 2)
        col += (band * m)[..., None] * np.array([2.2, 2.6, 2.4], np.float32) * amp
    out[y0:y1, x0:x1] = col * intensity
    return out


def composite_emissive(img, mask, light, occlusion=0.9):
    img *= (1 - mask * occlusion)[..., None]
    img += light


def shockwave(img, cx, cy, radius, width, amp, color=(0.5, 1.0, 0.8)):
    R = int(radius + width * 4)
    x0, x1 = max(0, int(cx - R)), min(W, int(cx + R) + 1)
    y0, y1 = max(0, int(cy - R)), min(H, int(cy + R) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + ((yy - cy) * 1.0) ** 2)
    ring = np.exp(-0.5 * ((r - radius) / width) ** 2)
    img[y0:y1, x0:x1] += (ring * amp)[..., None] * np.asarray(color, np.float32)
