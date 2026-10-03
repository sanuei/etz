"""Image/FX toolkit for the ETZ film: camera moves on stills, light FX, grading.

All compositing happens in linear light (float32, HDR allowed); the post chain
tone-maps to display space at the end.
"""
import math
import os

import cv2
import numpy as np

W, H = 1920, 804            # active 2.39:1 picture
FULL_W, FULL_H = 1920, 1080  # delivery frame (letterboxed)
BAR = (FULL_H - H) // 2
FPS = 24

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")

_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)


# --------------------------------------------------------------------------- math
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def smoother(t):
    t = clamp(t)
    return t * t * t * (t * (6 * t - 15) + 10)


def ease_in(t, p=2.0):
    return clamp(t) ** p


def ease_out(t, p=2.0):
    return 1 - (1 - clamp(t)) ** p


def ease_in_out(t, p=2.0):
    t = clamp(t)
    return 0.5 * (2 * t) ** p if t < 0.5 else 1 - 0.5 * (2 - 2 * t) ** p


def window(t, a, b, fade_in=0.2, fade_out=0.2):
    """1 inside [a, b] with smooth ramps."""
    if t < a or t > b:
        return 0.0
    v = 1.0
    if fade_in > 0:
        v = min(v, smooth((t - a) / fade_in))
    if fade_out > 0:
        v = min(v, smooth((b - t) / fade_out))
    return v


def srgb_to_lin(img8):
    return np.power(img8.astype(np.float32) / 255.0, 2.2)


def lin_to_srgb(x):
    return np.power(np.clip(x, 0.0, 1.0), 1 / 2.2)


# --------------------------------------------------------------------------- plates
class Plate:
    """A still image treated as a camera plate (linear RGB float32)."""

    def __init__(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1]
        self.img = srgb_to_lin(np.ascontiguousarray(img))
        self.h, self.w = self.img.shape[:2]
        self.base_scale = max(W / self.w, H / self.h)

    def matrix(self, cx, cy, zoom, rot=0.0):
        """Affine src->out putting src point (cx, cy) at the frame center."""
        s = zoom * self.base_scale
        a = math.radians(rot)
        ca, sa = math.cos(a) * s, math.sin(a) * s
        return np.array([[ca, -sa, W / 2 - (ca * cx - sa * cy)],
                         [sa, ca, H / 2 - (sa * cx + ca * cy)]], np.float64)

    def src_coords(self, M):
        """Source coordinates of every output pixel for affine M."""
        Mi = cv2.invertAffineTransform(M)
        sx = (Mi[0, 0] * _XX + Mi[0, 1] * _YY + Mi[0, 2]).astype(np.float32)
        sy = (Mi[1, 0] * _XX + Mi[1, 1] * _YY + Mi[1, 2]).astype(np.float32)
        return sx, sy

    def view(self, cx, cy, zoom, rot=0.0, img=None, warps=()):
        """Render the plate through the camera. `warps` are callables
        (sx, sy) -> (sx, sy) applied in source space (flow, swirl...)."""
        src = self.img if img is None else img
        M = self.matrix(cx, cy, zoom, rot)
        if not warps:
            out = cv2.warpAffine(src, M.astype(np.float32), (W, H), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REFLECT101)
        else:
            sx, sy = self.src_coords(M)
            for w in warps:
                sx, sy = w(sx, sy)
            out = cv2.remap(src, sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
        return np.maximum(out, 0.0), M


def to_out(M, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def mblur(fn, t, shutter=0.5 / FPS, n=1):
    """Average n sub-frame renders over the shutter interval (180 deg)."""
    if n <= 1:
        return fn(t)
    acc = None
    for k in range(n):
        f = fn(t + shutter * (k / (n - 1) - 0.5))
        acc = f if acc is None else acc + f
    return acc / n


# --------------------------------------------------------------------------- warps
class Flow:
    """Slow, smooth turbulence for gas and nebulae (source-space displacement)."""

    def __init__(self, w, h, cell=110, amp=4.0, speed=0.35, seed=1):
        rng = np.random.default_rng(seed)
        gw, gh = w // cell + 3, h // cell + 3
        self.size = (w, h)
        self.cell = cell
        self.ph = rng.uniform(0, 2 * np.pi, (4, gh, gw)).astype(np.float32)
        self.fr = rng.uniform(0.5, 1.0, (4, gh, gw)).astype(np.float32) * speed
        self.amp = amp

    def __call__(self, t):
        dx = np.sin(self.ph[0] + self.fr[0] * t) + 0.6 * np.sin(self.ph[1] + self.fr[1] * t * 1.7)
        dy = np.sin(self.ph[2] + self.fr[2] * t) + 0.6 * np.sin(self.ph[3] + self.fr[3] * t * 1.7)
        dx = cv2.resize(dx * self.amp, self.size, interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(dy * self.amp, self.size, interpolation=cv2.INTER_CUBIC)
        return dx, dy

    def warp(self, t):
        dx, dy = self(t)
        w, h = self.size

        def f(sx, sy):
            ix = np.clip(sx, 0, w - 1).astype(np.int32)
            iy = np.clip(sy, 0, h - 1).astype(np.int32)
            return sx + dx[iy, ix], sy + dy[iy, ix]
        return f


def swirl_warp(center, t, w0=0.35, r0=120.0, r_in=60.0, r_out=520.0, aspect=1.0, tilt=0.0, inflow=0.0,
               protect=0.0):
    """Differential (Keplerian-ish) rotation of a disk around `center`.
    w0 rad/s at radius r0; effect fades between r_in..r_out."""
    cx, cy = center
    ct, st = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))

    def f(sx, sy):
        x, y = sx - cx, sy - cy
        u = x * ct + y * st
        v = (-x * st + y * ct) / aspect
        r = np.sqrt(u * u + v * v) + 1e-3
        fade = np.clip((r - r_in * 0.6) / (r_in * 0.6), 0, 1) * np.clip((r_out - r) / (r_out * 0.35), 0, 1)
        if protect > 0:
            rs = np.sqrt(x * x + y * y)
            fade = fade * np.clip((rs - protect) / (protect * 0.4), 0, 1)
        ang = w0 * t * np.power(r0 / r, 1.5) * fade
        ang = np.clip(ang, -3.0, 3.0)
        ca, sa = np.cos(ang), np.sin(ang)
        k = 1.0 + inflow * t * fade
        u2 = (u * ca - v * sa) * k
        v2 = (u * sa + v * ca) * k * aspect
        x2 = u2 * ct - v2 * st
        y2 = u2 * st + v2 * ct
        return (x2 + cx).astype(np.float32), (y2 + cy).astype(np.float32)
    return f


def zoom_blur(img, cx, cy, amount=0.06, n=8):
    """Radial (zoom) blur toward (cx, cy) for warp speed."""
    acc = img.copy()
    for k in range(1, n):
        s = 1.0 + amount * k / (n - 1)
        M = np.array([[s, 0, cx - s * cx], [0, s, cy - s * cy]], np.float32)
        acc += cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return acc / n


# --------------------------------------------------------------------------- light
def add_glow(img, x, y, radius, color, amp, power=1.6):
    """Soft radial light (1/(1+r^2)^p) around a point, windowed for speed."""
    reach = math.sqrt(max(1.0, (abs(amp) / 2e-5) ** (1.0 / power)))
    R = int(min(radius * reach, 2.5 * W))
    x0, x1 = max(0, int(x - R)), min(W, int(x + R) + 1)
    y0, y1 = max(0, int(y - R)), min(H, int(y + R) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy = (np.arange(y0, y1, dtype=np.float32) - y)[:, None]
    xx = (np.arange(x0, x1, dtype=np.float32) - x)[None, :]
    g = 1.0 / np.power(1.0 + (xx * xx + yy * yy) / (radius * radius), power)
    img[y0:y1, x0:x1] += (amp * g)[..., None] * np.asarray(color, np.float32)


def splat(img, x, y, sigma, color, amp):
    r = int(math.ceil(sigma * 3)) + 1
    x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    gx = np.exp(-0.5 * ((np.arange(x0, x1, dtype=np.float32) - x) / sigma) ** 2)
    gy = np.exp(-0.5 * ((np.arange(y0, y1, dtype=np.float32) - y) / sigma) ** 2)
    img[y0:y1, x0:x1] += (amp * np.outer(gy, gx))[..., None] * np.asarray(color, np.float32)


def anamorphic(img, x, y, amp, color=(0.45, 0.75, 1.0), length=520.0, width=2.2, ghosts=True, rays=0.0):
    """Horizontal anamorphic streak + soft core + lens ghosts."""
    color = np.asarray(color, np.float32)
    y0, y1 = max(0, int(y - 48)), min(H, int(y + 49))
    if y0 < y1:
        yy = (np.arange(y0, y1, dtype=np.float32) - y)[:, None]
        xx = (np.arange(W, dtype=np.float32) - x)[None, :]
        ax = np.abs(xx)
        prof = np.exp(-ax / length) * np.exp(-0.5 * (yy / width) ** 2)
        prof += 0.35 * np.exp(-ax / (length * 0.35)) * np.exp(-0.5 * (yy / (width * 5)) ** 2)
        img[y0:y1] += (amp * 0.6 * prof)[..., None] * color
    add_glow(img, x, y, 18, (1.0, 0.97, 0.92), amp * 0.9, power=1.4)
    add_glow(img, x, y, 60, color * 0.6 + 0.4, amp * 0.18, power=1.2)
    if rays > 0:
        for k in range(6):
            a = k * math.pi / 3 + 0.3
            for s in np.linspace(8, 260, 26):
                splat(img, x + math.cos(a) * s, y + math.sin(a) * s, 1.4, (1, 1, 1),
                      rays * amp * 0.08 * math.exp(-s / 120))
    if ghosts:
        cx, cy = W / 2, H / 2
        for k, (s, r, c) in enumerate([(-0.45, 26, (0.3, 0.9, 0.7)), (-0.8, 60, (0.9, 0.6, 0.3)),
                                       (0.35, 14, (0.5, 0.7, 1.0)), (-1.25, 90, (0.3, 0.7, 0.9))]):
            gx, gy = cx + (x - cx) * s, cy + (y - cy) * s
            add_glow(img, gx, gy, r, c, amp * 0.012, power=3.0)


def godrays(img, x, y, strength=0.5, decay=0.93, n=24, threshold=0.25):
    """Volumetric light scattering from bright areas toward (x, y)."""
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    src = np.maximum(small - threshold, 0)
    acc = np.zeros_like(src)
    cx, cy = x / 4, y / 4
    wgt = 1.0
    cur = src
    for k in range(n):
        s = 1.0 - 0.035 * (k + 1)
        M = np.array([[s, 0, cx - s * cx], [0, s, cy - s * cy]], np.float32)
        cur = cv2.warpAffine(src, M, (W // 4, H // 4), flags=cv2.INTER_LINEAR)
        acc += cur * wgt
        wgt *= decay
    acc = cv2.resize(acc / n, (W, H), interpolation=cv2.INTER_LINEAR)
    return img + strength * acc


def bloom(img, threshold=0.7, strength=0.35, radii=(3, 10, 28, 70)):
    hi = np.maximum(img - threshold, 0)
    small = cv2.resize(hi, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(small)
    for r in radii:
        acc += cv2.GaussianBlur(small, (0, 0), r / 2)
    return img + strength * cv2.resize(acc / len(radii), (W, H), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- particles
class Dust:
    """Drifting motes in 3D for parallax and depth (rendered as soft dots)."""

    def __init__(self, n=220, seed=3, color=(0.75, 1.0, 0.9), zmin=0.6, zmax=6.0, spread=1.4):
        rng = np.random.default_rng(seed)
        self.p = np.stack([rng.uniform(-spread, spread, n) * 2.39, rng.uniform(-spread, spread, n),
                           rng.uniform(zmin, zmax, n)], 1)
        self.v = rng.normal(0, 0.02, (n, 3))
        self.b = rng.uniform(0.3, 1.0, n)
        self.tw = rng.uniform(0, 2 * np.pi, n)
        self.color = np.asarray(color, np.float32)

    def draw(self, img, t, cam=(0.0, 0.0, 0.0), amp=0.25, focus=2.5, f=1.0):
        p = self.p + self.v * t - np.asarray(cam)
        for (x, y, z), b, tw in zip(p, self.b, self.tw):
            if z <= 0.15:
                continue
            u = W / 2 + (x / z) * f * W / 2.39 / 1.0
            v = H / 2 + (y / z) * f * H
            if u < -50 or u > W + 50 or v < -50 or v > H + 50:
                continue
            coc = 0.8 + 6.0 * abs(1 / z - 1 / focus)
            a = amp * b * (0.75 + 0.25 * math.sin(tw + t * 2.0)) / (coc * coc) * 2.0
            splat(img, u, v, coc, self.color, a)


class StarField3D:
    """Stars in a tunnel for fly-through / hyperspace (streaks drawn AA in 8-bit)."""

    def __init__(self, n=2600, seed=7, radius=1.0, depth=40.0):
        rng = np.random.default_rng(seed)
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = radius * np.sqrt(rng.uniform(0.02, 1.0, n)) * 3.0
        self.x = np.cos(ang) * rad
        self.y = np.sin(ang) * rad * 0.8
        self.z = rng.uniform(0.2, depth, n)
        self.depth = depth
        self.b = rng.uniform(0.2, 1.0, n) ** 2
        tint = rng.uniform(0, 1, n)
        self.col = np.stack([0.75 + 0.25 * tint, 0.9 + 0.1 * tint, 1.0 - 0.2 * tint], 1)

    def draw(self, img, cz, speed, cx=W / 2, cy=H / 2, f=900.0, amp=1.0, streak=1.0, tint=(1, 1, 1)):
        """cz: camera z travelled; speed: z units/sec (sets streak length)."""
        z = (self.z - cz) % self.depth + 0.05
        z_prev = z + max(speed, 0.0) * (0.5 / FPS) * streak
        u = cx + self.x / z * f
        v = cy + self.y / z * f
        u0 = cx + self.x / z_prev * f
        v0 = cy + self.y / z_prev * f
        bright = np.clip(self.b * (1.5 / (z + 0.4)) * amp, 0, 6.0)
        layer = np.zeros((H, W, 3), np.uint8)
        vis = (u > -200) & (u < W + 200) & (v > -200) & (v < H + 200) & (bright > 0.01)
        gain = 3.0
        for i in np.nonzero(vis)[0]:
            c = np.clip(self.col[i] * np.asarray(tint) * bright[i] / gain, 0, 1) * 255
            p1 = (int(u[i] * 16), int(v[i] * 16))
            p0 = (int(u0[i] * 16), int(v0[i] * 16))
            th = 1 if bright[i] < 1.2 else 2
            cv2.line(layer, p0, p1, (int(c[0]), int(c[1]), int(c[2])), th, cv2.LINE_AA, shift=4)
        lin = srgb_to_lin(layer) * gain
        img += lin


# --------------------------------------------------------------------------- paths on plates
def color_mask(img_lin, kind="green", thr=0.08, line_px=9):
    """Mask of thin bright green light-lines in a linear plate (white top-hat)."""
    r, g, b = img_lin[..., 0], img_lin[..., 1], img_lin[..., 2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_px, line_px))
    th = g - cv2.morphologyEx(g, cv2.MORPH_OPEN, k)
    m = (th > thr) & (g > r * 1.5) & (g > b * 0.85)
    return m


def geodesic(mask, seeds):
    """Geodesic distance along mask from seed points (pixels)."""
    from skimage.graph import MCP_Geometric
    cost = np.where(mask, 1.0, np.inf)
    mcp = MCP_Geometric(cost)
    starts = []
    for (x, y) in seeds:
        ys, xs = np.nonzero(mask)
        k = np.argmin((xs - x) ** 2 + (ys - y) ** 2)
        starts.append((ys[k], xs[k]))
    dist, _ = mcp.find_costs(starts)
    dist[~np.isfinite(dist)] = -1
    return dist.astype(np.float32)


class PathPulses:
    """Light pulses travelling along light-lines already present in a plate."""

    def __init__(self, plate, seeds, kind="green", thr=0.08, dilate=1, roi=None, line_px=9, min_area=60):
        m = color_mask(plate.img, kind, thr, line_px)
        if roi is not None:
            x0, y0, x1, y1 = roi
            keep = np.zeros_like(m)
            keep[y0:y1, x0:x1] = True
            m &= keep
        m8 = m.astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m8, 8)
        big = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                big |= lab == i
        if dilate:
            big = cv2.dilate(big.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=dilate) > 0
        self.mask = big
        self.dist = geodesic(big, seeds)
        self.valid = self.dist >= 0
        ys, xs = np.nonzero(self.valid)
        self.box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None

    def field(self, t, speed=600.0, spacing=420.0, width=40.0, start=0.0):
        """Pulse intensity in source space (full plate size)."""
        out = np.zeros(self.dist.shape, np.float32)
        if self.box is None:
            return out
        x0, y0, x1, y1 = self.box
        d = self.dist[y0:y1, x0:x1]
        v = self.valid[y0:y1, x0:x1]
        front = speed * max(0.0, t - start)
        ph = (front - d) / spacing
        frac = ph - np.floor(ph)
        pulse = np.exp(-0.5 * ((frac - 0.5) * spacing / width) ** 2)
        pulse *= (d <= front) & v
        out[y0:y1, x0:x1] = pulse
        return cv2.GaussianBlur(out, (0, 0), 2.0)


# --------------------------------------------------------------------------- post
_vig = None
_ca_maps = None


def _vignette():
    global _vig
    if _vig is None:
        x = (_XX - W / 2) / (W / 2)
        y = (_YY - H / 2) / (H / 2) * (H / W) * 2.39 / 2.39
        r2 = x * x * 0.6 + y * y * 0.9
        _vig = (1.0 - 0.28 * np.power(r2, 1.1)).astype(np.float32)[..., None]
    return _vig


def _ca():
    global _ca_maps
    if _ca_maps is None:
        maps = []
        for s in (1.0012, 0.9988):
            mx = (W / 2 + (_XX - W / 2) / s).astype(np.float32)
            my = (H / 2 + (_YY - H / 2) / s).astype(np.float32)
            maps.append((mx, my))
        _ca_maps = maps
    return _ca_maps


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def grade(lin, exposure=1.0, bloom_amt=0.3, halation=0.12, sat=1.04, lift=(0.004, 0.010, 0.010),
          gain=(1.02, 1.0, 0.97), vignette=True, ca=True):
    """Linear HDR -> display-referred float RGB in [0, 1]."""
    x = lin * exposure
    if bloom_amt > 0:
        x = bloom(x, threshold=0.65, strength=bloom_amt)
    if halation > 0:
        hi = np.maximum(x - 0.8, 0)
        small = cv2.resize(hi, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        hal = cv2.GaussianBlur(small, (0, 0), 6)
        hal = cv2.resize(hal, (W, H), interpolation=cv2.INTER_LINEAR)
        x = x + halation * hal * np.array([1.0, 0.45, 0.25], np.float32)
    y = aces(x * 0.8)
    y = np.power(y, 1 / 2.2)
    if ca:
        (rx, ry), (bx, by) = _ca()
        r = cv2.remap(y[..., 0], rx, ry, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        b = cv2.remap(y[..., 2], bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        y = np.stack([r, y[..., 1], b], -1)
    lum = (y * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1, keepdims=True)
    y = lum + (y - lum) * sat
    shadow = np.power(np.clip(1 - lum, 0, 1), 3)
    y = y * np.asarray(gain, np.float32) + shadow * np.asarray(lift, np.float32)
    if vignette:
        y = y * _vignette()
    return np.clip(y, 0, 1)


def grain(y, rng, amt=0.022, size=0.7):
    n = rng.standard_normal((H, W)).astype(np.float32)
    if size > 0:
        n = cv2.GaussianBlur(n, (0, 0), size)
        n *= 1.0 / max(0.25, 0.65 - 0.25 * size)
    lum = y.mean(-1, keepdims=True)
    wgt = 0.35 + 1.6 * lum * (1 - lum) * 2
    return np.clip(y + amt * n[..., None] * wgt, 0, 1)


def letterbox(y_disp):
    out = np.zeros((FULL_H, FULL_W, 3), np.float32)
    out[BAR:BAR + H] = y_disp
    return out
