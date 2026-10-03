"""Shot-by-shot renderer for "ETZ: LIGHT BEYOND BOUNDARIES".

Every shot starts from a MiniMax image-01 plate and adds camera motion and
light effects. render_frame(T) returns the final 1920x1080 display frame.
"""
import functools
import json
import math
import os

import cv2
import numpy as np

import fx
import logo
import typo
from fx import H, W, FPS, Plate, add_glow, anamorphic, clamp, ease_in, ease_in_out, ease_out, lerp, smooth, splat

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "..", "assets", "images", "sel")
EDL = json.load(open(os.path.join(HERE, "..", "script", "edl.json")))
LANG = os.environ.get("ETZ_LANG", "cn")
VO_META = json.load(open(os.path.join(HERE, "..", "assets", "audio", "vo", LANG, "meta.json")))

EMERALD = np.array([0.16, 1.0, 0.55], np.float32)
MINT = np.array([0.55, 1.0, 0.82], np.float32)


@functools.lru_cache(maxsize=None)
def plate(name):
    return Plate(os.path.join(SEL, name + ".jpg"))


def cam(p, cx, cy, z, rot=0.0):
    """Clamp the camera so the (rotated) frame stays inside the plate."""
    s = z * p.base_scale
    a = math.radians(abs(rot))
    hw = (W / 2 * math.cos(a) + H / 2 * math.sin(a)) / s
    hh = (W / 2 * math.sin(a) + H / 2 * math.cos(a)) / s
    cx = min(max(cx, hw), p.w - hw) if p.w > 2 * hw else p.w / 2
    cy = min(max(cy, hh), p.h - hh) if p.h > 2 * hh else p.h / 2
    return cx, cy, z, rot


def rng_for(seed):
    return np.random.default_rng(seed)


# ============================================================================ S01
@functools.lru_cache(maxsize=None)
def _s01_dust():
    r = rng_for(11)
    n = 420
    return dict(r=r.uniform(140, 900, n), a=r.uniform(0, 2 * np.pi, n), b=r.uniform(0.2, 1.0, n) ** 2,
                tilt=r.normal(0, 0.05, n))


def s01(t):
    p = plate("S01")
    u = t / 5.25
    z = lerp(1.03, 1.20, ease_in_out(u, 1.6))
    cx, cy, z, rot = cam(p, lerp(1000, 880, ease_in_out(u)), lerp(430, 392, ease_in_out(u)), z, lerp(0, -1.2, u))
    bh = (817.0, 375.0)
    warps = (fx.swirl_warp(bh, t, w0=0.20, r0=150, r_in=105, r_out=430, aspect=1.0, inflow=0.006, protect=100),)
    img, M = p.view(cx, cy, z, rot, warps=warps)
    # dust spiralling inward (output space)
    c = fx.to_out(M, [bh])[0]
    d = _s01_dust()
    sc = M[0, 0] ** 2 + M[0, 1] ** 2
    scale = math.sqrt(sc)
    r = d["r"] * (1 - 0.035 * t)
    ang = d["a"] + 0.9 * np.power(150.0 / r, 1.5) * t
    xs = c[0] + np.cos(ang) * r * scale
    ys = c[1] + np.sin(ang) * r * scale * 0.55
    vis = 0.15 * smooth((t - 0.8) / 2.0)
    for x, y, b in zip(xs, ys, d["b"]):
        if 0 <= x < W and 0 <= y < H:
            splat(img, x, y, 1.1, (0.75, 0.95, 1.0), vis * b)
    # the singularity: a spark at the inner edge that swells into the big bang
    sp = fx.to_out(M, [(708.0, 367.0)])[0]
    breath = 0.5 + 0.5 * math.sin(t * 2.2)
    amp = 0.25 + 0.12 * breath + 9.0 * ease_in((t - 4.35) / 0.9, 3.0)
    anamorphic(img, sp[0], sp[1], amp * smooth((t - 0.6) / 1.2), color=(0.55, 0.85, 1.0), length=380 + 900 * ease_in((t - 4.3) / 0.95))
    # exposure: emerge from total black, white-out at the end
    expo = smooth((t - 0.35) / 2.8) * (1.0 + 14.0 * ease_in((t - 4.75) / 0.5, 2.5))
    return img * expo


# ============================================================================ S02
@functools.lru_cache(maxsize=None)
def _flow(name, cell, amp, speed, seed):
    p = plate(name)
    return fx.Flow(p.w, p.h, cell=cell, amp=amp, speed=speed, seed=seed)


@functools.lru_cache(maxsize=None)
def _sparks(seed, n):
    r = rng_for(seed)
    return dict(a=r.uniform(0, 2 * np.pi, n), v=r.uniform(40, 260, n), t0=r.uniform(-4, 5, n),
                b=r.uniform(0.2, 1.0, n) ** 2)


def s02(t):
    p = plate("S02")
    u = t / 5.25
    e = ease_out(u, 2.6)
    z = lerp(1.75, 1.02, e)
    cx, cy, z, rot = cam(p, lerp(1258, 1050, e), lerp(408, 432, e), z, lerp(1.5, 0.0, e))
    flow = _flow("S02", 120, 3.2, 0.5, 2)

    def frame(tt):
        uu = tt / 5.25
        ee = ease_out(uu, 2.6)
        c = cam(p, lerp(1258, 1050, ee), lerp(408, 432, ee), lerp(1.75, 1.02, ee), lerp(1.5, 0.0, ee))
        im, _ = p.view(*c, warps=(flow.warp(tt),))
        return im

    img = fx.mblur(frame, t, n=3 if t < 1.6 else 1)
    M = p.matrix(cx, cy, z, rot)
    star = fx.to_out(M, [(1258.0, 408.0)])[0]
    s = _sparks(21, 160)
    scale = math.hypot(M[0, 0], M[0, 1])
    for a, v, t0, b in zip(s["a"], s["v"], s["t0"], s["b"]):
        age = (t - t0) % 5.0
        rr = (40 + v * age) * scale
        x, y = star[0] + math.cos(a) * rr, star[1] + math.sin(a) * rr * 0.7
        if 0 <= x < W and 0 <= y < H:
            splat(img, x, y, 1.0, (1.0, 0.85, 0.6), 0.10 * b * math.exp(-age / 3.0))
    breath = 0.5 + 0.5 * math.sin(2 * math.pi * t / 2.6)
    anamorphic(img, star[0], star[1], 0.22 + 0.10 * breath, color=(1.0, 0.75, 0.45), length=420)
    flash = 1.0 + 12.0 * math.exp(-t / 0.22)
    return img * flash


# ============================================================================ S03
@functools.lru_cache(maxsize=None)
def _beam_particles():
    r = rng_for(31)
    n = 340
    return dict(x0=r.uniform(-0.2, 1.0, n), y=r.normal(0, 1, n), v=r.uniform(0.18, 0.42, n),
                b=r.uniform(0.25, 1.0, n) ** 2)


def s03(t):
    p = plate("S03")
    u = t / 4.5
    flow = _flow("S03", 70, 5.5, 1.1, 3)
    cx, cy, z, rot = cam(p, lerp(930, 1120, ease_in_out(u, 1.3)), 450, lerp(1.16, 1.24, u), lerp(0.4, -0.6, u))
    img, M = p.view(cx, cy, z, rot, warps=(flow.warp(t),))
    conv = fx.to_out(M, [(1017.0, 450.0)])[0]
    right = fx.to_out(M, [(2048.0, 452.0)])[0]
    left = fx.to_out(M, [(0.0, 440.0)])[0]
    d = _beam_particles()
    layer = np.zeros((H, W, 3), np.uint8)
    for x0, yn, v, b in zip(d["x0"], d["y"], d["v"], d["b"]):
        s = (x0 + v * t) % 1.25 - 0.1   # 0..1 along the river, >0.5 = after convergence
        if s < 0.5:
            k = s / 0.5
            x = lerp(left[0], conv[0], k)
            spread = 150 * (1 - k) ** 1.3 + 3
            y = lerp(left[1], conv[1], k) + yn * spread
            speed = 1.0 + 2.0 * k
        else:
            k = (s - 0.5) / 0.75
            x = lerp(conv[0], right[0] + 300, k ** 1.4)
            y = lerp(conv[1], right[1], k) + yn * 1.5
            speed = 3.0 + 8.0 * k
        L = 6 * speed
        c = int(min(255, 255 * b))
        cv2.line(layer, (int((x - L) * 16), int(y * 16)), (int(x * 16), int(y * 16)), (c, c, c), 1, cv2.LINE_AA, shift=4)
    lay = fx.srgb_to_lin(layer)
    img += lay * np.array([0.7, 1.6, 1.3], np.float32)
    # packets of light racing out along the thin line
    for t0 in (0.6, 2.0, 3.3):
        k = (t - t0) / 0.9
        if 0 <= k <= 1.2:
            x = lerp(conv[0], right[0] + 200, ease_in(k, 1.6))
            y = lerp(conv[1], right[1], k)
            anamorphic(img, x, y, 0.9 * (1 - k / 1.2), color=(0.5, 1.0, 0.85), length=260, ghosts=False)
    add_glow(img, conv[0], conv[1], 30, (0.6, 1.0, 0.9), 0.35)
    return img


# ============================================================================ S04
@functools.lru_cache(maxsize=None)
def _s04_arcs():
    cities = [(67, 563), (383, 467), (450, 683), (617, 650), (750, 550), (1367, 303), (1667, 350), (300, 520)]
    hub = (1060.0, -60.0)
    arcs = []
    r = rng_for(41)
    for i, (x, y) in enumerate(cities):
        c1 = (x + (hub[0] - x) * 0.15, y - 420 - r.uniform(0, 120))
        c2 = (hub[0] + (x - hub[0]) * 0.35, hub[1] + 260)
        ts = np.linspace(0, 1, 90)[:, None]
        P0, P1, P2, P3 = (np.array(v, np.float64) for v in ((x, y), c1, c2, hub))
        pts = (1 - ts) ** 3 * P0 + 3 * (1 - ts) ** 2 * ts * P1 + 3 * (1 - ts) * ts ** 2 * P2 + ts ** 3 * P3
        arcs.append((pts, 0.25 + 0.42 * i + r.uniform(0, 0.2)))
    return arcs


def s04(t):
    p = plate("S04")
    u = t / 5.25
    cx, cy, z, rot = cam(p, lerp(930, 1120, ease_in_out(u, 1.2)), lerp(470, 445, u), 1.14, lerp(-1.4, 0.9, u))
    fl = _flow("S04", 90, 6.0, 0.9, 4)
    dx, dy = fl(t)
    wy = np.clip((330 - np.arange(p.h, dtype=np.float32)) / 120, 0, 1)[:, None]

    def aurora(sx, sy):
        ix = np.clip(sx, 0, p.w - 1).astype(np.int32)
        iy = np.clip(sy, 0, p.h - 1).astype(np.int32)
        k = wy[iy, 0]
        return sx + dx[iy, ix] * k, sy + dy[iy, ix] * k * 0.4

    img, M = p.view(cx, cy, z, rot, warps=(aurora,))
    layer = np.zeros((H, W, 3), np.uint8)
    heads = []
    for pts, t0 in _s04_arcs():
        k = clamp((t - t0) / 1.6)
        if k <= 0:
            continue
        n = max(2, int(len(pts) * ease_out(k, 1.5)))
        q = fx.to_out(M, pts[:n])
        cv2.polylines(layer, [np.round(q * 16).astype(np.int32)], False, (150, 255, 210), 1, cv2.LINE_AA, shift=4)
        if k < 1:
            heads.append(q[-1])
        add_glow(img, *fx.to_out(M, [pts[0]])[0], 6, (1.0, 0.8, 0.5), 0.25 * smooth(k * 3))
    lay = fx.srgb_to_lin(layer)
    soft = cv2.GaussianBlur(lay, (0, 0), 3.0)
    img += lay * 0.9 * np.array([0.4, 1.0, 0.75], np.float32) + soft * 2.2 * np.array([0.25, 1.0, 0.6], np.float32)
    for hx, hy in heads:
        splat(img, hx, hy, 2.2, (0.7, 1.0, 0.9), 2.0)
        add_glow(img, hx, hy, 10, (0.4, 1.0, 0.7), 0.25)
    return img


# ============================================================================ S05
@functools.lru_cache(maxsize=None)
def _pulses(name, seeds, roi=None, thr=0.08, min_area=60):
    return fx.PathPulses(plate(name), list(seeds), thr=thr, roi=roi, min_area=min_area)


@functools.lru_cache(maxsize=None)
def _dust(seed, n, color=(0.75, 1.0, 0.9)):
    return fx.Dust(n=n, seed=seed, color=color)


def s05(t):
    p = plate("S05")
    u = t / 3.75
    cx, cy, z, rot = cam(p, lerp(1080, 1190, ease_in_out(u)), lerp(430, 400, u), lerp(1.04, 1.2, ease_in_out(u)),
                         lerp(0.6, -0.6, u))
    pp = _pulses("S05", ((262, 702), (2040, 402), (1840, 308)))
    field = pp.field(t + 0.8, speed=520, spacing=360, width=34)
    src = p.img + field[..., None] * np.array([0.5, 2.6, 1.6], np.float32)
    img, M = p.view(cx, cy, z, rot, img=src)
    _dust(51, 160).draw(img, t, cam=(0.0, 0.0, 0.45 * ease_in_out(u)), amp=0.18, focus=3.0)
    return img


# ============================================================================ S06 constellation chart
CHART = [0.30, 0.38, 0.34, 0.47, 0.43, 0.55, 0.51, 0.62, 0.58, 0.69, 0.76]


def s06(t):
    p = plate("S06")
    u = t / 4.5
    cx, cy, z, rot = cam(p, lerp(990, 1080, ease_in_out(u)), 432, lerp(1.10, 1.15, u), 0.0)
    img, M = p.view(cx, cy, z, rot)
    n = len(CHART)
    xs = np.linspace(470, 1590, n)
    yv = lambda v: 770 - v * 700
    layer = np.zeros((H, W, 3), np.uint8)
    stars = []
    prev = CHART[0] - 0.06
    for i, v in enumerate(CHART):
        t0 = 0.25 + i * 0.375
        k = clamp((t - t0) / 0.3)
        if k <= 0:
            prev = v
            continue
        o, c = prev, lerp(prev, v, ease_out(k))
        hi, lo = max(o, v) + 0.035, min(o, v) - 0.03
        up = v >= o
        col = (150, 255, 205) if up else (110, 190, 200)
        X = xs[i]
        top = fx.to_out(M, [(X, yv(max(o, c))), (X, yv(min(o, c))), (X, yv(hi)), (X, yv(lo))])
        bw = 11 * z
        x_out = top[0][0]
        pts = np.array([[x_out - bw, top[0][1]], [x_out + bw, top[0][1]], [x_out + bw, top[1][1]], [x_out - bw, top[1][1]]])
        cv2.polylines(layer, [np.round(pts * 16).astype(np.int32)], True, col, 1, cv2.LINE_AA, shift=4)
        wick_k = smooth((t - t0 - 0.15) / 0.3)
        if wick_k > 0:
            cv2.line(layer, (int(x_out * 16), int(top[0][1] * 16)), (int(x_out * 16), int(lerp(top[0][1], top[2][1], wick_k) * 16)),
                     col, 1, cv2.LINE_AA, shift=4)
            cv2.line(layer, (int(x_out * 16), int(top[1][1] * 16)), (int(x_out * 16), int(lerp(top[1][1], top[3][1], wick_k) * 16)),
                     col, 1, cv2.LINE_AA, shift=4)
        fill = np.zeros((H, W), np.uint8)
        cv2.fillPoly(fill, [np.round(pts * 16).astype(np.int32)], 60 if up else 35, cv2.LINE_AA, shift=4)
        img += (fx.srgb_to_lin(np.repeat(fill[..., None], 3, -1)) * (EMERALD if up else MINT) * 0.6)
        star = fx.to_out(M, [(X, yv(c))])[0]
        stars.append((star, k, t - t0))
        prev = v
    # constellation line through the closes
    if len(stars) > 1:
        q = np.array([s[0] for s in stars])
        cv2.polylines(layer, [np.round(q * 16).astype(np.int32)], False, (120, 255, 190), 1, cv2.LINE_AA, shift=4)
    lay = fx.srgb_to_lin(layer)
    img += lay * np.array([0.6, 1.5, 1.1], np.float32) + cv2.GaussianBlur(lay, (0, 0), 4) * np.array([0.3, 1.6, 0.9], np.float32)
    for (sx, sy), k, age in stars:
        flare = 1.0 + 3.0 * math.exp(-age / 0.25)
        splat(img, sx, sy, 1.6, (0.85, 1.0, 0.95), 1.8 * flare)
        add_glow(img, sx, sy, 9, (0.5, 1.0, 0.8), 0.18 * flare)
    return img


# ============================================================================ S07
S07_NODES = ((272, 563), (95, 322), (200, 345), (338, 330), (362, 400), (437, 380), (543, 383), (805, 337),
             (1040, 353), (1133, 338), (942, 470), (1017, 447), (1622, 488))


def s07(t):
    p = plate("S07")
    u = t / 3.0
    cx, cy, z, rot = cam(p, lerp(990, 1040, u), lerp(432, 446, u), lerp(1.02, 1.24, ease_in_out(u, 1.4)), lerp(-0.4, 0.4, u))
    pp = _pulses("S07", S07_NODES, thr=0.06, min_area=150)
    field = pp.field(t + 0.4, speed=700, spacing=420, width=36)
    src = p.img + field[..., None] * np.array([0.5, 2.8, 1.7], np.float32)
    img, M = p.view(cx, cy, z, rot, img=src)
    for nx, ny in S07_NODES:
        k = field[ny, nx] if 0 <= ny < field.shape[0] and 0 <= nx < field.shape[1] else 0
        q = fx.to_out(M, [(nx, ny)])[0]
        add_glow(img, q[0], q[1], 7, (0.5, 1.0, 0.8), 0.05 + 0.6 * k)
    _dust(71, 200).draw(img, t, cam=(0.0, 0.0, 0.5 * u), amp=0.2, focus=3.2)
    return img


# ============================================================================ S08
@functools.lru_cache(maxsize=None)
def _orbiters():
    r = rng_for(81)
    n = 70
    return dict(a=r.uniform(0, 2 * np.pi, n), rr=r.uniform(190, 520, n), b=r.uniform(0.3, 1.0, n))


def s08(t):
    p = plate("S08")
    u = t / 3.0
    cx, cy, z, rot = cam(p, lerp(1020, 1040, u), lerp(420, 410, u), lerp(1.07, 1.18, ease_in_out(u)), lerp(1.2, -2.6, u))
    bh = (1028.0, 408.0)
    warps = (fx.swirl_warp(bh, t, w0=0.12, r0=240, r_in=200, r_out=900, aspect=0.62, tilt=-12, inflow=0.0,
                           protect=185),)
    img, M = p.view(cx, cy, z, rot, warps=warps)
    c = fx.to_out(M, [bh])[0]
    scale = math.hypot(M[0, 0], M[0, 1])
    rot_m = math.atan2(M[1, 0], M[0, 0])
    # a glint orbiting the photon ring
    ang = -1.2 + 2.4 * t
    rr = 140 * scale
    gx, gy = c[0] + math.cos(ang + rot_m) * rr, c[1] + math.sin(ang + rot_m) * rr
    add_glow(img, gx, gy, 10, (1.0, 0.9, 0.7), 0.5)
    # emerald motes orbiting on the tilted disk
    o = _orbiters()
    tilt = math.radians(-12) + rot_m
    for a, r0, b in zip(o["a"], o["rr"], o["b"]):
        aa = a + 0.9 * (220 / r0) ** 1.5 * t
        x, y = math.cos(aa) * r0 * scale, math.sin(aa) * r0 * scale * 0.62
        X = c[0] + x * math.cos(tilt) - y * math.sin(tilt)
        Y = c[1] + x * math.sin(tilt) + y * math.cos(tilt)
        if math.hypot(X - c[0], Y - c[1]) > 135 * scale:
            splat(img, X, Y, 1.2, (0.4, 1.0, 0.7), 0.35 * b)
    return img


# ============================================================================ S09
def s09(t):
    p = plate("S09")
    u = t / 3.0

    def camf(tt):
        uu = tt / 3.0
        return cam(p, lerp(880, 1170, ease_in_out(uu, 1.3)), 445, 1.16, lerp(-1.6, 1.0, uu))

    pp = _pulses("S09", ((560, 312), (1100, 242)), roi=(0, 120, 2048, 430), thr=0.10)
    field = pp.field(t + 0.3, speed=650, spacing=520, width=60)
    src = p.img + field[..., None] * np.array([0.5, 2.4, 1.5], np.float32)

    def frame(tt):
        im, _ = p.view(*camf(tt), img=src)
        return im

    img = fx.mblur(frame, t, n=3)
    M = p.matrix(*camf(t))
    sun = fx.to_out(M, [(983.0, 250.0)])[0]
    rise = ease_in_out(u)
    img = fx.godrays(img, sun[0], sun[1], strength=0.25 + 0.5 * rise, threshold=0.35)
    anamorphic(img, sun[0], sun[1], 0.5 + 1.2 * rise, color=(1.0, 0.85, 0.65), length=900, rays=0.5)
    return img


# ============================================================================ S10
SCREEN = np.float32([[1054, 236], [1312.5, 372.5], [1252.5, 513.5], [956, 366]])
SCR_W, SCR_H = 576, 320


@functools.lru_cache(maxsize=None)
def _screen_series():
    r = rng_for(101)
    v = np.cumsum(r.normal(0.012, 0.05, 40)) + 0.3
    return (v - v.min()) / (v.max() - v.min())


def _screen_ui(t):
    """Abstract emerald trading UI (no readable text) for the phone screen."""
    ui = np.zeros((SCR_H, SCR_W, 3), np.float32)
    ui[:] = np.array([0.004, 0.02, 0.014], np.float32)
    lay = np.zeros((SCR_H, SCR_W, 3), np.uint8)
    for gy in range(60, SCR_H, 52):
        cv2.line(lay, (20, gy), (SCR_W - 20, gy), (18, 40, 32), 1, cv2.LINE_AA)
    s = _screen_series()
    n = int(clamp(0.35 + t / 1.6) * len(s))
    xs = np.linspace(30, SCR_W - 70, len(s))
    ys = SCR_H - 40 - s * (SCR_H - 110)
    for i in range(1, n):
        up = s[i] >= s[i - 1]
        col = (120, 255, 190) if up else (80, 170, 170)
        x = int(xs[i])
        y0, y1 = int(ys[i - 1]), int(ys[i])
        cv2.rectangle(lay, (x - 3, min(y0, y1)), (x + 3, max(y0, y1) + 1), col, -1, cv2.LINE_AA)
        cv2.line(lay, (x, min(y0, y1) - 6), (x, max(y0, y1) + 6), col, 1, cv2.LINE_AA)
    pts = np.stack([xs[:n], ys[:n]], 1)
    if n > 1:
        cv2.polylines(lay, [np.round(pts * 16).astype(np.int32)], False, (190, 255, 225), 2, cv2.LINE_AA, shift=4)
    # top bar: logo mark + abstract pills
    m = logo.raster(logo.MARK, 1.0, 14, 8, shape=(SCR_H, SCR_W))
    ui += m[..., None] * np.array([0.5, 1.4, 1.0], np.float32)
    for k, w in enumerate((70, 46, 58)):
        cv2.rectangle(lay, (60 + k * 90, 18), (60 + k * 90 + w, 26), (40, 90, 70), -1, cv2.LINE_AA)
    lin = fx.srgb_to_lin(lay)
    ui += lin * 1.3
    if n > 1:
        add = np.zeros_like(ui)
        x, y = pts[-1]
        yy, xx = np.mgrid[0:SCR_H, 0:SCR_W].astype(np.float32)
        add += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 6 ** 2))[..., None] * np.array([1.0, 2.5, 1.8], np.float32)
        ui += add
    return ui


def s10(t):
    p = plate("S10")
    u = t / 2.25
    cx, cy, z, rot = cam(p, lerp(1070, 1110, u), lerp(410, 392, u), lerp(1.04, 1.13, ease_in_out(u)), lerp(-0.3, 0.3, u))
    ui = _screen_ui(t)
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [SCR_W, 0], [SCR_W, SCR_H], [0, SCR_H]]), SCREEN)
    warped = cv2.warpPerspective(ui, Hm, (p.w, p.h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((SCR_H, SCR_W), np.float32), Hm, (p.w, p.h), flags=cv2.INTER_LINEAR)
    src = p.img * (1 - 0.92 * mask[..., None]) + warped * 0.95
    img, M = p.view(cx, cy, z, rot, img=src)
    c = fx.to_out(M, [SCREEN.mean(0)])[0]
    add_glow(img, c[0], c[1], 160, (0.2, 1.0, 0.6), 0.035 * (1 + 0.15 * math.sin(t * 7)))
    _dust(111, 140, color=(0.9, 1.0, 0.95)).draw(img, t, cam=(0.05 * u, 0.0, 0.2 * u), amp=0.12, focus=1.2)
    return img


# ============================================================================ S11
@functools.lru_cache(maxsize=None)
def _streaks():
    r = rng_for(121)
    n = 190
    return dict(x=r.uniform(0, 1, n), y=r.uniform(0, 1, n), v=r.uniform(0.6, 2.6, n), b=r.uniform(0.2, 1.0, n) ** 2)


def s11(t):
    p = plate("S11")
    u = t / 2.25
    jx = 2.0 * math.sin(t * 13.0) + 1.3 * math.sin(t * 29.0 + 1.0)
    jy = 1.6 * math.sin(t * 17.0 + 2.0) + 1.0 * math.sin(t * 31.0)
    cx, cy, z, rot = cam(p, 1000 + jx, 505 + jy, lerp(1.07, 1.16, u), 0.25 * math.sin(t * 5.0))
    shield_c = (1300.0, 492.0)
    pulse = 0.5 + 0.5 * math.sin(t * 9.0)

    def shield(sx, sy):
        dx, dy = sx - shield_c[0] + 160, sy - shield_c[1]
        r = np.sqrt(dx * dx + dy * dy) + 1e-3
        lead = np.clip((dx / r - 0.25) / 0.5, 0, 1)
        ring = np.exp(-0.5 * ((r - (300 + 30 * u)) / 26) ** 2) * lead
        k = 5.0 * ring / r
        return sx - dx * k, sy - dy * k

    img, M = p.view(cx, cy, z, rot, warps=(shield,))
    sc = fx.to_out(M, [shield_c])[0]
    scale = math.hypot(M[0, 0], M[0, 1])
    yy, xx = np.ogrid[0:H, 0:W]
    R = (300 + 30 * u) * scale
    ox = sc[0] - 160 * scale
    rr = np.sqrt((xx - ox) ** 2 + (yy - sc[1]) ** 2) + 1e-3
    cosang = (xx - ox) / rr
    lead = np.clip((cosang - 0.25) / 0.5, 0, 1) ** 2
    ang = np.arctan2(yy - sc[1], xx - ox)
    glint = np.exp(-0.5 * ((ang - (-1.0 + 2.0 * u)) / 0.18) ** 2)
    ring = np.exp(-0.5 * ((rr - R) / 3.5) ** 2) * lead * (0.05 + 0.05 * pulse + 0.5 * glint)
    img += ring[..., None].astype(np.float32) * np.array([0.3, 1.0, 0.7], np.float32)
    # streaming background stars (ship racing to the right)
    d = _streaks()
    layer = np.zeros((H, W, 3), np.uint8)
    for x0, y0, v, b in zip(d["x"], d["y"], d["v"], d["b"]):
        x = ((x0 - v * t * 0.55) % 1.2 - 0.1) * W
        y = y0 * H
        L = 30 + 140 * v
        c = int(255 * b * 0.8)
        cv2.line(layer, (int(x * 16), int(y * 16)), (int((x + L) * 16), int(y * 16)), (c, c, c), 1, cv2.LINE_AA, shift=4)
    img += fx.srgb_to_lin(layer) * np.array([0.7, 0.95, 1.0], np.float32) * 0.8
    # engine flicker
    eng = fx.to_out(M, [(800.0, 550.0)])[0]
    fl = 0.7 + 0.3 * math.sin(t * 41.0) * math.sin(t * 23.0)
    anamorphic(img, eng[0], eng[1], 0.6 * fl, color=(0.8, 1.0, 0.9), length=500, ghosts=False)
    return img


# ============================================================================ S12
_warp_stars = None


def _wstars():
    global _warp_stars
    if _warp_stars is None:
        _warp_stars = fx.StarField3D(n=2200, seed=131, depth=30.0)
    return _warp_stars


def s12(t):
    p = plate("S12")
    u = t / 2.25
    z = lerp(1.05, 1.75, ease_in(u, 2.2))
    cx, cy, z, rot = cam(p, 1092, 475, z, lerp(0, 2.0, ease_in(u)))
    img, M = p.view(cx, cy, z, rot)
    c = fx.to_out(M, [(1092.0, 475.0)])[0]
    img = fx.zoom_blur(img, c[0], c[1], amount=0.02 + 0.16 * ease_in(u, 2.0), n=8)
    speed = 4.0 + 60.0 * ease_in(u, 2.0)
    cz = 4.0 * t + 22.0 * ease_in(u, 3.0)
    _wstars().draw(img, cz, speed, cx=c[0], cy=c[1], f=820.0, amp=1.3, streak=2.6, tint=(0.75, 1.0, 0.95))
    add_glow(img, c[0], c[1], 40, (0.8, 1.0, 0.95), 0.6 + 2.5 * ease_in(u, 3.0))
    whiteout = 1.0 + 9.0 * ease_in((t - 1.85) / 0.4, 2.0)
    return img * whiteout


# ============================================================================ S13
def s13(t):
    p = plate("S13")
    u = t / 1.5

    def camf(tt):
        uu = tt / 1.5
        return cam(p, 1024, lerp(470, 520, uu), lerp(1.12, 1.30, ease_out(uu)), lerp(3.0, -2.5, ease_out(uu)))

    img = fx.mblur(lambda tt: p.view(*camf(tt))[0], t, n=4)
    M = p.matrix(*camf(t))
    c = fx.to_out(M, [(1023.0, 1333.0)])[0]
    scale = math.hypot(M[0, 0], M[0, 1])
    rot_m = math.atan2(M[1, 0], M[0, 0])
    for k, a0 in enumerate((-2.2, -1.2)):
        a = a0 + 2.6 * t + rot_m
        R = 1000 * scale
        x, y = c[0] + math.cos(a) * R, c[1] + math.sin(a) * R
        if -100 < x < W + 100 and -100 < y < H + 100:
            anamorphic(img, x, y, 1.2, color=(0.6, 1.0, 0.85), length=300, ghosts=False)
    flash = 1.0 + 3.0 * math.exp(-t / 0.12)
    return img * flash


# ============================================================================ S14
@functools.lru_cache(maxsize=None)
def _rays():
    r = rng_for(141)
    n = 120
    return dict(a=r.uniform(0, 2 * np.pi, n), d0=r.uniform(0, 1, n), v=r.uniform(0.6, 1.4, n), b=r.uniform(0.3, 1.0, n))


def s14(t):
    p = plate("S14")
    u = t / 1.5
    z = lerp(1.05, 1.42, ease_in(u, 1.6))
    cx, cy, z, rot = cam(p, 958, 437, z, lerp(-1.0, 1.0, u))
    img, M = p.view(cx, cy, z, rot)
    c = fx.to_out(M, [(958.0, 437.0)])[0]
    img = fx.zoom_blur(img, c[0], c[1], amount=0.03 + 0.06 * u, n=6)
    d = _rays()
    layer = np.zeros((H, W, 3), np.uint8)
    for a, d0, v, b in zip(d["a"], d["d0"], d["v"], d["b"]):
        k = (d0 - v * t * 0.9) % 1.0
        r1 = 60 + k * 1400
        r0 = r1 + 120 + 240 * k
        x1, y1 = c[0] + math.cos(a) * r1, c[1] + math.sin(a) * r1 * 0.75
        x0, y0 = c[0] + math.cos(a) * r0, c[1] + math.sin(a) * r0 * 0.75
        g = int(255 * b * min(1.0, k * 3))
        cv2.line(layer, (int(x0 * 16), int(y0 * 16)), (int(x1 * 16), int(y1 * 16)), (g, g, g), 2, cv2.LINE_AA, shift=4)
    img += fx.srgb_to_lin(layer) * np.array([0.35, 1.3, 0.75], np.float32)
    add_glow(img, c[0], c[1], 30, (0.7, 1.0, 0.85), 0.8 + 1.5 * u)
    flash = 1.0 + 2.5 * math.exp(-t / 0.12)
    return img * flash


# ============================================================================ S15-S17 logo
LOCK_T = 49.5
PIECE_FROM = [  # dx, dy (px), rot (deg), scale  -- start offsets for each mark piece
    (-980, -300, -60, 1.5), (900, 360, 50, 1.4), (-520, -470, 40, 1.6),
    (960, -420, -45, 1.3), (-860, 330, 70, 1.5), (480, 520, -30, 1.45),
]


def mark_layout(T):
    """Logo placement in output px: returns (scale, ox, oy) for lockup coords."""
    s_hero = 13.0
    cxm, cym = logo.MARK_CENTER
    ox_h, oy_h = W / 2 - cxm * s_hero, H / 2 - 10 - cym * s_hero
    s_lock = 6.6
    bx0, by0, bx1, by1 = logo.LOCKUP_BOX
    ox_l, oy_l = W / 2 - (bx0 + bx1) / 2 * s_lock, H / 2 - 14 - (by0 + by1) / 2 * s_lock
    oy_end = oy_l - 92
    k = ease_in_out((T - 50.55) / 1.0, 2.2)
    s = lerp(s_hero, s_lock, k)
    ox, oy = lerp(ox_h, ox_l, k), lerp(oy_h, oy_l, k)
    k2 = ease_in_out((T - 53.25) / 1.1, 2.0)
    oy += (oy_end - oy_l) * k2 * k
    push = 1.0 + 0.035 * smooth((T - LOCK_T) / 1.0) * (1 - k)
    if push != 1.0:
        cxw, cyw = W / 2, H / 2 - 10
        ox, oy, s = cxw + (ox - cxw) * push, cyw + (oy - cyw) * push, s * push
    return s, ox, oy, k


def draw_logo(img, T):
    s, ox, oy, k = mark_layout(T)
    if T < LOCK_T:
        prog = clamp((T - 47.25) / (LOCK_T - 47.25))

        def pieces_at(pr):
            ms = []
            e = ease_in(pr, 2.6)
            for poly, (dx, dy, rr, sc) in zip(logo.MARK, PIECE_FROM):
                cen = poly.mean(0)
                q = logo.transform_piece(poly, cen, dx * (1 - e) / s, dy * (1 - e) / s, rr * (1 - e), lerp(sc, 1.0, e))
                ms.append(q)
            return ms

        acc = np.zeros((H, W), np.float32)
        nsub = 12
        for j in range(nsub):
            pr = clamp(prog + (j / (nsub - 1) - 0.5) * 0.5 / FPS / (LOCK_T - 47.25) * 2.0)
            acc += logo.raster(pieces_at(pr), s, ox, oy)
        mask = np.clip(acc / nsub, 0, 1)
        mask = cv2.GaussianBlur(mask, (0, 0), 0.6 + 1.2 * ease_in(prog, 3.0)) * smooth(prog / 0.3)
        light = logo.shade(mask, T, intensity=0.7 + 1.3 * ease_in(prog, 3.0))
        logo.composite_emissive(img, mask, light, occlusion=0.7)
        return
    mask = logo.raster(logo.MARK, s, ox, oy)
    sweep = None
    ts = (T - 51.75) / 0.9
    if 0 <= ts <= 1:
        sweep = (lerp(-400, 2400, ease_in_out(ts)), math.radians(28), 60.0, 1.0)
    glow_hit = 1.0 + 2.5 * math.exp(-(T - LOCK_T) / 0.35)
    light = logo.shade(mask, T, sweep=sweep, intensity=glow_hit)
    logo.composite_emissive(img, mask, light, occlusion=0.95)
    wk = smooth((T - 50.65) / 0.9)
    if wk > 0:
        slide = (1 - ease_out(wk, 3)) * 50
        wm = logo.raster(logo.WORD, s, ox + slide, oy)
        if wk < 1:
            wm = cv2.GaussianBlur(wm, (0, 0), 0.1 + 3.0 * (1 - wk))
        wm *= wk
        wl = logo.shade(wm, T, sweep=sweep, intensity=1.0)
        logo.composite_emissive(img, wm, wl, occlusion=0.95)


@functools.lru_cache(maxsize=None)
def _burst():
    r = rng_for(151)
    n = 260
    return dict(a=r.uniform(0, 2 * np.pi, n), v=r.uniform(200, 1300, n), b=r.uniform(0.2, 1.0, n) ** 2)


def s_logo(T):
    p = plate("S16")
    t = T - 47.25
    z = lerp(1.0, 1.1, ease_in_out(clamp(t / 12.75), 1.5))
    cx, cy, z, rot = cam(p, 1000, 425, z, 0.0)
    fl = _flow("S16", 120, 4.0, 0.6, 16)
    img, M = p.view(cx, cy, z, rot, warps=(fl.warp(T),))
    # corona breathes up toward the lock, flares at the lock
    pre = ease_in(clamp(t / 2.25), 2.0)
    post = math.exp(-max(0.0, T - LOCK_T) / 0.6) if T >= LOCK_T else 0.0
    end_dim = 1.0 - 0.70 * smooth((T - 53.1) / 1.7)
    img *= (0.55 + 0.45 * pre + 0.9 * post) * end_dim
    c = fx.to_out(M, [(1000.0, 417.0)])[0]
    hot = fx.to_out(M, [(1333.0, 500.0)])[0]
    hot_amp = (0.15 + 0.6 * pre + 1.6 * post) * (1.0 - smooth((T - 52.4) / 1.3))
    anamorphic(img, hot[0], hot[1], hot_amp, color=(0.5, 1.0, 0.8), length=700, ghosts=True)
    if T >= LOCK_T:
        tt = T - LOCK_T
        scale = math.hypot(M[0, 0], M[0, 1])
        logo.shockwave(img, W / 2, H / 2 - 10, 60 + 1500 * tt ** 0.7, 10 + 30 * tt, 1.4 * math.exp(-tt / 0.5))
        b = _burst()
        for a, v, br in zip(b["a"], b["v"], b["b"]):
            rr = v * (1 - math.exp(-tt / 0.55)) * 0.75
            x, y = W / 2 + math.cos(a) * rr, H / 2 - 10 + math.sin(a) * rr * 0.6
            fade = math.exp(-tt / 1.4)
            if 0 <= x < W and 0 <= y < H and fade > 0.02:
                splat(img, x, y, 1.3, (0.6, 1.0, 0.85), 0.9 * br * fade)
    draw_logo(img, T)
    flash = 1.0 + (2.2 * math.exp(-(T - LOCK_T) / 0.16) if T >= LOCK_T else 0.0)
    fade_out = 1.0 - smooth((T - 58.9) / 1.0)
    return img * flash * fade_out


# ============================================================================ timeline
SHOTS = [
    (0.00, 5.25, s01), (5.25, 10.50, s02), (10.50, 15.00, s03), (15.00, 20.25, s04),
    (20.25, 24.00, s05), (24.00, 28.50, s06), (28.50, 31.50, s07), (31.50, 34.50, s08),
    (34.50, 37.50, s09), (37.50, 39.75, s10), (39.75, 42.00, s11), (42.00, 44.25, s12),
    (44.25, 45.75, s13), (45.75, 47.25, s14),
]
SUPER_POS = {  # placement of supers (fractions of picture width, height)
    "RWA · REAL-WORLD ASSETS": (0.5, 0.80), "SPOT TRADING · BTC/USDT": (0.5, 0.86),
    "7×24 · GLOBAL SERVICE": (0.5, 0.80), "TRADE ON THE APP": (0.27, 0.22), "ASSET SECURITY": (0.5, 0.17),
    "SPEED": (0.5, 0.50), "CONNECT": (0.5, 0.80), "GLOBAL": (0.5, 0.84),
}


def picture(T):
    if T >= 47.25:
        return s_logo(T)
    for a, b, fn in SHOTS:
        if a <= T < b:
            return fn(T - a)
    return np.zeros((H, W, 3), np.float32)


def endcard(disp, T):
    ec = EDL["endcard"]
    k1 = smooth((T - 53.75) / 0.9)
    k2 = smooth((T - 54.35) / 0.9)
    k3 = smooth((T - 55.3) / 0.8)
    fade = 1.0 - smooth((T - 58.9) / 1.0)
    y0 = H / 2 + 70
    if k1 > 0:
        a = typo.text_alpha(ec["slogan_en"].upper(), "en", 25, 320, round(0.26 + 0.04 * k1, 3))
        typo.place(disp, a, W / 2, y0 + 20, color=(0.92, 0.97, 0.95), opacity=k1 * fade, glow=0.15,
                   glow_color=(0.4, 1.0, 0.7))
    if k2 > 0:
        a = typo.text_alpha(ec["slogan_cn"], "serif", 27, 400, 0.42)
        typo.place(disp, a, W / 2, y0 + 64, color=(0.80, 0.92, 0.88), opacity=k2 * fade)
    if k3 > 0:
        a = typo.text_alpha(ec["url"], "en", 30, 500, 0.22)
        typo.place(disp, a, W / 2, y0 + 140, color=(0.36, 1.0, 0.70), opacity=k3 * fade, glow=0.35,
                   glow_color=(0.2, 1.0, 0.6))
        a = typo.text_alpha(ec["tag_en"].upper() + "   ·   " + ec["tag_cn"], "sans", 19, 380, 0.2)
        typo.place(disp, a, W / 2, y0 + 186, color=(0.62, 0.70, 0.68), opacity=k3 * fade * 0.9)


HITS = [(44.25, 7.0), (45.75, 7.0), (47.25, 7.0), (49.5, 10.0)]


def impact_shake(lin, T):
    """Short decaying camera shake on the big percussive hits."""
    dx = dy = 0.0
    zoom = 1.0
    for th, amp in HITS:
        u = T - th
        if 0 <= u < 0.45:
            a = amp * math.exp(-u / 0.12)
            dx += a * math.sin(2 * math.pi * 17 * u + th)
            dy += a * 0.7 * math.cos(2 * math.pi * 13 * u + 2 * th)
            zoom += 0.012 * math.exp(-u / 0.15)
    if abs(dx) < 0.05 and abs(dy) < 0.05 and zoom == 1.0:
        return lin
    M = np.array([[zoom, 0, (1 - zoom) * W / 2 + dx], [0, zoom, (1 - zoom) * H / 2 + dy]], np.float32)
    return cv2.warpAffine(lin, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def render_frame(T, rng=None):
    if rng is None:
        rng = np.random.default_rng(int(T * FPS) + 7)
    lin = impact_shake(picture(T), T)
    disp = fx.grade(lin, bloom_amt=0.32)
    for s in EDL["supers"]:
        px, py = SUPER_POS.get(s["en"], (0.5, 0.8))
        typo.title_card(disp, T, s["en"], s["cn"], s["start"], s["end"], cx=W * px, cy=H * py,
                        big=bool(s.get("big")))
    endcard(disp, T)
    disp = fx.grain(disp, rng, amt=0.020)
    full = fx.letterbox(disp)
    for v in EDL["vo"]:
        dur = VO_META[v["id"]]["ms"] / 1000.0
        cn, en = EDL["subs"][v["id"]]
        start = EDL.get("vo_start_" + LANG, {}).get(v["id"], v["start"])
        end = start + dur + 0.25
        if v["id"] == "V8":
            end = min(end, 53.3)
        typo.subtitle(full, T, cn, en, start + 0.05, end)
    return full
