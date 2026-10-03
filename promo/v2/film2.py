"""ETZ: LIGHT BEYOND BOUNDARIES — V2. Every frame is rendered from 3D scenes in
code (no generated images). Narration: MiniMax Speech-2.8-HD.
render_frame(T) -> 1920x1080 display-referred float RGB."""
import functools
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
import fx  # noqa: E402
import typo  # noqa: E402
from fx import anamorphic, add_glow, clamp, ease_in, ease_in_out, ease_out, lerp, smooth, splat  # noqa: E402

import bh  # noqa: E402
import gfx  # noqa: E402
import lines3d  # noqa: E402
import logo3d  # noqa: E402
import planet  # noqa: E402
import scenes  # noqa: E402
from gfx import Camera, H, W  # noqa: E402

EDL = json.load(open(os.path.join(HERE, "..", "script", "edl.json")))
LANG = os.environ.get("ETZ_LANG", "cn")
VO_META = json.load(open(os.path.join(HERE, "..", "assets", "audio", "vo", LANG, "meta.json")))
FPS = 24
FAST = os.environ.get("ETZ_FAST", "0") == "1"


def rng(seed):
    return np.random.default_rng(seed)


def sph(r, az_deg, el_deg):
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([r * math.cos(e) * math.sin(a), r * math.sin(e), r * math.cos(e) * math.cos(a)])


def sky(cam, gain=1.0, haze=1.0):
    return gfx.render_sky(cam, star_gain=gain, haze_gain=haze)


# ============================================================================ S01 black hole (cold open)
def s01(t):
    u = t / 5.25
    e = ease_in_out(u, 1.6)
    cam = Camera(sph(lerp(40.0, 23.0, e), lerp(-24.0, -6.0, e), lerp(15.0, 7.5, e)), (0, 0.35, 0),
                 fov=lerp(34.0, 38.0, e), roll=lerp(-4.0, -1.0, e))
    img = bh.render(cam, 2.0 + t, gfx.lens_sky(), gain=lerp(1.6, 2.4, e), sky_gain=1.6, ss=1.0 if FAST else 1.3)
    expo = smooth((t - 0.25) / 2.9) * (1.0 + 16.0 * ease_in((t - 4.65) / 0.6, 2.6))
    return img * expo


# ============================================================================ S02 big bang -> galaxy
@functools.lru_cache(maxsize=None)
def _galaxy():
    pos, col, flux = scenes.galaxy()
    d, sp = scenes.bigbang_dirs(len(pos))
    r = rng(8)
    order = r.permutation(len(pos))
    return pos, col, flux, d, sp, order


def _bang_state(tt):
    pos, col, flux, d, sp, order = _galaxy()
    # explosion radius (decelerating) then settle into the galaxy, which spins
    R = 1.5 * sp * (1.0 - math.exp(-max(tt, 0.0) / 0.55))
    boom = d * R[:, None]
    m = ease_in_out(clamp((tt - 1.15) / 2.6), 2.0)
    ang = 0.25 * tt
    ca, sa = math.cos(ang), math.sin(ang)
    g = pos @ np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]]).T
    return boom * (1 - m) + g * m, m


def s02(t):
    pos, col, flux, d, sp, order = _galaxy()
    e = ease_out(clamp(t / 5.25), 1.8)
    cam = Camera(sph(lerp(1.7, 2.75, e), lerp(10.0, 32.0, e), lerp(6.0, 38.0, e)), (0, 0, 0), fov=lerp(62.0, 50.0, e),
                 roll=lerp(-8.0, 0.0, e))
    img = sky(cam, 0.8, 0.6) * smooth((t - 0.8) / 2.0)
    p1, m = _bang_state(t)
    p0, _ = _bang_state(t - 0.5 / FPS * 1.6)
    heat = math.exp(-t / 0.9)
    _, _, _, d, sp, _ = _galaxy()
    k = clamp(t / 1.4)
    # fireball colours: white core cooling to gold, outer shell emerald (by particle speed)
    hot = np.array([1.0, 0.93, 0.8]) * (1 - k) + np.array([1.0, 0.62, 0.22]) * k
    shell = np.array([0.30, 1.0, 0.66])
    w = np.clip((sp - 0.55) / 0.4, 0, 1)[:, None]
    base = hot * (1 - w) + shell * w
    c = col * m + base * (1 - m)
    f = flux * (0.55 * m + (0.25 + 3.0 * heat) * (1 - m)) * 1.0
    scenes.splat_world(img, cam, p0, p1, c, f, sigma=0.85, z_ref=2.5)
    # the primordial flash at the centre
    cs = cam.project(np.zeros((1, 3)))
    add_glow(img, cs[0][0], cs[1][0], 30 + 120 * heat, (1.0, 0.92, 0.8), 2.5 * heat + 0.25 * m)
    anamorphic(img, cs[0][0], cs[1][0], 1.5 * heat + 0.35 * m, color=(1.0, 0.85, 0.6), length=900, ghosts=False)
    flash = 1.0 + 9.0 * math.exp(-t / 0.18)
    return img * flash


# ============================================================================ S03 rivers of light
@functools.lru_cache(maxsize=None)
def _rivers():
    r = rng(31)
    streams = []
    for i in range(14):
        a0 = r.uniform(0, 2 * np.pi)
        rad = r.uniform(2.0, 5.0)
        start = np.array([math.cos(a0) * rad, math.sin(a0) * rad * 0.6, r.uniform(-4, 6)])
        ctrl = start * 0.5 + np.array([r.normal(0, 1.5), r.normal(0, 1.0), 14.0])
        end = np.array([0.0, 0.0, 34.0])
        n = 3400
        u = r.uniform(0, 1, n)
        jitter = r.normal(0, 1, (n, 3)) * np.array([0.12, 0.12, 0.3])
        hue = r.uniform(0, 1)
        streams.append((start, ctrl, end, u, jitter, hue, r.uniform(0.10, 0.18)))
    return streams


def _bez(p0, p1, p2, s):
    s = s[:, None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s * s * p2


def s03(t):
    u = t / 4.5
    cz = lerp(-6.0, 4.0, ease_in_out(u, 1.3))
    cam = Camera((0.3 * math.sin(t * 0.5), 0.2, cz), (0.0, 0.0, cz + 20.0), fov=64, roll=lerp(-3, 3, u))
    img = sky(cam, 0.9, 0.8)
    for (p0, p1, p2, uu, jit, hue, speed) in _rivers():
        s1 = (uu + t * speed) % 1.0
        s0 = (uu + (t - 0.5 / FPS * 3) * speed) % 1.0
        ok = s1 > s0
        a = _bez(p0, p1, p2, s1) + jit * (1 - s1)[:, None]
        b = _bez(p0, p1, p2, s0) + jit * (1 - s0)[:, None]
        col = np.array([0.35 + 0.65 * hue, 1.0, 0.6 + 0.2 * hue]) if hue < 0.75 else np.array([1.0, 0.75, 0.35])
        cc = np.broadcast_to(col, (ok.sum(), 3)).copy()
        f = np.full(ok.sum(), 0.07) * (0.35 + 1.65 * s1[ok] ** 2)
        scenes.splat_world(img, cam, b[ok], a[ok], cc, f, sigma=0.75, z_ref=6.0, trail=1.0)
    core = cam.project(np.array([[0.0, 0.0, 34.0]]))
    anamorphic(img, core[0][0], core[1][0], 0.9 + 0.3 * math.sin(t * 3.0), color=(0.5, 1.0, 0.85), length=650,
               ghosts=False)
    return img


# ============================================================================ Earth helpers
EUROPE = [(51.5, -0.1), (48.9, 2.35), (52.5, 13.4), (40.4, -3.7), (41.9, 12.5), (59.3, 18.1), (55.8, 37.6),
          (52.2, 21.0), (41.0, 29.0), (52.4, 4.9), (45.5, 9.2), (48.2, 16.4), (60.2, 24.9), (50.1, 14.4)]
WORLD = [(40.7, -74.0), (37.8, -122.4), (-23.5, -46.6), (51.5, -0.1), (25.2, 55.3), (19.1, 72.9), (1.35, 103.8),
         (22.3, 114.2), (31.2, 121.5), (35.7, 139.7), (-33.9, 151.2), (6.5, 3.4), (-26.2, 28.0), (37.6, 127.0),
         (34.0, -118.2), (19.4, -99.1), (48.9, 2.35), (55.8, 37.6), (13.8, 100.5), (-6.2, 106.8)]


def horizon_cam(lat, lon, alt, hlat, hlon, pitch_down, fov, roll=0.0):
    P = planet.latlon_to_world(lat, lon, r=1.0 + alt)
    up = P / np.linalg.norm(P)
    Q = planet.latlon_to_world(hlat, hlon)
    fwd = Q - P
    fwd -= up * (fwd @ up)
    fwd /= np.linalg.norm(fwd)
    a = math.radians(pitch_down)
    look = fwd * math.cos(a) - up * math.sin(a)
    return Camera(P, P + look, fov=fov, up=up, roll=roll), fwd, up


def earth(cam, sun, **kw):
    img = sky(cam, 1.0, 0.8)
    col, a = planet.render(cam, sun, **kw)
    return img * (1 - a[..., None]) + col


# ============================================================================ S04 low orbit, Europe at night
FAR = [(40.7, -74.0), (25.2, 55.3), (1.35, 103.8), (35.7, 139.7), (-23.5, -46.6), (19.1, 72.9)]


def _eu_links():
    r = rng(41)
    links, arcs = [], []
    for i in range(len(EUROPE)):
        for j in r.choice(len(EUROPE), 2, replace=False):
            if j != i:
                links.append((i, ("eu", int(j)), 0.05, 0.3 + 0.22 * i + r.uniform(0, 0.3)))
    for i, f in enumerate(FAR):
        links.append((int(r.integers(0, len(EUROPE))), ("far", i), 0.22, 1.2 + 0.45 * i))
    for ia, (kind, j), lift, t0 in links:
        a = planet.latlon_to_world(*EUROPE[ia])
        b = planet.latlon_to_world(*(EUROPE[j] if kind == "eu" else FAR[j]))
        arcs.append(lines3d.slerp_arc(a, b, 90, lift=lift * (3.0 if kind == "eu" else 1.0)))
    return [(ia, j, lift, t0) for ia, (_, j), lift, t0 in links], arcs


EU_LINKS, EU_ARCS = _eu_links()
def s04(t):
    u = t / 5.25
    cam, fwd, up = horizon_cam(lerp(47.0, 49.5, u), lerp(6.0, 13.0, u), 0.062, 70, 22, pitch_down=lerp(16, 13, u),
                               fov=60, roll=lerp(-2.0, 1.5, u))
    sun = -up * 0.95 - fwd * 0.3
    img = earth(cam, sun, aurora=1.0, city_gain=2.6, t=t + 3.0)
    layer = np.zeros((H, W, 3), np.uint8)
    heads = []
    for i, (ia, ib, lift, t0) in enumerate(EU_LINKS):
        k = clamp((t - t0) / 1.3)
        if k <= 0:
            continue
        pts = EU_ARCS[i]
        n = max(2, int(len(pts) * ease_out(k, 1.6)))
        lines3d.draw_polyline(layer, cam, pts[:n], (150, 255, 205), 1, occlude_R=1.0)
        if k < 1:
            heads.append(pts[n - 1])
        a = planet.latlon_to_world(*EUROPE[ia], r=1.0005)
        lines3d.points_light(img, cam, a[None], 0.2 * smooth(k * 4), (1.0, 0.85, 0.6), 2.5, occlude_R=1.0)
    img += lines3d.layer_to_light(layer, (0.35, 1.0, 0.72), gain=1.1, glow=1.5)
    if heads:
        lines3d.points_light(img, cam, np.array(heads), 1.6, (0.7, 1.0, 0.85), 1.6, occlude_R=1.0)
    return img


# ============================================================================ S05 global network
@functools.lru_cache(maxsize=None)
def _routes():
    r = rng(51)
    pairs = []
    view = planet.latlon_to_world(25, 120)
    near = [i for i, c in enumerate(WORLD) if planet.latlon_to_world(*c) @ view > 0.12]
    for ii, i in enumerate(near):
        for j in near[ii + 1:]:
            if r.uniform() < 0.55:
                pairs.append((i, j, r.uniform(0, 2.4)))
    arcs = []
    for i, j, t0 in pairs:
        a = planet.latlon_to_world(*WORLD[i])
        b = planet.latlon_to_world(*WORLD[j])
        arcs.append((lines3d.slerp_arc(a, b, 90, lift=0.07), t0))
    return arcs


def s05(t):
    u = t / 3.75
    p = planet.latlon_to_world(lerp(22.0, 28.0, u), lerp(128.0, 112.0, u), r=lerp(2.75, 3.25, ease_out(u)))
    cam = Camera(p, (0, 0, 0), fov=42, roll=lerp(-6, -2, u))
    sun = planet.latlon_to_world(10, 25)
    img = earth(cam, sun, city_gain=3.2, t=t)
    layer = np.zeros((H, W, 3), np.uint8)
    heads = []
    for pts, t0 in _routes():
        k = clamp((t + 0.6 - t0 * 0.5) / 1.1)
        if k <= 0:
            continue
        n = max(2, int(len(pts) * ease_out(k)))
        lines3d.draw_polyline(layer, cam, pts[:n], (110, 255, 190), 1, occlude_R=1.0)
        ph = (t * 0.55 + t0) % 1.0
        heads.append(pts[int(ph * (n - 1))])
    img += lines3d.layer_to_light(layer, (0.3, 1.0, 0.7), gain=0.9, glow=1.4)
    if heads:
        lines3d.points_light(img, cam, np.array(heads), 1.4, (0.75, 1.0, 0.9), 1.4, occlude_R=1.0)
    cities = np.array([planet.latlon_to_world(la, lo, r=1.001) for la, lo in WORLD])
    lines3d.points_light(img, cam, cities, 0.6, (1.0, 0.85, 0.55), 2.0, occlude_R=1.0)
    return img


# ============================================================================ S06 constellation candles
CHART = [0.30, 0.38, 0.34, 0.47, 0.43, 0.55, 0.51, 0.62, 0.58, 0.69, 0.76]


@functools.lru_cache(maxsize=None)
def _dust3d():
    r = rng(61)
    n = 2500
    p = np.stack([r.uniform(-14, 14, n), r.uniform(-7, 7, n), r.uniform(2, 40, n)], 1)
    f = r.uniform(0.2, 1.0, n) ** 3 * 0.5
    return p, f


def s06(t):
    u = t / 4.5
    cam = Camera((lerp(-1.6, 1.4, ease_in_out(u)), lerp(0.4, -0.2, u), lerp(-1.0, 1.0, u)), (0.0, 0.4, 15.0), fov=52,
                 roll=lerp(1.5, -1.0, u))
    img = sky(cam, 1.1, 0.9)
    p, f = _dust3d()
    lines3d.points_light(img, cam, p, f, (0.85, 0.95, 1.0), 0.9)
    n = len(CHART)
    xs = np.linspace(9.5, -9.5, n)  # camera looks down +z, so +x is screen-left
    zc = 15.0
    yv = lambda v: (v - 0.5) * 12.0 + 0.4
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
        up = v >= o
        col = (150, 255, 205) if up else (110, 200, 215)
        z = zc + 0.9 * math.sin(i * 1.7)
        X = xs[i]
        bw = 0.16
        top, bot = yv(max(o, c)), yv(min(o, c))
        box = np.array([[X - bw, top, z], [X + bw, top, z], [X + bw, bot, z], [X - bw, bot, z], [X - bw, top, z]])
        lines3d.draw_polyline(layer, cam, box, col, 1)
        wk = smooth((t - t0 - 0.15) / 0.3)
        if wk > 0:
            lines3d.draw_polyline(layer, cam, np.array([[X, top, z], [X, lerp(top, yv(max(o, v) + 0.035), wk), z]]), col, 1)
            lines3d.draw_polyline(layer, cam, np.array([[X, bot, z], [X, lerp(bot, yv(min(o, v) - 0.03), wk), z]]), col, 1)
        stars.append((np.array([X, yv(c), z]), t - t0))
        prev = v
    if len(stars) > 1:
        lines3d.draw_polyline(layer, cam, np.array([s[0] for s in stars]), (120, 255, 190), 1)
    img += lines3d.layer_to_light(layer, (0.45, 1.0, 0.75), gain=1.2, glow=1.6)
    for pos, age in stars:
        fl = 1.0 + 3.5 * math.exp(-age / 0.25)
        lines3d.points_light(img, cam, pos[None], 2.2 * fl, (0.85, 1.0, 0.95), 1.6)
        sx, sy, _ = cam.project(pos[None])
        add_glow(img, sx[0], sy[0], 10, (0.5, 1.0, 0.8), 0.15 * fl)
    return img


# ============================================================================ S07 data lattice
@functools.lru_cache(maxsize=None)
def _lattice():
    r = rng(71)
    g = np.mgrid[-6:7, -4:5, 0:40].reshape(3, -1).T.astype(np.float64)
    g = g * np.array([2.2, 2.0, 2.2]) + r.normal(0, 0.35, g.shape)
    keep = r.uniform(0, 1, len(g)) < 0.55
    g = g[keep]
    from scipy.spatial import cKDTree
    tree = cKDTree(g)
    pairs = tree.query_pairs(2.9)
    edges = np.array([p for p in pairs if r.uniform() < 0.55])
    pulse = r.uniform(0, 1, len(edges))
    return g, edges, pulse


def s07(t):
    g, edges, pulse = _lattice()
    u = t / 3.0
    cz = lerp(-4.0, 16.0, ease_in(u, 1.3))
    cam = Camera((0.6 * math.sin(t), 0.4, cz), (0.0, 0.0, cz + 10.0), fov=70, roll=lerp(0, 8, u))
    img = sky(cam, 0.7, 0.5)
    d = g - cam.pos
    dist = np.linalg.norm(d, axis=1)
    layer = np.zeros((H, W, 3), np.uint8)
    a, b = edges[:, 0], edges[:, 1]
    near = (dist[a] < 22) & (dist[b] < 22) & ((d[a] @ cam.f) > 0.2) & ((d[b] @ cam.f) > 0.2)
    sxa, sya, _ = cam.project(g[a[near]])
    sxb, syb, _ = cam.project(g[b[near]])
    fa = np.clip(1.0 - dist[a[near]] / 22, 0, 1) ** 1.5
    for x0, y0, x1, y1, f in zip(sxa, sya, sxb, syb, fa):
        c = int(255 * f)
        cv2.line(layer, (int(x0 * 16), int(y0 * 16)), (int(x1 * 16), int(y1 * 16)), (c, c, c), 1, cv2.LINE_AA, shift=4)
    img += lines3d.layer_to_light(layer, (0.25, 0.95, 0.65), gain=1.0, glow=1.4)
    # pulses running along edges
    ph = (pulse[near] + t * 0.9) % 1.0
    pts = g[a[near]] + (g[b[near]] - g[a[near]]) * ph[:, None]
    lines3d.points_light(img, cam, pts, 0.9 * fa, (0.7, 1.0, 0.85), 1.3)
    nodes = dist < 22
    lines3d.points_light(img, cam, g[nodes], 1.3 * np.clip(1.0 - dist[nodes] / 22, 0, 1) ** 2, (0.6, 1.0, 0.85), 1.5)
    return img


# ============================================================================ S08 black hole, edge-on orbit
def s08(t):
    u = t / 3.0
    cam = Camera(sph(lerp(17.5, 15.0, u), lerp(30.0, 52.0, ease_in_out(u)), lerp(3.2, 2.0, u)), (0, 0.0, 0), fov=52,
                 roll=lerp(2.0, -3.0, u))
    return bh.render(cam, 40.0 + t, gfx.lens_sky(), gain=2.5, sky_gain=1.6, ss=1.0 if FAST else 1.3)


# ============================================================================ S09 orbital sunrise + ring
@functools.lru_cache(maxsize=None)
def _ring():
    s = np.linspace(0, 2 * np.pi, 360)
    tilt = math.radians(-18)
    pts = np.stack([np.cos(s) * 1.32, np.sin(s) * 1.32 * math.sin(tilt), np.sin(s) * 1.32 * math.cos(tilt)], 1)
    return pts


def s09(t):
    u = t / 3.0
    cam, fwd, up = horizon_cam(15.0, lerp(70.0, 74.0, u), 0.07, 14, 100, pitch_down=16.5, fov=60, roll=lerp(-4, 1, u))
    dip = math.acos(1 / 1.07)
    e = -dip + lerp(-0.035, 0.022, ease_in_out(u))
    sun = fwd * math.cos(e) + up * math.sin(e)
    img = earth(cam, sun, city_gain=2.4, t=t)
    # the sun itself once it clears the limb
    sp = cam.pos + sun * 50.0
    sx, sy, z = cam.project(sp[None])
    if z[0] > 0 and not lines3d._occluded(cam, sp[None])[0]:
        k = smooth((e + dip + 0.004) / 0.02)
        anamorphic(img, sx[0], sy[0], 4.0 * k, color=(1.0, 0.85, 0.65), length=1100, rays=1.0)
    # orbital ring of light around the planet (rotated to cross the view)
    R = planet.rot_matrix(0.0, 0.0)
    ring = _ring() @ np.array([[1, 0, 0], [0, 0.94, -0.34], [0, 0.34, 0.94]]).T
    q = planet.latlon_to_world(15, 72) * 0.0
    layer = np.zeros((H, W, 3), np.uint8)
    lines3d.draw_polyline(layer, cam, ring + q, (90, 230, 170), 1, occlude_R=1.0)
    img += lines3d.layer_to_light(layer, (0.3, 1.0, 0.7), gain=0.7, glow=1.2)
    for k in range(6):
        idx = int(((k / 6 + t * 0.12) % 1.0) * (len(ring) - 1))
        lines3d.points_light(img, cam, ring[idx][None], 3.0, (0.8, 1.0, 0.9), 2.0, occlude_R=1.0)
    return img


# ============================================================================ S10 glass phone
SCR_W, SCR_H = 300, 640


@functools.lru_cache(maxsize=None)
def _series():
    r = rng(101)
    v = np.cumsum(r.normal(0.012, 0.05, 44)) + 0.3
    return (v - v.min()) / (v.max() - v.min())


def phone_ui(t):
    ui = np.zeros((SCR_H, SCR_W, 3), np.float32)
    ui[:] = (0.004, 0.018, 0.013)
    lay = np.zeros((SCR_H, SCR_W, 3), np.uint8)
    for gy in range(150, 560, 48):
        cv2.line(lay, (18, gy), (SCR_W - 18, gy), (16, 36, 30), 1, cv2.LINE_AA)
    s = _series()
    n = int(clamp(0.35 + t / 1.6) * len(s))
    xs = np.linspace(26, SCR_W - 26, len(s))
    ys = 540 - s * 360
    for i in range(1, n):
        up = s[i] >= s[i - 1]
        col = (120, 255, 190) if up else (80, 170, 170)
        x = int(xs[i])
        y0, y1 = int(ys[i - 1]), int(ys[i])
        cv2.rectangle(lay, (x - 2, min(y0, y1)), (x + 2, max(y0, y1) + 1), col, -1, cv2.LINE_AA)
        cv2.line(lay, (x, min(y0, y1) - 6), (x, max(y0, y1) + 6), col, 1, cv2.LINE_AA)
    if n > 1:
        pts = np.stack([xs[:n], ys[:n]], 1)
        cv2.polylines(lay, [np.round(pts * 16).astype(np.int32)], False, (190, 255, 225), 2, cv2.LINE_AA, shift=4)
    import logo as logo2d
    m = logo2d.raster(logo2d.MARK, 1.6, 18, 22, shape=(SCR_H, SCR_W))
    ui += m[..., None] * np.array([0.5, 1.4, 1.0], np.float32)
    for k, w in enumerate((60, 40, 50)):
        cv2.rectangle(lay, (70 + k * 72, 34), (70 + k * 72 + w, 42), (40, 90, 70), -1, cv2.LINE_AA)
    cv2.rectangle(lay, (18, 580), (SCR_W - 18, 618), (40, 120, 85), -1, cv2.LINE_AA)
    ui += np.power(lay.astype(np.float32) / 255, 2.2) * 1.4
    return ui


def s10(t):
    u = t / 2.25
    cam, fwd, up = horizon_cam(42.0, lerp(-8.0, -5.0, u), 0.08, 52, 10, pitch_down=6.0, fov=48)
    sun = -up * 0.9 - fwd * 0.4
    img = earth(cam, sun, city_gain=2.0, aurora=0.6, t=t + 9.0)
    # phone floating ~3 units ahead of the camera, slowly turning
    center = cam.pos + cam.f * 6.2 + cam.r * 1.7 + cam.u * 0.1
    R = logo3d.rot_axis((0, 1, 0), lerp(0.55, 0.2, ease_in_out(u))) @ logo3d.rot_axis((1, 0, 0), -0.12)
    basisR = np.stack([cam.r, cam.u, -cam.f], 1)
    Rw = R @ basisR
    inv = np.zeros((4, 4))
    inv[:3, :3] = Rw.T
    inv[:3, 3] = -Rw.T @ center
    al = np.zeros((H, W), np.float32)
    out = np.zeros((H, W, 3), np.float32)
    scenes.render_phone(out, al, cam.pos, cam.basis(), cam.th, inv, phone_ui(t), gfx.lens_sky(),
                        np.array([0, 0, W, H], np.int64), 0.6)
    img = img * (1 - al[..., None]) + out
    return img


# ============================================================================ S11 shield around the mark
def mark_xfs(transforms):
    n = logo3d.N_PARTS
    xfs, vis = [], []
    for k in range(n):
        if k < logo3d.N_MARK and transforms is not None:
            R, T, s = transforms[k]
            xfs.append(logo3d.part_xf(R, T, s))
            vis.append(1.0)
        else:
            xfs.append(logo3d.part_xf(np.eye(3), np.zeros(3)))
            vis.append(0.0)
    return xfs, vis


def s11(t):
    u = t / 2.25
    cam = Camera(sph(lerp(68.0, 58.0, u), lerp(-18.0, 12.0, ease_in_out(u)), lerp(10.0, 4.0, u)), (0, 0, 0), fov=34)
    img = sky(cam, 1.0, 0.8)
    rot = logo3d.rot_axis((0, 1, 0), 0.5 * t)
    xfs, vis = mark_xfs([(rot, np.zeros(3), 0.62)] * logo3d.N_MARK)
    col, a = logo3d.render(cam, xfs, vis, gfx.lens_sky(), glow=1.0, rect=(560, 150, 1360, 660))
    img = img * (1 - a[..., None]) + col
    imp = np.array([[0.55, 0.35, 0.76, 0.55], [-0.7, 0.2, 0.68, 1.15], [0.2, -0.6, 0.77, 1.7]], np.float64)
    sh = np.zeros_like(img)
    scenes.render_shield(sh, cam.pos, cam.basis(), cam.th, 15.0, t, imp, 0.30, clamp(t / 0.9))
    img += sh
    # incoming streaks that strike the shield
    for (dx, dy, dz, t0) in imp:
        k = (t - (t0 - 0.35)) / 0.35
        if 0 <= k <= 1:
            dirv = np.array([dx, dy, dz])
            p1 = dirv * lerp(60, 15.2, k)
            p0 = dirv * lerp(60, 15.2, max(0, k - 0.25))
            lines3d.points_light(img, cam, p1[None], 6.0, (0.8, 1.0, 0.9), 1.6)
            layer = np.zeros((H, W, 3), np.uint8)
            lines3d.draw_polyline(layer, cam, np.array([p0, p1]), (200, 255, 230), 2)
            img += lines3d.layer_to_light(layer, (0.5, 1.0, 0.8), 1.0, 1.5)
    return img


# ============================================================================ S12 hyperspace
_WS = None


def s12(t):
    global _WS
    if _WS is None:
        _WS = fx.StarField3D(n=2600, seed=131, depth=30.0)
    u = t / 2.25
    img = np.zeros((H, W, 3), np.float32)
    speed = lerp(0.5, 9.0, ease_in(u, 2.0))
    scenes.render_tunnel(img, W / 2, H / 2, t, 2.0 * t + 6.0 * ease_in(u, 3.0), 0.25 + 0.9 * ease_in(u, 2.0),
                         1.0 + 2.0 * u)
    _WS.draw(img, 4.0 * t + 24.0 * ease_in(u, 3.0), 4.0 + 60.0 * ease_in(u, 2.0), cx=W / 2, cy=H / 2, f=820.0,
             amp=1.3, streak=2.6, tint=(0.75, 1.0, 0.95))
    img = fx.zoom_blur(img, W / 2, H / 2, amount=0.02 + 0.12 * ease_in(u, 2.0), n=6)
    add_glow(img, W / 2, H / 2, 40, (0.8, 1.0, 0.95), 0.4 + 2.5 * ease_in(u, 3.0))
    return img * (1.0 + 9.0 * ease_in((t - 1.85) / 0.4, 2.0))


# ============================================================================ S13 SPEED: warp exit
def s13(t):
    u = t / 1.5
    cam = Camera((0, 0, 0), (0, 0, 10), fov=60, roll=lerp(4, -2, u))
    img = sky(cam, 1.2, 0.8)
    global _WS
    if _WS is None:
        _WS = fx.StarField3D(n=2600, seed=131, depth=30.0)
    sp = 50.0 * math.exp(-t / 0.25) + 1.0
    _WS.draw(img, 40.0 + 6.0 * (1 - math.exp(-t / 0.25)) + t, sp, cx=W / 2, cy=H / 2, f=820.0, amp=1.2, streak=2.0,
             tint=(0.8, 1.0, 0.95))
    import logo as logo2d
    logo2d.shockwave(img, W / 2, H / 2, 40 + 1700 * t ** 0.6, 8 + 40 * t, 2.5 * math.exp(-t / 0.4))
    logo2d.shockwave(img, W / 2, H / 2, 20 + 900 * t ** 0.7, 4 + 20 * t, 1.5 * math.exp(-t / 0.3), (1.0, 0.95, 0.85))
    add_glow(img, W / 2, H / 2, 60, (0.9, 1.0, 0.95), 3.0 * math.exp(-t / 0.2))
    return img * (1.0 + 3.0 * math.exp(-t / 0.12))


# ============================================================================ S14 CONNECT: converge
@functools.lru_cache(maxsize=None)
def _conv():
    r = rng(141)
    n = 260
    return r.uniform(0, 2 * np.pi, n), r.uniform(0, 1, n), r.uniform(0.6, 1.4, n), r.uniform(0.3, 1.0, n)


def s14(t):
    u = t / 1.5
    cam = Camera((0, 0, 0), (0, 0, 10), fov=55, roll=lerp(-2, 2, u))
    img = sky(cam, 1.0, 0.7)
    a, d0, v, b = _conv()
    layer = np.zeros((H, W, 3), np.uint8)
    cx, cy = W / 2, H / 2
    for ai, di, vi, bi in zip(a, d0, v, b):
        k = (di - vi * t * 0.9) % 1.0
        r1 = 30 + k * 1500
        r0 = r1 + 160 + 300 * k
        x1, y1 = cx + math.cos(ai) * r1, cy + math.sin(ai) * r1 * 0.7
        x0, y0 = cx + math.cos(ai) * r0, cy + math.sin(ai) * r0 * 0.7
        g = int(255 * bi * min(1.0, k * 3))
        cv2.line(layer, (int(x0 * 16), int(y0 * 16)), (int(x1 * 16), int(y1 * 16)), (g, g, g), 2, cv2.LINE_AA, shift=4)
    img += lines3d.layer_to_light(layer, (0.3, 1.0, 0.65), gain=1.0, glow=1.6)
    add_glow(img, cx, cy, 30, (0.75, 1.0, 0.85), 1.0 + 2.0 * u)
    return img * (1.0 + 2.5 * math.exp(-t / 0.12))


# ============================================================================ S15+ logo
LOCK_T = 49.5
SHARDS = [  # start offsets per mark piece: translation (units), rotation axis, angle
    ((-38, 16, 26), (0.3, 1, 0.2), 2.2), ((34, -18, 20), (1, 0.2, 0.4), -1.9), ((-22, 30, -18), (0.2, 0.4, 1), 1.6),
    ((40, 22, -12), (1, 1, 0), -2.4), ((-34, -22, 14), (0.5, 0.2, 1), 2.0), ((18, -30, 30), (0.1, 1, 0.6), -1.7),
]


def logo_parts(T):
    """Per-part (R, T, scale) for the 6 mark shards + 9 wordmark shapes."""
    n = logo3d.N_PARTS
    out = []
    vis = []
    # 1) shards converge 47.25 -> 49.5
    p = clamp((T - 47.25) / (LOCK_T - 47.25))
    e = ease_in(p, 2.4)
    # 2) after 50.35 the mark slides left to make room for the wordmark
    k = ease_in_out((T - 50.35) / 0.8, 2.2)
    shift = -logo3d.LOCKUP_SHIFT * k
    for i in range(logo3d.N_MARK):
        tr, ax, ang = SHARDS[i]
        R = logo3d.rot_axis(ax, ang * (1 - e))
        Tt = np.array(tr, np.float64) * (1 - e) + shift
        out.append((R, Tt, 1.0))
        vis.append(smooth(p / 0.25) if T < LOCK_T else 1.0)
    wk = smooth((T - 50.75) / 0.7)
    for j in range(logo3d.N_PARTS - logo3d.N_MARK):
        R = logo3d.rot_axis((0, 1, 0), -0.9 * (1 - ease_out(wk, 3)))
        Tt = -logo3d.LOCKUP_SHIFT + np.array([12.0 * (1 - ease_out(wk, 3)), 0.0, 8.0 * (1 - wk)])
        out.append((R, Tt, 1.0))
        vis.append(wk)
    return out, vis


def logo_rect(cam, parts, vis, pad=40):
    v, st, cn, bx = logo3d.tables()
    pts = []
    for k, (R, Tt, sc) in enumerate(parts):
        if vis[k] <= 0.001:
            continue
        x0, y0, x1, y1 = bx[k]
        for X in (x0, x1):
            for Y in (y0, y1):
                for Z in (-1.6, 1.6):
                    pts.append(sc * (R @ np.array([X, Y, Z])) + Tt)
    if not pts:
        return (0, 0, 1, 1)
    sx, sy, z = cam.project(np.array(pts))
    if (z <= 0).any():
        return (0, 0, W, H)
    return (int(max(0, sx.min() - pad)), int(max(0, sy.min() - pad)), int(min(W, sx.max() + pad)),
            int(min(H, sy.max() + pad)))


def s_logo(T):
    t = T - 47.25
    # camera: slow push, small orbit after the lock, settles frontal for the end card
    orbit = 9.0 * (1 - ease_in_out((T - 50.3) / 2.2)) if T >= LOCK_T else lerp(16.0, 9.0, ease_in_out(t / 2.25))
    dist = lerp(118.0, 96.0, ease_in_out(clamp(t / 3.0))) + 118.0 * ease_in_out((T - 50.35) / 0.9)
    k2 = ease_in_out((T - 53.25) / 1.2)
    dist += 26.0 * k2
    off = np.array([0.0, -13.5 * k2, 0.0])
    cam = Camera(sph(dist, orbit, lerp(6.0, 2.0, ease_in_out(clamp(t / 4.0)))) + off, off, fov=30)
    img = sky(cam, 0.8, 0.5)
    # eclipse corona backdrop (screen space)
    pre = ease_in(clamp(t / 2.25), 2.0)
    post = math.exp(-max(0.0, T - LOCK_T) / 0.6) if T >= LOCK_T else 0.0
    end_dim = 1.0 - 0.72 * smooth((T - 53.1) / 1.7)
    R0 = 250.0 + 40.0 * ease_in_out(clamp(t / 4.0)) + 60 * ease_in_out((T - 50.35) / 0.9)
    cor = np.zeros_like(img)
    calm = 1.0 - 0.35 * smooth((T - 50.0) / 1.2)
    scenes.render_corona(cor, W / 2, H / 2 - 6, R0, T * 0.6, (0.16 + 0.22 * pre + 0.55 * post) * end_dim * calm, -0.55)
    img = img * 0.6 + cor
    # mark (and wordmark)
    parts, vis = logo_parts(T)
    xfs = [logo3d.part_xf(R, Tt, s) for (R, Tt, s) in parts]
    sweep = (0.0, 0.0, 0.0, 0.0)
    ts = (T - 51.75) / 0.9
    if 0 <= ts <= 1:
        sweep = (math.cos(0.5), math.sin(0.5), lerp(-80.0, 80.0, ease_in_out(ts)), 1.0)
    glow_hit = 0.75 + 2.2 * math.exp(-max(0.0, T - LOCK_T) / 0.35) * (T >= LOCK_T)
    col, a = logo3d.render(cam, xfs, vis, gfx.lens_sky(), t=T, glow=glow_hit, sweep=sweep,
                           rect=logo_rect(cam, parts, vis))
    img = img * (1 - a[..., None]) + col
    if T >= LOCK_T:
        tt = T - LOCK_T
        import logo as logo2d
        logo2d.shockwave(img, W / 2, H / 2, 60 + 1500 * tt ** 0.7, 10 + 30 * tt, 1.4 * math.exp(-tt / 0.5))
    flash = 1.0 + (2.2 * math.exp(-(T - LOCK_T) / 0.16) if T >= LOCK_T else 0.0)
    return img * flash * (1.0 - smooth((T - 58.9) / 1.0))


# ============================================================================ timeline + post
SHOTS = [
    (0.00, 5.25, s01), (5.25, 10.50, s02), (10.50, 15.00, s03), (15.00, 20.25, s04), (20.25, 24.00, s05),
    (24.00, 28.50, s06), (28.50, 31.50, s07), (31.50, 34.50, s08), (34.50, 37.50, s09), (37.50, 39.75, s10),
    (39.75, 42.00, s11), (42.00, 44.25, s12), (44.25, 45.75, s13), (45.75, 47.25, s14),
]
SUPER_POS = {
    "RWA · REAL-WORLD ASSETS": (0.5, 0.82), "SPOT TRADING · BTC/USDT": (0.5, 0.86),
    "7×24 · GLOBAL SERVICE": (0.5, 0.84), "TRADE ON THE APP": (0.27, 0.22), "ASSET SECURITY": (0.5, 0.88),
    "SPEED": (0.5, 0.5), "CONNECT": (0.5, 0.80), "GLOBAL": (0.5, 0.86),
}
HITS = [(44.25, 7.0), (45.75, 7.0), (47.25, 7.0), (49.5, 10.0)]


def picture(T):
    if T >= 47.25:
        return s_logo(T)
    for a, b, fn in SHOTS:
        if a <= T < b:
            return fn(T - a)
    return np.zeros((H, W, 3), np.float32)


def impact_shake(lin, T):
    dx = dy = 0.0
    zoom = 1.0
    for th, amp in HITS:
        u = T - th
        if 0 <= u < 0.45:
            a = amp * math.exp(-u / 0.12)
            dx += a * math.sin(2 * math.pi * 17 * u + th)
            dy += a * 0.7 * math.cos(2 * math.pi * 13 * u + 2 * th)
            zoom += 0.012 * math.exp(-u / 0.15)
    if zoom == 1.0:
        return lin
    M = np.array([[zoom, 0, (1 - zoom) * W / 2 + dx], [0, zoom, (1 - zoom) * H / 2 + dy]], np.float32)
    return cv2.warpAffine(lin, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def endcard(disp, T):
    ec = EDL["endcard"]
    k1 = smooth((T - 53.75) / 0.9)
    k2 = smooth((T - 54.35) / 0.9)
    k3 = smooth((T - 55.3) / 0.8)
    fade = 1.0 - smooth((T - 58.9) / 1.0)
    y0 = H / 2 + 92
    if k1 > 0:
        a = typo.text_alpha(ec["slogan_en"].upper(), "en", 25, 320, round(0.26 + 0.04 * k1, 3))
        typo.place(disp, a, W / 2, y0 + 20, color=(0.92, 0.97, 0.95), opacity=k1 * fade, glow=0.15,
                   glow_color=(0.4, 1.0, 0.7))
    if k2 > 0:
        a = typo.text_alpha(ec["slogan_cn"], "serif", 27, 400, 0.42)
        typo.place(disp, a, W / 2, y0 + 64, color=(0.80, 0.92, 0.88), opacity=k2 * fade)
    if k3 > 0:
        a = typo.text_alpha(ec["url"], "en", 30, 500, 0.22)
        typo.place(disp, a, W / 2, y0 + 132, color=(0.36, 1.0, 0.70), opacity=k3 * fade, glow=0.35,
                   glow_color=(0.2, 1.0, 0.6))
        a = typo.text_alpha(ec["tag_en"].upper() + "   ·   " + ec["tag_cn"], "sans", 19, 380, 0.2)
        typo.place(disp, a, W / 2, y0 + 176, color=(0.62, 0.70, 0.68), opacity=k3 * fade * 0.9)


def render_frame(T, rng_=None):
    if rng_ is None:
        rng_ = np.random.default_rng(int(T * FPS) + 7)
    lin = impact_shake(picture(T), T)
    disp = fx.grade(lin, bloom_amt=0.34)
    for s in EDL["supers"]:
        px, py = SUPER_POS.get(s["en"], (0.5, 0.8))
        typo.title_card(disp, T, s["en"], s["cn"], s["start"], s["end"], cx=W * px, cy=H * py, big=bool(s.get("big")))
    endcard(disp, T)
    disp = fx.grain(disp, rng_, amt=0.018)
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
