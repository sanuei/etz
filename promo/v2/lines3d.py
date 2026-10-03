"""3D polylines/points drawn as anti-aliased emissive light (with occlusion by a
sphere), used for city arcs, orbital rings, constellations and the data lattice."""
import math

import cv2
import numpy as np

from gfx import splat_points


def _occluded(cam, pts, R=1.0):
    """True where the segment camera->point hits the sphere |x|<R before the point."""
    o = cam.pos
    d = pts - o
    L = np.linalg.norm(d, axis=1)
    dn = d / L[:, None]
    b = dn @ o
    c = o @ o - R * R
    disc = b * b - c
    t0 = -b - np.sqrt(np.maximum(disc, 0))
    return (disc > 0) & (t0 > 0) & (t0 < L - 1e-4)


def draw_polyline(layer, cam, pts, color, width=1, occlude_R=None, fade=None):
    """Draw a 3D polyline into an 8-bit RGB layer (AA). fade: per-point 0..1."""
    sx, sy, z = cam.project(pts)
    vis = z > 0.01
    if occlude_R is not None:
        vis &= ~_occluded(cam, pts, occlude_R)
    for i in range(len(pts) - 1):
        if not (vis[i] and vis[i + 1]):
            continue
        f = 1.0 if fade is None else 0.5 * (fade[i] + fade[i + 1])
        if f <= 0.01:
            continue
        c = tuple(int(min(255, v * f)) for v in color)
        p0 = (int(sx[i] * 16), int(sy[i] * 16))
        p1 = (int(sx[i + 1] * 16), int(sy[i + 1] * 16))
        if abs(p0[0]) > 1e7 or abs(p1[0]) > 1e7 or abs(p0[1]) > 1e7 or abs(p1[1]) > 1e7:
            continue
        cv2.line(layer, p0, p1, c, width, cv2.LINE_AA, shift=4)


def layer_to_light(layer, tint, gain=1.0, glow=2.0, glow_sigma=3.0):
    lin = np.power(layer.astype(np.float32) / 255.0, 2.2)
    out = lin * gain
    if glow > 0:
        out = out + cv2.GaussianBlur(lin, (0, 0), glow_sigma) * glow
        out = out + cv2.GaussianBlur(lin, (0, 0), glow_sigma * 4) * glow * 0.5
    return out * np.asarray(tint, np.float32)


def points_light(img, cam, pts, flux, color, sigma=1.0, occlude_R=None):
    sx, sy, z = cam.project(pts)
    vis = z > 0.01
    if occlude_R is not None:
        vis &= ~_occluded(cam, pts, occlude_R)
    if not vis.any():
        return
    col = np.broadcast_to(np.asarray(color, np.float64), (vis.sum(), 3)).copy()
    splat_points(img, sx[vis], sy[vis], np.asarray(flux, np.float64)[vis] if np.ndim(flux) else
                 np.full(vis.sum(), float(flux)), col, sigma)


def slerp_arc(a, b, n=80, lift=0.25, base=1.0):
    """Great-circle arc between surface points a, b raised by `lift` at the middle."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    om = math.acos(np.clip(a @ b, -1, 1))
    s = np.linspace(0, 1, n)
    if om < 1e-4:
        dirs = np.repeat(a[None], n, 0)
    else:
        dirs = (np.sin((1 - s) * om)[:, None] * a + np.sin(s * om)[:, None] * b) / math.sin(om)
    h = base + lift * om / math.pi * 2.0 * np.sin(np.pi * s) + 0.004
    return dirs * h[:, None]
