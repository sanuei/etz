"""V2 render engine: true 3D, CPU (Numba). No generated images: every pixel is
computed from scenes defined in code (plus NASA Earth textures for the planet).

Conventions: right-handed, +y up. Linear HDR float32 RGB images (H, W, 3).
"""
import math
import os

import cv2
import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")
W, H = 1920, 804


# ============================================================================ camera
class Camera:
    def __init__(self, pos, target, fov=40.0, up=(0.0, 1.0, 0.0), roll=0.0, w=W, h=H):
        self.pos = np.asarray(pos, np.float64)
        f = np.asarray(target, np.float64) - self.pos
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(up, np.float64))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        if roll:
            a = math.radians(roll)
            r, u = r * math.cos(a) + u * math.sin(a), -r * math.sin(a) + u * math.cos(a)
        self.f, self.r, self.u = f, r, u
        self.fov = fov
        self.w, self.h = w, h
        self.th = math.tan(math.radians(fov) / 2)
        self.focal = (h / 2) / self.th

    def basis(self):
        return np.stack([self.r, self.u, self.f]).astype(np.float64)

    def project(self, pts):
        """World points (N,3) -> screen x, y, depth (N,) ; depth<=0 behind."""
        d = np.asarray(pts, np.float64) - self.pos
        z = d @ self.f
        x = d @ self.r
        y = d @ self.u
        zs = np.where(z > 1e-6, z, 1e-6)
        sx = self.w / 2 + x / zs * self.focal
        sy = self.h / 2 - y / zs * self.focal
        return sx, sy, z


# ============================================================================ noise
@njit(inline="always", fastmath=True)
def _hash3(ix, iy, iz):
    h = (ix * 374761393 + iy * 668265263 + iz * 1274126177) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) * (1.0 / 16777215.0)


@njit(inline="always", fastmath=True)
def vnoise3(x, y, z):
    ix = math.floor(x)
    iy = math.floor(y)
    iz = math.floor(z)
    fx = x - ix
    fy = y - iy
    fz = z - iz
    ix = int(ix)
    iy = int(iy)
    iz = int(iz)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    uz = fz * fz * (3.0 - 2.0 * fz)
    a = _hash3(ix, iy, iz)
    b = _hash3(ix + 1, iy, iz)
    c = _hash3(ix, iy + 1, iz)
    d = _hash3(ix + 1, iy + 1, iz)
    e = _hash3(ix, iy, iz + 1)
    f = _hash3(ix + 1, iy, iz + 1)
    g = _hash3(ix, iy + 1, iz + 1)
    h = _hash3(ix + 1, iy + 1, iz + 1)
    k0 = a + (b - a) * ux
    k1 = c + (d - c) * ux
    k2 = e + (f - e) * ux
    k3 = g + (h - g) * ux
    l0 = k0 + (k1 - k0) * uy
    l1 = k2 + (k3 - k2) * uy
    return l0 + (l1 - l0) * uz


@njit(inline="always", fastmath=True)
def fbm3(x, y, z, octaves):
    s = 0.0
    a = 0.5
    for _ in range(octaves):
        s += a * vnoise3(x, y, z)
        x = x * 2.03 + 17.1
        y = y * 2.03 + 3.7
        z = z * 2.03 + 11.3
        a *= 0.5
    return s


# ============================================================================ sky
@njit(inline="always", fastmath=True)
def sample_equirect(img, dx, dy, dz):
    h, w = img.shape[0], img.shape[1]
    th = math.atan2(dz, dx)
    ph = math.asin(max(-1.0, min(1.0, dy)))
    u = (th * (0.5 / math.pi) + 0.5) * w - 0.5
    v = (0.5 - ph / math.pi) * h - 0.5
    x0 = int(math.floor(u))
    y0 = int(math.floor(v))
    fx = u - x0
    fy = v - y0
    x0 = x0 % w
    x1 = (x0 + 1) % w
    y0 = min(max(y0, 0), h - 1)
    y1 = min(y0 + 1, h - 1)
    r = (img[y0, x0, 0] * (1 - fx) + img[y0, x1, 0] * fx) * (1 - fy) + (img[y1, x0, 0] * (1 - fx) + img[y1, x1, 0] * fx) * fy
    g = (img[y0, x0, 1] * (1 - fx) + img[y0, x1, 1] * fx) * (1 - fy) + (img[y1, x0, 1] * (1 - fx) + img[y1, x1, 1] * fx) * fy
    b = (img[y0, x0, 2] * (1 - fx) + img[y0, x1, 2] * fx) * (1 - fy) + (img[y1, x0, 2] * (1 - fx) + img[y1, x1, 2] * fx) * fy
    return r, g, b


GAL_N = np.array([0.32, 0.86, -0.40])
GAL_N = GAL_N / np.linalg.norm(GAL_N)


@njit(parallel=True, fastmath=True, cache=True)
def _sky_haze(out, gnx, gny, gnz):
    h, w = out.shape[0], out.shape[1]
    for y in prange(h):
        ph = (0.5 - (y + 0.5) / h) * math.pi
        for x in range(w):
            th = ((x + 0.5) / w - 0.5) * 2 * math.pi
            dx = math.cos(ph) * math.cos(th)
            dy = math.sin(ph)
            dz = math.cos(ph) * math.sin(th)
            lat = dx * gnx + dy * gny + dz * gnz           # distance from galactic plane
            band = math.exp(-(lat * lat) / 0.018)
            n = fbm3(dx * 3.0 + 5, dy * 3.0, dz * 3.0, 6)
            n2 = fbm3(dx * 7.0, dy * 7.0 + 9, dz * 7.0, 5)
            dust = max(0.0, n2 - 0.52) * 3.0
            core = math.exp(-((dx - 0.2) ** 2 + (dy + 0.1) ** 2 + (dz - 0.95) ** 2) / 0.6)
            v = band * (0.35 + 0.9 * n * n) * (1.0 - 0.85 * min(1.0, dust * band * 2.0))
            v += 0.02 * n
            neb = max(0.0, fbm3(dx * 2.2 + 30, dy * 2.2, dz * 2.2 - 4, 5) - 0.55) * 2.2
            out[y, x, 0] = 0.010 * v * (1.0 + 0.6 * core) + 0.012 * neb * 0.3
            out[y, x, 1] = 0.011 * v * (1.0 + 0.4 * core) + 0.012 * neb * 1.0
            out[y, x, 2] = 0.014 * v + 0.012 * neb * 0.75


def star_catalog(n=70000, seed=11):
    """Random stars, denser along the galactic band. Returns dirs (n,3), flux, color."""
    r = np.random.default_rng(seed)
    v = r.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    k = int(n * 0.45)
    lat = r.normal(0, 0.12, k)
    a = r.uniform(0, 2 * np.pi, k)
    e1 = np.cross(GAL_N, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(GAL_N, e1)
    band = (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2) * np.cos(lat)[:, None] + GAL_N * np.sin(lat)[:, None]
    v[:k] = band
    flux = (r.pareto(1.6, n) + 1.0) ** 1.0 * 0.004
    flux = np.minimum(flux, 2.5)
    t = r.uniform(0, 1, n)
    col = np.stack([0.75 + 0.3 * t, 0.85 + 0.12 * t, 1.1 - 0.35 * t], 1)
    return v.astype(np.float64), flux.astype(np.float64), col.astype(np.float64)


@njit(fastmath=True, cache=True)
def splat_points(img, sx, sy, flux, col, sigma_base):
    """Additive gaussian splats (serial: no write races)."""
    h, w = img.shape[0], img.shape[1]
    for i in range(sx.shape[0]):
        x = sx[i]
        y = sy[i]
        if x < -4 or y < -4 or x > w + 4 or y > h + 4:
            continue
        s = sigma_base * (1.0 + 0.35 * math.sqrt(min(flux[i], 4.0)))
        rad = int(math.ceil(s * 3.0))
        norm = flux[i] / (2 * math.pi * s * s)
        cx = int(x)
        cy = int(y)
        for yy in range(cy - rad, cy + rad + 1):
            if yy < 0 or yy >= h:
                continue
            dy = yy + 0.5 - y
            for xx in range(cx - rad, cx + rad + 1):
                if xx < 0 or xx >= w:
                    continue
                dx = xx + 0.5 - x
                g = norm * math.exp(-(dx * dx + dy * dy) / (2 * s * s))
                img[yy, xx, 0] += g * col[i, 0]
                img[yy, xx, 1] += g * col[i, 1]
                img[yy, xx, 2] += g * col[i, 2]


@njit(fastmath=True, cache=True)
def splat_streaks(img, x0, y0, x1, y1, flux, col, sigma, trail=0.0):
    """Motion-blurred particles: flux spread along segment (x0,y0)->(x1,y1).
    trail in [0,1]: 0 = energy-conserving blur, 1 = full-brightness light trail."""
    h, w = img.shape[0], img.shape[1]
    for i in range(x0.shape[0]):
        L = math.hypot(x1[i] - x0[i], y1[i] - y0[i])
        n = max(1, int(L / max(sigma * 0.8, 0.5)) + 1)
        f = flux[i] / n ** (1.0 - trail)
        s = sigma
        rad = int(math.ceil(s * 2.5))
        norm = f / (2 * math.pi * s * s)
        for k in range(n):
            t = (k + 0.5) / n
            x = x0[i] + (x1[i] - x0[i]) * t
            y = y0[i] + (y1[i] - y0[i]) * t
            if x < -4 or y < -4 or x > w + 4 or y > h + 4:
                continue
            cx = int(x)
            cy = int(y)
            for yy in range(cy - rad, cy + rad + 1):
                if yy < 0 or yy >= h:
                    continue
                dy = yy + 0.5 - y
                for xx in range(cx - rad, cx + rad + 1):
                    if xx < 0 or xx >= w:
                        continue
                    dx = xx + 0.5 - x
                    g = norm * math.exp(-(dx * dx + dy * dy) / (2 * s * s))
                    img[yy, xx, 0] += g * col[i, 0]
                    img[yy, xx, 1] += g * col[i, 1]
                    img[yy, xx, 2] += g * col[i, 2]


@njit(parallel=True, fastmath=True, cache=True)
def sky_background(out, basis, th, haze):
    h, w = out.shape[0], out.shape[1]
    asp = w / h
    for y in prange(h):
        py = (1.0 - 2.0 * (y + 0.5) / h) * th
        for x in range(w):
            px = (2.0 * (x + 0.5) / w - 1.0) * th * asp
            dx = basis[2, 0] + px * basis[0, 0] + py * basis[1, 0]
            dy = basis[2, 1] + px * basis[0, 1] + py * basis[1, 1]
            dz = basis[2, 2] + px * basis[0, 2] + py * basis[1, 2]
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            r, g, b = sample_equirect(haze, dx / n, dy / n, dz / n)
            out[y, x, 0] += r
            out[y, x, 1] += g
            out[y, x, 2] += b


_SKY = {}


def sky_maps():
    """(haze equirect 2048x1024, lens map with stars 6144x3072) cached per process."""
    if "haze" not in _SKY:
        cache = os.path.join(HERE, "..", "build", "v2cache")
        os.makedirs(cache, exist_ok=True)
        p = os.path.join(cache, "haze.npy")
        if os.path.exists(p):
            haze = np.load(p)
        else:
            haze = np.zeros((1024, 2048, 3), np.float32)
            _sky_haze(haze, *GAL_N)
            np.save(p, haze)
        _SKY["haze"] = haze
        _SKY["stars"] = star_catalog()
    return _SKY["haze"], _SKY["stars"]


def lens_sky():
    """High-res equirect sky including stars, for gravitationally lensed rays."""
    if "lens" not in _SKY:
        cache = os.path.join(HERE, "..", "build", "v2cache", "lens_sky.npy")
        if os.path.exists(cache):
            _SKY["lens"] = np.load(cache, mmap_mode="r")
        else:
            haze, (d, f, c) = sky_maps()
            Wl, Hl = 6144, 3072
            big = cv2.resize(haze, (Wl, Hl), interpolation=cv2.INTER_CUBIC)
            th = np.arctan2(d[:, 2], d[:, 0])
            ph = np.arcsin(np.clip(d[:, 1], -1, 1))
            sx = (th / (2 * np.pi) + 0.5) * Wl
            sy = (0.5 - ph / np.pi) * Hl
            splat_points(big, sx, sy, f * 1.6, c, 0.75)
            np.save(cache, big.astype(np.float32))
            _SKY["lens"] = np.load(cache, mmap_mode="r")
    return _SKY["lens"]


def render_sky(cam, img=None, star_gain=1.0, haze_gain=1.0, star_sigma=0.7):
    haze, (d, f, c) = sky_maps()
    if img is None:
        img = np.zeros((cam.h, cam.w, 3), np.float32)
    if haze_gain > 0:
        tmp = np.zeros_like(img)
        sky_background(tmp, cam.basis(), cam.th, haze)
        img += tmp * haze_gain
    pts = cam.pos + d * 1e6
    sx, sy, z = cam.project(pts)
    m = (z > 0) & (sx > -5) & (sx < cam.w + 5) & (sy > -5) & (sy < cam.h + 5)
    splat_points(img, sx[m], sy[m], f[m] * star_gain, c[m], star_sigma)
    return img
