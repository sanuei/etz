"""More V2 renderers: galaxy/big-bang particles, volumetric nebula, warp tunnel,
eclipse corona, hexagonal energy shield, floating glass phone."""
import math

import cv2
import numpy as np
from numba import njit, prange

from gfx import fbm3, sample_equirect, splat_streaks, vnoise3, W, H


# ============================================================================ galaxy / big bang
def galaxy(n=700000, seed=5):
    r = np.random.default_rng(seed)
    nb = int(n * 0.16)
    nd = n - nb
    # disk: exponential radius, two log-spiral arms + inter-arm scatter
    rad = r.exponential(0.32, nd) + 0.04
    rad = np.where(rad > 1.6, r.uniform(0.04, 1.6, nd), rad)
    arm = r.integers(0, 2, nd) * np.pi
    pitch = math.radians(14)
    theta = arm + np.log(rad / 0.04) / math.tan(pitch) + r.normal(0, 0.28 + 0.25 * (r.uniform(0, 1, nd) > 0.6), nd)
    y = r.normal(0, 0.018 + 0.02 * np.exp(-rad / 0.2), nd)
    disk = np.stack([np.cos(theta) * rad, y, np.sin(theta) * rad], 1)
    # bulge
    bul = r.normal(0, 1, (nb, 3)) * np.array([0.12, 0.08, 0.12])
    pos = np.concatenate([disk, bul])
    # colours: warm core, blue-white arms, emerald star-forming knots
    col = np.empty((n, 3))
    t = r.uniform(0, 1, nd)
    col[:nd] = np.stack([0.62 + 0.3 * t, 0.78 + 0.15 * t, 1.0 + 0.1 * t], 1)
    knots = r.uniform(0, 1, nd) < 0.07
    col[:nd][knots] = np.array([0.35, 1.25, 0.75])
    col[nd:] = np.array([1.05, 0.82, 0.55])
    flux = np.concatenate([r.uniform(0.2, 1.0, nd) ** 2, r.uniform(0.4, 1.0, nb)])
    flux[:nd][knots] *= 2.2
    return pos, col, flux


def bigbang_dirs(n, seed=6):
    r = np.random.default_rng(seed)
    d = r.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    sp = r.uniform(0.3, 1.0, n) ** 0.5
    return d, sp


def splat_world(img, cam, p0, p1, col, flux, sigma=0.9, z_ref=1.0, trail=0.0):
    """Project world points (motion from p0 to p1 during the shutter) and splat.
    Brightness falls off as (z_ref/z)^2 beyond z_ref."""
    sx1, sy1, z1 = cam.project(p1)
    sx0, sy0, z0 = cam.project(p0)
    m = (z1 > 0.05) & (z0 > 0.05) & (sx1 > -50) & (sx1 < cam.w + 50) & (sy1 > -50) & (sy1 < cam.h + 50)
    if not m.any():
        return
    f = flux[m] * np.minimum(1.0, (z_ref / np.maximum(z1[m], 0.05)) ** 2)
    splat_streaks(img, sx0[m], sy0[m], sx1[m], sy1[m], f, col[m], sigma, trail)


# ============================================================================ volumetric nebula
@njit(parallel=True, fastmath=True, cache=True)
def render_nebula(out, cam_pos, basis, th, t, steps, depth, density, tint):
    h, w = out.shape[0], out.shape[1]
    asp = w / h
    for y in prange(h):
        py0 = (1.0 - 2.0 * (y + 0.5) / h) * th
        for x in range(w):
            px0 = (2.0 * (x + 0.5) / w - 1.0) * th * asp
            dx = basis[2, 0] + px0 * basis[0, 0] + py0 * basis[1, 0]
            dy = basis[2, 1] + px0 * basis[0, 1] + py0 * basis[1, 1]
            dz = basis[2, 2] + px0 * basis[0, 2] + py0 * basis[1, 2]
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= n
            dy /= n
            dz /= n
            jit = (math.sin((x * 12.9898 + y * 78.233)) * 43758.5453) % 1.0
            ds = depth / steps
            tr = 1.0
            cr = 0.0
            cg = 0.0
            cb = 0.0
            for i in range(steps):
                s = (i + jit) * ds * (1.0 + i * 0.03)
                px = cam_pos[0] + dx * s
                py = cam_pos[1] + dy * s
                pz = cam_pos[2] + dz * s
                # domain-warped fbm, carved into a winding canyon around the flight path
                wx = fbm3(px * 0.35, py * 0.35, pz * 0.35 + t * 0.03, 3)
                wy = fbm3(px * 0.35 + 5.2, py * 0.35 + 1.3, pz * 0.35, 3)
                q = fbm3(px * 0.6 + wx * 2.0, py * 0.6 + wy * 2.0, pz * 0.6 + t * 0.05, 5)
                cx = math.sin(pz * 0.08) * 3.0
                cyy = math.cos(pz * 0.06) * 1.5
                tube = math.sqrt((px - cx) ** 2 + ((py - cyy) * 1.4) ** 2)
                shell = min(1.0, max(0.0, (tube - 1.2) / 2.5))
                qq = min(1.0, max(0.0, (q - 0.47) / 0.28))
                dens = qq * qq * (3.0 - 2.0 * qq) * density * shell
                if dens > 0.0:
                    hot = min(1.0, dens * 1.4)
                    gold = max(0.0, fbm3(px * 0.22 + 40.0, py * 0.22, pz * 0.22, 2) - 0.55) * 4.0
                    gold = min(1.0, gold)
                    # deep teal -> emerald -> mint-white, with occasional gold filaments
                    er = 0.03 + 0.20 * hot + 0.70 * hot ** 4
                    eg = 0.28 + 0.75 * hot + 0.10 * hot ** 4
                    eb = 0.38 + 0.12 * hot + 0.45 * hot ** 4
                    er = er * (1 - gold) + (0.95 * hot + 0.1) * gold
                    eg = eg * (1 - gold) + (0.62 * hot + 0.06) * gold
                    eb = eb * (1 - gold) + (0.22 * hot + 0.02) * gold
                    lum = dens * (0.5 + 0.9 * fbm3(px * 1.7, py * 1.7, pz * 1.7, 2)) * tint
                    a = 1.0 - math.exp(-dens * ds * 1.1)
                    cr += tr * lum * er * ds
                    cg += tr * lum * eg * ds
                    cb += tr * lum * eb * ds
                    tr *= 1.0 - a
                    if tr < 0.02:
                        break
            out[y, x, 0] = cr
            out[y, x, 1] = cg
            out[y, x, 2] = cb


def nebula(cam, t, scale=0.5, steps=56, depth=34.0, density=1.0, tint=0.30):
    w, h = int(cam.w * scale), int(cam.h * scale)
    out = np.zeros((h, w, 3), np.float32)
    render_nebula(out, cam.pos.astype(np.float64), cam.basis(), cam.th, float(t), int(steps), float(depth),
                  float(density), float(tint))
    return cv2.resize(out, (cam.w, cam.h), interpolation=cv2.INTER_CUBIC).clip(0, None)


# ============================================================================ warp tunnel
@njit(parallel=True, fastmath=True, cache=True)
def render_tunnel(out, cx, cy, t, speed, gain, twist):
    h, w = out.shape[0], out.shape[1]
    for y in prange(h):
        for x in range(w):
            ux = (x + 0.5 - cx) / h
            uy = (y + 0.5 - cy) / h
            rho = math.sqrt(ux * ux + uy * uy) + 1e-4
            ang = math.atan2(uy, ux)
            z = 0.35 / rho + speed
            a = ang + twist * z * 0.05
            n = fbm3(math.cos(a) * 4.0, math.sin(a) * 4.0, z * 1.4, 4)
            streak = max(0.0, n - 0.48) * 4.0
            streak = streak * streak
            fade = min(1.0, rho * 2.5) * math.exp(-rho * 0.8)
            core = math.exp(-rho * rho / 0.004) * 3.0
            v = gain * (streak * fade + core)
            out[y, x, 0] += v * 0.45
            out[y, x, 1] += v * 1.00
            out[y, x, 2] += v * 0.85


# ============================================================================ eclipse corona
@njit(parallel=True, fastmath=True, cache=True)
def render_corona(out, cx, cy, R0, t, gain, hot_angle):
    h, w = out.shape[0], out.shape[1]
    for y in prange(h):
        for x in range(w):
            dx = x + 0.5 - cx
            dy = y + 0.5 - cy
            r = math.sqrt(dx * dx + dy * dy)
            if r < R0:
                continue
            ang = math.atan2(dy, dx)
            ca = math.cos(ang)
            sa = math.sin(ang)
            lr = math.log(r / R0)
            streamer = fbm3(ca * 3.0, sa * 3.0, lr * 1.2 - t * 0.05, 5)
            fine = fbm3(ca * 22.0, sa * 22.0, lr * 4.0 - t * 0.12, 3)
            s = max(0.0, streamer - 0.28) * 2.2 * (0.6 + 0.8 * fine)
            fall = (R0 / r) ** 2.2
            rim = math.exp(-(r - R0) / (R0 * 0.012)) * 2.5
            da = ang - hot_angle
            da = math.atan2(math.sin(da), math.cos(da))
            hot = math.exp(-(da / 0.35) ** 2) * math.exp(-(r - R0) / (R0 * 0.08)) * 3.0
            v = gain * (s * fall * 1.6 + rim * (0.4 + 0.6 * fine) + hot)
            white = min(1.0, math.exp(-(r - R0) / (R0 * 0.05)) + hot * 0.3)
            out[y, x, 0] += v * (0.07 + 0.85 * white)
            out[y, x, 1] += v * (0.90 + 0.10 * white)
            out[y, x, 2] += v * (0.42 + 0.48 * white)


# ============================================================================ hex energy shield
@njit(inline="always", fastmath=True)
def _cell(px, py, pz):
    """3D cellular noise: (F1, F2) distances."""
    ix = math.floor(px)
    iy = math.floor(py)
    iz = math.floor(pz)
    f1 = 9.0
    f2 = 9.0
    for a in range(-1, 2):
        for b in range(-1, 2):
            for c in range(-1, 2):
                gx = ix + a
                gy = iy + b
                gz = iz + c
                h1 = math.sin(gx * 127.1 + gy * 311.7 + gz * 74.7) * 43758.5453
                h2 = math.sin(gx * 269.5 + gy * 183.3 + gz * 246.1) * 43758.5453
                h3 = math.sin(gx * 113.5 + gy * 271.9 + gz * 124.6) * 43758.5453
                fx = gx + (h1 - math.floor(h1)) * 0.8 + 0.1 - px
                fy = gy + (h2 - math.floor(h2)) * 0.8 + 0.1 - py
                fz = gz + (h3 - math.floor(h3)) * 0.8 + 0.1 - pz
                d = fx * fx + fy * fy + fz * fz
                if d < f1:
                    f2 = f1
                    f1 = d
                elif d < f2:
                    f2 = d
    return math.sqrt(f1), math.sqrt(f2)


@njit(parallel=True, fastmath=True, cache=True)
def render_shield(out, cam_pos, basis, th, R, t, impacts, gain, form):
    """Transparent sphere of radius R at the origin with glowing cell edges.
    impacts: (k,4) = dir xyz, start time. form: 0..1 build-up sweep."""
    h, w = out.shape[0], out.shape[1]
    asp = w / h
    for y in prange(h):
        py0 = (1.0 - 2.0 * (y + 0.5) / h) * th
        for x in range(w):
            px0 = (2.0 * (x + 0.5) / w - 1.0) * th * asp
            dx = basis[2, 0] + px0 * basis[0, 0] + py0 * basis[1, 0]
            dy = basis[2, 1] + px0 * basis[0, 1] + py0 * basis[1, 1]
            dz = basis[2, 2] + px0 * basis[0, 2] + py0 * basis[1, 2]
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= n
            dy /= n
            dz /= n
            ox, oy, oz = cam_pos[0], cam_pos[1], cam_pos[2]
            b = ox * dx + oy * dy + oz * dz
            c = ox * ox + oy * oy + oz * oz - R * R
            disc = b * b - c
            if disc <= 0.0:
                continue
            sq = math.sqrt(disc)
            acc_r = 0.0
            acc_g = 0.0
            acc_b = 0.0
            for side in range(2):
                tt = -b - sq if side == 0 else -b + sq
                if tt <= 0.0:
                    continue
                hx = (ox + dx * tt) / R
                hy = (oy + dy * tt) / R
                hz = (oz + dz * tt) / R
                ndv = abs(hx * dx + hy * dy + hz * dz)
                fres = (1.0 - ndv) ** 3
                f1, f2 = _cell(hx * 5.0, hy * 5.0, hz * 5.0)
                edge = math.exp(-(f2 - f1) / 0.045)
                # build-up: cells light up from the "south pole" upward
                lat = (hy + 1.0) * 0.5
                built = min(1.0, max(0.0, (form * 1.25 - lat) / 0.15))
                ripple = 0.0
                for k in range(impacts.shape[0]):
                    age = t - impacts[k, 3]
                    if age < 0.0 or age > 1.6:
                        continue
                    cosd = hx * impacts[k, 0] + hy * impacts[k, 1] + hz * impacts[k, 2]
                    ang = math.acos(max(-1.0, min(1.0, cosd)))
                    front = age * 1.6
                    ripple += math.exp(-((ang - front) / 0.10) ** 2) * math.exp(-age * 1.6) * 3.0
                    ripple += math.exp(-(ang / 0.18) ** 2) * math.exp(-age * 5.0) * 6.0
                wgt = 1.0 if side == 0 else 0.45
                v = wgt * built * (edge * (0.35 + 1.4 * fres) + 0.06 * fres + ripple * (0.25 + edge))
                acc_r += v * 0.25
                acc_g += v * 1.00
                acc_b += v * 0.70
            out[y, x, 0] += acc_r * gain
            out[y, x, 1] += acc_g * gain
            out[y, x, 2] += acc_b * gain


# ============================================================================ glass phone
@njit(inline="always", fastmath=True)
def _sd_rbox(px, py, pz, bx, by, bz, r):
    qx = abs(px) - bx + r
    qy = abs(py) - by + r
    qz = abs(pz) - bz + r
    ox = max(qx, 0.0)
    oy = max(qy, 0.0)
    oz = max(qz, 0.0)
    return math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(qx, max(qy, qz)), 0.0) - r


@njit(parallel=True, fastmath=True, cache=True)
def render_phone(out, alpha, cam_pos, basis, th, inv, screen, sky, rect, glow):
    """Phone body: rounded box 0.75 x 1.55 x 0.085 (half extents in inv-local space)."""
    h, w = out.shape[0], out.shape[1]
    asp = w / h
    BX, BY, BZ, RR = 0.75, 1.55, 0.085, 0.16
    sh, sw = screen.shape[0], screen.shape[1]
    for y in prange(rect[1], rect[3]):
        py0 = (1.0 - 2.0 * (y + 0.5) / h) * th
        for x in range(rect[0], rect[2]):
            px0 = (2.0 * (x + 0.5) / w - 1.0) * th * asp
            dx = basis[2, 0] + px0 * basis[0, 0] + py0 * basis[1, 0]
            dy = basis[2, 1] + px0 * basis[0, 1] + py0 * basis[1, 1]
            dz = basis[2, 2] + px0 * basis[0, 2] + py0 * basis[1, 2]
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            # into phone-local space
            ox = inv[0, 0] * cam_pos[0] + inv[0, 1] * cam_pos[1] + inv[0, 2] * cam_pos[2] + inv[0, 3]
            oy = inv[1, 0] * cam_pos[0] + inv[1, 1] * cam_pos[1] + inv[1, 2] * cam_pos[2] + inv[1, 3]
            oz = inv[2, 0] * cam_pos[0] + inv[2, 1] * cam_pos[1] + inv[2, 2] * cam_pos[2] + inv[2, 3]
            ldx = (inv[0, 0] * dx + inv[0, 1] * dy + inv[0, 2] * dz) / n
            ldy = (inv[1, 0] * dx + inv[1, 1] * dy + inv[1, 2] * dz) / n
            ldz = (inv[2, 0] * dx + inv[2, 1] * dy + inv[2, 2] * dz) / n
            tt = 0.0
            hit = False
            dmin = 1e9
            for i in range(90):
                d = _sd_rbox(ox + ldx * tt, oy + ldy * tt, oz + ldz * tt, BX, BY, BZ, RR)
                dmin = min(dmin, d)
                if d < 0.0008:
                    hit = True
                    break
                tt += d
                if tt > 60.0:
                    break
            if not hit:
                g = glow * math.exp(-dmin / 0.08) * 0.25
                out[y, x, 1] += g
                out[y, x, 0] += g * 0.2
                out[y, x, 2] += g * 0.6
                continue
            hx = ox + ldx * tt
            hy = oy + ldy * tt
            hz = oz + ldz * tt
            e = 0.0007
            nx = _sd_rbox(hx + e, hy, hz, BX, BY, BZ, RR) - _sd_rbox(hx - e, hy, hz, BX, BY, BZ, RR)
            ny = _sd_rbox(hx, hy + e, hz, BX, BY, BZ, RR) - _sd_rbox(hx, hy - e, hz, BX, BY, BZ, RR)
            nz = _sd_rbox(hx, hy, hz + e, BX, BY, BZ, RR) - _sd_rbox(hx, hy, hz - e, BX, BY, BZ, RR)
            nn = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
            nx /= nn
            ny /= nn
            nz /= nn
            ndv = max(0.0, -(nx * ldx + ny * ldy + nz * ldz))
            fres = 0.06 + 0.94 * (1.0 - ndv) ** 5
            rx = ldx + 2 * ndv * nx
            ry = ldy + 2 * ndv * ny
            rz = ldz + 2 * ndv * nz
            sr, sg, sb = sample_equirect(sky, rx, ry, rz)
            soft = math.exp(-((ry - 0.5) / 0.15) ** 2) * math.exp(-((rx - 0.2) / 0.6) ** 2) * 2.5
            soft += math.exp(-((rx + 0.7) / 0.12) ** 2) * math.exp(-((ry) / 0.6) ** 2) * 1.6
            refl_r = sr * 3 + soft
            refl_g = sg * 3 + soft * 1.05
            refl_b = sb * 3 + soft * 1.1
            cr = 0.006 + fres * refl_r
            cg = 0.008 + fres * refl_g
            cb = 0.010 + fres * refl_b
            # metal frame highlight on the rounded sides
            side = 1.0 - min(1.0, abs(nz) * 1.2)
            cr += side * (0.05 + 0.6 * fres) * 0.6
            cg += side * (0.06 + 0.6 * fres) * 0.7
            cb += side * (0.06 + 0.6 * fres) * 0.7
            # screen (front face, local -z side faces the camera)
            if hz > 0.0 and abs(nz) > 0.9:
                u = (hx + BX * 0.92) / (2 * BX * 0.92)
                v = (BY * 0.95 - hy) / (2 * BY * 0.95)
                if u > 0.0 and u < 1.0 and v > 0.0 and v < 1.0:
                    ix = int(u * (sw - 1))
                    iy = int(v * (sh - 1))
                    cr += screen[iy, ix, 0]
                    cg += screen[iy, ix, 1]
                    cb += screen[iy, ix, 2]
            out[y, x, 0] = cr
            out[y, x, 1] = cg
            out[y, x, 2] = cb
            alpha[y, x] = 1.0
