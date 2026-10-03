"""3D ETZ logo: extruded, bevelled polygon SDFs (official etz.com geometry),
sphere-traced on the CPU with glass/obsidian shading, emissive emerald edges,
studio reflections and a moving light sweep. Each part has its own transform
so the mark can assemble from flying shards."""
import math
import os
import sys

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
import logo as logo2d  # noqa: E402

from gfx import sample_equirect  # noqa: E402

# ---------------------------------------------------------------- geometry tables
_MARK_C = logo2d.MARK_CENTER.copy()


def _pack(polys, center):
    verts, starts, counts, boxes = [], [], [], []
    k = 0
    for p in polys:
        q = np.asarray(p, np.float64).copy()
        if np.allclose(q[0], q[-1]):
            q = q[:-1]
        q = q - center
        q[:, 1] *= -1.0  # svg y-down -> y-up
        verts.append(q)
        starts.append(k)
        counts.append(len(q))
        boxes.append([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()])
        k += len(q)
    return (np.concatenate(verts).astype(np.float64), np.array(starts, np.int64), np.array(counts, np.int64),
            np.array(boxes, np.float64))


def lockup_tables():
    """All parts of the horizontal lockup (6 mark shards + 9 wordmark shapes),
    centred on the mark centre (lockup offset handled by part transforms)."""
    polys = list(logo2d.MARK) + list(logo2d.WORD)
    return _pack(polys, _MARK_C)


N_MARK = len(logo2d.MARK)
N_PARTS = N_MARK + len(logo2d.WORD)
LOCKUP_SHIFT = np.array([(logo2d.LOCKUP_BOX[0] + logo2d.LOCKUP_BOX[2]) / 2 - _MARK_C[0], 0.0, 0.0])


@njit(inline="always", fastmath=True)
def _sd_poly(px, py, verts, s0, n):
    v0x = verts[s0, 0]
    v0y = verts[s0, 1]
    d = (px - v0x) ** 2 + (py - v0y) ** 2
    sg = 1.0
    j = s0 + n - 1
    for i in range(s0, s0 + n):
        vix = verts[i, 0]
        viy = verts[i, 1]
        ex = verts[j, 0] - vix
        ey = verts[j, 1] - viy
        wx = px - vix
        wy = py - viy
        ee = ex * ex + ey * ey
        t = 0.0
        if ee > 1e-12:
            t = min(1.0, max(0.0, (wx * ex + wy * ey) / ee))
        bx = wx - ex * t
        by = wy - ey * t
        dd = bx * bx + by * by
        if dd < d:
            d = dd
        c1 = py >= viy
        c2 = py < verts[j, 1]
        c3 = ex * wy > ey * wx
        if (c1 and c2 and c3) or ((not c1) and (not c2) and (not c3)):
            sg = -sg
        j = i
    return sg * math.sqrt(d)


@njit(inline="always", fastmath=True)
def _scene(x, y, z, verts, starts, counts, boxes, xf, vis, half, bev):
    """Distance to the union of visible parts. xf[k] = 3x4 inverse transform
    (world->part) with uniform scale stored in xf[k,3,0]. Returns (d, part, d2d)."""
    best = 1e9
    bp = -1
    bd2 = 0.0
    for k in range(starts.shape[0]):
        if vis[k] <= 0.001:
            continue
        sc = xf[k, 3, 0]
        lx = (xf[k, 0, 0] * x + xf[k, 0, 1] * y + xf[k, 0, 2] * z + xf[k, 0, 3])
        ly = (xf[k, 1, 0] * x + xf[k, 1, 1] * y + xf[k, 1, 2] * z + xf[k, 1, 3])
        lz = (xf[k, 2, 0] * x + xf[k, 2, 1] * y + xf[k, 2, 2] * z + xf[k, 2, 3])
        # cheap bound: distance to the part's padded bounding box
        bx = max(boxes[k, 0] - lx, 0.0, lx - boxes[k, 2])
        by = max(boxes[k, 1] - ly, 0.0, ly - boxes[k, 3])
        bz = max(abs(lz) - half, 0.0)
        lb = math.sqrt(bx * bx + by * by + bz * bz) / sc
        if lb > best:
            continue
        d2 = _sd_poly(lx, ly, verts, starts[k], counts[k])
        wx = d2 + bev
        wy = abs(lz) - half + bev
        d = (min(max(wx, wy), 0.0) + math.sqrt(max(wx, 0.0) ** 2 + max(wy, 0.0) ** 2) - bev) / sc
        if d < best:
            best = d
            bp = k
            bd2 = d2
    return best, bp, bd2


@njit(inline="always", fastmath=True)
def _env(dx, dy, dz, sky, t):
    """Studio-in-space environment for reflections."""
    sr, sg, sb = sample_equirect(sky, dx, dy, dz)
    r = sr * 3.0
    g = sg * 3.0
    b = sb * 3.0
    # two long softboxes above, one emerald kicker from below-left
    sb1 = math.exp(-((dy - 0.55) / 0.10) ** 2) * math.exp(-((dx - 0.1) / 0.9) ** 2)
    sb2 = math.exp(-((dy - 0.15) / 0.05) ** 2) * math.exp(-((dx + 0.75) / 0.25) ** 2)
    kick = math.exp(-((dy + 0.45) / 0.25) ** 2) * math.exp(-((dx + 0.6) / 0.5) ** 2)
    r += 2.4 * sb1 + 3.0 * sb2 + 0.10 * kick
    g += 2.5 * sb1 + 3.2 * sb2 + 1.40 * kick
    b += 2.6 * sb1 + 3.4 * sb2 + 0.80 * kick
    return r, g, b


@njit(parallel=True, fastmath=True, cache=True)
def render_logo(out, alpha, cam_pos, basis, th, verts, starts, counts, boxes, xf, vis, glow, half, bev,
                sky, t, sweep, rect, exposure):
    h, w = out.shape[0], out.shape[1]
    asp = w / h
    x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]
    for y in prange(y0, y1):
        py0 = (1.0 - 2.0 * (y + 0.5) / h) * th
        for x in range(x0, x1):
            px0 = (2.0 * (x + 0.5) / w - 1.0) * th * asp
            dx = basis[2, 0] + px0 * basis[0, 0] + py0 * basis[1, 0]
            dy = basis[2, 1] + px0 * basis[0, 1] + py0 * basis[1, 1]
            dz = basis[2, 2] + px0 * basis[0, 2] + py0 * basis[1, 2]
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= n
            dy /= n
            dz /= n
            ox, oy, oz = cam_pos[0], cam_pos[1], cam_pos[2]
            tt = 0.0
            hit = False
            part = -1
            d2 = 0.0
            dmin_seen = 1e9
            for i in range(110):
                d, part, d2 = _scene(ox + dx * tt, oy + dy * tt, oz + dz * tt, verts, starts, counts, boxes, xf, vis,
                                     half, bev)
                if d < dmin_seen:
                    dmin_seen = d
                if d < 0.0015 * tt:
                    hit = True
                    break
                tt += d * 0.9
                if tt > 400.0:
                    break
            if not hit:
                # soft emissive halo around the shards (cheap glow from nearest approach)
                g = glow * math.exp(-dmin_seen / 0.8)
                out[y, x, 0] += g * 0.10
                out[y, x, 1] += g * 0.55
                out[y, x, 2] += g * 0.32
                continue
            hx = ox + dx * tt
            hy = oy + dy * tt
            hz = oz + dz * tt
            e = 0.002 * tt + 0.0005
            a1, _, _ = _scene(hx + e, hy - e, hz - e, verts, starts, counts, boxes, xf, vis, half, bev)
            a2, _, _ = _scene(hx - e, hy - e, hz + e, verts, starts, counts, boxes, xf, vis, half, bev)
            a3, _, _ = _scene(hx - e, hy + e, hz - e, verts, starts, counts, boxes, xf, vis, half, bev)
            a4, _, _ = _scene(hx + e, hy + e, hz + e, verts, starts, counts, boxes, xf, vis, half, bev)
            nx = a1 - a2 - a3 + a4
            ny = -a1 - a2 + a3 + a4
            nz = -a1 + a2 - a3 + a4
            nn = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
            nx /= nn
            ny /= nn
            nz /= nn
            ndv = -(nx * dx + ny * dy + nz * dz)
            ndv = max(ndv, 0.0)
            fres = 0.10 + 0.90 * (1.0 - ndv) ** 5
            rx = dx + 2 * ndv * nx
            ry = dy + 2 * ndv * ny
            rz = dz + 2 * ndv * nz
            er, eg, eb = _env(rx, ry, rz, sky, t)
            # key light (upper right, warm white) + rim (behind, emerald)
            lx, ly, lz = 0.45, 0.75, -0.48
            ll = math.sqrt(lx * lx + ly * ly + lz * lz)
            lx /= ll
            ly /= ll
            lz /= ll
            hxv = lx - dx
            hyv = ly - dy
            hzv = lz - dz
            hl = math.sqrt(hxv * hxv + hyv * hyv + hzv * hzv) + 1e-9
            ndh = max(0.0, (nx * hxv + ny * hyv + nz * hzv) / hl)
            spec = ndh ** 90 * 6.0 + ndh ** 18 * 0.35
            diff = max(0.0, nx * lx + ny * ly + nz * lz)
            # base: deep emerald glass with a faint inner (subsurface) glow
            sss = math.exp(-abs(d2) / 1.6) * 0.5 + 0.5
            br = 0.006 + 0.012 * diff + 0.010 * sss
            bg = 0.040 + 0.070 * diff + 0.085 * sss
            bb = 0.026 + 0.045 * diff + 0.055 * sss
            cr = br + fres * er * 0.9 + spec
            cg = bg + fres * eg * 0.9 + spec
            cb = bb + fres * eb * 0.9 + spec * 0.95
            # emissive edge: front-face rim where the 2D outline is near
            face = 1.0 - min(1.0, abs(abs(nz) - 1.0) * 4.0)
            edge = math.exp(-abs(d2) / 0.13) * (0.30 + 0.70 * face) + (1.0 - face) * 0.22
            em = glow * edge
            cr += em * 0.25
            cg += em * 1.60
            cb += em * 0.95
            # light sweep: a bright band travelling across the faces
            if sweep[3] > 0.0:
                sd = (hx * sweep[0] + hy * sweep[1]) - sweep[2]
                band = math.exp(-(sd / 1.2) ** 2) * sweep[3]
                cr += band * (0.6 + 2.5 * fres)
                cg += band * (0.9 + 2.6 * fres)
                cb += band * (0.8 + 2.6 * fres)
            out[y, x, 0] = cr * exposure
            out[y, x, 1] = cg * exposure
            out[y, x, 2] = cb * exposure
            alpha[y, x] = 1.0


def part_xf(R, T, s=1.0):
    """world = s * R @ local + T ; returns the inverse as 3x4 plus scale slot."""
    Ri = R.T / s
    m = np.zeros((4, 4))
    m[:3, :3] = Ri
    m[:3, 3] = -Ri @ T
    m[3, 0] = 1.0 / s
    return m


def rot_axis(axis, ang):
    a = np.asarray(axis, np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    x, y, z = a
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


_TABLES = {}


def tables():
    if not _TABLES:
        v, s, c, b = lockup_tables()
        pad = 0.6
        b = b + np.array([-pad, -pad, pad, pad])
        _TABLES["t"] = (v, s, c, b)
    return _TABLES["t"]


def render(cam, xfs, vis, sky, t=0.0, glow=1.0, half=1.25, bev=0.32, sweep=(0.7, 0.7, -999.0, 0.0), rect=None,
           exposure=1.0, out=None, alpha=None):
    v, s, c, b = tables()
    if out is None:
        out = np.zeros((cam.h, cam.w, 3), np.float32)
    if alpha is None:
        alpha = np.zeros((cam.h, cam.w), np.float32)
    if rect is None:
        rect = (0, 0, cam.w, cam.h)
    render_logo(out, alpha, cam.pos.astype(np.float64), cam.basis(), cam.th, v, s, c, b,
                np.asarray(xfs, np.float64), np.asarray(vis, np.float64), float(glow), float(half), float(bev),
                sky, float(t), np.asarray(sweep, np.float64), np.asarray(rect, np.int64), float(exposure))
    return out, alpha
