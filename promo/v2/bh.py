"""Schwarzschild black hole with a thin accretion disk (units: r_s = 1).

Photon paths are integrated with the classic Newtonian-form geodesic
a = -1.5 h^2 x / r^5, which reproduces Schwarzschild light bending; the disk
gets Keplerian rotation, relativistic Doppler beaming and gravitational redshift.
"""
import math

import cv2
import numpy as np
from numba import njit, prange

from gfx import fbm3, sample_equirect


@njit(inline="always", fastmath=True)
def _ramp(T):
    # temperature -> colour (dark ember -> amber -> gold -> white-hot)
    if T < 0.25:
        k = T / 0.25
        return 0.35 * k + 0.05, 0.06 * k, 0.01 * k
    if T < 0.55:
        k = (T - 0.25) / 0.30
        return 0.40 + 0.60 * k, 0.06 + 0.42 * k, 0.01 + 0.10 * k
    if T < 0.95:
        k = (T - 0.55) / 0.40
        return 1.0, 0.48 + 0.40 * k, 0.11 + 0.47 * k
    k = min(1.0, (T - 0.95) / 0.8)
    return 1.0 - 0.08 * k, 0.88 + 0.08 * k, 0.58 + 0.45 * k


@njit(parallel=True, fastmath=True, cache=True)
def render_bh(out, cam_pos, basis, th, t, sky, r_in, r_out, gain, emerald, sky_gain, tilt):
    h, w = out.shape[0], out.shape[1]
    asp = w / h
    ct = math.cos(tilt)
    st = math.sin(tilt)
    for y in prange(h):
        py0 = (1.0 - 2.0 * (y + 0.5) / h) * th
        for x in range(w):
            px0 = (2.0 * (x + 0.5) / w - 1.0) * th * asp
            vx = basis[2, 0] + px0 * basis[0, 0] + py0 * basis[1, 0]
            vy = basis[2, 1] + px0 * basis[0, 1] + py0 * basis[1, 1]
            vz = basis[2, 2] + px0 * basis[0, 2] + py0 * basis[1, 2]
            n = math.sqrt(vx * vx + vy * vy + vz * vz)
            vx /= n
            vy /= n
            vz /= n
            # rotate world into the disk frame (disk tilted about the x axis)
            px = cam_pos[0]
            py = cam_pos[1] * ct + cam_pos[2] * st
            pz = -cam_pos[1] * st + cam_pos[2] * ct
            tvy = vy * ct + vz * st
            vz = -vy * st + vz * ct
            vy = tvy
            cx = py * vz - pz * vy
            cy = pz * vx - px * vz
            cz = px * vy - py * vx
            h2 = cx * cx + cy * cy + cz * cz
            cr = 0.0
            cg = 0.0
            cb = 0.0
            trans = 1.0
            for i in range(700):
                r2 = px * px + py * py + pz * pz
                r = math.sqrt(r2)
                if r < 1.0:
                    trans = 0.0
                    break
                dt = min(max(0.045 * r, 0.012), 0.9)
                if r < 2.6:
                    dt = min(dt, 0.03)
                a = -1.5 * h2 / (r2 * r2 * r)
                vx += a * px * dt
                vy += a * py * dt
                vz += a * pz * dt
                nx = px + vx * dt
                ny = py + vy * dt
                nz = pz + vz * dt
                if py * ny < 0.0:
                    s = py / (py - ny)
                    ix = px + (nx - px) * s
                    iz = pz + (nz - pz) * s
                    rc = math.sqrt(ix * ix + iz * iz)
                    if rc > r_in * 0.92 and rc < r_out:
                        phi = math.atan2(iz, ix)
                        om = 0.55 * (r_in / rc) ** 1.5
                        ang = phi + om * t
                        ca = math.cos(ang)
                        sa = math.sin(ang)
                        n1 = fbm3(rc * 2.6, ca * rc * 0.55 + 3.0, sa * rc * 0.55, 5)
                        n2 = fbm3(rc * 9.0 + 7.0, ca * 2.2, sa * 2.2 + 5.0, 4)
                        rings = 0.55 + 0.45 * math.sin(rc * 7.0 + n1 * 6.0)
                        tex = (0.35 + 1.3 * n1 * n1) * (0.6 + 0.4 * rings) * (0.75 + 0.5 * n2)
                        edge_in = min(1.0, max(0.0, (rc - r_in * 0.92) / (r_in * 0.12)))
                        edge_out = min(1.0, max(0.0, (r_out - rc) / (r_out * 0.45)))
                        prof = (r_in / rc) ** 2.0 * edge_in * edge_out * edge_out
                        # Doppler: disk velocity (counter-clockwise seen from +y)
                        beta = math.sqrt(0.5 / max(rc - 0.5, 0.6))
                        tx = -iz / rc
                        tz = ix / rc
                        vn = math.sqrt(vx * vx + vy * vy + vz * vz)
                        cosv = -(tx * vx + tz * vz) / vn
                        gam = 1.0 / math.sqrt(1.0 - beta * beta)
                        dop = 1.0 / (gam * (1.0 - beta * cosv))
                        grav = math.sqrt(max(0.0, 1.0 - 1.0 / rc))
                        gfac = dop * grav
                        temp = (r_in / rc) ** 0.75 * gfac * (0.8 + 0.35 * n2)
                        rr, gg, bb = _ramp(temp)
                        em = emerald * min(1.0, max(0.0, (rc - r_in * 1.6) / (r_out - r_in * 1.6)))
                        rr = rr * (1 - em) + 0.10 * em
                        gg = gg * (1 - em) + 0.95 * em
                        bb = bb * (1 - em) + 0.55 * em
                        inten = gain * prof * tex * gfac ** 3
                        alpha = min(0.95, (0.25 + 0.75 * tex) * edge_in * (0.4 + 0.6 * edge_out))
                        cr += trans * inten * rr
                        cg += trans * inten * gg
                        cb += trans * inten * bb
                        trans *= 1.0 - alpha
                        if trans < 0.01:
                            break
                px = nx
                py = ny
                pz = nz
                if r > 60.0 and (px * vx + py * vy + pz * vz) > 0.0:
                    break
            if trans > 0.0:
                vn = math.sqrt(vx * vx + vy * vy + vz * vz)
                # back to world frame for the sky lookup
                wy = (vy * ct - vz * st) / vn
                wz = (vy * st + vz * ct) / vn
                sr, sg, sb = sample_equirect(sky, vx / vn, wy, wz)
                cr += trans * sr * sky_gain
                cg += trans * sg * sky_gain
                cb += trans * sb * sky_gain
            out[y, x, 0] = cr
            out[y, x, 1] = cg
            out[y, x, 2] = cb


def render(cam, t, sky, r_in=3.0, r_out=16.0, gain=2.2, emerald=0.35, sky_gain=1.0, tilt=0.0, ss=1.0):
    w, h = int(cam.w * ss), int(cam.h * ss)
    out = np.zeros((h, w, 3), np.float32)
    render_bh(out, cam.pos.astype(np.float64), cam.basis(), cam.th, float(t), sky, float(r_in), float(r_out),
              float(gain), float(emerald), float(sky_gain), float(tilt))
    if ss != 1.0:
        out = cv2.resize(out, (cam.w, cam.h), interpolation=cv2.INTER_AREA)
    return out
