"""Earth from orbit: NASA Blue Marble / Black Marble / cloud textures on a sphere,
soft terminator, ocean glint, Rayleigh-ish limb, green airglow layer, aurora."""
import math
import os

import cv2
import numpy as np
from numba import njit, prange

from gfx import ASSETS, fbm3, vnoise3

TEX = os.path.join(ASSETS, "textures")
_T = {}


def textures():
    if not _T:
        def lin(path, size=None, gray=False):
            im = cv2.imread(os.path.join(TEX, path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
            if size:
                im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
            if not gray:
                im = im[:, :, ::-1]
            return np.ascontiguousarray(im)
        _T["day"] = lin("bluemarble_5400.jpg")
        _T["night"] = lin("blackmarble_3km.jpg", (8192, 4096))
        _T["cloud"] = lin("clouds_2048.jpg", gray=True)
        _T["spec"] = lin("earth_specular_2048.jpg", gray=True)
    return _T


@njit(inline="always", fastmath=True)
def _tex3(img, u, v):
    h, w = img.shape[0], img.shape[1]
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    fx = x - x0
    fy = y - y0
    x0 = x0 % w
    x1 = (x0 + 1) % w
    y0 = min(max(y0, 0), h - 1)
    y1 = min(y0 + 1, h - 1)
    r = (img[y0, x0, 0] * (1 - fx) + img[y0, x1, 0] * fx) * (1 - fy) + (img[y1, x0, 0] * (1 - fx) + img[y1, x1, 0] * fx) * fy
    g = (img[y0, x0, 1] * (1 - fx) + img[y0, x1, 1] * fx) * (1 - fy) + (img[y1, x0, 1] * (1 - fx) + img[y1, x1, 1] * fx) * fy
    b = (img[y0, x0, 2] * (1 - fx) + img[y0, x1, 2] * fx) * (1 - fy) + (img[y1, x0, 2] * (1 - fx) + img[y1, x1, 2] * fx) * fy
    return r / 255.0, g / 255.0, b / 255.0


@njit(inline="always", fastmath=True)
def _tex1(img, u, v):
    h, w = img.shape[0], img.shape[1]
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    fx = x - x0
    fy = y - y0
    x0 = x0 % w
    x1 = (x0 + 1) % w
    y0 = min(max(y0, 0), h - 1)
    y1 = min(y0 + 1, h - 1)
    return ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy) / 255.0


@njit(inline="always", fastmath=True)
def _sstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


@njit(parallel=True, fastmath=True, cache=True)
def render_planet(out, alpha, cam_pos, basis, th, rot, sun, day, night, cloud, spec, params):
    """Planet of radius 1 at the origin. rot: 3x3 world->planet matrix.
    params: [city_gain, day_gain, airglow, aurora, atm_gain, t, cloud_shift]"""
    h, w = out.shape[0], out.shape[1]
    asp = w / h
    city_gain = params[0]
    day_gain = params[1]
    airglow = params[2]
    aurora = params[3]
    atm_gain = params[4]
    tt = params[5]
    cshift = params[6]
    RA = 1.045
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
            c = ox * ox + oy * oy + oz * oz
            # closest approach (for limb effects)
            tca = -b
            d2 = c - b * b
            dmin = math.sqrt(max(d2, 0.0))
            disc_a = b * b - (c - RA * RA)
            if disc_a <= 0.0 or tca < -RA:
                continue
            ta0 = -b - math.sqrt(disc_a)
            ta1 = -b + math.sqrt(disc_a)
            ta0 = max(ta0, 0.0)
            disc_p = b * b - (c - 1.0)
            hit = False
            tp = 0.0
            if disc_p > 0.0:
                tp = -b - math.sqrt(disc_p)
                if tp > 0.0:
                    hit = True
            cr = 0.0
            cg = 0.0
            cb = 0.0
            a = 0.0
            if hit:
                hx = ox + dx * tp
                hy = oy + dy * tp
                hz = oz + dz * tp
                # planet-local coordinates
                lx = rot[0, 0] * hx + rot[0, 1] * hy + rot[0, 2] * hz
                ly = rot[1, 0] * hx + rot[1, 1] * hy + rot[1, 2] * hz
                lz = rot[2, 0] * hx + rot[2, 1] * hy + rot[2, 2] * hz
                lon = math.atan2(lz, lx)
                lat = math.asin(max(-1.0, min(1.0, ly)))
                u = 0.5 - lon / (2 * math.pi)
                v = 0.5 - lat / math.pi
                ndl = hx * sun[0] + hy * sun[1] + hz * sun[2]
                dr, dg, db = _tex3(day, u, v)
                dr = dr ** 2.2
                dg = dg ** 2.2
                db = db ** 2.2
                cl = _tex1(cloud, (u + cshift) % 1.0, v)
                cl = min(1.0, cl * 1.15) ** 1.4
                lit = _sstep(-0.08, 0.35, ndl)
                diff = max(ndl, 0.0) ** 0.8
                sr = (dr * (1 - cl) + 0.85 * cl) * diff
                sg = (dg * (1 - cl) + 0.88 * cl) * diff
                sb = (db * (1 - cl) + 0.92 * cl) * diff
                # ocean glint
                sm = _tex1(spec, u, v)
                rx = sun[0] - 2 * ndl * hx
                ry = sun[1] - 2 * ndl * hy
                rz = sun[2] - 2 * ndl * hz
                vd = -(rx * dx + ry * dy + rz * dz)
                gl = sm * (1 - cl) * max(vd, 0.0) ** 80 * 3.0 * lit
                nr, ng, nb = _tex3(night, u, v)
                nl = (nr * 0.6 + ng * 0.3 + nb * 0.1)
                nl = nl ** 2.0 * (1.0 - _sstep(-0.12, 0.06, ndl)) * (1.0 - 0.75 * cl)
                cr = day_gain * (sr + gl) + city_gain * nl * 1.00
                cg = day_gain * (sg + gl * 0.95) + city_gain * nl * 0.62
                cb = day_gain * (sb + gl * 0.85) + city_gain * nl * 0.30
                # terminator warmth on clouds
                tw = math.exp(-((ndl - 0.03) / 0.045) ** 2) * cl
                cr += 0.07 * tw * day_gain
                cg += 0.025 * tw * day_gain
                a = 1.0
                # atmospheric veil over the surface (more toward the limb)
                mu = -(dx * hx + dy * hy + dz * hz)
                veil = (1.0 - mu) ** 3
                cr = cr * (1 - 0.25 * veil) + atm_gain * veil * lit * 0.10
                cg = cg * (1 - 0.25 * veil) + atm_gain * veil * lit * 0.22
                cb = cb * (1 - 0.25 * veil) + atm_gain * veil * lit * 0.45
            # limb / atmosphere shell glow for rays near the edge
            if dmin > 0.97:
                hgt = (dmin - 1.0)
                # sun factor at the tangent point
                tx = ox + dx * tca
                ty = oy + dy * tca
                tz = oz + dz * tca
                tl = math.sqrt(tx * tx + ty * ty + tz * tz)
                sl = (tx * sun[0] + ty * sun[1] + tz * sun[2]) / tl
                mu_s = dx * sun[0] + dy * sun[1] + dz * sun[2]
                fw = max(mu_s, 0.0)
                phase = 0.10 + 0.8 * fw ** 4 + 4.0 * fw ** 32
                d_low = math.exp(-max(hgt, 0.0) / 0.0028)
                d_high = math.exp(-max(hgt, 0.0) / 0.0085)
                if hit:
                    d_low *= 0.35 * math.exp(-(1.0 - dmin) / 0.0025)
                    d_high *= 0.35 * math.exp(-(1.0 - dmin) / 0.007)
                day_l = _sstep(-0.2, 0.35, sl)
                twl = math.exp(-((sl + 0.02) / 0.11) ** 2)
                cr += atm_gain * (d_high * day_l * 0.27 + twl * phase * (d_low * 1.00 + d_high * 0.08))
                cg += atm_gain * (d_high * day_l * 0.50 + twl * phase * (d_low * 0.42 + d_high * 0.22))
                cb += atm_gain * (d_high * day_l * 0.95 + twl * phase * (d_low * 0.10 + d_high * 0.55))
                a = max(a, min(0.5, (d_high * (day_l + twl * phase)) * 0.5))
                # thin green airglow layer (~100 km), visible on the night limb
                ag = math.exp(-((hgt - 0.0155) / 0.0018) ** 2) * airglow * (1.0 - 0.85 * day_l)
                cr += ag * 0.10
                cg += ag * 1.00
                cb += ag * 0.45
            # aurora curtains: march the shell segment near the polar ovals
            if aurora > 0.0:
                t0 = ta0
                t1 = tp if hit else ta1
                steps = 40
                acc_g = 0.0
                acc_r = 0.0
                for k in range(steps):
                    s = t0 + (t1 - t0) * (k + 0.5) / steps
                    qx = ox + dx * s
                    qy = oy + dy * s
                    qz = oz + dz * s
                    lx = rot[0, 0] * qx + rot[0, 1] * qy + rot[0, 2] * qz
                    ly = rot[1, 0] * qx + rot[1, 1] * qy + rot[1, 2] * qz
                    lz = rot[2, 0] * qx + rot[2, 1] * qy + rot[2, 2] * qz
                    rr = math.sqrt(lx * lx + ly * ly + lz * lz)
                    hh = rr - 1.0
                    if hh < 0.0150 or hh > 0.042:
                        continue
                    lat = math.asin(ly / rr)
                    lon = math.atan2(lz, lx)
                    # wavy oval: the curtain's latitude meanders with longitude
                    lat0 = 1.16 + 0.05 * math.sin(lon * 5.0 + tt * 0.15) + 0.03 * vnoise3(lon * 6.0, tt * 0.1, 3.0)
                    band = math.exp(-((abs(lat) - lat0) / 0.035) ** 2)
                    if band < 0.03:
                        continue
                    rays = vnoise3(lon * 140.0, tt * 0.6, hh * 8.0)
                    rays = 0.25 + 0.75 * rays * rays
                    curtain = max(0.0, vnoise3(lon * 7.0 + tt * 0.25, 2.0, tt * 0.1) - 0.3) * 1.8
                    vert = (hh - 0.0150) / 0.027
                    e = band * curtain * rays * (t1 - t0) / steps * 16.0
                    acc_g += e * math.exp(-vert * 3.0) * (1.0 - math.exp(-vert * 40.0 - 0.2))
                    acc_r += e * vert * vert * 0.35
                cr += aurora * (acc_g * 0.10 + acc_r * 0.85)
                cg += aurora * (acc_g * 1.00 + acc_r * 0.10)
                cb += aurora * (acc_g * 0.35 + acc_r * 0.60)
            out[y, x, 0] = cr
            out[y, x, 1] = cg
            out[y, x, 2] = cb
            alpha[y, x] = a


def rot_matrix(yaw=0.0, tilt=0.0):
    """World->planet rotation: spin about the planet axis (yaw), axial tilt about z."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    ct, st = math.cos(tilt), math.sin(tilt)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    return (Ry @ Rz).astype(np.float64)


def render(cam, sun, yaw=0.0, tilt=0.0, city_gain=2.2, day_gain=1.0, airglow=0.07, aurora=0.0, atm_gain=1.0,
           t=0.0, cloud_shift=0.0):
    tx = textures()
    out = np.zeros((cam.h, cam.w, 3), np.float32)
    alpha = np.zeros((cam.h, cam.w), np.float32)
    sun = np.asarray(sun, np.float64)
    sun = sun / np.linalg.norm(sun)
    params = np.array([city_gain, day_gain, airglow, aurora, atm_gain, t, cloud_shift], np.float64)
    render_planet(out, alpha, cam.pos.astype(np.float64), cam.basis(), cam.th, rot_matrix(yaw, tilt), sun,
                  tx["day"], tx["night"], tx["cloud"], tx["spec"], params)
    return out, alpha


def latlon_to_world(lat_deg, lon_deg, yaw=0.0, tilt=0.0, r=1.0):
    """Inverse of the texture mapping in render_planet (planet->world)."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    # texture u = 0.5 - lon/2pi with lon = atan2(lz, lx); geographic lon L maps to u = (L+180)/360
    a = -math.radians(lon_deg)
    lx, ly, lz = math.cos(lat) * math.cos(a), math.sin(lat), math.cos(lat) * math.sin(a)
    R = rot_matrix(yaw, tilt)
    return (R.T @ np.array([lx, ly, lz])) * r
