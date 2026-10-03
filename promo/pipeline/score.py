"""Procedural trailer score + sound design + narration mix for the ETZ film.

MiniMax's music API is closed to new users, so the score is synthesised here,
cut-for-cut against the picture (80 BPM, D minor -> D major at the logo).
Writes build/audio/mix.wav (48 kHz stereo) and per-stem wavs.
"""
import json
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
EDL = json.load(open(os.path.join(ROOT, "script", "edl.json")))
SR = 48000
DUR = 60.0
N = int(SR * DUR)
RNG = np.random.default_rng(2026)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


# ----------------------------------------------------------------------------- building blocks
def lp(x, fc, order=2):
    sos = signal.butter(order, min(fc, SR * 0.45), "low", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def hp(x, fc, order=2):
    sos = signal.butter(order, fc, "high", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def bp(x, f0, q=2.0, order=2):
    lo, hi = f0 / (1 + 0.5 / q), min(f0 * (1 + 0.5 / q), SR * 0.45)
    sos = signal.butter(order, [lo, hi], "band", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def sweep_filter(x, fcs, kind="low", block=256, q=1.5):
    """Time-varying filter: fcs is an array (per sample) of cutoff/centre Hz."""
    out = np.zeros_like(x)
    zi = None
    for i in range(0, len(x), block):
        fc = float(np.clip(fcs[min(i + block // 2, len(fcs) - 1)], 30, SR * 0.44))
        if kind == "low":
            sos = signal.butter(2, fc, "low", fs=SR, output="sos")
        else:
            lo, hi = fc / (1 + 0.5 / q), min(fc * (1 + 0.5 / q), SR * 0.44)
            sos = signal.butter(1, [lo, hi], "band", fs=SR, output="sos")
        if zi is None or zi.shape[0] != sos.shape[0]:
            zi = np.zeros((sos.shape[0], 2))
        out[i:i + block], zi = signal.sosfilt(sos, x[i:i + block], zi=zi)
    return out


def saw(freq, t, phase=0.0):
    ph = np.cumsum(np.broadcast_to(freq, t.shape) / SR) + phase
    return 2.0 * (ph - np.floor(ph + 0.5))


def sine(freq, t, phase=0.0):
    ph = np.cumsum(np.broadcast_to(freq, t.shape) / SR) + phase
    return np.sin(2 * np.pi * ph)


def env_ar(n, a, r, curve=2.0):
    """Attack (s) then exponential-ish release (s) over n samples."""
    t = np.arange(n) / SR
    att = np.clip(t / max(a, 1e-4), 0, 1) ** curve
    rel = np.exp(-np.maximum(t - a, 0) / max(r, 1e-4))
    return att * rel


def env_hold(n, a, hold, r):
    t = np.arange(n) / SR
    att = np.clip(t / max(a, 1e-4), 0, 1)
    rel = np.clip(1 - (t - a - hold) / max(r, 1e-4), 0, 1)
    rel = np.where(t < a + hold, 1.0, rel)
    return att * rel ** 2


def pan(x, p):
    """p in [-1, 1] (constant power) -> stereo."""
    a = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], 1)


def softclip(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


def reverb_ir(rt60=3.5, pre=0.03, bright=6000, seed=5):
    r = np.random.default_rng(seed)
    n = int(SR * rt60 * 1.1)
    t = np.arange(n) / SR
    ir = r.standard_normal((n, 2)) * np.exp(-6.9 * t / rt60)[:, None]
    # darker tail: blend a low-passed copy in over time
    dark = lp(ir, 1800)
    mix = np.clip(t / (rt60 * 0.5), 0, 1)[:, None]
    ir = lp(ir * (1 - mix) + dark * mix, bright)
    ir = np.concatenate([np.zeros((int(pre * SR), 2)), ir])
    return ir / np.sqrt((ir ** 2).sum(0)).max()


def reverb(x, ir, wet=0.3):
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    y = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    return x * (1 - wet) + y * wet


def delay_pingpong(x, d=0.375, fb=0.38, n=5, damp=3500):
    out = x.copy()
    L = len(x)
    cur = x
    for k in range(1, n + 1):
        cur = lp(cur, damp) * fb
        s = int(d * k * SR)
        if s >= L:
            break
        sh = np.zeros_like(x)
        sh[s:] = cur[:L - s]
        if k % 2 == 1:
            sh = sh[:, ::-1]
        out += sh
    return out


class Bus:
    def __init__(self, name):
        self.name = name
        self.x = np.zeros((N, 2))

    def add(self, sig, at, gain=1.0, p=0.0):
        if sig.ndim == 1:
            sig = pan(sig, p)
        i = int(at * SR)
        if i >= N:
            return
        j = min(N, i + len(sig))
        if i < 0:
            sig = sig[-i:]
            i = 0
            j = min(N, len(sig))
        self.x[i:j] += sig[:j - i] * gain


# ----------------------------------------------------------------------------- instruments
def organ(notes, dur, attack=1.5, release=2.0, bright=0.5, seed=0):
    """Interstellar-ish pipe organ: additive stops, slow swell, gentle chorus."""
    t = tvec(dur + release)
    out = np.zeros(len(t))
    r = np.random.default_rng(seed)
    stops = [(1, 1.0), (2, 0.55 * bright + 0.2), (3, 0.25 * bright), (4, 0.22 * bright), (6, 0.08 * bright),
             (8, 0.06 * bright), (0.5, 0.35)]
    for m in notes:
        f = midi(m)
        for h, a in stops:
            for det in (-0.12, 0.12):
                ff = f * h * 2 ** (det / 1200 * 12 * 0.06)
                if ff < SR * 0.4:
                    vib = 1 + 0.0012 * np.sin(2 * np.pi * (4.6 + r.uniform(-0.3, 0.3)) * t)
                    out += a * sine(ff * vib, t, r.uniform(0, 1)) * 0.5
    env = env_hold(len(t), attack, dur - attack, release)
    return out * env / max(1, len(notes))


def choir(notes, dur, attack=1.2, release=2.0, seed=1):
    """Formant-filtered detuned saws ("aah")."""
    t = tvec(dur + release)
    src = np.zeros(len(t))
    r = np.random.default_rng(seed)
    for m in notes:
        for k in range(5):
            det = r.uniform(-9, 9)
            vib = 1 + 0.004 * np.sin(2 * np.pi * r.uniform(4.5, 5.5) * t + r.uniform(0, 6))
            src += saw(midi(m) * 2 ** (det / 1200) * vib, t, r.uniform(0, 1))
    y = bp(src, 700, 2.5) * 1.0 + bp(src, 1150, 3.0) * 0.6 + bp(src, 2600, 4.0) * 0.25
    y = lp(y, 5000)
    env = env_hold(len(t), attack, dur - attack, release)
    return y * env / max(1, len(notes)) * 0.7


def pluck(m, dur=0.9, bright=2500, seed=0):
    t = tvec(dur)
    f = midi(m)
    x = 0.6 * sine(f, t) + 0.25 * sine(f * 2, t) + 0.12 * np.sign(sine(f, t)) * 0.3 + 0.08 * sine(f * 3.01, t)
    e = env_ar(len(t), 0.004, 0.22)
    return lp(x * e, bright)


def bell(m, dur=3.0, amp=1.0):
    t = tvec(dur)
    f = midi(m)
    parts = [(1.0, 1.0, 1.6), (2.756, 0.55, 0.9), (5.404, 0.3, 0.5), (8.933, 0.15, 0.3), (0.5, 0.25, 2.0)]
    y = sum(a * np.sin(2 * np.pi * f * k * t) * np.exp(-t / d) for k, a, d in parts)
    return y * env_ar(len(t), 0.002, 10.0) * amp * 0.4


def braam(notes, dur=3.0, drive=2.5):
    t = tvec(dur)
    y = np.zeros(len(t))
    for m in notes:
        for det in (-14, -6, 0, 7, 13):
            y += saw(midi(m) * 2 ** (det / 1200), t, RNG.uniform(0, 1))
        y += 0.6 * np.sign(sine(midi(m) * 0.5, t))
    y /= len(notes) * 5
    fc = 120 + 2600 * np.exp(-t / 0.55) + 400 * np.exp(-t / 2.0)
    y = sweep_filter(y, fc, "low")
    y = softclip(y * 2.2, drive)
    e = env_hold(len(t), 0.03, dur * 0.35, dur * 0.65)
    return lp(y * e, 3500)


def taiko(big=1.0, pitch=1.0):
    t = tvec(1.6)
    f = (52 + 110 * np.exp(-t / 0.035)) * pitch
    body = sine(f, t) * np.exp(-t / (0.32 * big))
    skin = bp(RNG.standard_normal(len(t)), 900, 1.2) * np.exp(-t / 0.03) * 0.8
    thud = lp(RNG.standard_normal(len(t)), 300) * np.exp(-t / 0.08) * 1.5
    return softclip((body * 1.4 + skin + thud) * big, 1.5)


def kick(amp=1.0):
    t = tvec(0.6)
    f = 45 + 120 * np.exp(-t / 0.04)
    y = sine(f, t) * np.exp(-t / 0.22) + 0.3 * lp(RNG.standard_normal(len(t)), 2000) * np.exp(-t / 0.008)
    return y * amp


def tom(m=45, amp=1.0):
    t = tvec(0.7)
    f = midi(m) * (1 + 0.6 * np.exp(-t / 0.03))
    y = sine(f, t) * np.exp(-t / 0.25) + 0.2 * bp(RNG.standard_normal(len(t)), 1200, 1.0) * np.exp(-t / 0.02)
    return y * amp


def impact_noise(dur=3.0, hp_f=1500):
    t = tvec(dur)
    n = RNG.standard_normal((len(t), 2))
    y = hp(n, hp_f) * np.exp(-t / 0.9)[:, None] + lp(n, 400) * np.exp(-t / 0.35)[:, None] * 0.8
    return y * 0.5


def subdrop(f0=62, f1=27, dur=2.2):
    t = tvec(dur)
    f = f1 + (f0 - f1) * np.exp(-t / 0.45)
    return sine(f, t) * env_ar(len(t), 0.005, 0.9)


def whoosh(dur=1.1, f0=300, f1=4000, peak=0.6):
    t = tvec(dur)
    u = t / dur
    fc = f0 * (f1 / f0) ** np.sin(np.pi * u) ** 1.2
    n = RNG.standard_normal(len(t))
    y = sweep_filter(n, fc, "band", q=1.2)
    env = np.exp(-0.5 * ((u - peak) / 0.22) ** 2)
    st = pan(y * env, 0.0)
    p = np.linspace(-0.7, 0.7, len(t))
    a = (p + 1) * np.pi / 4
    st[:, 0] *= np.cos(a) * 1.4
    st[:, 1] *= np.sin(a) * 1.4
    return st


def riser(dur, f0=180, f1=5000, tone=(50, 62)):
    t = tvec(dur)
    u = t / dur
    fc = f0 * (f1 / f0) ** (u ** 1.6)
    n = RNG.standard_normal(len(t))
    y = sweep_filter(n, fc, "band", q=2.0) * (u ** 2.2) * 1.4
    for m in tone:  # pitch-rising tones (an octave over the riser)
        f = midi(m) * 2 ** (u ** 1.8)
        y += 0.25 * saw(f, t) * (u ** 2.5)
    y = lp(y, 9000)
    return y


def reverse_swell(dur=0.6):
    t = tvec(dur)
    n = RNG.standard_normal((len(t), 2))
    y = hp(n, 1200) * np.exp(-(dur - t) / 0.25)[:, None]
    return y * 0.6


def drone(dur, m=38):
    t = tvec(dur)
    f = midi(m)
    y = sine(f, t) + 0.35 * sine(f * 2, t) + 0.12 * sine(f * 3, t)
    y += 0.25 * lp(RNG.standard_normal(len(t)), 120)
    y *= 1 + 0.08 * np.sin(2 * np.pi * 0.13 * t)
    return y


# ----------------------------------------------------------------------------- score
CHORDS = {
    "Dm": [38, 50, 53, 57, 62], "Bb": [34, 46, 50, 53, 58], "F": [41, 48, 53, 57, 60],
    "C": [36, 48, 52, 55, 60], "Gm": [43, 50, 55, 58, 62], "A": [45, 52, 57, 61, 64],
    "D": [38, 50, 54, 57, 62, 66],
}
PROG = [(5.25, "Dm"), (10.5, "Dm"), (15.0, "Bb"), (20.25, "F"), (24.0, "C"), (28.5, "Dm"), (31.5, "Bb"),
        (34.5, "F"), (37.5, "C"), (39.75, "Gm"), (42.0, "A"), (44.25, "Dm"), (45.75, "Bb"), (47.25, "Gm"),
        (48.0, "A"), (49.0, None), (49.5, "D"), (60.0, None)]


def chord_at(t):
    cur = None
    for s, c in PROG:
        if t >= s:
            cur = c
    return cur


def build():
    pad, arp, perc, fxb, low = Bus("pad"), Bus("arp"), Bus("perc"), Bus("fx"), Bus("low")

    # --- sub drone across the film (swells into the bang, the climax, the logo)
    d = drone(DUR)
    tt = np.arange(N) / SR
    ramp = np.clip((tt - 0.8) / 3.7, 0, 1) ** 2
    g = (ramp * (0.55 - 0.18 * np.clip((tt - 8) / 4, 0, 1)) + 0.25 * np.exp(-((tt - 33) / 4) ** 2)
         + 0.35 * np.clip((tt - 42) / 7, 0, 1) * (tt < 49.0) + 0.5 * (tt >= 49.5) * np.exp(-(tt - 49.5) / 6.0))
    g *= np.clip((DUR - 0.3 - tt) / 1.6, 0, 1)
    low.add(d * g * 0.30, 0.0)

    # --- cold open: the spark, the inhale, the big bang
    fxb.add(pan(bell(93, 4.0, 0.5), 0.15), 0.6, 0.5)
    fxb.add(pan(riser(2.05, 120, 7000, (38, 45)), 0.0), 3.2, 0.55)
    perc.add(pan(taiko(1.6, 0.8), 0), 5.25, 0.9)
    low.add(pan(subdrop(70, 26, 3.0), 0), 5.25, 1.0)
    fxb.add(impact_noise(4.0, 900), 5.25, 0.9)
    pad.add(pan(braam(CHORDS["Dm"][:3], 3.5, 2.0), 0), 5.25, 0.35)
    pad.add(pan(choir([62, 65, 69], 5.0, 0.8, 3.0), 0), 5.4, 0.45)

    # --- organ / choir bed following the progression
    for i, (s, c) in enumerate(PROG[:-1]):
        e = PROG[i + 1][0]
        if c is None or s < 10.0 or s >= 49.0:
            continue
        dur = e - s
        notes = CHORDS[c]
        bright = 0.3 + 0.5 * min(1.0, (s - 10) / 34)
        vol = 0.24 + 0.46 * min(1.0, (s - 10) / 34) ** 1.3
        pad.add(pan(organ(notes[1:], dur, attack=min(1.2, dur * 0.4), release=1.2, bright=bright, seed=i), -0.1), s, vol)
        pad.add(pan(organ([notes[0]], dur, attack=0.4, release=1.0, bright=0.2, seed=50 + i), 0.1), s, vol * 0.8)
        if s >= 28.5:
            pad.add(pan(choir([n + 12 for n in notes[2:]], dur, 0.6, 1.0, seed=i), 0.0), s, 0.18 + 0.2 * (s > 41))

    # --- glassy 8th-note arpeggio 10.5 -> 49.0, ping-pong delayed
    step = 0.375
    k = 0
    tcur = 10.5
    pattern = [0, 2, 3, 4, 3, 2, 4, 1]
    while tcur < 48.95:
        c = chord_at(tcur + 1e-3)
        if c:
            notes = CHORDS[c]
            m = notes[pattern[k % len(pattern)] % len(notes)] + 24
            while m > 86:
                m -= 12
            prog = min(1.0, (tcur - 10.5) / 34)
            acc = 1.0 if k % 2 == 0 else 0.7
            arp.add(pan(pluck(m, 0.8, 1400 + 4200 * prog), 0.3 * np.sin(k * 0.9)), tcur, (0.16 + 0.22 * prog) * acc)
        tcur += step
        k += 1
    # end-card arpeggio: slower, fading
    for j in range(10):
        tcur = 50.25 + j * 0.75
        m = CHORDS["D"][[1, 3, 5, 4, 2, 3, 5, 4, 1, 3][j]] + 24
        arp.add(pan(pluck(m, 1.2, 3000), 0.4 * np.sin(j)), tcur, 0.22 * (1 - j / 11))

    # --- constellation pings on each candle (S06)
    for i in range(11):
        m = [74, 76, 77, 79, 81, 79, 81, 84, 81, 84, 86][i]
        fxb.add(pan(bell(m, 2.0, 0.35), -0.6 + 0.12 * i), 24.25 + i * 0.375, 0.5)

    # --- low string ostinato (16ths) 34.5 -> 49.0
    t16 = 0.1875
    j = 0
    tcur = 34.5
    while tcur < 48.95:
        c = chord_at(tcur + 1e-3)
        if c:
            root = CHORDS[c][0]
            while root < 38:
                root += 12
            m = root + (12 if j % 4 == 2 else 0)
            tt2 = tvec(0.3)
            x = saw(midi(m), tt2) * env_ar(len(tt2), 0.003, 0.09)
            x = lp(x, 700 + 1500 * min(1, (tcur - 34.5) / 12))
            v = (0.20 + 0.20 * min(1, (tcur - 34.5) / 12)) * (1.0 if j % 4 == 0 else 0.7)
            low.add(pan(x, -0.25 if j % 2 else 0.25), tcur, v)
        tcur += t16
        j += 1

    # --- pulse kicks
    for b in np.arange(28.5, 34.5, 1.5):
        perc.add(pan(kick(0.5), 0), b, 0.5)
    for b in np.arange(34.5, 42.0, 0.75):
        perc.add(pan(kick(0.8), 0), b, 0.65)
    # accelerating tom roll into the climax
    times = []
    tcur = 42.0
    gap = 0.375
    while tcur < 44.2:
        times.append(tcur)
        tcur += gap
        gap = max(0.07, gap * 0.86)
    for i, tm in enumerate(times):
        perc.add(pan(tom(45 + (i % 3) * 3, 0.5 + 0.5 * i / len(times)), 0.4 * np.sin(i)), tm, 0.7)

    # --- whooshes on movement and cuts
    for tm, dur in [(10.2, 1.0), (19.9, 1.0), (28.2, 0.9), (37.2, 0.8), (39.4, 0.9), (45.45, 0.7)]:
        fxb.add(whoosh(dur), tm, 0.55)
    fxb.add(whoosh(2.3, 200, 6000, 0.75), 41.9, 0.8)       # hyperspace jump
    fxb.add(pan(riser(3.6, 150, 8000, (45, 57)), 0), 40.65, 0.6)
    fxb.add(pan(riser(1.75, 200, 9000, (45, 57, 64)), 0), 47.25, 0.65)
    # soft section booms
    for tm, a in [(15.0, 0.35), (24.0, 0.3), (34.5, 0.55), (37.5, 0.35)]:
        perc.add(pan(taiko(0.8, 0.9), 0), tm, a)

    # --- climax hits: SPEED / CONNECT / GLOBAL
    for tm, c in [(44.25, "Dm"), (45.75, "Bb"), (47.25, "Gm")]:
        perc.add(pan(taiko(1.5, 1.0), 0), tm, 1.0)
        perc.add(pan(taiko(1.0, 1.5), 0), tm + 0.012, 0.45)
        low.add(pan(subdrop(58, 30, 1.4), 0), tm, 0.7)
        fxb.add(impact_noise(1.5, 2000), tm, 0.55)
        pad.add(pan(braam(CHORDS[c][:3], 1.45, 3.0), 0), tm, 0.55)

    # --- the inhale and the LOCK
    perc.add(pan(taiko(2.0, 0.85), 0), 49.5, 1.1)
    perc.add(pan(taiko(1.2, 1.4), 0), 49.51, 0.5)
    low.add(pan(subdrop(66, 25, 3.5), 0), 49.5, 1.1)
    fxb.add(impact_noise(5.0, 900), 49.5, 1.0)
    pad.add(pan(braam([38, 45, 50, 54], 4.5, 2.6), 0), 49.5, 0.8)
    pad.add(pan(choir([62, 66, 69, 74], 9.5, 0.3, 2.0, seed=9), 0), 49.5, 0.5)
    pad.add(pan(organ(CHORDS["D"], 9.0, attack=0.25, release=1.4, bright=0.8, seed=99), 0), 49.5, 0.55)
    for m, tm, pn in [(86, 49.55, -0.4), (90, 49.7, 0.4), (93, 49.9, 0.0)]:
        fxb.add(pan(bell(m, 4.0, 0.5), pn), tm, 0.45)
    fxb.add(pan(bell(98, 3.0, 0.5), 0.2), 51.75, 0.55)     # logo light sweep
    fxb.add(pan(bell(86, 3.5, 0.5), -0.2), 55.3, 0.4)      # url
    return dict(pad=pad.x, arp=arp.x, perc=perc.x, fx=fxb.x, low=low.x)


def load_vo(lang="cn"):
    vo = np.zeros(N)
    meta = json.load(open(os.path.join(ROOT, "assets", "audio", "vo", lang, "meta.json")))
    for v in EDL["vo"]:
        sr, x = wavfile.read(os.path.join(ROOT, "assets", "audio", "vo", lang, v["id"] + ".wav"))
        x = x.astype(np.float64) / 32768.0 if x.dtype == np.int16 else x.astype(np.float64)
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            from math import gcd
            gg = gcd(SR, sr)
            x = signal.resample_poly(x, SR // gg, sr // gg)
        start = EDL.get("vo_start_" + lang, {}).get(v["id"], v["start"])
        i = int(start * SR)
        j = min(N, i + len(x))
        vo[i:j] += x[:j - i]
    return vo


def process_vo(vo):
    x = hp(vo, 75)
    # warmth + presence
    x = x + 0.25 * bp(x, 160, 1.2) + 0.30 * bp(x, 3800, 1.5)
    # compressor (RMS, 4:1 above -22 dBFS)
    env = np.sqrt(signal.lfilter([1 - 0.995], [1, -0.995], x ** 2) + 1e-12)
    db = 20 * np.log10(env)
    gr = np.where(db > -22, (db + 22) * (1 - 1 / 4.0), 0)
    x = x * 10 ** (-gr / 20)
    x = x / (np.abs(x).max() + 1e-9) * 0.9
    st = pan(x, 0.0) * np.sqrt(2)
    return reverb(st, reverb_ir(1.6, 0.02, 7000, seed=11), wet=0.12), x


def mix(lang="cn"):
    stems = build()
    ir_big = reverb_ir(4.2, 0.04, 6500)
    music = {}
    music["pad"] = reverb(stems["pad"], ir_big, 0.35) * 0.9
    music["arp"] = reverb(delay_pingpong(stems["arp"]), ir_big, 0.45) * 0.9
    music["perc"] = reverb(stems["perc"], reverb_ir(2.4, 0.01, 5000, seed=7), 0.22) * 1.0
    music["fx"] = reverb(stems["fx"], ir_big, 0.3) * 0.9
    music["low"] = lp(stems["low"], 900) * 1.0
    bus = sum(music.values())
    # the held breath before the logo lock: music drops away, only the inhale remains
    tt = np.arange(N) / SR
    hold = 1 - (1 - 0.025) * np.clip((tt - 48.93) / 0.07, 0, 1) * np.clip((49.5 - tt) / 0.012, 0, 1)
    bus *= hold[:, None]
    inhale = np.zeros((N, 2))
    sw = reverse_swell(0.35)
    i0 = int(49.15 * SR)
    inhale[i0:i0 + len(sw)] = sw
    bus += reverb(inhale, ir_big, 0.15) * 0.45
    vo_st, vo_dry = process_vo(load_vo(lang))
    # sidechain duck: music -5.5 dB under narration
    env = np.abs(vo_dry)
    env = signal.lfilter([1 - 0.9993], [1, -0.9993], env)
    env = env / (env.max() + 1e-9)
    duck = 1 - 0.47 * np.clip(env * 4, 0, 1)
    bus *= duck[:, None]
    out = bus * 0.55 + vo_st * 0.78
    fade = np.clip((DUR - np.arange(N) / SR) / 1.2, 0, 1)
    out *= fade[:, None]
    out = master(out)
    return out, music, vo_st


def true_peak_env(x, os_=4):
    up = signal.resample_poly(x, os_, 1, axis=0)
    return np.abs(up).max(1).reshape(-1, os_).max(1)[:len(x)]


def limiter(x, ceiling_db=-1.2, look_ms=3.0, release_ms=90.0):
    c = 10 ** (ceiling_db / 20)
    pk = true_peak_env(x)
    req = np.minimum(1.0, c / (pk + 1e-12))
    L = max(1, int(look_ms * SR / 1000))
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    g = minimum_filter1d(req, size=2 * L + 1, origin=-L // 2)
    g = uniform_filter1d(g, size=L)
    a = np.exp(-1.0 / (release_ms * SR / 1000))
    out = np.empty_like(g)
    cur = 1.0
    for i in range(0, len(g), 64):  # block-wise release smoothing (attack is instant)
        blk = g[i:i + 64]
        m = blk.min()
        cur = min(m, cur * a ** 64 + (1 - a ** 64) * 1.0) if m < cur else cur + (min(1.0, m) - cur) * (1 - a ** 64)
        out[i:i + 64] = np.minimum(blk, cur)
    return x * out[:, None]


def master(x, target_lufs=-14.5):
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    for _ in range(3):
        lufs = meter.integrated_loudness(x)
        x = x * 10 ** ((target_lufs - lufs) / 20)
        x = limiter(x)
    return x


def write(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "cn"
    out_dir = os.path.join(ROOT, "build", "audio")
    out, music, vo = mix(lang)
    write(os.path.join(out_dir, f"mix_{lang}.wav"), out)
    for k, v in music.items():
        write(os.path.join(out_dir, f"stem_{k}.wav"), v * 0.5)
    write(os.path.join(out_dir, f"stem_vo_{lang}.wav"), vo * 0.6)
    import pyloudnorm as pyln
    print("peak", np.abs(out).max(), "LUFS", pyln.Meter(SR).integrated_loudness(out),
          "TP dBTP", 20 * np.log10(true_peak_env(out).max()))
