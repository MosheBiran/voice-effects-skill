"""Layered sound-effect synthesis engine (numpy only).

A sound is described as a *preset*: a list of layers (oscillators, FM pairs,
noise, dust) with their own pitch sweeps, filters, envelopes and LFOs, plus a
master section (drive, EQ, reverb, echo, stereo width).

Every render takes a handful of macros:

    pitch      semitones, shifts every layer that tracks pitch
    depth      0..1, effect-specific: each layer's *role* decides what depth
               does to it (more sub, longer tail, lower body, bigger room)
    length     time multiplier
    tone       -1..1 dark/bright tilt
    drive      0..1 saturation
    space      0..1 reverb amount
    width      0..1.5 stereo width
    variation  0..1 random detune/decay/gain jitter (use with seed)
"""

import json
import math
import os
import wave

import numpy as np

SR = 48000
PRESET_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "presets"))


# ---------------------------------------------------------------- utilities

def _rng(seed, *salt):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, *[int(s) for s in salt]])


def db(x):
    return 10 ** (x / 20.0)


def sweep(f0, f1, ft, n, curve=1.0, sr=SR):
    """Per-sample curve from f0 to f1 over ft seconds (log interpolation), then held.

    curve < 1 moves fast at the start, curve > 1 accelerates towards the end.
    """
    if f1 is None or ft is None or ft <= 0 or f1 == f0:
        return np.full(n, float(f0))
    u = np.clip(np.arange(n) / sr / ft, 0.0, 1.0) ** float(curve)
    f0, f1 = max(f0, 1e-3), max(f1, 1e-3)
    return f0 * (f1 / f0) ** u


def spec_curve(spec, scale, T, n, total, default_f0=220.0):
    """Frequency curve of an oscillator or filter spec, already multiplied by `scale`.

    Either f0/f1/ft/fc (one sweep) or points: [[t_seconds, hz], ...] (log-interpolated).
    Times are stretched by the time multiplier T.
    """
    pts = spec.get("points")
    if pts:
        ts = np.array([p[0] for p in pts], float) * T * SR
        vs = np.log(np.maximum(np.array([p[1] for p in pts], float), 1e-3))
        return np.exp(np.interp(np.arange(n), ts, vs)) * scale
    f1 = spec.get("f1")
    return sweep(spec.get("f0", default_f0) * scale, f1 * scale if f1 is not None else None,
                 spec.get("ft", total / T) * T, n, spec.get("fc", 1.0))


def envelope(n, a, h, d, ac=1.0, dc=2.0, sr=SR):
    """Attack (power curve ac), hold, decay ((1-u)^dc)."""
    t = np.arange(n) / sr
    env = np.zeros(n)
    m = t < a
    env[m] = (t[m] / max(a, 1e-6)) ** ac
    m = (t >= a) & (t < a + h)
    env[m] = 1.0
    m = t >= a + h
    u = np.clip((t[m] - a - h) / max(d, 1e-6), 0.0, 1.0)
    env[m] = (1.0 - u) ** dc
    return env


def lfo_wave(n, spec, rng, sr=SR):
    """Unipolar LFO in 0..1. spec: rate, to (end rate), shape, smooth."""
    rate = float(spec.get("rate", 5.0))
    r = sweep(rate, spec.get("to"), n / sr, n, spec.get("curve", 1.0), sr)
    ph = np.cumsum(r) / sr + spec.get("phase", 0.0)
    shape = spec.get("shape", "sine")
    if shape == "sine":
        v = 0.5 + 0.5 * np.sin(2 * np.pi * ph)
    elif shape == "square":
        v = (ph % 1.0 < spec.get("duty", 0.5)).astype(float)
    elif shape == "saw":
        v = 1.0 - (ph % 1.0)
    elif shape == "random":
        vals = rng.random(int(ph[-1]) + 3 if n else 3)
        v = vals[np.floor(ph).astype(int)]
    else:
        raise ValueError(f"unknown lfo shape {shape}")
    sm = spec.get("smooth", 0.0 if shape != "square" else 0.002)
    if sm > 0:
        k = max(1, int(sm * sr))
        v = np.convolve(v, np.ones(k) / k, mode="same")
    return v


def _polyblep(t, dt):
    y = np.zeros_like(t)
    dt = np.minimum(dt, 0.5)
    m = t < dt
    x = t[m] / dt[m]
    y[m] = x + x - x * x - 1.0
    m = t > 1.0 - dt
    x = (t[m] - 1.0) / dt[m]
    y[m] = x * x + x + x + 1.0
    return y


def oscillator(freq, wave_name, phase0=0.0, sr=SR):
    dt = freq / sr
    ph = (np.cumsum(dt) + phase0) % 1.0
    if wave_name == "sine":
        return np.sin(2 * np.pi * ph)
    if wave_name == "triangle":
        return 2.0 * np.abs(2.0 * ph - 1.0) - 1.0
    if wave_name == "saw":
        return 2.0 * ph - 1.0 - _polyblep(ph, dt)
    if wave_name == "square":
        return np.where(ph < 0.5, 1.0, -1.0) + _polyblep(ph, dt) - _polyblep((ph + 0.5) % 1.0, dt)
    raise ValueError(f"unknown wave {wave_name}")


def colored_noise(n, color, rng, sr=SR):
    w = rng.standard_normal(n)
    if color == "white" or n < 4:
        return w / (w.std() + 1e-12)
    expo = {"pink": -0.5, "brown": -1.0, "blue": 0.5, "violet": 1.0}[color]
    W = np.fft.rfft(w)
    f = np.maximum(np.fft.rfftfreq(n, 1.0 / sr), 15.0)
    y = np.fft.irfft(W * f ** expo, n)
    return y / (y.std() + 1e-12)


def dust(n, density, rng, sr=SR):
    y = np.zeros(n)
    k = rng.poisson(max(density, 0.0) * n / sr)
    if k:
        pos = rng.integers(0, n, k)
        y[pos] = rng.choice([-1.0, 1.0], k) * rng.uniform(0.25, 1.0, k)
    return y


# ------------------------------------------------------------------ filters

def filter_mag(f, fc, kind, q=0.707, order=1):
    """Magnitude response of an analog 2-pole prototype (order 2 = 4-pole)."""
    w = f / np.maximum(fc, 1e-3)
    q = max(q, 0.05)
    den = np.sqrt((1.0 - w * w) ** 2 + (w / q) ** 2) + 1e-12
    if kind == "lowpass":
        m = 1.0 / den
    elif kind == "highpass":
        m = w * w / den
    elif kind == "bandpass":
        m = (w / q) / den
    elif kind == "notch":
        m = np.abs(1.0 - w * w) / den
    else:
        raise ValueError(f"unknown filter {kind}")
    return m ** order


def stft_process(x, mag_fn, n_fft=2048, sr=SR):
    """Time-varying spectral filter. mag_fn(freqs, frame_centres_s) -> (frames, bins)."""
    n = len(x)
    if n == 0:
        return x
    hop = n_fft // 4
    win = np.sqrt(np.hanning(n_fft + 1)[:-1])
    pad = np.concatenate([np.zeros(n_fft), x, np.zeros(n_fft)])
    starts = np.arange(0, len(pad) - n_fft + 1, hop)
    frames = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[starts] * win
    X = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    centres = (starts + n_fft / 2 - n_fft) / sr
    X *= mag_fn(freqs[None, :], centres[:, None])
    y_frames = np.fft.irfft(X, n_fft, axis=1) * win
    out = np.zeros(len(pad))
    norm = np.zeros(len(pad))
    w2 = win * win
    for i, s in enumerate(starts):
        out[s:s + n_fft] += y_frames[i]
        norm[s:s + n_fft] += w2
    out = out / np.maximum(norm, 1e-6)
    return out[n_fft:n_fft + n]


def static_eq(x, mag_fn, sr=SR):
    """Zero-phase static EQ on a (channels, n) or (n,) array via one big FFT."""
    n = x.shape[-1]
    if n < 4:
        return x
    nfft = 1 << int(math.ceil(math.log2(n + 1)))
    X = np.fft.rfft(x, nfft, axis=-1)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    return np.fft.irfft(X * mag_fn(f), nfft, axis=-1)[..., :n]


def shelf(f, fc, gain_db, high=False):
    s = (f / fc) ** 2 / (1.0 + (f / fc) ** 2)
    g = db(gain_db)
    return 1.0 + (g - 1.0) * s if high else g + (1.0 - g) * s


def fft_convolve(a, b):
    n = len(a) + len(b) - 1
    nfft = 1 << int(math.ceil(math.log2(n)))
    return np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)[:n]


# ------------------------------------------------------------------- reverb

def make_ir(size, damp, predelay, rng, sr=SR):
    """Stereo room impulse: early reflections + band-split exponential tail."""
    size = max(size, 0.05)
    n = int(size * 1.15 * sr)
    t = np.arange(n) / sr
    f = np.fft.rfftfreq(n, 1.0 / sr)
    lo = 1.0 / (1.0 + (f / 500.0) ** 4)
    hi = (f / 4000.0) ** 4 / (1.0 + (f / 4000.0) ** 4)
    mid = np.clip(1.0 - lo - hi, 0.0, 1.0)
    rts = (size * (1.0 + 0.15 * damp), size, size * (1.0 - 0.7 * damp))
    irs = []
    for ch in range(2):
        N = np.fft.rfft(rng.standard_normal(n))
        tail = np.zeros(n)
        for band, rt in zip((lo, mid, hi), rts):
            tail += np.fft.irfft(N * band, n) * np.exp(-6.91 * t / max(rt, 0.02))
        tail *= np.clip(t / 0.008, 0.0, 1.0)
        er = np.zeros(n)
        for _ in range(10):
            p = int(rng.uniform(0.004, 0.07) * sr)
            if p < n:
                er[p] += rng.uniform(-1, 1) * math.exp(-p / sr / 0.05)
        ir = tail / (np.sqrt(np.sum(tail ** 2)) + 1e-12) + 0.6 * er
        ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
        irs.append(np.concatenate([np.zeros(int(predelay * sr)), ir]))
    return irs


# ------------------------------------------------------------------- layers

def role_mods(role, D):
    """(gain, decay, freq) multipliers that make depth mean something per role."""
    c = D - 0.5
    if role == "sub":
        return 0.15 + 1.7 * D, 0.7 + 0.9 * D, 2 ** (-c * 0.4)
    if role == "body":
        return 1.0, 0.85 + 0.5 * D, 2 ** (-c * 0.35)
    if role == "transient":
        return 1.15 - 0.45 * D, 1.0, 1.0
    if role == "air":
        return 1.1 - 0.55 * D, 1.0, 1.0
    if role == "tail":
        return 0.3 + 1.3 * D, 0.6 + 1.2 * D, 2 ** (-c * 0.3)
    if role == "tone":
        return 1.0, 0.9 + 0.3 * D, 1.0
    return 1.0, 1.0, 1.0


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _pan_gains(p):
    ang = (np.clip(p, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return np.cos(ang), np.sin(ang)


def render_layer(L, M, idx):
    """Render one layer to a stereo array. Returns (delay_seconds, (2, n) array)."""
    rng = _rng(M["seed"], idx + 1)
    role = L.get("role", "body")
    gmul, dmul, fmul = role_mods(role, M["D"])
    var = M["var"]
    fj = 2 ** (rng.normal() * 0.07 * var)
    dj = max(0.5, 1.0 + rng.normal() * 0.18 * var)
    gj = db(rng.normal() * 1.5 * var)
    T = M["T"]
    track = L.get("track", 1.0)
    pr = (M["r"] ** track) * fmul * fj

    env = L.get("env", {})
    a = env.get("a", 0.005) * T
    h = env.get("h", 0.0) * T
    d = env.get("d", 0.5) * T * dmul * dj
    n = max(32, int(round((a + h + d) * SR)))
    typ = L.get("type", "osc")
    stereo_src = False

    if typ in ("osc", "fm"):
        f = spec_curve(L, pr, T, n, a + h + d)
        for spec in _as_list(L.get("lfo")):
            if spec.get("target") == "freq":
                f = f * 2 ** (spec.get("depth", 1.0) * (2 * lfo_wave(n, spec, rng) - 1) / 12.0)
        f = np.clip(f, 5.0, 0.45 * SR)
        voices = int(L.get("voices", 1))
        if typ == "fm":
            ratio = L.get("ratio", 2.0)
            index = L.get("index", 2.0)
            idecay = L.get("idecay")
            t = np.arange(n) / SR
            I = index * (np.exp(-t / (idecay * T)) if idecay else 1.0)
            pm = 2 * np.pi * np.cumsum(f * ratio) / SR
            src = np.sin(2 * np.pi * (np.cumsum(f) / SR + rng.random()) + I * np.sin(pm))
        elif voices <= 1:
            src = oscillator(f, L.get("wave", "sine"), rng.random())
        else:
            spread = L.get("detune", 20.0)
            cents = np.linspace(-spread / 2, spread / 2, voices)
            pans = np.linspace(-1.0, 1.0, voices) * L.get("spread", 0.8) * M["width"]
            src = np.zeros((2, n))
            for c, p in zip(cents, pans):
                v = oscillator(f * 2 ** (c / 1200.0), L.get("wave", "saw"), rng.random())
                gl, gr = _pan_gains(p)
                src[0] += v * gl
                src[1] += v * gr
            src /= math.sqrt(voices)
            stereo_src = True
    elif typ == "noise":
        color = L.get("color", "white")
        if L.get("stereo"):
            src = np.stack([colored_noise(n, color, rng), colored_noise(n, color, rng)]) * 0.5
            stereo_src = True
        else:
            src = colored_noise(n, color, rng) * 0.5
    elif typ == "dust":
        src = dust(n, L.get("density", 40.0), rng)
    else:
        raise ValueError(f"unknown layer type {typ}")

    # filters run on the raw source, before the amplitude envelope, so attacks stay sharp
    for F in _as_list(L.get("filter")):
        ftr = F.get("track", track)
        fr = (M["r"] ** ftr) * fmul * fj
        fcurve = spec_curve(F, fr, T, n, a + h + d, 1000.0)
        for spec in _as_list(F.get("lfo")):
            fcurve = fcurve * 2 ** (spec.get("depth", 1.0) * (2 * lfo_wave(n, spec, rng) - 1))
        fcurve = np.clip(fcurve, 10.0, 0.45 * SR)
        kind, q, order = F.get("type", "lowpass"), F.get("q", 0.707), F.get("order", 1)

        def mag(freqs, centres, fcurve=fcurve, kind=kind, q=q, order=order):
            idx_ = np.clip((centres * SR).astype(int), 0, n - 1)
            return filter_mag(freqs, fcurve[idx_], kind, q, order)

        before = math.sqrt(np.mean(src ** 2)) + 1e-12
        src = np.stack([stft_process(c, mag) for c in src]) if stereo_src else stft_process(src, mag)
        if L.get("normalize", True):
            src *= before / (math.sqrt(np.mean(src ** 2)) + 1e-12)

    amp = envelope(n, a, h, d, env.get("ac", 1.0), env.get("dc", 2.0))
    for spec in _as_list(L.get("lfo")):
        if spec.get("target", "amp") == "amp":
            amp = amp * (1.0 - spec.get("depth", 1.0) * (1.0 - lfo_wave(n, spec, rng)))
    src = src * amp

    drv = L.get("drive", 0.0)
    if drv > 0:
        k = 1.0 + 12.0 * drv
        src = np.tanh(k * src) / math.tanh(k)
    if L.get("crush"):
        steps = 2 ** int(L["crush"])
        src = np.round(src * steps) / steps
    if L.get("hold"):
        hs = max(1, int(L["hold"]))
        src = np.repeat(src[..., ::hs], hs, axis=-1)[..., :n]

    if stereo_src:
        out = src
    else:
        p0 = L.get("pan", 0.0) * M["width"]
        p1 = L.get("pan_to", L.get("pan", 0.0)) * M["width"]
        p = p0 + (p1 - p0) * np.linspace(0.0, 1.0, n)
        gl, gr = _pan_gains(p)
        out = np.stack([src * gl, src * gr])

    if L.get("reverse"):
        out = out[:, ::-1]

    rep = L.get("repeat")
    if rep:
        count = int(rep.get("count", 2))
        interval = rep.get("interval", 0.25) * T
        decay = rep.get("decay", 0.7)
        jit = rep.get("jitter", 0.0)
        offs = [0]
        for i in range(1, count):
            step = interval * (1.0 + rng.uniform(-jit, jit))
            offs.append(offs[-1] + int(step * SR))
        tot = np.zeros((2, offs[-1] + out.shape[1]))
        for i, o in enumerate(offs):
            tot[:, o:o + out.shape[1]] += out * decay ** i
        out = tot

    return L.get("delay", 0.0) * T, out * L.get("gain", 1.0) * gmul * gj


# ------------------------------------------------------------------ presets

def load_preset(name_or_path):
    path = name_or_path
    if not os.path.exists(path):
        path = os.path.join(PRESET_DIR, f"{name_or_path}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no preset '{name_or_path}' (see: sfx.py list)")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def list_presets():
    out = []
    for fn in sorted(os.listdir(PRESET_DIR)):
        if fn.endswith(".json"):
            with open(os.path.join(PRESET_DIR, fn), encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out


def resolve_macros(P, pitch=None, depth=None, length=None, tone=None, drive=None,
                   space=None, width=None, variation=None):
    pr = P.get("pitch", {})
    dp = P.get("depth", {})
    return {
        "pitch": float(pr.get("default", 0.0) if pitch is None else pitch),
        "depth": float(np.clip(dp.get("default", 0.5) if depth is None else depth, 0.0, 1.0)),
        "length": float(P.get("length", 1.0) if length is None else length),
        "tone": float(np.clip(P.get("tone", 0.0) if tone is None else tone, -1.0, 1.0)),
        "drive": float(np.clip(P.get("drive", 0.0) if drive is None else drive, 0.0, 1.0)),
        "space": float(np.clip(P.get("space", {}).get("mix", 0.2) if space is None else space, 0.0, 1.0)),
        "width": float(np.clip(P.get("width", 1.0) if width is None else width, 0.0, 1.5)),
        "variation": float(np.clip(0.0 if variation is None else variation, 0.0, 1.0)),
    }


# ------------------------------------------------------------------- render

def render(P, seed=0, mute=(), solo=None, normalize=True, **macros):
    """Render preset dict P. Returns ((2, n) float array, info dict)."""
    m = resolve_macros(P, **macros)
    D = m["depth"]
    dp = P.get("depth", {})
    c = 2.0 * D - 1.0
    T = m["length"] * dp.get("time", 1.15) ** c
    st = m["pitch"] + dp.get("pitch", -2.0) * c
    M = {"T": T, "r": 2 ** (st / 12.0), "D": D, "var": m["variation"], "seed": seed, "width": m["width"]}

    parts = []
    for i, L in enumerate(P["layers"]):
        tag = L.get("name", L.get("role", "body"))
        if tag in mute or L.get("role") in mute:
            continue
        if solo and tag not in solo and L.get("role") not in solo:
            continue
        dl, y = render_layer(L, M, i)
        parts.append([int(dl * SR), y, bool(L.get("reverse")) and L.get("align", "end") == "end"])
    if not parts:
        raise ValueError("every layer is muted")
    # reversed layers swell *into* a point: line their ends up instead of their starts
    rev_end = max((o + y.shape[1] for o, y, al in parts if al), default=None)
    if rev_end is not None:
        for p in parts:
            if p[2]:
                p[0] = rev_end - p[1].shape[1]
    n = max(o + y.shape[1] for o, y, _ in parts)

    # master: drive -> EQ -> reverb -> echo -> width
    sp = P.get("space", {})
    size = sp.get("size", 1.2) * (0.5 + D) * min(T, 2.0) ** 0.3
    wet = min(0.9, m["space"] * (0.6 + 0.8 * D))
    tail = int((size * 1.15 + 0.05) * SR) if wet > 0.001 else int(0.05 * SR)
    x = np.zeros((2, n + tail))
    for o, y, _ in parts:
        x[:, o:o + y.shape[1]] += y

    if m["drive"] > 0:
        k = 1.0 + 8.0 * m["drive"]
        pk = np.max(np.abs(x)) + 1e-12
        x = np.tanh(k * x / pk) / math.tanh(k) * pk

    low_db = dp.get("low_db", 10.0) * (D - 0.3)
    tone = m["tone"]
    hp = P.get("highpass", 22.0)

    def eq(f):
        g = shelf(f, 110.0, low_db) * shelf(f, 4000.0, 10.0 * tone, high=True) * shelf(f, 250.0, -3.0 * tone)
        return g * filter_mag(f, hp, "highpass", 0.707, 2)

    x = static_eq(x, eq)

    if wet > 0.001:
        irs = make_ir(size, sp.get("damp", 0.5), 0.004 + 0.03 * D, _rng(seed, 999))
        dry = x
        rev = np.stack([fft_convolve(dry[ch], irs[ch])[: x.shape[1]] for ch in range(2)])
        rms_d = math.sqrt(np.mean(dry ** 2)) + 1e-12
        rms_r = math.sqrt(np.mean(rev ** 2)) + 1e-12
        x = dry * (1.0 - 0.5 * wet) + rev * (rms_d / rms_r) * wet * 1.2

    echo = P.get("echo")
    if echo and echo.get("mix", 0) > 0:
        dt = int(echo.get("time", 0.15) * T * SR)
        fb = echo.get("fb", 0.4)
        add = int(dt * math.log(0.001) / math.log(max(fb, 1e-3))) if fb > 0 else dt
        y = np.concatenate([x, np.zeros((2, add + dt))], axis=1)
        out = y.copy()
        g, k = echo["mix"], 1
        while g > 0.001 and k * dt < y.shape[1]:
            sh = np.zeros_like(y)
            sh[:, k * dt:] = y[:, : y.shape[1] - k * dt]
            if echo.get("pingpong", True) and k % 2:
                sh = sh[::-1]
            out += sh * g
            g *= fb
            k += 1
        x = out

    w = m["width"]
    mid = 0.5 * (x[0] + x[1])
    side = 0.5 * (x[0] - x[1]) * w
    x = np.stack([mid + side, mid - side])

    # trim silence, fade, normalise
    mono = np.max(np.abs(x), axis=0)
    pk = mono.max() + 1e-12
    above = np.nonzero(mono > pk * db(-70))[0]
    first = above[0] if len(above) else 0
    last = above[-1] if len(above) else x.shape[1] - 1
    first = max(0, first - int(0.002 * SR))
    x = x[:, first: min(x.shape[1], last + int(0.03 * SR))]
    fade = min(int(0.02 * SR), x.shape[1] // 4)
    if fade > 0:
        x[:, -fade:] *= np.linspace(1.0, 0.0, fade) ** 2
    if normalize:
        x *= db(-1.0) / (np.max(np.abs(x)) + 1e-12)

    info = {
        "preset": P.get("id", "custom"),
        "macros": m,
        "seed": seed,
        "duration": round(x.shape[1] / SR, 4),
        "sync": round(sync_point(x, P.get("sync", "onset")), 4),
        "sync_mode": P.get("sync", "onset"),
    }
    return x, info


def sync_point(x, mode="onset", sr=SR):
    """Where the effect lands on the picture.

    onset: first strong transient (impacts, clicks)
    peak:  loudest moment (whoosh centre, end of a riser)
    hit:   start of the biggest sudden jump in level (the impact inside a riser-into-impact)
    """
    mono = np.mean(np.abs(x), axis=0)
    k = max(1, int(0.01 * sr))
    env = np.sqrt(np.convolve(mono ** 2, np.ones(k) / k, mode="same"))
    if mode == "peak":
        return float(np.argmax(env)) / sr
    if mode == "hit":
        jump = np.zeros_like(env)
        jump[k:] = env[k:] - env[:-k]
        return max(0.0, (float(np.argmax(jump)) - k / 2) / sr)
    thr = 0.25 * env.max()
    idx = np.nonzero(env > thr)[0]
    return float(idx[0]) / sr if len(idx) else 0.0


# ---------------------------------------------------------------------- I/O

def write_wav(path, x, sr=SR):
    """24-bit PCM WAV. x: (channels, n) or (n,)."""
    x = np.atleast_2d(x)
    y = np.clip(x.T, -1.0, 1.0)
    i = np.round(y * 8388607.0).astype("<i4")
    b = i.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(x.shape[0])
        wf.setsampwidth(3)
        wf.setframerate(sr)
        wf.writeframes(b)


def read_wav(path):
    """Read a PCM WAV (16/24/32-bit). Returns ((channels, n) float array, sr)."""
    with wave.open(path, "rb") as wf:
        ch, sw, sr, n = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(n)
    if sw == 2:
        y = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        i = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        i = np.where(i >= 1 << 23, i - (1 << 24), i)
        y = i / 8388608.0
    elif sw == 4:
        y = np.frombuffer(raw, "<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    return y.reshape(-1, ch).T, sr


def resample(x, sr_in, sr_out):
    """FFT resampling along the last axis."""
    if sr_in == sr_out:
        return x
    n = x.shape[-1]
    m = int(round(n * sr_out / sr_in))
    X = np.fft.rfft(x, axis=-1)
    keep = m // 2 + 1
    if X.shape[-1] >= keep:
        X = X[..., :keep]
    else:
        X = np.concatenate([X, np.zeros(X.shape[:-1] + (keep - X.shape[-1],), complex)], axis=-1)
    return np.fft.irfft(X, m, axis=-1) * (m / n)
