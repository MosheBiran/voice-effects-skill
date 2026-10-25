"""Synthetic background beds (voice-over-like speech, music, room noise) for classifier training.

They are not meant to sound real; they reproduce what matters to the detector:
syllable-rate harmonic bursts for speech, a regular beat plus chords for music.
"""

import numpy as np

import sfx_engine as E
from analysis import ASR

VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480),
          (570, 840, 2410), (440, 1020, 2240), (660, 1720, 2410), (490, 1350, 1690)]


def _formants(x, f123, rng):
    q = [f / rng.uniform(70, 130) for f in f123]
    g = (1.0, 0.55, 0.3)

    def mag(freqs, centres):
        m = sum(gi * E.filter_mag(freqs, fi, "bandpass", qi) for fi, qi, gi in zip(f123, q, g))
        return m * np.ones_like(centres)

    return E.stft_process(x, mag, n_fft=512, sr=ASR)


def speech(dur, rng, sr=ASR):
    n = int(dur * sr)
    out = np.zeros(n)
    base = rng.uniform(95, 220)
    t = rng.uniform(0.0, 0.3)
    while t < dur - 0.25:
        syl = rng.uniform(0.11, 0.3)
        m = int(syl * sr)
        f0 = base * 2 ** (rng.normal(0, 0.12)) * 2 ** (rng.uniform(-0.15, 0.15) * np.linspace(0, 1, m))
        src = E.oscillator(f0, "saw", rng.random(), sr=sr)
        voiced = _formants(src, VOWELS[rng.integers(len(VOWELS))], rng)
        k = np.arange(m) / m
        env = np.minimum(1.0, k / 0.15) * np.minimum(1.0, (1 - k) / 0.3) ** 1.5
        seg = voiced / (voiced.std() + 1e-9) * env * rng.uniform(0.5, 1.0)
        o = int(t * sr)
        if rng.random() < 0.5:
            c = int(rng.uniform(0.03, 0.08) * sr)
            cons = E.colored_noise(c, "white", rng, sr)
            cons = E.stft_process(cons, lambda f, ctr: E.filter_mag(f, rng.uniform(2500, 5000), "highpass", 0.7)
                                  * np.ones_like(ctr), n_fft=256, sr=sr)
            cons *= np.hanning(c) * rng.uniform(0.15, 0.4)
            out[max(0, o - c):o] += cons[: o - max(0, o - c)]
        out[o:o + m] += seg[: n - o]
        t += syl + (rng.uniform(0.25, 0.6) if rng.random() < 0.15 else rng.uniform(0.02, 0.12))
    return out


def music(dur, rng, sr=ASR, drums=True):
    n = int(dur * sr)
    out = np.zeros(n)
    beat = 60.0 / rng.uniform(80, 140)
    root = 110.0 * 2 ** (rng.integers(-5, 6) / 12)
    scale = [0, 2, 3, 5, 7, 8, 10]
    t_bar = 0.0
    while t_bar < dur:
        degs = rng.choice(len(scale), 3, replace=False)
        m = int(min(4 * beat, dur - t_bar) * sr)
        if m <= 0:
            break
        pad = np.zeros(m)
        for dgi in degs:
            f = np.full(m, root * 2 ** (scale[dgi] / 12))
            pad += E.oscillator(f, "saw", rng.random(), sr=sr)
        pad = E.stft_process(pad, lambda f, c: E.filter_mag(f, 1400, "lowpass", 0.7) * np.ones_like(c), n_fft=1024, sr=sr)
        k = np.arange(m) / m
        pad *= np.minimum(1, k / 0.05) * np.minimum(1, (1 - k) / 0.05) * 0.12
        o = int(t_bar * sr)
        out[o:o + m] += pad
        t_bar += 4 * beat
    step = beat / 2
    i = 0
    while i * step < dur:
        o = int(i * step * sr)
        if rng.random() < 0.6:
            m = int(0.25 * sr)
            f = np.full(m, root * 4 * 2 ** (scale[rng.integers(len(scale))] / 12))
            pl = E.oscillator(f, "triangle", 0.0, sr=sr) * np.exp(-np.arange(m) / sr / 0.06) * 0.15
            out[o:o + m] += pl[: n - o]
        if drums:
            if i % 4 == 0:
                m = int(0.3 * sr)
                f = E.sweep(115, 48, 0.06, m, 1.0, sr)
                kick = E.oscillator(f, "sine", 0.0, sr=sr) * np.exp(-np.arange(m) / sr / 0.08) * 0.8
                out[o:o + m] += kick[: n - o]
            if i % 4 == 2:
                m = int(0.18 * sr)
                sn = E.colored_noise(m, "white", rng, sr) * np.exp(-np.arange(m) / sr / 0.05) * 0.3
                out[o:o + m] += sn[: n - o]
            m = int(0.05 * sr)
            hat = E.colored_noise(m, "violet", rng, sr) * np.exp(-np.arange(m) / sr / 0.012) * 0.12
            out[o:o + m] += hat[: n - o]
        i += 1
    return out


def room(dur, rng, sr=ASR):
    return E.colored_noise(int(dur * sr), "pink", rng, sr) * 0.05


def make(kind, dur, rng, sr=ASR):
    if kind == "speech":
        return speech(dur, rng, sr)
    if kind == "music":
        return music(dur, rng, sr, drums=rng.random() < 0.8)
    if kind == "speech+music":
        return speech(dur, rng, sr) + music(dur, rng, sr) * 0.35
    if kind == "room":
        return room(dur, rng, sr)
    return E.colored_noise(int(dur * sr), "white", rng, sr) * 1e-4
