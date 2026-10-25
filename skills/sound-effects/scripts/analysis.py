"""Find and describe sound effects inside finished videos.

Pipeline for one video:
  1. audio -> mono 22.05 kHz, STFT power
  2. background model per band (rolling low percentile over a few seconds) so the
     music/voice bed is subtracted and short foreground sounds stand out
  3. foreground regions -> candidate events (start / peak / end)
  4. 24 features per event (envelope shape, band balance, sweep direction,
     noisiness, harmonicity ...) -> k-nearest-neighbour classifier trained on
     this skill's own synthesized effects plus every event a person confirmed
  5. video context: scene cuts (ffmpeg scene score), offset of each event to the
     nearest cut, position in the video, frames around the event for review
"""

import json
import math
import os
import re
import shutil
import subprocess

import numpy as np

import sfx_engine as E

ASR = 22050
N_FFT = 1024
HOP = 256
FPS = ASR / HOP
FREQS = np.fft.rfftfreq(N_FFT, 1.0 / ASR)
NON_SFX = ("speech", "music", "other")
FEATURE_NAMES = [
    "log_dur", "attack_frac", "rise_corr", "peaks_per_s", "env_jitter",
    "band_sub", "band_low", "band_lowmid", "band_mid", "band_highmid", "band_presence", "band_high", "band_air",
    "flatness", "centroid", "centroid_delta", "centroid_arch", "low_delta", "harmonicity", "tonality",
    "contrast", "transient", "decay_ratio", "bandwidth",
]
FEATURE_WEIGHTS = np.array([1.4, 1.0, 1.2, 0.8, 0.8,
                            1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0,
                            1.2, 1.2, 1.3, 1.1, 1.1, 1.2, 1.0,
                            0.4, 1.0, 1.0, 0.8])
_BAND_EDGES = [20, 80, 160, 320, 640, 1280, 2560, 5120, 11025]


# -------------------------------------------------------------------- media

def have_ffmpeg():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe(path):
    info = {"duration": 0.0, "has_video": False, "has_audio": False, "fps": 30.0}
    if not have_ffmpeg():
        x, sr = E.read_wav(path)
        info.update(duration=x.shape[1] / sr, has_audio=True)
        return info
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    info["duration"] = float(j.get("format", {}).get("duration", 0) or 0)
    for s in j.get("streams", []):
        if s.get("codec_type") == "video" and not info["has_video"] and not s.get("disposition", {}).get("attached_pic"):
            num, den = (s.get("avg_frame_rate") or "30/1").split("/")
            fps = float(num) / float(den) if float(den or 0) else 30.0
            info.update(has_video=True, width=int(s["width"]), height=int(s["height"]), fps=fps if fps > 0 else 30.0)
        elif s.get("codec_type") == "audio":
            info["has_audio"] = True
    return info


def load_audio(path, sr=ASR):
    if have_ffmpeg():
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(out, "<f4").astype(np.float64)
    x, sr_in = E.read_wav(path)
    return E.resample(x.mean(axis=0), sr_in, sr)


def scene_cuts(path, duration=None):
    """Scene-change scores per frame and the cut times picked from them."""
    if not have_ffmpeg():
        return [], [], []
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-i", path, "-an", "-vf",
           "scale=192:-2,select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file=-", "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    times, scores = [], []
    t = None
    for line in out.splitlines():
        m = re.search(r"pts_time:([0-9.eE+-]+)", line)
        if m:
            t = float(m.group(1))
            continue
        m = re.search(r"scene_score=([0-9.]+)", line)
        if m and t is not None:
            times.append(t)
            scores.append(float(m.group(1)))
    times, scores = np.array(times), np.array(scores)
    cuts = []
    for i in np.argsort(-scores):
        s = scores[i]
        if s < 0.12:
            break
        near = np.abs(times - times[i]) <= 0.5
        local = np.median(scores[near]) if near.any() else 0.0
        if s >= 0.25 or s >= 6 * local + 0.08:
            if all(abs(times[i] - c) > 0.2 for c, _ in cuts):
                cuts.append((float(times[i]), float(s)))
    cuts.sort()
    return [round(c, 3) for c, _ in cuts], [round(s, 3) for _, s in cuts], (times, scores)


def frame_size(info, max_w=360, max_h=320):
    w, h = info.get("width", 16), info.get("height", 9)
    k = min(max_w / w, max_h / h)
    return max(2, int(w * k) // 2 * 2), max(2, int(h * k) // 2 * 2)


def grab_frame(path, t, size):
    w, h = size
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(t, 0.0):.3f}", "-i", path, "-frames:v", "1",
                          "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    if len(out) < w * h * 3:
        return np.zeros((h, w, 3), np.uint8) + 40
    return np.frombuffer(out[:w * h * 3], np.uint8).reshape(h, w, 3).copy()


# ---------------------------------------------------------------- spectrum

def stft_power(x):
    """(frames, bins) float32 power spectrum, computed in chunks to bound memory."""
    pad = np.concatenate([np.zeros(N_FFT // 2), x, np.zeros(N_FFT // 2 + HOP)])
    nfr = max(1, 1 + (len(pad) - N_FFT) // HOP)
    win = np.hanning(N_FFT)
    view = np.lib.stride_tricks.sliding_window_view(pad, N_FFT)
    P = np.empty((nfr, N_FFT // 2 + 1), np.float32)
    for s in range(0, nfr, 2048):
        idx = np.arange(s, min(nfr, s + 2048)) * HOP
        P[s:s + len(idx)] = np.abs(np.fft.rfft(view[idx] * win, axis=1)) ** 2
    return P


def _band_matrix(n_bands=24, fmin=40.0, fmax=10500.0):
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    B = np.zeros((len(FREQS), n_bands), np.float32)
    for b in range(n_bands):
        m = (FREQS >= edges[b]) & (FREQS < edges[b + 1])
        if not m.any():
            m[np.argmin(np.abs(FREQS - math.sqrt(edges[b] * edges[b + 1])))] = True
        B[m, b] = 1.0
    return B, np.sqrt(edges[:-1] * edges[1:])


BANDS, BAND_CENTRES = _band_matrix()
_BAND_TO_BIN = np.stack([np.interp(np.log(np.maximum(FREQS, 1.0)), np.log(BAND_CENTRES), row)
                         for row in np.eye(len(BAND_CENTRES))])


def background(Edb, window_s=4.0, pct=25, dec=4):
    """Rolling low percentile per band (frames, bands), computed on a decimated grid."""
    d = Edb[::dec]
    W = max(3, int(window_s * FPS / dec)) | 1
    padded = np.pad(d, ((W // 2, W // 2), (0, 0)), mode="reflect" if len(d) > W // 2 else "edge")
    bg = np.empty_like(d)
    for s in range(0, len(d), 4096):
        v = np.lib.stride_tricks.sliding_window_view(padded[s:s + 4096 + W - 1], W, axis=0)
        bg[s:s + v.shape[0]] = np.percentile(v, pct, axis=-1)
    idx = np.minimum(np.arange(len(Edb)) / dec, len(d) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(d) - 1)
    fr = (idx - lo)[:, None]
    return bg[lo] * (1 - fr) + bg[hi] * fr


class Analysis:
    """Spectral state of one audio track, shared by detection and feature extraction."""

    def __init__(self, x):
        self.x = x
        self.duration = len(x) / ASR
        self.P = stft_power(x)
        self.E = self.P @ BANDS + 1e-10
        self.Edb = 10 * np.log10(self.E)
        self.bg_db = background(self.Edb)
        # per-bin background power density, log-interpolated between band centres
        log_dens = self.bg_db * (math.log(10) / 10) - np.log(BANDS.sum(axis=0))
        self.bgP = np.exp(log_dens @ _BAND_TO_BIN).astype(np.float32)
        fg = np.clip(self.Edb - self.bg_db - 4.0, 0.0, None)
        loud = self.Edb.max(axis=1)
        gate = loud > (np.percentile(loud, 99.5) - 55.0)
        top = np.sort(fg, axis=1)[:, -6:].mean(axis=1) * gate
        self.novelty = np.convolve(top, np.ones(3) / 3, mode="same")
        # broadband onset strength: rise in band levels, ignoring bands near the noise floor
        floored = np.maximum(self.Edb, self.Edb.max() - 60.0)
        prev = np.vstack([np.full((1, floored.shape[1]), self.Edb.max() - 60.0), floored[:-1]])
        flux = np.sum(np.clip(floored - prev, 0, None), axis=1) / 24.0
        self.flux = flux

    def fg_power(self, s, e):
        return np.clip(self.P[s:e] - 1.5 * self.bgP[s:e], 0.0, None) + 1e-12


def detect_events(A, sensitivity=1.0, max_dur=8.0):
    """Foreground regions of the novelty curve -> [{'start','peak','end','contrast'}] in seconds."""
    N = A.novelty
    flux = A.flux
    if len(N) < 4:
        return []
    # N is already "dB above the bed", so ask for a fixed contrast over the quiet parts
    hi = max(8.0, float(np.percentile(N, 20)) + 8.0) / sensitivity
    lo = hi * 0.5
    above = N > lo
    events = []
    i = 0
    while i < len(N):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(N) and above[j]:
            j += 1
        if N[i:j].max() >= hi:
            events.append([i, j])
        i = j
    merged = []
    gap = int(0.1 * FPS)
    for s, e in events:
        if merged and s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    pieces = []
    for s, e in merged:
        pieces += _split_region(N, s, e)
    out = []
    for s, e in pieces:
        if e - s < 3:
            continue
        e = min(e, s + int(max_dur * FPS))
        p = s + int(np.argmax(N[s:e]))
        rise = np.nonzero(N[s:p + 1] >= max(hi, 0.5 * N[p]))[0]
        o = s + int(rise[0]) if len(rise) else p
        h = s + int(np.argmax(np.convolve(flux[s:e], [0.5, 0.5], mode="same")))
        out.append({"start": round(s / FPS, 3), "onset": round(o / FPS, 3), "hit": round(h / FPS, 3),
                    "peak": round(p / FPS, 3), "end": round(e / FPS, 3), "contrast": round(float(N[s:e].max()), 2)})
    return out


def _split_region(N, s, e, rise_db=12.0, drop_db=6.0):
    """Split a region where a new sound rises well above the decayed tail of the previous one."""
    W = int(0.3 * FPS)
    min_len = int(0.15 * FPS)
    segs, cur, t = [], s, s + min_len
    while t < e:
        w0 = max(cur, t - W)
        v = w0 + int(np.argmin(N[w0:t]))
        if (N[t] - N[v] >= rise_db and v - cur >= min_len and e - v >= min_len
                and N[cur:v].max() - N[v] >= drop_db):
            segs.append((cur, v))
            cur, t = v, v + min_len
            continue
        t += 1
    segs.append((cur, e))
    return segs


def _wmean(v, w):
    return float(np.sum(v * w) / (np.sum(w) + 1e-12))


def event_features(A, ev):
    """24-dim feature vector (see FEATURE_NAMES) for an event dict with start/end in seconds."""
    s = max(0, int(ev["start"] * FPS))
    e = max(s + 2, int(ev["end"] * FPS))
    e = min(e, len(A.P))
    s = min(s, e - 2)
    P = A.fg_power(s, e).astype(np.float64)
    n = P.shape[0]
    dur = n / FPS
    tot = P.sum(axis=1)
    edb = 10 * np.log10(tot + 1e-12)
    pk = int(np.argmax(edb))
    t = np.arange(n)
    rise = float(np.corrcoef(t, edb)[0, 1]) if n >= 3 and edb.std() > 1e-6 else 0.0
    sm = np.convolve(edb, np.ones(2) / 2, mode="same")
    peaks = 0
    for k in range(1, n - 1):
        if sm[k] >= sm[k - 1] and sm[k] > sm[k + 1]:
            left = sm[max(0, k - 6):k].min() if k > 0 else sm[k]
            right = sm[k + 1:k + 7].min()
            if sm[k] - max(left, right) > 6.0:
                peaks += 1
    jitter = float(np.std(np.diff(edb))) / 10.0 if n > 2 else 0.0

    S = P.sum(axis=0)
    bands = []
    for lo, hi in zip(_BAND_EDGES[:-1], _BAND_EDGES[1:]):
        m = (FREQS >= lo) & (FREQS < hi)
        bands.append(math.log10(S[m].sum() / (S.sum() + 1e-12) + 1e-3))

    mfl = (FREQS >= 150) & (FREQS <= 8000)
    Pm = P[:, mfl]
    flat_t = np.exp(np.mean(np.log(Pm), axis=1)) / (np.mean(Pm, axis=1) + 1e-12)
    flatness = _wmean(flat_t, tot)

    mc = (FREQS >= 50) & (FREQS <= 10000)
    cent_t = (P[:, mc] @ FREQS[mc]) / (P[:, mc].sum(axis=1) + 1e-12)
    q = max(1, n // 4)
    c_start = _wmean(cent_t[:q], tot[:q]) + 1.0
    c_end = _wmean(cent_t[-q:], tot[-q:]) + 1.0
    mid = slice(max(0, n // 2 - q // 2), max(1, n // 2 + (q + 1) // 2))
    c_mid = _wmean(cent_t[mid], tot[mid]) + 1.0
    c_mean = _wmean(cent_t, tot) + 1.0

    ml = (FREQS >= 30) & (FREQS <= 400)
    lc_t = (P[:, ml] @ FREQS[ml]) / (P[:, ml].sum(axis=1) + 1e-12)
    lw = P[:, ml].sum(axis=1)
    low_delta = math.log2((_wmean(lc_t[-q:], lw[-q:]) + 1) / (_wmean(lc_t[:q], lw[:q]) + 1))

    ac = np.fft.irfft(P, axis=1)
    lag_lo, lag_hi = int(ASR / 500), int(ASR / 70)
    harm_t = ac[:, lag_lo:lag_hi].max(axis=1) / (ac[:, 0] + 1e-12)
    harmonicity = _wmean(np.clip(harm_t, 0, 1), tot)

    mt = (FREQS >= 300) & (FREQS <= 10000)
    ton_t = np.log10(P[:, mt].max(axis=1) / (P[:, mt].mean(axis=1) + 1e-12) + 1e-12)
    tonality = _wmean(ton_t, tot)

    transient = float(A.flux[s:min(e, s + 4)].max()) / 3.0
    half = max(1, n // 2)
    decay_ratio = math.log10((tot[half:].sum() + 1e-12) / (tot[:half].sum() + 1e-12))
    lf = np.log2(np.maximum(FREQS[mc], 1.0))
    w = S[mc] / (S[mc].sum() + 1e-12)
    mu = np.sum(lf * w)
    bandwidth = math.sqrt(max(0.0, np.sum(w * (lf - mu) ** 2)))

    f = [math.log(max(dur, 1e-3)), pk / max(n - 1, 1), rise, math.log1p(peaks / max(dur, 1e-3)), jitter,
         *bands, flatness, math.log2(c_mean / 1000.0), math.log2(c_end / c_start), math.log2(c_mid / math.sqrt(c_start * c_end)),
         low_delta, harmonicity, tonality, ev.get("contrast", 10.0) / 20.0, transient, decay_ratio, bandwidth]
    return np.nan_to_num(np.array(f, float), nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------- classifier

def families():
    fam = {p["id"]: p.get("family", p["id"]) for p in E.list_presets()}
    for k in NON_SFX:
        fam[k] = k
    return fam


def family_of(label, fam=None):
    fam = fam or families()
    if label in fam:
        return fam[label]
    if label in set(fam.values()):
        return label
    return "other"


class Model:
    """Distance-weighted kNN over standardized, weighted features, plus a per-effect ridge
    regression that estimates the pitch/depth/length an event was made with (rough: ~3 st)."""

    def __init__(self, X, y, w, est, mu=None, sd=None, reg=None):
        self.X, self.y, self.w, self.est = np.asarray(X, float), np.asarray(y), np.asarray(w, float), np.asarray(est, float)
        self.mu = self.X.mean(axis=0) if mu is None else mu
        self.sd = (self.X.std(axis=0) + 1e-6) if sd is None else sd
        self.Z = (self.X - self.mu) / self.sd * FEATURE_WEIGHTS
        self.reg = self._fit_reg() if reg is None else reg

    def _fit_reg(self, lam=3.0):
        reg = {}
        for lab in np.unique(self.y):
            idx = np.nonzero((self.y == lab) & ~np.isnan(self.est).any(axis=1))[0]
            if len(idx) < 8:
                continue
            A = np.hstack([self.Z[idx], np.ones((len(idx), 1))])
            reg[str(lab)] = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ self.est[idx])
        return reg

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        reg = dict(zip(d["reg_labels"], d["reg_W"])) if "reg_labels" in d else None
        return cls(d["X"], d["y"], d["w"], d["est"], d["mu"], d["sd"], reg)

    def save(self, path):
        labs = sorted(self.reg)
        np.savez_compressed(path, X=self.X, y=self.y.astype("U32"), w=self.w, est=self.est, mu=self.mu, sd=self.sd,
                            reg_labels=np.array(labs, dtype="U32"),
                            reg_W=np.array([self.reg[k] for k in labs]) if labs else np.zeros((0, 25, 3)))

    def estimate(self, f, label):
        W = self.reg.get(str(label))
        if W is None:
            return None
        z = (np.asarray(f, float) - self.mu) / self.sd * FEATURE_WEIGHTS
        p, dpt, ln = np.append(z, 1.0) @ W
        return {"pitch": round(float(np.clip(p, -24, 24)), 1), "depth": round(float(np.clip(dpt, 0, 1)), 2),
                "length": round(float(np.clip(ln, 0.3, 3.0)), 2)}

    def predict(self, f, k=9):
        z = (np.asarray(f, float) - self.mu) / self.sd * FEATURE_WEIGHTS
        d = np.sqrt(np.sum((self.Z - z) ** 2, axis=1))
        nn = np.argsort(d)[:k]
        wt = self.w[nn] / (d[nn] + 0.5)
        probs = {}
        for i, wi in zip(nn, wt):
            probs[self.y[i]] = probs.get(self.y[i], 0.0) + wi
        tot = sum(probs.values()) + 1e-12
        ranked = sorted(((lab, p / tot) for lab, p in probs.items()), key=lambda kv: -kv[1])
        return ranked, self.estimate(f, ranked[0][0]), float(d[nn[0]])


# ------------------------------------------------------------------ context

ANCHOR = {"whoosh": "peak", "riser": "peak", "impact": "hit", "sub_drop": "hit"}  # everything else: onset


def is_likely(ev, min_conf=0.4):
    """Worth a human/Claude look: confirmed effects, or confident non-speech/music detections."""
    if ev.get("status") == "rejected" or ev.get("family") in NON_SFX:
        return False
    if ev.get("status") == "confirmed":
        return True
    return bool(ev.get("confidence", 0) >= min_conf and "repeats_on_grid" not in ev.get("flags", []))


def anchor_name(family):
    return ANCHOR.get(family, "onset")


def anchor_time(ev, family):
    return ev.get(anchor_name(family), ev["start"])


def add_context(ev, cuts, duration, family):
    t = anchor_time(ev, family)
    ctx = {"anchor": anchor_name(family), "anchor_time": t,
           "position": round(t / max(duration, 1e-6), 3), "in_hook": t < 2.0, "in_outro": t > duration - 3.0}
    if cuts:
        c = min(cuts, key=lambda c: abs(c - t))
        ctx.update(nearest_cut=c, offset_to_cut=round(t - c, 3), at_cut=abs(t - c) <= 0.3)
    else:
        ctx.update(nearest_cut=None, offset_to_cut=None, at_cut=False)
    ev["context"] = ctx
    return ev


def flag_repeats(events, model):
    """Mark near-identical events that repeat on a regular grid (likely a music beat)."""
    if model is None or len(events) < 4:
        return
    Z = np.array([(np.array(ev["features"]) - model.mu) / model.sd * FEATURE_WEIGHTS for ev in events])
    for i, ev in enumerate(events):
        d = np.sqrt(np.sum((Z - Z[i]) ** 2, axis=1))
        grp = sorted(events[j]["peak"] for j in np.nonzero(d < 2.0)[0])
        if len(grp) >= 4:
            ioi = np.diff(grp)
            base = np.median(ioi)
            if base > 0.15:
                ratio = ioi / base
                regular = np.mean(np.abs(ratio - np.round(ratio)) < 0.1)
                if regular >= 0.75:
                    ev["flags"] = sorted(set(ev.get("flags", [])) | {"repeats_on_grid"})


# ------------------------------------------------------------------- images

def overview_png(A, path, title, cuts, events):
    import viz
    return viz.overview(A.x, ASR, path, title, cuts=cuts, events=events)


def event_sheet_png(A, video, info, ev, path, title):
    import viz
    t = ev.get("context", {}).get("anchor_time", ev["peak"])
    frames, labels = [], []
    if video and info.get("has_video") and have_ffmpeg():
        size = frame_size(info)
        for dt, lab in ((-0.5, "-0.5S"), (0.0, "AT EVENT"), (0.5, "+0.5S")):
            frames.append(grab_frame(video, t + dt, size))
            labels.append(f"{lab} {t + dt:.2f}")
    width = max(sum(f.shape[1] for f in frames) + 6 * max(0, len(frames) - 1), 720)
    t0 = max(0.0, ev["start"] - 1.0)
    t1 = min(A.duration, ev["end"] + 1.0)
    marks = [(c, (255, 255, 255)) for c in ev.get("_cuts", []) if t0 <= c <= t1]
    crop = viz.spec_crop(A.x, ASR, t0, t1, width, 200, marks=marks, span=(ev["start"], ev["end"]))
    return viz.sheet(frames, crop, path, title, labels)
