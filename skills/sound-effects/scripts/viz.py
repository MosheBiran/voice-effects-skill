"""Dependency-free PNG rendering: spectrograms, SFX cards, video overviews, review sheets.

Everything here returns or writes plain RGB uint8 arrays so Claude can *look* at
sounds (Read the PNG) while designing or reviewing them.
"""

import struct
import zlib

import numpy as np

BG = np.array([14, 16, 24], np.uint8)
FG = np.array([226, 230, 238], np.uint8)
DIM = np.array([120, 128, 145], np.uint8)
CUT = np.array([255, 255, 255], np.uint8)

FAMILY_COLORS = {
    "impact": (255, 92, 72), "sub_drop": (190, 70, 255), "whoosh": (64, 200, 255),
    "riser": (255, 196, 40), "downlifter": (255, 140, 40), "ui": (120, 255, 140),
    "chime": (255, 120, 220), "glitch": (0, 255, 210), "texture": (170, 170, 200),
    "speech": (90, 90, 110), "music": (110, 110, 90), "other": (150, 150, 150),
}

_FONT = {
    "0": ".###. #...# #..## #.#.# ##..# #...# .###.", "1": "..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.",
    "2": ".###. #...# ....# ...#. ..#.. .#... #####", "3": "##### ...#. ..#.. ...#. ....# #...# .###.",
    "4": "...#. ..##. .#.#. #..#. ##### ...#. ...#.", "5": "##### #.... ####. ....# ....# #...# .###.",
    "6": "..##. .#... #.... ####. #...# #...# .###.", "7": "##### ....# ...#. ..#.. .#... .#... .#...",
    "8": ".###. #...# #...# .###. #...# #...# .###.", "9": ".###. #...# #...# .#### ....# ...#. .##..",
    "A": ".###. #...# #...# ##### #...# #...# #...#", "B": "####. #...# #...# ####. #...# #...# ####.",
    "C": ".###. #...# #.... #.... #.... #...# .###.", "D": "###.. #..#. #...# #...# #...# #..#. ###..",
    "E": "##### #.... #.... ####. #.... #.... #####", "F": "##### #.... #.... ####. #.... #.... #....",
    "G": ".###. #...# #.... #.### #...# #...# .####", "H": "#...# #...# #...# ##### #...# #...# #...#",
    "I": ".###. ..#.. ..#.. ..#.. ..#.. ..#.. .###.", "J": "..### ...#. ...#. ...#. ...#. #..#. .##..",
    "K": "#...# #..#. #.#.. ##... #.#.. #..#. #...#", "L": "#.... #.... #.... #.... #.... #.... #####",
    "M": "#...# ##.## #.#.# #.#.# #...# #...# #...#", "N": "#...# #...# ##..# #.#.# #..## #...# #...#",
    "O": ".###. #...# #...# #...# #...# #...# .###.", "P": "####. #...# #...# ####. #.... #.... #....",
    "Q": ".###. #...# #...# #...# #.#.# #..#. .##.#", "R": "####. #...# #...# ####. #.#.. #..#. #...#",
    "S": ".#### #.... #.... .###. ....# ....# ####.", "T": "##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
    "U": "#...# #...# #...# #...# #...# #...# .###.", "V": "#...# #...# #...# #...# #...# .#.#. ..#..",
    "W": "#...# #...# #...# #.#.# #.#.# #.#.# .#.#.", "X": "#...# #...# .#.#. ..#.. .#.#. #...# #...#",
    "Y": "#...# #...# .#.#. ..#.. ..#.. ..#.. ..#..", "Z": "##### ....# ...#. ..#.. .#... #.... #####",
    ".": "..... ..... ..... ..... ..... .##.. .##..", ":": "..... .##.. .##.. ..... .##.. .##.. .....",
    "-": "..... ..... ..... ##### ..... ..... .....", "+": "..... ..#.. ..#.. ##### ..#.. ..#.. .....",
    "#": ".#.#. .#.#. ##### .#.#. ##### .#.#. .#.#.", "/": "..... ....# ...#. ..#.. .#... #.... .....",
    "%": "##... ##..# ...#. ..#.. .#... #..## ...##", "_": "..... ..... ..... ..... ..... ..... #####",
    "(": "...#. ..#.. .#... .#... .#... ..#.. ...#.", ")": ".#... ..#.. ...#. ...#. ...#. ..#.. .#...",
    ",": "..... ..... ..... ..... .##.. ..#.. .#...", "=": "..... ..... ##### ..... ##### ..... .....",
    "?": ".###. #...# ....# ...#. ..#.. ..... ..#..", ">": ".#... ..#.. ...#. ....# ...#. ..#.. .#...",
    "<": "...#. ..#.. .#... #.... .#... ..#.. ...#.", "|": "..#.. ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
    " ": "..... ..... ..... ..... ..... ..... .....",
}
_GLYPHS = {k: np.array([[c == "#" for c in row] for row in v.split()], bool) for k, v in _FONT.items()}


def write_png(path, img):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def text_width(text, scale=1):
    return len(text) * 6 * scale


def draw_text(img, x, y, text, color=FG, scale=1, bg=None):
    """Draw ASCII text (unknown characters become '?'). Clips at the image edge."""
    text = str(text).upper()
    if bg is not None:
        fill_rect(img, x - 2, y - 2, x + text_width(text, scale) + 1, y + 7 * scale + 2, bg)
    cx = x
    for ch in text:
        g = _GLYPHS.get(ch, _GLYPHS["?"])
        if scale > 1:
            g = np.kron(g, np.ones((scale, scale), bool))
        h, w = g.shape
        y0, x0 = max(0, y), max(0, cx)
        y1, x1 = min(img.shape[0], y + h), min(img.shape[1], cx + w)
        if y1 > y0 and x1 > x0:
            sub = g[y0 - y:y1 - y, x0 - cx:x1 - cx]
            img[y0:y1, x0:x1][sub] = color
        cx += 6 * scale


def fill_rect(img, x0, y0, x1, y1, color, alpha=1.0):
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    if alpha >= 1.0:
        img[y0:y1, x0:x1] = color
    else:
        reg = img[y0:y1, x0:x1].astype(float)
        img[y0:y1, x0:x1] = (reg * (1 - alpha) + np.array(color, float) * alpha).astype(np.uint8)


def vline(img, x, y0, y1, color, dash=0):
    x = int(x)
    if not 0 <= x < img.shape[1]:
        return
    ys = np.arange(max(0, int(y0)), min(img.shape[0], int(y1)))
    if dash:
        ys = ys[(ys // dash) % 2 == 0]
    img[ys, x] = color


_CMAP = np.array([[10, 10, 22], [40, 18, 88], [118, 32, 128], [196, 58, 108],
                  [244, 120, 72], [252, 196, 96], [252, 252, 200]], float)


def colormap(v):
    v = np.clip(v, 0.0, 1.0) * (len(_CMAP) - 1)
    i = np.minimum(v.astype(int), len(_CMAP) - 2)
    f = (v - i)[..., None]
    return (_CMAP[i] * (1 - f) + _CMAP[i + 1] * f).astype(np.uint8)


def spectrogram(x, sr, width, height, t0=0.0, t1=None, fmin=30.0, fmax=None, db_range=75.0, n_fft=None):
    """Log-frequency spectrogram as a (height, width) array in 0..1 (row 0 = top = high freq)."""
    x = np.asarray(x, float)
    if x.ndim > 1:
        x = x.mean(axis=0)
    t1 = len(x) / sr if t1 is None else t1
    fmax = fmax or sr / 2 * 0.95
    n_fft = n_fft or (2048 if sr > 30000 else 1024)
    half = n_fft // 2
    centres = (np.linspace(t0, t1, width) * sr).astype(int)
    pad = np.concatenate([np.zeros(half), x, np.zeros(half + 1)])
    idx = np.clip(centres, 0, len(x)) [:, None] + np.arange(n_fft)[None, :]
    win = np.hanning(n_fft)
    mags = np.abs(np.fft.rfft(pad[idx] * win, axis=1)) + 1e-9
    dbv = 20 * np.log10(mags)
    freqs = np.geomspace(fmin, fmax, height)[::-1]
    bins = freqs / (sr / n_fft)
    b0 = np.clip(bins.astype(int), 0, mags.shape[1] - 2)
    fr = bins - b0
    img = dbv[:, b0] * (1 - fr) + dbv[:, b0 + 1] * fr
    top = dbv.max() - 4.0
    return np.clip((img.T - (top - db_range)) / db_range, 0, 1) ** 1.3


def freq_to_row(f, height, sr, fmin=30.0, fmax=None):
    fmax = fmax or sr / 2 * 0.95
    return int((1 - np.log(f / fmin) / np.log(fmax / fmin)) * (height - 1))


def waveform_strip(x, width, height, t0=0.0, t1=None, sr=48000, color=(150, 190, 255)):
    x = np.asarray(x, float)
    if x.ndim > 1:
        x = x.mean(axis=0)
    t1 = len(x) / sr if t1 is None else t1
    img = np.zeros((height, width, 3), np.uint8) + BG
    edges = np.linspace(t0 * sr, t1 * sr, width + 1).astype(int)
    pk = np.max(np.abs(x)) + 1e-12
    mid = height // 2
    for i in range(width):
        a, b = max(0, edges[i]), min(len(x), max(edges[i + 1], edges[i] + 1))
        if b <= a:
            continue
        seg = x[a:b] / pk
        y0 = int(mid - seg.max() * (mid - 2))
        y1 = int(mid - seg.min() * (mid - 2))
        img[max(0, y0):min(height, y1 + 1), i] = color
    return img


def _freq_grid(img, y_off, height, sr, fmin, fmax=None):
    for f, lab in ((100, "100"), (1000, "1K"), (10000, "10K")):
        if f < (fmax or sr / 2 * 0.95):
            r = y_off + freq_to_row(f, height, sr, fmin, fmax)
            img[r, ::4] = DIM
            draw_text(img, 3, r - 9, lab, DIM)


def sfx_card(x, sr, path, title, sync=None, width=900):
    """Waveform + spectrogram card for one generated effect."""
    dur = x.shape[-1] / sr
    head, wave_h, spec_h = 24, 70, 260
    img = np.zeros((head + wave_h + spec_h + 18, width, 3), np.uint8) + BG
    draw_text(img, 8, 8, f"{title}   {dur:.2f}S", FG)
    img[head:head + wave_h] = waveform_strip(x, width, wave_h, sr=sr)
    spec = spectrogram(x, sr, width, spec_h, fmin=25.0)
    img[head + wave_h:head + wave_h + spec_h] = colormap(spec)
    _freq_grid(img, head + wave_h, spec_h, sr, 25.0)
    step = 0.1 if dur < 1.5 else 0.25 if dur < 4 else 0.5 if dur < 8 else 1.0
    for k in range(int(dur / step) + 1):
        px = int(k * step / dur * (width - 1))
        vline(img, px, head + wave_h + spec_h, head + wave_h + spec_h + 5, DIM)
        if k % 2 == 0:
            draw_text(img, px + 2, head + wave_h + spec_h + 8, f"{k * step:g}", DIM)
    if sync is not None:
        px = int(sync / dur * (width - 1))
        vline(img, px, head, head + wave_h + spec_h, (80, 255, 160), dash=4)
        draw_text(img, min(px + 3, width - 60), head + 3, "SYNC", (80, 255, 160))
    write_png(path, img)
    return path


def overview(x, sr, path, title, cuts=(), events=(), width=None):
    """Whole-video timeline: spectrogram, scene cuts (white dashes), detected events (boxes with #id)."""
    dur = len(x) / sr
    width = width or int(np.clip(dur * 40, 1000, 2400))
    head, lane, wave_h, spec_h, foot = 26, 34, 50, 240, 20
    H = head + lane + wave_h + spec_h + foot
    img = np.zeros((H, width, 3), np.uint8) + BG
    draw_text(img, 8, 8, title, FG)
    y_w = head + lane
    img[y_w:y_w + wave_h] = waveform_strip(x, width, wave_h, sr=sr)
    y_s = y_w + wave_h
    img[y_s:y_s + spec_h] = colormap(spectrogram(x, sr, width, spec_h, fmin=30.0))
    _freq_grid(img, y_s, spec_h, sr, 30.0)

    def px(t):
        return int(t / max(dur, 1e-6) * (width - 1))

    step = 1 if dur <= 30 else 2 if dur <= 60 else 5 if dur <= 180 else 10
    for k in range(int(dur / step) + 1):
        vline(img, px(k * step), y_s + spec_h, y_s + spec_h + 5, DIM)
        draw_text(img, px(k * step) + 2, y_s + spec_h + 8, f"{k * step}", DIM)
    for c in cuts:
        vline(img, px(c), head, y_s + spec_h, CUT, dash=3)
    row_x_end = [-1, -1]
    for ev in sorted(events, key=lambda e: e.get("likely", True)):
        fam = ev.get("family") or "other"
        col = FAMILY_COLORS.get(fam, FAMILY_COLORS["other"])
        x0, x1 = px(ev["start"]), max(px(ev["end"]), px(ev["start"]) + 2)
        if not ev.get("likely", True):
            fill_rect(img, x0, y_s + spec_h - 2, x1, y_s + spec_h, DIM)
            continue
        fill_rect(img, x0, y_w, x1, y_s + spec_h, col, alpha=0.18)
        fill_rect(img, x0, y_s + spec_h - 4, x1, y_s + spec_h, col)
        lab = f"#{ev['id']} {ev.get('label') or ev.get('auto_label') or '?'}"
        r = 0 if row_x_end[0] < x0 else 1
        ty = head + 3 + r * 15
        draw_text(img, x0 + 1, ty, lab, col, bg=BG)
        row_x_end[r] = x0 + text_width(lab) + 6
    write_png(path, img)
    return path


def sheet(frames, spec_img, path, title, frame_labels=()):
    """Review sheet: a row of video frames (RGB arrays) above a spectrogram crop (RGB)."""
    gap = 6
    fw = sum(f.shape[1] for f in frames) + gap * max(0, len(frames) - 1) if frames else 0
    width = max(fw, spec_img.shape[1] if spec_img is not None else 0, 400)
    fh = max((f.shape[0] for f in frames), default=0)
    head = 24
    H = head + (fh + 16 if frames else 0) + (spec_img.shape[0] if spec_img is not None else 0)
    img = np.zeros((H, width, 3), np.uint8) + BG
    draw_text(img, 8, 8, title, FG)
    x = 0
    for i, f in enumerate(frames):
        img[head:head + f.shape[0], x:x + f.shape[1]] = f
        if i < len(frame_labels):
            draw_text(img, x + 4, head + f.shape[0] + 4, frame_labels[i], DIM)
        x += f.shape[1] + gap
    if spec_img is not None:
        y = head + (fh + 16 if frames else 0)
        img[y:y + spec_img.shape[0], :spec_img.shape[1]] = spec_img
    write_png(path, img)
    return path


def spec_crop(x, sr, t0, t1, width, height, marks=(), span=None):
    """Spectrogram crop t0..t1 as RGB with optional vertical marks [(t, color)] and a shaded span."""
    spec = colormap(spectrogram(x, sr, width, height, t0=t0, t1=t1, fmin=30.0))
    img = np.zeros((height + 14, width, 3), np.uint8) + BG
    img[:height] = spec

    def px(t):
        return int((t - t0) / max(t1 - t0, 1e-6) * (width - 1))

    if span:
        fill_rect(img, px(span[0]), height - 5, max(px(span[1]), px(span[0]) + 2), height, (80, 255, 160))
    for t, col in marks:
        vline(img, px(t), 0, height, col, dash=3)
    draw_text(img, 2, height + 4, f"{t0:.2f}S", DIM)
    lab = f"{t1:.2f}S"
    draw_text(img, width - text_width(lab) - 2, height + 4, lab, DIM)
    _freq_grid(img, 0, height, sr, 30.0)
    return img
