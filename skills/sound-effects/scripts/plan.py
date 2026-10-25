#!/usr/bin/env python3
"""Place sound effects on a new video the way the learned creators would.

  plan.py suggest <video> [--style all|NAME,NAME] [--intensity 1.0] [--seed 0] [--out cues.json]
                          [--review-dir DIR]
      detect cuts, apply the playbook (hook, riser into the strongest cut, effects on cuts at the
      learned rate and offset, learned pitch/depth) and write an editable cue sheet

  plan.py render <cues.json> [--video PATH] [--out DIR] [--preview] [--tc-start 01:00:00:00]
      render every cue, build one SFX stem aligned to the video (drop it at 00:00 in the edit),
      cues.csv, DaVinci Resolve marker EDL, and optionally preview.mp4 (video + stem mixed)
"""

import argparse
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import analysis as AN  # noqa: E402
import learn  # noqa: E402
import sfx_engine as E  # noqa: E402
import viz  # noqa: E402

DEFAULT_STATS = {
    "events": 0, "sfx_per_min": 12.0, "cut_coverage": 0.6, "hook_rate": 0.7, "hook_family": "impact",
    "families": {
        "whoosh": {"count": 50, "at_cut": 0.85, "offset_median": -0.04, "presets": {"whoosh": 3, "swipe": 2}, "est": None},
        "impact": {"count": 20, "at_cut": 0.7, "offset_median": 0.0, "presets": {"impact": 2, "punch": 2}, "est": None},
        "ui": {"count": 15, "at_cut": 0.3, "offset_median": 0.0, "presets": {"pop": 2, "click": 1}, "est": None},
        "riser": {"count": 8, "at_cut": 0.8, "offset_median": 0.0, "presets": {"riser": 1}, "est": None},
        "sub_drop": {"count": 5, "at_cut": 0.7, "offset_median": 0.0, "presets": {"sub_drop": 1}, "est": None},
        "glitch": {"count": 4, "at_cut": 0.6, "offset_median": 0.0, "presets": {"glitch": 1}, "est": None},
    },
}
GAIN_DB = {"impact": -3, "sub_drop": -3, "riser": -6, "downlifter": -6, "whoosh": -8, "ui": -10,
           "chime": -10, "glitch": -8, "texture": -10}
RESOLVE_COLOR = {"impact": "Red", "sub_drop": "Purple", "riser": "Yellow", "downlifter": "Rose",
                 "whoosh": "Blue", "ui": "Green", "chime": "Pink", "glitch": "Cyan", "texture": "Sand"}


def learned_stats(style):
    docs = [d for _, d in learn.all_docs()]
    if style and style != "all":
        names = {s.strip().lower() for s in style.split(",")}
        docs = [d for d in docs if d["creator"].lower() in names or learn.slug(d["creator"]) in names]
    if docs:
        for inc in (None, 0.75):
            st = learn.aggregate(docs, inc)
            if st["events"] >= 5:
                return st, "learned" + (" (incl. auto-labelled)" if inc else ""), len(docs)
    return DEFAULT_STATS, "defaults (nothing learned for this style yet)", 0


def pick_preset(fam, *counters):
    """First preset id found in the given usage counters (most used first), else the family default."""
    ids = {p["id"] for p in E.list_presets()}
    for counter in counters:
        for lab in (counter or {}):
            if lab in ids and AN.family_of(lab) == fam:
                return lab
    return learn.DEFAULT_PRESET.get(fam, "whoosh")


def make_cue(stats, fam, t, reason, rng, kind="cut", preset=None):
    fs = stats["families"].get(fam, {})
    preset = preset or pick_preset(fam, fs.get("cut_presets") if kind == "cut" else None, fs.get("presets"))
    P = E.load_preset(preset)
    est = fs.get("est") or {}
    p0 = P.get("pitch", {}).get("default", 0)
    d0 = P.get("depth", {}).get("default", 0.5)
    # learned values are rough (~3 st), so move halfway from the preset's tuned default
    pitch = round(p0 + 0.5 * (est["pitch"] - p0)) if est.get("pitch") is not None else p0
    depth = d0 + 0.7 * (est["depth"] - d0) if est.get("depth") is not None else d0
    length = 1.0 + 0.5 * (est["length"] - 1.0) if est.get("length") is not None else 1.0
    return {"time": round(float(t), 3), "effect": preset, "family": fam, "anchor": AN.anchor_name(fam),
            "pitch": round(float(pitch), 1), "depth": round(float(depth), 2), "length": round(float(length), 2),
            "gain_db": GAIN_DB.get(fam, -8), "seed": int(rng.integers(1000)), "kind": kind, "reason": reason}


def cmd_suggest(args):
    info = AN.probe(args.video)
    dur = info["duration"]
    cuts, scores, _ = AN.scene_cuts(args.video) if info.get("has_video") else ([], [], None)
    stats, source, n_docs = learned_stats(args.style)
    rng = np.random.default_rng(args.seed)
    fams = stats["families"]
    cues = []

    def free(t):
        return all(abs(c["time"] - t) > 0.3 for c in cues)

    hf = stats.get("hook_family")
    if (stats.get("hook_rate") or 0) >= 0.4 and hf:
        early = [c for c in cuts if c <= 2.0]
        t = early[0] if early else 0.0
        hp = pick_preset(hf, stats.get("hook_presets"), fams.get(hf, {}).get("presets"))
        cues.append(make_cue(stats, hf, t + (fams.get(hf, {}).get("offset_median") or 0.0 if early else 0.0),
                             f"hook: {stats['hook_rate']:.0%} of learned videos open with an effect ({hp})",
                             rng, "hook", preset=hp))

    riser = fams.get("riser")
    if riser and riser["count"] / max(1, sum(f["count"] for f in fams.values())) >= 0.04 and cuts:
        cand = [(s, c) for c, s in zip(cuts, scores) if c >= 3.0]
        if cand:
            s, c = max(cand)
            cues.append(make_cue(stats, "riser", c + (riser.get("offset_median") or 0.0),
                                 f"riser ending on the strongest cut (scene score {s:.2f})", rng, "build"))
            imp = fams.get("impact")
            if imp and imp.get("at_cut", 0) >= 0.4:
                cues.append(make_cue(stats, "impact", c + (imp.get("offset_median") or 0.0),
                                     "impact landing the riser's payoff", rng, "payoff"))

    cut_fams = [(f, s["count"] * s.get("at_cut", 0)) for f, s in fams.items() if f != "riser"]
    cut_fams = [(f, w) for f, w in cut_fams if w > 0]
    if cut_fams and cuts:
        names = [f for f, _ in cut_fams]
        p = np.array([w for _, w in cut_fams], float)
        p /= p.sum()
        n_cut = int(round((stats.get("cut_coverage") or 0.5) * len(cuts) * args.intensity))
        recent = []
        for c, s in sorted(zip(cuts, scores), key=lambda cs: -cs[1]):
            if n_cut <= 0:
                break
            fam = rng.choice(names, p=p)
            if len(recent) >= 2 and recent[-1] == recent[-2] == fam and len(names) > 1:
                fam = rng.choice(names, p=p)
            t = c + (fams[fam].get("offset_median") or 0.0)
            if not free(t):
                continue
            share = fams[fam]["count"] * fams[fam].get("at_cut", 0) / sum(w for _, w in cut_fams)
            cues.append(make_cue(stats, fam, t, f"cut (scene score {s:.2f}); {share:.0%} of cut effects are {fam}", rng))
            recent.append(fam)
            n_cut -= 1

    target = stats.get("sfx_per_min", 10) * dur / 60.0 * args.intensity
    if len(cues) > target * 1.3:
        keep = [c for c in cues if c["kind"] != "cut"]
        rest = [c for c in cues if c["kind"] == "cut"][: max(0, int(round(target * 1.3)) - len(keep))]
        cues = keep + rest
    cues = [c for c in cues if 0.0 <= c["time"] <= dur]
    cues.sort(key=lambda c: c["time"])
    for i, c in enumerate(cues, 1):
        c["id"] = i

    doc = {"video": os.path.abspath(args.video), "duration": round(dur, 3), "fps": round(info.get("fps", 30.0), 3),
           "style": args.style, "source": source, "learned_videos": n_docs, "cuts": cuts, "cut_scores": scores,
           "cues": cues}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)

    print(f"{os.path.basename(args.video)}: {dur:.1f}s, {len(cuts)} cuts; style={args.style} -> {source}")
    print(f"{'id':>3} {'time':>7}  {'effect':<14} {'pitch':>5} {'depth':>5} {'gain':>5}  reason")
    for c in cues:
        print(f"{c['id']:>3} {c['time']:>7.2f}  {c['effect']:<14} {c['pitch']:>+5.1f} {c['depth']:>5.2f} {c['gain_db']:>5}  {c['reason']}")
    uncovered = [c for c in cuts if all(abs(c - q["time"]) > 0.4 for q in cues)]
    print(f"\ncue sheet: {args.out}")
    if uncovered:
        print("cuts left without an effect: " + ", ".join(f"{c:.2f}" for c in uncovered))

    if args.review_dir and info.get("has_video") and AN.have_ffmpeg():
        os.makedirs(args.review_dir, exist_ok=True)
        size = AN.frame_size(info)
        x = AN.load_audio(args.video)
        A_x = x if len(x) else np.zeros(int(dur * AN.ASR) + 1)
        ev_like = [{"id": c["id"], "start": c["time"] - 0.05, "end": c["time"] + 0.25, "family": c["family"],
                    "label": c["effect"]} for c in cues]
        viz.overview(A_x, AN.ASR, os.path.join(args.review_dir, "plan_overview.png"),
                     f"PLAN {os.path.basename(args.video)}", cuts=cuts, events=ev_like)
        for k, c in enumerate(cuts, 1):
            frames = [AN.grab_frame(args.video, c - 0.25, size), AN.grab_frame(args.video, c + 0.25, size)]
            crop = viz.spec_crop(A_x, AN.ASR, max(0, c - 1.0), min(dur, c + 1.0), max(720, 2 * size[0] + 6), 140,
                                 marks=[(c, (255, 255, 255))])
            viz.sheet(frames, crop, os.path.join(args.review_dir, f"cut_{k:03d}.png"),
                      f"CUT {k} AT {c:.2f}S", [f"BEFORE {c - 0.25:.2f}", f"AFTER {c + 0.25:.2f}"])
        print(f"review: {args.review_dir}/plan_overview.png and cut_NNN.png (frames before/after each cut)")


def _tc(t, fps, start):
    hh, mm, ss, ff = (int(v) for v in start.split(":"))
    f = int(round(t * fps)) + ((hh * 60 + mm) * 60 + ss) * fps + ff
    return f"{f // (fps * 3600):02d}:{f // (fps * 60) % 60:02d}:{f // fps % 60:02d}:{f % fps:02d}"


def write_edl(path, cues, fps, start):
    fps = max(1, int(round(fps)))
    lines = ["TITLE: SFX cues", "FCM: NON-DROP FRAME", ""]
    for i, c in enumerate(cues, 1):
        a = _tc(c["time"], fps, start)
        b = _tc(c["time"] + 1.0 / fps, fps, start)
        color = RESOLVE_COLOR.get(c.get("family"), "Blue")
        lines += [f"{i:03d}  001      V     C        {a} {b} {a} {b}  ",
                  f" |C:ResolveColor{color} |M:{c['effect']} {c['pitch']:+g}st d{c['depth']:.2f} |D:1", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def cmd_render(args):
    with open(args.cues, encoding="utf-8") as fh:
        doc = json.load(fh)
    video = args.video or doc.get("video")
    dur = doc["duration"]
    os.makedirs(os.path.join(args.out, "cues"), exist_ok=True)
    n = int(dur * E.SR) + 1
    stem = np.zeros((2, n))
    rows = []
    for c in doc["cues"]:
        if c.get("mute"):
            continue
        P = E.load_preset(c["effect"])
        extra = {k: c[k] for k in ("tone", "drive", "space", "width", "variation") if c.get(k) is not None}
        x, info = E.render(P, seed=c.get("seed", 0), pitch=c.get("pitch"), depth=c.get("depth"),
                           length=c.get("length"), **extra)
        fn = os.path.join(args.out, "cues", f"{c['id']:02d}_{c['effect']}.wav")
        E.write_wav(fn, x)
        o = int(round((c["time"] - info["sync"]) * E.SR))
        seg = x * E.db(c.get("gain_db", -6))
        if o < 0:
            seg, o = seg[:, -o:], 0
        m = min(seg.shape[1], n - o)
        if m > 0:
            stem[:, o:o + m] += seg[:, :m]
        rows.append({"id": c["id"], "time": c["time"], "starts_at": round(c["time"] - info["sync"], 3),
                     "effect": c["effect"], "pitch": c.get("pitch"), "depth": c.get("depth"), "length": c.get("length"),
                     "gain_db": c.get("gain_db"), "file": os.path.relpath(fn, args.out), "reason": c.get("reason", "")})
    pk = np.max(np.abs(stem)) if stem.size else 0.0
    if pk > 0.98:
        stem *= 0.98 / pk
        print(f"stem peaked at {20 * np.log10(pk):+.1f} dBFS; scaled down to -0.2 dBFS")
    stem_path = os.path.join(args.out, "sfx_stem.wav")
    E.write_wav(stem_path, stem)
    with open(os.path.join(args.out, "cues.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    write_edl(os.path.join(args.out, "markers.edl"), [c for c in doc["cues"] if not c.get("mute")],
              doc.get("fps", 30), args.tc_start)
    print(f"stem:    {stem_path}  ({dur:.2f}s, starts at the video's first frame)")
    print(f"cues:    {os.path.join(args.out, 'cues')}/ ({len(rows)} files), cues.csv")
    print(f"markers: {os.path.join(args.out, 'markers.edl')} (Resolve: Timeline > Import > Timeline Markers from EDL)")
    if args.preview:
        if not video or not os.path.exists(video) or not AN.have_ffmpeg():
            print("preview skipped: video or ffmpeg not available")
            return
        info = AN.probe(video)
        out = os.path.join(args.out, "preview.mp4")
        if info.get("has_audio"):
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", video, "-i", stem_path, "-filter_complex",
                   "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[a]", "-map", "0:v", "-map", "[a]",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out]
        else:
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", video, "-i", stem_path, "-map", "0:v", "-map", "1:a",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out]
        subprocess.run(cmd, check=True)
        print(f"preview: {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("suggest")
    p.add_argument("video")
    p.add_argument("--style", default="all")
    p.add_argument("--intensity", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="cues.json")
    p.add_argument("--review-dir", default="sfx_plan_review")
    p.set_defaults(fn=cmd_suggest)
    p = sub.add_parser("render")
    p.add_argument("cues")
    p.add_argument("--video")
    p.add_argument("--out", default="sfx_render")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--tc-start", default="01:00:00:00")
    p.set_defaults(fn=cmd_render)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
