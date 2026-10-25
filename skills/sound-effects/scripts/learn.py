#!/usr/bin/env python3
"""Learn when creators use sound effects, from their finished videos.

  learn.py analyze <video...> --creator NAME   detect + classify SFX, write knowledge JSON and review PNGs
  learn.py review <video.json>                 print the event table again
  learn.py label <video.json> ID=LABEL[:TRIGGER[:NOTE]] ...
                                               confirm / correct / reject events (LABEL 'ok' keeps the
                                               auto label, 'x' rejects)
  learn.py confirm <video.json> --min 0.7      confirm every pending event at or above a confidence
  learn.py add <video.json> --start S --end E --label L [--trigger T] [--video PATH]
                                               add an effect the detector missed
  learn.py train [--rebuild-synthetic]         rebuild the classifier (synthetic + confirmed examples)
  learn.py playbook [--include-auto 0.75]      aggregate everything into PLAYBOOK.md / playbook.json
  learn.py status                              what has been learned so far

Knowledge lives in ../knowledge (override with SFX_KNOWLEDGE_DIR).
"""

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import analysis as AN  # noqa: E402
import sfx_engine as E  # noqa: E402

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOW = os.environ.get("SFX_KNOWLEDGE_DIR") or os.path.join(SKILL_DIR, "knowledge")
VIDEOS = os.path.join(KNOW, "videos")
MODEL = os.path.join(KNOW, "model.npz")
SYNTH = os.path.join(KNOW, "synthetic.npz")
PLAYBOOK_JSON = os.path.join(KNOW, "playbook.json")
PLAYBOOK_MD = os.path.join(KNOW, "PLAYBOOK.md")

TRIGGERS = ["cut", "transition", "text", "number", "zoom", "reveal", "before_after", "logo", "product",
            "face", "gesture", "punchline", "beat", "hook", "cta", "screen", "motion", "other"]
DEFAULT_PRESET = {"impact": "impact", "sub_drop": "sub_drop", "whoosh": "whoosh", "riser": "riser",
                  "downlifter": "downlifter", "ui": "pop", "chime": "notification", "glitch": "glitch",
                  "texture": "rumble"}


def slug(s):
    s = re.sub(r"[^\w]+", "-", str(s).strip().lower(), flags=re.UNICODE).strip("-")
    return s or "unnamed"


def load_doc(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_doc(doc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)


def all_docs():
    return [(p, load_doc(p)) for p in sorted(glob.glob(os.path.join(VIDEOS, "*", "*.json")))]


def load_model(quiet=False):
    if os.path.exists(MODEL):
        return AN.Model.load(MODEL)
    if not quiet:
        print("note: no classifier yet - run `learn.py train` (events will be unlabelled)", file=sys.stderr)
    return None


# ------------------------------------------------------------------ analyze

def classify(ev, model, fam):
    if model is None:
        ev.update(auto_label="other", confidence=0.0, alternatives=[], est=None)
    else:
        ranked, est, dist = model.predict(ev["features"])
        ev.update(auto_label=str(ranked[0][0]), confidence=round(float(ranked[0][1]), 2),
                  alternatives=[[str(lab), round(float(p), 2)] for lab, p in ranked[1:3]], est=est,
                  nn_dist=round(float(dist), 2))
    ev["family"] = AN.family_of(ev["label"] or ev["auto_label"], fam)


def print_table(doc):
    print(f"\n{doc['creator']} / {doc['video']}  ({doc['duration']:.1f}s, {len(doc.get('cuts', []))} cuts, "
          f"{len(doc['events'])} events)")
    print("* = likely sound effect (has a review sheet); cut = anchor time minus nearest cut")
    print(f"{'id':>4} {'time':>7} {'dur':>5}  {'label':<22} {'conf':>4}  {'alternatives':<26} {'cut':>7}  status / notes")
    for ev in doc["events"]:
        star = "*" if AN.is_likely(ev) else " "
        c = ev.get("context", {})
        off = c.get("offset_to_cut")
        cut = f"{off:+.2f}" if off is not None and abs(off) <= 1.5 else "-"
        lab = ev["label"] or ev["auto_label"]
        alts = ", ".join(f"{a}:{p:.2f}" for a, p in ev.get("alternatives", []))
        extra = [ev["status"]]
        if ev.get("trigger"):
            extra.append(f"trigger={ev['trigger']}")
        extra += ev.get("flags", [])
        print(f"{star}{ev['id']:>3} {ev['start']:>7.2f} {ev['end'] - ev['start']:>5.2f}  {lab:<22} "
              f"{ev.get('confidence', 0):>4.2f}  {alts:<26} {cut:>7}  {' '.join(extra)}")


def analyze_one(path, creator, review_dir, model, sheets="likely", sensitivity=1.0):
    info = AN.probe(path)
    x = AN.load_audio(path)
    A = AN.Analysis(x)
    cuts, cut_scores, _ = AN.scene_cuts(path) if info.get("has_video") else ([], [], None)
    evs = AN.detect_events(A, sensitivity)
    fam = AN.families()
    for i, ev in enumerate(evs, 1):
        ev["id"] = i
        ev["features"] = [round(float(v), 4) for v in AN.event_features(A, ev)]
        ev.update(label=None, trigger=None, note=None, status="pending")
        classify(ev, model, fam)
        AN.add_context(ev, cuts, A.duration, ev["family"])
    AN.flag_repeats(evs, model)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(VIDEOS, slug(creator), slug(stem) + ".json")
    if os.path.exists(out):
        evs = carry_over_labels(load_doc(out), evs, cuts, A.duration, fam)
    for ev in evs:
        ev["likely"] = AN.is_likely(ev)
    doc = {"version": 1, "video": os.path.basename(path), "source": os.path.abspath(path), "creator": creator,
           "analyzed": dt.date.today().isoformat(), "duration": round(A.duration, 3),
           "has_video": info.get("has_video", False), "fps": round(info.get("fps", 30.0), 3),
           "size": [info.get("width"), info.get("height")] if info.get("has_video") else None,
           "cuts": cuts, "cut_scores": cut_scores, "events": evs}
    save_doc(doc, out)

    rdir = os.path.join(review_dir, slug(creator), slug(stem))
    os.makedirs(rdir, exist_ok=True)
    AN.overview_png(A, os.path.join(rdir, "overview.png"), f"{creator} / {stem}", cuts, evs)
    made = 0
    if sheets != "none":
        for ev in evs:
            if sheets == "all" or ev["likely"]:
                ev["_cuts"] = cuts
                AN.event_sheet_png(A, path, info, ev, os.path.join(rdir, f"ev_{ev['id']:03d}.png"),
                                   f"#{ev['id']} {ev['auto_label']} ({ev['confidence']:.2f})  "
                                   f"{ev['start']:.2f}-{ev['end']:.2f}S")
                ev.pop("_cuts", None)
                made += 1
    print_table(doc)
    print(f"\nknowledge: {out}\nreview:    {rdir}/overview.png (+{made} event sheets for the * rows)")
    return out


def carry_over_labels(old, evs, cuts, duration, fam):
    """Re-analysis keeps earlier review work: copy labels to the new event that overlaps most."""
    reviewed = [e for e in old.get("events", []) if e.get("status") != "pending"]
    for oe in reviewed:
        best, best_ov = None, 0.0
        for ev in evs:
            ov = min(ev["end"], oe["end"]) - max(ev["start"], oe["start"])
            if ov > best_ov:
                best, best_ov = ev, ov
        if best is not None and best_ov >= 0.5 * min(oe["end"] - oe["start"], best["end"] - best["start"]):
            for k in ("label", "trigger", "note", "status"):
                best[k] = oe.get(k)
        elif "manual" in oe.get("flags", []) or oe.get("status") == "confirmed":
            evs.append(dict(oe, flags=sorted(set(oe.get("flags", [])) | {"kept_from_previous"})))
    evs.sort(key=lambda e: e["start"])
    for i, ev in enumerate(evs, 1):
        ev["id"] = i
        ev["family"] = AN.family_of(ev["label"] or ev["auto_label"], fam)
        AN.add_context(ev, cuts, duration, ev["family"])
    n = sum(e.get("status") != "pending" for e in evs)
    print(f"kept {n} reviewed events from the previous analysis", file=sys.stderr)
    return evs


def cmd_analyze(args):
    model = load_model()
    for path in args.videos:
        analyze_one(path, args.creator, args.review_dir, model, args.sheets, args.sensitivity)


def cmd_review(args):
    print_table(load_doc(args.doc))


# -------------------------------------------------------------------- label

def _apply_label(doc, ev, label, trigger, note, fam, model=None):
    if label in ("x", "reject", "no"):
        ev.update(status="rejected", label=None)
    else:
        if label == "ok":
            label = ev["auto_label"]
        if label not in fam and label not in set(fam.values()):
            raise SystemExit(f"unknown label '{label}'. Use a preset id ({', '.join(sorted(k for k in fam if k not in AN.NON_SFX))}), "
                             f"a family, speech, music, other, ok or x")
        if label != ev.get("auto_label"):
            ev["est"] = model.estimate(ev["features"], label) if model is not None else None
        ev.update(status="confirmed", label=label)
    if trigger:
        if trigger not in TRIGGERS:
            print(f"note: trigger '{trigger}' is not in the usual list ({', '.join(TRIGGERS)})", file=sys.stderr)
        ev["trigger"] = trigger
    if note:
        ev["note"] = note
    ev["family"] = AN.family_of(ev["label"] or ev["auto_label"], fam)
    AN.add_context(ev, doc.get("cuts", []), doc["duration"], ev["family"])
    ev["likely"] = AN.is_likely(ev)


def cmd_label(args):
    doc = load_doc(args.doc)
    fam = AN.families()
    model = load_model(quiet=True)
    by_id = {ev["id"]: ev for ev in doc["events"]}
    for a in args.assignments:
        if "=" not in a:
            raise SystemExit(f"expected ID=LABEL[:TRIGGER[:NOTE]], got '{a}'")
        ids, rest = a.split("=", 1)
        parts = rest.split(":", 2)
        for i in ids.split(","):
            ev = by_id.get(int(i))
            if ev is None:
                raise SystemExit(f"no event #{i}")
            _apply_label(doc, ev, parts[0], parts[1] if len(parts) > 1 else None,
                         parts[2] if len(parts) > 2 else None, fam, model)
    if args.rest:
        for ev in doc["events"]:
            if ev["status"] == "pending":
                _apply_label(doc, ev, "x" if args.rest == "reject" else "ok", None, None, fam)
    save_doc(doc, args.doc)
    print_table(doc)


def cmd_confirm(args):
    doc = load_doc(args.doc)
    fam = AN.families()
    n = 0
    for ev in doc["events"]:
        if ev["status"] == "pending" and ev.get("confidence", 0) >= args.min and "repeats_on_grid" not in ev.get("flags", []):
            _apply_label(doc, ev, "ok", None, None, fam)
            n += 1
    save_doc(doc, args.doc)
    print(f"confirmed {n} events")
    print_table(doc)


def cmd_add(args):
    doc = load_doc(args.doc)
    src = args.video or doc.get("source")
    if not src or not os.path.exists(src):
        raise SystemExit("the source video is not available; pass --video PATH")
    A = AN.Analysis(AN.load_audio(src))
    ev = {"id": max([e["id"] for e in doc["events"]], default=0) + 1, "start": args.start, "onset": args.start, "hit": args.start,
          "end": args.end,
          "peak": args.peak if args.peak is not None else round((args.start + args.end) / 2, 3), "contrast": 10.0,
          "label": None, "trigger": None, "note": "added by hand", "status": "pending", "flags": ["manual"]}
    ev["features"] = [round(float(v), 4) for v in AN.event_features(A, ev)]
    fam = AN.families()
    model = load_model(quiet=True)
    classify(ev, model, fam)
    _apply_label(doc, ev, args.label, args.trigger, args.note, fam, model)
    doc["events"].append(ev)
    doc["events"].sort(key=lambda e: e["start"])
    save_doc(doc, args.doc)
    print_table(doc)


# -------------------------------------------------------------------- train

def _synth_job(job):
    import beds
    pid, k, seed = job
    rng = np.random.default_rng(seed)
    P = E.load_preset(pid)
    pr, dp = P.get("pitch", {}), P.get("depth", {})
    lo, hi = pr.get("min", -12), pr.get("max", 12)
    m = {"pitch": float(np.clip(rng.normal(pr.get("default", 0), (hi - lo) / 4), lo, hi)),
         "depth": float(rng.uniform(0.05, 0.95)), "length": float(rng.uniform(0.75, 1.35)),
         "tone": float(rng.uniform(-0.4, 0.4)), "space": float(rng.uniform(0.0, 0.5)), "variation": 0.6}
    x, info = E.render(P, seed=int(rng.integers(1 << 30)), **m)
    y = E.resample(x.mean(axis=0), E.SR, AN.ASR)
    if rng.random() < 0.3:
        fc = rng.uniform(6000, 10000)
        y = E.static_eq(y, lambda f: E.filter_mag(f, fc, "lowpass", 0.707, 2), sr=AN.ASR)
    kind = rng.choice(["silence", "room", "speech", "music", "speech+music"], p=[0.15, 0.15, 0.25, 0.25, 0.2])
    pre = 1.5
    total = pre + len(y) / AN.ASR + 1.5
    bed = beds.make(kind, total, rng)
    act = y[np.abs(y) > 0.05 * np.abs(y).max()]
    sfx_rms = math.sqrt(np.mean(act ** 2)) if len(act) else 0.1
    if kind not in ("silence",):
        snr = rng.uniform(3, 20)
        bed *= sfx_rms / (math.sqrt(np.mean(bed ** 2)) + 1e-12) / 10 ** (snr / 20)
    mix = bed.copy()
    o = int(pre * AN.ASR)
    mix[o:o + len(y)] += y[: len(mix) - o]
    mix /= np.abs(mix).max() + 1e-9
    A = AN.Analysis(mix)
    best, ov_best = None, 0.0
    t0, t1 = pre, pre + len(y) / AN.ASR
    for ev in AN.detect_events(A):
        ov = max(0.0, min(ev["end"], t1) - max(ev["start"], t0))
        if ov > ov_best:
            best, ov_best = ev, ov
    if best is None:
        return None
    return AN.event_features(A, best), pid, [m["pitch"], m["depth"], m["length"]]


def _negative_job(job):
    import beds
    kind, seed = job
    rng = np.random.default_rng(seed)
    bed = beds.make(kind, 8.0, rng)
    A = AN.Analysis(bed / (np.abs(bed).max() + 1e-9))
    evs = AN.detect_events(A)
    rng.shuffle(evs)
    label = "speech" if kind == "speech" else "music"
    return [(AN.event_features(A, ev), label) for ev in evs[:4]]


def build_synthetic(n_per=16, seed=1234, workers=None):
    from multiprocessing import Pool
    ids = [p["id"] for p in E.list_presets()]
    rng = np.random.default_rng(seed)
    jobs = [(pid, k, int(rng.integers(1 << 30))) for pid in ids for k in range(n_per)]
    neg = [("speech", int(rng.integers(1 << 30))) for _ in range(40)] + [("music", int(rng.integers(1 << 30))) for _ in range(30)]
    workers = workers or max(1, min(8, (os.cpu_count() or 2)))
    X, y, est = [], [], []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_synth_job, jobs, chunksize=2)):
            if r is not None:
                X.append(r[0])
                y.append(r[1])
                est.append(r[2])
            if (i + 1) % 50 == 0:
                print(f"  synthetic effects {i + 1}/{len(jobs)}", file=sys.stderr)
        for rows in pool.imap_unordered(_negative_job, neg):
            for f, lab in rows:
                X.append(f)
                y.append(lab)
                est.append([np.nan] * 3)
    np.savez_compressed(SYNTH, X=np.array(X), y=np.array(y, dtype="U32"), est=np.array(est, float))
    print(f"synthetic set: {len(y)} examples -> {SYNTH}")


def confirmed_examples():
    X, y = [], []
    for _, doc in all_docs():
        for ev in doc["events"]:
            if ev["status"] == "confirmed" and ev.get("label"):
                X.append(ev["features"])
                y.append(ev["label"])
    return X, y


def loo_accuracy(model, idx):
    """Leave-one-out accuracy (preset level and family level) for rows `idx`."""
    fam = AN.families()
    hit = fhit = 0
    for i in idx:
        z = model.Z[i]
        d = np.sqrt(np.sum((model.Z - z) ** 2, axis=1))
        d[i] = np.inf
        nn = np.argsort(d)[:9]
        votes = defaultdict(float)
        for j in nn:
            votes[model.y[j]] += model.w[j] / (d[j] + 0.5)
        pred = max(votes, key=votes.get)
        hit += pred == model.y[i]
        fhit += AN.family_of(pred, fam) == AN.family_of(model.y[i], fam)
    n = max(1, len(idx))
    return hit / n, fhit / n


def cmd_train(args):
    os.makedirs(KNOW, exist_ok=True)
    if args.rebuild_synthetic or not os.path.exists(SYNTH):
        print("building the synthetic training set (one-off, a few minutes)...", file=sys.stderr)
        build_synthetic(args.per_preset)
    d = np.load(SYNTH)
    X, y, est = list(d["X"]), list(d["y"]), list(d["est"])
    # balance classes so the large speech/music sets don't outvote each effect
    counts = Counter(y)
    typical = np.median([c for lab, c in counts.items() if lab not in AN.NON_SFX] or [1])
    w = [min(1.0, typical / counts[lab]) ** 0.5 for lab in y]
    Xr, yr = confirmed_examples()
    X += Xr
    y += yr
    est += [[np.nan] * 3] * len(yr)
    w += [args.real_weight] * len(yr)
    model = AN.Model(np.array(X, float), np.array(y), np.array(w), np.array(est, float))
    model.save(MODEL)
    n_syn = len(d["y"])
    acc, facc = loo_accuracy(model, range(n_syn))
    print(f"model: {len(y)} examples ({n_syn} synthetic, {len(yr)} confirmed from videos) -> {MODEL}")
    print(f"leave-one-out on synthetic: preset {acc:.0%}, family {facc:.0%}")
    if yr:
        racc, rfacc = loo_accuracy(model, range(n_syn, len(y)))
        print(f"leave-one-out on confirmed video events: preset {racc:.0%}, family {rfacc:.0%}")
    print("classes:", ", ".join(f"{k}:{v}" for k, v in sorted(Counter(y).items())))


# ----------------------------------------------------------------- playbook

def _q(vals, q):
    return round(float(np.percentile(vals, q)), 3) if len(vals) else None


def usable_events(doc, include_auto=None):
    out = []
    for ev in doc["events"]:
        if ev["status"] == "confirmed":
            ok = True
        elif ev["status"] == "pending" and include_auto is not None:
            ok = ev.get("confidence", 0) >= include_auto and "repeats_on_grid" not in ev.get("flags", [])
        else:
            ok = False
        if ok and ev["family"] not in AN.NON_SFX:
            out.append(ev)
    return out


def aggregate(docs, include_auto=None):
    """Statistics of SFX usage across docs (used by PLAYBOOK.md and plan.py)."""
    minutes = sum(d["duration"] for d in docs) / 60.0
    fams = defaultdict(list)
    hook_videos, hook_fams, hook_presets = 0, Counter(), Counter()
    cuts_total = cuts_covered = 0
    n_events = 0
    for d in docs:
        evs = usable_events(d, include_auto)
        n_events += len(evs)
        for ev in evs:
            fams[ev["family"]].append(ev)
        hooks = [ev for ev in evs if ev["context"]["in_hook"]]
        if hooks:
            hook_videos += 1
            hook_fams[hooks[0]["family"]] += 1
            hook_presets[hooks[0]["label"] or hooks[0]["auto_label"]] += 1
        for c in d.get("cuts", []):
            cuts_total += 1
            if any(ev["context"].get("nearest_cut") == c and ev["context"].get("at_cut") for ev in evs):
                cuts_covered += 1
    out = {"videos": len(docs), "minutes": round(minutes, 2), "events": n_events,
           "sfx_per_min": round(n_events / minutes, 2) if minutes else 0.0,
           "cut_coverage": round(cuts_covered / cuts_total, 3) if cuts_total else None,
           "hook_rate": round(hook_videos / len(docs), 3) if docs else None,
           "hook_family": hook_fams.most_common(1)[0][0] if hook_fams else None,
           "hook_presets": dict(hook_presets.most_common()),
           "families": {}}
    for fam, evs in sorted(fams.items(), key=lambda kv: -len(kv[1])):
        at = [ev for ev in evs if ev["context"].get("at_cut")]
        offs = [ev["context"]["offset_to_cut"] for ev in at]
        ests = [ev["est"] for ev in evs if ev.get("est")]
        trig = Counter(ev["trigger"] for ev in evs if ev.get("trigger"))
        presets = Counter(ev["label"] or ev["auto_label"] for ev in evs)
        out["families"][fam] = {
            "count": len(evs), "videos": len({id(d) for d in docs for ev in usable_events(d, include_auto) if ev["family"] == fam}),
            "rate_per_min": round(len(evs) / minutes, 2) if minutes else 0.0,
            "share": round(len(evs) / max(1, n_events), 3),
            "at_cut": round(len(at) / len(evs), 3),
            "offset_median": _q(offs, 50), "offset_iqr": [_q(offs, 25), _q(offs, 75)] if offs else None,
            "anchor": AN.anchor_name(fam),
            "hook_frac": round(sum(ev["context"]["in_hook"] for ev in evs) / len(evs), 3),
            "outro_frac": round(sum(ev["context"]["in_outro"] for ev in evs) / len(evs), 3),
            "dur_median": _q([ev["end"] - ev["start"] for ev in evs], 50),
            "triggers": dict(trig.most_common()),
            "presets": dict(presets.most_common()),
            "cut_presets": dict(Counter(ev["label"] or ev["auto_label"] for ev in at).most_common()),
            "est": {k: _q([e[k] for e in ests], 50) for k in ("pitch", "depth", "length")} if ests else None,
        }
    return out


def _rules(stats):
    lines = []
    for fam, s in stats["families"].items():
        if s["count"] < 3:
            continue
        bits = [f"**{fam}** ({s['count']} uses, {s['rate_per_min']:.1f}/min)"]
        if s["at_cut"] >= 0.5 and s["offset_median"] is not None:
            when = "before" if s["offset_median"] < -0.02 else "after" if s["offset_median"] > 0.02 else "on"
            bits.append(f"{s['at_cut']:.0%} sit on a cut; the {s['anchor']} lands {abs(s['offset_median']):.2f}s {when} the cut")
        else:
            bits.append(f"mostly away from cuts ({s['at_cut']:.0%} on a cut)")
        if s["triggers"]:
            tot = sum(s["triggers"].values())
            bits.append("triggered by " + ", ".join(f"{k} {v / tot:.0%}" for k, v in list(s["triggers"].items())[:3]))
        if s["hook_frac"] >= 0.3:
            bits.append(f"{s['hook_frac']:.0%} in the first 2s (hook)")
        if s["est"]:
            e = s["est"]
            bits.append(f"typical pitch {e['pitch']:+g} st, depth {e['depth']:.2f}")
        lines.append("- " + "; ".join(bits) + ".")
    return lines


def _md_section(title, stats):
    out = [f"## {title}", "",
           f"{stats['videos']} videos, {stats['minutes']:.1f} min, {stats['events']} effects "
           f"({stats['sfx_per_min']:.1f} per minute)."]
    if stats["cut_coverage"] is not None:
        out.append(f"Cuts with an effect on them: {stats['cut_coverage']:.0%}.")
    if stats["hook_rate"] is not None:
        out.append(f"Videos with an effect in the first 2 seconds: {stats['hook_rate']:.0%}"
                   + (f" (usually {stats['hook_family']})." if stats["hook_family"] else "."))
    out.append("")
    if stats["families"]:
        out += ["| family | uses | /min | on cut | offset to cut | triggers | presets | pitch | depth |",
                "|---|---|---|---|---|---|---|---|---|"]
        for fam, s in stats["families"].items():
            trig = ", ".join(f"{k} {v}" for k, v in list(s["triggers"].items())[:3]) or "-"
            pre = ", ".join(f"{k} {v}" for k, v in list(s["presets"].items())[:3])
            off = f"{s['offset_median']:+.2f}s" if s["offset_median"] is not None else "-"
            e = s["est"] or {}
            out.append(f"| {fam} | {s['count']} | {s['rate_per_min']:.1f} | {s['at_cut']:.0%} | {off} | {trig} | {pre} | "
                       f"{e.get('pitch', '-')} | {e.get('depth', '-')} |")
        out.append("")
        rules = _rules(stats)
        if rules:
            out += ["Patterns:", ""] + rules + [""]
    return out


def cmd_playbook(args):
    docs = [d for _, d in all_docs()]
    if not docs:
        raise SystemExit("nothing learned yet - run `learn.py analyze` first")
    by_creator = defaultdict(list)
    for d in docs:
        by_creator[d["creator"]].append(d)
    pb = {"generated": dt.date.today().isoformat(), "include_auto": args.include_auto,
          "overall": aggregate(docs, args.include_auto),
          "creators": {c: aggregate(ds, args.include_auto) for c, ds in sorted(by_creator.items())}}
    with open(PLAYBOOK_JSON, "w", encoding="utf-8") as fh:
        json.dump(pb, fh, ensure_ascii=False, indent=1)
    pending = sum(ev["status"] == "pending" for d in docs for ev in d["events"])
    md = ["# SFX playbook", "",
          f"Learned from {len(docs)} videos by {len(by_creator)} creators (updated {pb['generated']}). "
          "Built from confirmed events" + (f" plus auto-labelled events with confidence >= {args.include_auto}"
                                           if args.include_auto is not None else "") +
          f"; {pending} events are still unreviewed.", "",
          "Offsets are measured from the cut to the effect's anchor (the transient for impacts, the peak for whooshes "
          "and risers, the onset for everything else). Negative = before the cut.", ""]
    md += _md_section("All creators", pb["overall"])
    for c, s in pb["creators"].items():
        md += _md_section(c, s)
    with open(PLAYBOOK_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {PLAYBOOK_MD} and {PLAYBOOK_JSON}")


def cmd_status(args):
    docs = all_docs()
    by = defaultdict(lambda: Counter())
    for _, d in docs:
        c = by[d["creator"]]
        c["videos"] += 1
        c["minutes"] += d["duration"] / 60
        for ev in d["events"]:
            c[ev["status"]] += 1
    if not docs:
        print("no videos analysed yet")
    for cr, c in sorted(by.items()):
        print(f"{cr}: {c['videos']} videos, {c['minutes']:.1f} min, {c['confirmed']} confirmed, "
              f"{c['pending']} pending, {c['rejected']} rejected")
    print(f"model: {'yes' if os.path.exists(MODEL) else 'missing - run learn.py train'}; "
          f"playbook: {'yes' if os.path.exists(PLAYBOOK_JSON) else 'not built yet'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze")
    p.add_argument("videos", nargs="+")
    p.add_argument("--creator", required=True)
    p.add_argument("--review-dir", default="sfx_review")
    p.add_argument("--sheets", choices=["likely", "all", "none"], default="likely")
    p.add_argument("--sensitivity", type=float, default=1.0, help=">1 finds quieter events")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("review")
    p.add_argument("doc")
    p.set_defaults(fn=cmd_review)
    p = sub.add_parser("label")
    p.add_argument("doc")
    p.add_argument("assignments", nargs="*")
    p.add_argument("--rest", choices=["reject", "ok"], help="apply to every event still pending")
    p.set_defaults(fn=cmd_label)
    p = sub.add_parser("confirm")
    p.add_argument("doc")
    p.add_argument("--min", type=float, default=0.7)
    p.set_defaults(fn=cmd_confirm)
    p = sub.add_parser("add")
    p.add_argument("doc")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--peak", type=float)
    p.add_argument("--label", required=True)
    p.add_argument("--trigger")
    p.add_argument("--note")
    p.add_argument("--video")
    p.set_defaults(fn=cmd_add)
    p = sub.add_parser("train")
    p.add_argument("--rebuild-synthetic", action="store_true")
    p.add_argument("--per-preset", type=int, default=40)
    p.add_argument("--real-weight", type=float, default=3.0)
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("playbook")
    p.add_argument("--include-auto", type=float, default=None, metavar="CONF")
    p.set_defaults(fn=cmd_playbook)
    p = sub.add_parser("status")
    p.set_defaults(fn=cmd_status)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
