#!/usr/bin/env python3
"""Sound-effect generator CLI.

  sfx.py list [--json]                      presets with pitch range and what depth does
  sfx.py show <preset>                      print a preset's full JSON
  sfx.py render <preset|spec.json> [opts]   render WAV (+ sidecar JSON, + PNG with --png)
  sfx.py save-preset <spec.json> [--force]  validate a custom spec and add it to the library

Render options:
  --pitch ST  --depth 0..1  --length X  --tone -1..1  --drive 0..1  --space 0..1
  --width 0..1.5  --variation 0..1  --seed N  --variants K  --mute a,b  --solo a,b
  --out DIR  --name BASENAME  --png
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sfx_engine as E  # noqa: E402
import viz  # noqa: E402

MACROS = ("pitch", "depth", "length", "tone", "drive", "space", "width", "variation")


def cmd_list(args):
    presets = E.list_presets()
    if args.json:
        print(json.dumps([{k: p.get(k) for k in ("id", "family", "name", "he", "use", "sync", "pitch", "depth")}
                          for p in presets], ensure_ascii=False, indent=1))
        return
    fam = None
    for p in sorted(presets, key=lambda p: (p.get("family", ""), p["id"])):
        if p.get("family") != fam:
            fam = p.get("family")
            print(f"\n[{fam}]")
        pr, dp = p.get("pitch", {}), p.get("depth", {})
        print(f"  {p['id']:<14} {p.get('name', '')} / {p.get('he', '')}")
        print(f"  {'':<14} pitch {pr.get('min', -12):+d}..{pr.get('max', 12):+d} (default {pr.get('default', 0):+g})"
              f" | depth default {dp.get('default', 0.5):.2f}: {dp.get('hint', '')}")
        print(f"  {'':<14} use: {p.get('use', '')}")


def cmd_show(args):
    print(json.dumps(E.load_preset(args.preset), ensure_ascii=False, indent=2))


def _fmt(v):
    s = f"{abs(v):g}".replace(".", "_")
    return "0" if v == 0 else ("up" if v > 0 else "dn") + s


def render_to_files(P, out_dir, seed=0, name=None, png=False, mute=(), solo=None, **macros):
    """Render and write <name>.wav + <name>.json (+ .png). Returns (wav_path, info)."""
    x, info = E.render(P, seed=seed, mute=mute, solo=solo, **macros)
    m = info["macros"]
    base = name or f"{P.get('id', 'custom')}_st{_fmt(m['pitch'])}_d{int(round(m['depth'] * 100))}_s{seed}"
    wav = os.path.join(out_dir, base + ".wav")
    E.write_wav(wav, x)
    info["file"] = os.path.basename(wav)
    with open(os.path.join(out_dir, base + ".json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    if png:
        viz.sfx_card(x, E.SR, os.path.join(out_dir, base + ".png"),
                     f"{P.get('id', 'custom')}  PITCH {m['pitch']:+g}  DEPTH {m['depth']:.2f}  SEED {seed}",
                     sync=info["sync"])
    return wav, info


def cmd_render(args):
    P = E.load_preset(args.preset)
    macros = {k: getattr(args, k) for k in MACROS}
    pr = P.get("pitch", {})
    if args.pitch is not None and not pr.get("min", -24) <= args.pitch <= pr.get("max", 24):
        print(f"note: pitch {args.pitch:+g} is outside {P.get('id')}'s usual range "
              f"{pr.get('min'):+d}..{pr.get('max'):+d}", file=sys.stderr)
    if args.variants > 1 and macros["variation"] is None:
        macros["variation"] = 0.5
    mute = tuple(s for s in (args.mute or "").split(",") if s)
    solo = tuple(s for s in (args.solo or "").split(",") if s) or None
    os.makedirs(args.out, exist_ok=True)
    for k in range(args.variants):
        seed = args.seed + k
        name = None
        if args.name:
            name = args.name if args.variants == 1 else f"{args.name}_v{k + 1}"
        wav, info = render_to_files(P, args.out, seed=seed, name=name, png=args.png, mute=mute, solo=solo, **macros)
        m = info["macros"]
        print(f"{wav}  dur={info['duration']:.2f}s  sync={info['sync']:.3f}s ({info['sync_mode']})  "
              f"pitch={m['pitch']:+g} depth={m['depth']:.2f} length={m['length']:g}")


def cmd_save(args):
    with open(args.spec, encoding="utf-8") as fh:
        P = json.load(fh)
    for key in ("id", "family", "layers"):
        if key not in P:
            sys.exit(f"spec is missing '{key}'")
    if not all(c.isalnum() or c == "_" for c in P["id"]):
        sys.exit("id must be letters, digits and underscores")
    E.render(P)  # raises on a broken spec
    dest = os.path.join(E.PRESET_DIR, P["id"] + ".json")
    if os.path.exists(dest) and not args.force:
        sys.exit(f"{dest} exists (use --force to replace)")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(P, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"saved {dest}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("show")
    p.add_argument("preset")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("render")
    p.add_argument("preset", help="preset id or path to a spec JSON")
    for k in MACROS:
        p.add_argument(f"--{k}", type=float)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", type=int, default=1)
    p.add_argument("--mute")
    p.add_argument("--solo")
    p.add_argument("--out", default="sfx_out")
    p.add_argument("--name")
    p.add_argument("--png", action="store_true")
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("save-preset")
    p.add_argument("spec")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_save)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
