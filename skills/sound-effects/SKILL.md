---
name: sound-effects
description: Generate rich, layered sound effects (whoosh, impact, riser, sub drop, pop, glitch, shimmer and more) whose pitch and depth adapt to each effect; learn from other creators' videos when and how they use each effect; and plan and render synced SFX for the user's own video (cue sheet, SFX stem WAV, DaVinci Resolve markers, preview). Use whenever the user asks for sound effects / SFX / sound design for a video, wants an effect richer, deeper, higher or lower, shares reference videos to learn SFX timing from, or wants effects placed on their edit. Hebrew triggers - אפקטים קוליים, סאונד אפקטים, אפקט סאונד, ווש, אימפקט, רייזר, ללמוד מסרטונים.
---

# Sound effects: generate, learn, place

Three jobs, one toolkit (`scripts/`, Python 3 + numpy; ffmpeg/ffprobe for anything with video):

| Job | Script | Output |
|---|---|---|
| A. Make an effect | `sfx.py` | 48 kHz / 24-bit stereo WAV + JSON sidecar (sync point) + optional spectrogram PNG |
| B. Learn from creators' videos | `learn.py` | per-video event JSON in `knowledge/videos/`, review PNGs, `knowledge/PLAYBOOK.md` |
| C. Put effects on the user's video | `plan.py` | `cues.json` → SFX stem, per-cue WAVs, `cues.csv`, `markers.edl`, `preview.mp4` |

`SKILL_DIR` below means the folder holding this file. Run scripts as `python3 SKILL_DIR/scripts/<name>.py ...`.
Before the first run in a session, check `python3 -c "import numpy"` and `ffmpeg -version`; install what is missing
(`pip install numpy`, `apt-get install -y ffmpeg`) or tell the user what is needed.
Answer the user in their language (often Hebrew). Write output files into the working directory (or wherever the user wants), never into `SKILL_DIR` except `knowledge/` and `presets/`.

## A. Make an effect

1. Pick a preset: `sfx.py list` prints every preset with its family, pitch range, what *depth* does for it, and when to use it.
   If `SKILL_DIR/knowledge/PLAYBOOK.md` exists, read it first. Learned creators' typical pitch and depth for that effect are a good starting point.
2. Choose pitch and depth for the moment, not just the effect:
   - **pitch** (semitones) shifts every tonal/filter layer. Up = lighter, faster, more "UI"; down = bigger, heavier, slower. Stay inside the preset's range unless asked.
   - **depth** (0-1) is effect-specific. Each layer has a role (sub, body, transient, air, tail, tone) and depth rebalances them. More depth means more sub, a longer tail, a lower body, a bigger room and a slightly longer, lower sound. Less depth makes it tight and dry. The `list` output says what it means per preset.
   - Rules of thumb: hero reveal / logo / before-after payoff → depth 0.7-0.9, pitch −2…−5. Fast cuts, text, UI → depth 0.2-0.4, pitch 0…+5. Phone-first vertical video: keep `--drive` 0.2-0.4 so the low end has harmonics phones can play.
   - Other macros: `--length` (time stretch), `--tone` (−1 dark … +1 bright), `--drive`, `--space` (reverb), `--width`, `--variation` with `--seed`.
3. Render a few options and look at them:
   `python3 SKILL_DIR/scripts/sfx.py render whoosh --pitch -2 --depth 0.6 --variants 3 --png --out sfx_out`
   Read the PNGs (waveform + log spectrogram; green dashed line = sync point) to confirm the shape matches the brief
   (a whoosh should arch, a riser should climb to the sync line, an impact should have a hard left edge).
4. Deliver the WAV paths and say in one line why that pitch/depth. The JSON sidecar's `sync` is the time inside the file
   that should land on the cut (onset for hits, peak for whooshes and risers).
5. Nothing fits? Write a custom spec (layer format in `reference/layer-spec.md`; copy a close preset with `sfx.py show <id>` and edit),
   render it with `sfx.py render path/to/spec.json`, iterate, and if the user likes it, `sfx.py save-preset spec.json` so it joins the library.
   Use `--mute` / `--solo` with layer names to hear (and show) what each layer contributes.

## B. Learn from creators' videos

Goal: find when each creator uses which effect (on which cut, on what visual trigger, how far before/after the cut, how often, with what pitch/depth) and keep that knowledge in `SKILL_DIR/knowledge/`.

Getting the videos: local files or uploads; Google Drive files via the Drive connector (download, then analyse); URLs only with `yt-dlp` if it is installed and the network allows it.
Videos are analysed, never stored: don't copy them into `SKILL_DIR` or commit them.

1. **Analyse** (one creator per call; any number of files):
   `python3 SKILL_DIR/scripts/learn.py analyze a.mp4 b.mp4 --creator "Name" --review-dir sfx_review`
   It prints a table of detected events: auto label, confidence, top alternatives, offset to the nearest cut. Rows marked `*` are likely effects.
   Speech syllables and music hits are detected too and mostly come out as `speech` / `music` or low confidence.
2. **Review. This is the step that makes the knowledge trustworthy.** Read `sfx_review/<creator>/<video>/overview.png`
   (whole timeline: spectrogram, white dashed lines = cuts, coloured boxes = likely effects with `#id label`), then each `ev_NNN.png`
   (frames 0.5 s before / at / after the event plus a zoomed spectrogram). For every `*` row decide:
   is it really an effect, which preset/family it is, and what visual **trigger** caused it (you can see it in the frames).
   Scan the overview for effects the detector missed (labelled grey or absent) too.
3. **Label** in one call per video (`ok` keeps the auto label, `x` rejects; `--rest reject` rejects everything not mentioned):
   `learn.py label knowledge/videos/name/a.json 2,7=ok:transition 18=pop:text 22=reveal:before_after 31=x --rest reject`
   Missed effect: `learn.py add <json> --start 12.4 --end 13.1 --label riser --trigger reveal`.
   Labels: any preset id, a family (impact, sub_drop, whoosh, riser, downlifter, ui, chime, glitch, texture), `speech`, `music`, `other`.
   Triggers: cut, transition, text, number, zoom, reveal, before_after, logo, product, face, gesture, punchline, beat, hook, cta, screen, motion, other.
   `learn.py confirm <json> --min 0.8` bulk-accepts confident auto labels. Use it only when the user prefers speed over precision.
4. **Retrain and summarise**: `learn.py train` (adds every confirmed event to the classifier, weighted ×3, so the next analysis is better),
   then `learn.py playbook` → `knowledge/PLAYBOOK.md` + `playbook.json`. Tell the user the findings in plain words:
   effects per minute, share of cuts with an effect, hook habits, which family goes on which trigger, typical offsets, and differences between creators.
5. **Keep it**: if `SKILL_DIR` is inside a git checkout of the skill repo, commit `knowledge/` (JSON, PNG-free) and push when the user wants the learning saved.
   `learn.py status` shows what has been learned so far.

Be straight about accuracy. Auto labels are a first pass (about 80% right at the family level on synthetic tests, lower under loud music),
offsets are measured to ~12 ms, and pitch/depth estimates are rough (±3 semitones). The playbook is built only from reviewed events unless
`playbook --include-auto 0.75` is used. Details and limits: `reference/learning.md`.

## C. Put effects on the user's video

1. `python3 SKILL_DIR/scripts/plan.py suggest edit.mp4 --style "Creator A,Creator B" --intensity 1.0 --out cues.json --review-dir sfx_plan`
   (`--style all` = everyone learned; with nothing learned it falls back to common editing defaults and says so). It detects cuts and applies
   the playbook: hook effect, a riser landing on the strongest cut, effects on cuts at the learned rate, offset, preset and pitch/depth.
2. Read `sfx_plan/plan_overview.png` and the `cut_NNN.png` sheets (frames before/after each cut). Then edit `cues.json` where the picture
   asks for it. Cuts only cover transitions: add pops/clicks where text or numbers appear, impacts on reveals, swap effects, adjust
   `pitch`/`depth`/`length`/`gain_db`, or set `"mute": true`. Cue fields: `time` (where the effect's sync point lands, seconds),
   `effect` (preset id), `pitch`, `depth`, `length`, `gain_db`, `seed`, optional `tone`/`drive`/`space`/`width`/`variation`.
3. `python3 SKILL_DIR/scripts/plan.py render cues.json --out sfx_render --preview`
   → `sfx_stem.wav` (same length as the video: drop it on a track at the first frame), `cues/NN_effect.wav`, `cues.csv`,
   `markers.edl` (DaVinci Resolve: Timeline › Import › Timeline Markers from EDL; timecode starts 01:00:00:00, change with `--tc-start`),
   and `preview.mp4` (video + original audio + SFX) for a quick check.
4. Deliver the stem and the preview, list the cues briefly, and offer tweaks ("deeper impact at 0:07", "fewer whooshes").

## Files

- `presets/*.json`: effect library (one file per effect; `family`, `use`, `sync`, pitch range, depth meaning, layers).
- `scripts/sfx_engine.py`: synthesis engine (oscillators with unison, FM, coloured noise, dust, time-varying filters, LFOs, repeats, reverse, drive, EQ, generated-IR reverb, echo, width).
- `scripts/analysis.py`: detection, features, classifier, scene cuts, review images. `scripts/beds.py`: synthetic speech/music used for training.
- `knowledge/`: `model.npz` (classifier), `synthetic.npz` (training set from the presets; rebuild with `learn.py train --rebuild-synthetic` after adding presets), `videos/<creator>/*.json`, `PLAYBOOK.md`, `playbook.json`. Override the location with `SFX_KNOWLEDGE_DIR`.
