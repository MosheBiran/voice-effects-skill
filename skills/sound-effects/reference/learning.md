# How the learning works (and where it is weak)

## Detection

1. Audio is decoded to mono 22.05 kHz; STFT 1024 / hop 256 (≈ 11.6 ms per frame).
2. Energy in 24 log-spaced bands (40 Hz-10.5 kHz). The **bed** (music, voice-over, room) is estimated per band as the
   25th percentile over a 4 s window. **Novelty** = mean of the 6 bands that rise most above the bed (dB).
3. Regions where novelty passes `max(8 dB, 20th percentile + 8 dB)` (divided by `--sensitivity`) become events. Hysteresis is half that.
   Regions closer than 0.1 s merge. A region splits where a new sound rises ≥ 12 dB after the previous one decayed ≥ 6 dB.
4. Each event stores `start`, `onset` (strong rise), `hit` (largest broadband transient), `peak` (max novelty), `end`.

Long, slow builds (risers, drones) partly sink into the bed estimate, so their detected start is late. Their *peak* is still right,
which is why risers are anchored on the peak.

## Features (24)

Duration, attack position, rise correlation, envelope peaks/s, envelope jitter, 8 band shares (sub ... air), spectral flatness,
centroid, centroid change start→end, centroid arch (middle vs ends), low-band pitch drift, harmonicity (autocorrelation),
tonality (peakiness), contrast, transient strength, decay ratio, bandwidth. All are measured on the bed-subtracted spectrum.

## Classifier

Distance-weighted kNN (k = 9) on standardised, weighted features. Training data:

- **Synthetic**: every preset rendered 40× with random pitch/depth/length/tone/space/variation, mixed over silence, room tone,
  synthetic voice-over, synthetic music or both at 3-20 dB SNR, then run through the same detector. Plus speech and music events
  from bare beds. Classes are balanced (√ weighting) so speech/music don't outvote effects.
- **Confirmed events** from real videos (weight ×3). This is the part that adapts to real creators: the more you review, the better
  the auto labels get for that style of content.

`learn.py train` prints leave-one-out accuracy. On the synthetic set it is about 75% per preset and 83% per family; close relatives
(boom/impact/sub drop, whoosh/swipe, click/typing/shutter) account for most errors. Expect lower on real videos with dense music until
some are confirmed.

Pitch/depth/length estimates come from a per-preset ridge regression fitted on the synthetic renders. They are rough
(≈ ±3 st pitch, ±0.16 depth), so `plan.py` moves only halfway from the preset default toward them.

## Context

- **Cuts**: ffmpeg scene score per frame (scaled to 192 px). A cut is a local peak ≥ 0.25, or ≥ 0.12 and well above its
  neighbourhood (≥ 6× the local median + 0.08), at least 0.2 s from the next cut. Soft dissolves and whip-pans may be missed;
  the overview shows the cuts that were found as white dashed lines.
- **Anchor** per family: impact and sub drop → `hit`, whoosh and riser → `peak`, everything else → `onset`.
  `offset_to_cut` = anchor − nearest cut (negative = before the cut); `at_cut` = |offset| ≤ 0.3 s.
- `in_hook` = anchor < 2 s; `in_outro` = within the last 3 s.
- `repeats_on_grid` flag: near-identical events on a regular time grid (usually a music beat, sometimes a creator who puts the same
  whoosh on beat-synced cuts; check the frames).

## Playbook

`learn.py playbook` aggregates reviewed events (optionally `--include-auto CONF`):
effects per minute, cut coverage (share of cuts with an effect), hook rate and hook presets, and per family: count, rate,
share, share on cuts, median offset and IQR, triggers, presets used (overall and on cuts), duration, median pitch/depth/length.
Everything is computed overall and per creator. `plan.py` reads the same numbers.

## Improving results

- Review more videos per creator (5-10 short videos give stable offsets and rates).
- Too many false events under music: `--sensitivity 0.8`. Quiet effects missed: `--sensitivity 1.3`.
- After adding presets, `learn.py train --rebuild-synthetic` (about 2 min) so the classifier knows them.
- A creator with a signature sound: design it as a preset (A.5 in SKILL.md), rebuild, and future analyses will recognise it.
