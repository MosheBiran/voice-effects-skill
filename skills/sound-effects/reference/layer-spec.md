# Preset / custom spec format

A preset is one JSON object. Copy a close one (`sfx.py show impact > my.json`), edit, render with
`sfx.py render my.json --png`, and save with `sfx.py save-preset my.json` when it is good.

```json
{
  "id": "door_slam",                 // letters, digits, underscores; file name in presets/
  "family": "impact",                // impact | sub_drop | whoosh | riser | downlifter | ui | chime | glitch | texture
  "name": "Door slam", "he": "טריקת דלת",
  "use": "When to reach for it (shown by sfx.py list)",
  "sync": "onset",                   // onset | peak | hit  (what lands on the cut, see below)
  "pitch": {"default": 0, "min": -12, "max": 7},
  "depth": {"default": 0.6, "hint": "what depth does here", "time": 1.3, "pitch": -2, "low_db": 9},
  "length": 1.0, "tone": 0.0, "drive": 0.2, "width": 1.0,
  "space": {"size": 1.4, "mix": 0.25, "damp": 0.5},
  "echo": {"time": 0.15, "fb": 0.35, "mix": 0.2, "pingpong": true},
  "highpass": 22,
  "layers": [ ... ]
}
```

## Macros and how depth is wired

Render macros: `pitch` (st), `depth` (0-1), `length`, `tone` (−1..1), `drive`, `space` (reverb mix), `width`, `variation` (+ `seed`).

`depth` acts in three places:

1. **Global time/pitch**: with c = 2·depth − 1, time is multiplied by `depth.time` ^ c and pitch moves by `depth.pitch` · c semitones.
   With time 1.3 and pitch −2, depth 1 is 30% longer and 2 st lower than depth 0.5; depth 0 is the reverse.
2. **Per layer role** (gain × decay × frequency):

   | role | depth 0 → 1 |
   |---|---|
   | `sub` | gain 0.15 → 1.85, decay 0.7 → 1.6, slightly lower |
   | `body` | decay 0.85 → 1.35, about −2 st lower at 1 |
   | `tail` | gain 0.3 → 1.6, decay 0.6 → 1.8 |
   | `transient` | gain 1.15 → 0.7 (softer attack when deep) |
   | `air` | gain 1.1 → 0.55 |
   | `tone` | decay 0.9 → 1.2, pitch untouched (musical parts) |

3. **Master**: low shelf at 110 Hz of `depth.low_db` · (depth − 0.3) dB, reverb size × (0.5 + depth), more reverb mix and pre-delay.

So a new preset gets sensible depth behaviour just by giving each layer the right `role`.

## Layers

Common keys:

| key | meaning |
|---|---|
| `name` | label used by `--mute` / `--solo` |
| `role` | sub, body, transient, air, tail, tone (see above) |
| `type` | `osc`, `fm`, `noise`, `dust` |
| `env` | `{"a": attack s, "h": hold s, "d": decay s, "ac": attack curve (>1 = swell), "dc": decay curve (higher = faster drop)}` |
| `gain` | linear level before the master normalises to −1 dBFS |
| `delay` | start time in seconds (stretched by length/depth) |
| `pan`, `pan_to` | −1..1, sweeps over the layer (mono sources) |
| `track` | how much global pitch moves this layer (1 = fully, 0 = fixed) |
| `drive` | per-layer tanh saturation 0-1 |
| `crush` | bit depth (e.g. 4) for lo-fi |
| `hold` | sample-and-hold factor (downsampling grit) |
| `reverse` | play the rendered layer backwards; reversed layers line up their ends |
| `repeat` | `{"count": 7, "interval": 0.085, "decay": 0.97, "jitter": 0.35}` |
| `lfo` | one or a list of `{"target": "amp"|"freq", "shape": "sine"|"square"|"saw"|"random", "rate": Hz, "to": end rate, "depth": amp 0-1 / freq semitones, "duty", "smooth", "phase"}` |
| `filter` | one or a list of filters (below) |
| `normalize` | default true: keep the layer's level constant through its filter |

Pitch curves (`osc`, `fm`) and filter cutoffs use either a single sweep
`"f0": 120, "f1": 40, "ft": 0.4, "fc": 0.6` (from f0 to f1 over ft seconds; fc < 1 = fast at first, > 1 = accelerating)
or explicit points `"points": [[0, 450], [0.36, 2600], [0.85, 600]]` (seconds, Hz; log-interpolated). Use points for arches like a whoosh.

Type-specific:

- `osc`: `wave` sine | triangle | saw | square (band-limited), `voices` (unison count), `detune` (cents spread), `spread` (stereo).
- `fm`: carrier follows the pitch curve; `ratio` (modulator/carrier), `index`, `idecay` (seconds for the index to fall, bell-like).
- `noise`: `color` white | pink | brown | blue | violet, `stereo: true` for decorrelated L/R.
- `dust`: random clicks, `density` per second (crackle, sparkle, debris). Usually `"normalize": false`.

Filters: `{"type": "lowpass"|"highpass"|"bandpass"|"notch", "f0", "f1", "ft", "fc" | "points", "q", "order": 1|2, "lfo": {"rate", "depth" (octaves)}}`.

## Sync modes

- `onset`: first strong transient. Impacts, clicks, pops, chimes.
- `peak`: loudest moment. The middle of a whoosh, the top of a riser or swell.
- `hit`: start of the biggest sudden jump. A riser-into-impact lands its impact on the cut.

`render` reports the sync time; `plan.py render` aligns that point to each cue's `time`.

## Design tips

- Big sounds = several narrow layers rather than one wide one: sub sine sweep + filtered brown noise body + short white-noise crack + pink-noise tail.
- Movement makes things feel expensive: sweep filters and pitch, pan across, add slow LFOs on long sounds.
- Phones can't play below ~100 Hz: give sub layers some `drive` or a `body` layer an octave up.
- Check every new design's PNG: the spectrogram should show the gesture you intend (arch, climb, hard attack, ringing partials).
