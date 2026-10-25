# voice-effects-skill

This repo is the `sound-effects` skill (in `skills/sound-effects/`, also linked as `.claude/skills/sound-effects`)
packaged as a Claude Code plugin (`.claude-plugin/`). Follow `skills/sound-effects/SKILL.md` for any sound-effect,
learning or SFX-placement request.

- Scripts need only Python 3 + numpy (ffmpeg for video). Keep it that way: no scipy/librosa/matplotlib.
- Never commit videos or audio (see `.gitignore`). Commit what was learned: `skills/sound-effects/knowledge/`
  (`videos/**/*.json`, `model.npz`, `synthetic.npz`, `PLAYBOOK.md`, `playbook.json`).
- After adding or changing presets, run `python3 skills/sound-effects/scripts/learn.py train --rebuild-synthetic`
  so the classifier knows them, and commit the new `model.npz` / `synthetic.npz`.
- Smoke test: `python3 skills/sound-effects/scripts/sfx.py render impact --png --out /tmp/sfx_check`.
