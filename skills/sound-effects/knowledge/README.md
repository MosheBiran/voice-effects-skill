# knowledge/

What the skill has learned. Written by `scripts/learn.py`; safe to commit (no media).

| file | what |
|---|---|
| `model.npz` | event classifier: synthetic examples + every confirmed video event, plus pitch/depth estimators |
| `synthetic.npz` | features of the presets rendered over voice/music/room beds (rebuild after adding presets: `learn.py train --rebuild-synthetic`) |
| `videos/<creator>/<video>.json` | one analysed video: cuts, detected events, labels, triggers, features |
| `PLAYBOOK.md`, `playbook.json` | how the learned creators use effects (`learn.py playbook`), read by `plan.py` |

No videos have been learned yet when `videos/` is empty. Analyse some with `learn.py analyze`.
