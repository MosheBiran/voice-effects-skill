# Sound Effects Skill · סקיל אפקטים קוליים ל-Claude

סקיל ל-Claude שעושה שלושה דברים:

1. **מייצר אפקטים קוליים עשירים** – 22 אפקטים (ווש, אימפקט, רייזר, סאב דרופ, פופ, גליץ', נצנוץ, מתכת ועוד), כל אחד בנוי מכמה שכבות (סאב, גוף, טרנזיינט, אוויר, זנב, טון) עם סוויפים, פילטרים, LFO, ריוורב ודיסטורשן.
   - **פיץ'** – מזיז את כל האפקט למעלה/למטה, בטווח שמתאים לכל אפקט.
   - **עומק** – משתנה לפי האפקט: באימפקט מוסיף בום סאב, זנב ארוך וחדר גדול; בווש הופך אותו לאיטי, נמוך וכבד; ברייזר בונה ארוך יותר ומתחיל נמוך יותר.
2. **לומד מסרטונים של יוצרים** – מזהה את האפקטים בסרטון, מסווג אותם, מודד מתי הם יושבים ביחס לקאטים ומה קורה בתמונה באותו רגע. אחרי ש-Claude עובר על הזיהויים (עם פריימים וספקטרוגרמה), הוא בונה **Playbook**: כמה אפקטים בדקה, על כמה מהקאטים יש אפקט, מה שמים בהוק, איזה אפקט על איזה טריגר (קאט, טקסט, ריביל, לפני/אחרי…), וכמה לפני/אחרי הקאט. כל זיהוי שמאושר משפר את המסווג לסרטון הבא.
3. **שם אפקטים על הסרטון שלך** – לפי הסגנון שנלמד: מזהה קאטים, מציע רשימת קיוז, ומרנדר טראק SFX אחד מסונכרן לסרטון, קבצי אפקט נפרדים, מרקרים ל-DaVinci Resolve ופריוויו.

## דוגמאות לבקשות

- "תן לי אימפקט עמוק לרגע של הלפני/אחרי, 3 גרסאות"
- "הווש הזה גבוה מדי – תוריד פיץ' ותעשה אותו יותר עמוק"
- "הנה 5 סרטונים של @creator – תלמד מתי הוא שם אפקטים"
- "תשים אפקטים על הסרטון הזה בסגנון של שני היוצרים שלמדת"

## התקנה

**Claude Code** (כפלאגין):

```
/plugin marketplace add MosheBiran/voice-effects-skill
/plugin install sound-effects@voice-effects-skill
```

**claude.ai / אפליקציית Claude** (כסקיל): לכווץ את התיקייה `skills/sound-effects` ל-ZIP ולהעלות ב-Settings › Capabilities › Skills.

```
cd skills && zip -r sound-effects.zip sound-effects
```

דרישות: Python 3 עם numpy. לניתוח סרטונים ולתכנון על וידאו צריך גם ffmpeg/ffprobe. יצירת אפקטים עובדת גם בלעדיהם.

## שימוש ישיר מהטרמינל

```bash
S=skills/sound-effects/scripts
python3 $S/sfx.py list                                            # כל האפקטים, טווחי פיץ' ומה העומק עושה
python3 $S/sfx.py render impact --pitch -3 --depth 0.8 --variants 3 --png --out sfx_out

python3 $S/learn.py analyze v1.mp4 v2.mp4 --creator "Creator A"   # זיהוי + תמונות לבדיקה
python3 $S/learn.py label skills/sound-effects/knowledge/videos/creator-a/v1.json 2,7=ok:transition 18=pop:text --rest reject
python3 $S/learn.py train && python3 $S/learn.py playbook

python3 $S/plan.py suggest my_edit.mp4 --style "Creator A" --out cues.json
python3 $S/plan.py render cues.json --out sfx_render --preview
```

## מבנה

```
skills/sound-effects/
  SKILL.md              ההוראות ל-Claude
  presets/              ספריית האפקטים (JSON לכל אפקט)
  scripts/              מנוע הסינתזה, ניתוח, למידה ותכנון
  reference/            פורמט השכבות ואיך הלמידה עובדת
  knowledge/            המסווג, מה שנלמד מסרטונים, PLAYBOOK.md
```

## מגבלות

הזיהוי האוטומטי הוא סבב ראשון. בבדיקות על חומר סינתטי הוא צודק בכ-80% ברמת משפחת האפקט, ופחות מתחת למוזיקה רועשת. לכן יש שלב בדיקה, וה-Playbook נבנה רק מזיהויים מאושרים. הערכת הפיץ'/עומק של אפקט מתוך סרטון היא גסה (בערך ±3 חצאי טונים). הסרטונים עצמם לא נשמרים בריפו, רק מה שנלמד מהם.
