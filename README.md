# transcribe — local Hebrew speech-to-text

Give it a podcast episode, a YouTube link or an audio file; get back a Hebrew transcript as plain text. Everything runs **on your own machine** (Apple Silicon via MLX, anything else via faster-whisper) with the open [ivrit.ai](https://huggingface.co/ivrit-ai) Whisper models. No audio leaves your computer, no API key, no cost.

**Why:** you want to *talk with your AI* about a podcast or lecture. The AI probably can't hear it, and even if it was trained on it, it won't quote it back to you. So you make the transcript yourself, locally, for your own use — and hand the text to your agent. Background: [the post](https://ohadzvishaboo.pages.dev/tech/local-transcribe/) (Hebrew).

## Install
Needs `uv` and `ffmpeg` (macOS: `brew install uv ffmpeg`).

```bash
uv tool install --python 3.12 git+https://github.com/ohadzs/transcribe
```

The first run downloads the model from Hugging Face (~1.6 GB).

## Use
```bash
transcribe "https://open.spotify.com/episode/..."
transcribe "https://www.youtube.com/watch?v=..."
transcribe ~/Downloads/lecture.m4a
```

| Flag | What it does |
|---|---|
| `--lang <code>` | spoken language (`he`, `en`, `fr`, …). Default `auto`: detected from a 30-second sample. Hebrew uses the ivrit.ai models, anything else stock Whisper (speaker labels become `Speaker 1:`) |
| `--accurate` | `large-v3` instead of `large-v3-turbo`: slightly better, ~3x slower |
| `--local` | keep the transcript on this machine (`~/Documents/Transcripts`) instead of uploading it |
| `-o <dir>` | local output folder (implies `--local`) |
| `--title <name>` | output file name instead of the episode title |
| `--no-speakers` | don't mark who is speaking (on by default, see below); ~10-20% faster |
| `--num-speakers N` | the number of speakers, if you know it (`1` = no labels, skip detection) |
| `--no-captions` | transcribe a YouTube video even if it has manual captions (see below) |

Output: `<title>.txt` (clean paragraphs) and `<title> - timestamps.txt` (one line per segment with start/end times).

### Speaker labels (on by default)
In interviews and panels every change of speaker starts a new paragraph labelled `דובר 1:` / `דובר 2:` (numbered in order of first appearance), and every line of the timestamps file carries its speaker. When there is only one speaker (a lecture, a monologue, a voice memo) the transcript has no labels at all. `--no-speakers` turns detection off.

How: a speaker-embedding model (NVIDIA TitaNet-small, ONNX via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), ~40 MB, downloaded once from the sherpa-onnx GitHub releases; **no Hugging Face token**) runs over 1.5-second windows of the audio on the CPU *while* Whisper transcribes on the GPU. The windows are grouped by spectral clustering, which also guesses how many speakers there are (groups that sound like the same voice are merged back, so a solo talk stays one speaker), and each speaker change is moved to the nearest sentence end or pause. Short interjections ("כן", "אוקיי") are folded into the surrounding speaker. It gets most turns right in a normal two- or three-person conversation; overlapping speech and very quick back-and-forth are where it slips. If it finds the wrong number of speakers, pass `--num-speakers`.

Where it goes: by default both files are uploaded with [rclone](https://rclone.org) to `gdrive:Transcripts` (your Google Drive, so your AI can read them from anywhere) and the local copies are deleted. Set `TRANSCRIBE_REMOTE` to use another rclone remote/folder. If rclone is missing or the upload fails, the files stay in `~/Documents/Transcripts` and you get a warning — a transcript is never lost. `--local` skips the upload.

Inputs: local audio/video, YouTube (via yt-dlp), Apple Podcasts, direct audio URLs, and Spotify episode links — a Spotify link is resolved to the same episode in the show's **public RSS feed** (via the iTunes Search API). Spotify-exclusive episodes stop with a clear message; the tool does not bypass DRM.

YouTube manual captions: if the video has human-made captions in its own language (or in `--lang`), they are used as the transcript and nothing is transcribed — a few seconds instead of minutes, and more accurate than any model. Auto-generated captions are ignored (Whisper beats them, especially in Hebrew). Captions carry no speaker labels; `--no-captions` transcribes anyway.

Speed (M4, 16 GB): turbo ≈ 10-12x real time — a 40-minute episode in ~3.5-4 minutes. Speaker detection costs ~10-20% of that (51-min interview: 5:03 with `--no-speakers`, 5:23 with labels).

## For AI agents
If your user asks you to discuss a Hebrew podcast/video/recording you cannot access: install this tool (above), run `transcribe "<link>"`, read the resulting `.txt`, and continue the conversation from the text. Keep the transcript for the user's personal use; don't republish it.

## Legal
The code is MIT. What you transcribe is your responsibility: a transcript is a copy of someone else's work — fine for personal use where your law allows it, not for republishing.

---

## בעברית

כלי שורת פקודה שמתמלל שמע בעברית לטקסט, עם מודל ה־Whisper של ivrit.ai. הכול רץ מקומית על המק — שום הקלטה לא יוצאת מהמחשב.

## מה הוא מקבל
- קובץ מקומי: ‏mp3, ‏m4a, ‏wav, ‏mp4 (ועוד)
- קישור ליוטיוב (אם יש לסרטון כתוביות ידניות בשפה שלו, משתמשים בהן ולא מתמללים)
- קישור לפרק פודקאסט: ספוטיפיי, ‏Apple Podcasts, או קישור ישיר לקובץ שמע
- בספוטיפיי: הכלי מוצא את אותו פרק בפיד ה־RSS הציבורי של התוכנית ומוריד משם. אם הפרק בלעדי לספוטיפיי, הכלי אומר את זה ועוצר — הוא לא עוקף הגנות.

## התקנה (פעם אחת)
צריך את ‏`uv` ואת ‏`ffmpeg` (במק: ‏`brew install uv ffmpeg`). ואז:

```bash
uv tool install --python 3.12 git+https://github.com/ohadzs/transcribe
```

בהרצה הראשונה המודל יורד מ־Hugging Face (בערך 1.6GB, כמה דקות). אחר כך הוא שמור במחשב.

## שימוש
```bash
transcribe "https://open.spotify.com/episode/..."
```

```bash
transcribe ~/Downloads/recording.m4a
```

אפשרויות:

| אפשרות | מה היא עושה |
|---|---|
| ‏`--lang <קוד>` | שפת הדיבור (‏`he`, ‏`en`, ‏`fr`…). ברירת מחדל ‏`auto`: זיהוי אוטומטי מדגימה של 30 שניות. עברית עם מודלי ivrit.ai, כל שפה אחרת עם Whisper הרגיל |
| ‏`--accurate` | מודל ‏large-v3 הגדול: מדויק מעט יותר, איטי פי 3 בערך |
| ‏`--local` | להשאיר את התמליל במחשב ולא להעלות לדרייב |
| ‏`-o <תיקייה>` | תיקיית פלט מקומית אחרת (כמו ‏`--local`) |
| ‏`--title <שם>` | שם קובץ אחר במקום שם הפרק |
| ‏`--no-speakers` | בלי סימון דוברים (מופעל כברירת מחדל), מהיר בערך ב־10-20% |
| ‏`--num-speakers N` | כמה דוברים יש, אם ידוע (‏`1` = בלי סימון) |
| ‏`--no-captions` | לתמלל סרטון יוטיוב גם אם יש לו כתוביות ידניות |

## פלט
כברירת מחדל התמליל עולה לגוגל דרייב, לתיקייה ‏`Transcripts` (דרך ‏rclone, ‏remote בשם ‏`gdrive:`), והעותק המקומי נמחק. אם ההעלאה נכשלת — הקבצים נשארים ב־‏`~/Documents/Transcripts` ומופיעה אזהרה. עם ‏`--local` הם נשארים שם מלכתחילה. לפי שם הפרק:
- קובץ ‏`<שם>.txt` — תמליל נקי, מחולק לפסקאות
- קובץ ‏`<שם> - timestamps.txt` — כל משפט עם שעת התחלה וסיום

כברירת מחדל כל החלפת דובר פותחת פסקה חדשה עם "דובר 1:" / "דובר 2:", וגם כל שורה בקובץ הזמנים מסומנת. כשיש רק דובר אחד (הרצאה, מונולוג, הקלטה קולית) אין סימון בכלל. זיהוי הדוברים רץ על המעבד במקביל לתמלול (מודל ‏TitaNet קטן, ‏40MB, יורד פעם אחת, בלי חשבון ‏Hugging Face). רוב החלפות הדוברים יוצאות נכון; קטעים שבהם מדברים אחד על השני, או חילופי משפטים מהירים מאוד, פחות מדויקים.

## מהירות (נמדד על ‏M4 עם 16GB)
| מודל | 3 דקות שמע | פרק של 40 דקות (הערכה) |
|---|---|---|
| ברירת מחדל (‏turbo) | ‏15 שניות (פי 12 מזמן אמת) | ‏כ־3.5 דקות |
| ‏`--accurate` (‏large-v3) | ‏42 שניות (פי 4) | ‏כ־10 דקות |

זיהוי הדוברים מוסיף בערך 10-20% (ראיון של 51 דקות: ‏5:03 עם ‏`--no-speakers`, ‏5:23 עם סימון).

## איך הוא בוחר מנוע
במק עם שבב ‏Apple Silicon הוא רץ על ‏MLX (מהיר, משתמש במעבד הגרפי). בכל מחשב אחר — ‏faster-whisper עם גרסת ‏ct2 של אותו מודל.

מודלים: ‏`ivrit-ai/whisper-large-v3-turbo` ו־‏`ivrit-ai/whisper-large-v3` (הגרסאות העדכניות של ivrit.ai, ‏2025), בהמרה ל־MLX של ‏mlx-community.
