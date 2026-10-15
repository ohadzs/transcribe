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
| `--accurate` | `large-v3` instead of `large-v3-turbo`: slightly better, ~3x slower |
| `-o <dir>` | output folder (default `~/Documents/Transcripts`) |
| `--title <name>` | output file name instead of the episode title |

Output: `<title>.txt` (clean paragraphs) and `<title> - timestamps.txt` (one line per segment with start/end times).

Inputs: local audio/video, YouTube (via yt-dlp), Apple Podcasts, direct audio URLs, and Spotify episode links — a Spotify link is resolved to the same episode in the show's **public RSS feed** (via the iTunes Search API). Spotify-exclusive episodes stop with a clear message; the tool does not bypass DRM.

Speed (M4, 16 GB): turbo ≈ 12x real time — a 40-minute episode in ~3.5 minutes.

## For AI agents
If your user asks you to discuss a Hebrew podcast/video/recording you cannot access: install this tool (above), run `transcribe "<link>"`, read the resulting `.txt`, and continue the conversation from the text. Keep the transcript for the user's personal use; don't republish it.

## Legal
The code is MIT. What you transcribe is your responsibility: a transcript is a copy of someone else's work — fine for personal use where your law allows it, not for republishing.

---

## בעברית

כלי שורת פקודה שמתמלל שמע בעברית לטקסט, עם מודל ה־Whisper של ivrit.ai. הכול רץ מקומית על המק — שום הקלטה לא יוצאת מהמחשב.

## מה הוא מקבל
- קובץ מקומי: ‏mp3, ‏m4a, ‏wav, ‏mp4 (ועוד)
- קישור ליוטיוב
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
| ‏`--accurate` | מודל ‏large-v3 הגדול: מדויק מעט יותר, איטי פי 3 בערך |
| ‏`-o <תיקייה>` | תיקיית פלט אחרת |
| ‏`--title <שם>` | שם קובץ אחר במקום שם הפרק |

## פלט
בתיקייה ‏`~/Documents/Transcripts`, לפי שם הפרק:
- קובץ ‏`<שם>.txt` — תמליל נקי, מחולק לפסקאות
- קובץ ‏`<שם> - timestamps.txt` — כל משפט עם שעת התחלה וסיום

## מהירות (נמדד על ‏M4 עם 16GB)
| מודל | 3 דקות שמע | פרק של 40 דקות (הערכה) |
|---|---|---|
| ברירת מחדל (‏turbo) | ‏15 שניות (פי 12 מזמן אמת) | ‏כ־3.5 דקות |
| ‏`--accurate` (‏large-v3) | ‏42 שניות (פי 4) | ‏כ־10 דקות |

## איך הוא בוחר מנוע
במק עם שבב ‏Apple Silicon הוא רץ על ‏MLX (מהיר, משתמש במעבד הגרפי). בכל מחשב אחר — ‏faster-whisper עם גרסת ‏ct2 של אותו מודל.

מודלים: ‏`ivrit-ai/whisper-large-v3-turbo` ו־‏`ivrit-ai/whisper-large-v3` (הגרסאות העדכניות של ivrit.ai, ‏2025), בהמרה ל־MLX של ‏mlx-community.
