"""Pick the fastest engine for this machine, detect the language, and run the right Whisper model
(ivrit.ai's Hebrew fine-tune for Hebrew, stock Whisper for everything else)."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

from tqdm import tqdm

# ivrit.ai Hebrew fine-tunes of Whisper large-v3 (released Apr/May 2025, the current ones).
MODELS = {
    "mlx": {
        "fast": "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx",
        "accurate": "mlx-community/ivrit-ai-whisper-large-v3-mlx",
    },
    "faster-whisper": {
        "fast": "ivrit-ai/whisper-large-v3-turbo-ct2",
        "accurate": "ivrit-ai/whisper-large-v3-ct2",
    },
}
# Stock multilingual Whisper, for any language other than Hebrew (the Hebrew fine-tunes
# translate other languages into broken Hebrew) and for detecting the language.
STOCK = {
    "mlx": {
        "fast": "mlx-community/whisper-large-v3-turbo",
        "accurate": "mlx-community/whisper-large-v3-mlx",
    },
    "faster-whisper": {
        "fast": "large-v3-turbo",
        "accurate": "large-v3",
    },
}
DETECT_SECONDS = 30


# The model learned invisible RTL/LTR control marks from subtitle data; strip them.
_BIDI = dict.fromkeys(map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"))


def clean(text: str) -> str:
    return " ".join(text.translate(_BIDI).split())


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: int | None = None  # 1, 2, ... with --speakers


def machine() -> dict:
    is_apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    ram_gb = 0.0
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError):
        pass
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True).stdout.strip() or cpu
        except OSError:
            pass
    return {"apple_silicon": is_apple, "ram_gb": ram_gb, "cpu": cpu}


def choose(quality: str | None, lang: str = "he") -> tuple[str, str, dict]:
    m = machine()
    engine = "mlx" if m["apple_silicon"] else "faster-whisper"
    if quality is None:
        # large-v3 needs ~6GB working memory; turbo is ~2-3x faster with slightly lower accuracy.
        quality = "fast"
    return engine, (MODELS if lang == "he" else STOCK)[engine][quality], m


def detect_language(path: str, engine: str) -> str:
    """Guess the spoken language from a 30-second sample, with stock Whisper turbo.

    The sample starts a fifth of the way in, past intro music and jingles."""
    import tempfile

    start = audio_duration(path) * 0.2
    with tempfile.TemporaryDirectory(prefix="transcribe-lang-") as tmp:
        clip = os.path.join(tmp, "sample.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.0f}", "-t", str(DETECT_SECONDS),
                        "-i", path, "-ac", "1", "-ar", "16000", clip], check=True)
        if engine == "mlx":
            import mlx_whisper

            return mlx_whisper.transcribe(clip, path_or_hf_repo=STOCK["mlx"]["fast"], language=None,
                                          verbose=None)["language"]
        from faster_whisper import WhisperModel

        _, info = WhisperModel(STOCK["faster-whisper"]["fast"], device="auto", compute_type="int8").transcribe(clip)
        return info.language


def audio_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def transcribe(path: str, engine: str, model: str, lang: str = "he") -> list[Segment]:
    if engine == "mlx":
        return _mlx(path, model, lang)
    return _faster_whisper(path, model, lang)


def _mlx(path: str, model: str, lang: str) -> list[Segment]:
    import mlx_whisper

    # verbose=False makes mlx_whisper draw its own progress bar over the audio.
    result = mlx_whisper.transcribe(
        path,
        path_or_hf_repo=model,
        language=lang,
        task="transcribe",
        verbose=False,
        condition_on_previous_text=False,
    )
    return [Segment(s["start"], s["end"], clean(s["text"])) for s in result["segments"] if clean(s["text"])]


def _faster_whisper(path: str, model: str, lang: str) -> list[Segment]:
    from faster_whisper import WhisperModel

    wm = WhisperModel(model, device="auto", compute_type="int8")
    segs, info = wm.transcribe(path, language=lang, task="transcribe", vad_filter=True,
                               condition_on_previous_text=False)
    out: list[Segment] = []
    with tqdm(total=round(info.duration), unit="s", desc="Transcribing") as bar:
        for s in segs:
            if clean(s.text):
                out.append(Segment(s.start, s.end, clean(s.text)))
            bar.update(round(s.end) - bar.n)
        bar.update(bar.total - bar.n)
    return out
