"""Pick the fastest engine for this machine and run the ivrit.ai Hebrew Whisper model."""

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


# The model learned invisible RTL/LTR control marks from subtitle data; strip them.
_BIDI = dict.fromkeys(map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"))


def clean(text: str) -> str:
    return " ".join(text.translate(_BIDI).split())


@dataclass
class Segment:
    start: float
    end: float
    text: str


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


def choose(quality: str | None) -> tuple[str, str, dict]:
    m = machine()
    engine = "mlx" if m["apple_silicon"] else "faster-whisper"
    if quality is None:
        # large-v3 needs ~6GB working memory; turbo is ~2-3x faster with slightly lower accuracy.
        quality = "fast"
    return engine, MODELS[engine][quality], m


def audio_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def transcribe(path: str, engine: str, model: str) -> list[Segment]:
    if engine == "mlx":
        return _mlx(path, model)
    return _faster_whisper(path, model)


def _mlx(path: str, model: str) -> list[Segment]:
    import mlx_whisper

    # verbose=False makes mlx_whisper draw its own progress bar over the audio.
    result = mlx_whisper.transcribe(
        path,
        path_or_hf_repo=model,
        language="he",
        task="transcribe",
        verbose=False,
        condition_on_previous_text=False,
    )
    return [Segment(s["start"], s["end"], clean(s["text"])) for s in result["segments"] if clean(s["text"])]


def _faster_whisper(path: str, model: str) -> list[Segment]:
    from faster_whisper import WhisperModel

    wm = WhisperModel(model, device="auto", compute_type="int8")
    segs, info = wm.transcribe(path, language="he", task="transcribe", vad_filter=True,
                               condition_on_previous_text=False)
    out: list[Segment] = []
    with tqdm(total=round(info.duration), unit="s", desc="Transcribing") as bar:
        for s in segs:
            if clean(s.text):
                out.append(Segment(s.start, s.end, clean(s.text)))
            bar.update(round(s.end) - bar.n)
        bar.update(bar.total - bar.n)
    return out
