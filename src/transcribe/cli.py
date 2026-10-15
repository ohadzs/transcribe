"""transcribe <file or link> -- Hebrew speech to text with ivrit.ai Whisper."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path

from . import engine, sources

DEFAULT_OUT = Path.home() / "Documents" / "Transcripts"
PARAGRAPH_GAP = 1.5      # a pause this long (seconds) starts a new paragraph
PARAGRAPH_CHARS = 700    # ...or a paragraph this long, at the next sentence end


def ts(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def safe_name(title: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", title)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:120] or "transcript"


def paragraphs(segs: list[engine.Segment]) -> list[str]:
    paras, cur, prev_end = [], [], None
    for s in segs:
        long_pause = prev_end is not None and s.start - prev_end >= PARAGRAPH_GAP
        too_long = sum(len(x) for x in cur) >= PARAGRAPH_CHARS and cur and cur[-1].endswith((".", "?", "!"))
        if cur and (long_pause or too_long):
            paras.append(" ".join(cur))
            cur = []
        cur.append(s.text)
        prev_end = s.end
    if cur:
        paras.append(" ".join(cur))
    return paras


def write_outputs(segs: list[engine.Segment], title: str, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = safe_name(title)
    txt, stamped = out_dir / f"{base}.txt", out_dir / f"{base} - timestamps.txt"
    n = 2
    while txt.exists() or stamped.exists():
        txt, stamped = out_dir / f"{base} ({n}).txt", out_dir / f"{base} ({n}) - timestamps.txt"
        n += 1
    txt.write_text("\n\n".join(paragraphs(segs)) + "\n", encoding="utf-8")
    stamped.write_text("".join(f"[{ts(s.start)} - {ts(s.end)}] {s.text}\n" for s in segs), encoding="utf-8")
    return txt, stamped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="transcribe", description="Transcribe Hebrew audio to text (ivrit.ai Whisper).")
    p.add_argument("source", help="audio/video file, YouTube link, or podcast episode link (Spotify / Apple / RSS)")
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help=f"output folder (default: {DEFAULT_OUT})")
    p.add_argument("--accurate", action="store_true", help="use the larger large-v3 model (slower, a bit more accurate)")
    p.add_argument("--title", help="override the output file name")
    a = p.parse_args(argv)

    eng, model, m = engine.choose("accurate" if a.accurate else None)
    print(f"Machine: {m['cpu']}, {m['ram_gb']:.0f} GB RAM -> engine {eng}, model {model}")

    with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp:
        try:
            src = sources.resolve(a.source, Path(tmp))
        except sources.SourceError as e:
            print(f"\nError: {e}", file=sys.stderr)
            return 2
        dur = engine.audio_duration(str(src.audio))
        print(f"Audio length: {ts(dur)}. Transcribing (first run downloads the model, ~1.6-3 GB)...")
        t0 = time.time()
        segs = engine.transcribe(str(src.audio), eng, model)
        took = time.time() - t0

    if not segs:
        print("No speech was recognized.", file=sys.stderr)
        return 1
    txt, stamped = write_outputs(segs, a.title or src.title, a.out)
    speed = f" ({dur / took:.1f}x real time)" if took and dur else ""
    print(f"\nDone in {ts(took)}{speed}.\n  {txt}\n  {stamped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
