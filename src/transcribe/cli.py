"""transcribe <file or link> -- Hebrew speech to text with ivrit.ai Whisper."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import engine, sources

DEFAULT_OUT = Path.home() / "Documents" / "Transcripts"
# rclone destination for transcripts (default mode); set TRANSCRIBE_REMOTE to change it.
REMOTE = os.environ.get("TRANSCRIBE_REMOTE", "gdrive:Transcripts")
PARAGRAPH_GAP = 1.5      # a pause this long (seconds) starts a new paragraph
PARAGRAPH_CHARS = 700    # ...or a paragraph this long, at the next sentence end
SPEAKER = "דובר {}: "   # speaker label (Hebrew: "Speaker N: ")


def ts(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def safe_name(title: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", title)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:120] or "transcript"


def speaker_tag(s: engine.Segment) -> str:
    return SPEAKER.format(s.speaker) if s.speaker else ""


def paragraphs(segs: list[engine.Segment]) -> list[str]:
    """Paragraphs break on long pauses, on length, and on every speaker change (labelled)."""
    paras, cur, prev = [], [], None
    for s in segs:
        new_turn = prev is not None and s.speaker != prev.speaker
        long_pause = prev is not None and s.start - prev.end >= PARAGRAPH_GAP
        too_long = sum(len(x) for x in cur) >= PARAGRAPH_CHARS and cur and cur[-1].endswith((".", "?", "!"))
        if cur and (new_turn or long_pause or too_long):
            paras.append(" ".join(cur))
            cur = []
        cur.append((speaker_tag(s) if prev is None or new_turn else "") + s.text)
        prev = s
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
    stamped.write_text("".join(f"[{ts(s.start)} - {ts(s.end)}] {speaker_tag(s)}{s.text}\n" for s in segs), encoding="utf-8")
    return txt, stamped


def _rclone() -> str | None:
    return shutil.which("rclone") or next((p for p in ("/opt/homebrew/bin/rclone", "/usr/local/bin/rclone") if Path(p).exists()), None)


def upload(files: tuple[Path, Path], remote: str) -> list[str] | None:
    """Move both files to the rclone remote; return their remote paths, or None (local files kept)."""
    rc = _rclone()
    if not rc:
        print("\nWarning: rclone not found -- transcript kept locally.", file=sys.stderr)
        return None
    run = lambda *args: subprocess.run([rc, *args], capture_output=True, text=True)
    listing = run("lsf", "--files-only", remote)
    existing = set(listing.stdout.splitlines()) if listing.returncode == 0 else set()
    # Don't overwrite an earlier transcript with the same title on the remote.
    txt, stamped = files
    base, n = txt.stem, 2
    names = (txt.name, stamped.name)
    while names[0] in existing or names[1] in existing:
        names = (f"{base} ({n}).txt", f"{base} ({n}) - timestamps.txt")
        n += 1
    done = []
    for f, name in zip(files, names):
        dest = f"{remote.rstrip('/')}/{name}"
        r = run("moveto", str(f), dest)
        if r.returncode != 0 or f.exists():
            print(f"\nWarning: upload to {remote} failed -- transcript kept locally.\n{r.stderr.strip()}", file=sys.stderr)
            return None
        done.append(dest)
    return done


def _with_speakers(audio: str, eng: str, model: str, num_speakers: int | None) -> list[engine.Segment]:
    """Transcribe (GPU) and diarize (CPU, separate process) at the same time, then merge."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from . import diarize

    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        turns = pool.submit(diarize.diarize, audio, num_speakers)
        segs = engine.transcribe(audio, eng, model)
        print("Waiting for speaker detection to finish...")
        turns = turns.result()
    labelled = diarize.label(segs, turns)
    print(f"Speakers found: {len({s.speaker for s in labelled})}")
    return labelled


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="transcribe", description="Transcribe Hebrew audio to text (ivrit.ai Whisper).")
    p.add_argument("source", help="audio/video file, YouTube link, or podcast episode link (Spotify / Apple / RSS)")
    p.add_argument("--local", action="store_true", help=f"keep the transcript on this machine instead of uploading it to {REMOTE}")
    p.add_argument("-o", "--out", type=Path, help=f"local output folder (implies --local; default: {DEFAULT_OUT})")
    p.add_argument("--accurate", action="store_true", help="use the larger large-v3 model (slower, a bit more accurate)")
    p.add_argument("--title", help="override the output file name")
    p.add_argument("--speakers", action="store_true", help='mark speaker turns ("דובר 1:", "דובר 2:", ...)')
    p.add_argument("--num-speakers", type=int, metavar="N", help="how many speakers there are, if known (implies --speakers)")
    a = p.parse_args(argv)
    a.speakers = a.speakers or bool(a.num_speakers)

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
        if a.speakers:
            segs = _with_speakers(str(src.audio), eng, model, a.num_speakers)
        else:
            segs = engine.transcribe(str(src.audio), eng, model)
        took = time.time() - t0

    if not segs:
        print("No speech was recognized.", file=sys.stderr)
        return 1
    txt, stamped = write_outputs(segs, a.title or src.title, a.out or DEFAULT_OUT)
    speed = f" ({dur / took:.1f}x real time)" if took and dur else ""
    where = None if a.local or a.out else upload((txt, stamped), REMOTE)
    print(f"\nDone in {ts(took)}{speed}.")
    print("".join(f"  {x}\n" for x in (where or (txt, stamped))), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
