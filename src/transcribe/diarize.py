"""Who spoke when: speaker embeddings over short sliding windows + spectral clustering.

The embedding model (NVIDIA TitaNet-small as ONNX, ~40 MB, run with sherpa-onnx) is downloaded once
from the sherpa-onnx GitHub releases -- no Hugging Face token or account needed. It runs on the CPU in
a separate process while Whisper uses the GPU, so on Apple Silicon it adds little wall-clock time.
The number of speakers is estimated from the eigengap of the affinity matrix unless it is given.
"""

from __future__ import annotations

import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache" / "transcribe" / "diarization"
EMBEDDING_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                 "speaker-recongition-models/nemo_en_titanet_small.onnx")
SAMPLE_RATE = 16000
WINDOW, HOP = 1.5, 0.75   # seconds; each window labels the HOP-wide stretch at its centre
MAX_SPEAKERS = 8
MAX_CLUSTER_WINDOWS = 4000  # cluster at most this many windows (memory); the rest join the nearest centroid
PRUNE = 0.1               # keep each window's top 10% affinities (NME-SC style p-pruning)


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


def _model() -> Path:
    dest = CACHE / Path(EMBEDDING_URL).name
    if not dest.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(EMBEDDING_URL, tmp)
        tmp.rename(dest)
    return dest


def load_audio(path: str) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.float32)


def _embed(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import sherpa_onnx as so

    ext = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(model=str(_model()), num_threads=4))
    w, h = int(WINDOW * SAMPLE_RATE), int(HOP * SAMPLE_RATE)
    starts = np.arange(0, max(len(x) - w, 0) + 1, h)
    embs, rms = [], []
    for a in starts:
        chunk = x[a:a + w]
        st = ext.create_stream()
        st.accept_waveform(SAMPLE_RATE, chunk)
        st.input_finished()
        embs.append(ext.compute(st))
        rms.append(float(np.sqrt(np.mean(chunk ** 2))))
    return np.asarray(embs, np.float32), starts / SAMPLE_RATE, np.asarray(rms)


def _spectral(X: np.ndarray, n: int | None) -> np.ndarray:
    from scipy.cluster.vq import kmeans2

    A = X @ X.T
    k = max(int(PRUNE * len(A)), 5)
    A = np.where(A >= np.partition(A, -k, axis=1)[:, -k][:, None], A, 0.0)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    d = 1 / np.sqrt(np.maximum(A.sum(1), 1e-9))
    lap = np.eye(len(A), dtype=A.dtype) - d[:, None] * A * d[None, :]
    vals, vecs = np.linalg.eigh(lap)
    n = n or int(np.argmax(np.diff(vals[:MAX_SPEAKERS + 1])[1:]) + 2)
    if n == 1:
        return np.zeros(len(X), int)
    Y = vecs[:, :n] / np.linalg.norm(vecs[:, :n], axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    best = None
    for _ in range(5):
        cent, lab = kmeans2(Y, n, minit="++", seed=rng)
        cost = ((Y - cent[lab]) ** 2).sum()
        if best is None or cost < best[0]:
            best = (cost, lab)
    return best[1]


def diarize(path: str, num_speakers: int | None = None) -> list[Turn]:
    x = load_audio(path)
    E, starts, rms = _embed(x)
    if len(E) < 4:
        return [Turn(0.0, len(x) / SAMPLE_RATE, 0)]
    speech = rms > np.percentile(rms, 10)  # skip the quietest windows (silence, music tails)
    X = E[speech] - E[speech].mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    sub = np.linspace(0, len(X) - 1, min(len(X), MAX_CLUSTER_WINDOWS)).astype(int)
    lab_sub = _spectral(X[sub], num_speakers)
    cent = np.stack([X[sub][lab_sub == c].mean(0) for c in np.unique(lab_sub)])
    lab = np.argmax(X @ cent.T, 1) if len(sub) < len(X) else lab_sub

    turns: list[Turn] = []
    for t0, sp in zip(starts[speech], lab):
        a, b = t0 + (WINDOW - HOP) / 2, t0 + (WINDOW + HOP) / 2
        if turns and turns[-1].speaker == sp and a - turns[-1].end < 1e-3:
            turns[-1].end = b
        else:
            turns.append(Turn(float(a), float(b), int(sp)))
    return turns


# A speaker run of fewer words than this between two runs of one other speaker is treated as
# diarization noise and absorbed (usually mis-assigned words at a turn edge, or an "uh-huh").
MIN_RUN_WORDS = 3
# Window labels are ~0.75 s coarse, so a turn change is moved to the nearest natural break
# (sentence end, segment start, pause) within this many words.
SNAP_WORDS = 4


def _speaker_at(turns: list[Turn], start: float, end: float) -> int | None:
    best, best_ov, nearest, near_d = None, 0.0, None, float("inf")
    for t in turns:
        ov = min(end, t.end) - max(start, t.start)
        if ov > best_ov:
            best, best_ov = t.speaker, ov
        d = max(t.start - end, start - t.end, 0.0)
        if d < near_d:
            nearest, near_d = t.speaker, d
    return best if best is not None else nearest


def _break_score(words: list, first: set, j: int) -> float:
    """How natural it is for a new speaker to start at word j."""
    prev, w = words[j - 1], words[j]
    tail = prev.text[-1:]
    score = 3.0 if tail in ".?!" else 1.0 if tail == "," else 0.0
    if j in first or w.start - prev.end >= 0.5:
        score = max(score, 3.0)
    return score


def _runs(spk: list) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(spk):
        j = i
        while j + 1 < len(spk) and spk[j + 1] == spk[i]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


@dataclass
class _Word:
    start: float
    end: float
    text: str


def _words(seg) -> list[_Word]:
    """Approximate word timings by spreading the segment's time over its words by length.

    Real word timestamps make Whisper ~80% slower; on our test episodes this estimate gave the
    same speaker for 99%+ of words, since the change points are snapped to punctuation anyway.
    """
    toks = seg.text.split()
    weights = [len(t) + 2 for t in toks]
    total, t, out = sum(weights), seg.start, []
    for tok, w in zip(toks, weights):
        dt = (seg.end - seg.start) * w / total
        out.append(_Word(t, t + dt, tok))
        t += dt
    return out


def label(segs: list, turns: list[Turn]) -> list:
    """Give every word a speaker, then split Whisper segments wherever the speaker changes.

    Speakers are renumbered 1, 2, ... in order of first appearance.
    """
    from .engine import Segment

    per_seg = [_words(s) for s in segs]
    words, first = [], set()
    for ws in per_seg:
        if ws:
            first.add(len(words))
        words += ws
    if not words:
        return segs
    spk = [_speaker_at(turns, w.start, w.end) for w in words]
    # 1) absorb short runs sandwiched between the same speaker
    for _ in range(2):
        for i, j in _runs(spk):
            if 0 < i and j + 1 < len(spk) and spk[i - 1] == spk[j + 1] and j - i + 1 < MIN_RUN_WORDS:
                spk[i:j + 1] = [spk[i - 1]] * (j - i + 1)
    # 2) snap each change to the most natural break nearby
    c = 0
    while True:
        runs = _runs(spk)
        if c + 1 >= len(runs):
            break
        (a0, _), (b0, b1) = runs[c], runs[c + 1]
        best, best_v = b0, _break_score(words, first, b0)
        for j in range(max(a0 + 1, b0 - SNAP_WORDS), min(b1, b0 + SNAP_WORDS) + 1):
            v = _break_score(words, first, j) - 0.5 * abs(j - b0)
            if v > best_v:
                best, best_v = j, v
        lo, hi = sorted((best, b0))
        spk[lo:hi] = [spk[b0] if best < b0 else spk[a0]] * (hi - lo)
        c += 1
    order: dict = {}
    for s in spk:
        order.setdefault(s, len(order) + 1)

    out, k = [], 0
    for ws in per_seg:
        cur: list[_Word] = []
        for w in ws:
            sp = order[spk[k]]
            k += 1
            if cur and sp != cur_spk:
                out.append(Segment(cur[0].start, cur[-1].end, " ".join(x.text for x in cur), cur_spk))
                cur = []
            cur.append(w)
            cur_spk = sp
        if cur:
            out.append(Segment(cur[0].start, cur[-1].end, " ".join(x.text for x in cur), cur_spk))
    return out
