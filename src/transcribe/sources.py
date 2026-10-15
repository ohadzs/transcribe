"""Turn an input (local file, YouTube link, podcast link) into a local audio file + a title."""

from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"
AUDIO_EXT = {".mp3", ".m4a", ".wav", ".mp4", ".aac", ".ogg", ".flac", ".webm", ".mov", ".mkv", ".opus"}


class SourceError(Exception):
    """A problem the user should see as a plain message."""


@dataclass
class Source:
    audio: Path
    title: str


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _get_json(url: str) -> dict:
    return json.loads(_get(url))


def _download(url: str, dest: Path, label: str) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0) or None
        ctype = r.headers.get("Content-Type", "")
        ext = Path(urllib.parse.urlparse(r.geturl()).path).suffix.lower()
        if ext not in AUDIO_EXT:
            ext = ".m4a" if "mp4" in ctype or "m4a" in ctype else ".mp3"
        out = dest.with_suffix(ext)
        with open(out, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=label) as bar:
            while chunk := r.read(1 << 16):
                f.write(chunk)
                bar.update(len(chunk))
    return out


# ---------- local files ----------

def _local(path: Path) -> Source:
    if not path.exists():
        raise SourceError(f"File not found: {path}")
    return Source(audio=path.resolve(), title=path.stem)


# ---------- YouTube and anything else yt-dlp understands ----------

def _yt_dlp(url: str, workdir: Path) -> Source:
    exe = shutil.which("yt-dlp") or str(Path(sys.executable).parent / "yt-dlp")
    info = subprocess.run(
        [exe, "--no-playlist", "--dump-single-json", url],
        capture_output=True, text=True,
    )
    if info.returncode != 0:
        raise SourceError(f"Could not read this link:\n{info.stderr.strip()[-500:]}")
    title = json.loads(info.stdout).get("title") or "audio"
    out = workdir / "download.%(ext)s"
    print("Downloading audio...")
    dl = subprocess.run(
        [exe, "--no-playlist", "-f", "bestaudio/best", "-x", "--audio-format", "m4a",
         "--progress", "--newline", "-o", str(out), url],
    )
    if dl.returncode != 0:
        raise SourceError("Download failed (see the yt-dlp messages above).")
    files = sorted(workdir.glob("download.*"))
    if not files:
        raise SourceError("Download finished but no audio file was produced.")
    return Source(audio=files[0], title=title)


# ---------- podcasts: RSS feeds ----------

def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", " ", s).strip().lower()


def _find_in_feed(feed_url: str, episode_title: str) -> tuple[str, str] | None:
    """Return (title, audio_url) of the best matching episode in the RSS feed."""
    try:
        root = ET.fromstring(_get(feed_url))
    except Exception:
        return None
    want = _norm(episode_title)
    best, best_score = None, 0.0
    for item in root.iter("item"):
        t = (item.findtext("title") or "").strip()
        enc = item.find("enclosure")
        if enc is None or not enc.get("url"):
            continue
        got = _norm(t)
        score = 1.0 if got == want else difflib.SequenceMatcher(None, want, got).ratio()
        if want and (want in got or got in want):
            score = max(score, 0.95)
        if score > best_score:
            best, best_score = (t, enc.get("url")), score
    return best if best_score >= 0.85 else None


def _itunes_feeds(show: str) -> list[str]:
    q = urllib.parse.urlencode({"term": show, "media": "podcast", "entity": "podcast", "limit": 10})
    res = _get_json(f"https://itunes.apple.com/search?{q}").get("results", [])
    want = _norm(show)
    res.sort(key=lambda r: -difflib.SequenceMatcher(None, want, _norm(r.get("collectionName", ""))).ratio())
    return [r["feedUrl"] for r in res if r.get("feedUrl")]


def _spotify(url: str, workdir: Path) -> Source:
    m = re.search(r"open\.spotify\.com/(?:intl-[a-z]+/)?episode/([A-Za-z0-9]+)", url)
    if not m:
        raise SourceError("Only Spotify *episode* links are supported (open.spotify.com/episode/...).")
    html = _get(f"https://open.spotify.com/embed/episode/{m.group(1)}").decode("utf-8", "replace")
    nd = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.S)
    try:
        ent = json.loads(nd.group(1))["props"]["pageProps"]["state"]["data"]["entity"]
        episode, show = ent["title"], ent["subtitle"]
    except Exception:
        raise SourceError("Could not read the episode details from Spotify.")
    print(f"Spotify episode: {episode!r} from show {show!r}")
    print("Looking for the same episode in the show's public RSS feed...")
    feeds = _itunes_feeds(show)
    for feed in feeds[:5]:
        hit = _find_in_feed(feed, episode)
        if hit:
            title, audio_url = hit
            print("Found it in the public feed.")
            return Source(audio=_download(audio_url, workdir / "download", "Downloading"), title=title)
    if not feeds:
        why = "the show is not listed in any public podcast directory"
    else:
        why = "the show's public RSS feed does not contain this episode"
    raise SourceError(
        f"This episode looks like a Spotify exclusive: {why}.\n"
        "There is no public audio file to download, so it cannot be transcribed.\n"
        "(Spotify's own audio is DRM-protected and this tool will not try to bypass it.)"
    )


def _apple(url: str, workdir: Path) -> Source:
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    ep_id = (q.get("i") or [None])[0]
    show_id = re.search(r"/id(\d+)", url)
    if ep_id:
        res = _get_json(f"https://itunes.apple.com/lookup?id={show_id.group(1) if show_id else ''}"
                        f"&entity=podcastEpisode&limit=300").get("results", [])
        for r in res:
            if str(r.get("trackId")) == ep_id and r.get("episodeUrl"):
                return Source(audio=_download(r["episodeUrl"], workdir / "download", "Downloading"),
                              title=r.get("trackName", "episode"))
    raise SourceError("Send a link to a specific Apple Podcasts episode (it contains '?i=').")


# ---------- dispatch ----------

def resolve(arg: str, workdir: Path) -> Source:
    if not re.match(r"^https?://", arg):
        return _local(Path(arg).expanduser())
    host = urllib.parse.urlparse(arg).netloc.lower()
    if "spotify.com" in host:
        return _spotify(arg, workdir)
    if "podcasts.apple.com" in host:
        return _apple(arg, workdir)
    path_ext = Path(urllib.parse.urlparse(arg).path).suffix.lower()
    if path_ext in AUDIO_EXT:  # direct link to an audio file
        name = urllib.parse.unquote(Path(urllib.parse.urlparse(arg).path).stem)
        return Source(audio=_download(arg, workdir / "download", "Downloading"), title=name)
    return _yt_dlp(arg, workdir)  # YouTube and most podcast sites
