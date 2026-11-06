"""Keyframe maps for videos, made by ../fast_mp4_grids (mp4cut).

mp4cut decodes only the keyframes (one sweep per job, VideoToolbox when it
can), so a video costs well under a second whatever its length. Its PNGs are
packed here into one sprite JPEG per video, turned upright for the display
rotation (mp4cut reads frames as stored), and the PNGs are thrown away.

The page gets, per video: {"n", "cols", "w", "h", "t": [seconds]}. Tile i sits
at column i % cols, row i // cols of thumbs/<key>_kf.jpg, w x h pixels each, and
shows the video from t[i] (the timeline starting at 0 for its first frame).
"""
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

MAX_KEYFRAMES = 240  # spread evenly over the video; short ones just get all of theirs
DECODE_WIDTH = 400   # mp4cut's pass width (stored frame width), before packing shrinks it
JOBS = 3             # ffmpeg processes per mp4cut run
TIMEOUT = 300

# Turns that make a frame upright, by display rotation (ffprobe: degrees counter-clockwise).
UPRIGHT = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
           270: Image.Transpose.ROTATE_270}


def default_mp4cut(here):
    return Path(os.environ.get("MP4CUT", here.parent / "fast_mp4_grids" / "bin" / "mp4cut.mjs"))


def find_tool(mp4cut):
    """[node, mp4cut.mjs] if both are there, else None (and keyframe maps are skipped this run)."""
    node = shutil.which("node")
    mp4cut = Path(mp4cut).expanduser()
    return [node, str(mp4cut)] if node and mp4cut.is_file() else None


def tile_long_side(n):
    """Fewer keyframes get bigger tiles: a handful fill a whole side panel."""
    return 400 if n <= 16 else 320 if n <= 64 else 240


def run_mp4cut(tool, src, out, hw):
    """(manifest, out dir) of a finished run, or None."""
    cmd = [*tool, "frames", str(src), "--count", str(MAX_KEYFRAMES),
           "--passes", f"kf:{DECODE_WIDTH}:key", "--sheet", "none",
           "--jobs", str(JOBS), "--out", str(out), *([] if hw else ["--no-hw"])]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    manifest = out / "manifest.json"
    if r.returncode != 0 or not manifest.exists():
        return None
    return json.loads(manifest.read_text()), out


def make(tool, src, sprite):
    """Keyframe map for one video, or None if mp4cut can't read it."""
    try:
        return _make(tool, src, sprite)
    except Exception:
        return None


def _make(tool, src, sprite):
    with tempfile.TemporaryDirectory(prefix="fvg_kf_") as tmp:
        tmp = Path(tmp)
        # mp4cut's software fallback can lose a race between its parallel
        # hardware runs (e.g. VP9), so a failed run is retried on the CPU.
        done = run_mp4cut(tool, src, tmp / "hw", True) or run_mp4cut(tool, src, tmp / "sw", False)
        if not done:
            return None
        m, base = done
        frames = m["passes"]["kf"]["frames"]
        if not frames:
            return None
        turn = UPRIGHT.get(round(m["source"].get("rotation", 0)) % 360)

        n = len(frames)
        tiles = []
        for f in frames:
            with Image.open(base / f["file"]) as im:
                im = im.convert("RGB")
                tiles.append(im.transpose(turn) if turn else im)
        w0, h0 = tiles[0].size
        s = min(1.0, tile_long_side(n) / max(w0, h0))
        tw, th = max(1, round(w0 * s)), max(1, round(h0 * s))
        cols = max(1, min(n, round(math.sqrt(n * th / tw))))  # a roughly square sheet
        rows = math.ceil(n / cols)
        sheet = Image.new("RGB", (cols * tw, rows * th), (30, 30, 30))
        for i, im in enumerate(tiles):
            if im.size != (tw, th):
                im = im.resize((tw, th), Image.Resampling.LANCZOS, reducing_gap=2.0)
            sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
        tmp_out = f"{sprite}.tmp"
        sheet.save(tmp_out, "JPEG", quality=80)
        os.replace(tmp_out, sprite)

    t0 = frames[0]["pts"]
    return {"n": n, "cols": cols, "w": tw, "h": th, "t": [round(f["pts"] - t0, 3) for f in frames]}
