"""Low-LOD chunked atlases: many tiny square thumbnails packed into a few big images.

Engine-agnostic: plain JPEG chunks plus one JSON manifest, and every lookup is
arithmetic. Nothing here knows about HTML; gallery.py is just one consumer.

Layout (manifest "fvg-lod-atlas", version 1)
--------------------------------------------
Items are numbered 0..count-1, in the order of `keys`. Each level packs them into
chunks: one chunk is one image of cols x rows square slots, filled row-major.
For item i at level L (lv = levels[L]):

    chunk = i // lv.per_chunk              -> image file lv.chunks[chunk]
    slot  = i %  lv.per_chunk
    x = (slot % lv.cols)  * lv.pitch + lv.pad   -> top-left of the item's square
    y = (slot // lv.cols) * lv.pitch + lv.pad
    the square is lv.size x lv.size, where size = pitch - 2 * pad

Design choices:
- Squares only (centre crops): a grid cell shows a square anyway, so no slot
  needs its own width/height, and position is pure arithmetic.
- `pad` pixels around each square repeat its edge, so bilinear/mipmapped
  sampling (and sub-pixel CSS positioning) never pulls in the neighbour.
- pitch is a multiple of 16: slots align with JPEG's 8x8 / 16x16 (4:2:0)
  blocks, so compression never bleeds between items, and with 4x4 BC7/ASTC blocks, if chunks are
  later transcoded for the GPU.
- Chunks are always full size (unused slots are dark grey), so every chunk of a
  level has the same dimensions and they can go straight into a texture array.
- Chunk files are named by a hash of their level settings and member keys. An
  unchanged chunk keeps its name across rebuilds (nothing redone, caches stay
  valid), and files no longer referenced are deleted. Keys must change when an
  item's image changes (gallery.py's cache keys do).
"""
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

FORMAT, VERSION = "fvg-lod-atlas", 1
MANIFEST = "atlas.json"
EMPTY = (40, 40, 40)


@dataclass(frozen=True)
class Level:
    pitch: int  # slot size in px (multiple of 16)
    pad: int    # edge-repeat border inside the slot
    cols: int   # slots per chunk side (chunks are cols x cols slots)
    quality: int = 80
    subsampling: int = 2  # JPEG chroma: 0 = 4:4:4 (full colour), 2 = 4:2:0 (half-res colour)

    @property
    def size(self):
        return self.pitch - 2 * self.pad

    @property
    def per_chunk(self):
        return self.cols * self.cols

    @property
    def px(self):
        return self.cols * self.pitch


# L0: 1024 items per 512px chunk, so a whole 6k library is ~6 small files that
# can be resident all the time. At 14px a square is mostly colour, so it gets
# full-res chroma and higher quality (it's tiny either way).
# L1: 256 items per 1024px chunk, loaded near view.
LEVELS = (Level(pitch=16, pad=1, cols=32, quality=90, subsampling=0),
          Level(pitch=64, pad=2, cols=16))


def _tile(src, lv):
    """The item's square centre crop at this level, padded out to the pitch with its own edges."""
    with Image.open(src) as im:
        im.draft("RGB", (lv.size, lv.size))  # JPEG: decode at reduced scale, still >= size
        sq = ImageOps.fit(im.convert("RGB"), (lv.size, lv.size), Image.Resampling.LANCZOS)
    if not lv.pad:
        return sq
    # Stretch to the full pitch, then put the exact square back in the middle:
    # the border ring that's left is (very nearly) the square's edge pixels.
    tile = sq.resize((lv.pitch, lv.pitch), Image.Resampling.BILINEAR)
    tile.paste(sq, (lv.pad, lv.pad))
    return tile


def _build_chunk(job):
    lv, paths, out = job
    sheet = Image.new("RGB", (lv.px, lv.px), EMPTY)
    for s, p in enumerate(paths):
        try:
            sheet.paste(_tile(p, lv), ((s % lv.cols) * lv.pitch, (s // lv.cols) * lv.pitch))
        except Exception:
            pass  # missing/broken source: the slot stays empty
    tmp = out + ".tmp"
    sheet.save(tmp, "JPEG", quality=lv.quality, subsampling=lv.subsampling)
    os.replace(tmp, out)  # never a half-written chunk under its final name


def chunk_name(level_index, lv, keys):
    h = hashlib.md5(json.dumps([VERSION, lv.pitch, lv.pad, lv.cols, lv.quality, lv.subsampling, keys]).encode())
    return f"l{level_index}_{h.hexdigest()[:16]}.jpg"


def build(items, out_dir, levels=LEVELS, workers=None):
    """Pack items = [(key, source image path), ...] (display order) into out_dir.

    Returns (manifest, number of chunks built). Only chunks whose members changed
    are rebuilt; the manifest is also written to out_dir/atlas.json.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = [k for k, _ in items]
    paths = [str(p) for _, p in items]
    manifest = {"format": FORMAT, "version": VERSION, "count": len(keys), "keys": keys, "levels": []}
    jobs = []
    for L, lv in enumerate(levels):
        assert lv.pitch % 16 == 0 and 0 <= lv.pad and lv.size > 0, lv
        names = []
        for start in range(0, len(keys), lv.per_chunk):
            name = chunk_name(L, lv, keys[start:start + lv.per_chunk])
            names.append(name)
            if not (out / name).exists():
                jobs.append((lv, paths[start:start + lv.per_chunk], str(out / name)))
        manifest["levels"].append({
            "pitch": lv.pitch, "pad": lv.pad, "size": lv.size, "cols": lv.cols, "rows": lv.cols,
            "per_chunk": lv.per_chunk, "width": lv.px, "height": lv.px, "chunks": names,
        })
    jobs.sort(key=lambda j: -len(j[1]))  # biggest chunks first, for even worker load
    if jobs:
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_build_chunk, jobs))
    keep = {n for lv in manifest["levels"] for n in lv["chunks"]} | {MANIFEST}
    for f in out.iterdir():
        if f.name not in keep:
            f.unlink()
    tmp = out / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest))
    os.replace(tmp, out / MANIFEST)
    return manifest, len(jobs)


def locate(manifest, level, i):
    """(chunk file, x, y, size) of item i at a level: the reference for any port of the lookup."""
    lv = manifest["levels"][level]
    c, s = divmod(i, lv["per_chunk"])
    return (lv["chunks"][c], (s % lv["cols"]) * lv["pitch"] + lv["pad"],
            (s // lv["cols"]) * lv["pitch"] + lv["pad"], lv["size"])
