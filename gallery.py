"""Fast browsable grid for a folder of images (PNG, JPEG, HEIC/HEIF) and videos.

Makes small JPEG thumbnails in parallel (cached, so re-runs only process new or
changed files) and writes an HTML page with a lazy-loading grid + lightbox.

Video thumbnails use the same trick as ../mp4_fast_editor: VideoToolbox
hardware decode, and only the nearest keyframe is decoded, so each video
costs one short ffmpeg call no matter how long it is.

Usage:
    python gallery.py [FOLDER] [--size 200] [--recursive] [--no-open]
"""
import argparse
import hashlib
import html
import json
import os
import subprocess
import time
import webbrowser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

import lod_atlas

register_heif_opener()

HERE = Path(__file__).resolve().parent
THUMBS = HERE / "thumbs"
META = THUMBS / "meta.json"
OUTPUT = HERE / "gallery.html"
# The item list lives in its own script next to the page, not inlined in it: the
# page stays a small fixed template, and filenames are never parsed as HTML.
# A <script src> (unlike fetch) still works when the page is opened from file://.
ITEMS_JS = HERE / "gallery_items.js"
ATLAS_DIR = THUMBS / "atlas"  # the page expects it at thumbs/atlas/

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
NEEDS_PREVIEW = {".heic", ".heif"}  # browsers (except Safari) can't show these
PREVIEW_SIZE = 1600
VIDEO_JOBS = 6  # VideoToolbox only runs a handful of decode sessions at once
# The sharp tier loaded near the cursor: short side = thumb size x this, enough
# to fill a square (cropped) cell on a 2x retina screen.
MID_SCALE = 2.4
CACHE_VERSION = 2  # bump when the thumbnail outputs change, to rebuild the cache


def cache_key(src, size):
    st = src.stat()
    key = f"v{CACHE_VERSION}|{src}|{st.st_mtime_ns}|{st.st_size}|{size}"
    return hashlib.md5(key.encode()).hexdigest()


def mid_path(thumb):
    return thumb[: -len(".jpg")] + "_m.jpg"


def fit_short(w, h, short):
    """Size with the short side at `short` (long side capped at 3x that), never upscaled."""
    s = min(1.0, short / min(w, h), 3 * short / max(w, h))
    return max(1, round(w * s)), max(1, round(h * s))


def flatten(img):
    """RGB copy of img, with any transparency composited onto dark grey."""
    if img.mode in ("RGBA", "LA", "P", "PA"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (40, 40, 40))
        bg.paste(img, mask=img.getchannel("A"))
        return bg
    return img.convert("RGB")


def make_image_thumb(job):
    """One decode, shrunk in steps: preview (HEIC only) -> mid -> thumb."""
    src, thumb, preview, size = job
    short = round(size * MID_SCALE)
    try:
        with Image.open(src) as img:
            img.draft("RGB", (short, short))  # JPEG only: decode at reduced scale
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.mode or img.mode == "P" else "RGB")
            if preview:
                img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS, reducing_gap=2.0)
                flatten(img).save(preview, "JPEG", quality=85)
            target = fit_short(*img.size, short)
            if target != img.size:
                img = img.resize(target, Image.Resampling.LANCZOS, reducing_gap=2.0)
            flatten(img).save(mid_path(thumb), "JPEG", quality=80)
            img.thumbnail((size, size), Image.Resampling.BILINEAR, reducing_gap=2.0)
            flatten(img).save(thumb, "JPEG", quality=80)
        return {}
    except Exception:
        return None


def video_duration(src):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def ffmpeg_frame(src, thumb, size, at, hw):
    """One decoded keyframe, written twice: the mid tier and the small thumb."""
    mid = mid_path(thumb)
    short = round(size * MID_SCALE)
    # Short side to `short` (never upscaled). JPEG wants full-range colour;
    # some recordings are limited-range.
    graph = (
        f"[0:v:0]scale='if(gt(iw,ih),-2,min({short},iw))':'if(gt(iw,ih),min({short},ih),-2)'"
        f":out_range=full,format=yuvj420p,split[m][t0];"
        f"[t0]scale={size}:{size}:force_original_aspect_ratio=decrease[t]"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        *(["-hwaccel", "videotoolbox"] if hw else []),
        "-skip_frame", "nokey",           # decode keyframes only
        "-noaccurate_seek", "-ss", f"{at:.3f}",
        "-i", src,
        "-filter_complex", graph,
        "-map", "[t]", "-frames:v", "1", "-q:v", "4", thumb,
        "-map", "[m]", "-frames:v", "1", "-q:v", "3", mid,
    ]
    r = subprocess.run(cmd, capture_output=True)
    ok = lambda p: os.path.exists(p) and os.path.getsize(p) > 0
    return r.returncode == 0 and ok(thumb) and ok(mid)


def make_video_thumb(job):
    src, thumb, size = job
    dur = video_duration(src)
    at = min(dur * 0.1, 10.0)  # a little way in, past black intro frames
    for seek, hw in ((at, True), (at, False), (0.0, False)):
        if ffmpeg_frame(src, thumb, size, seek, hw):
            return {"dur": dur}
    return None


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ margin:0; background:#1e1e1e; color:#ddd; font:14px system-ui, sans-serif; }}
  header {{ position:sticky; top:0; z-index:5; background:#1e1e1ef0; padding:10px 14px;
           display:flex; gap:12px; align-items:center; }}
  input, select {{ padding:6px 10px; border-radius:6px; border:1px solid #444;
          background:#2a2a2a; color:#eee; }}
  input {{ flex:1; max-width:400px; }}
  #grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax({size}px, 1fr));
          gap:6px; padding:6px 14px 40px; }}
  .c {{ position:relative; aspect-ratio:1; background:#2a2a2a no-repeat; border-radius:4px; overflow:hidden;
       cursor:pointer; content-visibility:auto; contain-intrinsic-size:{size}px; }}
  .c img {{ width:100%; height:100%; object-fit:cover; display:block; opacity:0; transition:opacity .12s; }}
  .c img.on {{ opacity:1; }}
  .c:hover {{ outline:2px solid #4a9eff; }}
  .c span {{ position:absolute; right:5px; bottom:5px; background:#000b; color:#fff;
            font-size:12px; padding:1px 6px; border-radius:4px; }}
  #lb {{ position:fixed; inset:0; background:#000e; display:none; z-index:10;
        flex-direction:column; align-items:center; justify-content:center; }}
  #lb.on {{ display:flex; }}
  #lb img, #lb video {{ max-width:95vw; max-height:88vh; object-fit:contain; }}
  #lb p {{ margin:10px; }} #lb a {{ color:#4a9eff; }}
</style></head><body>
<header><b>{title}</b><input id="q" placeholder="Filter by filename…" autofocus>
<select id="t"><option value="">All</option><option value="image">Images</option>
<option value="video">Videos</option></select><span id="n"></span></header>
<div id="grid"></div>
<div id="lb"><img id="big"><video id="vid" controls autoplay playsinline></video>
<p><span id="cap"></span> · <a id="open" target="_blank">open original</a>
 · ←/→ to browse, Esc to close</p></div>
<script src="{items_js}"></script>
<script>
const $ = id => document.getElementById(id);
// The one place an item becomes a URL. Tiers are files in thumbs/ today; a
// packed store would only change this function.
function loc(it, tier) {{
  if (tier === 'full') return BASE + it.name.split('/').map(encodeURIComponent).join('/');
  if (tier === 'view') return it.view ? 'thumbs/' + it.k + '_view.jpg' : loc(it, 'full');
  return 'thumbs/' + it.k + (tier === 'mid' ? '_m' : '') + '.jpg';
}}
const lower = items.map(it => it.name.toLowerCase());
const fmt = s => {{ s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60;
  const ss = String(s % 60).padStart(2, '0'); return h ? h + ':' + String(m).padStart(2, '0') + ':' + ss : m + ':' + ss; }};

// Low-LOD atlases (format in lod_atlas.py): every item also has a square in a
// few big chunk images, drawn as its cell's CSS background. L0 is tiny and fully
// preloaded, so even a flung scroll never shows empty cells; L1 is set on cells
// a few screens around the viewport; the real thumbnail <img> loads (and fades
// in over the atlas) only once scrolling slows down.
const AT = 'thumbs/atlas/';
function sprite(i, L) {{
  const lv = ATLAS.levels[L], s = i % lv.per_chunk, n = lv.size;
  const x = (s % lv.cols) * lv.pitch + lv.pad, y = Math.floor(s / lv.cols) * lv.pitch + lv.pad;
  // In percentages, so it's independent of the cell's on-screen size: the chunk
  // is scaled so one square fills the cell, then shifted to this item's square.
  const pct = (v, total) => total === n ? 0 : v / (total - n) * 100;
  return ['url(' + AT + lv.chunks[Math.floor(i / lv.per_chunk)] + ')',
          lv.width / n * 100 + '% ' + lv.height / n * 100 + '%',
          pct(x, lv.width) + '% ' + pct(y, lv.height) + '%'];
}}
function paint(i, levels) {{  // levels: finest first (drawn on top)
  const ps = levels.map(L => sprite(i, L)), st = cells[i].style;
  st.backgroundImage = ps.map(p => p[0]).join(',');
  st.backgroundSize = ps.map(p => p[1]).join(',');
  st.backgroundPosition = ps.map(p => p[2]).join(',');
}}
const keepL0 = ATLAS.levels[0].chunks.map(n => {{ const im = new Image(); im.src = AT + n; return im; }});

const tstate = new Uint8Array(items.length);  // thumbnail <img>: 0 none, 1 loading, 2 shown, 3 failed
const lod = new Uint8Array(items.length);     // 1 = L1 atlas set as background
const pending = new Set(), shown = new Set();
const cells = items.map((it, i) => {{
  const d = document.createElement('div'); d.className = 'c'; d.title = it.name;
  d.innerHTML = '<img decoding="async">' + (it.type === 'video' ? '<span>▶ ' + fmt(it.dur || 0) + '</span>' : '');
  const im = d.firstChild;
  im.onload = () => {{ im.classList.add('on'); if (tstate[i] === 1) {{ tstate[i] = 2; pending.delete(i); shown.add(i); }} }};
  im.onerror = () => {{ if (tstate[i] === 1) {{ tstate[i] = 3; pending.delete(i); }} }};
  d.onclick = () => show(i);
  $('grid').appendChild(d); return d;
}});
// Debug: gallery.html?lod=0 (or 1) shows only that atlas level, no thumbnails.
const FORCE = new URLSearchParams(location.search).get('lod');
cells.forEach((_, i) => paint(i, FORCE === null ? [0] : [+FORCE]));
function unload(i) {{
  const im = cells[i].firstChild; im.removeAttribute('src'); im.classList.remove('on');
  tstate[i] = 0; pending.delete(i); shown.delete(i);
}}
let visible = [], cur = -1;
function filter() {{
  const q = $('q').value.toLowerCase(), t = $('t').value;
  visible = [];
  items.forEach((it, i) => {{ const ok = lower[i].includes(q) && (!t || it.type === t);
    cells[i].style.display = ok ? '' : 'none'; if (ok) visible.push(i); }});
  $('n').textContent = visible.length + ' / ' + items.length;
  schedule();
}}

// Sharpen near the cursor: cells within RADIUS of it swap their thumb for the
// mid tier. Each is fetched and decoded off-screen first (no flicker), nearest
// first, MAX_LOADS at a time; loads the cursor has moved away from are
// cancelled. Only the MAX_SHARP most recently near cells stay sharp, to cap memory.
const RADIUS = 2, MAX_LOADS = 6, MAX_SHARP = 200;
const state = new Uint8Array(items.length);  // 0 thumb, 1 loading, 2 sharp, 3 failed
const loads = new Map(), sharp = new Set();
let mouse = null, raf = 0, cols = 1, GAP = 0;
function schedule() {{ if (!raf) raf = requestAnimationFrame(update); }}
function measure() {{
  const cs = getComputedStyle($('grid'));
  cols = cs.gridTemplateColumns.split(' ').length; GAP = parseFloat(cs.columnGap) || 0; schedule();
}}
// Scroll speed decides what loads. Faster than FAST viewport-heights/s is a
// fling: atlases only. SETTLE_MS after the last fast scroll event, it has slowed
// down: load real thumbnails for the viewport (+ a little margin).
const FAST = 2.5, SETTLE_MS = 140, L1_MARGIN = 3, THUMB_MARGIN = 0.5, MAX_THUMBS = 1500;
let lastY = scrollY, lastT = performance.now(), fast = false, settle = 0;
function onScroll() {{
  const now = performance.now(), v = Math.abs(scrollY - lastY) / Math.max(1, now - lastT) * 1000;
  lastY = scrollY; lastT = now;
  if (v > FAST * innerHeight) {{
    fast = true; clearTimeout(settle);
    settle = setTimeout(() => {{ fast = false; schedule(); }}, SETTLE_MS);
  }}
  schedule();
}}
// The grid is regular, so which cells are on screen is plain arithmetic:
// one rect read per frame, no hit-testing or observers.
function rows(g, margin) {{  // [first, end) positions in `visible` within viewport ± margin screens
  const rowH = g.height + GAP, top = -g.top - margin * innerHeight, bot = innerHeight - g.top + margin * innerHeight;
  return [Math.max(0, Math.floor(top / rowH) * cols), Math.min(visible.length, Math.ceil(bot / rowH) * cols)];
}}
function update() {{
  raf = 0;
  if (lb.classList.contains('on') || !visible.length || FORCE !== null) return;
  const g = cells[visible[0]].getBoundingClientRect();
  let [a, b] = rows(g, L1_MARGIN);  // L1: one chunk covers hundreds of cells, so this is cheap
  for (let p = a; p < b; p++) {{ const i = visible[p]; if (!lod[i]) {{ lod[i] = 1; paint(i, [1, 0]); }} }}
  if (fast) return;
  [a, b] = rows(g, THUMB_MARGIN);
  const want = new Set();
  for (let p = a; p < b; p++) {{
    const i = visible[p]; want.add(i);
    if (!tstate[i]) {{ tstate[i] = 1; pending.add(i); cells[i].firstChild.src = loc(items[i], 'thumb'); }}
  }}
  for (const i of pending) if (!want.has(i) && state[i] === 0) unload(i);  // scrolled past before it loaded
  for (const i of shown) {{  // oldest first: drop far-away thumbnails, the atlas stands in
    if (shown.size <= MAX_THUMBS) break;
    if (!want.has(i) && state[i] === 0) unload(i);
  }}
  if (mouse) sharpen(g);
}}
function sharpen(g) {{
  const c0 = Math.floor((mouse[0] - g.left) / (g.width + GAP));
  const r0 = Math.floor((mouse[1] - g.top) / (g.height + GAP));
  const want = [];
  for (let dr = -RADIUS; dr <= RADIUS; dr++) for (let dc = -RADIUS; dc <= RADIUS; dc++) {{
    const r = r0 + dr, c = c0 + dc, p = r * cols + c;
    if (r < 0 || c < 0 || c >= cols || p >= visible.length) continue;
    want.push([dr * dr + dc * dc, visible[p]]);
  }}
  want.sort((a, b) => a[0] - b[0]);
  const near = new Set(want.map(w => w[1]));
  for (const [i, im] of loads) if (!near.has(i)) {{ im.src = ''; loads.delete(i); state[i] = 0; }}
  for (const [, i] of want) {{
    if (state[i] === 2) {{ sharp.delete(i); sharp.add(i); }}  // still near: keep it fresh
    else if (state[i] === 0 && loads.size < MAX_LOADS) load(i);
  }}
}}
function load(i) {{
  const im = new Image(), url = loc(items[i], 'mid');
  state[i] = 1; loads.set(i, im); im.src = url;
  im.decode().then(() => {{
    if (loads.get(i) !== im) return;  // cancelled meanwhile
    loads.delete(i); state[i] = 2; cells[i].firstChild.src = url; sharp.add(i);
    if (sharp.size > MAX_SHARP) {{
      const j = sharp.values().next().value; sharp.delete(j);
      cells[j].firstChild.src = loc(items[j], 'thumb'); state[j] = 0;
    }}
    schedule();  // a slot freed up: start the next nearest
  }}, () => {{ if (loads.get(i) === im) {{ loads.delete(i); state[i] = 3; schedule(); }} }});
}}
document.addEventListener('mousemove', e => {{ mouse = [e.clientX, e.clientY]; schedule(); }}, {{ passive: true }});
addEventListener('scroll', onScroll, {{ passive: true }});
addEventListener('resize', measure);
measure();

$('q').oninput = filter; $('t').onchange = filter; filter();
const lb = $('lb'), big = $('big'), vid = $('vid');
function show(i) {{
  cur = i; const it = items[i], isVid = it.type === 'video';
  vid.pause(); vid.style.display = isVid ? '' : 'none'; big.style.display = isVid ? 'none' : '';
  if (isVid) {{ vid.src = loc(it, 'full'); big.removeAttribute('src'); }}
  else {{ big.src = loc(it, 'view'); vid.removeAttribute('src'); vid.load(); }}
  $('cap').textContent = it.name; $('open').href = loc(it, 'full'); lb.classList.add('on');
  // Warm the previous and next images so ←/→ shows them instantly.
  const p = visible.indexOf(i);
  for (const d of [1, -1]) {{
    const n = items[visible[(p + d + visible.length) % visible.length]];
    if (p >= 0 && n.type !== 'video') new Image().src = loc(n, 'view');
  }}
}}
function close() {{ lb.classList.remove('on'); vid.pause(); schedule(); }}
function step(d) {{ const p = visible.indexOf(cur); if (p < 0) return;
  show(visible[(p + d + visible.length) % visible.length]); }}
lb.onclick = e => {{ if (e.target === lb) close(); }};
document.onkeydown = e => {{ if (!lb.classList.contains('on')) return;
  if (e.key === 'Escape') close();
  if (e.key === 'ArrowRight') step(1); if (e.key === 'ArrowLeft') step(-1); }};
</script></body></html>
"""


def load_meta():
    try:
        return json.loads(META.read_text())
    except (OSError, ValueError):
        return {}


def save_meta(meta):
    """Write to a temp file and rename, so a crash can't leave a half-written cache."""
    tmp = META.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, META)


def progress(label, i, total):
    if i % 50 == 0 or i == total:
        print(f"  {label} {i}/{total}", end="\r", flush=True)
    if i == total:
        print()


def main():
    p = argparse.ArgumentParser(description="Build a browsable grid of images and videos.")
    p.add_argument("folder", nargs="?", default="~/Desktop", help="folder to scan (default: ~/Desktop)")
    p.add_argument("--size", type=int, default=200, help="thumbnail size in px (default: 200)")
    p.add_argument("--recursive", action="store_true", help="include subfolders")
    p.add_argument("--no-open", action="store_true", help="don't open the browser when done")
    args = p.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    t0 = time.time()
    pattern = "**/*" if args.recursive else "*"
    files = sorted(
        f for f in folder.glob(pattern)
        if f.suffix.lower() in MEDIA_EXTS and f.is_file() and not f.name.startswith("._")
    )
    if not files:
        raise SystemExit(f"No images or videos found in {folder}")

    THUMBS.mkdir(exist_ok=True)
    meta = load_meta()

    entries = []  # (file, key, kind)
    image_jobs, video_jobs = [], []
    for f in files:
        key = cache_key(f, args.size)
        kind = "video" if f.suffix.lower() in VIDEO_EXTS else "image"
        entries.append((f, key, kind))
        if key in meta:
            continue
        thumb = str(THUMBS / f"{key}.jpg")
        if kind == "video":
            video_jobs.append((key, (str(f), thumb, args.size)))
        else:
            preview = str(THUMBS / f"{key}_view.jpg") if f.suffix.lower() in NEEDS_PREVIEW else None
            image_jobs.append((key, (str(f), thumb, preview, args.size)))

    # Biggest files first, handed out one at a time, so no worker is left
    # holding a run of slow files at the end while the others sit idle.
    image_jobs.sort(key=lambda j: -os.path.getsize(j[1][0]))

    print(f"{len(files)} files found ({sum(k == 'video' for *_, k in entries)} videos), "
          f"{len(files) - len(image_jobs) - len(video_jobs)} cached, "
          f"{len(image_jobs)} images + {len(video_jobs)} videos to process")

    failed = set()

    def collect(label, jobs, results):
        for i, ((key, _), res) in enumerate(zip(jobs, results), 1):
            if res is None:
                failed.add(key)
            meta[key] = res  # None = unreadable, remembered so it isn't retried
            progress(label, i, len(jobs))

    # Videos run on threads (the work is in ffmpeg) alongside the image process pool.
    # The cache is saved even on Ctrl-C, so finished work isn't redone.
    try:
        with ThreadPoolExecutor(VIDEO_JOBS) as vex, ProcessPoolExecutor(os.cpu_count()) as iex:
            vids = vex.map(make_video_thumb, [j for _, j in video_jobs])
            if image_jobs:
                collect("images", image_jobs, iex.map(make_image_thumb, [j for _, j in image_jobs]))
            if video_jobs:
                collect("videos", video_jobs, vids)
    finally:
        save_meta(meta)
    if failed:
        print(f"  skipped {len(failed)} unreadable file(s)")

    # An item is an id plus attributes. Where its bytes live (thumbs/ files,
    # the original) is worked out in the page by loc(), not stored per item.
    items = []
    for f, key, kind in entries:
        res = meta.get(key)
        if res is None:
            continue
        item = {"k": key, "name": f.relative_to(folder).as_posix(), "type": kind}
        if kind == "video":
            item["dur"] = round(res.get("dur", 0), 1)
        elif f.suffix.lower() in NEEDS_PREVIEW:
            item["view"] = 1
        items.append(item)

    # Low-LOD atlases from the small thumbnails, in page order. Only chunks whose
    # members changed are rebuilt; chunks for items no longer shown are removed.
    t1 = time.time()
    atlas, built = lod_atlas.build([(it["k"], THUMBS / f"{it['k']}.jpg") for it in items], ATLAS_DIR)
    atlas.pop("keys")  # the page's item order *is* the slot order
    print(f"  atlas: {built} of {sum(len(lv['chunks']) for lv in atlas['levels'])} chunks built "
          f"in {time.time() - t1:.1f}s")

    title = html.escape(f"{folder.name} ({len(items)} files)")
    # Top-level consts in a classic script are visible to the page's own script.
    ITEMS_JS.write_text(f"const items = {json.dumps(items)}, BASE = {json.dumps(folder.as_uri() + '/')},\n"
                        f"  ATLAS = {json.dumps(atlas)};\n", encoding="utf-8")
    # ?v= so a reload never pairs the new page with a cached old item list.
    src = f"{ITEMS_JS.name}?v={time.time_ns()}"
    OUTPUT.write_text(PAGE.format(title=title, size=args.size, items_js=src), encoding="utf-8")
    print(f"Wrote {OUTPUT} in {time.time() - t0:.1f}s")

    if not args.no_open:
        webbrowser.open(OUTPUT.as_uri())


if __name__ == "__main__":
    main()
