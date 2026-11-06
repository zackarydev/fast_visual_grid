"""Stress test for gallery.py: scale, hostile inputs, cache and concurrency.

Every run uses a private copy of gallery.py under $TMPDIR/fvg_stress/apps/, so its
thumbs/ and gallery.html never touch the repo's real cache. Generated fixtures
(noise images from fake_images.py, ffmpeg "life" videos) are kept between runs
and reused, so only the first run pays for generating them.

Usage:
    python stress_gallery.py                       # all suites, 6000 images
    python stress_gallery.py scale --images 6000 --videos 60 --budget 60
    python stress_gallery.py hostile cache
    python stress_gallery.py scale --links 20      # + the real gallery/ folder x20 via symlinks
"""
import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, ImageStat
from pillow_heif import register_heif_opener

import gallery  # only for cache_key / mid_path, to find a file's outputs
import keyframes
import lod_atlas
from fake_images import noise_image

register_heif_opener()

HERE = Path(__file__).resolve().parent
ROOT = Path(tempfile.gettempdir()) / "fast_visual"
PY = sys.executable
MP4CUT = keyframes.default_mp4cut(HERE)  # the app copies can't find ../fast_mp4_grids themselves
STATS = re.compile(r"(\d+) files found \((\d+) videos\), (\d+) cached, (\d+) images \+ (\d+) videos to process")

results = []  # (suite, name, status, detail)


def check(suite, name, ok, detail=""):
    """ok: True = PASS, False = FAIL, None = INFO (a number worth seeing, no verdict)."""
    status = "INFO" if ok is None else "PASS" if ok else "FAIL"
    results.append((suite, name, status, detail))
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""), flush=True)


# ---------------------------------------------------------------- running gallery.py

def make_app(name):
    """A private copy of gallery.py; its thumbs/ and gallery.html land next to it."""
    app = ROOT / "apps" / name
    shutil.rmtree(app, ignore_errors=True)
    app.mkdir(parents=True)
    for f in ("gallery.py", "lod_atlas.py", "keyframes.py"):
        shutil.copy(HERE / f, app / f)
    return app


def tree_rss_mb(root):
    """Summed RSS of a process and all its descendants (overcounts shared pages a bit)."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    kids, rss = {}, {}
    for line in out.splitlines():
        pid, ppid, r = map(int, line.split())
        kids.setdefault(ppid, []).append(pid)
        rss[pid] = r
    total, stack = 0, [root]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack += kids.get(p, [])
    return total / 1024


class Run:
    def __init__(self, code, out, secs, peak_mb, hung=False):
        self.code, self.out, self.secs, self.peak_mb, self.hung = code, out, secs, peak_mb, hung
        m = STATS.search(out)
        self.found, self.videos, self.cached, self.img_todo, self.vid_todo = (
            map(int, m.groups()) if m else (None,) * 5)
        self.todo = None if m is None else self.img_todo + self.vid_todo
        s = re.search(r"skipped (\d+)", out)
        self.skipped = int(s.group(1)) if s else 0
        k = re.search(r"keyframe maps: (\d+) videos to map", out)
        self.kf_todo = int(k[1]) if k else None
        a = re.search(r"atlas: (\d+) of (\d+) chunks built in ([\d.]+)s", out)
        self.atlas_built, self.atlas_chunks, self.atlas_secs = (int(a[1]), int(a[2]), float(a[3])) if a else (None,) * 3

    def tail(self, n=300):
        return self.out.strip().replace("\r", "\n")[-n:]


def start(app, folder, *extra, env=None):
    cmd = [PY, str(app / "gallery.py"), str(folder), "--no-open", "--mp4cut", str(MP4CUT), *extra]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=env, start_new_session=True)  # own process group, like a terminal


def finish(p, t0, kill_after=None, sig=signal.SIGINT, timeout=900):
    """Wait for p (optionally signalling its whole group after kill_after s), sampling memory."""
    peak, stop = [0.0], threading.Event()

    def sample():
        while not stop.wait(0.2):
            peak[0] = max(peak[0], tree_rss_mb(p.pid))

    threading.Thread(target=sample, daemon=True).start()
    out, hung = "", False
    try:
        if kill_after is not None:
            try:
                out, _ = p.communicate(timeout=kill_after)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, sig)
                timeout = 30  # after Ctrl-C it should wind down quickly
        if p.returncode is None:
            out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        hung = True
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
    stop.set()
    return Run(p.returncode, out or "", time.time() - t0, peak[0], hung)


def run(app, folder, *extra, **kw):
    t0 = time.time()
    return finish(start(app, folder, *extra, env=kw.pop("env", None)), t0, **kw)


def page(app):
    """(html text, items list) of the page a run wrote."""
    text = (app / "gallery.html").read_text(encoding="utf-8")
    js = app / "gallery_items.js"
    data = js.read_text(encoding="utf-8") if js.exists() else text  # older layout: inlined
    m = re.search(r"const items = (\[.*?\]), BASE = (\".*?\")[,;]", data, re.S)
    return text, json.loads(m.group(1)), json.loads(m.group(2))


def load_meta(app):
    try:
        return json.loads((app / "thumbs" / "meta.json").read_text())
    except (OSError, ValueError):
        return None


def thumb_files(app, key, kind_view=False, kf=False):
    """Where gallery.py stores a key's generated sizes. The one place to change if thumbnails get packed."""
    t = str(app / "thumbs" / f"{key}.jpg")
    files = [t, gallery.mid_path(t)]
    if kind_view:
        files.append(str(app / "thumbs" / f"{key}_view.jpg"))
    if kf:
        files.append(kf_sprite(app, key))
    return files


def kf_sprite(app, key):
    return str(app / "thumbs" / f"{key}_kf.jpg")


def kf_problems(app, it):
    """What's wrong with an item's keyframe map, checked against its sprite. [] = fine."""
    k = it.get("kf")
    if not k:
        return ["no keyframe map"]
    bad = []
    if not (k["n"] == len(k["t"]) >= 1):
        bad.append(f"n={k['n']} but {len(k['t'])} times")
    if k["t"] != sorted(k["t"]) or k["t"][0] != 0:
        bad.append(f"times not ascending from 0: {k['t'][:5]}")
    if it.get("dur") and k["t"][-1] > it["dur"] + 0.5:
        bad.append(f"last keyframe {k['t'][-1]}s past the end ({it['dur']}s)")
    try:
        with Image.open(kf_sprite(app, it["k"])) as im:
            want = (k["cols"] * k["w"], -(-k["n"] // k["cols"]) * k["h"])
            if im.size != want:
                bad.append(f"sprite {im.size}, layout wants {want}")
    except OSError as e:
        bad.append(f"sprite unreadable: {e}")
    return bad


def dir_stats(d):
    files = [f for f in Path(d).iterdir() if f.is_file()] if Path(d).exists() else []
    return len(files), sum(f.stat().st_size for f in files) / 1e6


# ---------------------------------------------------------------- fixtures

def photo_dims(rng, max_side):
    long = rng.randint(max_side // 2, max_side)
    short = round(long / rng.choice([4 / 3, 3 / 2, 16 / 9, 1.0]))
    return (long, short) if rng.random() < 0.6 else (short, long)


def _make_image(job):
    path, w, h, seed, fmt = job
    img = noise_image(w, h, seed)
    if fmt == "PNG":
        img.save(path, "PNG", compress_level=1)
    elif fmt == "HEIF":
        img.save(path, "HEIF", quality=60)
    else:
        img.save(path, "JPEG", quality=85)


def make_video(path, w, h, secs, seed, codec=("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p")):
    src = f"life=s={w}x{h}:seed={seed}:r=30:mold=10:ratio=0.4:life_color=#ff9933:death_color=#1a2a40"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", src,
                    "-t", str(secs), *codec, "-g", "60", str(path)], check=True)


def camera_roll(n_images, n_videos, max_side, seed=1):
    """A reusable folder that looks like a phone's camera roll: mostly JPEG, some HEIC/PNG, some video."""
    d = ROOT / "fixtures" / f"roll_{n_images}_{n_videos}_{max_side}_{seed}"
    if (d / ".complete").exists():
        return d
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    rng = random.Random(seed)
    jobs = []
    for i in range(n_images):
        r = rng.random()
        fmt, ext, side = (("JPEG", "jpg", max_side) if r < 0.75 else
                          ("HEIF", "heic", max_side) if r < 0.9 else
                          ("PNG", "png", max_side // 2))  # screenshots: smaller, but PNG is heavy
        w, h = photo_dims(rng, side)
        jobs.append((str(d / f"IMG_{i:05d}.{ext}"), w, h, rng.getrandbits(32), fmt))
    t0 = time.time()
    print(f"  generating {n_images} images + {n_videos} videos in {d} (one-off, reused after)...", flush=True)
    with ProcessPoolExecutor() as ex:
        for i, _ in enumerate(ex.map(_make_image, jobs, chunksize=8), 1):
            if i % 250 == 0 or i == n_images:
                print(f"    images {i}/{n_images}", end="\r", flush=True)
    print()
    vjobs = []
    for i in range(n_videos):
        w, h = rng.choice([(1280, 720), (1920, 1080), (720, 1280)])
        vjobs.append((d / f"VID_{i:04d}.{rng.choice(['mp4', 'mov'])}", w, h, rng.randint(3, 15), rng.getrandbits(31)))
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda j: make_video(*j), vjobs))
    (d / ".complete").write_text("")
    print(f"  generated in {time.time() - t0:.0f}s, {dir_stats(d)[1]:.0f} MB")
    return d


# ---------------------------------------------------------------- suite: scale

def suite_scale(args):
    print(f"\n== scale: {args.images} images + {args.videos} videos")
    folder = camera_roll(args.images, args.videos, args.max_side)
    if args.links:
        # Multiply the real gallery/ folder via symlinks: new paths = new cache keys = real work.
        real = [f for f in (HERE / "gallery").iterdir() if f.suffix.lower() in gallery.MEDIA_EXTS]
        folder_l = ROOT / "fixtures" / f"links_{args.links}"
        shutil.rmtree(folder_l, ignore_errors=True)
        folder_l.mkdir(parents=True)
        for f in folder.iterdir():
            if not f.name.startswith("."):
                (folder_l / f.name).symlink_to(f)
        for i in range(args.links):
            for f in real:
                (folder_l / f"copy{i:03d}_{f.name}").symlink_to(f)
        folder = folder_l
    expected = sum(1 for f in folder.iterdir() if f.suffix.lower() in gallery.MEDIA_EXTS)

    app = make_app("scale")
    cold = run(app, folder, "--size", str(args.size))
    check("scale", "cold run exits cleanly", cold.code == 0 and not cold.hung, cold.tail() if cold.code else "")
    check("scale", "all files found", cold.found == expected, f"{cold.found} / {expected}")
    check("scale", f"cold run under {args.budget}s", cold.secs <= args.budget,
          f"{cold.secs:.1f}s, {expected / cold.secs:.0f} files/s, peak RSS (tree) {cold.peak_mb:.0f} MB")
    check("scale", "nothing skipped", cold.skipped == 0, f"{cold.skipped} skipped")

    warm = run(app, folder, "--size", str(args.size))
    check("scale", "warm run processes nothing", warm.todo == 0, f"{warm.todo} to process")
    check("scale", f"warm run under {args.warm_budget}s", warm.secs <= args.warm_budget, f"{warm.secs:.2f}s")

    text, items, _ = page(app)
    check("scale", "page lists every file", len(items) == expected, f"{len(items)} items")
    check("scale", "item names unique", len({i['name'] for i in items}) == len(items))
    missing = [i["name"] for i in items
               if not all(os.path.exists(f) for f in thumb_files(app, i["k"], i.get("view"), "kf" in i))]
    check("scale", "every item has its generated files", not missing, f"{len(missing)} missing, e.g. {missing[:3]}")
    vids = [i for i in items if i["type"] == "video"]
    check("scale", "cold run maps every video's keyframes", cold.kf_todo == len(vids), f"{cold.kf_todo} / {len(vids)}")
    check("scale", "warm run maps nothing", warm.kf_todo == 0, f"{warm.kf_todo} to map")
    bad = [f"{i['name']}: {p}" for i in vids for p in kf_problems(app, i)]
    check("scale", "every video has a sound keyframe map", not bad, f"{len(bad)} problems, e.g. {bad[:3]}")
    if vids and not bad:
        sprites = [Path(kf_sprite(app, i["k"])) for i in vids]
        mb = sum(f.stat().st_size for f in sprites) / 1e6
        check("scale", "keyframe maps", None, f"{sum(i['kf']['n'] for i in vids)} keyframes in {len(vids)} sprites, "
              f"{mb:.1f} MB, {mb * 1e3 / len(vids):.0f} KB/video")
    n, mb = dir_stats(app / "thumbs")
    check("scale", "thumbs/ footprint", None, f"{n} files, {mb:.0f} MB, {mb * 1e3 / max(n, 1):.1f} KB/file avg")
    suite_atlas_checks(app, items, cold, warm)
    js = app / "gallery_items.js"
    check("scale", "page size", None, f"{len(text.encode()) / 1e3:.0f} KB HTML + "
          f"{js.stat().st_size / 1e6 if js.exists() else 0:.2f} MB item list (all cells built at load)")


def suite_atlas_checks(app, items, cold, warm):
    adir = app / "thumbs" / "atlas"
    check("scale", "atlas built on cold run", bool(cold.atlas_built) and cold.atlas_built == cold.atlas_chunks,
          f"{cold.atlas_chunks} chunks in {cold.atlas_secs}s")
    check("scale", "atlas reused on warm run", warm.atlas_built == 0, f"{warm.atlas_built} rebuilt in {warm.atlas_secs}s")
    m = json.loads((adir / lod_atlas.MANIFEST).read_text())
    check("scale", "atlas covers every item in page order", m["keys"] == [i["k"] for i in items], f"{m['count']} items")
    for L, lv in enumerate(m["levels"]):
        files = [adir / c for c in lv["chunks"]]
        mb = sum(f.stat().st_size for f in files) / 1e6
        check("scale", f"atlas L{L} footprint", None,
              f"{len(files)} chunks of {lv['width']}px, {lv['size']}px squares, {mb:.2f} MB total")
    # The arithmetic is the contract: the square locate() points at must be the
    # item's own, i.e. match its thumbnail better than the neighbouring slots do.
    # Separately, quality: how far each level's squares are from the thumbnail.
    sheets = {}

    def square(L, j):
        name, x, y, s = lod_atlas.locate(m, L, j)
        if name not in sheets:
            sheets[name] = Image.open(adir / name).convert("RGB")
        return name, sheets[name].crop((x, y, x + s, y + s))

    diff = lambda a, b: sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3
    sample = random.Random(0).sample(range(len(items)), min(200, len(items)))
    for L in range(len(m["levels"])):
        own, misses = [], []
        for i in sample:
            with Image.open(app / "thumbs" / f"{items[i]['k']}.jpg") as t:
                s = m["levels"][L]["size"]
                ref = ImageOps.fit(t.convert("RGB"), (s, s), Image.Resampling.LANCZOS)
            name, sq = square(L, i)
            d = {i: diff(sq, ref)}
            for j in range(max(0, i - 3), min(len(items), i + 4)):
                if j != i and square(L, j)[0] == name:
                    d[j] = diff(square(L, j)[1], ref)
            own.append(d[i])
            if min(d, key=d.get) != i:
                misses.append(i)
        own.sort()
        # Rare misses are fine (near-identical neighbours, e.g. dark video frames); a layout bug misses ~all.
        check("scale", f"atlas L{L}: located square is the item's own", len(misses) <= len(sample) // 50,
              f"own slot best match in {len(sample) - len(misses)}/{len(sample)}")
        limit = 12 if L == 0 else 6  # L0 squares are ~14px: mostly colour, detail is expected to go
        check("scale", f"atlas L{L}: quality vs thumbnail", own[-1] < limit,
              f"mean pixel diff median {own[len(own) // 2]:.1f}, max {own[-1]:.1f} /255 (limit {limit})")


# ---------------------------------------------------------------- suite: hostile

def hostile_fixtures(d):
    """Build every awkward input we can think of. Returns {relative name: expected outcome}:
    ok = shown with a thumbnail, skip = listed then rejected as unreadable, hidden = never picked up."""
    exp = {}
    rnd = lambda w, h, mode="RGB": noise_image(w, h, seed=w * 7 + h, mode=mode)

    def img(name, im, fmt, expect="ok", **kw):
        im.save(d / name, fmt, **kw)
        exp[name] = expect

    img("normal.jpg", rnd(800, 600), "JPEG")
    img("UPPER.JPG", rnd(640, 480), "JPEG")
    img("cmyk.jpg", rnd(600, 400).convert("CMYK"), "JPEG")
    img("gray.jpg", rnd(600, 400, "L"), "JPEG")
    img("16bit.png", rnd(600, 400, "L").convert("I").point(lambda v: v * 257).convert("I;16"), "PNG")
    img("rgba.png", rnd(500, 500, "RGBA"), "PNG")
    pal = rnd(400, 300).convert("P", palette=Image.Palette.ADAPTIVE)
    img("palette_transparent.png", pal, "PNG", transparency=0)
    img("la.png", rnd(300, 300, "LA"), "PNG")
    img("tiny_1x1.png", rnd(1, 1), "PNG")
    img("sliver_4000x10.png", rnd(4000, 10), "PNG")
    frames = [rnd(300, 200 + i).resize((300, 200)) for i in range(3)]
    frames[0].save(d / "anim.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    exp["anim.gif"] = "hidden"  # .gif isn't in IMAGE_EXTS
    frames[0].save(d / "anim.png", save_all=True, append_images=frames[1:], duration=100)
    exp["anim.png"] = "ok"
    exif = Image.Exif()
    exif[0x0112] = 6  # stored landscape, displays rotated 90° -> portrait
    img("exif_rotated.jpg", rnd(600, 300), "JPEG", exif=exif.tobytes())
    img("real.heic", rnd(1200, 900), "HEIF")
    img("big_90mp.png", rnd(9500, 9500, "L"), "PNG", compress_level=1)  # just over Pillow's bomb *warning*
    img("bomb_196mp.png", Image.new("1", (14000, 14000)), "PNG", "skip")  # over Pillow's bomb *error*

    # broken files
    (d / "zero.jpg").write_bytes(b"")
    data = (d / "normal.jpg").read_bytes()
    (d / "truncated.jpg").write_bytes(data[: len(data) // 2])
    (d / "garbage.png").write_bytes(os.urandom(5000))
    (d / "garbage.heic").write_bytes(os.urandom(5000))
    (d / "text.mp4").write_text("this is not a video\n")
    (d / "zero.mov").write_bytes(b"")
    (d / "unreadable.jpg").write_bytes(data)
    os.chmod(d / "unreadable.jpg", 0)
    exp.update({n: "skip" for n in ["zero.jpg", "garbage.png", "garbage.heic",
                                     "text.mp4", "zero.mov", "unreadable.jpg"]})
    exp["truncated.jpg"] = "skip"  # Pillow refuses truncated JPEGs by default

    # videos
    lavfi = lambda src, name, *a: subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", src, *a, str(d / name)],
        check=True)
    lavfi("sine=d=2", "audio_only.mp4", "-c:a", "aac")
    exp["audio_only.mp4"] = "skip"
    make_video(d / "one_frame.mp4", 320, 240, 0.034, 1)
    make_video(d / "clip.webm", 640, 360, 2, 2, codec=("-c:v", "libvpx-vp9", "-deadline", "realtime"))
    make_video(d / "clip.mkv", 640, 360, 2, 3)
    make_video(d / "clip.avi", 640, 360, 2, 4, codec=("-c:v", "mpeg4"))
    make_video(d / "portrait.mov", 720, 1280, 3, 5)
    make_video(d / "long_gop.mp4", 640, 360, 20, 6, codec=("-c:v", "libx264", "-preset", "ultrafast",
                                                           "-pix_fmt", "yuv420p", "-g", "9999"))
    # Stored landscape, flagged to display rotated 90° (as phones record): portrait on screen.
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "90",
                    "-i", str(d / "clip.mkv"), "-c", "copy", str(d / "rotated_meta.mp4")], check=True)
    exp.update({n: "ok" for n in ["one_frame.mp4", "clip.webm", "clip.mkv", "clip.avi",
                                   "portrait.mov", "long_gop.mp4", "rotated_meta.mp4"]})

    # names that test escaping in HTML, JSON and URLs
    for name in ["spaces and 'quotes\".jpg", "café 日本語 🎉.jpg", "<!--<script>.jpg",
                 "<b>x.jpg", "100%#?&=.jpg", "new\nline.jpg", "back\\slash.jpg"]:
        shutil.copy(d / "normal.jpg", d / name)
        exp[name] = "ok"
    # A name can't hold "/", but a relative path can: dir "x<" + "script>.jpg" = "x</script>.jpg".
    (d / "x<").mkdir()
    shutil.copy(d / "normal.jpg", d / "x<" / "script>.jpg")
    exp["x</script>.jpg"] = "hidden"  # until --recursive

    # things that should never be picked up
    shutil.copy(d / "normal.jpg", d / "._resource_fork.jpg")
    (d / "broken_link.jpg").symlink_to(d / "does_not_exist.jpg")
    (d / "folder.jpg").mkdir()
    shutil.copy(d / "normal.jpg", d / "folder.jpg" / "inside.jpg")
    (d / "sub").mkdir()
    shutil.copy(d / "normal.jpg", d / "sub" / "nested.jpg")
    (d / "notes.txt").write_text("hi")
    exp.update({n: "hidden" for n in ["._resource_fork.jpg", "broken_link.jpg", "folder.jpg",
                                       "notes.txt", "sub/nested.jpg", "folder.jpg/inside.jpg"]})
    return exp


def outcome(app, folder, name, size, meta):
    key = gallery.cache_key(folder / name, size) if (folder / name).is_file() else None
    if key is None or key not in meta:
        return "hidden", key
    return ("skip" if meta[key] is None else "ok"), key


def suite_hostile(args):
    print("\n== hostile inputs")
    base = ROOT / "fixtures" / "hostile"
    shutil.rmtree(base, ignore_errors=True)
    folder = base / 'tricky <b>&"name'  # the folder name lands in <title> and BASE
    folder.mkdir(parents=True)
    exp = hostile_fixtures(folder)
    folder = folder.resolve()

    app = make_app("hostile")
    r = run(app, folder, "--size", str(args.size), timeout=300)
    check("hostile", "run exits cleanly", r.code == 0 and not r.hung, r.tail() if r.code else f"{r.secs:.1f}s")
    if r.code != 0:
        return
    meta = load_meta(app)
    text, items, base_url = page(app)
    by_name = {i["name"]: i for i in items}

    wrong = []
    for name, want in exp.items():
        got, key = outcome(app, folder, name, args.size, meta)
        if got != want:
            wrong.append(f"{name!r}: expected {want}, got {got}")
    check("hostile", "each file handled as expected", not wrong, "; ".join(wrong))

    def thumb(name):
        it = by_name.get(name)
        return Image.open(thumb_files(app, it["k"])[0]) if it else None

    bad = []
    for name, want in exp.items():
        if want == "ok" and name in by_name:
            t = thumb(name)
            if not (0 < max(t.size) <= args.size and min(t.size) >= 1):
                bad.append(f"{name} {t.size}")
    check("hostile", "thumbnails within size bounds", not bad, ", ".join(bad))
    for name in ["exif_rotated.jpg", "portrait.mov", "rotated_meta.mp4"]:
        t = thumb(name)
        check("hostile", f"{name} thumbnail is portrait", t is not None and t.height > t.width,
              f"{t.size}" if t else "no thumbnail")
    t = thumb("16bit.png")
    if t:
        mean = sum(t.convert("L").getdata()) / (t.width * t.height)
        check("hostile", "16-bit PNG not blown out to white", mean < 240, f"mean brightness {mean:.0f}/255")
    # Keyframe maps: every readable video gets one, upright, from any container
    # but AVI: mp4cut is told its MPEG-4 Part 2 frames are keyframes that the
    # decoder then skips. Browsers can't play AVI anyway; it opens without a map.
    vids = [n for n, w in exp.items() if w == "ok" and Path(n).suffix.lower() in gallery.VIDEO_EXTS
            and n != "clip.avi"]
    bad = [f"{n}: {p}" for n in vids for p in (kf_problems(app, by_name[n]) if n in by_name else ["not listed"])]
    check("hostile", "every readable video has a keyframe map", not bad, "; ".join(bad))
    for name, portrait in [("portrait.mov", True), ("rotated_meta.mp4", True), ("clip.mkv", False)]:
        k = by_name.get(name, {}).get("kf")
        check("hostile", f"{name} keyframes are {'portrait' if portrait else 'landscape'}",
              bool(k) and (k["h"] > k["w"]) == portrait, f"{k['w']}x{k['h']}" if k else "no map")
    for name in ["long_gop.mp4", "one_frame.mp4", "clip.avi"]:
        k = by_name.get(name, {}).get("kf")
        check("hostile", f"{name} keyframe map", None, f"{k['n']} keyframes" if k else "none")
    it = by_name.get("real.heic")
    check("hostile", "HEIC gets a browser-viewable preview",
          bool(it and it.get("view") and os.path.exists(thumb_files(app, it["k"], True)[2])))

    # The page has exactly the template's script tags; filenames can't open or close another.
    def script_tags(html_text):
        low = html_text.lower()
        return low.count("<script"), low.count("</script")

    want_tags = script_tags(gallery.PAGE)
    check("hostile", "filenames can't break out of <script>", script_tags(text) == want_tags,
          f"(open, close) tags: {script_tags(text)}, template has {want_tags}")
    title = re.search(r"<title>(.*?)</title>", text, re.S).group(1)
    check("hostile", "folder name escaped in <title>", "<b>" not in title, title)
    # Rebuild each "open original" URL the way the page does, and check it points at the file.
    enc = lambda s: urllib.parse.quote(s, safe="!'()*-._~")  # = encodeURIComponent
    broken = []
    for it in items:
        url = base_url + "/".join(enc(p) for p in it["name"].split("/"))
        if not Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).exists():
            broken.append(it["name"])
    check("hostile", "original-file URLs resolve", not broken, f"{broken}")

    r2 = run(app, folder, "--size", str(args.size), "--recursive")
    _, items2, _ = page(app)
    names2 = {i["name"] for i in items2}
    check("hostile", "--recursive finds sub/nested.jpg", "sub/nested.jpg" in names2)
    tags2 = script_tags((app / "gallery.html").read_text(encoding="utf-8"))
    check("hostile", "path 'x</script>.jpg' can't close <script>",
          "x</script>.jpg" in names2 and tags2 == want_tags, f"(open, close) tags: {tags2}")
    check("hostile", "--recursive skips files inside a dir named *.jpg?", None,
          "included" if "folder.jpg/inside.jpg" in names2 else "not included")

    # Folders with nothing usable.
    empty = base / "empty"
    empty.mkdir()
    r = run(app, empty)
    check("hostile", "empty folder: clean error", r.code == 1 and "No images" in r.out, r.tail(120))
    locked = base / "locked"
    locked.mkdir()
    shutil.copy(folder / "normal.jpg", locked)
    os.chmod(locked, 0)
    r = run(app, locked)
    os.chmod(locked, 0o755)
    check("hostile", "unreadable folder: says why", "ermission" in r.out, r.tail(120))
    r = run(app, base / "nope")
    check("hostile", "missing folder: says why", "exist" in r.out.lower() or "not found" in r.out.lower(), r.tail(120))
    os.chmod(folder / "unreadable.jpg", 0o644)


# ---------------------------------------------------------------- suite: cache

def suite_cache(args):
    print("\n== cache & concurrency")
    folder = camera_roll(args.cache_images, 6, args.max_side, seed=2)
    n = args.cache_images + 6
    size = str(args.size)

    app = make_app("cache")
    cold = run(app, folder, "--size", size)
    check("cache", "cold run", cold.code == 0, f"{cold.secs:.1f}s for {n} files")
    T = cold.secs

    warm = run(app, folder, "--size", size)
    check("cache", "warm run processes nothing", warm.todo == 0, f"{warm.todo} to process")

    victim = sorted(folder.glob("IMG_*.jpg"))[0]
    st = victim.stat()
    os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    r = run(app, folder, "--size", size)
    os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns))
    check("cache", "touched file is the only one redone", r.todo == 1, f"{r.todo} to process")

    before = dir_stats(app / "thumbs")[0]
    r = run(app, folder, "--size", str(args.size - 50))
    after = dir_stats(app / "thumbs")[0]
    check("cache", "--size change rebuilds everything", r.todo == n, f"{r.todo} to process")
    check("cache", "stale thumbnails cleaned up", after <= before,
          f"thumbs/ went {before} -> {after} files; old size's files are never removed")

    # Ctrl-C halfway: the cache should survive and the next run should resume.
    app = make_app("cache_sigint")
    r = run(app, folder, "--size", size, kill_after=T * 0.5, sig=signal.SIGINT)
    check("cache", "Ctrl-C mid-run exits (doesn't hang)", not r.hung, f"exit {r.code}")
    meta = load_meta(app)
    done = sum(v is not None for v in (meta or {}).values())
    check("cache", "Ctrl-C keeps finished work", done > 0, f"{done} entries saved")
    r = run(app, folder, "--size", size)
    check("cache", "run after Ctrl-C resumes", r.code == 0 and r.cached == done, f"{r.cached} cached")

    # Hard kill (crash, power loss, kill -9): nothing gets to run a finally block.
    app = make_app("cache_sigkill")
    run(app, folder, "--size", size, kill_after=T * 0.5, sig=signal.SIGKILL)
    r = run(app, folder, "--size", size)
    check("cache", "hard kill keeps finished work", (r.cached or 0) > 0,
          f"{r.cached} cached after kill -9 at {T * 0.5:.1f}s; meta.json is only written at the end")

    # Two runs at once sharing one thumbs/ (e.g. two folders opened back to back).
    app = make_app("cache_concurrent")
    halves = [ROOT / "fixtures" / "half_a", ROOT / "fixtures" / "half_b"]
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in gallery.MEDIA_EXTS)
    for h, part in zip(halves, (files[::2], files[1::2])):
        shutil.rmtree(h, ignore_errors=True)
        h.mkdir()
        for f in part:
            (h / f.name).symlink_to(f)
    t0 = time.time()
    procs = [start(app, h, "--size", size) for h in halves]
    runs = [finish(p, t0) for p in procs]
    check("cache", "concurrent runs both succeed", all(r.code == 0 for r in runs), f"exits {[r.code for r in runs]}")
    redo = [run(app, h, "--size", size).todo for h in halves]
    check("cache", "concurrent runs both keep their cache", redo == [0, 0],
          f"to redo afterwards: {redo} (last writer of meta.json wins)")

    # A transient video failure (ffmpeg missing/busy) should be retried next time.
    app = make_app("cache_transient")
    vids = ROOT / "fixtures" / "vids_only"
    shutil.rmtree(vids, ignore_errors=True)
    vids.mkdir()
    for f in folder.glob("VID_*"):
        (vids / f.name).symlink_to(f)
    no_ffmpeg = dict(os.environ, PATH="/usr/bin:/bin")
    r = run(app, vids, "--size", size, env=no_ffmpeg)
    check("cache", "no ffmpeg on PATH: fails gracefully", r.code == 0, r.tail(200) if r.code else f"{r.skipped} skipped")
    r = run(app, vids, "--size", size)
    check("cache", "videos retried once ffmpeg is back", r.vid_todo == 6, f"{r.vid_todo} of 6 retried")
    check("cache", "keyframe maps made once ffmpeg is back", r.kf_todo == 6, f"{r.kf_todo} of 6 mapped")

    # Keyframe maps added to a gallery built without them: only the maps are made.
    app = make_app("cache_kf_later")
    run(app, vids, "--size", size, "--no-keyframes")
    r = run(app, vids, "--size", size)
    check("cache", "keyframe maps added later: thumbnails kept", r.todo == 0 and r.kf_todo == 6,
          f"{r.todo} thumbnails redone, {r.kf_todo} of 6 mapped")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suites", nargs="*", default=["scale", "hostile", "cache"],
                   help="any of: scale hostile cache (default: all)")
    p.add_argument("--images", type=int, default=6000, help="scale: generated images (default 6000)")
    p.add_argument("--videos", type=int, default=60, help="scale: generated videos (default 60)")
    p.add_argument("--max-side", type=int, default=4032, help="long side of generated photos (default 4032, a 12MP phone)")
    p.add_argument("--links", type=int, default=0, help="scale: also symlink the real gallery/ folder this many times")
    p.add_argument("--budget", type=float, default=60, help="scale: max seconds for the cold run (default 60)")
    p.add_argument("--warm-budget", type=float, default=2, help="scale: max seconds for the warm run (default 2)")
    p.add_argument("--cache-images", type=int, default=600, help="cache suite: generated images (default 600)")
    p.add_argument("--size", type=int, default=200)
    p.add_argument("--regen", action="store_true", help="throw away generated fixtures and make new ones")
    args = p.parse_args()
    for s in args.suites:
        if s not in ("scale", "hostile", "cache"):
            p.error(f"unknown suite {s!r}")
    if args.regen:
        shutil.rmtree(ROOT / "fixtures", ignore_errors=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"work dir: {ROOT}")

    t0 = time.time()
    for s in args.suites:
        globals()[f"suite_{s}"](args)

    fails = [r for r in results if r[2] == "FAIL"]
    print(f"\n{sum(r[2] == 'PASS' for r in results)} passed, {len(fails)} failed "
          f"in {time.time() - t0:.0f}s")
    for suite, name, _, detail in fails:
        print(f"  FAIL [{suite}] {name}" + (f": {detail}" if detail else ""))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
