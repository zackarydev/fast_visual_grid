"""Cheap random test images: smooth multi-octave value noise, Pillow only (no numpy).

Each octave is a tiny grid of random pixels blown up with bicubic resampling, so
all the per-pixel work happens in Pillow's C code. A few octaves blended together
give coloured Perlin-ish clouds: never flat, different per seed, and smooth enough
that JPEG sizes stay realistic instead of exploding like white noise would.
"""
import random

from PIL import Image


def noise_image(w, h, seed=None, mode="RGB", octaves=4, base_cells=4):
    """A w x h image of value noise. mode: RGB, RGBA, L or LA."""
    rng = random.Random(seed)
    bands = len(Image.new(mode, (1, 1)).getbands())
    long = max(w, h)
    out = None
    for o in range(octaves):
        cells = base_cells * 2 ** o  # each octave has twice the detail...
        gw = max(2, round(cells * w / long) + 1)
        gh = max(2, round(cells * h / long) + 1)
        grid = Image.frombytes(mode, (gw, gh), rng.randbytes(gw * gh * bands))
        layer = grid.resize((w, h), Image.Resampling.BICUBIC)
        # ...and half the weight, so the coarse shapes dominate.
        out = layer if out is None else Image.blend(out, layer, 0.6 * 0.5 ** o)
    return out


if __name__ == "__main__":
    import sys
    noise_image(800, 600, seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0).show()
