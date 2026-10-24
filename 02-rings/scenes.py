"""Scenes of one tree's life, from a seed falling at night to a sunrise, all drawn in light.

Scene functions take (A, c): A holds the shared assets built once per worker process,
c is the frame context (times, section progress, audio pulses; see FrameCtx in
engine/video/render.py, which loads this file and puts engine/video on the path for gfx).
They return a float32 RGB image of the picture area (letterbox bars are added later); the engine blooms
whatever goes above about 0.75, so bright lines glow.

The world is a hillside seen through a camera (View): units are picture heights, the foot of the tree is the
origin and y points up. Everything is light on dark: the ridges are glowing outlines, rings of light lie on the
ground around the tree, and the tree is made of filaments and points of light in the season's colours, with a
pulse of light climbing from the root to the tips on every kick. Each scene has its own palette. The story
keeps count: each scene covers some years of the tree's life (YEARS), and in the years that grew twice
(FALSE_YEARS) a second ring is laid down, so the rings outrun the years.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gfx

FONT = Path(__file__).resolve().parents[1] / ".tools" / "fonts" / "CormorantGaramond[wght].ttf"

# skies, top of the picture to the bottom (display-space RGB 0..1); the ridges hide most of the lower half
NIGHT = [(0.0, (0.004, 0.008, 0.035)), (0.55, (0.02, 0.04, 0.10)), (1.0, (0.05, 0.07, 0.15))]
DUSK = [(0.0, (0.03, 0.03, 0.12)), (0.22, (0.26, 0.07, 0.27)), (0.38, (0.90, 0.32, 0.22)),
        (0.50, (1.0, 0.58, 0.30)), (1.0, (0.30, 0.10, 0.12))]
STORM = [(0.0, (0.01, 0.0, 0.03)), (0.6, (0.05, 0.01, 0.09)), (1.0, (0.09, 0.02, 0.12))]
HOT = [(0.0, (0.12, 0.04, 0.06)), (0.25, (0.32, 0.09, 0.07)), (0.45, (0.85, 0.36, 0.10)),
       (0.58, (1.0, 0.76, 0.40)), (1.0, (1.0, 0.86, 0.55))]
EMBER = [(0.0, (0.09, 0.02, 0.07)), (0.25, (0.42, 0.06, 0.12)), (0.40, (0.95, 0.33, 0.10)),
         (0.52, (1.0, 0.62, 0.24)), (1.0, (0.35, 0.08, 0.05))]
FROST = [(0.0, (0.0, 0.01, 0.04)), (0.5, (0.03, 0.08, 0.18)), (0.62, (0.14, 0.28, 0.48)), (1.0, (0.06, 0.10, 0.16))]
ROSE = [(0.0, (0.12, 0.10, 0.32)), (0.25, (0.45, 0.22, 0.50)), (0.42, (0.95, 0.50, 0.60)),
        (0.55, (1.0, 0.75, 0.62)), (1.0, (1.0, 0.85, 0.70))]
SUNRISE = [(0.0, (0.03, 0.04, 0.15)), (0.22, (0.22, 0.12, 0.38)), (0.38, (0.80, 0.36, 0.45)),
           (0.50, (1.0, 0.66, 0.38)), (1.0, (1.0, 0.85, 0.60))]

AMBER = np.array([1.0, 0.62, 0.28], np.float32)
GOLD = np.array([1.0, 0.82, 0.46], np.float32)
PALE = np.array([1.0, 0.93, 0.80], np.float32)
WATER = np.array([0.35, 0.75, 1.0], np.float32)
BARK = np.array([0.15, 0.10, 0.075], np.float32)
EARLYWOOD = np.array([0.40, 0.26, 0.15], np.float32)
LATEWOOD = np.array([0.17, 0.095, 0.05], np.float32)
DARK = np.array([0.020, 0.017, 0.015], np.float32)
# the young tree through the year (spring, summer, autumn, winter): colour of its tips, share of its leaves out
SEASON_TIPS = np.array([[1.0, 0.55, 0.85], [0.35, 1.0, 0.6], [1.0, 0.5, 0.15], [0.55, 0.8, 1.0]], np.float32)
SEASON_LEAVES = np.array([0.6, 1.0, 0.75, 0.12], np.float32)

FALSE_YEARS = (5, 11, 14, 16, 19, 21, 23)  # the hard years, when growth stopped and started again
YEARS = {"seed": (0.0, 0.0), "sapling": (0.0, 9.0), "lull": (9.0, 9.0), "storm": (9.0, 13.0),
         "drought": (13.0, 17.0), "autumn": (17.0, 18.0), "winter": (18.0, 22.0), "counting": (22.0, 22.0),
         "spring": (22.0, 24.0), "dawn": (24.0, 25.0)}
FULL_GROWN = 22.0
FLAT = 0.12  # how flat a circle on the ground looks from the camera
RINGS = 16  # rings of light on the ground around the tree


def rows(stops, n):
    """Colour of each of n rows for (position, colour) stops: (n, 3)."""
    pos = np.array([p for p, _ in stops], np.float32)
    cols = np.array([col for _, col in stops], np.float32)
    v = np.linspace(0, 1, n, dtype=np.float32)
    return np.stack([np.interp(v, pos, cols[:, i]) for i in range(3)], -1).astype(np.float32)


def ease(e0, e1, x):
    """smoothstep for one number, as a plain float (a numpy scalar would turn float32 pictures into float64)."""
    return float(gfx.smoothstep(e0, e1, x))


def noise_1d(n, beta, seed):
    """Smooth 1-D noise (zero mean, unit std) of length n, from one row of a fractal texture."""
    row = gfx.fbm_texture(1024, beta, seed)[0]
    return np.interp(np.linspace(0, 1023, n), np.arange(1024), row).astype(np.float32)


def year_of(c, scene):
    a, b = YEARS[scene]
    return a + (b - a) * c.p


def rings_at(year):
    """Rings laid down by this point of the tree's life: one a year, two in the years that grew twice."""
    return int(year) + sum(year >= f - 0.5 for f in FALSE_YEARS)


def season_mix(season, table):
    """A per-season value (spring, summer, autumn, winter) blended smoothly round the year."""
    s = season % 4.0
    i = int(s)
    f = ease(0.0, 1.0, s - i)
    return table[i] + (table[(i + 1) % 4] - table[i]) * f


_fonts = {}


def font(size):
    if size not in _fonts:
        try:
            _fonts[size] = ImageFont.truetype(str(FONT), size)
        except OSError:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


class Assets:
    """Shapes, textures and particles shared by every frame (deterministic, built once per process)."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.s = w / 1920
        self.lw, self.lh = w // 4, h // 4  # low-res buffer for skies and soft light
        self.lyy, self.lxx = np.mgrid[0:self.lh, 0:self.lw].astype(np.float32)
        self.yy = np.arange(h, dtype=np.float32)[:, None]
        self.xx = np.arange(w, dtype=np.float32)[None]
        rng = np.random.default_rng(3)

        self.gx = np.linspace(-8, 8, 4096).astype(np.float32)
        self.gn = noise_1d(4096, 1.6, 41)
        self.fn = noise_1d(4096, 1.9, 42)
        self.cloud = gfx.fbm_texture(512, 1.35, 13)
        haze = gfx.blur(np.pad(self.cloud, 48, mode="wrap")[..., None], 6)[48:-48, 48:-48, 0]
        self.haze = (haze / haze.std()).astype(np.float32)  # the cloud texture blurred; still tiles
        self.soil = gfx.fbm_texture(256, 1.2, 8)
        self.mud = self._mud()
        self.stars = np.c_[rng.uniform(0, 1, 700), rng.uniform(0, 0.7, 700), rng.power(3.0, 700),
                           rng.uniform(0, 6.3, 700)].astype(np.float32)
        self.motes = np.c_[rng.uniform(0, 1, 220), rng.uniform(0.05, 0.85, 220), rng.uniform(0.3, 1.0, 220),
                           rng.uniform(0, 1, 220)].astype(np.float32)
        self.rain = np.c_[rng.uniform(0, 1, 700), rng.uniform(0, 1, 700), rng.uniform(1.4, 2.2, 700),
                          rng.uniform(0.04, 0.09, 700)].astype(np.float32)
        self.snow = np.c_[rng.uniform(0, 1, 1400), rng.uniform(0, 1, 1400), rng.uniform(0.05, 0.16, 1400),
                          rng.uniform(0.6, 2.6, 1400), rng.uniform(0, 6.3, 1400)].astype(np.float32)
        self.flakes = np.c_[rng.uniform(0, 1, 160), rng.uniform(0, 1, 160), rng.uniform(0.06, 0.16, 160),
                            rng.uniform(0.6, 1.0, 160), rng.uniform(0, 6.3, 160), rng.integers(0, 3, 160)
                            ].astype(np.float32)
        self.bolts = [self._bolt(rng) for _ in range(6)]
        glints = np.c_[rng.uniform(0, 1, w), rng.uniform(0, 1, w), rng.uniform(0, 6.3, w)].astype(np.float32)
        # points of light in the grass along the near ridge: world x, depth below the ridge, kind
        self.dots = np.c_[rng.uniform(-1.6, 1.6, 900), rng.uniform(0.002, 0.09, 900) ** 1.3,
                          rng.integers(0, 3, 900)].astype(np.float32)

        self.tree = self._tree(rng)
        self.roots = self._roots(rng)
        self._rings(rng)
        cum = np.cumsum(self.tree["length"]) / self.tree["length"].sum()  # frost: more on the longer branches
        self.frost_seg = np.minimum(np.searchsorted(cum, glints[:, 0]), len(cum) - 1)
        self.frost_frac, self.frost_phase = glints[:, 1], glints[:, 2]
        self.vig = gfx.vignette(h, w, 0.36)
        self.grain = [rng.normal(0, 1, (h // 2, w // 2, 1)).astype(np.float32) for _ in range(6)]

    @staticmethod
    def _mud(n=256, cells=48, seed=17):
        """Dried mud as a tileable texture: polygonal cells (Voronoi) with wavering edges, 1 along the cracks
        between them; returned sharp and at three blurs, for seeing it from further off."""
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        x = xx + 2.5 * gfx.fbm_texture(n, 1.8, seed + 1)
        y = yy + 2.5 * gfx.fbm_texture(n, 1.8, seed + 2)
        d1 = np.full((n, n), np.inf, np.float32)
        d2 = d1.copy()
        for px, py in rng.uniform(0, n, (cells, 2)):
            dx = (x - px + n / 2) % n - n / 2
            dy = (y - py + n / 2) % n - n / 2
            d = np.sqrt(dx * dx + dy * dy)
            d2 = np.minimum(d2, np.maximum(d1, d))
            d1 = np.minimum(d1, d)
        mud = gfx.smoothstep(3.0, 0.4, d2 - d1).astype(np.float32)
        tile = np.pad(mud, 32, mode="wrap")[..., None]  # blurred as a tile, so it still wraps
        return [mud] + [gfx.blur(tile, sg)[32:-32, 32:-32, 0] for sg in (1.0, 2.0, 4.0)]

    @staticmethod
    def _bolt(rng):
        """A lightning bolt as a jagged line in picture fractions, with one fork."""
        x, y, pts = rng.uniform(0.15, 0.85), 0.0, []
        end = rng.uniform(0.45, 0.7)
        while y < end:
            pts.append((x, y))
            x += rng.normal(0, 0.018)
            y += rng.uniform(0.02, 0.05)
        k = len(pts) // 3
        fork = [pts[k]]
        fx, fy = pts[k]
        for _ in range(len(pts) // 2):
            fx += rng.normal(0.012, 0.016)
            fy += rng.uniform(0.015, 0.035)
            fork.append((fx, fy))
        return np.array(pts, np.float32), np.array(fork, np.float32)

    def _tree(self, rng):
        """A lone tree grown the way one grows: a leading stem keeps going up and sends out side shoots to
        alternating sides, and every shoot does the same on a smaller scale, bending back towards the light,
        so lengths, angles and spacing all vary. Segments are (parent, angle from the parent, length, width),
        lengths in picture heights; each bows a little when drawn. Leaves sit in clusters along the outer
        shoots."""
        par, ang, length, width, depth, order, bend = [], [], [], [], [], [], []

        def add(parent, a, ln, wd, d, o):
            par.append(parent)
            ang.append(a)
            length.append(ln)
            width.append(wd)
            depth.append(d)
            order.append(o)
            bend.append(rng.normal(0, 0.03 if o == 0 else 0.10))
            return len(par) - 1

        def shoot(parent, rel, heading, ln, wd, d, o):
            side = 1 if rng.uniform() < 0.5 else -1
            steps = int(rng.integers(*((4, 7), (4, 7), (3, 6), (3, 5), (2, 4))[o]))
            for s in range(steps):
                i = add(parent, rel, ln, wd, d, o)
                if o < 4 and (o > 0 or s >= 1) and (o < 3 or rng.uniform() < 0.75):
                    turn = side * rng.uniform(*((0.8, 1.25), (0.6, 1.05), (0.5, 0.95), (0.45, 0.9))[o])
                    grow = rng.uniform(0.85, 1.0) if o == 0 else rng.uniform(0.62, 0.82)
                    shoot(i, turn, heading + turn, ln * grow, wd * rng.uniform(0.5, 0.65), d + 1, o + 1)
                    side = -side
                parent, d = i, d + 1
                # each step bends back towards the light: the trunk firmly, the spreading limbs only a little
                rel = rng.normal(0, 0.05 if o == 0 else 0.11) - (0.25 if o == 0 else 0.08) * heading
                heading += rel
                ln *= rng.uniform(0.82, 0.93)
                wd *= 0.80

        shoot(-1, 0.04, 0.04, 0.072, 0.046, 0, 0)  # leaning a little, the way the wind taught it
        par, depth, order = np.array(par), np.array(depth), np.array(order)
        n = len(par)
        terminal = np.ones(n, bool)
        terminal[par[par >= 0]] = False
        birth = np.empty(n, np.float32)  # growth order: trunk first, twigs last
        birth[np.lexsort((np.arange(n), depth))] = np.linspace(0, 1, n)
        leaf_seg, leaf_frac, leaf_off, leaf_size = [], [], [], []
        for i in np.where(order >= 3)[0]:
            k = 4 if terminal[i] else 2
            leaf_seg += [i] * k
            leaf_frac.append(rng.uniform(0.2, 1.0, k))
            leaf_off.append(rng.normal(0, 0.011, (k, 2)))
            leaf_size.append(rng.uniform(0.004, 0.008, k))
        m = len(leaf_seg)
        return {"par": par, "ang": np.array(ang), "length": np.array(length), "width": np.array(width),
                "depth": depth, "order": order, "bend": np.array(bend), "terminal": terminal, "birth": birth,
                "phase": rng.uniform(0, 2 * np.pi, n),
                "levels": [np.where(depth == d)[0] for d in range(depth.max() + 1)],
                "leaf_seg": np.array(leaf_seg), "leaf_frac": np.concatenate(leaf_frac),
                "leaf_off": np.concatenate(leaf_off), "leaf_size": np.concatenate(leaf_size),
                "leaf_phase": rng.uniform(0, 2 * np.pi, m), "leaf_keep": rng.uniform(0, 1, m),
                "leaf_tone": rng.uniform(0, 1, m), "leaf_kind": rng.integers(0, 3, m),
                "blossom": rng.uniform(0, 1, m) < 0.45}

    def _roots(self, rng):
        """Roots under the tree as segments in picture heights from the trunk base (y down), each with its
        distance from the base along the root; the longest roots end in water."""
        segs = []

        def grow(x, y, a, wd, s, steps, lateral):
            pts = [(x, y, s)]
            for _ in range(steps):
                a = a * 0.96 + rng.normal(0, 0.16)  # wander, but keep heading down
                step = 0.016
                nx, ny = x + step * np.sin(a), y + step * np.cos(a)
                segs.append((x, y, nx, ny, wd, s, s + step))
                x, y, s = nx, ny, s + step
                wd *= 0.965
                pts.append((x, y, s))
                if lateral and rng.uniform() < 0.13:
                    side = 1 if rng.uniform() < 0.5 else -1
                    grow(x, y, a + side * rng.uniform(0.6, 1.2), wd * 0.6, s, int(rng.integers(5, 13)), False)
            return pts

        water = []
        for i, a in enumerate(np.linspace(-1.05, 1.05, 6)):
            steps = int(rng.integers(26, 34)) if i in (1, 3, 4) else int(rng.integers(14, 22))
            pts = grow(0.0, 0.0, a, 0.009, 0.0, steps, True)
            if i in (1, 3, 4):
                water.append(np.array(pts[::-1]))  # path from the tip back up to the trunk
        segs = np.array(segs, np.float32)
        return {"seg": segs[np.argsort(segs[:, 5])], "water": water, "reach": max(p[0, 2] for p in water),
                "glints": rng.normal(0, 1, (60, 2)).astype(np.float32), "glint_phase": rng.uniform(0, 6.3, 60)}

    def _rings(self, rng):
        """Ring boundaries (radius as a fraction of the grown trunk) and the year each was laid down, the
        false rings of the hard years among them; angular tables for the wobble, wood rays and checks."""
        width = rng.lognormal(0, 0.25, 25)
        width[np.array(FALSE_YEARS) - 1] *= 0.6  # the hard years stayed narrow
        b, when, false, acc = [0.0], [0.0], [False], 0.0
        for y in range(1, 26):
            if y in FALSE_YEARS:  # growth stopped mid-season and started again: one year, two rings
                b.append(acc + width[y - 1] * 0.5)
                when.append(y - 0.5)
                false.append(True)
            acc += width[y - 1]
            b.append(acc)
            when.append(float(y))
            false.append(False)
        self.ring_b = (np.array(b) / acc).astype(np.float32)
        self.ring_t = np.array(when, np.float32)
        self.ring_false = np.array(false)
        self.n_rings = len(b) - 1
        gap = np.diff(self.ring_b)
        self.ring_gap = np.r_[gap[0], 0.5 * (gap[:-1] + gap[1:]), gap[-1]].astype(np.float32)
        self.ring_i = np.arange(self.n_rings + 1, dtype=np.float32)
        th = np.linspace(-np.pi, np.pi, 4096, endpoint=False).astype(np.float32)
        self.th = th
        self.wob = (1 + 0.045 * np.sin(th + 0.7) + 0.03 * np.sin(2 * th + 2.1) + 0.015 * np.sin(3 * th + 4.0)
                    + 0.006 * noise_1d(4096, 1.5, 51)).astype(np.float32)
        self.fiber = (0.5 + 0.5 * np.tanh(noise_1d(4096, 0.6, 52))).astype(np.float32)
        self.bark_tex = (0.5 + 0.5 * np.tanh(1.4 * noise_1d(4096, 0.9, 53))).astype(np.float32)
        crack = np.zeros_like(th)
        for a in rng.uniform(-np.pi, np.pi, 6):
            d = np.angle(np.exp(1j * (th - a)))
            crack = np.maximum(crack, np.exp(-(d / rng.uniform(0.004, 0.010)) ** 2))
        self.crack = crack.astype(np.float32)
        self.log_r = 0.47 * self.h
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.count_c = (0.5 * self.w, 0.52 * self.h)
        dx, dy = xx - self.count_c[0], self.count_c[1] - yy
        self.count_th, self.count_r = np.arctan2(dy, dx).astype(np.float32), np.hypot(dx, dy).astype(np.float32)


# ------------------------------------------------------------------ camera and ground
class View:
    """Camera on the hillside: world units are picture heights, the foot of the tree is the origin, y is up."""

    def __init__(self, A, cx, cy, zoom):
        # plain floats: a numpy scalar here would turn every float32 picture computed from the view into float64
        self.A, self.cx, self.cy, self.zoom = A, float(cx), float(cy), float(zoom)
        self.k = self.zoom * A.h

    def sx(self, x):
        return self.A.w / 2 + (x - self.cx) * self.k

    def sy(self, y):
        return self.A.h / 2 - (y - self.cy) * self.k

    def wx(self, sx):
        return self.cx + (sx - self.A.w / 2) / self.k

    def wy(self, sy):
        return self.cy - (sy - self.A.h / 2) / self.k

    def far(self, depth):
        """The same camera seen from a layer `depth` times as far: it pans and zooms less."""
        return View(self.A, self.cx / depth, self.cy / depth - 0.02 * depth, self.zoom ** (1 / depth))


def ground_h(A, x):
    """Height of the near hill at world x; the tree stands on its crest at x = 0."""
    return 0.11 * np.exp(-(x / 0.8) ** 2) - 0.11 + 0.006 * np.interp(x, A.gx, A.gn)


def ridge(A, x, freq, shift):
    """Rough 1-D noise along a skyline (wraps, so it never runs out)."""
    return np.interp((x * freq + shift + 8.0) % 16.0 - 8.0, A.gx, A.fn)


def far_h(A, x):
    """The ridge behind the hill."""
    return (-0.05 + 0.04 * ridge(A, x, 2.0, 1.0) + 0.012 * ridge(A, x, 18.0, 4.0)
            + 0.10 * np.exp(-((x + 0.7) / 0.3) ** 2) + 0.07 * np.exp(-((x - 1.2) / 0.35) ** 2))


def farther_h(A, x):
    """The mountains on the horizon."""
    return (-0.02 + 0.05 * ridge(A, x, 1.5, 5.0) + 0.015 * ridge(A, x, 14.0, 2.0)
            + 0.20 * np.exp(-((x - 0.8) / 0.45) ** 2) + 0.13 * np.exp(-((x + 1.0) / 0.4) ** 2))


def hill(A, v, height):
    """Coverage of a hill (picture size) and its outline row for every column."""
    gy = v.sy(height(A, v.wx(A.xx[0]))).astype(np.float32)
    cover = np.zeros((A.h, A.w), np.float32)
    top = int(max(0, gy.min() - 2))
    cover[top:] = gfx.smoothstep(-1.0, 1.0, A.yy[top:] - gy[None])
    return cover, gy


def ring_radius(k):
    """Radius of ring k of light on the ground around the tree (world units)."""
    return 0.09 * k + 0.012 * k * k


def glow_hills(A, img, v, rim, ring_col, kick=0.0, fill=(0.0, 0.0, 0.0), ground=0.15, glow=0.12, lit=None,
               ripples=(), pool=0.0):
    """Ridges as glowing outlines over dark fills, each lit softly down from its edge, and the near hill with
    rings of light round the tree on it: `lit` per ring (fading outward by default), `ripples` = [(radius,
    brightness)] running out across them, a `pool` of light at the foot. Returns the picture, the near hill's
    coverage and its outline row."""
    rim = np.asarray(rim, np.float32)
    fill = np.asarray(fill, np.float32)
    for height, depth, k in ((farther_h, 4.0, 0.35), (far_h, 2.2, 0.6)):
        fv = v.far(depth)
        gy = fv.sy(height(A, fv.wx(A.xx[0]))).astype(np.float32)
        tone = fill * (0.6 if depth > 3 else 0.35)
        y0 = int(max(0, gy.min() - 6 * A.s))
        y1 = int(min(A.h, gy.max() + 160 * A.s))  # below this the range covers all and its glow has faded
        below = A.yy[y0:y1] - gy[None]
        cover = gfx.smoothstep(-1.0, 1.0, below)
        light = np.exp(-(below / (1.3 * A.s)) ** 2) + glow * np.exp(-np.abs(below) / (28 * A.s)) * cover
        img[y0:y1] = img[y0:y1] * (1 - 0.96 * cover[..., None]) + cover[..., None] * tone + (light * k)[..., None] * rim
        img[y1:] = img[y1:] * 0.04 + tone
    nm, gy = hill(A, v, ground_h)
    y0 = int(max(0, gy.min() - 6 * A.s))
    below = A.yy[y0:] - gy[None]
    m = nm[y0:]
    light = np.exp(-(below / (1.4 * A.s)) ** 2) * (1.3 + 0.8 * kick) + glow * np.exp(-np.abs(below) / (40 * A.s)) * m
    img[y0:] = img[y0:] * (1 - 0.92 * m[..., None]) + m[..., None] * (fill * ground) + light[..., None] * rim
    if lit is None:
        lit = (0.55 * (1 - np.arange(RINGS + 1) / (RINGS + 1)) ** 1.5).astype(np.float32)
    img += ground_rings(A, v, gy, nm, lit, ripples, pool)[..., None] * np.asarray(ring_col, np.float32)
    return img, nm, gy


def ground_rings(A, v, gy, nm, lit, ripples=(), pool=0.0):
    """Rings of light lying on the near hill round the foot of the tree, flattened by the view, with ripples
    running out across them and a pool of light at the foot: an (h, w) intensity, kept smooth by measuring
    each pixel's distance on screen to the nearest ring."""
    out = np.zeros((A.h, A.w), np.float32)
    top = int(max(0, gy.min() - 2))
    ex = A.xx - v.sx(0.0)
    ey = (A.yy[top:] - v.sy(0.0)) / FLAT
    e = np.sqrt(ex * ex + ey * ey) + 1e-3  # radius of the ground circle through each pixel, in pixels
    px = v.k / (np.sqrt(ex * ex + (ey / FLAT) ** 2) / e)  # screen pixels per world unit of that radius
    r = e / v.k
    near = np.clip(np.rint((np.sqrt(0.0081 + 0.048 * r) - 0.09) / 0.024), 1, RINGS)
    val = np.exp(-(np.abs(r - ring_radius(near)) * px / (0.9 * A.s)) ** 2) * lit[near.astype(np.int32)]
    for radius, power in ripples:
        if power > 0.01:
            val += np.exp(-((r - radius) * px / (1.6 * A.s)) ** 2) * power
    if pool > 0:
        val += np.exp(-r / 0.3) * pool
    out[top:] = val * nm[top:]
    return out


def bar_ripples(c, speed=0.32, count=3, power=1.6):
    """A ring of light sent out across the ground from the foot of the tree on every bar: [(radius, brightness)]."""
    bar = 4 * c.beat
    since = c.tau % bar
    return [((since + j * bar) * speed, power * float(np.exp(-(since + j * bar) / (2.0 * bar))))
            for j in range(count)]


def ground_lights(A, img, v, c, colors, amount, size=1.0):
    """Points of light in the grass along the near ridge, twinkling; bigger the nearer they lie."""
    if amount <= 0.01:
        return
    D = A.dots
    x, y = v.sx(D[:, 0]), v.sy(ground_h(A, D[:, 0]) - D[:, 1])
    tw = 0.55 + 0.45 * np.sin(c.t * 1.7 + D[:, 0] * 40.0)
    col = np.asarray(colors, np.float32).reshape(-1, 3)[D[:, 2].astype(np.int32) % len(colors)]
    gfx.splat_points(img, x, y, (col * (amount * tw)[:, None]).astype(np.float32),
                     (0.7 + 10 * D[:, 1]) * size * v.zoom ** 0.5 * A.s)


def cracks(A, v, gy, nm):
    """The dried ground split into a network of cracks, seen in perspective so they grow finer towards the
    crest: an (h, w) intensity, worked at half res."""
    bx, by = v.sx(0.0), v.sy(0.0)
    top = int(max(0, gy.min()))
    ys = np.arange(top, A.h, 2, dtype=np.float32)[:, None] + 0.5
    xs = np.arange(0, A.w, 2, dtype=np.float32)[None] + 0.5
    lift = np.maximum(ys - (by - 0.12 * A.h), 1.0)  # rows below where the ground plane meets the sky
    gx, gz = (xs - bx) / lift, A.h / lift  # on the ground, in camera heights
    level = np.clip(np.log2(np.maximum(100 * A.h / lift ** 2, 1.0)), 0, 3)  # from the texels a pixel spans
    tex = np.zeros((len(ys), xs.shape[1]), np.float32)
    for j, mud in enumerate(A.mud):
        weight = np.clip(1 - np.abs(level - j), 0, 1)
        if weight.max() > 0:
            tex += gfx.sample(mud, gx * 150 + 31, gz * 150) * weight
    hot = 0.45 + 0.55 * gfx.smoothstep(-0.6, 1.2, gfx.sample(A.haze, gx * 40 + 7, gz * 40))
    half = (tex * hot * gfx.smoothstep(by + 4 * A.s, by + 70 * A.s, ys)).astype(np.float32)  # too fine at the crest
    out = np.zeros((A.h, A.w), np.float32)
    out[top:] = gfx.resize(half[..., None], A.w, A.h - top)[..., 0]
    return out * nm


def glow_earth(A, v, c, year, front):
    """Under the hill: dark earth with faint strata, and the roots in it as filaments of amber light grown out to
    `front` (distance along the root), bright at their growing tips; where a root has reached water the water
    glints, and after each kick it climbs the root as a spark."""
    wx, wy = v.wx(A.lxx * (A.w / A.lw)), v.wy(A.lyy * (A.h / A.lh))
    wav = (wy + 0.02 * gfx.sample(A.soil, wx * 30, wy * 30)) * 16.0
    strata = np.exp(-((wav - np.rint(wav)) / 0.06) ** 2) * 0.05
    under = np.array([0.010, 0.008, 0.020], np.float32) + strata[..., None] * np.array([0.5, 0.35, 1.0], np.float32)
    under = gfx.resize(under.astype(np.float32), A.w, A.h)
    S = A.stars  # specks of mineral in the soil, catching the light
    tw = (0.3 + 0.7 * np.sin(c.t * (0.8 + S[:, 2]) + S[:, 3]) ** 2) * S[:, 2] * 0.35
    speck = np.where((S[:, 3] > 3.0)[:, None], AMBER, WATER) * tw[:, None]
    gfx.splat_points(under, v.sx((S[:, 0] - 0.5) * 2.0), v.sy(-0.06 - S[:, 1] * 0.9), speck.astype(np.float32),
                     np.full(len(S), 1.0 * A.s))
    R = A.roots
    _, rsize = age_growth(year)
    seg = R["seg"]
    vis = seg[:, 5] < front
    bx, by = v.sx(0.0), v.sy(0.0)
    sc = v.k * (0.5 + 0.5 * rsize)
    layer = Image.new("L", (A.w, A.h))
    d = ImageDraw.Draw(layer)
    for x0, y0, x1, y1, wd, s0, s1 in seg[vis]:
        k = min(1.0, (front - s0) / (s1 - s0))
        d.line([(bx + x0 * sc, by + y0 * sc), (bx + (x0 + (x1 - x0) * k) * sc, by + (y0 + (y1 - y0) * k) * sc)],
               fill=255, width=max(1, int(round(1.2 * wd * sc))))
    rl = np.asarray(layer, np.float32)[..., None] / 255
    under += (gfx.blur(rl, 0.6 * A.s) * 1.1 + gfx.blur(rl, 5 * A.s) * 0.7) * AMBER * (0.8 + 0.5 * c.kick_pulse)
    tips = seg[vis & (seg[:, 6] >= front)]
    if len(tips):
        k = np.clip((front - tips[:, 5]) / (tips[:, 6] - tips[:, 5]), 0, 1)
        gfx.splat_points(under, bx + (tips[:, 0] + (tips[:, 2] - tips[:, 0]) * k) * sc,
                         by + (tips[:, 1] + (tips[:, 3] - tips[:, 1]) * k) * sc,
                         np.tile(GOLD * 1.8, (len(tips), 1)).astype(np.float32), np.full(len(tips), 2.4 * A.s))
    since = c.t - c.last_kick
    for path in R["water"]:
        reached = ease(path[0, 2] - 0.02, path[0, 2], front)
        g = R["glints"]
        tw = 0.5 + 0.5 * np.sin(c.t * 2.0 + R["glint_phase"])
        gx = bx + path[0, 0] * sc + g[:, 0] * 10 * A.s
        gy = by + path[0, 1] * sc + 12 * A.s + g[:, 1] * 6 * A.s
        gfx.splat_points(under, gx, gy, (WATER[None] * (0.15 + 1.2 * reached) * tw[:, None]).astype(np.float32),
                         np.full(len(g), 1.6 * A.s))
        pool = np.array([[0.30], [0.10]], np.float32) * WATER * (0.2 + 0.8 * reached)  # the water, glowing
        gfx.splat_points(under, np.full(2, gx.mean()), np.full(2, gy.mean()), pool, np.array([9.0, 26.0]) * A.s)
        if reached > 0.99 and since < 1.4:  # water climbs the root after each kick
            s = path[:, 2]
            pos = path[0, 2] * (1 - since / 1.4) ** 0.7
            px = bx + np.interp(-pos, -s, path[:, 0]) * sc
            py = by + np.interp(-pos, -s, path[:, 1]) * sc
            col = gfx.lerp(WATER, PALE, since / 1.4) * 2.4 * (1 - since / 1.4)
            gfx.splat_points(under, np.array([px]), np.array([py]), col[None].astype(np.float32),
                             np.array([3.5 * A.s]))
    return under


# ------------------------------------------------------------------ sky
def glow_sky(A, c, stops, stars=1.0, haze=None, haze_amount=0.0, drift=2.0, aurora=None, clouds=0.0,
             cloud_body=(0.0, 0.0, 0.0), cloud_rim=(0.0, 0.0, 0.0), cloud_drift=6.0, light=None):
    """A deep gradient sky, worked at low res and scaled up: a drifting haze, aurora curtains (bottom colour,
    top colour, amount), a bank of dark clouds whose lower edges catch the light (and which `light` =
    (x, y, power, colour), in picture fractions, lights from inside), then stars."""
    low = np.repeat(rows(stops, A.lh)[:, None], A.lw, 1)
    y = A.lyy / A.lh
    if haze is not None and haze_amount > 0:
        u = (A.lxx + c.t * drift) * (512 / A.lw) * 0.4
        neb = gfx.smoothstep(-0.3, 1.2, gfx.sample(A.haze, u, A.lyy * (512 / A.lh) * 0.3)) * gfx.smoothstep(1.0, 0.2, y)
        low += (neb * haze_amount)[..., None] * np.asarray(haze, np.float32)
    if aurora is not None:
        low += curtains(A, c, *aurora)
    if clouds > 0:  # soft banks stretched along the horizon, a little torn at the edges
        u = (A.lxx + c.t * cloud_drift) * (512 / A.lw) * 0.35
        v = A.lyy * (512 / A.lh) * 0.3
        dens = gfx.sample(A.haze, u, v) + 0.12 * gfx.sample(A.cloud, u * 2.0, v * 2.0)
        cover = gfx.smoothstep(1.0 - 2.4 * clouds, 2.0 - 2.4 * clouds, dens) * gfx.smoothstep(0.62, 0.25, y)
        edge = np.clip(cover - np.roll(cover, -3, 0), 0, 1) * 2.5  # cloud here and clear just below: a lower edge
        shade = (0.6 + 0.4 * gfx.smoothstep(-1.0, 1.5, gfx.sample(A.haze, u * 1.7 + 50, v * 1.7)))[..., None]
        low = (low * (1 - 0.92 * cover[..., None]) + cover[..., None] * np.asarray(cloud_body, np.float32) * shade
               + np.clip(edge, 0, 1)[..., None] * np.asarray(cloud_rim, np.float32))
        if light is not None and light[2] > 0.01:
            lx, ly, power, col = light
            d = np.hypot(A.lxx / A.lw - lx, (y - ly) * (A.lh / A.lw))
            low += (cover * np.exp(-d / 0.10) * power)[..., None] * np.asarray(col, np.float32)
    img = gfx.resize(low.astype(np.float32), A.w, A.h)
    if stars > 0:
        S = A.stars
        tw = 0.6 + 0.4 * np.sin(S[:, 3] + c.t * (1.5 + 2 * S[:, 2]))
        col = np.array([0.85, 0.9, 1.0], np.float32)[None] * (S[:, 2] * tw * stars * 1.6)[:, None]
        gfx.splat_points(img, S[:, 0] * A.w, S[:, 1] * A.h, col.astype(np.float32), (0.6 + 1.1 * S[:, 2] ** 3) * A.s)
    return img


def curtains(A, c, bottom, top, amount, height=0.26, spread=0.13):
    """Aurora: curtains of light hanging in the sky, rippling slowly, brightest along their lower hem and fading
    upward from the `bottom` colour to the `top` one (low res)."""
    x = A.lxx / A.lw
    hem = height + 0.06 * np.sin(x * 5.0 + c.t * 0.15) + 0.03 * np.sin(x * 13.0 - c.t * 0.25)
    rise = (hem - A.lyy / A.lh) / spread  # 0 on the hem, 1 one spread above it
    veil = np.where(rise > 0, np.exp(-rise * rise), np.exp(-(6 * rise) ** 2)) + 0.6 * np.exp(-(rise / 0.12) ** 2)
    folds = ((0.5 + 0.5 * np.sin(x * 40.0 + 3 * np.sin(x * 7.0 + c.t * 0.2) + c.t * 0.6))
             * (0.6 + 0.4 * np.sin(x * 17.0 - c.t * 0.4 + 2 * np.sin(x * 3.0))))
    k = np.clip(rise, 0, 1)[..., None]
    col = np.asarray(bottom, np.float32) * (1 - k) + np.asarray(top, np.float32) * k
    return ((veil * (0.25 + 0.75 * folds) * amount)[..., None] * col).astype(np.float32)


def sun_halo(A, x, y, power, color):
    """A wide soft halo around a light (low res, upscaled)."""
    d = np.hypot(A.lxx - x * A.lw / A.w, A.lyy - y * A.lh / A.h) * (A.w / A.lw)
    g = np.exp(-d / (0.04 * A.w)) * 0.6 + np.exp(-d / (0.14 * A.w)) * 0.25
    return gfx.resize((g[..., None] * np.asarray(color, np.float32) * power).astype(np.float32), A.w, A.h)


def sun_disc(A, img, x, y, radius, color, power=1.0):
    """A low sun: a hot disc with a soft edge and a wide halo around it."""
    col = np.asarray(color, np.float32)
    img += sun_halo(A, x, y, power, col)
    x0, x1 = int(max(0, x - radius - 3)), int(min(A.w, x + radius + 4))
    y0, y1 = int(max(0, y - radius - 3)), int(min(A.h, y + radius + 4))
    if x1 > x0 and y1 > y0:
        d = np.hypot(A.xx[:, x0:x1] - x, A.yy[y0:y1] - y)
        disc = gfx.smoothstep(radius + 1.0, radius - 1.0, d)
        img[y0:y1, x0:x1] += (disc * 2.0 * power)[..., None] * (0.4 * col + 0.6)


def shimmer(A, img, top, bottom, t, amount):
    """Heat haze: the rows between top and bottom waver sideways, most in the middle of the band."""
    y0, y1 = int(max(0, top)), int(min(A.h, bottom))
    if y1 - y0 < 4 or amount <= 0.01:
        return img
    ys = A.yy[y0:y1]
    k = np.sin(np.pi * (ys - y0) / (y1 - y0))
    off = amount * A.s * k * (1.5 * np.sin(ys * 0.35 + t * 5.0) + 0.8 * np.sin(ys * 0.9 - t * 7.3))
    xs = A.xx + off
    img[y0:y1] = gfx.sample_clamped(img, xs, np.broadcast_to(ys, xs.shape))
    return img


# ------------------------------------------------------------------ the tree
def age_growth(year):
    """How much of the tree has grown (by branch order) and its overall size, for its age in years."""
    a = min(1.0, max(0.0, year) / FULL_GROWN)
    return 0.01 + 0.99 * a ** 0.7, 0.12 + 0.88 * a ** 0.7


def tree_pose(A, base, scale, t, wind, grow=1.0, seg_grow=None, lean=0.0):
    """Segment end points for this frame: the crown sways more than the trunk, and leans with the wind."""
    T = A.tree
    k = T["order"] / 4.0
    sway = wind * ((0.008 + 0.05 * k ** 1.5) * np.sin(1.2 * t + T["phase"]) + 0.012 * np.sin(0.4 * t))
    sway += lean * (0.02 + 0.06 * k)
    L = T["length"] * scale * A.h * np.where(T["terminal"], grow, 1.0)
    if seg_grow is not None:
        L = L * seg_grow
    n = len(L)
    a, x0, y0, x1, y1 = (np.empty(n) for _ in range(5))
    for d, idx in enumerate(T["levels"]):
        if d == 0:
            a[idx] = T["ang"][idx] + sway[idx]
            x0[idx], y0[idx] = base
        else:
            p = T["par"][idx]
            a[idx] = a[p] + T["ang"][idx] + sway[idx]
            x0[idx], y0[idx] = x1[p], y1[p]
        x1[idx] = x0[idx] + L[idx] * np.sin(a[idx])
        y1[idx] = y0[idx] - L[idx] * np.cos(a[idx])
    return x0, y0, x1, y1, a


def glow_tree(A, img, v, c, year, wind, base, tip, leaf_cols, leaves=1.0, lean=0.0, grow=1.0, twinkle=1.0,
              pulse=1.0, inner=0.0, frost=0.0, ss=2):
    """The tree drawn in light, grown to its age and posed in the wind: filaments from a hot trunk (`base`) out
    to tips in the season's colour, white-hot along the thicker limbs; a crown of leaf lights (a share `leaves`
    of them out) that also glows as a whole and breathes with the voice; on every kick a pulse of light
    climbing from the root to the tips. `inner` lights a trembling seam inside the trunk, `frost` glitters
    along the branches. Limbs are drawn at ss times the resolution and scaled down, so they stay smooth.
    Returns the picture and the pose (picture coordinates)."""
    T = A.tree
    g, size = age_growth(year)
    seg_grow = np.clip((g - T["birth"]) / 0.05, 0, 1)
    scale = v.zoom * size
    pose = tree_pose(A, (v.sx(0.0), v.sy(-0.004)), scale, c.t, wind, grow, seg_grow, lean)
    x0, y0, x1, y1, a = pose
    since = c.t - c.last_kick
    climb = pulse * np.exp(-((T["depth"] - since * 16.0) / 1.8) ** 2) * np.exp(-since / 1.4)
    k = (T["order"] / 4.0)[:, None]
    col = np.asarray(base, np.float32) * (1 - k) + np.asarray(tip, np.float32) * k
    col = col * (0.75 + 0.6 * (1 - k)) + climb[:, None] * PALE * 1.6
    trunk = T["order"] == 0
    if inner > 0:  # a seam of light trembling inside the trunk
        flick = 0.65 + 0.35 * np.sin(c.t * 31.0 + T["phase"][trunk] * 5.0) * np.sin(c.t * 17.0)
        col[trunk] += PALE * (inner * 1.6 * flick * (0.8 + 0.4 * c.kick_pulse))[:, None]
    W = T["width"] * scale * A.h * 0.45
    shown = seg_grow > 0
    xs, ys = np.r_[x0[shown], x1[shown]], np.r_[y0[shown], y1[shown]]
    pad = int(24 * A.s) + 4
    bx0, bx1 = int(max(0, xs.min() - pad)), int(min(A.w, xs.max() + pad))
    by0, by1 = int(max(0, ys.min() - pad)), int(min(A.h, ys.max() + pad))
    if bx1 - bx0 > 2 and by1 - by0 > 2:
        im = Image.new("RGB", ((bx1 - bx0) * ss, (by1 - by0) * ss))
        d = ImageDraw.Draw(im)
        X0, Y0, X1, Y1 = (x0 - bx0) * ss, (y0 - by0) * ss, (x1 - bx0) * ss, (y1 - by0) * ss
        CX = (X0 + X1) / 2 - 2 * (Y1 - Y0) * T["bend"]  # each limb bows a little off the straight line
        CY = (Y0 + Y1) / 2 + 2 * (X1 - X0) * T["bend"]
        tones = [np.clip(col * f / 2.0, 0, 1) for f in (0.5, 0.95)]  # stored at half intensity, doubled back
        tones.append(np.clip((0.35 * col + 0.9) / 2.0, 0, 1))  # the white-hot core
        u = np.linspace(0, 1, 7)
        bands = ([], [], [])  # (colour, outline, joint) per tone, drawn tone by tone so the core runs on unbroken
        for i in np.argsort(-T["order"], kind="stable"):  # fine twigs first, the trunk on top
            if not shown[i]:
                continue
            rgb = [tuple(int(255 * q) for q in tone[i]) for tone in tones]
            if W[i] < 3.0:  # a twig: one bowed line
                mid = ((X0[i] + X1[i]) / 4 + CX[i] / 2, (Y0[i] + Y1[i]) / 4 + CY[i] / 2)
                d.line([(X0[i], Y0[i]), mid, (X1[i], Y1[i])], fill=rgb[1], width=max(2, int(round(W[i] * ss))),
                       joint="curve")
                continue
            # a limb: a bowed band narrowing along its length (the trunk flaring into the ground at its foot)
            bx = (1 - u) ** 2 * X0[i] + 2 * u * (1 - u) * CX[i] + u ** 2 * X1[i]
            by = (1 - u) ** 2 * Y0[i] + 2 * u * (1 - u) * CY[i] + u ** 2 * Y1[i]
            ln = max(1e-3, float(np.hypot(X1[i] - X0[i], Y1[i] - Y0[i])))
            nx, ny = -(Y1[i] - Y0[i]) / ln, (X1[i] - X0[i]) / ln
            wu = W[i] * ss * (1 - 0.2 * u) * (1 + (0.7 * np.exp(-u / 0.06) if i == 0 else 0))
            for tone, wf in enumerate((1.0, 0.55, 0.2)):
                half = 0.5 * wf * wu
                left = list(zip((bx + nx * half).tolist(), (by + ny * half).tolist()))
                right = list(zip((bx - nx * half).tolist(), (by - ny * half).tolist()))
                r = float(half[-1])
                bands[tone].append((rgb[tone], left + right[::-1], (X1[i] - r, Y1[i] - r, X1[i] + r, Y1[i] + r)))
        for band in bands:
            for rgb, poly, cap in band:
                d.polygon(poly, fill=rgb)
                d.ellipse(cap, fill=rgb)  # round the joint
        arr = np.asarray(im.resize((bx1 - bx0, by1 - by0), Image.BOX), np.float32) * (2.0 / 255)
        img[by0:by1, bx0:bx1] += arr + gfx.blur(arr, 6 * A.s) * 0.8  # the filaments and their halo
    if leaves > 0:
        keep = T["leaf_keep"] < leaves
        keep[keep] = seg_grow[T["leaf_seg"][keep]] > 0.5
        i = T["leaf_seg"][keep]
        fr, off, ph = T["leaf_frac"][keep], T["leaf_off"][keep] * scale * A.h, T["leaf_phase"][keep]
        ca, sa = np.cos(a[i]), np.sin(a[i])
        lx = x0[i] + (x1[i] - x0[i]) * fr + off[:, 0] * ca - off[:, 1] * sa + wind * 2 * A.s * np.sin(3 * c.t + ph)
        ly = y0[i] + (y1[i] - y0[i]) * fr + off[:, 0] * sa + off[:, 1] * ca
        lc = np.asarray(leaf_cols, np.float32).reshape(-1, 3)[T["leaf_kind"][keep] % len(leaf_cols)]
        tw = 0.8 + 0.3 * twinkle * np.sin(c.t * 2.2 + ph * 3)
        cols = (lc * (tw * (0.9 + 1.2 * climb[i]))[:, None]).astype(np.float32)
        sizes = (1.3 + 1.6 * T["leaf_tone"][keep]) * A.s * (0.7 + 0.3 * min(scale, 2.0))
        gfx.splat_points(img, lx, ly, cols * 1.2, sizes)
        low = np.zeros((A.lh, A.lw, 3), np.float32)  # the crown glows as a whole
        gfx.splat_points(low, lx * (A.lw / A.w), ly * (A.lh / A.h), cols * 0.12, np.full(len(lx), 1.0))
        img += gfx.resize(gfx.blur(low, 2.5), A.w, A.h) * (0.8 + 0.5 * c.vocal)
    if frost > 0:
        i = A.frost_seg
        ok = seg_grow[i] > 0.5
        i, f, ph = i[ok], A.frost_frac[ok], A.frost_phase[ok]
        tw = np.maximum(0.0, np.sin(c.t * 2.5 + ph * 7.0)) ** 12
        col = np.array([0.8, 0.92, 1.0], np.float32)[None] * ((0.12 + 1.8 * tw) * frost)[:, None]
        gfx.splat_points(img, x0[i] + (x1[i] - x0[i]) * f, y0[i] + (y1[i] - y0[i]) * f, col.astype(np.float32),
                         np.where(tw > 0.5, 1.5, 0.8) * A.s)
    return img, pose


LEAF = np.stack([np.cos(np.linspace(0, 2 * np.pi, 10, endpoint=False)),
                 0.36 * np.sin(np.linspace(0, 2 * np.pi, 10, endpoint=False))
                 * np.abs(np.sin(np.linspace(0, 2 * np.pi, 10, endpoint=False))) ** 0.35], -1)  # pointed oval


def sprout(A, img, x, y, height, t, color, ss=2):
    """A shoot just out of the ground drawn in light: a curved stem and its first two leaves, outlined bright."""
    if height < 1.0:
        return
    pad = int(0.6 * height) + 8
    bx0, bx1 = int(max(0, x - pad)), int(min(A.w, x + pad))
    by0, by1 = int(max(0, y - height - pad)), int(min(A.h, y + 4))
    im = Image.new("RGB", ((bx1 - bx0) * ss, (by1 - by0) * ss))
    d = ImageDraw.Draw(im)
    k = np.linspace(0, 1, 12)
    sx = (x + 0.12 * height * np.sin(k * 2.2 + 0.3 * np.sin(t)) - bx0) * ss
    sy = (y - height * k - by0) * ss
    bright = tuple(int(255 * q) for q in np.clip(np.asarray(color) / 2, 0, 1))
    dim = tuple(int(0.3 * q) for q in bright)
    d.line(list(zip(sx.tolist(), sy.tolist())), fill=bright, width=max(2, int(2 * A.s * ss)), joint="curve")
    tx, ty, hs = float(sx[-1]), float(sy[-1]), height * ss
    for cx, cy, size, turn in ((tx - 0.3 * hs, ty, 0.32 * hs, 0.35), (tx + 0.28 * hs, ty - 0.05 * hs, 0.28 * hs, -0.4)):
        turn += 0.05 * np.sin(t)
        ca, sa = np.cos(turn), np.sin(turn)
        poly = [(cx + (px * ca - py * sa) * size, cy + (px * sa + py * ca) * size) for px, py in LEAF]
        d.polygon(poly, fill=dim, outline=bright, width=max(2, int(1.5 * A.s * ss)))
    arr = np.asarray(im.resize((bx1 - bx0, by1 - by0), Image.BOX), np.float32) * (2.0 / 255)
    img[by0:by1, bx0:bx1] += arr + gfx.blur(arr, 5 * A.s)


# ------------------------------------------------------------------ weather and small lights
def fireflies(A, img, c, color, amount):
    """Slow points of light drifting through the air, a few big and out of focus close to the camera."""
    if amount <= 0.01:
        return
    P = A.motes
    x = ((P[:, 0] + c.t * 0.006 * P[:, 2] + 0.02 * np.sin(c.t * 0.7 + P[:, 3] * 9)) % 1.0) * A.w
    y = (P[:, 1] + 0.03 * np.sin(c.t * 0.5 + P[:, 3] * 6)) * A.h
    br = amount * (0.4 + 0.6 * np.sin(c.t * 1.3 + P[:, 3] * 9) ** 2)
    big = P[:, 3] > 0.8
    col = np.asarray(color, np.float32)[None] * (br * np.where(big, 0.12, 0.5))[:, None]
    gfx.splat_points(img, x, y, col.astype(np.float32), np.where(big, 9.0, 1.3) * A.s)


def light_rain(A, img, t, amount, slant, color):
    """Rain as faint streaks of light, slanted by the wind."""
    R = A.rain[: int(len(A.rain) * amount)]
    if not len(R):
        return
    y = ((R[:, 1] + t * R[:, 2]) % 1.15 - 0.08) * A.h
    x = ((R[:, 0] + t * R[:, 2] * slant * 0.5) % 1.05 - 0.02) * A.w
    ln = R[:, 3] * A.h
    layer = Image.new("L", (A.w, A.h))
    d = ImageDraw.Draw(layer)
    for x0, y0, l0, k in zip(x.tolist(), y.tolist(), ln.tolist(), R[:, 0].tolist()):
        d.line([(x0, y0), (x0 - l0 * slant, y0 - l0)], fill=int(40 + 110 * k), width=1)
    img += (np.asarray(layer, np.float32) / 255)[..., None] * np.asarray(color, np.float32)


def drift_lights(A, img, t, colors, amount, wind=0.3, fall=1.0, size=1.0, trail=0):
    """Leaves, petals or embers of light letting go and tumbling down, flickering as they turn; `trail` adds
    fading copies behind each, for the fast ones."""
    F = A.flakes[: int(len(A.flakes) * amount)]
    if not len(F):
        return
    cols = np.asarray(colors, np.float32).reshape(-1, 3)[F[:, 5].astype(np.int32) % len(colors)]
    for j in range(trail + 1):
        tj = t - 0.03 * j
        y = ((F[:, 1] + tj * F[:, 2] * fall) % 1.1 - 0.05) * A.h
        x = ((F[:, 0] + tj * F[:, 2] * wind + 0.03 * np.sin(tj * 1.4 + F[:, 4])) % 1.0) * A.w
        turn = (0.35 + 0.65 * np.abs(np.sin(tj * 3.0 + F[:, 4]))) / (1 + 1.5 * j)
        gfx.splat_points(img, x, y, (cols * turn[:, None]).astype(np.float32),
                         (1.1 + 1.5 * F[:, 3]) * size * A.s * (1 - 0.15 * j))


def snow_lights(A, img, t, amount, wind, color=(0.85, 0.93, 1.0)):
    """Snow as drifting points of light; the nearest flakes big, soft and out of focus."""
    S = A.snow[: int(len(A.snow) * amount)]
    y = ((S[:, 1] + t * S[:, 2]) % 1.05 - 0.02) * A.h
    x = ((S[:, 0] + t * S[:, 2] * wind + 0.015 * np.sin(t * 1.3 + S[:, 4])) % 1.0) * A.w
    k = S[:, 3] / 2.6
    near = k > 0.9
    br = np.where(near, 0.10, 0.25 + 0.45 * k)
    gfx.splat_points(img, x, y, (np.asarray(color, np.float32)[None] * br[:, None]).astype(np.float32),
                     np.where(near, 4.0 * S[:, 3], S[:, 3]) * A.s)


def strike(c, every=3):
    """Some snare hits strike lightning: the flash (0..1) for this frame and which bolt."""
    since = c.t - c.last_snap
    key = int(round(c.last_snap * 10))
    if since > 0.4 or key % every:
        return 0.0, key
    return float(np.exp(-since / 0.09)), key


def bolt(A, img, key, flash, color):
    """A lightning bolt from the clouds, with one fork: a hot core and a glow."""
    if flash <= 0.01:
        return
    main, fork = A.bolts[key % len(A.bolts)]
    layer = Image.new("L", (A.w, A.h))
    d = ImageDraw.Draw(layer)
    for pts, wd in ((main, 3), (fork, 2)):
        d.line([(float(x) * A.w, float(y) * A.h) for x, y in pts], fill=255, width=max(1, int(wd * A.s)))
    m = np.asarray(layer, np.float32)[..., None] / 255
    img += (m * 2.5 + gfx.blur(m, 6 * A.s) * 1.5) * np.asarray(color, np.float32) * flash


def wisp(A, img, x, y, t, power):
    """The small light that keeps the tree company: a soft glow with a few sparks circling it."""
    k = np.arange(5)
    sx = x + np.cos(t * 2.1 + k * 1.26) * 7 * A.s
    sy = y + np.sin(t * 2.7 + k * 1.26) * 5 * A.s
    gfx.splat_points(img, np.r_[x, sx], np.r_[y, sy],
                     (PALE[None] * np.r_[2.4, np.full(5, 0.35)][:, None] * power).astype(np.float32),
                     np.r_[3.2, np.full(5, 1.1)] * A.s)


# ------------------------------------------------------------------ the tally and the rings
def counter(A, img, x, y, year, alpha, size=26):
    """The tally in a corner: years lived over rings laid down."""
    if alpha <= 0.01:
        return
    s = int(size * A.s)
    rings = rings_at(year)
    fnt = font(s)
    w, h = int(330 * A.s * size / 26), int(2.7 * s)
    layer = Image.new("L", (w, h))
    d = ImageDraw.Draw(layer)
    d.text((0, 0), "YEAR", font=fnt, fill=190)
    d.text((int(4.6 * s), 0), f"{int(year)}", font=fnt, fill=255)
    d.text((0, int(1.3 * s)), "RINGS", font=fnt, fill=190)
    d.text((int(4.6 * s), int(1.3 * s)), f"{rings}", font=fnt, fill=255)
    m = np.asarray(layer, np.float32) / 255 * alpha
    x, y = int(x), int(y)
    region = img[y:y + h, x:x + w]
    m = m[: region.shape[0], : region.shape[1], None]
    halo = gfx.blur(m, 3 * A.s)  # a soft shadow first, so the tally reads on bright skies too
    region *= 1 - 0.45 * np.clip(halo * 2, 0, 1)
    region += m * PALE


def ring_disc(A, img, cx, cy, radius, year, fresh=0.0, alpha=1.0, anchor=None):
    """The rings laid down so far as a small glowing cut face floating beside the trunk; the newest one
    flashes as it forms, the false rings shine pale. A thin line ties it to the trunk at `anchor`."""
    if alpha <= 0.01:
        return
    edge = float(np.interp(year, A.ring_t, A.ring_b))
    if edge <= 0.002:
        return
    nb = rings_at(year)
    R = radius
    x0, x1 = int(max(0, cx - 1.2 * R)), int(min(A.w, cx + 1.2 * R))
    y0, y1 = int(max(0, cy - 1.2 * R)), int(min(A.h, cy + 1.2 * R))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, cy - yy
    th = np.arctan2(dy, dx)
    q = np.hypot(dx, dy) / R * edge / np.interp(th, A.th, A.wob, period=2 * np.pi).astype(np.float32)
    u = np.interp(q, A.ring_b, A.ring_i).astype(np.float32)
    near = np.rint(u)
    kb = near.astype(np.int32)
    d_px = np.abs(u - near) * A.ring_gap[kb] * (R / edge)
    lit = np.full(A.n_rings + 1, 0.55, np.float32)
    lit[A.ring_false] = 0.9
    lit[min(nb, A.n_rings)] += 2.0 * fresh
    lit[0] = 0.0
    lit[nb + 1:] = 0.0
    tint = np.where(A.ring_false[:, None], PALE, AMBER).astype(np.float32)
    inside = gfx.smoothstep(edge * 1.01, edge * 0.99, q)
    line = np.exp(-(d_px / (1.1 * A.s)) ** 2) * lit[kb] * (q < edge * 1.01)
    outline = np.exp(-((q / edge - 1.06) * R / (1.2 * A.s)) ** 2) * 0.5
    region = img[y0:y1, x0:x1]
    region *= 1 - (0.55 * alpha * inside)[..., None]
    region += (EARLYWOOD * 0.18 * (inside * alpha)[..., None]
               + line[..., None] * tint[kb] * alpha + outline[..., None] * PALE * alpha)
    if anchor is not None:
        ax, ay = anchor
        k = np.linspace(0, 1, 40)
        ex, ey = cx - R * 1.06 * np.sign(cx - ax) * 0.7, cy + R * 0.75
        gfx.splat_points(img, ax + (ex - ax) * k, ay + (ey - ay) * k,
                         np.tile(PALE * 0.06 * alpha, (40, 1)).astype(np.float32), np.full(40, 0.8 * A.s))


def fresh_ring(year):
    """How brightly the ring being laid down flares: on each new quarter of a year, then settling."""
    return float(np.exp(-((year * 4) % 1.0) * 4))


def trunk_mid(pose):
    """Picture point halfway up the first stretch of the trunk."""
    return pose[2][0], (pose[1][0] + pose[3][0]) / 2


def cross_section(A, zoom, lit, fill, tint, edge=1.0, base=1.0, rot=0.0):
    """The trunk's cut face filling the picture, grown out to `edge` (fraction of the grown trunk):
    rings and false rings, wood rays, radial checks and bark."""
    th, r0 = A.count_th, A.count_r
    if rot:
        th = (th + rot + np.pi) % (2 * np.pi) - np.pi
    q = r0 / (A.log_r * zoom) * edge / np.interp(th, A.th, A.wob, period=2 * np.pi).astype(np.float32)
    u = np.interp(q, A.ring_b, A.ring_i).astype(np.float32)
    k = np.minimum(u.astype(np.int32), A.n_rings - 1)
    late = gfx.smoothstep(0.5, 0.97, u - np.floor(u))
    img = EARLYWOOD + (LATEWOOD - EARLYWOOD) * late[..., None]
    fiber = np.interp(th, A.th, A.fiber, period=2 * np.pi).astype(np.float32)
    crack = np.interp(th, A.th, A.crack, period=2 * np.pi).astype(np.float32) * gfx.smoothstep(0.35, 0.9, q / edge)
    shade = (0.82 + 0.18 * fiber) * (1 - 0.8 * crack) * (0.5 + 0.5 * gfx.smoothstep(0.0, 0.05, q))
    img = img * shade[..., None]
    rough = np.interp(th * 3, A.th, A.bark_tex, period=2 * np.pi).astype(np.float32)
    img = gfx.lerp(img, BARK * (0.6 + 0.6 * rough)[..., None], gfx.smoothstep(edge, edge * 1.006, q)[..., None])
    img = gfx.lerp(img, DARK, gfx.smoothstep(edge * 1.06, edge * 1.075, q)[..., None]) * base
    near = np.rint(u)
    kb = near.astype(np.int32)
    d_px = np.abs(u - near) * A.ring_gap[kb] * (A.log_r * zoom / edge)
    line = np.exp(-(d_px / (1.3 * A.s)) ** 2) * (q < edge * 1.004) * lit[kb]
    img += line[..., None] * tint[kb]
    img += ((fill[k] * 0.10) * (q < edge))[..., None] * tint[k]
    return img


# ------------------------------------------------------------------ scenes
def seed(A, c):
    """Intro: a starry night over an empty hill, faint northern lights; a small light falls in a long arc and
    lands on the crest, rings of light run out across the ground from where it landed, and a shoot of light
    comes up."""
    p = c.p
    v = View(A, 0.0, 0.16 - 0.08 * ease(0, 1, p), 1.0 + 0.6 * ease(0.2, 1.0, p))
    img = glow_sky(A, c, NIGHT, aurora=((0.15, 0.9, 0.7), (0.45, 0.3, 1.0), 0.3))
    land = 0.55
    since = max(0.0, (p - land) * c.dur)
    k = np.arange(RINGS + 1, dtype=np.float32)
    lit = (0.5 * (1 - k / (RINGS + 1)) ** 1.5 * np.clip((since - 0.1 * k) / 0.4, 0, 1) * (p >= land))
    ripples = [(since * 1.1, 2.0 * float(np.exp(-since / 1.6)))] if p >= land else ()
    img, _, _ = glow_hills(A, img, v, (0.35, 0.5, 1.0), (0.45, 0.65, 1.0), fill=(0.01, 0.015, 0.04),
                           lit=lit.astype(np.float32), ripples=ripples, pool=0.25 * ease(0.0, 1.0, since))
    ground_lights(A, img, v, c, [(0.4, 0.6, 1.0), (0.5, 0.9, 1.0), (0.7, 0.6, 1.0)], 0.25 + 0.3 * ease(0, 2, since))
    if p < land:  # the light falls in a long arc
        q = p / land
        x, y = v.sx(-0.9 + 0.9 * q), v.sy(0.55 * (1 - q) ** 1.4)
        trail = np.maximum(q - np.linspace(0, 0.08, 16), 0)
        tx, ty = v.sx(-0.9 + 0.9 * trail), v.sy(0.55 * (1 - trail) ** 1.4)
        gfx.splat_points(img, tx, ty, (PALE[None] * np.linspace(0.8, 0.0, 16)[:, None]).astype(np.float32),
                         np.linspace(2.4, 0.8, 16) * A.s)
        wisp(A, img, x, y, c.t, 1.0)
    else:
        bx, by = v.sx(0.0), v.sy(0.0)
        burst = float(np.exp(-since / 0.4))
        img += sun_halo(A, bx, by, 1.2 * burst, GOLD)
        sprout(A, img, bx, by, 0.05 * v.k * ease(0, 1, (p - land) / (1 - land)), c.t, (0.55, 1.0, 0.6))
        wisp(A, img, bx + 14 * A.s, by - 22 * A.s, c.t, 0.6 * burst + 0.15)
    fireflies(A, img, c, (0.5, 0.8, 1.0), 0.3)
    c.bloom_strength = 1.0
    c.punch = 0.0
    return img


def sapling(A, c):
    """First verse: years flicker past under the northern lights and the young tree grows on its windy hill,
    its tips changing colour with the seasons; then the view sinks into the earth, where the roots walk on
    until they find water."""
    p = c.p
    year = year_of(c, "sapling")
    season = (year % 1.0) * 4.0
    pull = ease(0.0, 0.5, p)
    down = ease(0.52, 0.85, p)
    _, size = age_growth(year)
    v = View(A, 0.0, gfx.lerp(0.12 * size + 0.02, -0.42, down), gfx.lerp(3.0, 1.25, pull))
    gust = 0.5 + 0.5 * np.sin(c.t * 0.9)
    img = glow_sky(A, c, NIGHT, aurora=((0.1, 1.0, 0.6), (0.55, 0.3, 1.0), 0.7))
    img, nm, _ = glow_hills(A, img, v, (0.35, 0.9, 1.0), (0.3, 0.8, 1.0), c.kick_pulse, fill=(0.02, 0.05, 0.08),
                            ripples=bar_ripples(c))
    ground_lights(A, img, v, c, [(0.4, 1.0, 0.8), (0.5, 0.8, 1.0), (1.0, 0.9, 0.6)], 0.5)
    tip = season_mix(season, SEASON_TIPS)
    img, _ = glow_tree(A, img, v, c, year, 0.8 + 0.6 * gust, (1.0, 0.8, 0.55), tip, [tip, PALE],
                       leaves=float(season_mix(season, SEASON_LEAVES)), lean=0.25 * gust)
    if down > 0:  # the earth opens up, the roots in it
        under = glow_earth(A, v, c, year, A.roots["reach"] * ease(0.45, 0.95, p))
        m = (nm * ease(0.0, 0.3, down))[..., None]
        img = img * (1 - m) + under * m
    fireflies(A, img, c, (0.6, 1.0, 0.8), 0.7)
    counter(A, img, A.w - 300 * A.s, 40 * A.s, year, ease(0.08, 0.2, p))
    c.bloom_strength = 1.15
    c.streak = 0.15
    c.punch = 0.006
    return img


def lull(A, c):
    """Pre-chorus: a summer dusk; clouds pile up over the setting sun and flicker far off, and not a leaf moves,
    but inside the trunk a seam of light trembles and grows."""
    p = c.p
    year = year_of(c, "lull")
    v = View(A, 0.02, 0.20, 1.35 + 0.25 * ease(0, 1, p))
    flash, key = strike(c, every=2)
    far = (0.15 + 0.12 * (key % 4), 0.22, 0.5 * flash, (0.85, 0.6, 1.0))  # far off, the storm is coming
    img = glow_sky(A, c, DUSK, stars=0.2, clouds=0.35 + 0.45 * p, cloud_body=(0.06, 0.02, 0.08),
                   cloud_rim=(0.9, 0.35, 0.3), cloud_drift=4.0, light=far)
    sun_disc(A, img, 0.66 * A.w, 0.385 * A.h, 34 * A.s, (1.0, 0.5, 0.25), 0.9 * (1 - 0.5 * p))  # setting
    img, _, _ = glow_hills(A, img, v, (1.0, 0.45, 0.35), (1.0, 0.5, 0.3), fill=(0.06, 0.02, 0.05))
    leaf = [(0.55, 1.0, 0.45), (1.0, 0.85, 0.4)]
    img, _ = glow_tree(A, img, v, c, year, 0.0, (1.0, 0.75, 0.5), leaf[0], leaf, twinkle=0.0, pulse=0.0,
                       inner=ease(0.0, 1.0, p))
    counter(A, img, A.w - 300 * A.s, 40 * A.s, year, 1.0)
    c.bloom_strength = 1.05
    c.punch = 0.0  # the leaves must not give it away
    return img


def storm(A, c):
    """Chorus: night and a summer storm; rain, gusts, lightning on the snare lighting the clouds from inside.
    The tree holds, and beside it the rings it lays down light up one by one."""
    p = c.p
    year = year_of(c, "storm")
    gust = 0.5 + 0.5 * np.sin(c.t * 1.7) * np.sin(c.t * 0.6)
    shake = 0.004 * gust * np.sin(c.t * 23)
    pull = ease(0.0, 0.2, p)  # out from the still tree as the storm breaks
    v = View(A, gfx.lerp(0.02, -0.05, pull) + shake, gfx.lerp(0.20, 0.22, pull),
             gfx.lerp(1.6, 1.15, pull) + 0.05 * ease(0.2, 1, p))
    flash, key = strike(c)
    top = float(A.bolts[key % len(A.bolts)][0][0, 0])
    img = glow_sky(A, c, STORM, stars=0.0, haze=(0.30, 0.06, 0.40), haze_amount=0.6, drift=30.0, clouds=0.5,
                   cloud_body=(0.06, 0.01, 0.10), cloud_rim=(0.42, 0.12, 0.54), cloud_drift=40.0,
                   light=(top, 0.1, 1.5 * flash, (0.75, 0.6, 1.0)))
    bolt(A, img, key, flash, (0.85, 0.8, 1.0))
    img += flash * 0.12 * np.array([0.5, 0.3, 0.8], np.float32)
    img, _, _ = glow_hills(A, img, v, (0.9, 0.4, 1.0), (0.8, 0.35, 1.0), c.kick_pulse + flash,
                           fill=(0.05, 0.01, 0.07), ripples=bar_ripples(c))
    leaf = [(0.3, 1.0, 0.85), (0.8, 0.5, 1.0)]
    img, pose = glow_tree(A, img, v, c, year, 1.6 + 1.2 * gust, (1.0, 0.8, 0.6), leaf[0], leaf, leaves=0.85,
                          lean=0.6 * gust)
    light_rain(A, img, c.t, 0.7, 0.35 + 0.25 * gust, np.array([0.5, 0.55, 1.0], np.float32) * (0.8 + 0.8 * flash))
    drift_lights(A, img, c.t * 1.6, leaf[:1], 0.4, wind=1.6, fall=0.8, trail=3)  # leaves torn off
    ring_disc(A, img, A.w * 0.83, A.h * 0.40, 0.15 * A.h, year, fresh_ring(year) * (0.5 + 0.5 * c.kick_pulse), 1.0,
              trunk_mid(pose))
    counter(A, img, A.w * 0.83 - 60 * A.s, A.h * 0.40 + 0.17 * A.h, year, 1.0)
    c.bloom_strength = 1.2
    c.streak = 0.25 + 0.4 * flash
    c.punch = 0.012
    return img


def drought(A, c):
    """Second verse: a burning summer with no rain; the earth splits into glowing cracks, the air shimmers and
    the leaves dim and fall. Growth stops; then a little rain comes, green comes back to the tips, and a second
    ring forms in the same year."""
    p = c.p
    year = year_of(c, "drought")
    shower = ease(0.72, 0.8, p) * (1 - ease(0.92, 1.0, p))
    revive = ease(0.78, 0.95, p)
    dry = 1.0 - revive
    v = View(A, gfx.lerp(-0.05, 0.05, ease(0, 1, p)), gfx.lerp(0.22, 0.20, ease(0, 0.3, p)),
             1.2 - 0.2 * ease(0, 0.3, p) + 0.1 * ease(0.3, 1, p))
    sun = (0.24 * A.w, 0.27 * A.h)
    img = glow_sky(A, c, HOT, stars=0.0, haze=(0.5, 0.22, 0.06), haze_amount=0.35, drift=3.0, clouds=0.5 * shower,
                   cloud_body=(0.10, 0.06, 0.08), cloud_rim=(0.5, 0.35, 0.18), cloud_drift=12.0)
    sun_disc(A, img, sun[0], sun[1], 40 * A.s, (1.0, 0.8, 0.45), 1.3 * (1 - 0.6 * shower))
    img, nm, gy = glow_hills(A, img, v, (1.0, 0.72, 0.3), (1.0, 0.6, 0.2), c.kick_pulse, fill=(0.07, 0.03, 0.02),
                             glow=0.2)
    heat = (0.8 + 0.7 * c.kick_pulse) * (1 - 0.8 * shower) * (0.4 + 0.6 * dry)
    img += (cracks(A, v, gy, nm) * heat)[..., None] * np.array([1.0, 0.55, 0.15], np.float32)
    img = shimmer(A, img, gy.min() - 0.18 * A.h, gy.min() + 2, c.t, 1.5 * dry)
    img += gfx.god_rays(img, sun[0], sun[1], threshold=0.95, strength=0.3)
    gfx.lens_flare(img, sun[0], sun[1], (1.0 + 0.6 * c.kick_pulse) * (1 - shower))
    tip = gfx.lerp(np.array([0.85, 0.65, 0.25], np.float32), np.array([0.4, 1.0, 0.5], np.float32), revive)
    img, pose = glow_tree(A, img, v, c, year, 0.3, (1.0, 0.75, 0.4), tip, [tip, (1.0, 0.8, 0.35)],
                          leaves=0.35 + 0.4 * revive)
    drift_lights(A, img, c.t, [(1.0, 0.75, 0.3)], 0.3 * dry, wind=0.25)
    light_rain(A, img, c.t, 0.5 * shower, 0.1, (0.7, 0.8, 1.0))
    fireflies(A, img, c, (1.0, 0.75, 0.35), 0.6 * dry)  # dust in the light
    ring_disc(A, img, A.w * 0.83, A.h * 0.40, 0.14 * A.h, year, fresh_ring(year) * 0.6, ease(0.0, 0.1, p),
              trunk_mid(pose))
    counter(A, img, A.w * 0.83 - 60 * A.s, A.h * 0.40 + 0.16 * A.h, year, 1.0)
    c.bloom_strength = 1.0
    c.punch = 0.005
    return img


def autumn(A, c):
    """Second pre-chorus: a red-gold evening; the leaves burn orange and let go, drifting down like embers."""
    p = c.p
    year = year_of(c, "autumn")
    turn = ease(0, 1, p)
    v = View(A, gfx.lerp(0.05, -0.1, turn), gfx.lerp(0.20, 0.22, turn), gfx.lerp(1.1, 1.3, turn))
    sun = (0.18 * A.w, 0.40 * A.h)
    img = glow_sky(A, c, EMBER, stars=0.1, haze=(0.45, 0.12, 0.05), haze_amount=0.3)
    sun_disc(A, img, sun[0], sun[1], 30 * A.s, (1.0, 0.55, 0.2), 1.0)
    img, _, _ = glow_hills(A, img, v, (1.0, 0.55, 0.2), (1.0, 0.45, 0.15), fill=(0.07, 0.02, 0.02))
    img += gfx.god_rays(img, sun[0], sun[1], threshold=0.9, strength=0.35)
    leaf = [(1.0, 0.45, 0.1), (1.0, 0.22, 0.08), (1.0, 0.75, 0.2)]
    img, _ = glow_tree(A, img, v, c, year, 0.5, (1.0, 0.75, 0.45), (1.0, 0.5, 0.15), leaf, leaves=1.0 - 0.6 * p)
    drift_lights(A, img, c.t, leaf, 0.4 + 0.6 * p, wind=0.35, trail=2)
    fireflies(A, img, c, (1.0, 0.6, 0.25), 0.5)
    counter(A, img, A.w - 300 * A.s, 40 * A.s, year, 1.0)
    c.bloom_strength = 1.05
    c.punch = 0.0
    return img


def winter(A, c):
    """Second chorus: a snowstorm in blue light; the bare tree stands in it, frost glittering along its branches,
    and the rings keep coming."""
    p = c.p
    year = year_of(c, "winter")
    gust = 0.5 + 0.5 * np.sin(c.t * 1.3)
    settle = ease(0.0, 0.3, p)
    v = View(A, gfx.lerp(-0.1, 0.0, settle) + 0.04 * settle * np.sin(c.t * 0.3), gfx.lerp(0.22, 0.21, settle),
             gfx.lerp(1.3, 1.1, settle) + 0.06 * ease(0.3, 1, p))
    img = glow_sky(A, c, FROST, stars=0.3, haze=(0.25, 0.4, 0.6), haze_amount=0.45, drift=25.0)
    img, _, _ = glow_hills(A, img, v, (0.55, 0.85, 1.0), (0.5, 0.8, 1.0), c.kick_pulse, fill=(0.06, 0.10, 0.16),
                           ground=0.6, ripples=bar_ripples(c))
    img, pose = glow_tree(A, img, v, c, year, 0.9 + 0.8 * gust, (0.8, 0.92, 1.0), (0.45, 0.75, 1.0),
                          [(0.7, 0.9, 1.0)], leaves=0.05, lean=0.4 * gust, frost=1.0)
    snow_lights(A, img, c.t, 1.0, 0.6 + 0.8 * gust)
    ring_disc(A, img, A.w * 0.83, A.h * 0.40, 0.15 * A.h, year, fresh_ring(year) * (0.5 + 0.5 * c.kick_pulse), 1.0,
              trunk_mid(pose))
    counter(A, img, A.w * 0.83 - 60 * A.s, A.h * 0.40 + 0.17 * A.h, year, 1.0)
    c.bloom_strength = 1.1
    c.streak = 0.1
    c.punch = 0.012
    return img


def counting(A, c):
    """Bridge: a still winter night, close in on the heart of the trunk; the small light comes back and, with
    another beside it, goes round the rings from the pith outward, counting, and every ring it has counted stays
    lit, the false ones brightest."""
    year = YEARS["counting"][1]
    edge = float(np.interp(year, A.ring_t, A.ring_b))
    nb = rings_at(year)
    k = np.arange(A.n_rings + 1)
    reach = 1.0 + (nb - 1) * ease(0.04, 0.92, c.p)  # rings counted so far
    lit = (0.10 + 0.75 * np.clip(reach - k, 0, 1) * (k <= nb)).astype(np.float32)
    lit[A.ring_false & (k <= reach)] += 0.6
    lit[0] = 0.0
    tint = np.where(A.ring_false[:, None], PALE, AMBER).astype(np.float32)
    zoom = 1.05 + 0.15 * ease(0, 1, c.p)
    img = cross_section(A, zoom, lit, lit * 0.5, tint, edge=edge, base=0.4, rot=0.01 * c.t)
    cx, cy = A.count_c
    i = int(min(nb, max(1, np.floor(reach))))
    for off in (0.0, 0.9):  # two lights, one a little ahead
        phi = c.t * 1.8 + off
        r = A.ring_b[i] / edge * np.interp(phi, A.th, A.wob, period=2 * np.pi) * A.log_r * zoom
        tr = phi - np.linspace(0, 0.8, 28)
        rr = A.ring_b[i] / edge * np.interp(tr, A.th, A.wob, period=2 * np.pi) * A.log_r * zoom
        fade = np.linspace(1, 0, 28) ** 2
        gfx.splat_points(img, cx + rr * np.cos(tr), cy - rr * np.sin(tr),
                         (PALE[None] * (2.2 * fade)[:, None]).astype(np.float32), (1.4 + 2.4 * fade) * A.s)
        wisp(A, img, cx + r * np.cos(phi), cy - r * np.sin(phi), c.t + off, 0.8)
    counter(A, img, A.w - 300 * A.s, 40 * A.s, year, 1.0)
    snow_lights(A, img, c.t, 0.12, 0.1)
    c.bloom_strength = 1.0
    c.punch = 0.0
    return img


def spring(A, c):
    """Final chorus: spring comes back all at once; under a rose sky the tree flowers in light, petals blow
    across, flowers light up along the hill, the camera swings round, and the tally shows more rings than
    years."""
    p = c.p
    year = year_of(c, "spring")
    v = View(A, 0.25 - 0.2 * ease(0, 1, p), 0.24 - 0.04 * ease(0, 1, p), 1.75 - 0.75 * ease(0, 1, p))
    sun = (0.13 * A.w, 0.33 * A.h)
    img = glow_sky(A, c, ROSE, stars=0.15, haze=(0.5, 0.25, 0.4), haze_amount=0.25)
    sun_disc(A, img, sun[0], sun[1], 26 * A.s, (1.0, 0.8, 0.6), 1.3)
    img, _, _ = glow_hills(A, img, v, (1.0, 0.75, 0.85), (1.0, 0.6, 0.8), c.kick_pulse, fill=(0.22, 0.10, 0.20),
                           ripples=bar_ripples(c))
    img += gfx.god_rays(img, sun[0], sun[1], threshold=0.9, strength=0.55)
    ground_lights(A, img, v, c, [(1.0, 0.7, 0.85), (1.0, 0.95, 0.7), (0.8, 0.75, 1.0)], 0.9)
    leaf = [(1.0, 0.65, 0.85), (1.0, 0.92, 0.95), (0.55, 1.0, 0.65)]
    img, pose = glow_tree(A, img, v, c, year, 0.7, (1.0, 0.85, 0.6), (1.0, 0.6, 0.8), leaf, leaves=1.0)
    drift_lights(A, img, c.t, [(1.0, 0.75, 0.88), (1.0, 0.9, 0.95)], 1.0, wind=0.9, size=0.9)
    fireflies(A, img, c, (1.0, 0.85, 0.9), 0.6)
    ring_disc(A, img, A.w * 0.83, A.h * 0.40, 0.15 * A.h, year, fresh_ring(year) * 0.6 + 0.4 * c.snap_pulse, 1.0,
              trunk_mid(pose))
    counter(A, img, A.w * 0.83 - 80 * A.s, A.h * 0.40 + 0.17 * A.h, year, 1.0, size=34)
    c.bloom_strength = 1.2
    c.streak = 0.2
    c.punch = 0.012
    return img


def dawn(A, c):
    """Outro: sunrise on the hill; the small light comes back and settles on the highest twig, a new leaf of
    light opens there, and the tree is still growing."""
    p = c.p
    year = year_of(c, "dawn")
    rise = ease(0, 1, p)
    v = View(A, 0.0, 0.2, 1.0 - 0.05 * rise)
    sun = (0.71 * A.w, A.h * (0.40 - 0.14 * rise))  # rising from behind the far range
    img = glow_sky(A, c, SUNRISE, stars=0.4 * (1 - rise), haze=(0.6, 0.3, 0.25), haze_amount=0.2)
    sun_disc(A, img, sun[0], sun[1], 30 * A.s, (1.0, 0.75, 0.45), 0.6 + 0.8 * rise)
    img, _, _ = glow_hills(A, img, v, (1.0, 0.7, 0.45), (1.0, 0.75, 0.4), 0.5 * c.kick_pulse, fill=(0.10, 0.05, 0.08),
                           ripples=bar_ripples(c, speed=0.22, power=1.0))
    img += gfx.god_rays(img, sun[0], sun[1], threshold=0.9, strength=0.45)
    ground_lights(A, img, v, c, [(1.0, 0.85, 0.5), (0.6, 1.0, 0.6), (1.0, 0.7, 0.8)], 0.5)
    leaf = [(0.5, 1.0, 0.55), (1.0, 0.9, 0.6)]
    img, pose = glow_tree(A, img, v, c, year, 0.5, (1.0, 0.8, 0.5), leaf[0], leaf, leaves=0.8,
                          grow=1.0 + 0.08 * ease(0.2, 0.9, p))
    term = np.where(A.tree["terminal"])[0]
    tip = term[np.argmin(pose[3][term])]  # the highest twig
    tx, ty = pose[2][tip], pose[3][tip]
    k = ease(0.0, 0.45, p)
    wisp(A, img, gfx.lerp(-0.1 * A.w, tx, k) + (1 - k) * 60 * A.s * np.sin(c.t * 2),
         gfx.lerp(0.2 * A.h, ty - 6 * A.s, k), c.t, 0.9)
    bud = ease(0.5, 0.8, p)
    gfx.splat_points(img, np.array([tx, tx]), np.array([ty, ty]),
                     (np.array([[0.9, 1.8, 1.0], [0.12, 0.25, 0.14]]) * bud).astype(np.float32),
                     np.array([2.5, 10.0]) * A.s)
    fireflies(A, img, c, GOLD, 0.4)
    counter(A, img, A.w - 300 * A.s, 40 * A.s, year, 1.0 - ease(0.6, 0.85, p))
    c.bloom_strength = 1.05
    c.punch = 0.002
    return img


SCENES = {"seed": seed, "sapling": sapling, "lull": lull, "storm": storm, "drought": drought, "autumn": autumn,
          "winter": winter, "counting": counting, "spring": spring, "dawn": dawn}
FLASH_INTO = {"storm", "spring"}  # scenes that open with a burst of light
