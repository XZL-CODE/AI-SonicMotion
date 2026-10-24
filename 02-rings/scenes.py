"""Scenes of one tree's life, from a seed falling at night to a spring morning. Each scene function draws the
picture for one moment.

Scene functions take (A, c): A holds the shared assets built once per worker process,
c is the frame context (times, section progress, audio pulses; see FrameCtx in
engine/video/render.py, which loads this file and puts engine/video on the path for gfx).
They return a float32 RGB image of the picture area (letterbox bars are added later).

The world is a hillside seen through a camera (View): units are picture heights, the foot of the tree is the
origin and y points up. The story keeps count: each scene covers some years of the tree's life (YEARS), and in
the years that grew twice (FALSE_YEARS) a second ring is laid down, so the rings outrun the years.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gfx

FONT = Path(__file__).resolve().parents[1] / ".tools" / "fonts" / "CormorantGaramond[wght].ttf"

# skies, top of the picture to the bottom (display-space RGB 0..1)
NIGHT = [(0.0, (0.008, 0.012, 0.035)), (0.55, (0.025, 0.035, 0.08)), (0.8, (0.06, 0.06, 0.11)),
         (1.0, (0.03, 0.03, 0.05))]
DAY = [(0.0, (0.26, 0.47, 0.80)), (0.5, (0.55, 0.72, 0.90)), (0.78, (0.86, 0.88, 0.86)), (1.0, (0.72, 0.70, 0.64))]
DUSK = [(0.0, (0.10, 0.10, 0.24)), (0.45, (0.50, 0.30, 0.40)), (0.72, (1.0, 0.55, 0.30)), (1.0, (0.45, 0.25, 0.20))]
STORM = [(0.0, (0.020, 0.026, 0.040)), (0.6, (0.06, 0.07, 0.095)), (1.0, (0.04, 0.04, 0.055))]
HOT = [(0.0, (0.66, 0.71, 0.76)), (0.5, (0.96, 0.92, 0.82)), (1.0, (1.0, 0.88, 0.66))]
GOLDEN = [(0.0, (0.32, 0.38, 0.60)), (0.45, (0.90, 0.68, 0.45)), (0.75, (1.0, 0.76, 0.46)), (1.0, (0.55, 0.35, 0.22))]
OVERCAST = [(0.0, (0.36, 0.40, 0.48)), (0.6, (0.60, 0.64, 0.70)), (1.0, (0.74, 0.77, 0.82))]
MORNING = [(0.0, (0.36, 0.58, 0.88)), (0.45, (0.76, 0.82, 0.92)), (0.72, (1.0, 0.88, 0.82)), (1.0, (0.86, 0.80, 0.74))]
SUNRISE = [(0.0, (0.11, 0.17, 0.30)), (0.40, (0.36, 0.38, 0.46)), (0.66, (0.92, 0.62, 0.40)),
           (0.76, (1.0, 0.80, 0.54)), (1.0, (0.62, 0.42, 0.30))]

AMBER = np.array([1.0, 0.62, 0.28], np.float32)
GOLD = np.array([1.0, 0.82, 0.46], np.float32)
PALE = np.array([1.0, 0.93, 0.80], np.float32)
WATER = np.array([0.35, 0.75, 1.0], np.float32)
BARK = np.array([0.15, 0.10, 0.075], np.float32)
EARLYWOOD = np.array([0.40, 0.26, 0.15], np.float32)
LATEWOOD = np.array([0.17, 0.095, 0.05], np.float32)
DARK = np.array([0.020, 0.017, 0.015], np.float32)
SPRING_LEAF = np.array([0.52, 0.78, 0.30], np.float32)
SUMMER_LEAF = np.array([0.20, 0.44, 0.17], np.float32)
DRY_LEAF = np.array([0.66, 0.58, 0.26], np.float32)
AUTUMN_LEAF = np.array([[0.96, 0.62, 0.16], [0.88, 0.32, 0.12], [0.80, 0.50, 0.14]], np.float32)
BLOSSOM = np.array([1.0, 0.80, 0.88], np.float32)

FALSE_YEARS = (5, 11, 14, 16, 19, 21, 23)  # the hard years, when growth stopped and started again
YEARS = {"seed": (0.0, 0.0), "sapling": (0.0, 9.0), "lull": (9.0, 9.0), "storm": (9.0, 13.0),
         "drought": (13.0, 17.0), "autumn": (17.0, 18.0), "winter": (18.0, 22.0), "counting": (22.0, 22.0),
         "spring": (22.0, 24.0), "dawn": (24.0, 25.0)}
FULL_GROWN = 22.0


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
        self.lw, self.lh = w // 4, h // 4  # low-res buffer for skies, clouds and soil
        self.lyy, self.lxx = np.mgrid[0:self.lh, 0:self.lw].astype(np.float32)
        self.yy = np.arange(h, dtype=np.float32)[:, None]
        rng = np.random.default_rng(3)

        self.gx = np.linspace(-8, 8, 4096).astype(np.float32)
        self.gn = noise_1d(4096, 1.6, 41)
        self.fn = noise_1d(4096, 1.9, 42)
        self.cloud = gfx.fbm_texture(512, 1.35, 13)
        self.soil = gfx.fbm_texture(256, 1.2, 8)
        cracks = gfx.fbm_texture(256, 1.0, 9)
        self.cracks = gfx.smoothstep(0.10, 0.0, np.abs(cracks)).astype(np.float32)
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

        self.tree = self._tree(rng)
        self.roots = self._roots(rng)
        self._rings(rng)
        self.vig = gfx.vignette(h, w, 0.36)
        self.grain = [rng.normal(0, 1, (h // 2, w // 2, 1)).astype(np.float32) for _ in range(6)]

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
        """A lone tree as branch segments (parent, angle, length, width), lengths in picture heights."""
        par, ang, length, width, depth = [], [], [], [], []

        def grow(parent, a, ln, wd, d):
            i = len(par)
            par.append(parent)
            ang.append(a)
            length.append(ln)
            width.append(wd)
            depth.append(d)
            if d == 7:
                return
            kids = 2 if d < 2 else (3 if rng.uniform() < 0.3 else 2)
            for k in range(kids):
                spread = rng.uniform(0.30, 0.62) * (1.25 if d < 2 else 1.0)
                grow(i, (-1) ** k * spread + rng.normal(0, 0.1), ln * rng.uniform(0.70, 0.83), wd * 0.66, d + 1)

        grow(-1, 0.05, 0.16, 0.040, 0)  # leaning a little, the way the wind taught it
        par, depth = np.array(par), np.array(depth)
        n = len(par)
        terminal = np.ones(n, bool)
        terminal[par[par >= 0]] = False
        birth = np.empty(n, np.float32)  # growth order: trunk first, twigs last
        birth[np.lexsort((np.arange(n), depth))] = np.linspace(0, 1, n)
        leaf_seg, leaf_off, leaf_size = [], [], []
        for i in np.where(depth >= 5)[0]:
            k = 4 if depth[i] == 7 else 2
            leaf_seg += [i] * k
            leaf_off.append(rng.normal(0, 0.012, (k, 2)))
            leaf_size.append(rng.uniform(0.0035, 0.0075, k))
        m = len(leaf_seg)
        return {"par": par, "ang": np.array(ang), "length": np.array(length), "width": np.array(width),
                "depth": depth, "terminal": terminal, "birth": birth, "phase": rng.uniform(0, 2 * np.pi, n),
                "levels": [np.where(depth == d)[0] for d in range(depth.max() + 1)],
                "leaf_seg": np.array(leaf_seg), "leaf_off": np.concatenate(leaf_off),
                "leaf_size": np.concatenate(leaf_size), "leaf_phase": rng.uniform(0, 2 * np.pi, m),
                "leaf_keep": rng.uniform(0, 1, m), "leaf_tone": rng.uniform(0, 1, m),
                "leaf_kind": rng.integers(0, 3, m), "blossom": rng.uniform(0, 1, m) < 0.45}

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
        self.A, self.cx, self.cy, self.zoom, self.k = A, cx, cy, zoom, zoom * A.h

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
    x = v.wx(np.arange(A.w, dtype=np.float32))
    gy = v.sy(height(A, x)).astype(np.float32)
    return gfx.smoothstep(-1.0, 1.0, A.yy - gy[None]).astype(np.float32), gy


def sky(A, c, stops, sun=None, sun_power=0.0, clouds=0.0, cloud_col=(0.55, 0.55, 0.6), drift=6.0,
        stars=0.0):
    """Sky gradient with an optional sun, a cloud layer of the given cover and stars at night."""
    low = np.repeat(rows(stops, A.lh)[:, None], A.lw, 1)
    if sun is not None and sun_power > 0:
        d = np.hypot(A.lxx - sun[0] * A.lw / A.w, A.lyy - sun[1] * A.lh / A.h) * (A.w / A.lw)
        g = np.exp(-d / (0.03 * A.w)) * 0.9 + np.exp(-d / (0.16 * A.w)) * 0.35
        low += g[..., None] * np.array([1.0, 0.75, 0.46], np.float32) * sun_power
    if clouds > 0:
        u = (A.lxx + c.t * drift) * (512 / A.lw) * 0.7
        v = A.lyy * (512 / A.lh) * 0.45
        dens = gfx.sample(A.cloud, u, v)
        cover = gfx.smoothstep(1.2 - 2.0 * clouds, 1.9 - 2.0 * clouds, dens) * gfx.smoothstep(0.95, 0.4, A.lyy / A.lh)
        shade = 0.75 + 0.25 * gfx.smoothstep(-1, 1.5, gfx.sample(A.cloud, u + 9, v + 7))
        low = gfx.lerp(low, np.asarray(cloud_col, np.float32) * shade[..., None], cover[..., None])
    img = gfx.resize(low.astype(np.float32), A.w, A.h)
    if stars > 0:
        S = A.stars
        tw = 0.7 + 0.3 * np.sin(S[:, 3] + c.t * (1.5 + 2 * S[:, 2]))
        col = (np.array([0.85, 0.9, 1.0], np.float32)[None] * (S[:, 2] * tw * stars)[:, None]).astype(np.float32)
        gfx.splat_points(img, S[:, 0] * A.w, S[:, 1] * A.h, col * 1.4, (0.55 + 1.0 * S[:, 2] ** 3) * A.s)
    return img


def soil(A, v, dry=0.0):
    """Earth seen through the camera (low res, upscaled), drying into cracks near the surface."""
    sx, sy = A.lxx * (A.w / A.lw), A.lyy * (A.h / A.lh)
    wx, wy = v.wx(sx), v.wy(sy)
    depth = np.clip((ground_h(A, wx) - wy) / 0.9, 0, 1)
    tex = gfx.sample(A.soil, wx * 160, wy * 160)
    img = gfx.lerp(np.array([0.20, 0.14, 0.09], np.float32), np.array([0.05, 0.038, 0.032], np.float32),
                   depth[..., None]) * (0.8 + 0.12 * tex)[..., None]
    if dry > 0:
        cr = gfx.sample(A.cracks, wx * 90, wy * 90) * gfx.smoothstep(0.06, 0.0, depth * 0.9)
        img *= (1 - 0.7 * dry * cr)[..., None]
    return gfx.resize(img.astype(np.float32), A.w, A.h)


# ------------------------------------------------------------------ the tree
def age_growth(year):
    """How much of the tree has grown (by branch order) and its overall size, for its age in years."""
    a = min(1.0, max(0.0, year) / FULL_GROWN)
    return 0.01 + 0.99 * a ** 0.7, 0.12 + 0.88 * a ** 0.7


def tree_pose(A, base, scale, t, wind, grow=1.0, seg_grow=None, lean=0.0):
    """Segment end points for this frame: the crown sways more than the trunk, and leans with the wind."""
    T = A.tree
    k = T["depth"] / 7.0
    sway = wind * ((0.010 + 0.09 * k ** 1.7) * np.sin(1.2 * t + T["phase"]) + 0.02 * np.sin(0.4 * t))
    sway += lean * (0.04 + 0.10 * k)
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


LEAF = np.stack([np.cos(np.linspace(0, 2 * np.pi, 10, endpoint=False)),
                 0.36 * np.sin(np.linspace(0, 2 * np.pi, 10, endpoint=False))
                 * np.abs(np.sin(np.linspace(0, 2 * np.pi, 10, endpoint=False))) ** 0.35], -1)  # pointed oval


def draw_leaves(draw, x, y, size, angle, f=1.0, fills=None):
    """Leaves as pointed ovals: centres x, y, half-length size (pixels), orientation angle, one fill each."""
    ca, sa = np.cos(angle)[:, None], np.sin(angle)[:, None]
    lx = (x[:, None] + (LEAF[None, :, 0] * ca - LEAF[None, :, 1] * sa) * size[:, None]) * f
    ly = (y[:, None] + (LEAF[None, :, 0] * sa + LEAF[None, :, 1] * ca) * size[:, None]) * f
    for k, poly in enumerate(np.stack([lx, ly], -1).tolist()):
        draw.polygon([tuple(v) for v in poly], fill=255 if fills is None else fills[k])


def foliage(A, season, dry=0.0):
    """Which leaves are out and their colours for a point in the year: 0 spring, 1 summer, 2 autumn, 3 winter
    (wrapping), dry 0..1 for a drought."""
    T = A.tree
    s = season % 4.0
    tone = T["leaf_tone"][:, None]
    if s < 1:
        keep = 0.25 + 0.75 * s
        col = np.where(T["blossom"][:, None] & (s < 0.7), BLOSSOM, gfx.lerp(SPRING_LEAF, SUMMER_LEAF, s))
    elif s < 2:
        keep, col = 1.0, np.repeat(SUMMER_LEAF[None], len(tone), 0)
    elif s < 3:
        keep = 1.0 - 0.75 * (s - 2)
        col = gfx.lerp(SUMMER_LEAF, AUTUMN_LEAF[T["leaf_kind"]], min(1.0, (s - 2) * 1.6))
    else:
        keep, col = 0.06, AUTUMN_LEAF[T["leaf_kind"]] * 0.6
    col = gfx.lerp(col, DRY_LEAF, dry) * (0.8 + 0.4 * tone)
    return keep * (1 - 0.45 * dry), col.astype(np.float32)


def draw_tree(A, pose, scale, t, flutter, bark, foliage_state=None, light=1.0, res=2):
    """The tree in colour: an image and its coverage at picture size (drawn at 1/res)."""
    x0, y0, x1, y1, a = pose
    f = 1.0 / res
    im = Image.new("RGBA", (A.w // res, A.h // res))
    d = ImageDraw.Draw(im)
    bc = tuple(int(255 * v) for v in np.clip(np.asarray(bark) * light, 0, 1)) + (255,)
    widths = A.tree["width"] * scale * A.h * f
    for i in range(len(x0)):
        wd = max(1, int(round(widths[i])))
        d.line([(x0[i] * f, y0[i] * f), (x1[i] * f, y1[i] * f)], fill=bc, width=wd)
        if wd > 3:  # round the joints of the thick limbs
            r = wd / 2
            d.ellipse([x1[i] * f - r, y1[i] * f - r, x1[i] * f + r, y1[i] * f + r], fill=bc)
    if foliage_state is not None:
        keep, col = foliage_state
        T = A.tree
        sel = T["leaf_keep"] < keep
        if sel.any():
            i = T["leaf_seg"][sel]
            off = T["leaf_off"][sel] * scale * A.h
            ca, sa = np.cos(a[i]), np.sin(a[i])
            ph = T["leaf_phase"][sel]
            lx = x1[i] + off[:, 0] * ca - off[:, 1] * sa + flutter * 2.0 * A.s * np.sin(3.0 * t + ph)
            ly = y1[i] + off[:, 0] * sa + off[:, 1] * ca
            turn = a[i] - np.pi / 2 + ph + 0.25 * flutter * np.sin(2.3 * t + ph)
            size = np.maximum(T["leaf_size"][sel] * scale * A.h * 1.4, 1.2 * res)
            fills = [tuple(int(255 * v) for v in np.clip(cc * light, 0, 1)) + (255,) for cc in col[sel]]
            draw_leaves(d, lx, ly, size, turn, f, fills)
    arr = gfx.resize(np.asarray(im, np.float32) / 255, A.w, A.h)
    return arr[..., :3], arr[..., 3]


def tree_in(A, img, v, year, t, wind, season, dry=0.0, light=1.0, lean=0.0, rim_col=None, snow=0.0, grow=1.0):
    """Grow the tree to its age, pose it in the wind, dress it for the season and set it into the picture.
    Returns the pose (picture coordinates) and the coverage."""
    g, size = age_growth(year)
    seg_grow = np.clip((g - A.tree["birth"]) / 0.05, 0, 1)
    scale = v.zoom * size
    pose = tree_pose(A, (v.sx(0.0), v.sy(-0.004)), scale, t, wind, grow, seg_grow, lean)
    rgb, cov = draw_tree(A, pose, scale, t, 0.6 * wind, BARK, foliage(A, season, dry), light,
                         res=1 if v.zoom * size > 2.0 else 2)
    img = img * (1 - cov[..., None]) + rgb * cov[..., None]
    if rim_col is not None:
        edge = np.clip(cov - np.roll(cov, (2, -3), (0, 1)), 0, 1)
        img += edge[..., None] * np.asarray(rim_col, np.float32)
    if snow > 0:  # snow settles on the upper side of every branch
        top = np.clip(cov - np.roll(cov, -3, 0), 0, 1)
        img = gfx.lerp(img, np.array([0.92, 0.94, 0.98], np.float32), (top * snow)[..., None])
    return img, pose, cov


def landscape(A, c, v, stops, ground_col, far_col, sun=None, sun_power=0.0, clouds=0.0, cloud_col=(0.55, 0.55, 0.6),
              drift=6.0, stars=0.0, mist_col=None):
    """Sky, the far hills and the near hill seen through the camera (the tree is set in afterwards)."""
    img = sky(A, c, stops, sun, sun_power, clouds, cloud_col, drift, stars)
    horizon = rows(stops, 16)[12]
    fc = np.asarray(far_col, np.float32)
    for height, depth, haze in ((farther_h, 4.0, 0.55), (far_h, 2.2, 0.2)):  # farthest first, paler with distance
        fm, _ = hill(A, v.far(depth), height)
        img = gfx.lerp(img, gfx.lerp(fc, horizon, haze), fm[..., None])
    if mist_col is not None:  # a band of haze where the far hills meet the near one
        _, gy = hill(A, v, ground_h)
        band = np.exp(-((A.yy - (gy.mean() - 0.02 * A.h)) / (0.06 * A.h)) ** 2)
        img = gfx.lerp(img, np.asarray(mist_col, np.float32), (0.5 * band)[..., None])
    nm, gy = hill(A, v, ground_h)
    below = A.yy - gy[None]
    shade = gfx.smoothstep(0.0, 0.5 * A.h, below)
    wx, wy = v.wx(A.lxx * (A.w / A.lw)), v.wy(A.lyy * (A.h / A.lh))
    grass = gfx.resize(gfx.sample(A.soil, wx * 60, wy * 140)[..., None].astype(np.float32), A.w, A.h)[..., 0]
    tone = (1 - 0.45 * shade) * (0.92 + 0.08 * grass) + 0.18 * np.exp(-np.abs(below) / (2.5 * A.s))
    gc = np.asarray(ground_col, np.float32)
    img = gfx.lerp(img, gc * tone[..., None], nm[..., None])
    return img, nm


# ------------------------------------------------------------------ weather and small lights
def motes(A, img, c, amount, color=GOLD):
    """Dust or pollen drifting in the light; the snare makes some of it flash."""
    P = A.motes
    x = ((P[:, 0] + c.t * 0.004 * P[:, 2]) % 1.0) * A.w
    y = (P[:, 1] + 0.012 * np.sin(c.t * 0.5 + P[:, 3] * 6)) * A.h
    br = amount * (0.5 + 0.5 * np.sin(c.t * 1.3 + P[:, 3] * 9)) * (0.6 + 1.0 * c.snap_pulse * (P[:, 3] > 0.7))
    gfx.splat_points(img, x, y, (np.asarray(color)[None] * br[:, None] * 0.10).astype(np.float32),
                     (0.7 + 1.1 * P[:, 2]) * A.s)


def rain(A, img, t, amount, slant=0.3, color=(0.55, 0.62, 0.75)):
    """Streaks of rain, slanted by the wind."""
    R = A.rain[: int(len(A.rain) * amount)]
    if not len(R):
        return
    y = ((R[:, 1] + t * R[:, 2]) % 1.15 - 0.08) * A.h
    x = ((R[:, 0] + t * R[:, 2] * slant * 0.5) % 1.05 - 0.02) * A.w
    ln = R[:, 3] * A.h
    layer = Image.new("L", (A.w, A.h))
    d = ImageDraw.Draw(layer)
    for x0, y0, l0 in zip(x.tolist(), y.tolist(), ln.tolist()):
        d.line([(x0, y0), (x0 - l0 * slant, y0 - l0)], fill=150, width=1)
    img += (np.asarray(layer, np.float32) / 255)[..., None] * np.asarray(color, np.float32) * 0.6


def snowfall(A, img, t, amount, wind=0.3):
    """Snow drifting down and sideways."""
    S = A.snow[: int(len(A.snow) * amount)]
    y = ((S[:, 1] + t * S[:, 2]) % 1.05 - 0.02) * A.h
    x = ((S[:, 0] + t * S[:, 2] * wind + 0.015 * np.sin(t * 1.3 + S[:, 4])) % 1.0) * A.w
    col = np.array([0.95, 0.97, 1.0], np.float32)[None] * (0.35 + 0.25 * S[:, 3:4] / 2.6)
    gfx.splat_points(img, x, y, col.astype(np.float32), S[:, 3] * A.s)


def falling(A, img, t, colors, amount, wind=0.3, size=1.0, light=1.0):
    """Leaves or petals letting go and tumbling down: colours chosen per flake from `colors`."""
    F = A.flakes[: int(len(A.flakes) * amount)]
    if not len(F):
        return img
    y = ((F[:, 1] + t * F[:, 2]) % 1.1 - 0.05) * A.h
    x = ((F[:, 0] + t * F[:, 2] * wind + 0.03 * np.sin(t * 1.4 + F[:, 4])) % 1.0) * A.w
    layer = Image.new("RGBA", (A.w, A.h))
    d = ImageDraw.Draw(layer)
    cols = np.asarray(colors, np.float32).reshape(-1, 3)
    fills = [tuple(int(255 * v) for v in np.clip(cols[int(k) % len(cols)] * light, 0, 1)) + (235,) for k in F[:, 5]]
    spin = t * 3.0 + F[:, 4]
    draw_leaves(d, x, y, F[:, 3] * 7 * size * A.s * (0.6 + 0.4 * np.abs(np.sin(spin))), spin, 1.0, fills)
    arr = np.asarray(layer, np.float32) / 255
    a = arr[..., 3:4]
    return img * (1 - a) + arr[..., :3] * a


def lightning(A, img, c):
    """Some snare hits strike: a bolt from the clouds and a flash over everything. Returns the flash."""
    since = c.t - c.last_snap
    key = int(round(c.last_snap * 10))
    if since > 0.4 or key % 3:
        return 0.0
    flash = float(np.exp(-since / 0.09))
    main, fork = A.bolts[key % len(A.bolts)]
    layer = Image.new("L", (A.w, A.h))
    d = ImageDraw.Draw(layer)
    for pts, wd in ((main, 3), (fork, 2)):
        d.line([(float(x) * A.w, float(y) * A.h) for x, y in pts], fill=255, width=max(1, int(wd * A.s)))
    m = np.asarray(layer, np.float32) / 255
    glow = gfx.blur(m[..., None], 6 * A.s)[..., 0]
    img += (m * 2.5 + glow * 1.5)[..., None] * np.array([0.85, 0.9, 1.0], np.float32) * flash
    img += flash * 0.35 * np.array([0.7, 0.75, 0.9], np.float32)
    return flash


def wisp(A, img, x, y, t, power):
    """The small light that keeps the tree company: a soft glow with a few sparks circling it."""
    k = np.arange(5)
    sx = x + np.cos(t * 2.1 + k * 1.26) * 7 * A.s
    sy = y + np.sin(t * 2.7 + k * 1.26) * 5 * A.s
    gfx.splat_points(img, np.r_[x, sx], np.r_[y, sy],
                     (PALE[None] * np.r_[2.4, np.full(5, 0.35)][:, None] * power).astype(np.float32),
                     np.r_[3.2, np.full(5, 1.1)] * A.s)


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
    """Intro: a starry night over an empty hill; a small light falls, lands on the crest, and a shoot comes up."""
    p = c.p
    v = View(A, 0.0, 0.16 - 0.08 * ease(0, 1, p), 1.0 + 0.6 * ease(0.2, 1.0, p))
    img, _ = landscape(A, c, v, NIGHT, (0.025, 0.03, 0.04), (0.04, 0.05, 0.08), stars=1.0)
    land = 0.55
    if p < land:  # the light falls in a long arc
        k = p / land
        x, y = v.sx(-0.9 + 0.9 * k), v.sy(0.55 * (1 - k) ** 1.4)
        trail = np.linspace(0, 0.08, 16)
        tx, ty = v.sx(-0.9 + 0.9 * np.maximum(k - trail, 0)), v.sy(0.55 * (1 - np.maximum(k - trail, 0)) ** 1.4)
        gfx.splat_points(img, tx, ty, (PALE[None] * np.linspace(0.8, 0.0, 16)[:, None]).astype(np.float32),
                         np.linspace(2.4, 0.8, 16) * A.s)
        wisp(A, img, x, y, c.t, 1.0)
    else:
        since = (p - land) * c.dur
        bx, by = v.sx(0.0), v.sy(0.0)
        burst = np.exp(-since / 0.4)
        img += sky_burst(A, bx, by, burst)
        img = sprout(A, img, bx, by, 0.05 * v.k * ease(0, 1, (p - land) / (1 - land)), c.t)
        wisp(A, img, bx + 14 * A.s, by - 22 * A.s, c.t, 0.6 * burst + 0.15)
    c.bloom_strength = 0.9
    c.punch = 0.0
    return img


def sprout(A, img, x, y, height, t):
    """A shoot just out of the ground: a curved stem and its first two leaves, lit by the small light."""
    if height < 1.0:
        return img
    layer = Image.new("RGBA", (A.w, A.h))
    d = ImageDraw.Draw(layer)
    k = np.linspace(0, 1, 12)
    sx = x + 0.12 * height * np.sin(k * 2.2 + 0.3 * np.sin(t))
    sy = y - height * k
    col = (70, 120, 52, 255)
    d.line(list(zip(sx.tolist(), sy.tolist())), fill=col, width=max(1, int(2 * A.s)))
    tx, ty = float(sx[-1]), float(sy[-1])
    size = np.array([0.32, 0.28]) * height
    draw_leaves(d, np.array([tx - 0.3 * height, tx + 0.28 * height]), np.array([ty, ty - 0.05 * height]), size,
                np.array([0.35, -0.4]) + 0.05 * np.sin(t), 1.0, [(96, 160, 70, 255), (110, 172, 78, 255)])
    arr = np.asarray(layer, np.float32) / 255
    a = arr[..., 3:4]
    return img * (1 - a) + arr[..., :3] * 0.6 * a


def sky_burst(A, x, y, amount):
    """A soft burst of light at one point (low res)."""
    d = np.hypot(A.lxx - x * A.lw / A.w, A.lyy - y * A.lh / A.h) * (A.w / A.lw)
    g = np.exp(-d / (0.05 * A.w)) * amount
    return gfx.resize((g[..., None] * GOLD * 0.8).astype(np.float32), A.w, A.h)


def sapling(A, c):
    """First verse: the seasons flicker past and the young tree grows on its windy hill; then the view sinks
    into the earth, where the roots walk on until they find water."""
    p = c.p
    year = year_of(c, "sapling")
    season = (year % 1.0) * 4.0
    pull = ease(0.0, 0.5, p)
    down = ease(0.52, 0.85, p)
    _, size = age_growth(year)
    v = View(A, 0.0, gfx.lerp(0.12 * size + 0.02, -0.42, down), gfx.lerp(3.0, 1.25, pull))
    sun_y = A.h * (0.35 - 0.12 * np.sin(season * np.pi / 2))
    img, ground = landscape(A, c, v, DAY, gfx.lerp(np.array([0.22, 0.32, 0.14]), np.array([0.30, 0.30, 0.20]),
                                                   (season % 4) / 4),
                            (0.42, 0.52, 0.62), sun=(0.78 * A.w, sun_y), sun_power=0.6, clouds=0.25,
                            cloud_col=(0.95, 0.95, 0.97), drift=10.0)
    gust = 0.5 + 0.5 * np.sin(c.t * 0.9)
    img, pose, _ = tree_in(A, img, v, year, c.t, 0.8 + 0.6 * gust, season, lean=0.25 * gust,
                           rim_col=GOLD * 0.25)
    if down > 0:  # the earth opens up, the roots in it
        under = soil(A, v)
        R = A.roots
        _, rsize = age_growth(year)
        front = R["reach"] * ease(0.45, 0.95, p)
        seg = R["seg"]
        vis = seg[:, 5] < front
        bx, by = v.sx(0.0), v.sy(0.0)
        sc = v.k * (0.5 + 0.5 * rsize)
        layer = Image.new("L", (A.w, A.h))
        d = ImageDraw.Draw(layer)
        for x0, y0, x1, y1, wd, s0, s1 in seg[vis]:
            k = min(1.0, (front - s0) / (s1 - s0))
            d.line([(bx + x0 * sc, by + y0 * sc), (bx + (x0 + (x1 - x0) * k) * sc, by + (y0 + (y1 - y0) * k) * sc)],
                   fill=255, width=max(1, int(round(1.5 * wd * sc))))
        rl = gfx.blur(np.asarray(layer, np.float32)[..., None] / 255, 0.8 * A.s)
        under = under * (1 - 0.4 * rl) + rl * AMBER * (0.32 + 0.12 * c.kick_pulse)
        tips = seg[vis & (seg[:, 6] >= front)]
        if len(tips):
            k = np.clip((front - tips[:, 5]) / (tips[:, 6] - tips[:, 5]), 0, 1)
            gfx.splat_points(under, bx + (tips[:, 0] + (tips[:, 2] - tips[:, 0]) * k) * sc,
                             by + (tips[:, 1] + (tips[:, 3] - tips[:, 1]) * k) * sc,
                             np.tile(GOLD * 0.9, (len(tips), 1)).astype(np.float32), np.full(len(tips), 2.6 * A.s))
        for path in R["water"]:
            reached = ease(path[0, 2] - 0.02, path[0, 2], front)
            g = R["glints"]
            tw = 0.5 + 0.5 * np.sin(c.t * 2.0 + R["glint_phase"])
            gx = bx + path[0, 0] * sc + g[:, 0] * 10 * A.s
            gy = by + path[0, 1] * sc + 12 * A.s + g[:, 1] * 6 * A.s
            gfx.splat_points(under, gx, gy, (WATER[None] * (0.08 + 0.55 * reached) * tw[:, None]).astype(np.float32),
                             np.full(len(g), 1.4 * A.s))
            since = c.t - c.last_kick
            if reached > 0.99 and since < 1.4:  # water climbs the root after each kick
                s = path[:, 2]
                pos = path[0, 2] * (1 - since / 1.4)
                px = bx + np.interp(-pos, -s, path[:, 0]) * sc
                py = by + np.interp(-pos, -s, path[:, 1]) * sc
                col = gfx.lerp(WATER, GOLD, since / 1.4) * 1.6 * (1 - since / 1.4)
                gfx.splat_points(under, np.array([px]), np.array([py]), col[None].astype(np.float32),
                                 np.array([3.5 * A.s]))
        m = (ground * ease(0.0, 0.3, down))[..., None]
        img = img * (1 - m) + under * m
    motes(A, img, c, 0.5)
    counter(A, img, A.w - 300 * A.s, 40 * A.s, year, ease(0.08, 0.2, p))
    c.bloom_strength = 0.75
    c.punch = 0.004
    return img


def lull(A, c):
    """Pre-chorus: a summer dusk; the clouds pile up and the far sky flickers, and not a leaf moves.
    Inside the trunk a seam of light trembles and grows."""
    p = c.p
    year = year_of(c, "lull")
    v = View(A, 0.02, 0.20, 1.35 + 0.25 * ease(0, 1, p))
    img, _ = landscape(A, c, v, DUSK, (0.10, 0.08, 0.08), (0.30, 0.20, 0.26), sun=(0.80 * A.w, 0.62 * A.h),
                       sun_power=0.7 * (1 - 0.6 * p), clouds=0.35 + 0.45 * p, cloud_col=(0.22, 0.18, 0.24), drift=4.0)
    if c.snap_pulse > 0.5 and int(round(c.last_snap * 10)) % 2 == 0:  # far off, the storm is coming
        img += c.snap_pulse * 0.08 * np.array([0.8, 0.8, 1.0], np.float32)
    img, pose, cov = tree_in(A, img, v, year, c.t, 0.0, 1.4, light=0.55, rim_col=AMBER * 0.5)
    x0, y0, x1, y1, _ = pose  # the trunk: the first segment
    ys = np.arange(A.h, dtype=np.float32)
    k = np.clip((ys - y1[0]) / max(1.0, y0[0] - y1[0]), 0, 1)
    seam = x1[0] + (x0[0] - x1[0]) * k + (1.0 + 4.0 * p) * A.s * c.kick_pulse * np.sin(ys / (19 * A.s) + c.t * 37)
    xx = np.arange(A.w, dtype=np.float32)[None]
    inside = (ys >= y1[0]) & (ys <= y0[0])
    glow = np.exp(-((xx - seam[:, None]) / ((1.0 + 2.5 * p) * A.s)) ** 2) * inside[:, None] * cov
    img += (glow * (0.3 + 1.6 * p * p + 0.6 * p * c.kick_pulse))[..., None] * gfx.lerp(AMBER, PALE, p)
    counter(A, img, A.w - 300 * A.s, 40 * A.s, year, 1.0)
    c.bloom_strength = 0.95
    c.punch = 0.0  # the leaves must not give it away
    return img


def storm(A, c):
    """Chorus: night and a summer storm; rain, gusts, lightning on the snare. The tree holds, and beside it
    the rings it lays down light up one by one."""
    p = c.p
    year = year_of(c, "storm")
    gust = 0.5 + 0.5 * np.sin(c.t * 1.7) * np.sin(c.t * 0.6)
    shake = 0.004 * gust * np.sin(c.t * 23)
    v = View(A, -0.05 + shake, 0.22, 1.15 + 0.05 * ease(0, 1, p))
    img, _ = landscape(A, c, v, STORM, (0.03, 0.035, 0.04), (0.05, 0.055, 0.07), clouds=0.85,
                       cloud_col=(0.10, 0.11, 0.14), drift=40.0)
    flash = lightning(A, img, c)
    img, pose, _ = tree_in(A, img, v, year, c.t, 1.6 + 1.2 * gust, 1.3, light=0.30 + 0.9 * flash,
                           lean=0.6 * gust, rim_col=np.array([0.6, 0.7, 0.9]) * (0.1 + 0.8 * flash))
    rain(A, img, c.t, 1.0, slant=0.35 + 0.25 * gust)
    img = falling(A, img, c.t * 1.6, SUMMER_LEAF, 0.5, wind=1.6, light=0.3 + 0.8 * flash)
    fresh = float(np.exp(-((year * 4) % 1.0) * 4))  # each new quarter-ring flares, the ring settles
    trunk = (pose[2][0], (pose[1][0] + pose[3][0]) / 2)
    ring_disc(A, img, A.w * 0.76, A.h * 0.42, 0.17 * A.h, year, fresh * (0.5 + 0.5 * c.kick_pulse), 1.0, trunk)
    counter(A, img, A.w * 0.76 - 60 * A.s, A.h * 0.42 + 0.19 * A.h, year, 1.0)
    c.bloom_strength = 1.0
    c.streak = 0.15 + 0.4 * flash
    c.punch = 0.012
    return img


def drought(A, c):
    """Second verse: a white summer with no rain; the leaves yellow and fall, the earth cracks. Growth stops,
    then a little rain comes and it grows again, and a second ring forms in the same year."""
    p = c.p
    year = year_of(c, "drought")
    shower = ease(0.72, 0.8, p) * (1 - ease(0.92, 1.0, p))
    dry = 1.0 - 0.6 * ease(0.72, 0.9, p)
    v = View(A, 0.05 * ease(0, 1, p), 0.20, 1.0 + 0.1 * ease(0, 1, p))
    img, _ = landscape(A, c, v, HOT, gfx.lerp(np.array([0.30, 0.32, 0.18]), np.array([0.56, 0.45, 0.28]), dry),
                       (0.78, 0.74, 0.70), sun=(0.76 * A.w, 0.12 * A.h), sun_power=1.6 * (1 - 0.5 * shower),
                       clouds=0.15 + 0.5 * shower, cloud_col=(0.80, 0.82, 0.86))
    under = soil(A, v, dry)
    _, gy = hill(A, v, ground_h)
    crack_band = gfx.smoothstep(0, 4 * A.s, A.yy - gy[None]) * gfx.smoothstep(0.25 * A.h, 0.05 * A.h, A.yy - gy[None])
    img = gfx.lerp(img, img * (0.6 + 0.4 * under / (under.max() + 1e-6)), (crack_band * dry * 0.6)[..., None])
    img, pose, _ = tree_in(A, img, v, year, c.t, 0.3, 1.5, dry=dry, light=0.95, rim_col=GOLD * 0.35)
    img = falling(A, img, c.t, DRY_LEAF, 0.35 * dry, wind=0.25, light=0.95)
    rain(A, img, c.t, 0.5 * shower, slant=0.1, color=(0.75, 0.78, 0.85))
    motes(A, img, c, 0.9 * dry)
    gfx.lens_flare(img, 0.76 * A.w, 0.12 * A.h, (1.0 + 0.6 * c.kick_pulse) * (1 - shower))
    trunk = (pose[2][0], (pose[1][0] + pose[3][0]) / 2)
    fresh = float(np.exp(-((year * 4) % 1.0) * 4))
    ring_disc(A, img, A.w * 0.77, A.h * 0.40, 0.15 * A.h, year, fresh * 0.6, ease(0.0, 0.1, p), trunk)
    counter(A, img, A.w * 0.77 - 60 * A.s, A.h * 0.40 + 0.17 * A.h, year, 1.0)
    c.bloom_strength = 0.8
    c.punch = 0.005
    return img


def autumn(A, c):
    """Second pre-chorus: a golden afternoon; the leaves turn and let go, the light gets low."""
    p = c.p
    year = year_of(c, "autumn")
    v = View(A, -0.12 + 0.2 * ease(0, 1, p), 0.22, 1.3)
    img, _ = landscape(A, c, v, GOLDEN, (0.36, 0.26, 0.12), (0.52, 0.42, 0.46), sun=(0.18 * A.w, 0.58 * A.h),
                       sun_power=1.0, clouds=0.25, cloud_col=(0.98, 0.80, 0.62), mist_col=(0.95, 0.72, 0.52))
    img, pose, _ = tree_in(A, img, v, year, c.t, 0.5, 2.0 + 0.85 * p, light=0.9, rim_col=AMBER * 0.6)
    img = falling(A, img, c.t, AUTUMN_LEAF, 0.4 + 0.6 * p, wind=0.35, light=0.95)
    motes(A, img, c, 0.7)
    counter(A, img, A.w - 300 * A.s, 40 * A.s, year, 1.0)
    c.bloom_strength = 0.85
    c.punch = 0.0
    return img


def winter(A, c):
    """Second chorus: a snowstorm; the bare tree stands in the white, and the rings keep coming."""
    p = c.p
    year = year_of(c, "winter")
    gust = 0.5 + 0.5 * np.sin(c.t * 1.3)
    v = View(A, 0.04 * np.sin(c.t * 0.3), 0.21, 1.1 + 0.06 * ease(0, 1, p))
    img, _ = landscape(A, c, v, OVERCAST, (0.80, 0.83, 0.88), (0.62, 0.66, 0.74), clouds=0.9,
                       cloud_col=(0.66, 0.69, 0.75), drift=25.0, mist_col=(0.85, 0.87, 0.92))
    img, pose, _ = tree_in(A, img, v, year, c.t, 0.9 + 0.8 * gust, 3.5, light=0.55, lean=0.4 * gust, snow=0.8)
    snowfall(A, img, c.t, 1.0, wind=0.6 + 0.8 * gust)
    trunk = (pose[2][0], (pose[1][0] + pose[3][0]) / 2)
    fresh = float(np.exp(-((year * 4) % 1.0) * 4))
    ring_disc(A, img, A.w * 0.76, A.h * 0.42, 0.17 * A.h, year, fresh * (0.5 + 0.5 * c.kick_pulse), 1.0, trunk)
    counter(A, img, A.w * 0.76 - 60 * A.s, A.h * 0.42 + 0.19 * A.h, year, 1.0)
    c.bloom_strength = 0.9
    c.streak = 0.1
    c.punch = 0.012
    return img


def counting(A, c):
    """Bridge: a still winter night, close in to the heart of the trunk; the small light comes back and,
    with another beside it, goes round the rings from the pith outward, counting."""
    year = YEARS["counting"][1]
    edge = float(np.interp(year, A.ring_t, A.ring_b))
    nb = rings_at(year)
    k = np.arange(A.n_rings + 1)
    reach = 1.0 + (nb - 1) * ease(0.04, 0.92, c.p)  # rings counted so far
    lit = (0.06 + 0.34 * np.clip(reach - k, 0, 1) * (k <= nb)).astype(np.float32)
    lit[A.ring_false & (k <= reach)] += 0.35
    lit[0] = 0.0
    tint = np.where(A.ring_false[:, None], PALE, AMBER).astype(np.float32)
    zoom = 1.05 + 0.15 * ease(0, 1, c.p)
    img = cross_section(A, zoom, lit, lit * 0.6, tint, edge=edge, base=0.62, rot=0.01 * c.t)
    cx, cy = A.count_c
    i = int(min(nb, max(1, np.floor(reach))))
    for off in (0.0, 0.9):  # two lights, one a little ahead
        phi = c.t * 1.8 + off
        r = A.ring_b[i] / edge * np.interp(phi, A.th, A.wob, period=2 * np.pi) * A.log_r * zoom
        tr = phi - np.linspace(0, 0.55, 20)
        rr = A.ring_b[i] / edge * np.interp(tr, A.th, A.wob, period=2 * np.pi) * A.log_r * zoom
        fade = np.linspace(1, 0, 20) ** 2
        gfx.splat_points(img, cx + rr * np.cos(tr), cy - rr * np.sin(tr),
                         (PALE[None] * (2.2 * fade)[:, None]).astype(np.float32), (1.4 + 2.4 * fade) * A.s)
        wisp(A, img, cx + r * np.cos(phi), cy - r * np.sin(phi), c.t + off, 0.8)
    counter(A, img, A.w - 300 * A.s, 40 * A.s, year, 1.0)
    snowfall(A, img, c.t, 0.12, wind=0.1)
    c.bloom_strength = 0.85
    c.punch = 0.0
    return img


def spring(A, c):
    """Final chorus: spring comes back all at once; the hill greens, the tree flowers, petals blow across,
    the camera swings round, and the tally shows more rings than years."""
    p = c.p
    year = year_of(c, "spring")
    v = View(A, 0.25 - 0.2 * ease(0, 1, p), 0.24 - 0.04 * ease(0, 1, p), 1.75 - 0.75 * ease(0, 1, p))
    img, _ = landscape(A, c, v, MORNING, (0.24, 0.44, 0.16), (0.50, 0.62, 0.70), sun=(0.82 * A.w, 0.20 * A.h),
                       sun_power=1.2, clouds=0.2, cloud_col=(1.0, 0.97, 0.96), mist_col=(1.0, 0.92, 0.92))
    img += gfx.god_rays(img, 0.82 * A.w, 0.20 * A.h, threshold=0.95, strength=0.45 + 0.25 * c.kick_pulse)
    img, pose, _ = tree_in(A, img, v, year, c.t, 0.7, 0.35 + 0.5 * p, light=1.0, rim_col=PALE * 0.35)
    img = falling(A, img, c.t, BLOSSOM, 1.0, wind=0.9, size=0.9, light=1.0)
    motes(A, img, c, 0.8, PALE)
    trunk = (pose[2][0], (pose[1][0] + pose[3][0]) / 2)
    fresh = float(np.exp(-((year * 4) % 1.0) * 4))
    ring_disc(A, img, A.w * 0.77, A.h * 0.40, 0.17 * A.h, year, fresh * 0.6 + 0.4 * c.snap_pulse, 1.0, trunk)
    counter(A, img, A.w * 0.77 - 80 * A.s, A.h * 0.40 + 0.19 * A.h, year, 1.0, size=34)
    c.bloom_strength = 1.0
    c.streak = 0.2
    c.punch = 0.012
    return img


def dawn(A, c):
    """Outro: sunrise on the hill; the small light settles on a branch, a new leaf opens, and the tree is
    still growing."""
    p = c.p
    year = year_of(c, "dawn")
    light = 0.80 + 0.20 * ease(0, 1, p)
    v = View(A, 0.0, 0.2, 1.0 - 0.05 * ease(0, 1, p))
    img, _ = landscape(A, c, v, SUNRISE, (0.07, 0.06, 0.05), (0.56, 0.44, 0.42),
                       sun=(0.71 * A.w, A.h * (0.80 - 0.24 * light)), sun_power=0.4 + 1.1 * light, clouds=0.15,
                       cloud_col=(1.0, 0.80, 0.62), mist_col=(0.95, 0.74, 0.58))
    img, pose, _ = tree_in(A, img, v, year, c.t, 0.5, 1.1, light=0.45, rim_col=AMBER * 0.7,
                           grow=1.0 + 0.08 * ease(0.2, 0.9, p))
    term = np.where(A.tree["terminal"])[0]
    tip = term[np.argmin(pose[3][term])]  # the highest twig
    tx, ty = pose[2][tip], pose[3][tip]
    k = ease(0.0, 0.45, p)
    wisp(A, img, gfx.lerp(-0.1 * A.w, tx, k) + (1 - k) * 60 * A.s * np.sin(c.t * 2),
         gfx.lerp(0.2 * A.h, ty - 6 * A.s, k), c.t, 0.9)
    bud = ease(0.5, 0.8, p)
    gfx.splat_points(img, np.array([tx]), np.array([ty]), (SPRING_LEAF * 1.5 * bud)[None].astype(np.float32),
                     np.array([2.5 * A.s]))
    motes(A, img, c, 0.6)
    counter(A, img, A.w - 300 * A.s, 40 * A.s, year, 1.0 - ease(0.6, 0.85, p))
    c.bloom_strength = 0.85
    c.punch = 0.002
    return img


SCENES = {"seed": seed, "sapling": sapling, "lull": lull, "storm": storm, "drought": drought, "autumn": autumn,
          "winter": winter, "counting": counting, "spring": spring, "dawn": dawn}
FLASH_INTO = {"storm", "spring"}  # scenes that open with a burst of light
