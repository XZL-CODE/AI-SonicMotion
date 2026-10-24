"""Scenes of a tree with more rings than years. Each scene function draws the picture for one moment.

Scene functions take (A, c): A holds the shared assets built once per worker process,
c is the frame context (times, section progress, audio pulses; see FrameCtx in
engine/video/render.py, which loads this file and puts engine/video on the path for gfx).
They return a float32 RGB image of the picture area (letterbox bars are added later).
"""
import numpy as np
from PIL import Image, ImageDraw

import gfx

# palettes (display-space RGB 0..1); sky stops run from the top of the picture to the bottom
PREDAWN = [(0.0, (0.016, 0.022, 0.055)), (0.45, (0.045, 0.055, 0.105)), (0.72, (0.16, 0.12, 0.15)),
           (0.82, (0.12, 0.08, 0.09)), (1.0, (0.05, 0.035, 0.04))]
SUNRISE = [(0.0, (0.11, 0.17, 0.30)), (0.40, (0.36, 0.38, 0.46)), (0.66, (0.92, 0.62, 0.40)),
           (0.76, (1.0, 0.80, 0.54)), (1.0, (0.62, 0.42, 0.30))]
DRY_SKY = [(0.0, (0.58, 0.64, 0.72)), (0.5, (0.94, 0.85, 0.68)), (1.0, (1.0, 0.82, 0.56))]
AMBER = np.array([1.0, 0.62, 0.28], np.float32)
GOLD = np.array([1.0, 0.82, 0.46], np.float32)
PALE = np.array([1.0, 0.93, 0.80], np.float32)
WATER = np.array([0.35, 0.75, 1.0], np.float32)
SEASONS = np.array([[0.55, 1.0, 0.55], [1.0, 0.85, 0.40], [1.0, 0.45, 0.22], [0.72, 0.85, 1.0]], np.float32)
EARLYWOOD = np.array([0.40, 0.26, 0.15], np.float32)
LATEWOOD = np.array([0.17, 0.095, 0.05], np.float32)
BARK = np.array([0.085, 0.060, 0.045], np.float32)
DARK = np.array([0.020, 0.017, 0.015], np.float32)
TREE_X = 0.40  # the lone tree stands on the near crest, this far across the picture


def rows(stops, h):
    """Colour of each picture row for a list of (position, colour) stops: (h, 3)."""
    pos = np.array([p for p, _ in stops], np.float32)
    cols = np.array([col for _, col in stops], np.float32)
    v = np.linspace(0, 1, h, dtype=np.float32)
    return np.stack([np.interp(v, pos, cols[:, i]) for i in range(3)], -1).astype(np.float32)


def ease(e0, e1, x):
    """smoothstep for one number, as a plain float (a numpy scalar would turn float32 pictures into float64)."""
    return float(gfx.smoothstep(e0, e1, x))


def noise_1d(n, beta, seed):
    """Smooth 1-D noise (zero mean, unit std) of length n, from one row of a fractal texture."""
    row = gfx.fbm_texture(1024, beta, seed)[0]
    return np.interp(np.linspace(0, 1023, n), np.arange(1024), row).astype(np.float32)


LEAF = np.stack([np.cos(np.linspace(0, 2 * np.pi, 10, endpoint=False)),
                 0.36 * np.sin(np.linspace(0, 2 * np.pi, 10, endpoint=False))
                 * np.abs(np.sin(np.linspace(0, 2 * np.pi, 10, endpoint=False))) ** 0.35], -1)  # pointed oval


def draw_leaves(draw, x, y, size, angle, f=1.0, fill=255):
    """Leaves as pointed ovals: centres x, y, half-length size (pixels), orientation angle."""
    ca, sa = np.cos(angle)[:, None], np.sin(angle)[:, None]
    lx = (x[:, None] + (LEAF[None, :, 0] * ca - LEAF[None, :, 1] * sa) * size[:, None]) * f
    ly = (y[:, None] + (LEAF[None, :, 0] * sa + LEAF[None, :, 1] * ca) * size[:, None]) * f
    for poly in np.stack([lx, ly], -1).tolist():
        draw.polygon([tuple(v) for v in poly], fill=fill)


def beat_steps(c):
    """Section progress that moves on in steps, one step per beat: 0 at the start, 1 at the end."""
    nb = max(1.0, c.dur / c.beat)
    return float(np.clip((np.floor(c.beat_count) + gfx.smoothstep(0.0, 0.35, c.beat_count % 1.0)) / nb, 0, 1))


class Assets:
    """Textures, shapes and layouts shared by every frame (deterministic, built once per process)."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.s = w / 1920
        self.lw, self.lh = w // 4, h // 4  # low-res buffer for skies, mist and soil
        self.lyy, self.lxx = np.mgrid[0:self.lh, 0:self.lw].astype(np.float32)
        rng = np.random.default_rng(3)
        yy = np.arange(h, dtype=np.float32)[:, None]
        x = np.linspace(0, 1, w, dtype=np.float32)

        # hills: a near crest under the tree, a far line in the haze
        self.near_y = h * (0.86 - 0.13 * np.exp(-((x - TREE_X) / 0.30) ** 2)) + 0.006 * h * noise_1d(w, 1.6, 41)
        self.far_y = h * (0.75 - 0.035 * np.exp(-((x - 0.78) / 0.22) ** 2)) + 0.012 * h * noise_1d(w, 1.9, 42)
        self.near = gfx.smoothstep(-1.0, 1.0, yy - self.near_y[None]).astype(np.float32)
        self.far = gfx.smoothstep(-1.0, 1.0, yy - self.far_y[None]).astype(np.float32)
        self.base = (TREE_X * w, float(self.near_y[int(TREE_X * w)]) + 0.004 * h)
        self.mist = gfx.smoothstep(-0.3, 1.3, gfx.fbm_texture(512, 1.3, 7)).astype(np.float32)
        self.soil = gfx.fbm_texture(256, 1.2, 8)
        self.motes = np.c_[rng.uniform(0, 1, 260), rng.uniform(0.05, 0.85, 260), rng.uniform(0.3, 1.0, 260),
                           rng.uniform(0, 1, 260)]
        self.falling = np.c_[rng.uniform(0, 1, 40), rng.uniform(0, 1, 40), rng.uniform(0.5, 1.0, 40),
                             rng.uniform(0, 2 * np.pi, 40)]

        self.tree = self._tree(rng)
        self.roots = self._roots(rng)
        self._still(rng)
        self._rings(rng)

        self.vig = gfx.vignette(h, w, 0.36)
        self.grain = [rng.normal(0, 1, (h // 2, w // 2, 1)).astype(np.float32) for _ in range(6)]

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
        for i in np.where(depth >= 6)[0]:
            k = 4 if depth[i] == 7 else 2
            leaf_seg += [i] * k
            leaf_off.append(rng.normal(0, 0.011, (k, 2)))
            leaf_size.append(rng.uniform(0.0035, 0.0075, k))
        return {"par": par, "ang": np.array(ang), "length": np.array(length), "width": np.array(width),
                "depth": depth, "terminal": terminal, "birth": birth, "phase": rng.uniform(0, 2 * np.pi, n),
                "levels": [np.where(depth == d)[0] for d in range(depth.max() + 1)],
                "leaf_seg": np.array(leaf_seg), "leaf_off": np.concatenate(leaf_off),
                "leaf_size": np.concatenate(leaf_size), "leaf_phase": rng.uniform(0, 2 * np.pi, len(leaf_seg)),
                "leaf_keep": rng.uniform(0, 1, len(leaf_seg))}

    def _roots(self, rng):
        """Roots under the tree as segments in picture heights from the trunk base (y down), each with its
        distance from the base along the root; the longest roots end in water."""
        segs = []  # x0, y0, x1, y1, width, s0, s1

        def grow(x, y, a, wd, s, steps, lateral):
            pts = [(x, y, s)]
            for _ in range(steps):
                a = a * 0.96 + rng.normal(0, 0.16)  # wander, but keep heading down
                step = 0.018
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
            steps = int(rng.integers(30, 42)) if i in (1, 3, 4) else int(rng.integers(16, 26))
            pts = grow(0.0, 0.0, a, 0.010, 0.0, steps, True)
            if i in (1, 3, 4):
                water.append(np.array(pts[::-1]))  # path from the tip back up to the trunk
        segs = np.array(segs, np.float32)
        order = np.argsort(segs[:, 5])
        return {"seg": segs[order], "water": water, "reach": max(p[0, 2] for p in water),
                "glints": rng.normal(0, 1, (60, 2)).astype(np.float32), "glint_phase": rng.uniform(0, 6.3, 60)}

    def _still(self, rng):
        """Static layers of the close-up: warm haze with bokeh, the trunk and its bark, leaves held still."""
        w, h, s = self.w, self.h, self.s
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        bg = rows([(0.0, (0.62, 0.50, 0.38)), (0.5, (0.42, 0.30, 0.22)), (1.0, (0.20, 0.13, 0.09))], h)
        bg = np.repeat(bg[:, None], w, 1)
        n = 70
        gfx.splat_points(bg, rng.uniform(0.45, 1.0, n) * w, rng.uniform(0, 0.9, n) * h,
                         (GOLD[None] * rng.uniform(0.05, 0.22, (n, 1))).astype(np.float32),
                         rng.uniform(14, 46, n) * s)
        self.still_bg = bg.astype(np.float32)
        v = yy / h
        left = w * (0.27 + 0.025 * np.sin(v * 2.4 + 0.5))
        right = left + w * (0.17 - 0.03 * v)
        self.still_trunk = (gfx.smoothstep(left - 2, left + 2, xx) * gfx.smoothstep(right + 2, right - 2, xx)
                            ).astype(np.float32)
        ridge = gfx.sample(gfx.fbm_texture(512, 1.1, 17), xx * 0.35, yy * 0.04)
        self.still_bark = (0.55 + 0.45 * np.tanh(1.5 * ridge)).astype(np.float32)
        self.seam_x = (left[:, 0] + (right - left)[:, 0] * 0.46 + 5 * s * noise_1d(h, 1.4, 18)).astype(np.float32)
        im = Image.new("L", (w, h))
        d = ImageDraw.Draw(im)
        for x0, y0, a, ln in [(1.02, 0.08, 3.5, 0.55), (0.98, 0.42, 3.0, 0.40), (0.62, -0.05, 2.0, 0.42),
                              (-0.02, 0.02, -0.25, 0.30)]:  # twigs reaching in from the edges
            k = np.linspace(0, 1, 40)
            bend = a + 0.5 * k * (1 if x0 > 0.5 else -1)
            tx = (x0 + np.cumsum(np.cos(bend)) * ln / 40 * 0.56) * w
            ty = (y0 + np.cumsum(np.sin(bend)) * ln / 40) * h
            d.line(list(zip(tx.tolist(), ty.tolist())), fill=255, width=max(1, int(3 * s)))
            n = 34
            at = rng.integers(4, 40, n)
            side = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
            ang = bend[at] + side * rng.uniform(0.5, 1.0, n)
            size = rng.uniform(16, 34, n) * s
            draw_leaves(d, tx[at] + np.cos(ang) * size, ty[at] + np.sin(ang) * size, size, ang)
        self.still_leaves = gfx.blur(np.asarray(im, np.float32)[..., None] / 255, 2.5 * s)[..., 0]

    def _rings(self, rng):
        """The cut face: ring boundaries (normalised radius) including the false rings of the dry years,
        angular tables for the wobble, the wood rays and the radial checks, and polar maps per view."""
        years = 34
        width = rng.lognormal(0, 0.30, years)
        dry = rng.uniform(0, 1, years) < 0.28
        width[dry] *= 0.55
        b, false, acc = [0.0], [False], 0.0
        for k in range(years):
            if dry[k]:  # growth stopped in the drought and started again: one year, two rings
                b.append(acc + width[k] * rng.uniform(0.45, 0.6))
                false.append(True)
            acc += width[k]
            b.append(acc)
            false.append(False)
        self.ring_b = (np.array(b) / acc).astype(np.float32)
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
        self.view_main = self._view(0.5 * self.w, 0.5 * self.h)
        self.view_count = self._view(0.30 * self.w, 0.60 * self.h)

    def _view(self, cx, cy):
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        dx, dy = xx - cx, cy - yy
        return np.arctan2(dy, dx).astype(np.float32), np.hypot(dx, dy).astype(np.float32), (cx, cy)


# ------------------------------------------------------------------ the tree
def tree_pose(A, base, scale, t, wind, grow=1.0, seg_grow=None):
    """Segment end points for this frame: the crown sways more than the trunk; grow lengthens the twigs."""
    T = A.tree
    sway = wind * ((0.010 + 0.09 * (T["depth"] / 7.0) ** 1.7) * np.sin(1.2 * t + T["phase"])
                   + 0.02 * np.sin(0.4 * t))
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


def leaf_points(A, pose, scale, t, flutter, keep=1.0):
    """Leaf centres, sizes (pixels) and orientations hung on the outer twigs."""
    T = A.tree
    _, _, x1, y1, a = pose
    i = T["leaf_seg"]
    sel = T["leaf_keep"] < keep
    off = T["leaf_off"] * scale * A.h
    ca, sa = np.cos(a[i]), np.sin(a[i])
    wave = flutter * 2.0 * A.s * np.sin(3.0 * t + T["leaf_phase"])
    lx = x1[i] + off[:, 0] * ca - off[:, 1] * sa + wave
    ly = y1[i] + off[:, 0] * sa + off[:, 1] * ca
    turn = a[i] - np.pi / 2 + T["leaf_phase"] + 0.25 * flutter * np.sin(2.3 * t + T["leaf_phase"])
    return lx[sel], ly[sel], (T["leaf_size"] * scale * A.h)[sel] * 1.4, turn[sel]


def tree_mask(A, pose, scale, leaves=None, res=2):
    """Silhouette of the tree as a soft mask at picture size (drawn at 1/res; the upscale smooths it)."""
    x0, y0, x1, y1, _ = pose
    f = 1.0 / res
    im = Image.new("L", (A.w // res, A.h // res))
    d = ImageDraw.Draw(im)
    widths = A.tree["width"] * scale * A.h * f
    for i in range(len(x0)):
        wd = max(1, int(round(widths[i])))
        d.line([(x0[i] * f, y0[i] * f), (x1[i] * f, y1[i] * f)], fill=255, width=wd)
        if wd > 3:  # round the joints of the thick limbs
            r = wd / 2
            d.ellipse([x1[i] * f - r, y1[i] * f - r, x1[i] * f + r, y1[i] * f + r], fill=255)
    if leaves is not None:
        lx, ly, size, turn = leaves
        draw_leaves(d, lx, ly, np.maximum(size, 1.6 * res), turn, f)
    m = np.asarray(im, np.float32)[..., None] / 255
    return gfx.resize(m, A.w, A.h)[..., 0]


def rim(mask, dy, dx, amount):
    """Edge of a silhouette on the side facing the light (dy, dx: pixels towards the light)."""
    return np.clip(mask - np.roll(mask, (dy, -dx), (0, 1)), 0, 1) * amount


# ------------------------------------------------------------------ shared layers
def mist(A, t, y_center, thickness, speed, amount):
    """Low-res mist band drifting sideways: coverage 0..1 at picture size."""
    u = (A.lxx + t * speed) * (512 / A.lw) * 0.6
    v = A.lyy * (512 / A.lh) * 0.35 + 40
    band = np.exp(-((A.lyy / A.lh - y_center) / thickness) ** 2)
    m = gfx.sample(A.mist, u, v) * band * amount
    return gfx.resize(m[..., None].astype(np.float32), A.w, A.h)[..., 0]


def motes(A, img, c, amount):
    """Dust drifting in the low sun; the snare makes some of it flash."""
    P = A.motes
    x = ((P[:, 0] + c.t * 0.004 * P[:, 2]) % 1.0) * A.w
    y = (P[:, 1] + 0.012 * np.sin(c.t * 0.5 + P[:, 3] * 6)) * A.h
    br = amount * (0.5 + 0.5 * np.sin(c.t * 1.3 + P[:, 3] * 9)) * (0.6 + 1.0 * c.snap_pulse * (P[:, 3] > 0.7))
    gfx.splat_points(img, x, y, (GOLD[None] * br[:, None] * 0.10).astype(np.float32), (0.7 + 1.1 * P[:, 2]) * A.s)


def sun_glow(A, x, y, size, power, color=(1.0, 0.70, 0.42)):
    """Halo of a low sun (computed at low res)."""
    d = np.hypot(A.lxx - x * A.lw / A.w, A.lyy - y * A.lh / A.h) * (A.w / A.lw)
    g = np.exp(-d / (0.03 * size * A.w)) * 0.9 + np.exp(-d / (0.16 * size * A.w)) * 0.35
    return gfx.resize((g[..., None] * np.asarray(color, np.float32) * power).astype(np.float32), A.w, A.h)


def landscape(A, c, light, wind=1.0, grow=1.0):
    """Hills at first light and the lone tree on the near crest; light 0 (before dawn) .. 1 (sun up)."""
    h, w = A.h, A.w
    light = float(light)
    img = np.repeat(gfx.lerp(rows(PREDAWN, h), rows(SUNRISE, h), light)[:, None], w, 1)
    sun_x, sun_y = 0.71 * w, h * (0.80 - 0.24 * light)
    img += sun_glow(A, sun_x, sun_y, 1.0, 0.25 + 1.1 * light) * (0.9 + 0.2 * c.low)
    far_col = gfx.lerp(np.array([0.06, 0.06, 0.10]), np.array([0.58, 0.44, 0.42]), light).astype(np.float32)
    img = gfx.lerp(img, far_col, A.far[..., None])
    mist_col = gfx.lerp(np.array([0.10, 0.10, 0.16]), np.array([0.95, 0.74, 0.58]), light).astype(np.float32)
    img = gfx.lerp(img, mist_col, (mist(A, c.t, 0.79, 0.07, 6.0, 0.85))[..., None])
    pose = tree_pose(A, A.base, 1.12, c.t, wind, grow)
    tm = tree_mask(A, pose, 1.12, leaf_points(A, pose, 1.12, c.t, wind))
    near_col = gfx.lerp(np.array([0.010, 0.010, 0.018]), np.array([0.07, 0.05, 0.045]), light).astype(np.float32)
    sil = np.clip(A.near + tm, 0, 1)[..., None]
    img = gfx.lerp(img, near_col, sil)
    img += rim(tm, 2, 3, 0.55 * light)[..., None] * AMBER
    img = gfx.lerp(img, mist_col, (mist(A, c.t + 40, 0.93, 0.06, 11.0, 0.35 * light))[..., None])
    motes(A, img, c, 0.4 + 0.8 * light)
    return img, pose


# ------------------------------------------------------------------ scenes
def hill(A, c):
    """Intro: before dawn, a lone tree on the crest; the sky starts to lighten."""
    img, _ = landscape(A, c, 0.04 + 0.32 * ease(0, 1, c.p), wind=0.5)
    c.bloom_strength = 0.75
    c.punch = 0.003
    return img


def soil(A, c, shift):
    """Earth below the crest for picture rows seen `shift` pixels down (low res, upscaled)."""
    wy = (A.lyy * (A.h / A.lh) + shift) / A.h
    depth = np.clip((wy - 0.74) / 0.9, 0, 1)
    tex = gfx.sample(A.soil, A.lxx * 0.8, wy * A.lh * 0.8)
    base = gfx.lerp(np.array([0.18, 0.12, 0.08], np.float32), np.array([0.055, 0.040, 0.034], np.float32),
                    depth[..., None])
    img = base * (0.8 + 0.12 * tex)[..., None]
    for path in A.roots["water"]:  # damp earth around the water the roots are heading for
        px, py = A.base[0] + path[0, 0] * A.h, A.base[1] + path[0, 1] * A.h - shift
        d = np.hypot(A.lxx * (A.w / A.lw) - px, A.lyy * (A.h / A.lh) - py) / A.h
        img += (np.exp(-d / 0.06) * 0.05)[..., None] * WATER
    return gfx.resize(img.astype(np.float32), A.w, A.h)


def roots(A, c):
    """First verse: from the tree the view sinks into the earth; the roots go down until they find water,
    and then every kick sends a little of it up towards the tree."""
    h, w = A.h, A.w
    img, _ = landscape(A, c, 0.36 + 0.24 * c.p, wind=0.7)
    e = ease(0.06, 0.55, c.p)
    shift = int(round(0.70 * h * e * e * (3 - 2 * e)))
    ground = np.concatenate([A.near[shift:], np.ones((shift, w), np.float32)])
    img = np.concatenate([img[shift:], np.zeros((shift, w, 3), np.float32)])
    reveal = ease(0.0, 0.25, e)
    under = soil(A, c, shift)

    R = A.roots
    front = R["reach"] * ease(0.15, 0.92, beat_steps(c))
    seg = R["seg"]
    vis = seg[:, 5] < front
    bx, by = A.base[0], A.base[1] - shift
    layer = Image.new("L", (w, h))
    d = ImageDraw.Draw(layer)
    for x0, y0, x1, y1, wd, s0, s1 in seg[vis]:
        k = min(1.0, (front - s0) / (s1 - s0))
        d.line([(bx + x0 * h, by + y0 * h), (bx + (x0 + (x1 - x0) * k) * h, by + (y0 + (y1 - y0) * k) * h)],
               fill=255, width=max(1, int(round(1.5 * wd * h))))
    rl = gfx.blur(np.asarray(layer, np.float32)[..., None] / 255, 0.8 * A.s)
    under = under * (1 - 0.4 * rl) + rl * AMBER * (0.30 + 0.12 * c.kick_pulse)
    tips = seg[vis & (seg[:, 6] >= front)]
    if len(tips):  # the ends still growing glow
        k = np.clip((front - tips[:, 5]) / (tips[:, 6] - tips[:, 5]), 0, 1)
        tx = bx + (tips[:, 0] + (tips[:, 2] - tips[:, 0]) * k) * h
        ty = by + (tips[:, 1] + (tips[:, 3] - tips[:, 1]) * k) * h
        gfx.splat_points(under, tx, ty, np.tile(GOLD * 0.9, (len(tips), 1)).astype(np.float32),
                         np.full(len(tips), 2.6 * A.s))
    for path in R["water"]:
        reached = gfx.smoothstep(path[0, 2] - 0.02, path[0, 2], front)
        g = R["glints"]
        tw = 0.5 + 0.5 * np.sin(c.t * 2.0 + R["glint_phase"])
        gx = bx + path[0, 0] * h + g[:, 0] * 10 * A.s
        gy = by + path[0, 1] * h + 12 * A.s + g[:, 1] * 6 * A.s
        gfx.splat_points(under, gx, gy, (WATER[None] * (0.08 + 0.55 * reached) * tw[:, None]).astype(np.float32),
                         np.full(len(g), 1.4 * A.s))
        since = c.t - c.last_kick
        if reached > 0.99 and since < 1.4:  # water climbs the root after each kick
            s = path[:, 2]
            pos = path[0, 2] * (1 - since / 1.4)
            px = bx + np.interp(-pos, -s, path[:, 0]) * h
            py = by + np.interp(-pos, -s, path[:, 1]) * h
            col = gfx.lerp(WATER, GOLD, since / 1.4) * 1.6 * (1 - since / 1.4)
            gfx.splat_points(under, np.array([px]), np.array([py]), col[None].astype(np.float32),
                             np.array([3.5 * A.s]))
    m = (ground * reveal)[..., None]
    img = img * (1 - m) + under * m
    c.bloom_strength = 0.8
    c.punch = 0.005
    return img


def still(A, c):
    """Pre-chorus: up close, not a leaf moves; inside the trunk a seam of light trembles and grows."""
    h, w, s = A.h, A.w, A.s
    img = A.still_bg * (0.85 + 0.2 * c.low)
    trunk = A.still_trunk[..., None]
    img = img * (1 - trunk) + trunk * BARK * A.still_bark[..., None]
    img = img * (1 - 0.92 * A.still_leaves[..., None])
    p = c.p
    yy = np.arange(h, dtype=np.float32)
    jitter = (1.5 + 5.0 * p) * s * c.kick_pulse * np.sin(yy / (23 * s) + c.t * 37)
    xs = A.seam_x + jitter
    xx = np.arange(w, dtype=np.float32)[None]
    sigma = (1.2 + 5.0 * p * p) * s
    glow = np.exp(-((xx - xs[:, None]) / sigma) ** 2) * A.still_trunk
    level = 0.25 + 1.4 * p * p + 0.9 * p * c.kick_pulse
    img += (glow * level)[..., None] * gfx.lerp(AMBER, PALE, p)
    motes(A, img, c, 0.5)
    c.bloom_strength = 0.95
    c.punch = 0.0  # the leaves must not give it away
    return img


def cross_section(A, view, zoom, lit, fill, tint, base=1.0, rot=0.0):
    """The cut face of the trunk: rings and their false rings, wood rays, radial checks and bark.
    lit / fill: light on each ring boundary and inside each ring; tint: their colours (n_rings + 1, 3)."""
    th, r0, _ = view
    if rot:
        th = (th + rot + np.pi) % (2 * np.pi) - np.pi
    q = r0 / (A.log_r * zoom) / np.interp(th, A.th, A.wob, period=2 * np.pi).astype(np.float32)
    u = np.interp(q, A.ring_b, A.ring_i).astype(np.float32)
    k = np.minimum(u.astype(np.int32), A.n_rings - 1)
    late = gfx.smoothstep(0.5, 0.97, u - np.floor(u))
    img = EARLYWOOD + (LATEWOOD - EARLYWOOD) * late[..., None]
    fiber = np.interp(th, A.th, A.fiber, period=2 * np.pi).astype(np.float32)
    crack = np.interp(th, A.th, A.crack, period=2 * np.pi).astype(np.float32) * gfx.smoothstep(0.35, 0.9, q)
    shade = (0.82 + 0.18 * fiber) * (1 - 0.8 * crack) * (0.5 + 0.5 * gfx.smoothstep(0.0, 0.05, q))
    img = img * shade[..., None]
    rough = np.interp(th * 3, A.th, A.bark_tex, period=2 * np.pi).astype(np.float32)
    bark = BARK * (0.6 + 0.6 * rough)[..., None]
    img = gfx.lerp(img, bark, gfx.smoothstep(1.0, 1.006, q)[..., None])
    img = gfx.lerp(img, DARK, gfx.smoothstep(1.06, 1.075, q)[..., None]) * base
    near = np.rint(u)
    kb = near.astype(np.int32)
    d_px = np.abs(u - near) * A.ring_gap[kb] * (A.log_r * zoom)
    line = np.exp(-(d_px / (1.3 * A.s)) ** 2) * (q < 1.004) * lit[kb]
    img += line[..., None] * tint[kb]
    img += ((fill[k] * 0.10) * (q < 1.0))[..., None] * tint[k]
    return img


def ring_tint(A):
    return np.where(A.ring_false[:, None], PALE, AMBER).astype(np.float32)


def ring_lights(A, c, front, held):
    """Per-ring light: the rings counted so far stay lit, the front pulses with the kick, the snare
    sends a ripple out from the pith, and the spectrum shimmers across them."""
    k = np.arange(A.n_rings + 1, dtype=np.float32)
    lit = 0.06 + held * np.clip(front - k, 0, 1)
    lit += (1.1 + 1.0 * c.kick_pulse) * np.exp(-((k - front) / 1.3) ** 2)
    since = c.t - c.last_snap
    lit += 0.7 * np.exp(-((k - since * 45.0) / 1.5) ** 2) * np.exp(-since / 0.35)
    if c.spectrum is not None:
        lit += 0.25 * np.asarray(c.spectrum)[(k / (A.n_rings + 1) * 24).astype(int)]
    lit[A.ring_false] *= 1.35
    lit[0] = 0.0
    return lit.astype(np.float32)


def rings(A, c):
    """Chorus: the cut face; a light counts the rings outward on the beat, the false rings shine pale."""
    front = beat_steps(c) * (A.n_rings + 2)
    lit = ring_lights(A, c, front, 0.30)
    if c.occurrence:  # the second time, the first count is still there
        lit[1:] += 0.18
    img = cross_section(A, A.view_main, 1.04 + 0.10 * c.p, lit, lit * 0.5, ring_tint(A),
                        base=0.55 + 0.1 * c.low, rot=0.02 * c.t)
    c.bloom_strength = 1.0
    c.streak = 0.15 + 0.25 * c.kick_pulse
    c.punch = 0.012
    return img


def branches(A, c):
    """Second verse: a dry, cloudless summer; the crown keeps reaching for the light, one branch a beat."""
    h, w, s = A.h, A.w, A.s
    img = np.repeat(rows(DRY_SKY, h)[:, None], w, 1)
    sun_x, sun_y = 0.80 * w, 0.13 * h
    img += sun_glow(A, sun_x, sun_y, 1.3, 1.3, (1.0, 0.92, 0.75))
    gfx.splat_points(img, np.array([sun_x]), np.array([sun_y]), (PALE * 4.0)[None].astype(np.float32),
                     np.array([20 * s]))
    g = 0.42 + 0.58 * beat_steps(c)
    seg_grow = np.clip((g - A.tree["birth"]) / 0.05, 0, 1)
    scale = 2.1
    pose = tree_pose(A, (0.42 * w, 1.30 * h), scale, c.t, 0.25, seg_grow=seg_grow)
    tm = tree_mask(A, pose, scale, leaf_points(A, pose, scale, c.t, 0.3, keep=0.22 * g))
    img = gfx.lerp(img, np.array([0.12, 0.085, 0.065], np.float32), tm[..., None])
    img += rim(tm, 3, 3, 0.7)[..., None] * GOLD
    new = (seg_grow > 0) & (A.tree["birth"] > g - 0.05)  # every new branch catches the light as it grows
    if new.any():
        x1, y1 = pose[2][new], pose[3][new]
        age = np.clip((g - A.tree["birth"][new]) / 0.05, 0, 1)
        gfx.splat_points(img, x1, y1, (PALE[None] * (1.2 * (1 - age))[:, None]).astype(np.float32),
                         np.full(len(x1), 2.5 * s))
    F = A.falling  # dry leaves let go, one by one
    fy = ((F[:, 1] + c.t * 0.05 * F[:, 2]) % 1.1 - 0.05) * h
    fx = (F[:, 0] + 0.02 * np.sin(c.t * 1.5 + F[:, 3])) * w
    leaf = Image.new("L", (w, h))
    d = ImageDraw.Draw(leaf)
    for x, y, r in zip(fx, fy, F[:, 2] * 6 * s):
        d.ellipse([x - r, y - r * 0.5, x + r, y + r * 0.5], fill=200)
    img = img * (1 - 0.8 * np.asarray(leaf, np.float32)[..., None] / 255)
    motes(A, img, c, 0.8)
    gfx.lens_flare(img, sun_x, sun_y, 1.0 + 0.8 * c.kick_pulse)
    c.bloom_strength = 0.8
    c.punch = 0.006
    return img


def counting(A, c):
    """Bridge: up close to the pith; two small lights go round the rings together, one ring a bar."""
    zoom = 1.9
    bar = c.beat_count / 4
    cur = int(min(A.n_rings, 4 + int(bar)))
    k = np.arange(A.n_rings + 1)
    lit = (0.05 + 0.32 * (k < cur) + 0.45 * (k == cur)).astype(np.float32)
    lit[A.ring_false] *= 1.3
    lit[0] = 0.0
    img = cross_section(A, A.view_count, zoom, lit, lit * 0.6, ring_tint(A), base=0.7)
    cx, cy = A.view_count[2]
    for off in (0.0, 1.15):  # one a little ahead of the other
        phi = 1.2 + off + (bar % 1.0) * 2 * np.pi * 0.9 - np.linspace(0, 0.6, 24)
        r = A.ring_b[cur] * np.interp(phi, A.th, A.wob, period=2 * np.pi) * A.log_r * zoom
        fade = np.linspace(1, 0, 24) ** 2
        gfx.splat_points(img, cx + r * np.cos(phi), cy - r * np.sin(phi),
                         (PALE[None] * (2.4 * fade + 0.8 * c.snap_pulse * fade)[:, None]).astype(np.float32),
                         (1.6 + 2.6 * fade) * A.s)
    c.bloom_strength = 0.85
    c.punch = 0.0
    return img


def seasons(A, c):
    """Final chorus: every ring lights from the pith outward, the seasons turning in them, as the view
    pulls back to the whole trunk."""
    front = ease(0.0, 0.85, c.p) * (A.n_rings + 2)
    lit = ring_lights(A, c, front, 0.55)
    lit[A.ring_false] += 0.8 * c.snap_pulse
    k = np.arange(A.n_rings + 1)
    tint = SEASONS[(k + int(c.beat_count // 4)) % 4] * 0.55 + AMBER * 0.45
    tint[A.ring_false] = PALE
    zoom = 1.30 - 0.50 * ease(0.05, 0.95, c.p)
    img = cross_section(A, A.view_main, zoom, lit, lit * 0.6, tint.astype(np.float32), base=0.6 + 0.1 * c.low,
                        rot=-0.015 * c.t)
    c.bloom_strength = 1.1
    c.streak = 0.3
    c.punch = 0.014
    return img


def dawn(A, c):
    """Outro: the sun is up over the hill, and the tree is still growing."""
    grow = 1.0 + 0.12 * ease(0.1, 0.95, c.p)
    img, pose = landscape(A, c, 0.80 + 0.20 * ease(0, 1, c.p), wind=0.6, grow=grow)
    term = A.tree["terminal"]
    bud = ease(0.2, 0.8, c.p) * (0.6 + 0.4 * np.sin(c.t * 2 + A.tree["phase"][term]))
    gfx.splat_points(img, pose[2][term], pose[3][term], (GOLD[None] * (0.25 * bud)[:, None]).astype(np.float32),
                     np.full(term.sum(), 1.4 * A.s))
    c.bloom_strength = 0.85
    c.punch = 0.002
    return img


SCENES = {"hill": hill, "roots": roots, "still": still, "rings": rings, "branches": branches,
          "counting": counting, "seasons": seasons, "dawn": dawn}
FLASH_INTO = {"rings", "seasons"}  # scenes that open with a burst of light
