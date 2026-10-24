"""Scenes of the night flight. Each scene function draws the picture for one moment.

Scene functions take (A, c): A holds the shared assets built once per worker process,
c is the frame context (times, section progress, audio pulses; see FrameCtx in
engine/video/render.py, which loads this file and puts engine/video on the path for gfx).
They return a float32 RGB image of the picture area (letterbox bars are added later).
"""
import numpy as np
from PIL import Image, ImageDraw

import gfx

# palettes (display-space RGB 0..1)
NIGHT_SKY = [(-0.2, (0.012, 0.010, 0.022)), (0.0, (0.16, 0.08, 0.10)), (0.04, (0.07, 0.05, 0.12)),
             (0.25, (0.02, 0.025, 0.07)), (1.0, (0.004, 0.006, 0.02))]
SUNSET_SKY = [(-0.15, (0.20, 0.06, 0.10)), (0.0, (1.0, 0.56, 0.26)), (0.05, (0.95, 0.42, 0.30)),
              (0.16, (0.55, 0.24, 0.40)), (0.4, (0.16, 0.10, 0.28)), (1.0, (0.03, 0.03, 0.10))]
DUSK_SKY = [(-0.15, (0.10, 0.04, 0.10)), (0.0, (0.80, 0.36, 0.30)), (0.06, (0.52, 0.22, 0.36)),
            (0.2, (0.20, 0.10, 0.30)), (0.5, (0.06, 0.05, 0.16)), (1.0, (0.01, 0.01, 0.05))]
CITY_SKY = [(-1.0, (0.030, 0.016, 0.014)), (-0.06, (0.070, 0.034, 0.030)), (0.0, (0.20, 0.10, 0.10)),
            (0.05, (0.08, 0.05, 0.10)), (0.3, (0.02, 0.022, 0.06)), (1.0, (0.004, 0.006, 0.02))]
WARM = np.array([1.0, 0.72, 0.42], np.float32)
WHITE = np.array([1.0, 0.95, 0.88], np.float32)
CYAN = np.array([0.45, 0.85, 1.0], np.float32)
GOLD = np.array([1.0, 0.78, 0.38], np.float32)


class Assets:
    """Textures and layouts shared by every frame (deterministic, built once per process)."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.lw, self.lh = w // 3, h // 3  # low-res buffer for skies, clouds and fog
        self.cam = gfx.Camera(w, h, 58)
        self.lcam = gfx.Camera(self.lw, self.lh, 58)

        base = gfx.fbm_texture(1024, 1.55, 11)
        detail = np.tile(gfx.fbm_texture(512, 1.15, 12), (2, 2))
        height = 0.75 * base + 0.35 * detail
        self.dens = gfx.smoothstep(-0.55, 0.9, height).astype(np.float32)
        self.dens_far = gfx.blur(self.dens[..., None], 6)[..., 0]
        self.gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)).astype(np.float32)
        self.gz = (np.roll(height, -1, 0) - np.roll(height, 1, 0)).astype(np.float32)
        self.fog = gfx.smoothstep(-0.4, 1.4, gfx.fbm_texture(512, 1.35, 13)).astype(np.float32)
        self.fog_soft = gfx.blur(self.fog[..., None], 3)[..., 0]
        self.hn = (gfx.smoothstep(-0.9, 1.5, height) ** 0.7).astype(np.float32)  # cloud-top height 0..1
        # smooth geometry for marching (no sub-texel noise to alias); fine detail only shades
        self.hn_geo = gfx.blur(gfx.smoothstep(-0.9, 1.5, 0.85 * base + 0.15 * detail)[..., None] ** 0.7, 3)[..., 0]
        self.hgx = (0.5 * (np.roll(self.hn_geo, -1, 1) - np.roll(self.hn_geo, 1, 1))).astype(np.float32)
        self.hgz = (0.5 * (np.roll(self.hn_geo, -1, 0) - np.roll(self.hn_geo, 1, 0))).astype(np.float32)

        rng = np.random.default_rng(5)
        n = 2200
        self.star_az = rng.uniform(-np.pi, np.pi, n)
        self.star_el = np.arcsin(rng.uniform(0.02, 1.0, n))
        self.star_mag = rng.power(3.2, n)
        self.star_phase = rng.uniform(0, 2 * np.pi, n)
        self.star_col = gfx.lerp(np.array([0.75, 0.85, 1.0]), np.array([1.0, 0.85, 0.7]), rng.uniform(0, 1, (n, 1)))

        self.runway = self._runway()
        self.city = self._city(rng)
        self.answers = np.c_[rng.uniform(-260, 260, 420), rng.uniform(8, 140, 420), rng.uniform(0, 900, 420),
                             rng.uniform(0, 1, 420)]
        self.river = np.c_[rng.uniform(0, 1, 2600), rng.normal(0, 1, 2600), rng.uniform(0.5, 1.5, 2600),
                           rng.uniform(0, 2 * np.pi, 2600)]
        t = (1 + 5 ** 0.5) / 2
        v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t],
                      [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
        self.ico_v = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.ico_f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
                      (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
                      (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
        self.lyy, self.lxx = np.mgrid[0:self.lh, 0:self.lw].astype(np.float32)
        self.vig = gfx.vignette(h, w, 0.38)
        self.grain = [rng.normal(0, 1, (h // 2, w // 2, 1)).astype(np.float32) for _ in range(6)]

    @staticmethod
    def _runway():
        pts, kind = [], []
        for z in np.arange(0, 3001, 60):  # edge lights
            pts += [(-22.5, 0.4, z), (22.5, 0.4, z)]
            kind += [0, 0]
        for z in np.arange(15, 3000, 30):  # centreline (red towards the far end)
            pts.append((0, 0.2, z))
            kind.append(1 if z < 2100 else 2)
        for x in np.arange(-21, 22, 3):  # threshold (green) and far end (red)
            pts += [(x, 0.3, 0), (x, 0.3, 3000)]
            kind += [3, 2]
        for z in np.arange(-900, 0, 30):  # approach lights: crossbars + sequenced strobes
            for x in np.arange(-6, 7, 3):
                pts.append((x, 1.0, z))
                kind.append(4)
            pts.append((0, 1.6, z))
            kind.append(5)
        for z in np.arange(-200, 3201, 50):  # parallel taxiway, blue edges
            pts += [(-150, 0.3, z), (-118, 0.3, z)]
            kind += [6, 6]
        for x, z in [(620, 1500), (700, 1700), (780, 1900), (660, 2150), (820, 2400)]:  # apron floodlights
            pts.append((x, 18, z))
            kind.append(7)
        pts.append((520, 45, 1850))  # beacon on the tower
        kind.append(8)
        return {"pts": np.array(pts, float), "kind": np.array(kind)}

    def _city(self, rng):
        """Street grid thinned by a district-density field, a bright downtown, and highways."""
        dens = gfx.smoothstep(-0.9, 1.1, gfx.fbm_texture(256, 1.8, 31))
        self.city_dens = dens.astype(np.float32)

        def density(x, z):
            return gfx.sample(dens, (x + 8000) / 16000 * 256, (z + 2000) / 16000 * 256)

        sodium, warm, led, cool = [1.0, 0.58, 0.24], [1.0, 0.80, 0.52], [0.92, 0.94, 1.0], [0.55, 0.85, 1.0]
        pts, col = [], []

        def add(p, c, keep_prob):
            keep = rng.uniform(0, 1, len(p)) < keep_prob
            keep &= (np.abs(p[:, 0]) > 260) | (p[:, 2] < 2200) | (p[:, 2] > 6500)  # the airfield stays dark
            pts.append(p[keep])
            col.append(np.repeat(np.array([c], np.float32), keep.sum(), 0))

        for x0 in np.arange(-7000, 7001, 230):  # avenues with a gentle wander
            z = np.arange(-1500, 14000, 38.0) + rng.uniform(0, 38)
            x = x0 + 14 * np.sin(z / 700 + x0)
            add(np.c_[x, np.zeros_like(z), z], sodium if rng.uniform() < 0.7 else warm, 0.4 + 0.6 * density(x, z))
        for z0 in np.arange(-1500, 14001, 260):  # cross streets
            x = np.arange(-7000, 7001, 38.0) + rng.uniform(0, 38)
            z = z0 + 10 * np.sin(x / 500 + z0)
            add(np.c_[x, np.zeros_like(x), z], sodium if rng.uniform() < 0.6 else led, 0.4 + 0.6 * density(x, z))
        n = 50000  # windows, yards, signs: follow the density field
        x = rng.uniform(-7000, 7000, n)
        z = rng.uniform(-1500, 14000, n)
        add(np.c_[x, rng.uniform(0, 25, n), z], warm, 0.15 + 0.85 * density(x, z) ** 2)
        d = rng.normal(0, 1, (5000, 2)) * [520, 700] + [2600, 7600]  # downtown: dense, cool, some tall
        add(np.c_[d[:, 0], rng.exponential(30, 5000), d[:, 1]], led, 0.9)
        add(np.c_[d[:1500, 0], rng.exponential(60, 1500), d[:1500, 1]], cool, 0.5)
        tops = rng.normal(0, 1, (70, 2)) * [480, 620] + [2600, 7600]  # red obstruction lights on towers
        self.beacons = np.c_[tops[:, 0], rng.uniform(90, 260, 70), tops[:, 1]]

        self.highways = []
        for x0, amp, ph in [(-2600, 900, 0.3), (1500, 700, 2.1), (4300, 1100, 4.0)]:
            z = np.linspace(-1500, 14000, 900)
            x = x0 + amp * np.sin(z / 2600 + ph)
            line = np.c_[x, np.full_like(z, 6.0), z]
            seg = np.sqrt(np.sum(np.diff(line, axis=0) ** 2, axis=1))
            self.highways.append((line, np.r_[0, np.cumsum(seg)]))
            lamps = line[::3] + np.array([12.0, 4.0, 0])
            add(lamps, sodium, 1.0)
        pts = np.concatenate(pts)
        col = np.concatenate(col)
        return {"pts": pts, "col": col, "mag": rng.uniform(0.35, 1.0, len(pts)).astype(np.float32),
                "phase": rng.uniform(0, 2 * np.pi, len(pts))}


def airport_lights(A, c, wake=None, strobe=3.0):
    """Colours and brightness of the airport lights; the strobes and the beacon follow the beat."""
    R = A.runway
    kind = R["kind"]
    cols = np.zeros((len(kind), 3), np.float32)
    cols[kind == 0] = WARM
    cols[kind == 1] = WHITE
    cols[kind == 2] = [1.0, 0.18, 0.12]
    cols[kind == 3] = [0.3, 1.0, 0.5]
    cols[kind == 4] = [1.0, 0.9, 0.75]
    cols[kind == 6] = [0.25, 0.45, 1.0]
    cols[kind == 7] = [1.0, 0.85, 0.6]
    beacon_green = int(c.t / c.beat) % 2 == 0
    cols[kind == 8] = [0.3, 1.0, 0.5] if beacon_green else [1.0, 1.0, 1.0]
    seq = R["pts"][:, 2]
    rabbit = np.exp(-(((seq + 900) / 900 - c.eighth_phase) % 1.0) * 30.0) * (kind == 5)
    base = np.where(kind == 7, 0.6, np.where(kind == 6, 0.5, 1.0))
    if wake is not None:
        base = base * wake
    mags = base * (kind != 5) * (kind != 8) * (0.8 + 0.25 * c.kick_pulse) + rabbit * strobe
    mags = mags + (kind == 8) * 2.5 * np.exp(-c.beat_phase * 6)
    return cols, mags.astype(np.float32)


# ------------------------------------------------------------------ shared layers
def sky(A, cam, pitch, yaw, stops, sun=None, roll=0.0):
    """Sky by elevation angle (stops keyed by sin(elevation)), optional sun disc and halo."""
    dx, dy, dz = cam.rays(pitch, yaw, roll)
    el = dy / np.sqrt(dx * dx + dy * dy + dz * dz)
    pos = np.array([p for p, _ in stops], np.float32)
    cols = np.array([c for _, c in stops], np.float32)
    img = np.stack([np.interp(el, pos, cols[:, i]) for i in range(3)], -1).astype(np.float32)
    if sun is not None:
        sdir, power = sun
        n = np.sqrt(dx * dx + dy * dy + dz * dz)
        cosang = (dx * sdir[0] + dy * sdir[1] + dz * sdir[2]) / n
        ang = np.arccos(np.clip(cosang, -1, 1))
        halo = np.exp(-ang / 0.045) * 0.8 + np.exp(-ang / 0.25) * 0.4
        disc = gfx.smoothstep(0.0165, 0.0125, ang) * 2.4
        img += (halo[..., None] * np.array([1.0, 0.62, 0.32]) + disc[..., None] * np.array([1.0, 0.9, 0.7])) * power
    return img, el


def stars(A, img, cam, pitch, yaw, t, amount, twinkle=0.0, roll=0.0):
    """Stars on the celestial sphere; amount scales brightness, twinkle adds a beat sparkle."""
    if amount <= 0:
        return
    d = np.c_[np.cos(A.star_el) * np.sin(A.star_az), np.sin(A.star_el), np.cos(A.star_el) * np.cos(A.star_az)]
    sx, sy, z = cam.project(d * 1e4, (0, 0, 0), pitch, yaw, roll)
    vis = z > 0
    tw = 0.7 + 0.3 * np.sin(A.star_phase + t * (1.5 + 2 * A.star_mag)) + twinkle * (A.star_mag > 0.7)
    col = (A.star_col * (A.star_mag * tw * amount)[:, None]).astype(np.float32)
    gfx.splat_points(img, sx[vis], sy[vis], col[vis] * 1.6, (0.55 + 1.1 * A.star_mag[vis] ** 3))


def cloud_deck(A, cam, pos, pitch, yaw, sun_dir, light, shadow, gap, fog_col, height=0.0, scale=1 / 260.0,
               drift=(0.0, 0.0), roll=0.0, fog_dist=5000.0):
    """Sea of clouds on the plane y=height seen from pos; returns colour and coverage (alpha)."""
    dx, dy, dz = cam.rays(pitch, yaw, roll)
    X, Z, dist, valid = gfx.plane_hits(dx, dy, dz, height, pos[1])
    n = A.dens.shape[0]
    u = ((pos[0] + X) * scale + drift[0]) * n / 4
    v = ((pos[2] + Z) * scale + drift[1]) * n / 4
    near = gfx.smoothstep(2500, 600, dist)
    d = gfx.sample(A.dens, u, v) * near + gfx.sample(A.dens_far, u, v) * (1 - near)
    gx = gfx.sample(A.gx, u, v) * near
    gz = gfx.sample(A.gz, u, v) * near
    nx, ny, nz = -gx * 3.0, np.ones_like(gx), -gz * 3.0
    inv = 1 / np.sqrt(nx * nx + ny * ny + nz * nz)
    lam = np.clip((nx * sun_dir[0] + ny * sun_dir[1] + nz * sun_dir[2]) * inv, 0, 1)
    lit = gfx.lerp(np.asarray(shadow, np.float32), np.asarray(light, np.float32), lam[..., None] ** 1.5)
    col = gfx.lerp(np.asarray(gap, np.float32), lit, d[..., None])
    # forward scattering towards the sun, and aerial perspective
    ray_n = np.sqrt(dx * dx + dy * dy + dz * dz)
    fwd = np.clip((dx * sun_dir[0] + dz * sun_dir[2]) / ray_n, 0, 1) ** 6
    col += (fwd * d)[..., None] * np.asarray(light, np.float32) * 0.5
    fog = 1 - np.exp(-dist / fog_dist)
    col = gfx.lerp(col, np.asarray(fog_col, np.float32), fog[..., None])
    return col.astype(np.float32), valid.astype(np.float32)


def cloud_sea(A, cam, pos, pitch, yaw, sun_dir, light, shadow, gap, fog_col, sky_amb, roll=0.0,
              drift=(0.0, 0.0), thick=80.0, texel=2.5, fog_dist=4500.0, steps=18, reach=1800.0):
    """Sea of clouds seen from above: a ray-marched height field, so near puffs occlude each other.

    Cloud tops are y = thick * hn(x, z) over y = 0 (one texel = `texel` metres). Marching covers
    `reach` metres past the cloud-top plane; beyond that the deck is shaded as seen from afar.
    Rays that slip between puffs see the dark gap colour below.
    """
    dx, dy, dz = cam.rays(pitch, yaw, roll)
    inv = 1 / np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx * inv, dy * inv, dz * inv
    down = dy < -1e-4
    dyc = np.where(down, dy, -1e-4)
    t0 = np.maximum((thick - pos[1]) / dyc, 0)
    t_floor = -pos[1] / dyc
    t1 = np.minimum(t_floor, t0 + reach)
    hit = np.zeros(dy.shape, bool)
    t_hit = t1.copy()
    prev_t = prev_gap = None
    for k in range(steps + 1):
        t = t0 + (t1 - t0) * (k / steps)
        u = (pos[0] + dx * t) / texel + drift[0]
        v = (pos[2] + dz * t) / texel + drift[1]
        gap_now = (pos[1] + dy * t) - thick * gfx.sample(A.hn_geo, u, v)
        if prev_gap is not None:
            new = (~hit) & (gap_now <= 0) & down
            frac = prev_gap / np.maximum(prev_gap - gap_now, 1e-4)
            t_hit = np.where(new, prev_t + (t - prev_t) * frac, t_hit)
            hit |= new
        prev_t, prev_gap = t, gap_now
    far = (~hit) & down & (t_floor > t1 + 1.0)  # still above the floor: deck beyond the march
    hit |= far
    u = (pos[0] + dx * t_hit) / texel + drift[0]
    v = (pos[2] + dz * t_hit) / texel + drift[1]
    detail = gfx.smoothstep(2500, 400, t_hit)
    hn = gfx.sample(A.hn_geo, u, v)
    gx = gfx.sample(A.hgx, u, v) * thick / texel
    gz = gfx.sample(A.hgz, u, v) * thick / texel
    inv_n = 1 / np.sqrt(gx * gx + 1 + gz * gz)
    ndl = (-gx * sun_dir[0] + sun_dir[1] - gz * sun_dir[2]) * inv_n
    diff = np.clip((ndl + 0.2) / 1.2, 0, 1)
    ao = 0.4 + 0.6 * hn
    tex = 1 + 0.12 * (gfx.sample(A.hn, u * 2.0, v * 2.0) - 0.5) * detail
    col = (gfx.lerp(np.asarray(shadow, np.float32), np.asarray(light, np.float32), (diff ** 1.4)[..., None])
           * (ao * tex)[..., None] + np.asarray(sky_amb, np.float32) * (0.3 * inv_n)[..., None])
    # silver lining when looking towards a low sun, strongest on thin edges
    view_sun = np.clip(dx * sun_dir[0] + dy * sun_dir[1] + dz * sun_dir[2], 0, 1) ** 12
    col += ((view_sun * (1.15 - hn)) * 0.8)[..., None] * np.asarray(light, np.float32)
    col = np.where(hit[..., None], col, np.asarray(gap, np.float32))
    fog = 1 - np.exp(-np.where(hit, t_hit, t_floor) / fog_dist)
    col = gfx.lerp(col, np.asarray(fog_col, np.float32), fog[..., None])
    # rays that graze a crest between two march steps leave single-pixel holes; soften them
    col = gfx.blur(col.astype(np.float32), 0.8)
    return col, down.astype(np.float32)


def city_glow(A, cam, pos, pitch, yaw, roll=0.0, amount=1.0):
    """Light pollution on the ground: districts glow warm under their lights (low-res, additive)."""
    dx, dy, dz = cam.rays(pitch, yaw, roll)
    X, Z, dist, valid = gfx.plane_hits(dx, dy, dz, 0.0, pos[1])
    x, z = pos[0] + X, pos[2] + Z
    d = gfx.sample(A.city_dens, (x + 8000) / 16000 * 256, (z + 2000) / 16000 * 256)
    core = np.exp(-(((x - 2600) / 900) ** 2 + ((z - 7600) / 1200) ** 2))  # downtown
    strip = (np.abs(x) < 260) & (z > 2200) & (z < 6500)
    glow = (0.06 + 0.5 * d ** 2 + 0.6 * core) * valid * ~strip * np.exp(-dist / 9000.0)
    return (glow[..., None] * np.array([0.30, 0.15, 0.07], np.float32) * amount).astype(np.float32)


def upscale(A, low):
    return gfx.resize(low, A.w, A.h)


def lights_layer(A, img, cam, pts, cols, mags, pos, pitch, yaw, roll=0.0, size=1.0, streak=None):
    """Project point lights; near lights get bigger and optional motion streaks along the view."""
    sx, sy, z = cam.project(pts, pos, pitch, yaw, roll)
    vis = (z > 1.0) & (mags > 0.01)
    sx, sy, z, c, m = sx[vis], sy[vis], z[vis], cols[vis], mags[vis]
    sig = np.clip(size * 900.0 / z, 0.45, 9.0)
    bright = (c * (m * np.clip(1600.0 / z, 0.25, 3.0))[:, None]).astype(np.float32)
    gfx.splat_points(img, sx, sy, bright, sig)
    if streak is not None and streak > 0:
        prev = np.asarray(pos, float) - np.array([0, 0, streak])
        px, py, pz = cam.project(pts[vis], prev, pitch, yaw, roll)
        near = (pz > 1.0) & (z < 400)
        draw_lines(img, sx[near], sy[near], px[near], py[near], bright[near] * 0.12)


def light_haze(layer, amount, sigma=5.0):
    """Wide soft haze around a light layer (city glow, wet air), computed at 1/8 resolution."""
    h, w = layer.shape[:2]
    small = gfx.resize(layer, max(1, w // 8), max(1, h // 8))
    return gfx.resize(gfx.blur(small, sigma), w, h) * amount


def wet_reflection(layer, horizon, amount):
    """Vertical smears under lights below the horizon, like a rain-wet runway."""
    h, w = layer.shape[:2]
    half = gfx.resize(layer, w // 2, h // 2)
    smear = gfx.box(gfx.box(half, 12, 0), 12, 0)
    smear = np.roll(smear, 10, axis=0)
    out = gfx.resize(smear, w, h) * amount
    out[:int(max(0, horizon)) + 2] = 0
    return out


def draw_lines(img, x0, y0, x1, y1, cols, width=2):
    if len(x0) == 0:
        return
    layer = Image.new("RGB", (img.shape[1], img.shape[0]))
    dr = ImageDraw.Draw(layer)
    scale = 255.0 / max(1e-6, float(cols.max()))
    for a, b, c, d, col in zip(x0, y0, x1, y1, cols):
        dr.line([(a, b), (c, d)], fill=tuple(int(v * scale) for v in col), width=width)
    img += np.asarray(layer, np.float32) / scale


def fog_tunnel(A, t, speed, density, tint, top_light, under_light=(0.0, 0.0, 0.0)):
    """Flying through cloud: a bright, nearly opaque haze whose wisps rush outward (low-res)."""
    h, w = A.lh, A.lw
    u = (A.lxx - w / 2) / w
    v = (A.lyy - h / 2) / w
    n = A.fog_soft.shape[0]
    lum = np.zeros((h, w), np.float32)
    wsum = 0.0
    layers = 5
    for k in range(layers):
        ph = (t * speed + k / layers) % 1.0
        zoom = 0.4 + 3.0 * (1 - ph) ** 2
        a = gfx.smoothstep(0.0, 0.3, ph) * gfx.smoothstep(1.0, 0.7, ph)
        lum += a * gfx.sample(A.fog_soft, u * zoom * n + k * 97.0, v * zoom * n + k * 53.0)
        wsum += a
    lum = lum / max(wsum, 1e-3)
    vert = np.clip(1.0 - A.lyy / h, 0, 1)[..., None]
    col = (np.asarray(tint, np.float32) * (0.7 + 0.6 * lum[..., None]) * (0.75 + 0.45 * vert)
           + np.asarray(top_light, np.float32) * vert ** 2
           + np.asarray(under_light, np.float32) * (1 - vert) ** 2 * (0.6 + 0.8 * lum[..., None]))
    alpha = np.clip(density * (0.78 + 0.3 * lum), 0, 1)
    return col.astype(np.float32), alpha.astype(np.float32)


def shooting_star(img, c, area=(0.05, 0.95, 0.04, 0.45), every=2):
    """A meteor after some snares (chosen and placed by the snare time, so each one differs)."""
    age = c.t - c.last_snap
    k = int(round(c.last_snap * 100))
    if not (0 <= age < 0.65) or k % every:
        return
    rng = np.random.default_rng(k)
    h, w = img.shape[:2]
    x0, y0 = rng.uniform(area[0], area[1]) * w, rng.uniform(area[2], area[3]) * h
    ang = rng.uniform(np.radians(148), np.radians(168))
    length = rng.uniform(0.12, 0.24) * w
    head, tail = gfx.smoothstep(0, 0.3, age), gfx.smoothstep(0.12, 0.65, age)
    hx, hy = x0 + np.cos(ang) * length * head, y0 + np.sin(ang) * length * head
    tx, ty = x0 + np.cos(ang) * length * tail, y0 + np.sin(ang) * length * tail
    fade = 1 - gfx.smoothstep(0.4, 0.65, age)
    draw_lines(img, np.array([tx]), np.array([ty]), np.array([hx]), np.array([hy]),
               np.array([[0.8, 0.85, 1.0]], np.float32) * 0.9 * fade, width=2)
    gfx.splat_points(img, np.array([hx]), np.array([hy]), np.array([[1.2, 1.2, 1.4]], np.float32) * fade,
                     np.array([2.0]))


def hud(img, pitch, yaw, alpha, f, kick=0.0, spectrum=None):
    """Thin head-up display: pitch ladder, flight path marker, heading tape, a ring of live spectrum."""
    if alpha <= 0.01:
        return
    h, w = img.shape[:2]
    layer = Image.new("L", (w, h))
    dr = ImageDraw.Draw(layer)
    cx, cy = w / 2, h / 2
    if spectrum is not None:  # the instruments listen too: one tick per band around the marker
        bands = np.r_[spectrum, spectrum[::-1]]
        su = w / 1920
        for k, val in enumerate(bands):
            ang = -np.pi / 2 + k / len(bands) * 2 * np.pi
            r0, r1 = 44 * su, (52 + 52 * val) * su
            dr.line([(cx + r0 * np.cos(ang), cy + r0 * np.sin(ang)), (cx + r1 * np.cos(ang), cy + r1 * np.sin(ang))],
                    fill=int(120 + 120 * val), width=2)
    for deg in range(-20, 25, 5):
        y = cy + np.tan(pitch - np.radians(deg)) * f
        if not (40 < y < h - 40):
            continue
        half = 150 if deg == 0 else 70
        gap = 0 if deg == 0 else 26
        dr.line([(cx - half - gap, y), (cx - gap, y)], fill=200, width=2)
        dr.line([(cx + gap, y), (cx + half + gap, y)], fill=200, width=2)
    r = 14 + 4 * kick
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=2)
    dr.line([(cx - r - 26, cy), (cx - r, cy)], fill=255, width=2)
    dr.line([(cx + r, cy), (cx + r + 26, cy)], fill=255, width=2)
    dr.line([(cx, cy - r - 12), (cx, cy - r)], fill=255, width=2)
    heading = (np.degrees(yaw) + 270) % 360
    for k in range(-6, 7):
        hd = int(heading // 5 * 5 + k * 5)
        x = cx + (hd - heading) * 14
        tick = 14 if hd % 10 == 0 else 7
        dr.line([(x, 40), (x, 40 + tick)], fill=170, width=2)
    a = np.asarray(layer, np.float32)[..., None] / 255 * alpha
    img += a * CYAN * 1.3


# ------------------------------------------------------------------ scenes
def runway(A, c):
    """Intro and first verse: night runway, lights waking one by one, the roll begins."""
    p = c.p
    speed = 0.0 if c.tag == "Intro" else gfx.smoothstep(0.0, 1.0, p) ** 1.6
    z = 0.0 if c.tag == "Intro" else 900 * gfx.smoothstep(0, 1, p) ** 2.2 + 18 * c.tau
    pos = (0.0, 3.2, -420 + z)
    pitch = np.radians(-1.2 + 0.4 * np.sin(c.t * 0.3))
    yaw = np.radians(0.6 * np.sin(c.t * 0.21))
    low, _ = sky(A, A.lcam, pitch, yaw, NIGHT_SKY)
    img = upscale(A, low)
    hy = A.cam.horizon_y(pitch)
    yy = np.arange(A.h, dtype=np.float32)[:, None, None]
    img += np.exp(-np.abs(yy - hy) / 26.0) * np.array([0.22, 0.12, 0.08], np.float32) * (0.8 + 0.4 * c.low)
    img[int(max(0, hy)):] *= 0.35
    stars(A, img, A.cam, pitch, yaw, c.t, 0.5, roll=0.0)

    R = A.runway
    kind = R["kind"]
    order = np.clip(R["pts"][:, 2] / 3000, 0, 1)
    if c.tag == "Intro":  # lights switch on along the strip, one step per beat
        wake = np.clip((c.beat_count - order * 14) * 0.9, 0, 1)
    else:
        wake = np.ones(len(order))
    cols, mags = airport_lights(A, c, wake)
    L = np.zeros_like(img)
    if c.tag != "Intro" or c.beat_count > 3:  # packets of light race out along the edges, one per beat
        ph = c.beat_phase
        zc = pos[2] + 25 + ph ** 0.7 * 2200 - np.arange(6) * 35
        for x_edge in (-22.5, 22.5):
            pts = np.c_[np.full(6, x_edge), np.full(6, 0.6), zc]
            glow = (CYAN[None] * 4.0 * (1 - np.arange(6) / 6)[:, None] * (1 - ph) ** 0.5).astype(np.float32)
            lights_layer(A, L, A.cam, pts, glow, np.ones(6, np.float32), pos, pitch, yaw, size=0.6)
    lights_layer(A, L, A.cam, R["pts"], cols, mags.astype(np.float32), pos, pitch, yaw,
                 size=0.3, streak=8 + 70 * speed if speed > 0.3 else None)
    img += L + wet_reflection(L, hy, 0.55) + light_haze(L, 0.35)
    if c.tag != "Intro":  # rain, lit from below by the runway
        rng = np.random.default_rng(9)
        n = 380
        sp = rng.uniform(900, 1500, n)
        y = (rng.uniform(0, A.h, n) + c.t * sp) % (A.h + 80) - 40
        x = (rng.uniform(0, A.w, n) - c.t * sp * 0.18) % A.w
        ln = rng.uniform(14, 34, n) * (A.w / 1920)
        br = rng.uniform(0.03, 0.10, n) * (0.6 + 0.8 * y / A.h) * gfx.smoothstep(0.0, 0.15, c.p)
        draw_lines(img, x, y, x + ln * 0.18, y - ln, np.c_[br, br, br * 1.1].astype(np.float32), width=1)

    # the line: faint threads joining consecutive edge lights that are awake (signal path)
    edge = np.where(kind == 0)[0]
    sx, sy, zz = A.cam.project(R["pts"][edge], pos, pitch, yaw)
    ok = (zz > 2) & (wake[edge] > 0.5)
    for side in (0, 1):
        idx = np.where(ok & (np.arange(len(edge)) % 2 == side))[0]
        if len(idx) > 1:
            draw_lines(img, sx[idx[:-1]], sy[idx[:-1]], sx[idx[1:]], sy[idx[1:]],
                       np.tile(CYAN * 0.10 * (0.6 + 0.4 * c.snap_pulse), (len(idx) - 1, 1)), width=1)
    # distant city as out-of-focus bokeh along the horizon
    rng = np.random.default_rng(21)
    bx = rng.uniform(-200, A.w + 200, 70) + (c.t * 6) % 40
    by = hy + rng.normal(6, 5, 70)
    bc = np.array([1.0, 0.6, 0.3], np.float32) * rng.uniform(0.04, 0.14, (70, 1)).astype(np.float32)
    gfx.splat_points(img, bx, by, bc * (0.8 + 0.2 * np.sin(c.t + rng.uniform(0, 6, (70, 1)))),
                     rng.uniform(4, 11, 70))
    c.bloom_strength = 0.9
    c.punch = 0.006
    return img


def climb(A, c):
    """Pre-chorus: rotate, lift off, sink into grey; the instruments stay crisp."""
    p = c.p
    pitch = np.radians(2 + 7 * gfx.smoothstep(0.0, 0.4, p))
    alt = 3 + 900 * gfx.smoothstep(0.05, 1.0, p) ** 1.5
    pos = (0.0, alt, 1000 + 120 * c.tau)
    yaw = np.radians(0.8 * np.sin(c.t * 0.4))
    roll = np.radians(1.5 * np.sin(c.t * 0.7)) + np.radians(0.6) * c.kick_pulse * np.sin(c.t * 40)
    low, _ = sky(A, A.lcam, pitch, yaw, CITY_SKY, roll=roll)
    low += city_glow(A, A.lcam, pos, pitch, yaw, roll, 1.0 - 0.6 * p)
    img = upscale(A, low)
    stars(A, img, A.cam, pitch, yaw, c.t, 0.35 * (1 - p), roll=roll)
    R = A.runway
    cols, mags = airport_lights(A, c)
    lights_layer(A, img, A.cam, R["pts"], cols, mags * (1 - p), pos, pitch, yaw, roll, size=0.35)
    cy = A.city
    lights_layer(A, img, A.cam, cy["pts"], cy["col"], cy["mag"] * 0.5 * (1 - p), pos, pitch, yaw, roll, size=0.6)
    dens = gfx.smoothstep(0.12, 0.7, p)
    fcol, acc = fog_tunnel(A, c.t, 0.35 + 0.25 * p, 1.4 * dens, (0.16, 0.18, 0.24), (0.10, 0.10, 0.14))
    img = gfx.lerp(img, upscale(A, fcol), upscale(A, acc[..., None]))
    rng = np.random.default_rng(4)  # speed lines streaming past inside the cloud
    n = 80
    ang = rng.uniform(0, 2 * np.pi, n)
    ph = (rng.uniform(0, 1, n) + c.t * rng.uniform(0.8, 1.6, n)) % 1.0
    r0 = (0.1 + ph * 0.9) * A.w * 0.55
    r1 = r0 + (0.04 + 0.12 * ph) * A.w
    a = dens * (1 - gfx.smoothstep(0.85, 1.0, p)) * 0.16 * np.sin(np.pi * ph)
    draw_lines(img, A.w / 2 + np.cos(ang) * r0, A.h / 2 + np.sin(ang) * r0 * 0.6, A.w / 2 + np.cos(ang) * r1,
               A.h / 2 + np.sin(ang) * r1 * 0.6, np.c_[a, a, a * 1.1].astype(np.float32), width=1)
    hud(img, pitch, yaw, gfx.smoothstep(0.2, 0.5, p), A.cam.f, c.kick_pulse, c.spectrum)
    flash = gfx.smoothstep(0.88, 1.0, p)  # the cloud top breaks open
    img += flash ** 2 * np.array([1.4, 1.0, 0.7], np.float32)
    c.bloom_strength = 0.7
    c.punch = 0.008
    return img


def _above(A, c, sky_stops, sun_el, sun_az, light, shadow, gap, fog_col, extra_pitch=0.0, yaw=0.0):
    sun_dir = np.array([np.cos(sun_el) * np.sin(sun_az), np.sin(sun_el), np.cos(sun_el) * np.cos(sun_az)])
    pitch = np.radians(-6.5 + extra_pitch)
    pos = (0.0, 150.0 + 8 * np.sin(c.t * 0.25), 2000 + c.t * 38)
    roll = np.radians(1.2 * np.sin(c.t * 0.18))
    low, el = sky(A, A.lcam, pitch, yaw, sky_stops, sun=(sun_dir, 0.9 + 0.35 * c.kick_pulse), roll=roll)
    ccol, cov = cloud_sea(A, A.lcam, pos, pitch, yaw, sun_dir, light, shadow, gap, fog_col,
                          sky_amb=(0.10, 0.08, 0.20), roll=roll, drift=(c.t * 0.8, 0.0))
    low = gfx.lerp(low, ccol, cov[..., None])
    img = upscale(A, low)
    sx, sy, sz = A.cam.project(sun_dir[None] * 1e5, (0, 0, 0), pitch, yaw, roll)
    if sz[0] > 0:  # shafts of light from the sun through the haze, and the lens catching it
        img += gfx.god_rays(img, sx[0], sy[0], threshold=1.0, strength=0.45 + 0.35 * c.kick_pulse)
        on_screen = gfx.smoothstep(-0.2, 0.05, min(sx[0], A.w - sx[0]) / A.w)
        gfx.lens_flare(img, sx[0], sy[0], (0.7 + 0.5 * c.kick_pulse) * on_screen)
    return img, pos, pitch, roll, sun_dir


def above(A, c):
    """Chorus: above the clouds at sunset; answers float up like light, light as stars."""
    second = c.occurrence > 0
    sun_el = np.radians(3.5 if not second else 1.8)
    img, pos, pitch, roll, sun_dir = _above(
        A, c, SUNSET_SKY, sun_el, np.radians(18), light=(1.25, 0.78, 0.50), shadow=(0.26, 0.17, 0.32),
        gap=(0.16, 0.10, 0.20), fog_col=(0.95, 0.55, 0.38))
    stars(A, img, A.cam, pitch, 0.0, c.t, 0.25 + 0.15 * second, twinkle=0.6 * c.snap_pulse, roll=roll)
    # answers: drifting motes of light; on the third line some sink but fade above the clouds
    P = A.answers
    rel_z = (P[:, 2] - c.t * 22) % 900 + 20
    rise = (c.t * (6 + 10 * P[:, 3])) % 160
    sink = gfx.smoothstep(0.45, 0.8, c.p) * (P[:, 3] > 0.6)
    y = np.where(sink > 0, P[:, 1] - sink * 120 * (c.p - 0.45), P[:, 1] + rise - 40)
    pts = np.c_[P[:, 0], pos[1] - 140 + y, pos[2] + rel_z]
    fade = gfx.smoothstep(pos[1] - 140 + 2, pos[1] - 140 + 30, pts[:, 1])  # never touch the cloud tops
    sx, sy, z = A.cam.project(pts, pos, pitch, 0.0, roll)
    vis = z > 5
    tw = 0.65 + 0.35 * np.sin(P[:, 3] * 40 + c.t * 3) + 0.8 * c.snap_pulse * (P[:, 3] > 0.5)
    col = GOLD[None] * (tw * fade * np.clip(300 / z, 0.15, 1.6))[:, None]
    gfx.splat_points(img, sx[vis], sy[vis], col[vis].astype(np.float32) * 1.2, np.clip(380 / z[vis], 0.6, 3.5))
    # nearby answers join into constellations, brighter on the snare
    near = vis & (z < 700) & (fade > 0.3)
    px, py = sx[near], sy[near]
    if len(px) > 1:
        reach = 170 * A.w / 1920
        d = np.hypot(px[:, None] - px[None], py[:, None] - py[None])
        i, j = np.where(np.triu((d < reach) & (d > 4), 1))
        i, j = i[:240], j[:240]
        a = (1 - d[i, j] / reach) * 0.16 * (0.6 + 0.9 * c.snap_pulse)
        draw_lines(img, px[i], py[i], px[j], py[j], (GOLD[None] * a[:, None]).astype(np.float32), width=1)
    shooting_star(img, c)
    c.bloom_strength = 0.75
    c.streak = 0.35 + 0.35 * c.kick_pulse
    c.punch = 0.010
    return img


def holding(A, c):
    """Second verse: circling at dusk, drawing the same loop again; the wind ignores it."""
    yaw = c.tau * 0.05
    img, pos, pitch, roll, sun_dir = _above(
        A, c, DUSK_SKY, np.radians(1.0), np.radians(-40), light=(0.95, 0.52, 0.42), shadow=(0.15, 0.10, 0.24),
        gap=(0.10, 0.07, 0.15), fog_col=(0.62, 0.32, 0.36), extra_pitch=-3.0 - 2.0 * gfx.smoothstep(0.85, 1, c.p),
        yaw=yaw)
    stars(A, img, A.cam, pitch, yaw, c.t, 0.55, twinkle=0.5 * c.snap_pulse, roll=roll)
    # loops traced on a layer just above the clouds; older loops fade, later the wind smears them
    centre = np.array([pos[0], 25.0, pos[2] + 900.0])
    loops = int(c.tau / (c.beat * 4)) + 1
    smear = gfx.smoothstep(0.5, 0.95, c.p)
    layer_pts, layer_col = [], []
    for k in range(max(0, loops - 6), loops):
        prog = min(1.0, c.tau / (c.beat * 4) - k)
        ang = np.linspace(0, 2 * np.pi * prog, 160)
        rx, rz = 520 + 40 * k, 260 + 18 * k
        px = centre[0] + rx * np.cos(ang) + smear * 160 * np.sin(ang * 3 + c.t)
        pz = centre[2] + rz * np.sin(ang) + smear * 90 * np.cos(ang * 2 - c.t)
        layer_pts.append(np.c_[px, np.full_like(px, centre[1]) + 5 * k, pz])
        age = loops - 1 - k
        layer_col.append(np.tile(GOLD * (0.55 * 0.72 ** age) * (1 - 0.8 * smear), (len(px), 1)))
    # a faint chart grid lies on the cloud tops, the map that the wind will not read
    for axis in (0, 2):
        for off in np.arange(-3000, 3001, 500):
            run = np.linspace(-3000, 3000, 40)
            pts = np.zeros((40, 3))
            pts[:, axis] = np.round((pos[axis] + off) / 500) * 500
            pts[:, 2 - axis] = pos[2 - axis] + run
            pts[:, 1] = 30.0
            sx, sy, z = A.cam.project(pts, pos, pitch, yaw, roll)
            ok = (z[:-1] > 20) & (z[1:] > 20)
            fade = (np.exp(-z[:-1] / 2500.0) * 0.05 * (1 - 0.7 * smear))[:, None]
            draw_lines(img, sx[:-1][ok], sy[:-1][ok], sx[1:][ok], sy[1:][ok], (CYAN[None] * fade[ok]).astype(np.float32),
                       width=1)
    for pts, col in zip(layer_pts, layer_col):
        sx, sy, z = A.cam.project(pts, pos, pitch, yaw, roll)
        ok = z[:-1] > 5
        draw_lines(img, sx[:-1][ok], sy[:-1][ok], sx[1:][ok], sy[1:][ok], col[:-1][ok], width=2)
    if layer_pts:  # the pen: a comet at the tip of the loop being drawn
        hx, hy, hz = A.cam.project(layer_pts[-1][-1:], pos, pitch, yaw, roll)
        if hz[0] > 5:
            gfx.splat_points(img, hx, hy, (GOLD * 2.5 * (1 - 0.7 * smear))[None].astype(np.float32), np.array([2.5]))
            gfx.lens_flare(img, hx[0], hy[0], 0.35 * (1 - smear) * (0.7 + 0.6 * c.kick_pulse))
    # wind: long thin streaks sweeping across the view
    rng = np.random.default_rng(3)
    n = 90
    wx = (rng.uniform(0, 1, n) * A.w * 1.4 + c.t * rng.uniform(260, 520, n)) % (A.w * 1.4) - A.w * 0.2
    wy = rng.uniform(A.h * 0.15, A.h * 0.95, n)
    ln = rng.uniform(60, 220, n)
    draw_lines(img, wx, wy, wx - ln, wy + ln * 0.04,
               np.tile(np.array([0.5, 0.45, 0.6], np.float32) * (0.10 + 0.08 * c.high), (n, 1)), width=1)
    shooting_star(img, c, every=3)
    c.bloom_strength = 0.7
    c.streak = 0.25
    c.punch = 0.008
    return img


def river_stone(A, c):
    """Bridge: a river that keeps flowing, a stone that keeps its shape; then each returns home."""
    h, w = A.h, A.w
    img = gfx.gradient(h, w, [(0, (0.01, 0.01, 0.04)), (0.62, (0.03, 0.03, 0.09)), (1, (0.015, 0.012, 0.03))])
    stars(A, img, A.cam, np.radians(18), 0.0, c.t, 0.35 + 0.6 * gfx.smoothstep(0.5, 0.75, c.p))
    hy = h * 0.70
    img[int(hy):int(hy) + 2] += np.array([0.12, 0.10, 0.16], np.float32)
    P = A.river
    x = (P[:, 0] + c.t * 0.035 * P[:, 2]) % 1.0
    base = hy - 90 + 55 * np.sin(x * 7.0 + c.t * 0.6) + 22 * np.sin(x * 17 - c.t * 0.9)
    y = base + P[:, 1] * (16 + 6 * np.sin(P[:, 3] + c.t))
    # "give the wings back to the sky": each mote lets go at its own moment, left to right
    t_go = 0.5 + 0.2 * P[:, 0] + 0.06 * np.sin(P[:, 3] * 5)
    lift = gfx.smoothstep(t_go, t_go + 0.22, c.p)
    e = lift ** 1.4
    tx = ((P[:, 0] * 0.62 + P[:, 3] / (2 * np.pi) * 0.9) % 1.0) * w
    ty = h * (0.05 + 0.55 * ((P[:, 2] - 0.5) % 1.0))
    sx = gfx.lerp(x * (w * 0.62), tx, e) + 70 * np.sin(P[:, 3] * 3 + c.t * 0.7) * e * (1 - e) * 4
    y = gfx.lerp(y, ty, e)
    col = gfx.lerp(np.array([0.25, 0.75, 1.0], np.float32)[None], np.array([0.95, 0.92, 1.0], np.float32)[None],
                   lift[:, None])
    twinkle = 0.65 + 0.35 * np.sin(c.t * 3 + P[:, 3] * 7)
    bright = (0.5 + 0.25 * c.vocal + 0.5 * c.snap_pulse * (P[:, 3] > 4.5)) * gfx.lerp(1.0, 0.7 * twinkle, lift)
    gfx.splat_points(img, sx, y, (col * (bright * (0.6 + 0.4 * P[:, 2]))[:, None]).astype(np.float32),
                     0.9 + 0.5 * lift)
    # the stone: a slowly turning icosahedron that lands on the horizon on the last line
    drop = gfx.smoothstep(0.76, 0.93, c.p)
    cx, cy = w * 0.76, gfx.lerp(h * 0.40, hy - 70, drop ** 2)
    R = gfx.Camera._rot(c.t * 0.12, c.t * 0.21, 0.3)
    v = A.ico_v @ R.T
    layer = Image.new("RGB", (w, h))
    dr = ImageDraw.Draw(layer)
    light = np.array([0.4, 0.7, -0.6]) / np.linalg.norm([0.4, 0.7, -0.6])
    faces = sorted(A.ico_f, key=lambda f: -v[list(f), 2].mean())
    for f in faces:
        n = v[list(f)].mean(0)
        n /= np.linalg.norm(n)
        if n[2] > 0:
            continue
        shade = 0.12 + 0.55 * max(0.0, float(n @ light))
        poly = [(cx + 95 * v[i, 0], cy - 95 * v[i, 1]) for i in f]
        tone = gfx.lerp(np.array([0.30, 0.34, 0.48]), GOLD, drop) * shade
        edge = tone * (1.6 + 1.2 * c.kick_pulse) + 0.08
        dr.polygon(poly, fill=tuple(int(255 * min(1, t)) for t in tone),
                   outline=tuple(int(255 * min(1, t)) for t in edge))
    # a fine orbit around the stone: steady, exact, turning with the beat
    rr = 150 * A.w / 1920
    dr.ellipse([cx - rr, cy - rr * 0.32, cx + rr, cy + rr * 0.32], outline=(60, 66, 90), width=1)
    orb = c.t / c.beat * np.pi / 2
    ox, oy = cx + rr * np.cos(orb), cy + rr * 0.32 * np.sin(orb)
    dr.ellipse([ox - 3, oy - 3, ox + 3, oy + 3], fill=(230, 220, 200))
    img += np.asarray(layer, np.float32) / 255
    landed = c.t - (c.start + c.dur * 0.93)
    if landed > 0:  # ripple where the weight meets the ground, and the dust it throws up
        rr = 40 + landed * 420
        ring = Image.new("L", (w, h))
        ImageDraw.Draw(ring).ellipse([cx - rr, hy - rr * 0.12, cx + rr, hy + rr * 0.12], outline=255, width=2)
        img += (np.asarray(ring, np.float32) / 255 * np.exp(-landed * 1.6))[..., None] * GOLD
        rng = np.random.default_rng(8)
        ang = rng.uniform(np.pi, 2 * np.pi, 240)
        speed = rng.uniform(80, 420, 240)
        bx = cx + np.cos(ang) * speed * landed
        by = hy + np.sin(ang) * speed * landed * 0.35 + 60 * landed ** 2
        gfx.splat_points(img, bx, by, np.tile(GOLD * 0.5 * np.exp(-landed * 1.4), (240, 1)).astype(np.float32),
                         np.full(240, 1.1))
    c.bloom_strength = 0.85
    c.punch = 0.0
    return img


def descent(A, c):
    """Pre-chorus 2: gear down through cloud; holes open onto the city below, which beats."""
    p = c.p
    shake = c.kick_pulse * 0.9
    pitch = np.radians(-14 + 1.2 * shake * np.sin(c.t * 37))
    roll = np.radians(1.4 * shake * np.sin(c.t * 29) + 2 * np.sin(c.t * 0.5))
    yaw = np.radians(1.0 * np.sin(c.t * 0.3))
    pos = (0.0, 1500 - 500 * p, -1800 + 90 * c.tau)
    low, _ = sky(A, A.lcam, pitch, yaw, CITY_SKY, roll=roll)
    low += city_glow(A, A.lcam, pos, pitch, yaw, roll, 3.5 * (1 + 0.6 * c.kick_pulse))
    img = upscale(A, low) * 0.8
    cy = A.city
    beat = 1.0 + 0.9 * c.kick_pulse
    L = np.zeros_like(img)
    lights_layer(A, L, A.cam, cy["pts"], cy["col"], 4.0 * cy["mag"] * beat, pos, pitch, yaw, roll, size=0.45)
    img += L + light_haze(L, 2.4, 7.0)
    open_amt = gfx.smoothstep(0.25, 0.95, p)
    fcol, acc = fog_tunnel(A, c.t, 0.3, 1.5, (0.11, 0.12, 0.17), (0.05, 0.05, 0.08),
                           under_light=np.array([0.30, 0.15, 0.07]) * (0.4 + open_amt) * (1 + 0.5 * c.kick_pulse))
    # soft, slowly drifting breaks in the cloud; even an open break keeps a veil of mist
    if c.kick_pulse > 0.05:  # each kick lights the cloud from inside, somewhere new
        rng = np.random.default_rng(int(round(c.last_kick * 100)))
        gx, gy = rng.uniform(0.15, 0.85) * A.lw, rng.uniform(0.25, 0.8) * A.lh
        r2 = ((A.lxx - gx) ** 2 + (A.lyy - gy) ** 2) / (A.lw * 0.2) ** 2
        fcol = fcol + (np.exp(-r2) * c.kick_pulse * 0.5)[..., None] * np.array([1.0, 0.75, 0.5], np.float32)
    holes = gfx.sample(A.fog_soft, A.lxx * 0.7 + c.t * 6.0, A.lyy * 0.7 - c.t * 3.0)
    # breaks open in the upper part of the view, where the far city is bright; near ground stays veiled
    reach = gfx.smoothstep(A.lh * 0.92, A.lh * 0.5, A.lyy)
    alpha = acc * (1 - 0.8 * reach * gfx.smoothstep(1.0 - open_amt, 1.45 - open_amt, holes))
    img = gfx.lerp(img, upscale(A, fcol), upscale(A, alpha[..., None]))
    hud(img, pitch, yaw, 0.8 * (1 - open_amt), A.cam.f, c.kick_pulse, c.spectrum)
    c.bloom_strength = 0.9
    c.punch = 0.012
    return img


def city(A, c):
    """Final chorus: under the clouds the lights are heavy and close; the approach, the touchdown."""
    p = c.p
    land = gfx.smoothstep(0.0, 0.92, p)
    alt = gfx.lerp(420.0, 2.6, land ** 1.25)
    pos = (0.0, alt, gfx.lerp(-1400.0, 3300.0, land ** 1.1))
    pitch = np.radians(gfx.lerp(-11.0, -1.5, land ** 0.8))
    touchdown = c.t - (c.start + c.dur * 0.92)
    bump = np.exp(-max(0.0, touchdown) * 6) * (touchdown > 0)
    roll = np.radians(1.2 * np.sin(c.t * 0.4) * (1 - land) + 0.8 * bump * np.sin(c.t * 50))
    yaw = np.radians(0.8 * np.sin(c.t * 0.27) * (1 - land))
    low, el = sky(A, A.lcam, pitch, yaw, CITY_SKY, roll=roll)
    # the cloud ceiling above, lit from below by the city
    ccol, cov = cloud_deck(A, A.lcam, pos, pitch, yaw, np.array([0, -1.0, 0]), (0.30, 0.16, 0.10),
                           (0.07, 0.05, 0.07), (0.02, 0.02, 0.05), (0.10, 0.06, 0.06), height=900.0,
                           scale=1 / 420.0, roll=roll, fog_dist=6000)
    low = gfx.lerp(low, ccol, (cov * 0.9 * gfx.smoothstep(0.03, 0.2, el))[..., None])
    low += city_glow(A, A.lcam, pos, pitch, yaw, roll, 1.2 * (1 + 0.4 * c.kick_pulse))
    img = upscale(A, low)
    cy = A.city
    ripple_r = ((c.t - c.last_kick) * 2600.0)
    dist = np.sqrt(cy["pts"][:, 0] ** 2 + (cy["pts"][:, 2] - 3000) ** 2)
    ring = np.exp(-((dist - ripple_r) / 220.0) ** 2) * np.exp(-(c.t - c.last_kick) * 2.5)
    mags = 4.0 * cy["mag"] * (0.85 + 0.08 * np.sin(cy["phase"] + c.t * 2) + 1.2 * ring + 0.35 * c.kick_pulse)
    L = np.zeros_like(img)
    lights_layer(A, L, A.cam, cy["pts"], cy["col"], mags.astype(np.float32), pos, pitch, yaw, roll, size=0.45)
    img += L + light_haze(L, 1.6, 6.0)
    blink = 3.0 * np.exp(-c.beat_phase * 5)  # obstruction lights on the towers, on the beat
    lights_layer(A, img, A.cam, A.beacons, np.tile(np.array([1.0, 0.12, 0.08], np.float32), (len(A.beacons), 1)),
                 np.full(len(A.beacons), blink, np.float32), pos, pitch, yaw, roll, size=0.4)
    traffic(A, img, c, pos, pitch, yaw, roll)
    R = A.runway
    cols, mags = airport_lights(A, c, strobe=4.0)
    L = np.zeros_like(img)
    lights_layer(A, L, A.cam, R["pts"] + np.array([0, 0, 3000.0]), cols, mags.astype(np.float32), pos, pitch,
                 yaw, roll, size=0.35, streak=30 * land if land > 0.6 else None)
    img += L + wet_reflection(L, A.cam.horizon_y(pitch), 0.4 * land)
    img += bump * np.array([0.5, 0.35, 0.2], np.float32)
    if bump > 0.02:
        gfx.lens_flare(img, A.w / 2, A.h * 0.62, 1.6 * bump)
    c.bloom_strength = 1.0
    c.streak = 0.3
    c.punch = 0.014
    return img


def traffic(A, img, c, pos, pitch, yaw, roll):
    """Head- and tail-lights streaming along the highways; they surge a little on each kick."""
    rng = np.random.default_rng(17)
    for line, cum in A.highways:
        n = 160
        s0 = rng.uniform(0, cum[-1], n)
        speed = rng.uniform(22, 34, n) * (1 + 0.25 * c.kick_pulse)
        for direction, colour, offset in ((1, (1.0, 0.95, 0.85), -5.0), (-1, (1.0, 0.12, 0.08), 5.0)):
            s = (s0 + direction * speed * c.t) % cum[-1]
            x = np.interp(s, cum, line[:, 0]) + offset
            z = np.interp(s, cum, line[:, 2])
            pts = np.c_[x, np.full(n, 6.5), z]
            cols = np.tile(np.array(colour, np.float32), (n, 1))
            lights_layer(A, img, A.cam, pts, cols, np.full(n, 1.6, np.float32), pos, pitch, yaw, roll, size=0.35)


def ground(A, c):
    """Outro: rolled to a stop, looking up; the runway lines up again under the open sky."""
    p = c.p
    pitch = np.radians(gfx.lerp(-1.0, 32.0, gfx.smoothstep(0.1, 0.8, p)))
    pos = (0.0, 2.4, 3800 + 20 * gfx.smoothstep(0, 0.3, p))
    low, _ = sky(A, A.lcam, pitch, 0.0, NIGHT_SKY)
    img = upscale(A, low)
    stars(A, img, A.cam, pitch, 0.0, c.t, 0.6 + 0.6 * gfx.smoothstep(0.2, 0.8, p), twinkle=0.6 * c.snap_pulse)
    R = A.runway
    cols, mags = airport_lights(A, c)
    L = np.zeros_like(img)
    lights_layer(A, L, A.cam, R["pts"] + np.array([0, 0, 3000.0]), cols, mags, pos, pitch, 0.0, size=0.35)
    img += L + wet_reflection(L, A.cam.horizon_y(pitch), 0.45) + light_haze(L, 0.3)
    shooting_star(img, c, area=(0.05, 0.95, 0.02, 0.55), every=1)
    c.bloom_strength = 0.8
    c.punch = 0.004
    return img


SCENES = {"runway": runway, "climb": climb, "above": above, "holding": holding, "river_stone": river_stone,
          "descent": descent, "city": city, "ground": ground}
FLASH_INTO = {"above", "city"}  # scenes that open with a burst of light
