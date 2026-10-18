"""Rendering primitives: tileable noise, perspective camera, light splats, bloom, grading.

Images are float32 RGB arrays shaped (H, W, 3) in display space; values above 1 are
allowed until the final tone curve, so additive light can glow instead of clipping.
"""
import numpy as np
from PIL import Image, ImageFilter


def fbm_texture(size, beta, seed):
    """Tileable fractal noise from a 1/f^beta spectrum, normalised to zero mean, unit std."""
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(size)[:, None]
    ky = np.fft.fftfreq(size)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1.0
    spec = (rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))) / k ** beta
    spec[0, 0] = 0
    tex = np.real(np.fft.ifft2(spec))
    return ((tex - tex.mean()) / tex.std()).astype(np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def sample(tex, x, y):
    """Bilinear lookup with wrap-around; x, y in texel units (any shape)."""
    n_y, n_x = tex.shape[:2]
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    x0 = x0.astype(np.int64) % n_x
    y0 = y0.astype(np.int64) % n_y
    x1 = (x0 + 1) % n_x
    y1 = (y0 + 1) % n_y
    if tex.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    top = tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx
    bottom = tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def lerp(a, b, t):
    return a + (b - a) * t


def gradient(h, w, stops):
    """Vertical gradient; stops = [(position 0..1, (r, g, b)), ...] with colours in 0..1."""
    pos = np.array([p for p, _ in stops], np.float32)
    cols = np.array([c for _, c in stops], np.float32)
    v = np.linspace(0, 1, h, dtype=np.float32)
    col = np.stack([np.interp(v, pos, cols[:, i]) for i in range(3)], -1)
    return np.repeat(col[:, None, :], w, axis=1)


class Camera:
    """Pinhole camera; world axes x right, y up, z forward."""

    def __init__(self, w, h, fov_deg):
        self.w, self.h = w, h
        self.f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
        u = (np.arange(w, dtype=np.float32) - w / 2 + 0.5) / self.f
        v = (np.arange(h, dtype=np.float32) - h / 2 + 0.5) / self.f
        self.u, self.v = np.meshgrid(u, v)

    @staticmethod
    def _rot(pitch, yaw, roll):
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cr, sr = np.cos(roll), np.sin(roll)
        r_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        r_pitch = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
        r_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        return r_yaw @ r_pitch @ r_roll

    def rays(self, pitch=0.0, yaw=0.0, roll=0.0):
        """World-space ray directions for every pixel: three (H, W) arrays."""
        r = self._rot(pitch, yaw, roll).astype(np.float32)
        x, y, z = self.u, -self.v, np.ones_like(self.u)
        return (r[0, 0] * x + r[0, 1] * y + r[0, 2] * z,
                r[1, 0] * x + r[1, 1] * y + r[1, 2] * z,
                r[2, 0] * x + r[2, 1] * y + r[2, 2] * z)

    def project(self, pts, pos, pitch=0.0, yaw=0.0, roll=0.0):
        """World points (N, 3) -> screen x, y and camera depth (N,)."""
        r = self._rot(pitch, yaw, roll)
        cam = (np.asarray(pts, np.float64) - np.asarray(pos)) @ r  # world -> camera (r is orthonormal)
        z = cam[:, 2]
        safe = np.where(z > 1e-3, z, 1e-3)
        sx = self.w / 2 + cam[:, 0] / safe * self.f
        sy = self.h / 2 - cam[:, 1] / safe * self.f
        return sx, sy, z

    def horizon_y(self, pitch):
        return self.h / 2 + np.tan(pitch) * self.f


def plane_hits(dx, dy, dz, height, cam_y):
    """Intersections of rays with the plane y = height: world (x, z) offsets and distance."""
    t = (height - cam_y) / np.where(np.abs(dy) > 1e-5, dy, 1e-5)
    valid = t > 0
    t = np.where(valid, t, 0)
    return dx * t, dz * t, t, valid


def splat_points(img, xs, ys, colors, sizes):
    """Additive soft points; sizes are sigma in pixels. Small points are binned for speed."""
    h, w = img.shape[:2]
    keep = (xs > -20) & (xs < w + 20) & (ys > -20) & (ys < h + 20)
    xs, ys, colors, sizes = xs[keep], ys[keep], colors[keep], sizes[keep]
    small = sizes < 1.2
    if small.any():
        xi = np.clip(xs[small].astype(int), 0, w - 1)
        yi = np.clip(ys[small].astype(int), 0, h - 1)
        flat = img.reshape(-1, 3)
        np.add.at(flat, yi * w + xi, colors[small])
    for x, y, c, s in zip(xs[~small], ys[~small], colors[~small], sizes[~small]):
        r = int(3 * s) + 1
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        gx = np.exp(-0.5 * ((np.arange(x0, x1) - x) / s) ** 2)
        gy = np.exp(-0.5 * ((np.arange(y0, y1) - y) / s) ** 2)
        img[y0:y1, x0:x1] += (gy[:, None, None] * gx[None, :, None]) * c


def box(a, r, axis):
    """Running-mean box filter of radius r along one axis (edge padded)."""
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float32)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def blur(img, sigma):
    """Gaussian blur approximated by three box passes per axis (float precision, any channels)."""
    r = max(1, int(round((np.sqrt(4 * sigma * sigma + 1) - 1) / 2)))
    out = img
    for axis in (0, 1):
        for _ in range(3):
            out = box(out, r, axis)
    return out


def resize(img, w, h):
    """Bilinear resize of an (H, W, C) float image (area-averaged when shrinking)."""
    chans = [np.asarray(Image.fromarray(np.ascontiguousarray(img[..., i], dtype=np.float32), "F")
                        .resize((w, h), Image.BILINEAR)) for i in range(img.shape[2])]
    return np.stack(chans, -1)


def bloom(img, threshold=0.8, strength=0.6, streak=0.0):
    """Soft glow from bright areas plus an optional anamorphic horizontal streak."""
    h, w = img.shape[:2]
    small = np.clip(resize(img, w // 4, h // 4) - threshold, 0, None)
    glow = strength * (0.6 * blur(small, 3) + 0.4 * blur(small, 12))
    if streak > 0:
        # squash horizontally (area-averaged) and stretch back: bright points smear sideways
        strip = resize(resize(small, w // 48, h // 4), w // 4, h // 4)
        glow += streak * strip * np.array([0.55, 0.75, 1.0], np.float32)
    return img + resize(glow, w, h)


def tone(img):
    """Filmic shoulder (ACES-like fit) so stacked light rolls off softly instead of clipping."""
    x = np.clip(img, 0, None) * 0.9
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def vignette(h, w, strength=0.35):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    return (1 - strength * smoothstep(0.4, 1.4, r))[..., None].astype(np.float32)
