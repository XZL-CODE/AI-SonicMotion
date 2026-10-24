"""Frame-by-frame renderer for a song: scenes per section, beat-driven light, lyric subtitles.

Inputs are the song's song.json (sections, scenes, lyrics + translations), its scenes.py
(one drawing function per scene name) and the analysis json from engine/music/analyze.py
(beats, kicks, per-frame energy, word timing).

Usage:
    python3 engine/video/render.py 1-ground-truth 1-ground-truth/outputs/analysis/take_7.json
    python3 engine/video/render.py ... --stills 12,48,95        # PNG previews at those seconds
    python3 engine/video/render.py ... --start 40 --end 70      # render a clip
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / ".tools" / "fonts"
FONTS = {  # OFL fonts from the Google Fonts repository
    "serif": "ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf",
    "serif_italic": "ofl/cormorantgaramond/CormorantGaramond-Italic%5Bwght%5D.ttf",
    "cjk": "ofl/notoserifsc/NotoSerifSC%5Bwght%5D.ttf",
}
TRANSITION = 1.2  # seconds of crossfade between sections


FALLBACK_FONTS = {  # used when the OFL fonts cannot be downloaded (offline, blocked network)
    "serif": ["/System/Library/Fonts/Supplemental/Baskerville.ttc", "/System/Library/Fonts/Supplemental/Didot.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
    "serif_italic": ["/System/Library/Fonts/Supplemental/Baskerville.ttc",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
    "cjk": ["/System/Library/Fonts/Supplemental/Songti.ttc", "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"],
}


def font_path(key):
    path = FONT_DIR / Path(FONTS[key]).name.replace("%5B", "[").replace("%5D", "]")
    if not path.exists():
        FONT_DIR.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/google/fonts/main/" + FONTS[key]
        print("downloading font", path.name, flush=True)
        try:
            urllib.request.urlretrieve(url, path)
        except OSError as err:
            path.unlink(missing_ok=True)
            for alt in FALLBACK_FONTS[key]:
                if Path(alt).exists():
                    print(f"  download failed ({err}); using {alt}", flush=True)
                    return alt
            raise
    return str(path)


def load_font(key, size, weight):
    f = ImageFont.truetype(font_path(key), size)
    try:
        f.set_variation_by_axes([weight])
    except OSError:
        pass  # a static fallback font has no weight axis
    return f


@dataclass
class FrameCtx:
    t: float
    start: float
    dur: float
    tag: str
    occurrence: int
    beat: float
    beat_count: float
    beat_phase: float
    eighth_phase: float
    kick_pulse: float
    snap_pulse: float
    last_kick: float
    last_snap: float
    rms: float
    low: float
    high: float
    vocal: float
    spectrum: object = None  # 24 log-spaced band levels 0..1 for this frame (None for old analyses)
    bloom_strength: float = 0.8
    streak: float = 0.0
    punch: float = 0.008  # zoom kick on each bass drum hit, set per scene

    @property
    def tau(self):
        return self.t - self.start

    @property
    def p(self):
        return float(np.clip(self.tau / max(self.dur, 1e-3), 0, 1))


class Timeline:
    def __init__(self, song, an):
        self.song, self.an = song, an
        self.secs = an["sections"]
        self.beats = np.array(an["beats"])
        self.kicks = np.array(an["kicks"] or [-10.0])
        self.snaps = np.array(an["snaps"] or [-10.0])
        self.frames = {k: np.array(v, np.float32) for k, v in an["frames"].items()}
        self.spec = self.frames.pop("spectrum", None)
        self.fps = an["fps"]
        self.beat = float(np.median(np.diff(self.beats))) if len(self.beats) > 8 else 60.0 / an["tempo"]
        seen = {}
        self.occ = []
        for s in self.secs:
            self.occ.append(seen.get(s["scene"], 0))
            seen[s["scene"]] = self.occ[-1] + 1

    def feature(self, name, t):
        arr = self.frames[name]
        return float(arr[min(len(arr) - 1, max(0, int(t * self.fps)))])

    @staticmethod
    def _pulse(times, t, decay):
        i = np.searchsorted(times, t, side="right") - 1
        return (float(np.exp(-(t - times[i]) / decay)), float(times[i])) if i >= 0 else (0.0, -10.0)

    def ctx(self, i, t):
        s = self.secs[i]
        b = np.searchsorted(self.beats, t, side="right") - 1
        if 0 <= b < len(self.beats) - 1:
            phase = (t - self.beats[b]) / (self.beats[b + 1] - self.beats[b])
        else:
            phase = ((t - (self.beats[0] if len(self.beats) else 0)) / self.beat) % 1.0
        kick, last_kick = self._pulse(self.kicks, t, 0.16)
        snap, last_snap = self._pulse(self.snaps, t, 0.10)
        return FrameCtx(t=t, start=s["start"], dur=s["end"] - s["start"], tag=s["tag"], occurrence=self.occ[i],
                        beat=self.beat, beat_count=(t - s["start"]) / self.beat, beat_phase=phase,
                        eighth_phase=(phase * 2) % 1.0, kick_pulse=kick, snap_pulse=snap, last_kick=last_kick,
                        last_snap=last_snap,
                        rms=self.feature("rms", t), low=self.feature("low", t), high=self.feature("high", t),
                        vocal=self.feature("vocal", t), spectrum=self.spectrum(t))

    def spectrum(self, t):
        if self.spec is None:
            return None
        return self.spec[min(len(self.spec) - 1, max(0, int(t * self.fps)))]

    def blend(self, t):
        """[(section index, weight)] — one section, or two during a crossfade."""
        for i, s in enumerate(self.secs):
            if s["start"] <= t < s["end"] or i == len(self.secs) - 1:
                if i > 0 and t - s["start"] < TRANSITION / 2:
                    w = gfx.smoothstep(-TRANSITION / 2, TRANSITION / 2, t - s["start"])
                    return [(i - 1, 1 - w), (i, w)]
                if i + 1 < len(self.secs) and self.secs[i + 1]["start"] - t < TRANSITION / 2:
                    w = gfx.smoothstep(-TRANSITION / 2, TRANSITION / 2, t - self.secs[i + 1]["start"])
                    return [(i, 1 - w), (i + 1, w)]
                return [(i, 1.0)]
        return [(0, 1.0)]


def ease_out_back(q, k=1.6):
    """0..1 ease with a small overshoot: a word dropped into place settles like a landing."""
    q = q - 1
    return 1 + (k + 1) * q ** 3 + k * q ** 2


class Typesetter:
    """Text layers: subtitles in the lower letterbox bar (words light up as sung), hook lines set
    large in the picture (each word lands as it is sung), chapter marks and title cards."""

    def __init__(self, W, H, bar, song, an, scale):
        self.W, self.H, self.bar = W, H, bar
        self.song, self.lines = song, an["lines"]
        self.sections = an["sections"]
        self.duration = an["duration"]
        s = scale
        self.en = load_font("serif", int(42 * s), 500)
        self.zh = load_font("cjk", int(27 * s), 400)
        self.title = load_font("serif", int(96 * s), 400)
        self.title_zh = load_font("cjk", int(40 * s), 300)
        self.small = load_font("serif_italic", int(26 * s), 400)
        self.chapter = load_font("serif", int(30 * s), 500)
        self.chapter_zh = load_font("cjk", int(22 * s), 400)
        self.scale = s
        self.starts = [ln["start"] for ln in self.lines]
        # display windows: a line fades in a little before it is sung and has fully faded
        # before the next one appears, so two lines never share the bar
        self.windows = []
        prev_out = -1.0
        for k, ln in enumerate(self.lines):
            nxt = self.starts[k + 1] if k + 1 < len(self.lines) else self.duration
            t0 = max(ln["start"] - 0.45, prev_out)
            t1 = max(t0 + 0.4, min(nxt - 0.55, ln["end"] + 2.2))
            self.windows.append((t0, t1))
            prev_out = t1 + 0.3
        self.hook_fonts = {}

    def _line(self, ln):
        return self.song["sections"][ln["section"]]["lines"][ln["line"]]

    def _spaced(self, draw, xy, text, font, fill, spacing, anchor_center=True):
        widths = [draw.textlength(ch, font=font) for ch in text]
        total = sum(widths) + spacing * (len(text) - 1)
        x = xy[0] - total / 2 if anchor_center else xy[0]
        for ch, wch in zip(text, widths):
            draw.text((x, xy[1]), ch, font=font, fill=fill)
            x += wch + spacing

    def draw(self, frame, t):
        W, H = self.W, self.H
        layer = Image.new("RGBA", (W, H))
        dr = ImageDraw.Draw(layer)
        glow = Image.new("L", (W, H))
        gd = ImageDraw.Draw(glow)
        shade = Image.new("L", (W, H))
        sd = ImageDraw.Draw(shade)
        for k, ln in enumerate(self.lines):
            t0, t1 = self.windows[k]
            if t0 <= t <= t1 + 0.3:
                a = gfx.smoothstep(t0, t0 + 0.3, t) * (1 - gfx.smoothstep(t1, t1 + 0.3, t))
                self._subtitle(dr, gd, ln, t, a)
            if self._line(ln).get("hook") and t0 <= t <= t1 + 0.9:
                self._hook(dr, gd, sd, ln, t, t1)
        self._chapters(dr, t)
        self._titles(dr, gd, t)
        arr = np.asarray(frame, np.float32)
        sh = np.asarray(shade, np.float32)
        if sh.max() > 0:  # soft shadow under large type keeps it readable over bright skies
            small = gfx.blur(np.asarray(Image.fromarray(sh).resize((W // 4, H // 4), Image.BILINEAR),
                                        np.float32)[..., None], 3.0)[..., 0]
            sh = np.asarray(Image.fromarray(small).resize((W, H), Image.BILINEAR), np.float32) / 255
            arr = arr * (1 - 0.55 * sh[..., None])
        g = np.asarray(glow, np.float32)
        if g.max() > 0:
            small = Image.fromarray(g).resize((W // 4, H // 4), Image.BILINEAR)
            gb = gfx.blur(np.asarray(small, np.float32)[..., None], 2.5)[..., 0]
            gb = np.asarray(Image.fromarray(gb).resize((W, H), Image.BILINEAR), np.float32) / 255
            arr = arr + gb[..., None] * np.array([255, 170, 90]) * 0.8
        frame_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        frame_img.paste(layer, (0, 0), layer)
        return np.asarray(frame_img)

    def _subtitle(self, dr, gd, ln, t, a):
        W, H, bar, s = self.W, self.H, self.bar, self.scale
        line = self._line(ln)
        if line.get("hook"):  # the picture carries the English; the bar keeps the translation
            self._spaced(dr, (W / 2, H - bar + bar * 0.36), line["zh"], self.zh, (230, 222, 214, int(215 * a)),
                         3 * s)
            return
        words = ln["words"]
        gap = self.en.getlength(" ")
        widths = [self.en.getlength(w["w"]) for w in words]
        x = W / 2 - (sum(widths) + gap * (len(words) - 1)) / 2
        y_en = H - bar + bar * 0.18
        for w, wd in zip(words, widths):
            sung = gfx.smoothstep(w["start"] - 0.05, w["start"] + 0.12, t)
            now = sung * (1 - gfx.smoothstep(w["end"], w["end"] + 0.5, t))
            level = (0.42 + 0.58 * sung) * a
            col = (255, int(236 + 10 * now), int(214 + 20 * now), int(255 * level))
            dr.text((x, y_en), w["w"], font=self.en, fill=col)
            if now > 0.01:
                gd.text((x, y_en), w["w"], font=self.en, fill=int(200 * now * a))
            x += wd + gap
        self._spaced(dr, (W / 2, H - bar + bar * 0.58), line["zh"], self.zh, (230, 222, 214, int(205 * a)), 2 * s)

    def _hook_font(self, words):
        """Largest size (96 px down to 60 px at 1080p) at which the line fits in one or two rows."""
        text = " ".join(w["w"].upper() for w in words)
        for size in (96, 88, 80, 72, 64, 60):
            if size not in self.hook_fonts:
                self.hook_fonts[size] = load_font("serif", int(size * self.scale), 600)
            f = self.hook_fonts[size]
            spacing = 0.08 * size * self.scale
            width = f.getlength(text) + spacing * len(text)
            if width < self.W * 0.82:
                return f, spacing, [words]
            if width < self.W * 1.5 and size <= 80:
                half = len(words) // 2 + len(words) % 2
                return f, spacing, [words[:half], words[half:]]
        return f, spacing, [words[:len(words) // 2], words[len(words) // 2:]]

    def _hook(self, dr, gd, sd, ln, t, t_out):
        """Hook line set large in the picture; each word drops in and settles as it is sung."""
        W, s = self.W, self.scale
        font, spacing, rows = self._hook_font(ln["words"])
        fade = 1 - gfx.smoothstep(t_out, t_out + 0.9, t)
        drift = gfx.smoothstep(t_out, t_out + 0.9, t)
        size = font.size
        y0 = self.bar + (self.H - 2 * self.bar) * 0.40 - len(rows) * size * 0.62 - drift * 26 * s
        for r, row in enumerate(rows):
            texts = [w["w"].upper() for w in row]
            widths = [font.getlength(x) + spacing * len(x) for x in texts]
            gap = font.getlength(" ") + spacing * 2
            x = W / 2 - (sum(widths) + gap * (len(row) - 1)) / 2
            y = y0 + r * size * 1.18
            for w, text, wd in zip(row, texts, widths):
                q = float(np.clip((t - (w["start"] - 0.14)) / 0.45, 0, 1))
                if q <= 0:
                    x += wd + gap
                    continue
                e = ease_out_back(q)
                yy = y - (1 - e) * 70 * s
                alpha = gfx.smoothstep(0, 0.35, q) * fade
                land = float(np.exp(-max(0.0, t - w["start"]) / 0.3)) if t >= w["start"] - 0.05 else 0.0
                self._spaced(sd, (x, yy + 6 * s), text, font, int(200 * alpha), spacing, anchor_center=False)
                self._spaced(dr, (x, yy), text, font, (255, 244, 228, int(250 * alpha)), spacing,
                             anchor_center=False)
                self._spaced(gd, (x, yy), text, font, int((90 + 160 * land) * alpha), spacing, anchor_center=False)
                x += wd + gap

    def _chapters(self, dr, t):
        """Chapter mark at the top left as a section begins: number, a line drawing out, the name."""
        s = self.scale
        for sec, info in zip(self.sections, self.song["sections"]):
            ch = info.get("chapter")
            if not ch:
                continue
            t0 = sec["start"] + 0.6
            if not (t0 <= t <= t0 + 6.5):
                continue
            a = gfx.smoothstep(t0, t0 + 0.6, t) * (1 - gfx.smoothstep(t0 + 5.2, t0 + 6.4, t))
            x, y = 64 * s, self.bar + 40 * s
            dr.text((x, y), ch["no"], font=self.small, fill=(255, 225, 190, int(230 * a)))
            grow = gfx.smoothstep(t0, t0 + 1.2, t)
            dr.line([(x + 44 * s, y + 17 * s), (x + 44 * s + 90 * s * grow, y + 17 * s)],
                    fill=(255, 225, 190, int(200 * a)), width=max(1, int(2 * s)))
            reveal = gfx.smoothstep(t0 + 0.3, t0 + 1.4, t)
            name = ch["en"][:max(0, int(round(len(ch["en"]) * reveal)))]
            self._spaced(dr, (x + 150 * s, y), name, self.chapter, (255, 244, 230, int(240 * a)), 6 * s,
                         anchor_center=False)
            self._spaced(dr, (x + 150 * s, y + 42 * s), ch["zh"], self.chapter_zh,
                         (235, 225, 215, int(200 * a * reveal)), 6 * s, anchor_center=False)

    def _titles(self, dr, gd, t):
        W, H, s = self.W, self.H, self.scale
        first_line = self.starts[0] if self.starts else 12.0
        a = gfx.smoothstep(1.8, 4.0, t) * (1 - gfx.smoothstep(first_line - 3.2, first_line - 1.4, t))
        # the end card waits until the last hook line has left the picture
        hook_out = [w[1] + 0.9 for ln, w in zip(self.lines, self.windows) if self._line(ln).get("hook")]
        t_end = min(max([self.duration - 14] + hook_out), self.duration - 4.0)
        end = gfx.smoothstep(t_end, t_end + 1.5, t) * (1 - gfx.smoothstep(self.duration - 1.2, self.duration - 0.1, t))
        for alpha, credits in ((a, False), (end, True)):
            if alpha <= 0.01:
                continue
            cy = H * 0.33
            self._spaced(dr, (W / 2, cy - 70 * s), self.song["title"].upper(), self.title,
                         (255, 240, 225, int(235 * alpha)), 18 * s + 22 * s * (1 - alpha))
            # a band of light sweeps across the title while it is up
            sweep = (t * 0.35) % 1.0
            self._spaced(gd, (W / 2, cy - 70 * s), self.song["title"].upper(), self.title,
                         int((110 + 90 * np.exp(-((sweep - 0.5) / 0.12) ** 2)) * alpha), 18 * s + 22 * s * (1 - alpha))
            self._spaced(dr, (W / 2, cy + 52 * s), self.song["title_zh"], self.title_zh,
                         (240, 225, 210, int(210 * alpha)), 26 * s)
            if credits:
                self._spaced(dr, (W / 2, cy + 130 * s), "words, music & every frame made with code and AI",
                             self.small, (220, 210, 200, int(170 * alpha)), 1 * s)
                self._spaced(dr, (W / 2, cy + 168 * s), "AI 生成内容", self.zh, (220, 210, 200, int(150 * alpha)), 4 * s)


# per-process state for the worker pool
G = {}


def load_scenes(work):
    """The song's own scenes.py: Assets, SCENES (scene name -> drawing function), optional FLASH_INTO."""
    spec = importlib.util.spec_from_file_location("scenes", Path(work) / "scenes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def init_worker(work, analysis, W, H, scale):
    song = json.loads((Path(work) / "song.json").read_text(encoding="utf-8"))
    an = json.loads(Path(analysis).read_text(encoding="utf-8"))
    scenes = load_scenes(work)
    bar = int(round((H - W / 2.35) / 2))
    G.update(song=song, an=an, W=W, H=H, bar=bar, scale=scale, tl=Timeline(song, an), scenes=scenes,
             A=scenes.Assets(W, H - 2 * bar), ts=Typesetter(W, H, bar, song, an, scale))


def zoom(img8, f):
    """Scale an image up by f about its centre (keeps the size), for the beat punch."""
    h, w = img8.shape[:2]
    cw, ch = w / f, h / f
    box = ((w - cw) / 2, (h - ch) / 2, (w + cw) / 2, (h + ch) / 2)
    return np.asarray(Image.fromarray(img8).resize((w, h), Image.BILINEAR, box=box))


def render_frame(t):
    A, tl, scenes = G["A"], G["tl"], G["scenes"]
    pic, main = None, None
    blend = tl.blend(t)
    for i, w in blend:
        if w < 1e-3:
            continue
        c = tl.ctx(i, t)
        img = scenes.SCENES[tl.secs[i]["scene"]](A, c)
        img = gfx.bloom(img, 0.75, c.bloom_strength * (0.85 + 0.3 * c.kick_pulse), c.streak)
        pic = img * w if pic is None else pic + img * w
        if main is None or w > main[1]:
            main = (c, w)
    c = main[0]
    for sec in tl.secs:  # a burst of light where the song opens into a new scene
        if sec["scene"] in getattr(scenes, "FLASH_INTO", ()) and 0 <= t - sec["start"] < 0.9:
            pic += 0.5 * np.exp(-(t - sec["start"]) / 0.2) * np.array([1.0, 0.9, 0.78], np.float32)
    if len(blend) == 2:  # a warm light leak sweeps across every transition
        w2 = blend[1][1]
        x = np.linspace(0, 1, A.w, dtype=np.float32)
        leak = np.exp(-((x - (1.3 * w2 - 0.15)) / 0.22) ** 2) * 4 * w2 * (1 - w2) * 0.35
        pic += leak[None, :, None] * np.array([1.0, 0.62, 0.32], np.float32)
    pic = gfx.tone(pic) * A.vig
    grain = A.grain[int(t * 24) % len(A.grain)]
    pic += gfx.resize(grain, A.w, A.h) * 0.010
    dur = tl.an["duration"]
    pic *= gfx.smoothstep(0.0, 1.6, t) * (1 - gfx.smoothstep(dur - 3.0, dur - 0.2, t))
    pic8 = np.clip(pic * 255 + 0.5, 0, 255).astype(np.uint8)
    # beat punch: the picture leans in on each kick, with a slight colour fringe
    if c.punch * c.kick_pulse > 0.0015:
        pic8 = zoom(pic8, 1 + c.punch * c.kick_pulse)
    shift = int(round(c.kick_pulse * c.punch * 300))
    if shift:
        pic8 = pic8.copy()
        pic8[..., 0] = np.roll(pic8[..., 0], shift, 1)
        pic8[..., 2] = np.roll(pic8[..., 2], -shift, 1)
    out = np.zeros((G["H"], G["W"], 3), np.uint8)
    out[G["bar"]:G["bar"] + A.h] = pic8
    return G["ts"].draw(out, t)[..., :3].tobytes()


def main():
    ap = argparse.ArgumentParser(description="Render the music video")
    ap.add_argument("work", help="song directory containing song.json and scenes.py")
    ap.add_argument("analysis", help="analysis json from engine/music/analyze.py")
    ap.add_argument("--audio", help="audio to mux (default: the analysed take)")
    ap.add_argument("--out", help="output mp4 (default: <song>/outputs/<take>.mp4)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--stills", help="comma-separated seconds; writes PNGs instead of a video")
    ap.add_argument("--workers", type=int, default=0, help="processes (default: all cores)")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality: 18 near-lossless, 23 smaller file")
    args = ap.parse_args()

    an = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    W = args.width
    H = int(round(W * 9 / 16 / 2)) * 2
    scale = W / 1920
    work = str(Path(args.work).resolve())
    take = Path(args.analysis).stem
    out_dir = Path(work) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stills:
        init_worker(work, args.analysis, W, H, scale)
        for s in args.stills.split(","):
            t = float(s)
            img = np.frombuffer(render_frame(t), np.uint8).reshape(H, W, 3)
            path = out_dir / "stills" / f"{take}_{t:06.1f}.png"
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(img).save(path)
            print(path, flush=True)
        return

    end = args.end if args.end else an["duration"]
    times = np.arange(args.start, end, 1.0 / args.fps)
    out = Path(args.out) if args.out else out_dir / f"{take}.mp4"
    audio = args.audio or str(Path(work) / an["audio"])  # analyze.py stores it relative to the song
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(args.fps), "-i", "-", "-ss", str(args.start), "-t", str(end - args.start), "-i", audio,
           "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
           "-af", "volume=-1dB", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    workers = args.workers or None
    with Pool(workers, initializer=init_worker, initargs=(work, args.analysis, W, H, scale)) as pool:
        for k, data in enumerate(pool.imap(render_frame, times, chunksize=2)):
            ff.stdin.write(data)
            if k % 150 == 0:
                print(f"frame {k}/{len(times)}  t={times[k]:.1f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
