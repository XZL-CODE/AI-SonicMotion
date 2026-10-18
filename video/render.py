"""Frame-by-frame renderer for a work: scenes per section, beat-driven light, lyric subtitles.

Inputs are the work's song.json (sections, scenes, lyrics + translations) and the analysis
json from music/analyze.py (beats, kicks, per-frame energy, word timing).

Usage:
    python3 video/render.py works/ground-truth outputs/ground-truth/analysis/take_7.json
    python3 video/render.py ... --stills 12,48,95        # PNG previews at those seconds
    python3 video/render.py ... --start 40 --end 70      # render a clip
"""
import argparse
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
import scenes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / ".tools" / "fonts"
FONTS = {  # OFL fonts from the Google Fonts repository
    "serif": "ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf",
    "serif_italic": "ofl/cormorantgaramond/CormorantGaramond-Italic%5Bwght%5D.ttf",
    "cjk": "ofl/notoserifsc/NotoSerifSC%5Bwght%5D.ttf",
}
TRANSITION = 1.2  # seconds of crossfade between sections


def font_path(key):
    path = FONT_DIR / Path(FONTS[key]).name.replace("%5B", "[").replace("%5D", "]")
    if not path.exists():
        FONT_DIR.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/google/fonts/main/" + FONTS[key]
        print("downloading font", path.name, flush=True)
        urllib.request.urlretrieve(url, path)
    return str(path)


def load_font(key, size, weight):
    f = ImageFont.truetype(font_path(key), size)
    f.set_variation_by_axes([weight])
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
    rms: float
    low: float
    high: float
    vocal: float
    bloom_strength: float = 0.8
    streak: float = 0.0

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
        snap, _ = self._pulse(self.snaps, t, 0.10)
        return FrameCtx(t=t, start=s["start"], dur=s["end"] - s["start"], tag=s["tag"], occurrence=self.occ[i],
                        beat=self.beat, beat_count=(t - s["start"]) / self.beat, beat_phase=phase,
                        eighth_phase=(phase * 2) % 1.0, kick_pulse=kick, snap_pulse=snap, last_kick=last_kick,
                        rms=self.feature("rms", t), low=self.feature("low", t), high=self.feature("high", t),
                        vocal=self.feature("vocal", t))

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


class Typesetter:
    """Lyric subtitles in the lower letterbox bar (words light up as sung) and title cards."""

    def __init__(self, W, H, bar, song, an, scale):
        self.W, self.H, self.bar = W, H, bar
        self.song, self.lines = song, an["lines"]
        self.duration = an["duration"]
        s = scale
        self.en = load_font("serif", int(42 * s), 500)
        self.zh = load_font("cjk", int(27 * s), 400)
        self.title = load_font("serif", int(96 * s), 400)
        self.title_zh = load_font("cjk", int(40 * s), 300)
        self.small = load_font("serif_italic", int(26 * s), 400)
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

    def _spaced(self, draw, xy, text, font, fill, spacing, anchor_center=True):
        widths = [draw.textlength(ch, font=font) for ch in text]
        total = sum(widths) + spacing * (len(text) - 1)
        x = xy[0] - total / 2 if anchor_center else xy[0]
        for ch, wch in zip(text, widths):
            draw.text((x, xy[1]), ch, font=font, fill=fill)
            x += wch + spacing

    def draw(self, frame, t):
        W, H, bar, s = self.W, self.H, self.bar, self.scale
        layer = Image.new("RGBA", (W, H))
        dr = ImageDraw.Draw(layer)
        glow = Image.new("L", (W, H))
        gd = ImageDraw.Draw(glow)
        for k, ln in enumerate(self.lines):
            t0, t1 = self.windows[k]
            if not (t0 <= t <= t1 + 0.3):
                continue
            a = gfx.smoothstep(t0, t0 + 0.3, t) * (1 - gfx.smoothstep(t1, t1 + 0.3, t))
            sec = self.song["sections"][ln["section"]]["lines"][ln["line"]]
            words = ln["words"]
            gap = self.en.getlength(" ")
            widths = [self.en.getlength(w["w"]) for w in words]
            x = W / 2 - (sum(widths) + gap * (len(words) - 1)) / 2
            y_en = H - bar + bar * 0.18
            for w, wd in zip(words, widths):
                sung = gfx.smoothstep(w["start"] - 0.05, w["start"] + 0.12, t)
                now = gfx.smoothstep(w["start"] - 0.05, w["start"] + 0.1, t) * (1 - gfx.smoothstep(w["end"], w["end"] + 0.5, t))
                level = (0.42 + 0.58 * sung) * a
                col = (255, int(236 + 10 * now), int(214 + 20 * now), int(255 * level))
                dr.text((x, y_en), w["w"], font=self.en, fill=col)
                if now > 0.01:
                    gd.text((x, y_en), w["w"], font=self.en, fill=int(200 * now * a))
                x += wd + gap
            self._spaced(dr, (W / 2, H - bar + bar * 0.58), sec["zh"], self.zh, (230, 222, 214, int(205 * a)),
                         2 * s)
        self._titles(dr, gd, t)
        frame_img = Image.fromarray(frame)
        g = np.asarray(glow, np.float32)
        if g.max() > 0:
            small = Image.fromarray(g).resize((W // 4, H // 4), Image.BILINEAR)
            gb = gfx.blur(np.asarray(small, np.float32)[..., None], 2.5)[..., 0]
            gb = np.asarray(Image.fromarray(gb).resize((W, H), Image.BILINEAR), np.float32) / 255
            arr = np.asarray(frame_img, np.float32) + gb[..., None] * np.array([255, 170, 90]) * 0.8
            frame_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        frame_img.paste(layer, (0, 0), layer)
        return np.asarray(frame_img)

    def _titles(self, dr, gd, t):
        W, H, s = self.W, self.H, self.scale
        first_line = self.starts[0] if self.starts else 12.0
        a = gfx.smoothstep(1.8, 4.0, t) * (1 - gfx.smoothstep(first_line - 3.2, first_line - 1.4, t))
        end = gfx.smoothstep(self.duration - 14, self.duration - 11, t) * (1 - gfx.smoothstep(self.duration - 2.5,
                                                                                               self.duration - 0.6, t))
        for alpha, credits in ((a, False), (end, True)):
            if alpha <= 0.01:
                continue
            cy = H * 0.33
            self._spaced(dr, (W / 2, cy - 70 * s), self.song["title"].upper(), self.title,
                         (255, 240, 225, int(235 * alpha)), 18 * s + 22 * s * (1 - alpha))
            self._spaced(gd, (W / 2, cy - 70 * s), self.song["title"].upper(), self.title, int(110 * alpha),
                         18 * s + 22 * s * (1 - alpha))
            self._spaced(dr, (W / 2, cy + 52 * s), self.song["title_zh"], self.title_zh,
                         (240, 225, 210, int(210 * alpha)), 26 * s)
            if credits:
                self._spaced(dr, (W / 2, cy + 130 * s), "words, music & every frame made with code and AI",
                             self.small, (220, 210, 200, int(170 * alpha)), 1 * s)
                self._spaced(dr, (W / 2, cy + 168 * s), "AI 生成内容", self.zh, (220, 210, 200, int(150 * alpha)), 4 * s)


# per-process state for the worker pool
G = {}


def init_worker(work, analysis, W, H, scale):
    song = json.loads((Path(work) / "song.json").read_text(encoding="utf-8"))
    an = json.loads(Path(analysis).read_text(encoding="utf-8"))
    bar = int(round((H - W / 2.35) / 2))
    G.update(song=song, an=an, W=W, H=H, bar=bar, scale=scale, tl=Timeline(song, an),
             A=scenes.Assets(W, H - 2 * bar), ts=Typesetter(W, H, bar, song, an, scale))


def render_frame(t):
    A, tl = G["A"], G["tl"]
    pic = None
    for i, w in tl.blend(t):
        if w < 1e-3:
            continue
        c = tl.ctx(i, t)
        img = scenes.SCENES[tl.secs[i]["scene"]](A, c)
        img = gfx.bloom(img, 0.75, c.bloom_strength * (0.85 + 0.3 * c.kick_pulse), c.streak)
        pic = img * w if pic is None else pic + img * w
    pic = gfx.tone(pic) * A.vig
    grain = A.grain[int(t * 24) % len(A.grain)]
    pic += gfx.resize(grain, A.w, A.h) * 0.010
    dur = tl.an["duration"]
    pic *= gfx.smoothstep(0.0, 1.6, t) * (1 - gfx.smoothstep(dur - 3.0, dur - 0.2, t))
    frame = np.zeros((G["H"], G["W"], 3), np.float32)
    frame[G["bar"]:G["bar"] + A.h] = pic
    out = np.clip(frame * 255 + 0.5, 0, 255).astype(np.uint8)
    return G["ts"].draw(out, t)[..., :3].tobytes()


def main():
    ap = argparse.ArgumentParser(description="Render the music video")
    ap.add_argument("work")
    ap.add_argument("analysis")
    ap.add_argument("--audio", help="audio to mux (default: the analysed take)")
    ap.add_argument("--out", help="output mp4 (default: outputs/<work>/<take>.mp4)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--stills", help="comma-separated seconds; writes PNGs instead of a video")
    ap.add_argument("--workers", type=int, default=0, help="processes (default: all cores)")
    args = ap.parse_args()

    an = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    W = args.width
    H = int(round(W * 9 / 16 / 2)) * 2
    scale = W / 1920
    work = str(Path(args.work).resolve())
    take = Path(args.analysis).stem
    out_dir = ROOT / "outputs" / Path(work).name
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
    audio = args.audio or an["audio"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(args.fps), "-i", "-", "-ss", str(args.start), "-t", str(end - args.start), "-i", audio,
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(out)]
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
