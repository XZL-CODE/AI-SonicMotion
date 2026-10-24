"""Analyse a rendered take for the video: beats, kicks, per-frame energy, lyric timing.

Lyric timing comes from faster-whisper word timestamps aligned to the known lyrics in
song.json with a global (Needleman-Wunsch) alignment, so repeated choruses map in order
and words the recogniser misses are interpolated from their neighbours.

Usage:
    python3 engine/music/analyze.py 1-ground-truth 1-ground-truth/outputs/takes/take_7.wav
"""
import argparse
import difflib
import json
import os
import re
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SR = 22050
HOP = 512
WHISPER = os.environ.get("WHISPER_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")  # repo id or local dir


def norm_word(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def band_flux(S, freqs, lo, hi):
    band = S[(freqs >= lo) & (freqs < hi)].sum(0)
    band = np.log1p(band)
    flux = np.maximum(0, np.diff(band, prepend=band[0]))
    return flux / (flux.max() + 1e-9)


def pick_peaks(env, times, min_gap, thresh):
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=8, post_avg=8,
                                   delta=thresh, wait=int(min_gap * SR / HOP))
    return [round(float(times[p]), 3) for p in peaks]


def per_frame(values, times, fps, duration):
    grid = np.arange(int(duration * fps)) / fps
    v = np.interp(grid, times, values)
    hi = np.percentile(v, 98) + 1e-9
    return np.round(np.clip(v / hi, 0, 1), 3).tolist()


def rhythm(y, fps, duration):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=SR, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, start_bpm=64, tightness=200)
    beats = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)
    kick_env = band_flux(S, freqs, 30, 150)
    snap_env = band_flux(S, freqs, 2500, 9000)
    # downbeat phase: the beat offset (mod 4) carrying the most kick energy
    kick_at = np.interp(beats, times, kick_env)
    phase = int(np.argmax([kick_at[p::4].mean() if len(kick_at[p::4]) else 0 for p in range(4)]))
    rms = librosa.feature.rms(S=S)[0]
    bands = {name: S[(freqs >= lo) & (freqs < hi)].sum(0) for name, (lo, hi) in
             {"low": (30, 200), "mid": (200, 2500), "high": (2500, 11000)}.items()}
    vocal = S[(freqs >= 300) & (freqs < 3400)].sum(0)
    frames = {"rms": per_frame(rms, times, fps, duration),
              "kick": per_frame(kick_env, times, fps, duration),
              "snap": per_frame(snap_env, times, fps, duration),
              "vocal": per_frame(vocal, times, fps, duration)}
    frames.update({k: per_frame(v, times, fps, duration) for k, v in bands.items()})
    edges = np.geomspace(40, 11000, 25)  # 24 log-spaced bands for the instrument rings
    spectrum = [per_frame(np.log1p(10 * S[(freqs >= lo) & (freqs < hi)].mean(0)), times, fps, duration)
                for lo, hi in zip(edges[:-1], edges[1:])]
    frames["spectrum"] = np.round(np.array(spectrum).T, 2).tolist()
    return {
        "tempo": round(float(np.atleast_1d(tempo)[0]), 2),
        "beats": [round(float(b), 3) for b in beats],
        "downbeats": [round(float(b), 3) for b in beats[phase::4]],
        "kicks": pick_peaks(kick_env, times, 0.25, 0.12),
        "snaps": pick_peaks(snap_env, times, 0.12, 0.10),
        "frames": frames,
    }


def transcribe(path):
    from faster_whisper import WhisperModel

    wav, _ = librosa.load(path, sr=16000, mono=True)
    model = WhisperModel(WHISPER, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(wav, language="en", beam_size=5, word_timestamps=True,
                                   condition_on_previous_text=False, vad_filter=False)
    words = []
    for seg in segments:
        for w in seg.words or []:
            if norm_word(w.word):
                words.append({"w": norm_word(w.word), "start": float(w.start), "end": float(w.end)})
    return words


def align(lyric, heard):
    """Global alignment of lyric words to recognised words; returns lyric index -> heard index."""
    n, m = len(lyric), len(heard)

    def score(a, b):
        if a == b:
            return 2.0
        return 1.0 if difflib.SequenceMatcher(None, a, b).ratio() >= 0.75 else -1.0

    gap = -0.6
    D = np.zeros((n + 1, m + 1))
    D[1:, 0] = gap * np.arange(1, n + 1)
    D[0, 1:] = gap * np.arange(1, m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = max(D[i - 1, j - 1] + score(lyric[i - 1], heard[j - 1]["w"]),
                          D[i - 1, j] + gap, D[i, j - 1] + gap)
    match, i, j = {}, n, m
    while i > 0 and j > 0:
        s = score(lyric[i - 1], heard[j - 1]["w"])
        if D[i, j] == D[i - 1, j - 1] + s:
            if s > 0:
                match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1
    return match


def lyric_timing(song, heard, duration):
    words = []  # (section, line, text, normalised)
    for si, sec in enumerate(song["sections"]):
        for li, line in enumerate(sec["lines"]):
            for w in line["en"].split():
                words.append((si, li, w, norm_word(w)))
    match = align([w[3] for w in words], heard)
    starts = np.full(len(words), np.nan)
    ends = np.full(len(words), np.nan)
    for k, h in match.items():
        starts[k], ends[k] = heard[h]["start"], heard[h]["end"]
    # interpolate unmatched words on the word index axis (keeps order, spreads gaps evenly)
    known = np.where(~np.isnan(starts))[0]
    if len(known) == 0:
        raise SystemExit("no lyric words recognised; check the take has vocals")
    idx = np.arange(len(words))
    starts = np.interp(idx, known, starts[known])
    ends = np.maximum(np.interp(idx, known, ends[known]), starts + 0.15)
    lines = []
    for k, (si, li, text, _) in enumerate(words):
        if not lines or (lines[-1]["section"], lines[-1]["line"]) != (si, li):
            lines.append({"section": si, "line": li, "words": []})
        lines[-1]["words"].append({"w": text, "start": round(float(starts[k]), 3),
                                   "end": round(float(min(ends[k], duration)), 3), "heard": k in match})
    for line in lines:
        line["start"], line["end"] = line["words"][0]["start"], line["words"][-1]["end"]
    return lines, len(match) / len(words)


def sections_timing(song, lines, downbeats, duration):
    first = {}
    for line in lines:
        first.setdefault(line["section"], line["start"])
    starts = []
    for si in range(len(song["sections"])):
        if si == 0:
            starts.append(0.0)
        elif si in first:
            # cut on the last downbeat at least half a bar before the first sung word
            before = [d for d in downbeats if d <= first[si] - 0.9]
            starts.append(before[-1] if before else first[si] - 1.0)
        else:
            starts.append(starts[-1])
    starts = list(np.maximum.accumulate(starts))
    out = []
    for si, sec in enumerate(song["sections"]):
        end = starts[si + 1] if si + 1 < len(starts) else duration
        out.append({"tag": sec["tag"], "scene": sec["scene"], "start": round(float(starts[si]), 3),
                    "end": round(float(end), 3)})
    return out


def main():
    ap = argparse.ArgumentParser(description="Analyse a take for video rendering")
    ap.add_argument("work", help="song directory containing song.json")
    ap.add_argument("audio", help="take audio (wav)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", help="output json (default: <song>/outputs/analysis/<take>.json)")
    args = ap.parse_args()

    work = Path(args.work).resolve()
    song = json.loads((work / "song.json").read_text(encoding="utf-8"))
    y, _ = librosa.load(args.audio, sr=SR, mono=True)
    duration = len(y) / SR
    audio = Path(args.audio).resolve()
    try:
        audio = audio.relative_to(work)  # stays valid when the song directory moves or is cloned
    except ValueError:
        pass
    result = {"audio": str(audio), "duration": round(duration, 3), "fps": args.fps}
    result.update(rhythm(y, args.fps, duration))
    heard = transcribe(args.audio)
    result["lines"], coverage = lyric_timing(song, heard, duration)
    result["sections"] = sections_timing(song, result["lines"], result["downbeats"], duration)
    result["heard_ratio"] = round(coverage, 3)
    result["transcript"] = " ".join(w["w"] for w in heard)

    out = Path(args.out) if args.out else work / "outputs" / "analysis" / (Path(args.audio).stem + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"tempo {result['tempo']}  beats {len(result['beats'])}  kicks {len(result['kicks'])}  "
          f"lyric words recognised {coverage:.0%}  -> {out}")
    for sec in result["sections"]:
        print(f"  {sec['start']:7.2f}-{sec['end']:7.2f}  {sec['tag']:<13} {sec['scene']}")


if __name__ == "__main__":
    main()
