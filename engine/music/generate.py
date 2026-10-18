"""Generate song takes for a work with acestep.cpp (ACE-Step 1.5).

Each seed gives one take: ace-lm plans the song (metadata + audio codes) from the
caption and lyrics in <song>/song.json, then ace-synth renders it to audio,
matching the tonal quality of the work's timbre reference when song.json names one.

Usage:
    python3 engine/music/generate.py ground-truth --seeds 7 11 23
    # keep an earlier composition, synthesise it again (e.g. with the timbre reference)
    python3 engine/music/generate.py ground-truth --seeds 7 --suffix _b \\
        --codes ground-truth/composition-take7.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def lyrics_text(sections):
    """Build ACE-Step's tagged lyrics ("[Verse 1]\\nline\\n...") from song.json sections."""
    blocks = []
    for sec in sections:
        lines = [f"[{sec['tag']}]"] + [line["en"] for line in sec["lines"]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def run(cmd, log_path):
    start = time.time()
    with open(log_path, "w") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    if code != 0:
        sys.exit(f"{cmd[0]} failed with exit code {code}, see {log_path}")
    return time.time() - start


def main():
    ap = argparse.ArgumentParser(description="Generate song takes with acestep.cpp")
    ap.add_argument("work", help="song directory containing song.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7], help="one take per seed")
    ap.add_argument("--acestep", default=os.environ.get("ACESTEP_DIR", str(ROOT / ".tools" / "acestep.cpp")),
                    help="acestep.cpp checkout with build/ and models/ (see engine/music/setup_acestep.sh)")
    ap.add_argument("--out", help="output directory (default: <song>/outputs/takes)")
    ap.add_argument("--codes", help="planned request from an earlier run (an ace-lm output such as "
                                    "take_70.json): skip the LM and re-synthesise that composition")
    ap.add_argument("--ref-audio", help="timbre reference (default: song.json timbre_reference; 'none' to disable)")
    ap.add_argument("--suffix", default="", help="appended to take names, e.g. _b")
    args = ap.parse_args()

    work = Path(args.work).resolve()
    song = json.loads((work / "song.json").read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else work / "outputs" / "takes"
    out.mkdir(parents=True, exist_ok=True)
    ace = Path(args.acestep)
    models = ace / "models"
    ref = args.ref_audio or song.get("timbre_reference")
    ref = None if ref in (None, "none") else (work / ref if not Path(ref).is_absolute() else Path(ref))

    for seed in args.seeds:
        name = f"take_{seed}{args.suffix}"
        planned = out / f"{name}0.json"  # ace-lm writes <name>0.json; ace-synth then writes <name>00.wav
        if args.codes:
            request = json.loads(Path(args.codes).read_text(encoding="utf-8"))
            request["seed"] = seed
            planned.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
            t_lm = 0.0
        else:
            request = dict(song["request"], lyrics=lyrics_text(song["sections"]), seed=seed)
            req_path = out / f"{name}.json"
            req_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
            t_lm = run([str(ace / "build" / "ace-lm"), "--models", str(models), "--request", str(req_path)],
                       out / f"{name}.lm.log")
        synth = [str(ace / "build" / "ace-synth"), "--models", str(models), "--request", str(planned)]
        if ref:
            synth += ["--ref-audio", str(ref)]
        t_synth = run(synth, out / f"{name}.synth.log")
        shutil.move(out / f"{name}00.wav", out / f"{name}.wav")
        print(f"{name}.wav  lm {t_lm:.0f}s  synth {t_synth:.0f}s  timbre {ref.name if ref else 'none'}",
              flush=True)


if __name__ == "__main__":
    main()
