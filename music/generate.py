"""Generate song takes for a work with acestep.cpp (ACE-Step 1.5).

Each seed gives one take: ace-lm plans the song (metadata + audio codes) from the
caption and lyrics in works/<name>/song.json, then ace-synth renders it to audio.

Usage:
    python3 music/generate.py works/ground-truth --seeds 7 11 23
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    ap.add_argument("work", help="work directory containing song.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7], help="one take per seed")
    ap.add_argument("--acestep", default=os.environ.get("ACESTEP_DIR", str(ROOT / ".tools" / "acestep.cpp")),
                    help="acestep.cpp checkout with build/ and models/ (see music/setup_acestep.sh)")
    ap.add_argument("--out", help="output directory (default: outputs/<work>/takes)")
    args = ap.parse_args()

    work = Path(args.work).resolve()
    song = json.loads((work / "song.json").read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else ROOT / "outputs" / work.name / "takes"
    out.mkdir(parents=True, exist_ok=True)
    ace = Path(args.acestep)
    models = ace / "models"

    for seed in args.seeds:
        name = f"take_{seed}"
        request = dict(song["request"], lyrics=lyrics_text(song["sections"]), seed=seed)
        req_path = out / f"{name}.json"
        req_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")

        # ace-lm writes <name>0.json next to the request; ace-synth writes <name>00.wav.
        t_lm = run([str(ace / "build" / "ace-lm"), "--models", str(models), "--request", str(req_path)],
                   out / f"{name}.lm.log")
        t_synth = run([str(ace / "build" / "ace-synth"), "--models", str(models),
                       "--request", str(out / f"{name}0.json")], out / f"{name}.synth.log")
        shutil.move(out / f"{name}00.wav", out / f"{name}.wav")
        print(f"{name}.wav  lm {t_lm:.0f}s  synth {t_synth:.0f}s", flush=True)


if __name__ == "__main__":
    main()
