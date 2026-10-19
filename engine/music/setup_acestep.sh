#!/bin/bash
# Fetch a pinned acestep.cpp (C++ runtime for ACE-Step 1.5), build ace-lm / ace-synth,
# and download the quantized models (~7.7 GB, Q8_0 turbo + 4B LM; re-running resumes).
#
# macOS (Apple Silicon) builds with Metal automatically; Linux builds for the CPU.
# Needs: git, cmake, a C++17 compiler, python3, and the `hf` CLI (pip install -U huggingface_hub).
#
# Usage: bash engine/music/setup_acestep.sh          # into .tools/acestep.cpp
#        ACESTEP_DIR=/path bash engine/music/setup_acestep.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="${ACESTEP_DIR:-$ROOT/.tools/acestep.cpp}"
REPO="https://github.com/ServeurpersoCom/acestep.cpp"
COMMIT="de858d8e1a94f0e1939cf9bdd8ff7d9631953077"

command -v hf > /dev/null || { echo "missing 'hf' CLI: python3 -m pip install -U huggingface_hub"; exit 1; }
command -v cmake > /dev/null || { echo "missing cmake (macOS: brew install cmake)"; exit 1; }

if [ ! -d "$DIR/.git" ]; then
    mkdir -p "$DIR"
    git -C "$DIR" init -q
    git -C "$DIR" remote add origin "$REPO"
fi
git -C "$DIR" fetch -q --depth 1 origin "$COMMIT"
git -C "$DIR" checkout -q FETCH_HEAD
git -C "$DIR" submodule update --init --depth 1 || git -C "$DIR" submodule update --init

# Local patches (idempotent):
# 1. ACE_THREADS sets the CPU thread count. Upstream uses half the cores, which suits SMT
#    machines but leaves cores idle on machines without hyper-threading.
# 2. Apple's libc++ ships float std::from_chars but marks it available only from macOS 26,
#    so older deployment targets fail to compile; use acestep's own parser there too.
python3 - "$DIR/src" <<'EOF'
import sys
from pathlib import Path

src = Path(sys.argv[1])

path = src / "backend.h"
text = path.read_text()
old = """static int backend_cpu_n_threads(void) {
    int n = (int) std::thread::hardware_concurrency() / 2;"""
new = """static int backend_cpu_n_threads(void) {
    const char * env = getenv("ACE_THREADS");
    if (env && atoi(env) > 0) {
        return atoi(env);
    }
    int n = (int) std::thread::hardware_concurrency() / 2;"""
if "ACE_THREADS" not in text:
    assert old in text, "backend.h changed upstream; update the patch"
    path.write_text("#include <cstdlib>\n" + text.replace(old, new))

path = src / "pipeline-synth-ops.cpp"
text = path.read_text()
old = "#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION < 200000"
new = "#if defined(_LIBCPP_VERSION) && (_LIBCPP_VERSION < 200000 || defined(__APPLE__))"
if new not in text:
    assert old in text, "pipeline-synth-ops.cpp changed upstream; update the patch"
    path.write_text(text.replace(old, new))
EOF

if [ "$(uname)" = "Darwin" ]; then
    FLAGS=()  # Metal + Accelerate are enabled by default on macOS
else
    FLAGS=(-DGGML_BLAS=OFF -DGGML_NATIVE=ON)
fi
cmake -S "$DIR" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release ${FLAGS[@]+"${FLAGS[@]}"}
cmake --build "$DIR/build" --config Release -j "$(getconf _NPROCESSORS_ONLN)" --target ace-lm ace-synth

# Models: the Q8_0 turbo set (~7.7 GB) that upstream's models.sh picks by default, fetched here
# with progress bars. Interrupted downloads resume. Where huggingface.co is slow or unreachable:
#   export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
echo "downloading models from ${HF_ENDPOINT:-https://huggingface.co}"
mkdir -p "$DIR/models"
for f in vae-BF16.gguf Qwen3-Embedding-0.6B-Q8_0.gguf acestep-v15-turbo-Q8_0.gguf acestep-5Hz-lm-4B-Q8_0.gguf; do
    if [ -f "$DIR/models/$f" ]; then
        echo "[OK] $f"
        continue
    fi
    echo "[Download] $f"
    hf download Serveurperso/ACE-Step-1.5-GGUF "$f" --local-dir "$DIR/models"
done
echo "acestep.cpp ready in $DIR"
