#!/bin/bash
# Fetch a pinned acestep.cpp (C++ runtime for ACE-Step 1.5), build ace-lm / ace-synth,
# and download the quantized models (~7.7 GB, Q8_0 turbo + 4B LM).
#
# macOS (Apple Silicon) builds with Metal automatically; Linux builds for the CPU.
# Needs: git, cmake, a C++17 compiler, python3, and the `hf` CLI (pip install -U huggingface_hub).
#
# Usage: bash music/setup_acestep.sh          # into .tools/acestep.cpp
#        ACESTEP_DIR=/path bash music/setup_acestep.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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

# Local patch: let ACE_THREADS set the CPU thread count. Upstream uses half the cores,
# which suits SMT machines but leaves cores idle on machines without hyper-threading.
python3 - "$DIR/src/backend.h" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = """static int backend_cpu_n_threads(void) {
    int n = (int) std::thread::hardware_concurrency() / 2;"""
new = """static int backend_cpu_n_threads(void) {
    const char * env = getenv("ACE_THREADS");
    if (env && atoi(env) > 0) {
        return atoi(env);
    }
    int n = (int) std::thread::hardware_concurrency() / 2;"""
if "ACE_THREADS" not in src:
    assert old in src, "backend.h changed upstream; update the patch"
    src = "#include <cstdlib>\n" + src.replace(old, new)
    open(path, "w").write(src)
EOF

if [ "$(uname)" = "Darwin" ]; then
    FLAGS=()  # Metal + Accelerate are enabled by default on macOS
else
    FLAGS=(-DGGML_BLAS=OFF -DGGML_NATIVE=ON)
fi
cmake -S "$DIR" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release ${FLAGS[@]+"${FLAGS[@]}"}
cmake --build "$DIR/build" --config Release -j "$(getconf _NPROCESSORS_ONLN)" --target ace-lm ace-synth

(cd "$DIR" && bash ./models.sh)
echo "acestep.cpp ready in $DIR"
