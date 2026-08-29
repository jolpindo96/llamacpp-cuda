#!/usr/bin/env bash
# =============================================================================
# llamacpp-cuda bootstrap - KLD / perplexity rig on RunPod A100 / H100
#
# Binaries ship BAKED in the image; the volume build is an escape hatch only
# (set LLAMACPP_REF to a commit that differs from the baked REVISION).
# Idempotent across stop/start. Durable state on /workspace only.
#
# Log: /workspace/kld-bootstrap.log   Status: /workspace/STATUS.md
# =============================================================================
set -u
WS=/workspace
mkdir -p "$WS"
exec >>"$WS/kld-bootstrap.log" 2>&1
echo "=== bootstrap start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

BAKED_BIN=/opt/llama/bin
BAKED_REV="$(cat /opt/llama/REVISION 2>/dev/null || echo '')"
VENV=/opt/llama/venv
LLAMA_DIR=$WS/llama.cpp
VOL_BIN=$LLAMA_DIR/build/bin
CORPUS_DIR=$WS/corpora
LLAMA_REF="${LLAMACPP_REF:-}"          # unset => use baked binaries

fail(){ echo "FAILED: $1"; echo "FAILED: $1 ($(date -u))" > "$WS/STATUS.md"; }

# --- 1. SSH ------------------------------------------------------------------
# This is a generic (non-runpod/*) image, so nothing wires up sshd for us:
# RunPod's PUBLIC_KEY env has to be installed by hand. Host keys were deleted at
# build time (a public image must not ship a shared private host key), so they
# are generated fresh per pod here.
setup_ssh(){
    ssh-keygen -A >/dev/null 2>&1
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    if [ -n "${PUBLIC_KEY:-}" ]; then
        grep -qxF "$PUBLIC_KEY" /root/.ssh/authorized_keys 2>/dev/null \
            || echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    else
        echo "WARN: PUBLIC_KEY unset - no SSH access will be possible"
    fi
    mkdir -p /run/sshd
    # PermitRootLogin: RunPod execs as root; key-only, passwords stay disabled.
    sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
    /usr/sbin/sshd
    echo "sshd started"
}
setup_ssh

# --- 2. Which binaries -------------------------------------------------------
# Order: baked (ref unset or matches baked) -> matching volume build -> build it.
ref_matches(){ [ -n "$1" ] && [ -n "$2" ] && case "$1" in "$2"*) return 0;; esac; return 1; }

BIN=""; ACTIVE_REV=""; SOURCE=""
if [ -x "$BAKED_BIN/llama-perplexity" ] && { [ -z "$LLAMA_REF" ] || ref_matches "$BAKED_REV" "$LLAMA_REF"; }; then
    BIN="$BAKED_BIN"; ACTIVE_REV="$BAKED_REV"; SOURCE="baked image"
    echo "using BAKED binaries at ${BAKED_REV:0:9} (LLAMACPP_REF=${LLAMA_REF:-<unset>})"
elif [ -x "$VOL_BIN/llama-perplexity" ] \
     && ref_matches "$(cd "$LLAMA_DIR" 2>/dev/null && git rev-parse HEAD 2>/dev/null)" "$LLAMA_REF"; then
    BIN="$VOL_BIN"; ACTIVE_REV="$(cd "$LLAMA_DIR" && git rev-parse HEAD)"; SOURCE="volume build"
    echo "using VOLUME build at ${ACTIVE_REV:0:9}"
else
    echo "building llama.cpp at ref '$LLAMA_REF' on the volume (escape hatch)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq \
        build-essential cmake ccache libcurl4-openssl-dev cuda-toolkit-13-0 || fail "apt toolchain"
    [ -d "$LLAMA_DIR/.git" ] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" || fail "git clone"
    cd "$LLAMA_DIR" && git fetch --all --tags -q && git checkout -q "$LLAMA_REF" || fail "checkout $LLAMA_REF"
    # Same numerics gate as the image build: strip fast-math, then prove it.
    sed -i 's/[[:space:]]*-use_fast_math//g' ggml/src/ggml-cuda/CMakeLists.txt
    if grep -rEn 'use_fast_math|ffast-math|funsafe-math' ggml/src/ggml-cuda/CMakeLists.txt; then
        fail "fast-math survived the strip in volume build"
    else
        cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;90" \
              -DLLAMA_CURL=ON -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null \
          && cmake --build build -j"$(nproc)" --target \
              llama-perplexity llama-quantize llama-imatrix llama-bench llama-server \
          && { BIN="$VOL_BIN"; ACTIVE_REV="$(git rev-parse HEAD)"; SOURCE="volume build"; } \
          || fail "cmake build"
    fi
fi

# --- 3. Test corpus ----------------------------------------------------------
# wikitext-2 test split: the constant across every measurement in this campaign,
# so cross-model numbers stay comparable.
#
# Verified by CONTENT, not by curl's exit code. The bare .raw path under this
# dataset repo does not exist; curl -sL cheerfully wrote the 15-byte
# "Entry not found" body and exited 0. STATUS.md then reported READY over a
# corpus that would have produced a perplexity run across zero chunks. Fetch the
# zip (a path that does exist) and gate on the extracted size.
mkdir -p "$CORPUS_DIR"
CORPUS="$CORPUS_DIR/wiki.test.raw"
if [ ! -s "$CORPUS" ] || [ "$(wc -c < "$CORPUS")" -lt 1000000 ]; then
    curl -sL --retry 5 --retry-delay 3 -o "$CORPUS_DIR/w.zip" \
      https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip \
      || fail "corpus download"
    # python rather than unzip: the venv is already here, one less apt package.
    "$VENV/bin/python" -c \
      "import zipfile; zipfile.ZipFile('$CORPUS_DIR/w.zip').extractall('$CORPUS_DIR')" \
      || fail "corpus extract"
    cp "$CORPUS_DIR/wikitext-2-raw/wiki.test.raw" "$CORPUS" || fail "corpus place"
    rm -f "$CORPUS_DIR/w.zip"
    GOT="$(wc -c < "$CORPUS")"
    [ "$GOT" -gt 1000000 ] || fail "corpus is $GOT bytes, expected ~1.29MB"
fi

mkdir -p "$WS/models" "$WS/kld" "$WS/hf-cache"

# --- 4. Status breadcrumb for agents ----------------------------------------
if [ -n "$BIN" ]; then
cat > "$WS/STATUS.md" <<EOF
# Pod bootstrap: READY ($(date -u +%Y-%m-%dT%H:%M:%SZ))
- llama.cpp: ${ACTIVE_REV:0:9} (source: $SOURCE), binaries in $BIN
- numerics: fast-math STRIPPED at build (see /opt/llama/NUMERICS.txt)
- python: $VENV ($($VENV/bin/python --version 2>&1); torch $($VENV/bin/python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo '?'), CPU-only by design)
- convert: $VENV/bin/python /opt/llama/convert_hf_to_gguf.py --outtype bf16 <src> --outfile <dst>
- corpus: $CORPUS_DIR/wiki.test.raw ($(wc -c < "$CORPUS_DIR/wiki.test.raw" 2>/dev/null || echo 0) bytes)
- GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')
## Conventions
- Models -> $WS/models ; KLD refs -> $WS/kld ; hf cache -> $WS/hf-cache
- KLD is two flags: --kl-divergence-base <ref.dat> generates/points at the reference,
  --kl-divergence (no value) switches on comparison mode. Path always on -base.
- Everything outside /workspace is WIPED on pod stop; binaries are baked so that is cheap.
EOF
fi
echo "=== bootstrap complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Hold the container open; sshd is backgrounded above.
sleep infinity
