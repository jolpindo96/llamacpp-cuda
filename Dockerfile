# syntax=docker/dockerfile:1
#
# Pinned llama.cpp CUDA build for A100 (sm_80) / H100 (sm_90), baked into an image.
#
# Purpose: quantization-quality measurement -- KL-divergence, perplexity, imatrix
# -- on rented NVIDIA GPUs. The entire value of this image is that the numbers it
# produces are trustworthy, which is why the fast-math gate below is a hard build
# failure rather than a warning.
#
# Base: nvidia/cuda:13.0.3-{devel,runtime}-ubuntu24.04
#   Build and runtime share the SAME CUDA patch version on purpose: identical
#   cuBLAS under the measurement. Unlike the ROCm sibling repo (llamacpp-hip),
#   CUDA publishes a slim runtime tag at the same version, so taking the slim
#   base costs no version parity here -- this is not the "downgrade to get slim"
#   tradeoff that repo documents under "Fat vs slim".
#
# CUDA 13 is deliberate (matches the local RTX 5090 stack). It narrows the
# RunPod Community Cloud host pool to newer drivers, so pin the template's
# allowedCudaVersions to 13.x rather than discovering it as a boot failure.

ARG CUDA_VERSION=13.0.3
ARG BUILD_BASE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04
ARG RUNTIME_BASE=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

# ---------------------------------------------------------------- build stage
FROM ${BUILD_BASE} AS build

ARG LLAMA_REF
# A100 = sm_80, H100 = sm_90, B200 = sm_100, B300 (Blackwell Ultra) = sm_103.
# CI build runs on GitHub's dime, so carrying all four means renting whichever
# card is available on a given day never requires a new image.
#
# CAVEAT on the datacenter-Blackwell entries, which is a property of llama.cpp
# and not of this image: ggml-cuda defines GGML_CUDA_CC_BLACKWELL as 1200 (the
# consumer / RTX PRO family) and states in common.cuh that it integrates only
# that family's tensor-core instructions. CC 10.0 and 10.3 therefore fall in
# the Hopper tier (>= 900, < 1200): blackwell_mma_available() is false, and the
# Blackwell branches in fattn.cu / mmvq.cu are not taken. B200/B300 run
# Hopper-era kernels -- no tcgen05, no native FP4. They work; they are not
# tuned. A llama.cpp B300-vs-MI355X number measures engine coverage, not
# silicon, because gfx950 does have CDNA4-specific paths (incl. MXFP4).
ARG CUDA_ARCHS="80;90;100;103"

# Flash-attention KV-cache type combinations to compile. Upstream replaced
# GGML_CUDA_FA_ALL_QUANTS with this in #28079; an uncompiled combination now
# falls back at runtime with a warning rather than being unavailable, which is
# easy to miss in a log and silently changes what you are measuring.
#
# The upstream default is "q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16". Passing this
# REPLACES that default, so every entry we still need must be repeated here --
# in particular f16-f16, which is the default KV cache type and therefore what
# every standard perplexity/KLD run uses. Dropping it would break the baseline.
#
# Beyond the defaults we add the combinations James actually serves with, so the
# rig can measure a model at its real serving config rather than only at f16:
#   q5_1-q4_1  Qwen3.8-27B CUDA server config (-ctk q5_1 -ctv q4_1)
#   bf16-q8_0  Gemma-4-31B no-image config
#   q5_1-q5_1  Qwen3.8 Vulkan config
ARG CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q5_1-q4_1;bf16-q8_0;q5_1-q5_1"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ccache libcurl4-openssl-dev libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN test -n "${LLAMA_REF}" || { echo "LLAMA_REF build-arg is required"; exit 1; } \
 && git clone https://github.com/ggml-org/llama.cpp . \
 && git checkout -q "${LLAMA_REF}" \
 && git rev-parse HEAD | tee /REVISION

# --- Numerics gate: CUDA builds must NOT use fast-math. ----------------------
# Upstream ggml-cuda/CMakeLists.txt ships `set(CUDA_FLAGS -use_fast_math -extended-lambda)`.
# -use_fast_math implies -ftz=true, -prec-div=false, -prec-sqrt=false and lets nvcc
# contract/reassociate FP ops. For a KLD/perplexity rig that is disqualifying: this
# image exists to MEASURE precision loss, so the build must not silently introduce
# its own.
#
# This is the inverse of the HIP sibling repo's gate. There, upstream had already
# removed the offending flag and the check merely VERIFIES its absence. Here the
# flag is present on master, so we strip it and then PROVE the strip landed.
#
# Why prove it: a bare `sed` silently no-ops if upstream reorders or renames the
# flag, yielding a fast-math binary that reports plausible-but-wrong divergence.
# The grep turns that silent failure into a red build.
RUN set -eu; \
    echo "--- CUDA_FASTMATH_STRIP ---"; \
    cp ggml/src/ggml-cuda/CMakeLists.txt /tmp/cuda-cmake.orig; \
    sed -i 's/[[:space:]]*-use_fast_math//g' ggml/src/ggml-cuda/CMakeLists.txt; \
    if grep -rEn 'use_fast_math|ffast-math|funsafe-math|fassociative-math' \
            ggml/src/ggml-cuda/CMakeLists.txt ggml/CMakeLists.txt; then \
        echo "FAIL: fast-math flag survived the strip"; \
        exit 1; \
    fi; \
    { echo "# fast-math strip applied at build time"; \
      echo "# pin: ${LLAMA_REF}"; \
      diff -u /tmp/cuda-cmake.orig ggml/src/ggml-cuda/CMakeLists.txt || true; \
    } > /NUMERICS.txt; \
    cat /NUMERICS.txt; \
    echo "PASS: CUDA build is clean of fast-math"

# --allow-shlib-undefined is required, not cosmetic: libggml-cuda.so references
# CUDA *driver* API symbols (cuGetErrorString, the VMM entry points) that live in
# libcuda.so.1. That library is not in the devel image -- it is injected by
# nvidia-container-toolkit at `docker run --gpus`, and only a non-functional stub
# exists at build time. Without this flag the link fails with
# "undefined reference to `cuGetErrorString'". Upstream's own .devops/cuda.Dockerfile
# carries the identical flag for the identical reason.
RUN cmake -S . -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA_FA_QUANTS="${CUDA_FA_QUANTS}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
 && cmake --build build --config Release -j"$(nproc)" \
        --target llama-perplexity llama-quantize llama-imatrix llama-bench llama-server

# Collect binaries, the shared libs they were built against, and the pieces the
# HF->GGUF conversion path needs (convert_hf_to_gguf.py imports from gguf-py).
RUN set -eu; \
    mkdir -p /opt/llama/bin /opt/llama/lib; \
    cp build/bin/llama-perplexity build/bin/llama-quantize build/bin/llama-imatrix \
       build/bin/llama-bench build/bin/llama-server /opt/llama/bin/; \
    find build -name '*.so*' -exec cp -P {} /opt/llama/lib/ \; ; \
    cp convert_hf_to_gguf.py /opt/llama/; \
    cp -r gguf-py /opt/llama/gguf-py; \
    cp -r conversion /opt/llama/conversion; \
    cp /REVISION /opt/llama/REVISION; \
    cp /NUMERICS.txt /opt/llama/NUMERICS.txt

# Record real linkage so the runtime package list is derived, never guessed.
#
# libcuda.so.1 is EXPECTED to be unresolved here and is the only such exemption.
# It is the driver library, injected by nvidia-container-toolkit at
# `docker run --gpus`. Note the stubs directory cannot stand in for it: the stub
# is named libcuda.so while the soname required is libcuda.so.1, and the loader
# matches on exact filename -- so putting stubs on LD_LIBRARY_PATH resolves
# nothing. It is left out rather than left in looking useful.
#
# Everything else unresolved means we failed to ship a library, which must fail
# the build.
RUN set -eu; \
    LD_LIBRARY_PATH=/opt/llama/lib \
        ldd /opt/llama/bin/llama-perplexity | sort > /opt/llama/DEPS.txt; \
    cat /opt/llama/DEPS.txt; \
    MISSING="$(grep 'not found' /opt/llama/DEPS.txt | grep -v 'libcuda\.so\.1' || true)"; \
    if [ -n "$MISSING" ]; then \
        echo "FAIL: unresolved shared libraries in llama-perplexity:"; \
        echo "$MISSING"; exit 1; \
    fi; \
    echo "PASS: only the driver library is unresolved (supplied at runtime)"

# -------------------------------------------------------------- runtime stage
FROM ${RUNTIME_BASE} AS runtime

ARG LLAMA_REF
LABEL org.opencontainers.image.revision="${LLAMA_REF}" \
      org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      org.opencontainers.image.title="llamacpp-cuda" \
      org.opencontainers.image.description="Pinned llama.cpp CUDA (sm_80/sm_90) KLD+perplexity image for RunPod A100/H100"

# libgomp1 is the price of the slim runtime base: ggml links the GNU OpenMP
# runtime, which arrives implicitly with the toolchain in -devel but is absent
# from -runtime. The build-stage DEPS gate enumerated every unresolved library
# and this was the only one, so the list below is derived, not guessed.
#
# The rest: curl (corpus fetch), python3-venv (hf CLI + conversion),
# openssh-server (RunPod exec), ca-certificates (HTTPS to HF/GHCR),
# git (volume-build escape hatch).
#
# apt's openssh-server postinst generates SSH host keys at BUILD time. Baked into
# a public image that means every pod boots with the same host keypair, whose
# private half anyone can pull from the registry. They are deleted here; the boot
# script runs `ssh-keygen -A` to generate fresh per-pod keys.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl ca-certificates python3-venv python3-dev git openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_*

COPY --from=build /opt/llama /opt/llama

# CUDA runtime debs do register with the dynamic linker, but /opt/llama/lib
# (the ggml/llama shared objects) does not exist as far as ld.so is concerned.
RUN printf '/opt/llama/lib\n' > /etc/ld.so.conf.d/llamacpp.conf \
 && ldconfig

# Conversion toolchain: CPU-only torch on purpose. convert_hf_to_gguf.py is
# I/O-bound repacking, never touches the GPU, and the CPU wheel is ~1GB against
# ~3GB for the CUDA build. Baked so a fresh pod converts BF16 safetensors
# immediately instead of pip-installing on the meter.
#
# numpy is pinned to match upstream's requirements/*.txt (numpy~=2.2.6), NOT
# left to `-U`. Upstream bumped to 2.4.6 (#28649) and reverted it within a day
# (#28654); an unpinned install here would pull exactly the version they backed
# out of, into the toolchain that produces the BF16 KLD reference. Bump this
# only when upstream's requirements files bump.
RUN python3 -m venv /opt/llama/venv \
 && /opt/llama/venv/bin/pip install -q -U pip \
 && /opt/llama/venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch \
 && /opt/llama/venv/bin/pip install -q -U \
        "huggingface_hub[hf_transfer,cli]" transformers safetensors \
        sentencepiece protobuf "numpy~=2.2.6" \
 && /opt/llama/venv/bin/pip install -q /opt/llama/gguf-py

COPY kld-bootstrap.sh /usr/local/bin/kld-bootstrap.sh
RUN chmod +x /usr/local/bin/kld-bootstrap.sh

ENV PATH=/opt/llama/bin:/opt/llama/venv/bin:${PATH}
ENV HF_HOME=/workspace/hf-cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ENV above only reaches the container's own entrypoint. An `ssh pod "command"`
# session is spawned by sshd, not by that process, and a non-interactive bash
# skips /etc/profile and returns early from ~/.bashrc -- so `llama-perplexity`
# is "command not found" over SSH despite being on the image's PATH. Since SSH
# is how this pod is actually driven, that made the baked binaries reachable
# only by absolute path.
#
# /etc/environment is read by pam_env for SSH logins including non-interactive
# ones, so it is the one place that fixes all of them. It performs no variable
# expansion, hence the literal list. profile.d covers interactive shells too.
RUN printf 'PATH="/opt/llama/bin:/opt/llama/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\nHF_HOME="/workspace/hf-cache"\nHF_HUB_ENABLE_HF_TRANSFER="1"\n' \
        > /etc/environment \
 && printf '#!/bin/sh\nexport PATH=/opt/llama/bin:/opt/llama/venv/bin:$PATH\nexport HF_HOME=/workspace/hf-cache\nexport HF_HUB_ENABLE_HF_TRANSFER=1\n' \
        > /etc/profile.d/llama.sh \
 && chmod +x /etc/profile.d/llama.sh

# Verify the baked artifacts actually work in the runtime image rather than
# trusting COPY (exit codes lie). --version does not initialise CUDA, so it runs
# without a GPU present; libcuda.so.1 is the one lib allowed to be missing here
# (driver-injected at `docker run --gpus`, see DEPS.txt note in the build stage).
#
# The convert_hf_to_gguf.py --help line is not redundant with the import check
# above it. That script also imports its sibling `conversion` package (upstream
# split the per-architecture converters out of the monolithic script), and an
# earlier build shipped the script without that package. Importing
# torch/transformers/gguf succeeded, so the image looked fine and the
# ModuleNotFoundError only surfaced when a conversion was first attempted -- on
# the pod, on the meter. --help exercises the real import chain here instead.
#
# Comments stay outside the RUN: a `#` line between backslash continuations
# depends on the parser stripping it, which is not a thing to rely on in a gate.
RUN set -eu; \
    MISSING="$(ldd /opt/llama/bin/llama-perplexity 2>/dev/null \
                 | grep 'not found' | grep -v 'libcuda\.so\.1' || true)"; \
    if [ -n "$MISSING" ]; then \
        echo "FAIL: missing non-driver library:"; echo "$MISSING"; exit 1; \
    fi; \
    test -x /opt/llama/bin/llama-perplexity; \
    test -x /opt/llama/bin/llama-quantize; \
    test -s /opt/llama/REVISION; \
    test -s /opt/llama/NUMERICS.txt; \
    /opt/llama/venv/bin/python -c 'import torch, transformers, gguf; print("conv deps OK", torch.__version__)'; \
    cd /opt/llama && /opt/llama/venv/bin/python convert_hf_to_gguf.py --help >/dev/null; \
    echo "PASS: runtime image verified at $(cat /opt/llama/REVISION)"

CMD ["/usr/local/bin/kld-bootstrap.sh"]
