# llamacpp-cuda

Pinned llama.cpp CUDA builds for **A100 (sm_80)** and **H100 (sm_90)**, baked into a
bootable image and published to GHCR. Built for one job: **measuring quantization
quality** — KL-divergence, perplexity, imatrix — on rented NVIDIA GPUs.

CUDA sibling of [`llamacpp-hip`](https://github.com/jolpindo96/llamacpp-hip) (MI300X / gfx942).

```
ghcr.io/jolpindo96/llamacpp-cuda:<llama.cpp-commit>
```

## Why bake it

The previous approach (`kld-bench-v1` RunPod template) compiled llama.cpp on the pod
at every fresh boot: 5–8 minutes of A100 time spent on `cmake` before any measurement
could start, repeated per campaign. Compilation is not GPU work, so it has no business
running on a GPU meter. Here it runs on GitHub's free runners and the pod boots
straight into measurement.

## Numerics: the fast-math floor

This is the reason the repo exists, so it is a **hard build failure**, not a warning.

Upstream `ggml/src/ggml-cuda/CMakeLists.txt` ships:

```cmake
set(CUDA_FLAGS -use_fast_math -extended-lambda)
```

`-use_fast_math` implies `-ftz=true`, `-prec-div=false`, `-prec-sqrt=false`, and lets
nvcc contract and reassociate floating-point operations. For a rig whose entire output
is *how much precision a quantization loses*, a build that introduces its own precision
loss is disqualifying — the measurement would silently include the compiler's error
alongside the quantizer's.

The build strips the flag and then **proves the strip landed**:

```dockerfile
sed -i 's/[[:space:]]*-use_fast_math//g' ggml/src/ggml-cuda/CMakeLists.txt
grep -rEn 'use_fast_math|ffast-math|funsafe-math' ... && exit 1
```

The `grep` is the load-bearing half. A bare `sed` silently no-ops if upstream reorders
or renames the flag, producing a fast-math binary that reports plausible-but-wrong
divergence numbers — the worst possible failure mode, because nothing looks broken.
The resulting diff is preserved in the image at `/opt/llama/NUMERICS.txt`.

Note this is the **inverse** of the HIP sibling's gate. There, upstream had already
removed `-funsafe-math-optimizations` (`e79e4bf66`), so that check merely verifies
absence and refuses pins older than the fix. Here the flag is live on master, so it
must be actively removed on every build.

## What ships

| Path | Contents |
|---|---|
| `/opt/llama/bin` | `llama-perplexity`, `llama-quantize`, `llama-imatrix`, `llama-bench`, `llama-server` |
| `/opt/llama/lib` | ggml/llama shared objects (registered via `ld.so.conf.d`) |
| `/opt/llama/venv` | CPU-only torch, transformers, gguf, hf CLI |
| `/opt/llama/convert_hf_to_gguf.py` | HF safetensors → GGUF, with `gguf-py` installed |
| `/opt/llama/REVISION` | exact llama.cpp commit |
| `/opt/llama/NUMERICS.txt` | the fast-math strip diff |
| `/opt/llama/DEPS.txt` | `ldd` of the real binary — derived, not guessed |

**CPU-only torch is deliberate.** `convert_hf_to_gguf.py` is I/O-bound repacking and
never touches the GPU; the CPU wheel is ~1GB against ~3GB for the CUDA build.

## Linking against a driver that isn't there

The build passes `-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined`. This is
required, not cosmetic.

`libggml-cuda.so` references CUDA **driver** API symbols (`cuGetErrorString`, the VMM
entry points) that live in `libcuda.so.1` — a library that is deliberately absent from
every CUDA container image. It is injected by `nvidia-container-toolkit` at
`docker run --gpus`; only a non-functional stub exists at build time. Without the flag
the build dies at link with:

```
/usr/bin/ld: libggml-cuda.so: undefined reference to `cuGetErrorString'
```

Upstream's own `.devops/cuda.Dockerfile` carries the identical flag for the identical
reason. The consequence is that a *complete* runtime verification is impossible in CI —
there is no GPU on the runner — so the build-time checks are structural (binaries
exist, non-driver libraries all resolve) and the first real execution happens on the pod.
`kld-bootstrap.sh` writing `STATUS.md` is that confirmation.

## Base image

Both stages use `nvidia/cuda:13.0.3-*-ubuntu24.04` — build on `devel`, runtime on the
slim `runtime` tag. Same CUDA patch version on both, so cuBLAS is identical under the
measurement.

Unlike the ROCm sibling (see its "Fat vs slim" note), taking the slim base costs
nothing here: CUDA publishes a runtime tag at the *same* version, so there is no
downgrade-to-get-slim tradeoff to refuse.

**CUDA 13 narrows the host pool.** RunPod Community Cloud hosts range from CUDA 12.4
to 13.2; a 13.0 image will not start on a 12.x host. Pin the template's
`allowedCudaVersions` to 13.x so the scheduler only places you on compatible hosts,
rather than discovering it as a boot failure. (Three pods were lost to exactly this
class of mismatch before the pattern was understood.)

## Building a pin

Tag push encodes the pin:

```bash
git tag llama-a30273376 && git push origin llama-a30273376
```

Or run the workflow manually with a `ref` input. `:latest` is opt-in only — pinned
tags are the product, and a moving `:latest` would defeat the reproducibility the pin
exists for.

## Running on RunPod

Boot runs `kld-bootstrap.sh`, which:

1. Installs `PUBLIC_KEY` and starts `sshd`. This is a generic (non-`runpod/*`) image,
   so nothing wires SSH up for you. Host keys are **deleted at build time** and
   regenerated per pod — a public image must never ship a shared private host key.
2. Selects binaries: baked → matching volume build → volume build as escape hatch
   (set `LLAMACPP_REF` to a commit differing from the baked `REVISION`). The volume
   build applies the same numerics gate.
3. Fetches the wikitext-2 test split to `/workspace/corpora/`.
4. Writes `/workspace/STATUS.md` (READY/FAILED, versions, commit, conventions).

Only `/workspace` survives a stop/start; binaries are baked, so that is cheap.

### KLD syntax

Two flags, easy to get wrong:

```bash
llama-perplexity -m quant.gguf -f wiki.test.raw \
    --kl-divergence-base reference.dat \
    --kl-divergence
```

`--kl-divergence-base` takes the path (it both generates and reads the reference);
`--kl-divergence` takes **no value** and switches on comparison mode.

A KLD number is only meaningful when reference and test share **identical underlying
weights** — a quant against *its own* BF16. Comparing a quant against a different
fine-tune or merge measures the model difference, not the quantization.
