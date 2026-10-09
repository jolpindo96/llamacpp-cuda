# llamacpp-cuda

Pinned llama.cpp CUDA builds for **A100 (sm_80)**, **H100 (sm_90)**, **B200 (sm_100)**,
**B300 / Blackwell Ultra (sm_103)** and **RTX PRO 6000 Blackwell (sm_120a)**, baked into
a bootable image and published to GHCR. Built for one job: **measuring quantization
quality** — KL-divergence, perplexity, imatrix — on rented NVIDIA GPUs.

> **Of the three Blackwells, only sm_120 is actually optimized.** ggml-cuda defines its
> Blackwell tier as CC 1200 (consumer / RTX PRO) and integrates only that family's
> tensor-core instructions. CC 10.0 / 10.3 land in the Hopper tier:
> `blackwell_mma_available()` is false and the Blackwell branches in `fattn.cu` /
> `mmvq.cu` are skipped. B200/B300 work but are **not tuned** — no tcgen05, no native
> FP4. A llama.cpp B300-vs-MI355X comparison therefore measures engine coverage, not
> silicon; gfx950 *does* have CDNA4-specific paths. Use SGLang/vLLM for a hardware
> comparison.
>
> sm_120 is spelled `120a-real`, matching upstream's CMake — the `a` suffix is what
> enables the family-specific instructions. Plain `120` would compile and silently
> skip them.

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
| `/opt/llama/lib` | ggml/llama shared objects (registered via `ld.so.conf.d`) and, from `-r2` tags, the backend modules `libggml-cuda.so` + `libggml-cpu-<level>.so` |
| `/opt/llama/venv` | CPU-only torch, transformers, gguf, hf CLI |
| `/opt/llama/convert_hf_to_gguf.py` | HF safetensors → GGUF, with `gguf-py` installed |
| `/opt/llama/REVISION` | exact llama.cpp commit |
| `/opt/llama/NUMERICS.txt` | the fast-math strip diff |
| `/opt/llama/DEPS.txt` | `ldd` of the binary and of every backend module — derived, not guessed |
| `/opt/llama/CPU_VARIANTS.txt` | the CPU backend variants built into the image (`-r2` tags on) |
| `/opt/llama/PYDEPS.txt` | `pip freeze` of the conversion venv — the versions this build actually got (`-r2` tags on) |
| `/opt/llama/CCCL_VERSION.txt` | CCCL the CUDA module was compiled against (`a7fb71fab` on); the build refuses CUB `DeviceTopK` below 3.4.3 |

**CPU-only torch is deliberate.** `convert_hf_to_gguf.py` is I/O-bound repacking and
never touches the GPU; the CPU wheel is ~1GB against ~3GB for the CUDA build.

## CPU backend: portable variants, not `-march=native`

Tags up to `26394b4e6` compiled ggml's CPU backend with `-march=native` — for
whichever CPU the GitHub runner happened to have (`GGML_NATIVE` defaults ON; the
build log says `Adding CPU backend variant ggml-cpu: -march=native`). Two problems: an
image built on an AVX-512 runner dies with an illegal-instruction error on a host
without AVX-512, and two tags' CPU code can differ only because they landed on
different runners. The CPU backend runs even when every layer is on the GPU — the
token-embedding lookup stays on the host.

From `26394b4e6-r2` on, the build uses upstream's own flags
(`.devops/cuda.Dockerfile`): `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=ON`,
`GGML_CPU_ALL_VARIANTS=ON`. That compiles one CPU backend per x86 feature level —
`x64` baseline, `sse42`, `sandybridge`, `ivybridge`, `piledriver`, `haswell`,
`skylakex`, `cannonlake`, `cascadelake`, `icelake`, `cooperlake`, `zen4`,
`alderlake`, `sapphirerapids` — each a separate `libggml-cpu-<level>.so`. At
startup ggml asks each one to score itself against the host's CPUID and loads the
best; `/opt/llama/verify-backends.py` reports which one (see below for why the
tools' own log line can't be trusted for that). On the GitHub runner, an AMD EPYC
9V74 whose VM hides AVX-512, it picks `haswell` — exactly the case where a
`-march=native` build would have baked in the build machine's view of the CPU.

`GGML_BACKEND_DIR=/opt/llama/lib` is compiled in so the loader finds the modules
there (by default it only looks next to the executable and in the current
directory).

**One new failure mode, and its guard.** With dynamic backends, a Release build
skips a module that fails to load *without a message* — a pod whose driver cannot
serve `libggml-cuda.so` would quietly measure on the CPU instead of refusing to
start, which a statically linked binary did. `kld-bootstrap.sh` therefore runs
`/opt/llama/verify-backends.py --require CUDA --require CPU` and fails `STATUS.md`
unless both registered; the script's output (backends, loaded modules, devices) goes
into `STATUS.md`. The build runs the same script with `--require CPU` (the CI runner
has no GPU, so the CUDA module can only be checked on a pod):

```
backends: CPU
modules:  libggml-cpu-haswell.so
device:   CPU = AMD EPYC 9V74 80-Core Processor
```

The script asks ggml's registry directly (ctypes on `libggml.so`, then
`ggml_backend_load_all()`) and reads the loaded CPU variant from `/proc/self/maps`.
It deliberately does **not** grep the tools' `load_backend: loaded ...` lines:
llama.cpp's logger is asynchronous and `--list-devices` exits before it drains, so
those lines are dropped even when the load succeeded. The first version of the gate
relied on them and failed a working image.

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

## Flash-attention KV-cache combinations

`GGML_CUDA_FA_QUANTS` (upstream #28079 replaced `GGML_CUDA_FA_ALL_QUANTS`) controls
which KV-cache type pairs get FA kernels compiled. An uncompiled pair now falls back
at runtime **with only a warning** — easy to miss in a log, and it silently changes
what you are measuring.

Passing the flag *replaces* the upstream default
(`q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`), so this image repeats those and adds the
combinations actually served with:

| Pair | Why |
|---|---|
| `f16-f16` | default KV type — **every** standard PPL/KLD run; never drop it |
| `bf16-bf16` | BF16 reference runs |
| `q8_0-q8_0` | Gemma-4-31B, Muse Glimmer server configs |
| `q5_1-q4_1` | Qwen3.8-27B CUDA server config |
| `bf16-q8_0` | Gemma-4-31B no-image config |
| `q5_1-q5_1` | Qwen3.8 Vulkan config |
| `q4_0-q4_0` | upstream default, kept |

This is what makes it possible to measure a model **at its real serving config**, not
only at f16 — e.g. running perplexity twice, once with quantized KV cache and once at
BF16, to price the cache quantization itself.

## Base image

Both stages use `nvidia/cuda:13.4.2-*-ubuntu24.04` — build on `devel`, runtime on the
slim `runtime` tag. Same CUDA patch version on both, so cuBLAS is identical under the
measurement. Tags up to `3057bb66c` were built on `13.0.3`, the `26394b4e6` tags on
`13.4.1`; see [Pins](#pins).

**Why 13.4.2 and not 13.4.1 — a race in `DeviceTopK`.** CUDA 13.4.1 bundles CCCL
3.4.2, and CUB's `DeviceTopK` before 3.4.3 has a race (NVIDIA/cccl#10627): thread 0
can mark a CTA's merged histogram done while other threads are still adding their
counts, so a top-k selection can be silently wrong and differ run to run. The
`26394b4e6` and `26394b4e6-r2` images were built on 13.4.1 at a pin that enabled
`DeviceTopK` from CCCL 3.2, so they ship the racy kernel. Their CUB symbols read
`cub::_V_300402_SM_800_900_1000_1030_1200`, which is how this was confirmed without
Docker. Upstream now enables it only from 3.4.3 (`767767850`, #29792) and uses the sort
fallback below that. 13.4.2 bundles 3.4.3, and the build records the version in
`/opt/llama/CCCL_VERSION.txt` and **fails** if `DeviceTopK` was compiled against
anything older. Only `TOP_K` users are affected: sparse-attention indexers
(Qwen3.8-Flash-Next, DeepSeek V4, GLM-DSA). MoE routing uses `ARGSORT`.

**Why 13.4 and not 13.0.** ggml-cuda's `TOP_K` uses `cub::DeviceTopK` only when the
toolkit's bundled CCCL is ≥ 3.2 (`top-k.cu`); CUDA 13.0 ships CCCL 3.0, 13.2 → 3.2,
13.4 → 3.4. Below 3.2 every row longer than 1024 columns falls back to a full
segmented argsort — the kernel that sparse-attention indexers (Qwen3.8-Flash-Next's
QSA, DeepSeek's DSA) hit in every full-attention layer for every token, and the one
that was also called with aliased CUB key buffers until upstream `b23701f77`
(#28389). Dense models and MoE routing (`n_expert` ≤ 512 takes the bitonic path)
never reach it, which is why the 13.0 tags stay valid for everything measured on
them. Upstream's own release builds moved to 13.4.1 in #29202.

**Host compatibility is unchanged.** The 13.4.x bases' `NVIDIA_REQUIRE_CUDA` accepts
driver branches 580 / 590 / 595 / 610 — CUDA 13.0 through 13.3 hosts — through
minor-version compatibility, so the image starts wherever the 13.0 one did. The
≥ 13.0 host check below still applies.

Unlike the ROCm sibling (see its "Fat vs slim" note), the slim base costs no *version
parity* here: CUDA publishes a runtime tag at the same version, so there is no
downgrade-to-get-slim tradeoff to refuse.

It does cost exactly one package. `libgomp1` (GNU OpenMP, which ggml links) arrives
implicitly with the toolchain in `-devel` and is absent from `-runtime`, so it is
installed explicitly. That was found by the linkage gate rather than by reasoning —
which is the argument for having the gate: the runtime package list is derived from
what `ldd` actually reports, never guessed.

**CUDA 13 narrows the host pool.** RunPod Community Cloud hosts range from CUDA 12.4
to 13.2; a 13.0 image will not start on a 12.x host. Pin the template's
`allowedCudaVersions` to 13.x so the scheduler only places you on compatible hosts,
rather than discovering it as a boot failure. (Three pods were lost to exactly this
class of mismatch before the pattern was understood.)

## Pins

Tags are per llama.cpp commit and never `:latest`. **Never mix tags within one
comparison set** — a bump changes kernels, tile configs and the CUDA base, and the
numbers stop being comparable even when every model still loads.

| Tag | llama.cpp | CUDA base | Use it for |
|---|---|---|---|
| `de7fa0a3c` | b11514, 2026-10-08 | 13.4.2 | **Current default, and the tag for Qwen3.8-Flash-Next and other `TOP_K` models.** Everything `a7fb71fab` has, plus the MMQ out-of-bounds-read fix (`fc9ce6b9d`; crash-class, present in every earlier tag since July), radix top-k and indexer tiling (Flash-Next top-k at 34k tokens 5.8 s → 0.9 s), faster GDN prefill. Same pin as the HIP b11514 images. Qwen3.5/3.6/3.8 differ from `a7fb71fab` at their numeric floor (KLD ~0.008, GDN kernel reorder); Gemma, Nemotron and Muse are bit-identical. |
| `a7fb71fab` | b11401, 2026-10-05 | 13.4.2 | Superseded by `de7fa0a3c` before any measurement was made on it. Was the first tag with race-free `DeviceTopK` (CCCL 3.4.3), `llama-imatrix --nextn` / `-md` and the deprecation sweep. |
| `26394b4e6-r2` | b11067+7, 2026-09-21 | 13.4.1 | **Do not use for `TOP_K` models** — racy `DeviceTopK` (CCCL 3.4.2, see [Base image](#base-image)). Otherwise `26394b4e6` with the portable CPU backend. |
| `26394b4e6` | b11067+7, 2026-09-21 | 13.4.1 | **Do not use for `TOP_K` models** — racy `DeviceTopK` (CCCL 3.4.2), the very thing this tag was cut for. Was the Qwen3.8-Flash-Next (`qwen4exp`) tag: needs `41abbfd59` rms_norm+mul fusion, `37b53fd45` hc ops, `3cf03257f` sparse FA and the `b23701f77` argsort fix, all post-`3057bb66c`. Also carries `ce8caa6e6` (Gemma 4 FA tile retune) — same precision, different summation order, so Gemma 4 numbers from this tag are not comparable with the 13.0 tags. |
| `3057bb66c` | b10931, 2026-09-12 | 13.0.3 | The existing measurement set (Qwen3.5/3.6/3.8, Gemma 4, Nemotron 3 Nano, Muse Glimmer); B200 campaign. Both CUDA correctness fixes (`b74f590ea`, `73a43d1f6`). |
| `91f6a6cf3` | b10883, 2026-09-09 | 13.0.3 | Superseded by `3057bb66c`: identical CUDA path (the two commits between them are HIP-only), sm_80/sm_90 only. |
| `0cae43063` | b10839, 2026-09-07 | 13.0.3 | Superseded — sm_80/sm_90 only, upstream-default FA pairs only. |
| `c589f0ed1`, `fe2120bc9` | b10688 / b10740 | 13.0.3 | Superseded; no reason to use either. |
| `a30273376` | b10545, 2026-08-20 | 13.0.3 | Reproduces the 2026-08-29 Qwen3.8 / fusion / Muse Glimmer numbers; FA divergent-barrier UB (`b74f590ea`) was live in-path. |

The standing test before cutting a new tag, run against the tag you would replace:

```bash
git log --oneline <pin>..origin/master -- ggml/src/ggml-cuda tools/perplexity
```

Empty means the pin is fine. Only three things justify a bump: a model the pin cannot
load, a correctness fix that is actually on the measurement path (read the kernel's
dispatch gate, not the commit title), or a new quant type. If you bump mid-campaign,
re-measure the whole set.

## Building a pin

Tag push encodes the pin:

```bash
git tag llama-a30273376 && git push origin llama-a30273376
```

Or run the workflow manually with a `ref` input. `:latest` is opt-in only — pinned
tags are the product, and a moving `:latest` would defeat the reproducibility the pin
exists for.

**Rebuilding an existing pin takes a revision suffix, never an overwrite.** When the
image changes but the llama.cpp pin does not (a build-flag fix, a bootstrap fix), run
the workflow with `image_rev`:

```bash
gh workflow run build.yml -f ref=26394b4e6 -f image_rev=r2
```

That publishes `:26394b4e6-r2` and leaves `:26394b4e6` byte-identical, so every number
already measured on it still points at the same image. The workflow enforces this: it
refuses to build a tag that is already published, before the hour of compiling.
Same convention as `llamacpp-hip`.

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
