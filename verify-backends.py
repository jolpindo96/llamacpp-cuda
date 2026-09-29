#!/usr/bin/env python3
"""Load ggml's backends the way the llama.cpp tools do and report what loaded.

The backends are dlopen'd modules (GGML_BACKEND_DL). A module that fails to load
is skipped silently in a Release build, so "the binary starts" no longer proves
that CUDA is in use. The tools' own "load_backend: loaded ... backend" lines
cannot prove it either: llama.cpp's logger is asynchronous, and a
`--list-devices` run exits before it drains, so those lines are dropped even
when the load succeeds. (Verified 2026-09-29: LD_DEBUG showed
libggml-cpu-haswell.so loaded while no log line appeared.)

So this asks ggml's registry directly, through the same libggml.so and its
compiled-in GGML_BACKEND_DIR, and reads the CPU variant the host selected from
/proc/self/maps.

Usage: verify-backends.py [--require NAME ...]   NAME is a registry name: CPU, CUDA
Exits 1 if a required backend did not register.
"""
import argparse
import ctypes
import glob
import os
import re
import sys

LIB_DIR = "/opt/llama/lib"


def load_ggml():
    path = os.path.join(LIB_DIR, "libggml.so")
    if not os.path.exists(path):
        found = sorted(glob.glob(os.path.join(LIB_DIR, "libggml.so*")))
        if not found:
            sys.exit(f"no libggml.so in {LIB_DIR}")
        path = found[0]
    ggml = ctypes.CDLL(path)
    ptr, size, name = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p
    for fn, res, args in (
        ("ggml_backend_load_all", None, []),
        ("ggml_backend_reg_count", size, []),
        ("ggml_backend_reg_get", ptr, [size]),
        ("ggml_backend_reg_name", name, [ptr]),
        ("ggml_backend_dev_count", size, []),
        ("ggml_backend_dev_get", ptr, [size]),
        ("ggml_backend_dev_name", name, [ptr]),
        ("ggml_backend_dev_description", name, [ptr]),
    ):
        f = getattr(ggml, fn)
        f.restype, f.argtypes = res, args
    return ggml


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--require", action="append", default=[], metavar="NAME",
                    help="backend that must register (repeatable), e.g. CPU, CUDA")
    args = ap.parse_args()

    ggml = load_ggml()
    ggml.ggml_backend_load_all()

    regs = [ggml.ggml_backend_reg_name(ggml.ggml_backend_reg_get(i)).decode()
            for i in range(ggml.ggml_backend_reg_count())]
    devs = []
    for i in range(ggml.ggml_backend_dev_count()):
        d = ggml.ggml_backend_dev_get(i)
        devs.append((ggml.ggml_backend_dev_name(d).decode(),
                     ggml.ggml_backend_dev_description(d).decode()))
    with open("/proc/self/maps") as f:
        mapped = sorted(set(re.findall(r"/\S*/libggml-(?:cpu-[\w.]+|cuda)\.so", f.read())))

    print("backends:", ", ".join(regs) or "(none)")
    print("modules: ", ", ".join(os.path.basename(m) for m in mapped) or "(none)")
    for dname, desc in devs:
        print(f"device:   {dname} = {desc}")

    missing = [r for r in args.require if r not in regs]
    if missing:
        print("MISSING backend(s): " + ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
