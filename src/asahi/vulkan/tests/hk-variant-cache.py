#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run real cache round trips and process-isolated compiler-key checks."""
import os
import subprocess
import sys
import tempfile

binary = sys.argv[1]
env = {k: v for k, v in os.environ.items()
       if not k.startswith(("AGX_", "HK_APPLE9_", "MESA_SHADER_CACHE"))}
with tempfile.TemporaryDirectory(prefix="hk-variant-cache-") as directory:
    env["MESA_SHADER_CACHE_DIR"] = directory
    subprocess.run([binary], env=env, check=True)
    subprocess.run([binary, "--cold-process"], env=env, check=True)
    subprocess.run([binary, "--warm-process"], env=env, check=True)
    verify = subprocess.run([binary, "--verify-process"],
                            env=env | {"HK_APPLE9_VARIANT_CACHE_VERIFY": "1"},
                            capture_output=True, text=True, check=True)
    assert "cache mismatch" not in verify.stderr, verify.stderr
    print("verification mode recompiled the warm entry without a mismatch")

    def key(**settings):
        return subprocess.check_output([binary, "--key"],
                                       env=env | settings, text=True).strip()

    original = key()
    knobs = subprocess.check_output([binary, "--codegen-knobs"],
                                    env=env, text=True).splitlines()
    for knob in knobs:
        zero, one = key(**{knob: "0"}), key(**{knob: "1"})
        assert len({original, zero, one}) == 3, knob
    assert key(HK_APPLE9_VARIANT_CACHE_VERIFY="1") == original
    assert key(AGX_APPLE9_TRACE="first/path") == key(AGX_APPLE9_TRACE="second/path")
    assert key(AGX_APPLE9_TRACE="first/path") != original
    assert key(HK_APPLE9_UNKNOWN_TEST="1") != original
    assert key(HK_APPLE9_UNKNOWN_TEST="1") == key(HK_APPLE9_UNKNOWN_TEST="2")
    assert key(UNRELATED_TEST="1") == original
    assert key(AGX_APPLE9_GATED="1", AGX_APPLE9_LICM="0") == key(
        AGX_APPLE9_LICM="0", AGX_APPLE9_GATED="1")
    print(f"cache keys: all {len(knobs)} active codegen knob values, ignored verification, "
          "name-only tracing/unknown knobs and environment order pass")
