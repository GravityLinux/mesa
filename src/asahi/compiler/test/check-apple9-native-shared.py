#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Check the allocated local-root field of every shared memory instruction.

The executable checks shared/output address separation and dataflow in NIR.
This independently checks the root encoded in loads, stores and atomics after
selection/allocation. It does not simulate the GPU or qualify a native launch.
"""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

spec = importlib.util.spec_from_file_location(
    "mesh_check", Path(__file__).with_name("check-apple9-native-mesh.py"))
mesh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mesh)


def check(rows, root):
    counts = dict.fromkeys(("SHARED_LOAD", "SHARED_STORE", "SHARED_ATOMIC"), 0)
    for row in rows:
        if row["op"] not in counts:
            continue
        counts[row["op"]] += 1
        code = row["code"]
        atomic = row["op"] == "SHARED_ATOMIC"
        assert len(code) == (12 if atomic else 14)
        assert code[:3] == bytes([0x67 if atomic or row["op"] == "SHARED_LOAD" else 0xe7,
                                 3 if atomic else 2, 0x54])
        # Atomic roots use a split halfword index; ordinary local memory uses
        # the root's four-halfword selector directly in byte 4.
        encoded = (((code[4] << 1) | (code[3] >> 7)) // 4
                   if atomic else code[4] // 4)
        assert not code[4] % (2 if atomic else 4)
        if atomic:
            assert not code[3] & 128
        assert encoded == row["imm"] == root, (row, encoded, root)
    assert all(counts.values()), counts
    return counts


def main():
    run = subprocess.run(sys.argv[1:], env={**os.environ, "AGX_APPLE9_ASM": "1"},
                         capture_output=True, text=True)
    if run.returncode:
        sys.stdout.write(run.stdout)
        sys.stderr.write(run.stderr)
        return run.returncode
    names = mesh.opcode_names()
    programs, rows = [], None
    for line in run.stderr.splitlines():
        if line.startswith("APPLE9_ASM_BEGIN"):
            assert rows is None and line == "APPLE9_ASM_BEGIN stage=7"
            rows = []
        elif line == "APPLE9_ASM_END":
            programs.append(rows)
            rows = None
        elif line.startswith("APPLE9_ASM offset="):
            r = dict(x.split("=", 1) for x in line.split()[1:])
            r.update(op=names[int(r["op"])], imm=int(r["imm"], 0),
                     offset=int(r["offset"]), code=bytes.fromhex(r["bytes"]))
            assert rows is not None
            assert r["offset"] == (rows[-1]["offset"] + len(rows[-1]["code"]) if rows else 0)
            rows.append(r)
    fixtures = [dict(x.split("=", 1) for x in line.split())
                for line in run.stdout.splitlines() if line.startswith("shared_fixture=")]
    assert rows is None and len(programs) == len(fixtures) == 14
    for program, fixture in zip(programs, fixtures):
        check(program, int(fixture["root"]))
    guards = 0
    for op in ("SHARED_LOAD", "SHARED_STORE", "SHARED_ATOMIC"):
        for coherent in (False, True):
            bad = copy.deepcopy(programs[0])
            row = next(r for r in bad if r["op"] == op)
            code = bytearray(row["code"])
            code[4] = 10 if op == "SHARED_ATOMIC" else 20
            row["code"] = bytes(code)
            if coherent:
                row["imm"] = 5  # The ordinary compute shared root.
            try:
                check(bad, 6)
            except AssertionError:
                guards += 1
            else:
                raise AssertionError("wrong shared root escaped checker")
    sys.stdout.write(run.stdout)
    print(f"checked {len(programs)} allocated programs; {guards} wrong-root mutations rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
