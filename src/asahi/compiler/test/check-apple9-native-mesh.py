#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Check complete compiler output using the existing generic ASM listing.

The executable executes lowered NIR and checks the local output components. This check
independently decodes the six units at emitted instruction boundaries, follows
their allocated input definitions, and checks their enclosing execution masks.
It is a bounded def-use check, not an emulator for the ordinary GPU program.
No compiler callback, replay path, or instruction-byte search is used.
"""
import copy
import os
from pathlib import Path
import re
import subprocess
import sys


def opcode_names():
    header = (Path(__file__).resolve().parents[1] / "agx_apple9_ir.h").read_text()
    body = header.split("enum agx_apple9_vir_opcode {")[1].split("};")[0]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    names = [x.strip() for x in body.split(",") if x.strip()]
    assert all(re.fullmatch(r"AGX_APPLE9_VIR_\w+", x) for x in names)
    return [x.removeprefix("AGX_APPLE9_VIR_") for x in names]


def check_program(rows, channels=None):
    if channels is None: channels = [[c, 32 + 16*c] for c in range(4)]
    units = [i for i, r in enumerate(rows) if r["op"].startswith("MESH_")]
    assert [rows[i]["op"] for i in units] == [
        "MESH_CONTROL", *["MESH_SUBMIT_CHANNEL"] * len(channels), "MESH_SUBMIT_SCALAR"]

    def decoded(i):
        r = rows[i]
        code = r["code"]
        assert len(code) == 6
        word = int.from_bytes(code, "little")
        if r["op"] == "MESH_CONTROL":
            mask = (255 << 24) | (255 << 33) | (1 << 42)
            assert word & ~mask == 0x300000540357
            a, b = (word >> 24) & 255, (word >> 33) & 255
            assert a % 2 == b % 2 == 0
            assert word & (1 << 42)
            regs = [f"r{a // 2}", f"r{b // 2}"]
        else:
            assert word & ~((255 << 24) | (3 << 35)) == 0x0006005406d7
            base, release, mode = (word >> 24) & 127, (word >> 31) & 1, (word >> 35) & 3
            channel = r["op"] == "MESH_SUBMIT_CHANNEL"
            assert mode == (2 if channel else 0) and release == channel
            regs = [f"r{base}", f"r{base + 1}"]
        assert regs == r["src"], (regs, r)
        assert all(int(x[1:]) < 96 for x in regs)
        return regs

    def value(reg, before):
        for i in range(before - 1, -1, -1):
            r = rows[i]
            if r["op"].startswith("MESH_"):
                clobber = decoded(i)
                if r["op"] == "MESH_CONTROL":
                    clobber = clobber[:1]
                assert reg not in clobber, ("read of released/cleared source", reg, i)
            if r["dst"] != reg:
                continue
            if r["op"] == "IMM":
                if len(r["code"]) == 2:
                    assert r["code"][0] == (int(reg[1:]) << 4) | 12
                    assert r["code"][1] == r["imm"] < 128
                else:
                    code = r["code"]; dst = int(reg[1:])
                    assert len(code) == 8 and dst < 64
                    assert code[0] == ((dst & 15) << 4) | 12
                    assert code[2] == ((dst >> 4) << 6) | 2 and code[1] & 128
                    literal = ((code[1] & 127) | ((code[4] & 30) << 6) |
                               ((code[5] & 12) << 9) | (code[6] << 13) |
                               ((code[7] & 15) << 21) | ((code[3] & 254) << 24))
                    assert literal == r["imm"]
                return r["imm"]
            if r["op"] == "GET_SR":
                assert r["imm"] == 0xA7
                return "lane"
            if r["op"] == "IOR":
                a, b = [value(s, i) for s in r["src"]]
                assert a == b  # The only such operations on this slice are copies.
                return a
            if r["op"] == "SHARED_LOAD":
                offset = 4 * value(r["src"][0], i)
                return ("local", r["imm"], offset)
            raise AssertionError(("unmodelled input definition", r))
        raise AssertionError(("undefined allocated input", reg, before))

    # U0 clears A and preserves B. Check all subsequent ordinary reads of A,
    # not just the def-use slices feeding mesh units.
    killed = set(decoded(units[0])[:1])
    for r in rows[units[0] + 1:]:
        assert not killed.intersection(r["src"]), ("post-U0 read", killed, r)
        killed.discard(r["dst"])

    expected = [[("local", 6, 16), ("local", 6, 20)],
                *channels, [("local", 6, 20), 0]]
    for i, want in zip(units, expected):
        got = [value(s, i) for s in decoded(i)]
        assert got == want, (rows[i]["op"], got, want)

    # The application body has reconverged before U0. The two
    # remaining masks must be lane < actual vertex count and lane == 0.
    stack, last_compare = [], None
    unit_masks = []
    for i, r in enumerate(rows):
        if r["op"] == "PREDICATE_COMPARE":
            last_compare = i
        elif r["op"] == "EXEC_MASK_PUSH":
            stack.append((last_compare, r["imm"]))
        elif r["op"] == "EXEC_MASK_POP":
            assert stack
            stack.pop()
        elif i in units:
            unit_masks.append(list(stack))
    assert not stack and unit_masks[0] == []
    assert all(m == unit_masks[1] for m in unit_masks[1:-1])
    for mask, kind in [(unit_masks[1], "channel"), (unit_masks[-1], "scalar")]:
        assert len(mask) == 1
        index, push = mask[0]
        compare = rows[index]
        assert push == 1
        if kind == "channel":
            assert compare["imm"] == 5
            assert [value(s, index) for s in compare["src"]] == ["lane", ("local", 6, 20)]
        else:
            assert compare["imm"] == 7 and len(compare["code"]) == 10
            assert [value(s, index) for s in compare["src"]] == ["lane"]
            # The extended integer compare's zero literal operand.
            word = int.from_bytes(compare["code"], "little")
            assert (word >> 57) & 1
            assert ((word >> 24) & 127) == 0 and ((word >> 52) & 1) == 0
            assert ((word >> 20) & 1) == 0
    return [rows[i]["code"].hex() for i in units]


def mutation_checks(rows):
    control = next(i for i, r in enumerate(rows) if r["op"] == "MESH_CONTROL")
    a, b = rows[control]["src"]
    assert a != b
    for coherent in (False, True):
        bad = copy.deepcopy(rows)
        word = int.from_bytes(bad[control]["code"], "little")
        # Allocate B to the primitive-count register instead of vertex count.
        word = (word & ~(255 << 33)) | (2 * int(a[1:]) << 33)
        bad[control]["code"] = word.to_bytes(6, "little")
        if coherent:
            bad[control]["src"][1] = a
        try:
            check_program(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("mis-allocation mutation escaped checker")
    bad = copy.deepcopy(rows)
    channel = next(r for r in bad if r["op"] == "MESH_SUBMIT_CHANNEL")
    channel["src"].reverse()
    try:
        check_program(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("swapped tuple mutation escaped checker")

    # Mutate the last definition of the actual scalar neighbor, after all
    # channel submissions. Do not change a shared zero definition that would
    # make an earlier channel fail before testing the scalar's own contract.
    scalar = next(i for i, r in enumerate(rows) if r["op"] == "MESH_SUBMIT_SCALAR")
    reg = rows[scalar]["src"][1]
    definition = max(i for i in range(scalar) if rows[i]["dst"] == reg)
    assert definition > max(i for i, r in enumerate(rows) if r["op"] == "MESH_SUBMIT_CHANNEL")
    for mutation in ("missing-source", "undefined", "garbage"):
        bad = copy.deepcopy(rows)
        if mutation == "missing-source":
            bad[scalar]["src"].pop()
        elif mutation == "undefined":
            del bad[definition]
        else:
            # Replace only the last allocated definition with a valid IMM 0x20.
            dst = int(reg[1:]); assert dst < 64
            code = bytes([(dst & 15) << 4 | 12, 0xa0,
                          (dst >> 4) << 6 | 2, 0, 0, 0, 0, 0])
            bad[definition].update(op="IMM", src=[], code=code, imm=0x20)
        try:
            check_program(bad)
        except AssertionError as error:
            if mutation == "garbage":
                assert error.args[0][0] == "MESH_SUBMIT_SCALAR", error
            elif mutation == "undefined":
                assert error.args[0][0] in ("read of released/cleared source",
                                             "undefined allocated input"), error
                assert error.args[0][1] == reg, error
        else:
            raise AssertionError(f"scalar base+1 {mutation} mutation escaped checker")


def main():
    run = subprocess.run(sys.argv[1:], env={**os.environ, "AGX_APPLE9_ASM": "1"},
                         capture_output=True, text=True)
    if run.returncode:
        sys.stdout.write(run.stdout)
        sys.stderr.write(run.stderr)
        return run.returncode
    fixtures = [x for x in run.stdout.splitlines() if x.startswith("fixture=")]
    names = opcode_names()
    rows, checked = None, []
    programs = []
    for line in run.stderr.splitlines():
        if line.startswith("APPLE9_ASM_BEGIN"):
            assert rows is None
            rows = []
        elif line == "APPLE9_ASM_END":
            match = re.search(r"channels=(\S+)", fixtures[len(checked)])
            channels = [list(map(int, x.split(':'))) for x in match[1].split(',')] if match else None
            checked.append(check_program(rows, channels))
            programs.append(rows)
            rows = None
        elif line.startswith("APPLE9_ASM offset="):
            assert rows is not None
            r = dict(x.split("=", 1) for x in line.split()[1:])
            r.update(op=names[int(r["op"])], imm=int(r["imm"], 0),
                     offset=int(r["offset"]), code=bytes.fromhex(r["bytes"]),
                     src=r["src"].split(",") if r["src"] else [])
            assert r["offset"] == (rows[-1]["offset"] + len(rows[-1]["code"]) if rows else 0)
            rows.append(r)
    fixtures = [x for x in run.stdout.splitlines() if x.startswith("fixture=")]
    assert rows is None and len(checked) == len(fixtures) and len(fixtures) >= 4
    assert any("fixture=H1_BASE " in x for x in fixtures)
    assert any("fixture=H1_SHIFT_X " in x for x in fixtures)
    base = next(i for i, x in enumerate(fixtures)
                if "fixture=H1_BASE " in x and "pressure=0" in x)
    pressured = next(i for i, x in enumerate(fixtures) if "pressure=1" in x)
    def gprs(line):
        return int(re.search(r"gprs=(\d+)", line).group(1))
    assert gprs(fixtures[pressured]) > gprs(fixtures[base])
    mutation_checks(programs[base])
    mutation_checks(programs[pressured])
    wide = next(i for i, x in enumerate(fixtures) if "fixture=ATTR_LIMIT " in x)
    channels = [list(map(int, x.split(':'))) for x in
                re.search(r"channels=(\S+)", fixtures[wide])[1].split(',')]
    final_channel = max(i for i, r in enumerate(programs[wide]) if r['op'] == 'MESH_SUBMIT_CHANNEL')
    plane = max(i for i, r in enumerate(programs[wide][:final_channel])
                if r['op'] == 'IMM' and r['imm'] >= 128)
    for coherent in (False, True):
        bad = copy.deepcopy(programs[wide]);code = bytearray(bad[plane]['code']);code[1] ^= 1
        bad[plane]['code'] = bytes(code)
        if coherent:bad[plane]['imm'] ^= 1
        try:check_program(bad, channels)
        except AssertionError:pass
        else:raise AssertionError('corrupt high plane immediate accepted')
    sys.stdout.write(run.stdout)
    print("six allocation, six scalar base+1 and two high-plane immediate mutations rejected; live pressure increases the GPR high-water mark")
    print(f"checked {len(checked)} complete programs: boundaries, U units, allocated inputs, masks")
    print("first program U units:", " ".join(checked[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
