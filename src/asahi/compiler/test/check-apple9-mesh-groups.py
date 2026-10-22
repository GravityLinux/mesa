#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Check the allocated group-count load/compare slice at instruction boundaries.

This is a bounded data-flow oracle for eight volatile stores, not a GPU or
control-flow emulator. Decode load selectors and SELECT operands from bytes;
bind different compute/child grids to detect an otherwise valid wrong root.
"""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

spec = importlib.util.spec_from_file_location(
    'native', Path(__file__).with_name('check-apple9-native-mesh.py'))
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def check(rows, grid):
    stores = [i for i, r in enumerate(rows) if r['op'] == 'DEVICE_STORE'][:8]
    assert len(stores) == 8
    roots = {3: (0x1234, 0x5678, 0x9abc), 9: grid}
    loads = set()

    def value(reg, before):
        i = next(i for i in range(before - 1, -1, -1) if rows[i]['dst'] == reg)
        r = rows[i]
        code = r['code']
        word = int.from_bytes(code, 'little')

        def field(bit, bits=1):
            return (word >> bit) & ((1 << bits) - 1)

        def gpr(index):
            return value('r' + str(index), i)

        if r['op'] == 'IMM':
            if len(code) == 2:
                return code[1]
            assert len(code) == 8
            return ((code[1] & 127) | ((code[4] & 30) << 6) |
                    ((code[5] & 12) << 9) | (code[6] << 13) |
                    ((code[7] & 15) << 21) | ((code[3] & 254) << 24))
        if r['op'] == 'IOR':
            return value(r['src'][0], i) | value(r['src'][1], i)
        if r['op'] == 'DEVICE_LOAD':
            assert len(code) == 14 and code[0] == 0x67 and field(53)
            assert (code[8] & 15) == 1  # one 32-bit component
            root = code[4]
            assert root == r['imm'] and root in roots
            component = gpr(code[5] & 127)
            assert component < 3
            loads.add((root, component))
            return roots[root][component]
        if r['op'] == 'SELECT':
            assert len(code) == 10 and field(48, 4) == 7 and field(34)
            assert not field(55)  # comparison source A is a GPR
            a = gpr(field(9, 6) | (field(56) << 6))
            # Integer comparison literals use bit 52 as their low bit.
            b = ((code[3] & 126) | field(52) | (field(20) << 7)
                 if field(57) else gpr(field(25, 6) | (field(58) << 6)))

            def selected(bit, high, file, discard, cache):
                if field(file):
                    assert field(cache) and not field(high)
                    return field(bit - 1, 7) | (field(discard) << 7)
                return gpr(field(bit, 6) | (field(high) << 6))

            yes = selected(41, 38, 37, 39, 47)
            no = selected(73, 70, 69, 71, 79)
            return yes if a == b else no
        raise AssertionError(('unmodelled definition', r))

    got = []
    for c, i in enumerate(stores):
        r = rows[i]
        assert len(r['src']) == 4
        address = value(r['src'][2], i) | (value(r['src'][3], i) << 32)
        assert address + 4 * value(r['src'][1], i) == 0x20040 + 4 * c
        got.append(value(r['src'][0], i))
    same = [int(a == b) for a, b in zip(grid, (5, 6, 7))]
    assert got == [*grid, int(all(same)), int(not all(same)), *same], (grid, got)
    assert loads == {(9, c) for c in range(3)}, loads


def main():
    run = subprocess.run([*sys.argv[1:], '--num-workgroups'],
                         env={**os.environ, 'AGX_APPLE9_ASM': '1'},
                         capture_output=True, text=True, check=True)
    names = native.opcode_names()
    programs = []
    rows = None
    for line in run.stderr.splitlines():
        if line.startswith('APPLE9_ASM_BEGIN'):
            assert rows is None
            rows = []
        elif line == 'APPLE9_ASM_END':
            programs.append(rows)
            rows = None
        elif line.startswith('APPLE9_ASM offset='):
            r = dict(x.split('=', 1) for x in line.split()[1:])
            r.update(op=names[int(r['op'])], imm=int(r['imm'], 0),
                     code=bytes.fromhex(r['bytes']), offset=int(r['offset']),
                     src=r['src'].split(',') if r['src'] else [])
            assert r['offset'] == (rows[-1]['offset'] + len(rows[-1]['code']) if rows else 0)
            rows.append(r)
    assert rows is None and len(programs) == 4
    # All eight combinations of matching/mismatching XYZ exercise both vector
    # reductions. Distinct values expose swapped components and wrong offsets.
    for rows in programs:
        for mask in range(8):
            check(rows, tuple(x if mask & (1 << c) else x + 10
                              for c, x in enumerate((5, 6, 7))))
        bad = copy.deepcopy(rows)
        for r in bad:
            if r['op'] == 'DEVICE_LOAD' and r['imm'] == 9:
                code = bytearray(r['code'])
                code[4] = 3
                r.update(code=bytes(code), imm=3)
        try:
            check(bad, (5, 6, 7))
        except AssertionError:
            pass
        else:
            raise AssertionError('wrong compute-root mutation escaped checker')
    sys.stdout.write(run.stdout)
    print('32 allocated load/compare checks and 4 wrong-root mutations passed')


if __name__ == '__main__':
    main()
