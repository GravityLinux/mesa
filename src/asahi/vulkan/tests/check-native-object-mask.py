#!/usr/bin/env python3
# Copyright 2026 Asahi Linux Contributors
# SPDX-License-Identifier: MIT
"""Check encoded caller masks, seeded from the actual object dispatch packet.

This models the established structured-if semantics, not GPU execution. It
deliberately cannot invent lanes absent from the launch. No opcode byte scan
or assumed initial ALL_LANES is used: listings are checked against full binaries.
"""
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def opcodes():
    p = Path(__file__).resolve().parents[2] / 'compiler/agx_apple9_ir.h'
    s = p.read_text().split('enum agx_apple9_vir_opcode {')[1].split('};')[0]
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    return [x.strip().removeprefix('AGX_APPLE9_VIR_') for x in s.split(',') if x.strip()]


def caller_mask(rows, entry_lanes):
    assert 0 < entry_lanes <= 16
    active = (1 << entry_lanes) - 1
    stack, trace = [], []
    regs = [[None] * 16 for _ in range(128)]
    by_offset = {r['offset']: i for i, r in enumerate(rows)}
    predicate = None
    pc = 0
    while pc < len(rows):
        r = rows[pc]; pc += 1
        b = r['code']; w = int.from_bytes(b, 'little'); op = r['op']
        write = None
        if op == 'IMM':
            dst = int(r['dst'][1:])
            assert b[0] == (dst & 15) << 4 | 12
            if len(b) == 2:
                value = b[1]
            else:
                assert len(b) == 8
                value = ((b[1] & 127) | ((b[4] & 30) << 6) | ((b[5] & 12) << 9) |
                         b[6] << 13 | (b[7] & 15) << 21 | (b[3] & 254) << 24)
            assert value == r['imm']
            write = [value] * 16
        elif op == 'GET_SR':
            dst = int(r['dst'][1:])
            assert b == bytes([((dst & 15) << 4) | 4, 0xa7, 0x10 | ((dst >> 4) << 6), 6,
                               ((dst & 15) << 4) | 3, 0, (dst >> 4) << 6, 1])
            write = list(range(16))
        elif op == 'IOR':
            assert r['src'][0] == r['src'][1]
            write = list(regs[int(r['src'][0][1:])])
        elif op == 'PREDICATE_COMPARE':
            short = len(b) == 6
            assert short or len(b) == 10
            assert b[0] == 0x0a and b[1] & 1
            a = ((w >> 9) & 63) | (((w >> (40 if short else 56)) & 1) << 6)
            assert r['src'][0] == 'r' + str(a)
            literal = (w >> (41 if short else 57)) & 1
            if literal:
                value = (b[3] & 126) | (((w >> 20) & 1) << 7) | ((w >> (36 if short else 52)) & 1)
                rhs = [value] * 16
            else:
                src = ((w >> 25) & 63) | (((w >> (42 if short else 58)) & 1) << 6)
                assert r['src'][1] == 'r' + str(src)
                rhs = regs[src]
            cond = (b[4] if short else b[6]) & 7
            assert cond == r['imm'] and cond in (5, 7)
            predicate = 0
            for lane in range(16):
                if not active & (1 << lane): continue
                assert regs[a][lane] is not None and rhs[lane] is not None
                yes = regs[a][lane] < rhs[lane] if cond == 5 else regs[a][lane] == rhs[lane]
                predicate |= int(yes) << lane
        elif op == 'EXEC_MASK_PUSH':
            assert b in (bytes.fromhex('0f055401'), bytes.fromhex('0f055421'))
            assert predicate is not None
            stack.append(active)
            active &= predicate ^ (0xffff if b[3] & 32 else 0)
        elif op == 'EXEC_MASK_POP':
            assert b == bytes.fromhex('0f0604010000') and stack
            active = stack.pop()
        elif op == 'EXEC_MASK_ELSE':
            assert b == bytes.fromhex('0f040419') and stack
            active = stack[-1] & ~active
        elif op == 'JMP_EXEC_NONE':
            assert b[:3] == bytes.fromhex('0f0154') and len(b) == 10
            target = r['offset'] + int.from_bytes(b[3:9], 'little', signed=True)
            assert target in by_offset
            if not active: pc = by_offset[target]
        elif op == 'WORKGROUP_BARRIER':
            assert not stack
        elif op == 'OBJECT_CALL':
            assert not stack, ('CALL inside an unclosed mask', r['offset'])
            assert b[:3] == bytes.fromhex('8f1054') and len(b) == 10
            target = r['offset'] + int.from_bytes(b[3:9], 'little', signed=True)
            assert target in by_offset and target > r['offset']
            trace.append(dict(offset=r['offset'], op=op, active=hex(active)))
            return active, trace
        else:
            assert op in ('IOR_UNIFORM', 'DEVICE_LOAD', 'SHARED_STORE', 'WAIT'), op
        if write is not None:
            dst = int(r['dst'][1:])
            for lane in range(16):
                if active & (1 << lane): regs[dst][lane] = write[lane]
        if op in ('EXEC_MASK_PUSH', 'EXEC_MASK_POP', 'EXEC_MASK_ELSE', 'WORKGROUP_BARRIER'):
            trace.append(dict(offset=r['offset'], op=op, active=hex(active)))
    raise AssertionError('no collective CALL reached')


def main():
    env = {k: v for k, v in os.environ.items() if not k.startswith(('HK_', 'AGX_', 'NIR_'))}
    env['AGX_APPLE9_ASM'] = '1'
    run = subprocess.run([sys.argv[1], '--mask-fixtures'], env=env, capture_output=True, text=True)
    if run.returncode:
        sys.stdout.write(run.stdout); sys.stderr.write(run.stderr)
        return run.returncode
    names = opcodes(); programs = []; rows = None
    for line in run.stderr.splitlines():
        if line.startswith('APPLE9_ASM_BEGIN'):
            assert rows is None; rows = []
        elif line == 'APPLE9_ASM_END':
            assert rows is not None; programs.append(rows); rows = None
        elif line.startswith('APPLE9_ASM '):
            r = dict(x.split('=', 1) for x in line.split()[1:])
            r.update(op=names[int(r['op'])], offset=int(r['offset']), imm=int(r['imm'], 0),
                     src=r['src'].split(',') if r['src'] else [], code=bytes.fromhex(r['bytes']))
            assert r['offset'] == sum(len(x['code']) for x in rows)
            rows.append(r)
    fixtures = [dict(x.split('=',1) for x in l.split()[1:])
                for l in run.stdout.splitlines() if l.startswith('object-mask ')]
    assert len(fixtures) == len(programs) == 3
    assert [int(f['logical']) for f in fixtures] == [1,3,16]
    results = []
    for f,rows in zip(fixtures, programs):
        assert b''.join(r['code'] for r in rows) == bytes.fromhex(f['binary'])
        total, y, z, local, ly, lz = map(int, f['dispatch'].split(','))
        assert (total, local) == (int(f['physical']), int(f['physical']))
        assert (y,z,ly,lz) == (1,1,1,1)
        active, trace = caller_mask(rows, local)
        assert active == 0xffff, ('insufficient CALL participation', f['logical'], hex(active))
        # This is the historical broken packet, not an invented shader change.
        assert caller_mask(rows, 1)[0] == 1
        # A missing encoded mask pop must not pass merely because NIR reconverges.
        bad = copy.deepcopy(rows)
        pop = next(r for r in reversed(bad[:next(i for i,r in enumerate(bad) if r['op']=='OBJECT_CALL')]) if r['op']=='EXEC_MASK_POP')
        pop.update(op='WAIT', code=bytes.fromhex('0600'))
        try: caller_mask(bad, local)
        except AssertionError: pass
        else: raise AssertionError('missing POP mutation escaped')
        results.append(dict(logical=int(f['logical']), entry=hex((1<<local)-1), call=hex(active), trace=trace))
    print(json.dumps(dict(passed=True, hardware_execution=False, fixtures=results), indent=2))
    return 0

if __name__ == '__main__':
    sys.exit(main())
