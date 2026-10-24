#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise real cached parsing and compiler keys in independent processes."""
import os
import subprocess
import sys

probe = sys.argv[1]
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('AGX_', 'HK_'))}

def read(**settings):
    output = subprocess.check_output([probe], env=env | settings, text=True)
    return tuple(map(int, output.split()))

base, flags = read()
assert flags == 0
# Presence-only aliases keep enabling with "0" and the empty string.
presence = {
    'asm': 'AGX_APPLE9_ASM', 'binary': 'AGX_APPLE9_DUMP_BINARY',
    'stats': 'AGX_APPLE9_STATS', 'trace': 'AGX_APPLE9_TRACE',
    'nir': 'AGX_APPLE9_TRACE_NIR', 'final_nir': 'AGX_APPLE9_TRACE_FINAL_NIR',
    'shared_ra': 'AGX_APPLE9_TRACE_SHARED_RA', 'ra_audit': 'AGX_APPLE9_RA_AUDIT',
    'ra_webs': 'AGX_APPLE9_RA_WEBS', 'pressure': 'HK_APPLE9_PRESSURE_STATS',
    'vectorize': 'HK_APPLE9_TRACE_VECTORIZE', 'eager': 'HK_APPLE9_EAGER_TRACE',
}
boolean = {
    'spill_loops': 'AGX_SPILL_TRACE_LOOPS', 'merge': 'HK_APPLE9_MERGE_TRACE',
    'tiling': 'HK_TILING_ELISION_STATS',
    'launch_check': 'HK_APPLE9_LAUNCH_CHECK', 'bo_labels': 'AGX_BO_TRACE_LABELS',
    'replay': 'HK_REPLAY_DEBUG',
}
seen = 0
for name, alias in (presence | boolean).items():
    key, flag = read(AGX_APPLE9_DEBUG=name)
    assert key == base and flag and not flag & (flag - 1), name
    assert not seen & flag, name
    seen |= flag
    for value in ('1', 'true'):
        assert read(**{alias: value}) == (base, flag), (alias, value)
    if alias in presence.values():
        for value in ('0', ''):
            assert read(**{alias: value}) == (base, flag), (alias, value)
    else:
        for value in ('0', 'false'):
            assert read(**{alias: value}) == (base, 0), (alias, value)
assert read(AGX_APPLE9_DEBUG=','.join(presence | boolean)) == (base, seen)
assert read(AGX_APPLE9_DEBUG='') == (base, 0)
a = read(AGX_APPLE9_DEBUG='asm')[1]
b = read(AGX_APPLE9_STATS='1')[1]
assert read(AGX_APPLE9_DEBUG='asm', AGX_APPLE9_STATS='0') == (base, a | b)
# Runtime-only controls and dump paths share the same compiler key.
for name in ('HK_APPLE9_VARIANT_CACHE_VERIFY', 'HK_APPLE9_DUMP_FRAGMENT_DIR',
             'HK_APPLE9_EAGER_VARIANTS', 'HK_APPLE9_SPARSE_SCRATCH',
             'HK_APPLE9_TESS_BATCH', 'HK_APPLE9_TESS_SCAN_BLOCK',
             'HK_APPLE9_MERGE_RENDERS', 'HK_APPLE9_MERGE_RESUME',
             'HK_APPLE9_MERGE_HOIST_BARRIERS', 'HK_APPLE9_MERGE_MOVE_COMPUTE',
             'HK_APPLE9_FRAGMENT_BARRIER', 'HK_APPLE9_NO_TILING_ELISION',
             'HK_APPLE9_TESS_ROUTE_LOG'):
    for value in ('0', '1', '/tmp/trace-a', '/tmp/trace-b'):
        assert read(**{name: value}) == (base, 0), (name, value)
# Code-generation controls continue to separate cache entries.
for name in ('AGX_SPILL_COALESCE', 'AGX_APPLE9_HOIST_LOADS',
             'AGX_APPLE9_SAFE_HOIST_STAGES', 'HK_APPLE9_NATIVE_MESH',
             'HK_APPLE9_TESS_NATIVE'):
    zero = read(**{name: '0'})[0]
    one = read(**{name: '1'})[0]
    assert len({base, zero, one}) == 3, name
# A longer, unknown name must not be mistaken for a diagnostic alias.
assert read(AGX_APPLE9_ASM_SUFFIX='1')[0] != base
print('19 diagnostic aliases preserve parsing; diagnostics and runtime controls preserve cache keys; codegen keys remain distinct')
