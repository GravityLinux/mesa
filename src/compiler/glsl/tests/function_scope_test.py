#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the ESSL 1.00 function-body scope and later-language restrictions."""
import pathlib
import subprocess
import sys
import tempfile

cases = [
    ('shadow_parameter', 'int f(int x) { int x = 7; return x; }', 'f(4)', 'essl100'),
    ('nested_shadow', 'int f(int x) { { int x = 7; return x; } }', 'f(4)', 'all'),
    ('duplicate_local', 'int f(int x) { int y = 1; int y = 2; return y; }', 'f(4)', 'none'),
    ('duplicate_parameter', 'int f(int x, int x) { return x; }', 'f(4, 5)', 'none'),
]
failed = False
with tempfile.TemporaryDirectory(prefix='glsl-function-scope-') as directory:
    for version in (100, 110, 300):
        for name, function, call, expected in cases:
            suffix = ' es' if version == 300 else ''
            precision = 'precision highp float;\nprecision highp int;\n' if version != 110 else ''
            path = pathlib.Path(directory) / f'{name}-{version}.vert'
            path.write_text(f'#version {version}{suffix}\n{precision}{function}\n'
                            f'void main() {{ gl_Position = vec4(float({call})); }}\n')
            result = subprocess.run([sys.argv[1], '--version', str(version), '--just-log', str(path)],
                                    text=True, capture_output=True)
            should_compile = expected == 'all' or (expected == 'essl100' and version == 100)
            # The standalone compiler reports shader errors in its log and
            # returns zero even for a rejected shader.
            compiled = 'error:' not in (result.stdout + result.stderr)
            if result.returncode != 0 or compiled != should_compile:
                print(f'FAIL {name} / {version}: expected compile={should_compile}')
                print(result.stdout + result.stderr)
                failed = True
if failed:
    sys.exit(1)
print('12 function-scope checks passed')
