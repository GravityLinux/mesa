#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Feed the exact kernel formatter's output to Mesa's production parser."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--kernel-repo', type=Path, required=True)
p.add_argument('--kernel-commit', required=True)
p.add_argument('--parser', type=Path, required=True)
p.add_argument('--class-def', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
source = subprocess.check_output(['git', 'show', a.kernel_commit+':drivers/gpu/drm/asahi/g16_mesh.rs'], cwd=a.kernel_repo)
generations = [0, 1, 127, 256, (1 << 56) - 1]
modes = [0, 1, 2, 3, 4, 5, 255]
with tempfile.TemporaryDirectory(prefix='native-mesh-contract-') as tmp:
    root = Path(tmp)
    (root/'g16_mesh.rs').write_bytes(source)
    (root/'main.rs').write_text('''mod g16_mesh;
fn main() {
    for generation in [0, 1, 127, 256, (1u64 << 56) - 1] {
        for mode in [0, 1, 2, 3, 4, 5, 255] {
            let mut line = String::new();
            g16_mesh::advertise(generation, mode, &mut line).unwrap();
            assert!(line.len() <= g16_mesh::ADVERTISEMENT_MAX);
            print!("{}", line);
        }
    }
}
''')
    subprocess.run(['rustc', '--edition=2021', '-A', 'dead_code', str(root/'main.rs'), '-o', str(root/'format')], check=True)
    lines = subprocess.check_output([str(root/'format')])
result = subprocess.check_output([str(a.parser.resolve()), '--parse-stdin'], input=lines)
expected = ''.join(f'{int(mode == 4)} {generation if mode == 4 else 0}\n'
                   for generation in generations for mode in modes).encode()
assert result == expected, (result, expected)
import re
classes = re.findall(r'^HK_NATIVE_INDEX_SHAPE\((\d+), (\d+), (\d+)\)$', a.class_def.read_text(), re.M)
canonical = ''.join('T%sV%sP%s\n' % row for row in classes).encode()
digest = hashlib.sha256(canonical).hexdigest()
assert all(b'classes='+digest.encode()+b'\n' in line+b'\n' for line in lines.splitlines() if b'profile=qualified' in line)
a.out.write_text(json.dumps(dict(verdict='PASS', kernel_commit=a.kernel_commit,
    kernel_source_sha256=hashlib.sha256(source).hexdigest(),
    formatter_lines=lines.decode().splitlines(), parser_rows=result.decode().splitlines(),
    cases=len(generations)*len(modes), class_sha256=digest,
    generation_max=(1 << 56)-1, device_access=False), indent=2)+'\n')
print('PASS: 35 exact-kernel formatter lines, class hash, all profiles and generation boundaries')
