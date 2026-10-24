#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare complete serialized driver shader objects; never mask bytes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--parent', type=Path, required=True)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--qualification', type=Path, required=True)
p.add_argument('--kill', type=Path, action='append', default=[])
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
qualified = set(json.loads(a.qualification.read_text())['proposed_default_shapes'])
assert len(qualified) == 31 and 'T16V16P16' not in qualified

def records(root):
    rows = list(csv.DictReader((root/'PROGRAMS.tsv').open(), delimiter='\t'))
    result = {}
    for row in rows:
        name = row['case'] + '-' + row['object'] + '.bin'
        content = (root/name).read_bytes()
        assert len(content) == int(row['bytes']) and name not in result
        result[name] = (row, content)
    assert len(list(root.glob('*.bin'))) == len(rows)
    return result

parent, candidate = records(a.parent), records(a.candidate)
assert parent.keys() == candidate.keys()
changed, identical, native_cases, manifest = [], [], set(), {}
for name, (old, old_bytes) in parent.items():
    new, new_bytes = candidate[name]
    expected = old['case'].endswith('-R0') and old['case'][:-3] in qualified
    assert old['native'] == '0'
    assert old['stage'] == new['stage'] and old['object'] == new['object']
    assert int(new['native']) == int(expected), (name, 'route', new['native'], expected)
    if expected:
        assert old_bytes != new_bytes, (name, 'missing native variant')
        changed.append(name)
        native_cases.add(old['case'][:-3])
    else:
        assert old == new and old_bytes == new_bytes, (name, 'nonqualified changed')
        identical.append(name)
    manifest[name] = {'qualified': expected, 'parent_sha256': hashlib.sha256(old_bytes).hexdigest(),
                      'candidate_sha256': hashlib.sha256(new_bytes).hexdigest()}
assert native_cases == qualified
kills = []
for root in a.kill:
    killed = records(root)
    assert parent.keys() == killed.keys()
    for name, (old, old_bytes) in parent.items():
        row, content = killed[name]
        assert old == row and old_bytes == content, (root, name, 'kill changed')
    kills.append({'path': str(root), 'identical_objects': len(killed)})
result = {'verdict': 'PASS', 'parent': str(a.parent), 'candidate': str(a.candidate),
          'pipeline_cases': len({v[0]['case'] for v in parent.values()}),
          'shader_objects': len(parent), 'nonqualified_identical_objects': len(identical),
          'qualified_changed_objects': len(changed), 'native_classes': sorted(native_cases),
          'kill_controls': kills, 'masked_bytes': 0, 'GPU_execution': False,
          'scope': 'exhaustive bounded declaration grid plus listed interface, workgroup, GS, TES and VS-FS fixtures; not a proof over arbitrary shader programs',
          'objects': manifest}
a.out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='objects'}, indent=2))
