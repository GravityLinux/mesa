#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Hash the exact .def used to compile the native-index class table."""
import hashlib
from pathlib import Path
import re
import sys

text = re.sub(r'/\*.*?\*/', '', Path(sys.argv[1]).read_text(), flags=re.S)
classes = []
for line in text.splitlines():
    if not line.strip():
        continue
    match = re.fullmatch(r'HK_NATIVE_INDEX_SHAPE\(([1-9][0-9]*), ([1-9][0-9]*), ([1-9][0-9]*)\)', line.strip())
    if not match:
        raise ValueError('invalid native-index class: '+line)
    classes.append('T%sV%sP%s\n' % match.groups())
if not classes or len(set(classes)) != len(classes):
    raise ValueError('empty or duplicate native-index classes')
digest = hashlib.sha256(''.join(classes).encode('ascii')).hexdigest()
Path(sys.argv[2]).write_text('/* Generated from hk_apple9_native_index_shapes.def. */\n'
                           '#define HK_APPLE9_NATIVE_INDEX_CLASSES_SHA256 "'+digest+'"\n')
