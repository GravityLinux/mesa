#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preserve every dEQP status, including partial runs and unsupported cases."""
import argparse
import collections
import json
import pathlib
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('qpa', type=pathlib.Path)
parser.add_argument('--cts-source', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path)
args = parser.parse_args()
sys.path.insert(0, str(args.cts_source / 'scripts' / 'log'))
from log_parser import BatchResultParser

results = BatchResultParser().parseFile(str(args.qpa))
with args.qpa.open('rb') as stream:
    stream.seek(0, 2)
    stream.seek(max(0, stream.tell() - 1024))
    complete = b'#endSession' in stream.read()
summary = {
    'log': str(args.qpa),
    'session_complete': complete,
    'counts': dict(collections.Counter(result.statusCode for result in results)),
    'tests': [{'name': result.name, 'status': result.statusCode,
               'details': result.statusDetails} for result in results],
}
output = args.output or args.qpa.with_suffix('.json')
output.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: value for key, value in summary.items() if key != 'tests'}))
accepted = {'Pass', 'NotSupported', 'QualityWarning', 'CompatibilityWarning'}
if not complete or any(result.statusCode not in accepted for result in results):
    sys.exit(1)
