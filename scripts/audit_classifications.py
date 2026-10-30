#!/usr/bin/env python3
"""Audit storm-ID classification TXT lists; optionally cross-check a Control CSV.

Uses only the Python standard library. Does not change source lists.
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path


def normalize(raw):
    name = str(raw).strip().strip("\"'")
    if not name:
        return ''
    while name.lower().endswith('.npy'):
        name = name[:-4]
    return name + '.npy'


def load_list(path):
    raw = [x.strip() for x in re.split(r'[,\n\r]+', path.read_text(encoding='utf-8')) if x.strip()]
    normalized = [normalize(x) for x in raw]
    counts = Counter(normalized)
    doubled = [x for x in raw if x.lower().endswith('.npy.npy')]
    print(f'{path}: {len(raw):,} entries / {len(counts):,} unique; '
          f'{len(raw)-len(counts):,} duplicates; {len(doubled):,} doubled suffixes')
    if doubled:
        print('  Example doubled suffix:', doubled[0])
    return set(counts)


def control_keys(path, column='storm_key'):
    ids = set()
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f'Column {column!r} not found. Available columns: {reader.fieldnames}')
        for row in reader:
            if row[column]:
                ids.add(normalize(row[column]))
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True, help='First classification TXT')
    parser.add_argument('--right', type=Path, help='Second classification TXT')
    parser.add_argument('--require-disjoint', action='store_true', help='Exit nonzero if lists intersect')
    parser.add_argument('--control', type=Path, help='Optional Control prediction CSV for availability audit')
    parser.add_argument('--control-key', default='storm_key')
    parser.add_argument('--missing-out', type=Path, help='Optional TXT output for missing storm IDs')
    args = parser.parse_args()

    left = load_list(args.left)
    if args.right:
        right = load_list(args.right)
        overlap = left & right
        print(f'Overlap: {len(overlap):,} unique storm IDs')
        print(f'Left-only: {len(left - right):,}; right-only: {len(right - left):,}')
        if overlap:
            print('Examples:', ', '.join(sorted(overlap)[:5]))
        if args.require_disjoint and overlap:
            raise SystemExit('FAILED: classifications are not disjoint')
    elif args.require_disjoint:
        parser.error('--require-disjoint requires --right')

    if args.control:
        available = control_keys(args.control, args.control_key)
        for label, ids in [('left', left)] + ([('right', right)] if args.right else []):
            missing = ids - available
            print(f'{label} in Control CSV: {len(ids & available):,}; '
                  f'not in Control CSV: {len(missing):,}')
            if label == 'left' and args.missing_out:
                args.missing_out.parent.mkdir(parents=True, exist_ok=True)
                args.missing_out.write_text('\n'.join(sorted(missing)) + ('\n' if missing else ''), encoding='utf-8')
                print('Wrote missing IDs:', args.missing_out)


if __name__ == '__main__':
    main()
