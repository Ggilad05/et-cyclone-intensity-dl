#!/usr/bin/env python3
"""Reproduce Table 1: cyclone counts by season and geographic region.

The Control storm-ID list defines the study cohort. Six-hourly track
positions are selected from the original 3-hourly tracking NetCDFs by t % 2 == 1.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import xarray as xr

SEASONS = ('DJF', 'MAM', 'JJA', 'SON')
REGIONS = ('North Atlantic', 'North Pacific', 'Mediterranean',
           'South Atlantic', 'Southern Ocean')

REGION_FILES = {
    'North Atlantic': ('original/classification_NH_NA.txt', 'classification_NH_NA.txt'),
    'North Pacific': ('original/classification_NH_NP.txt', 'classification_NH_NP.txt'),
    'Mediterranean': ('original/classification_NH_MED.txt', 'classification_NH_MED.txt'),
    'South Atlantic': ('revised/classification_SH_SA_corrected.txt', 'classification_SH_SA_corrected.txt'),
    'Southern Ocean': ('revised/classification_SH_SO_exclusive.txt', 'classification_SH_SO_exclusive.txt'),
}


def normalize_id(raw: str) -> str:
    value = str(raw).strip().strip('"\'')
    while value.lower().endswith('.npy'):
        value = value[:-4]
    return f'{value}.npy' if value else ''


def load_control(path: Path) -> set[str]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or 'storm_key' not in reader.fieldnames:
            raise ValueError(f'Missing storm_key column: {path}')
        ids = {normalize_id(row['storm_key']) for row in reader}
    ids.discard('')
    return ids


def load_regions(directory: Path) -> dict[str, set[str]]:
    result = {}
    for region, candidates in REGION_FILES.items():
        path = next((directory / name for name in candidates
                     if (directory / name).is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Missing classification for {region} in {directory}')
        values = {normalize_id(x) for x in re.split(r'[,\s]+',
                  path.read_text(encoding='utf-8-sig'))}
        values.discard('')
        result[region] = values
    if result['South Atlantic'] & result['Southern Ocean']:
        raise ValueError('South Atlantic and Southern Ocean lists must be disjoint')
    return result


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-csv', type=Path, default=Path(__file__).resolve().parent /
                        'reference/control_34_storm_keys.csv')
    parser.add_argument('--data-dir', type=Path, default=Path(
                        '/arctic2/shreibshtein/MasterProject/OrCyclones/ERA_5_Track'))
    parser.add_argument('--classifications-dir', type=Path, default=repo_root/'classifications')
    parser.add_argument('--out-dir', type=Path, default=repo_root/'outputs'/'table_1')
    parser.add_argument('--start-year', type=int, default=1979)
    parser.add_argument('--end-year', type=int, default=2020)
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error('start-year must not exceed end-year')
    ids = load_control(args.control_csv)
    ids = {sid for sid in ids if any(f'_{year}_' in sid for year in range(args.start_year, args.end_year+1))}
    membership = load_regions(args.classifications_dir)
    stats = {hemi: {category: [0, 0] for category in SEASONS + REGIONS}
             for hemi in ('NH', 'SH')}
    seen = set()
    files = sorted(args.data_dir.glob('*.nc'))
    if not files:
        raise FileNotFoundError(f'No tracking NetCDF files in {args.data_dir}')

    for path in files:
        match = re.search(r'_(\d{4})_', path.stem)
        season = path.stem.split('_')[0]
        if not match or season not in SEASONS:
            continue
        year = int(match.group(1))
        if not args.start_year <= year <= args.end_year:
            continue
        with xr.open_dataset(path) as ds:
            data = ds['data'].to_numpy()
            track_ids = ds['trackid'].to_numpy()
        if data.ndim != 3 or data.shape[0] < 4:
            raise ValueError(f'Unexpected tracking array shape in {path}: {data.shape}')

        for column, track_id in enumerate(track_ids):
            sid = normalize_id(f'{path.stem}_{int(track_id)}.npy')
            if sid not in ids:
                continue
            if sid in seen:
                raise ValueError(f'Duplicate storm: {sid}')
            seen.add(sid)

            t = data[0, :, column]
            valid6 = np.isfinite(t) & (t != 0) & (np.mod(t, 2) == 1)
            valid6 &= np.isfinite(data[1, :, column]) & np.isfinite(data[2, :, column])
            nsteps = int(np.count_nonzero(valid6))
            if nsteps == 0:
                raise ValueError(f'No valid 6h points for {sid}')
            if '_NH_' in sid:
                hemi = 'NH'
            elif '_SH_' in sid:
                hemi = 'SH'
            else:
                hemi = 'NH' if data[2, valid6, column][0] >= 0 else 'SH'

            stats[hemi][season][0] += 1
            stats[hemi][season][1] += nsteps
            for region in REGIONS:
                if sid in membership[region]:
                    stats[hemi][region][0] += 1
                    stats[hemi][region][1] += nsteps

    missing = ids - seen
    if missing:
        raise ValueError(f'{len(missing):,} Control storm IDs not found in NetCDF tracks; '
                         f'example: {sorted(missing)[:3]}')

    output = []
    for category in SEASONS + REGIONS + ('Total',):
        if category == 'Total':
            nh = [sum(stats['NH'][s][j] for s in SEASONS) for j in (0, 1)]
            sh = [sum(stats['SH'][s][j] for s in SEASONS) for j in (0, 1)]
        else:
            nh, sh = stats['NH'][category], stats['SH'][category]
        output.append((category, *nh, *sh))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir/'Table_1_Cyclone_Counts_by_Season_and_Region.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Classification','NH_Storms','NH_Steps_6h','SH_Storms','SH_Steps_6h'])
        writer.writerows(output)

    print('Classification | NH storms | NH steps | SH storms | SH steps')
    for category, nh_storms, nh_steps, sh_storms, sh_steps in output:
        print(f'{category:<19} {nh_storms:>10,} {nh_steps:>10,} '
              f'{sh_storms:>10,} {sh_steps:>10,}')
    print(f'Total matched storms: {len(seen):,}')
    print(f'Saved: {csv_path}')

if __name__ == '__main__':
    main()
