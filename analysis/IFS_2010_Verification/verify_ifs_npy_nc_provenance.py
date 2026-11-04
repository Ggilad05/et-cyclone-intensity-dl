#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit IFS NPY fields against converted MARS NetCDF fields (read-only).

The historical MARS request uses initialization 00 UTC and steps 6/12/18/24.
Consequently the EXPECTED sources are:
  valid 06 -> same day's +6   (var151_2)
  valid 12 -> same day's +12  (var151_2)
  valid 18 -> same day's +18  (var151_2)
  valid 00 -> PREVIOUS day's +24 (var151_2)

This script proves or rejects the *spatial field match* of selected storm NPY
frames against these candidate fields, and compares the 00 UTC NPY against the
same-day static var151 where available. A match to a converted NetCDF field is
NOT, by itself, proof of GRIB step/type if the conversion stripped provenance.

Run on Arctic:
    python verify_ifs_npy_nc_provenance.py
Optional:
    python verify_ifs_npy_nc_provenance.py --nc-root /path/to/converted/root \
      --samples-per-hour 3 --storm DJF_2010_NH_neg_1

Outputs: ifs_npy_nc_audit.csv, ifs_npy_nc_summary.csv, nc_metadata_audit.csv
Does not modify any input CSV or NPY.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE = Path('data/steps_analysis')
LEAD_TO_CONTROL = {6: 'control_31.csv', 12: 'control_32.csv',
                   18: 'control_33.csv', 24: 'control_34.csv'}
HOUR_TO_LEAD = {0: 24, 6: 6, 12: 12, 18: 18}
SEASON_START_MONTH = {'DJF': 12, 'MAM': 3, 'JJA': 6, 'SON': 9}


def cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nc-root', type=Path, default=None,
                   help='Directory containing YYYY_MM/mslp_YYYY_MM_DD_converted_2.nc; auto-detect otherwise')
    p.add_argument('--search-root', type=Path, default=Path('data'))
    p.add_argument('--ifs-dir', type=Path, default=BASE / 'Forecast 2010' / '2010')
    p.add_argument('--era5-root', type=Path, default=Path('data/era5'))
    p.add_argument('--control-dir', type=Path, default=BASE / 'updated_data' / 'control')
    p.add_argument('--out', type=Path, default=BASE / 'IFS_NPY_vs_NC_verification')
    p.add_argument('--samples-per-hour', type=int, default=2)
    p.add_argument('--storm', default=None, help='Optional specific storm name, with or without .npy')
    p.add_argument('--max-search-depth', type=int, default=8)
    p.add_argument('--tolerance-pa', type=float, default=0.1,
                   help='Full-patch max absolute difference required for confirming match')
    p.add_argument('--radius-deg-lon', type=float, default=17.5)
    p.add_argument('--radius-deg-lat', type=float, default=15.0)
    p.add_argument('--resolution-deg', type=float, default=1.25)
    p.add_argument('--limit-scan-days', type=int, default=100,
                   help='Max candidate valid calendar dates per sampled frame if day NPY absent')
    return p.parse_args()


def nc_path(root, day):
    return root / day.strftime('%Y_%m') / f'mslp_{day:%Y_%m_%d}_converted_2.nc'


def detect_root(a):
    if a.nc_root:
        if not a.nc_root.is_dir():
            raise FileNotFoundError(f'--nc-root not found: {a.nc_root}')
        return a.nc_root
    probe = dt.date(2010, 7, 15)
    options = [Path('data'), Path('data/era5'),
               Path('data')]
    for candidate in options:
        if nc_path(candidate, probe).is_file() or nc_path(candidate, probe-dt.timedelta(days=1)).is_file():
            return candidate
    base = a.search_root
    if not base.is_dir():
        raise FileNotFoundError(f'NC root not found. Provide --nc-root. Search root missing: {base}')
    targets = {nc_path(Path('/'), probe).name, nc_path(Path('/'), probe-dt.timedelta(days=1)).name}
    print(f'Searching NC root below {base}, max depth={a.max_search_depth} ...', flush=True)
    for here, folders, files in os.walk(base):
        depth = len(Path(here).relative_to(base).parts)
        folders[:] = [x for x in folders if not x.startswith('.') and x not in ('venv', '.venv', '__pycache__')]
        if depth >= a.max_search_depth:
            folders[:] = []
        if Path(here).name == '2010_07' and targets.intersection(files):
            return Path(here).parent
    raise FileNotFoundError('Could not locate converted NC files. Supply --nc-root ROOT '
                            '(must contain YYYY_MM/mslp_YYYY_MM_DD_converted_2.nc).')


def clean_key(s):
    return re.sub(r'(?i)\.npy$', '', str(s).strip())


def inputs_from_control(a):
    """Use center longitude/latitude and truth from precisely matched Control steps."""
    all_candidates = []
    for lead, fname in LEAD_TO_CONTROL.items():
        fp = a.control_dir / fname
        if not fp.is_file():
            raise FileNotFoundError(f'Control file missing: {fp}')
        cols = ['storm_key', 'step_index', 'lon', 'lat', 'true_mslp_corr']
        t = pd.read_csv(fp, usecols=cols, dtype={'storm_key': str})
        t['storm_key'] = t['storm_key'].map(clean_key)
        t = t[t.storm_key.str.contains('_2010_', regex=False)].copy()
        if a.storm:
            t = t[t.storm_key == clean_key(a.storm)]
        if t.empty:
            print(f'  WARNING: no Control records for {lead}h', flush=True)
            continue
        t['step_index'] = pd.to_numeric(t['step_index'], errors='raise').astype(int)
        if t.duplicated(['storm_key', 'step_index']).any():
            raise ValueError(f'Nonunique Control keys for lead={lead}')
        t['lead_h'] = lead
        all_candidates.append(t)
    if not all_candidates:
        raise ValueError('No Control candidates found for 2010 (check --storm/paths)')
    control = pd.concat(all_candidates, ignore_index=True)
    selected = []
    seen_hour = {h: 0 for h in HOUR_TO_LEAD}
    # Prefer the same, earliest storms across horizons to ease source NC access.
    for row in control.sort_values(['storm_key', 'step_index', 'lead_h']).itertuples(index=False):
        if seen_hour[row.lead_h % 24] >= a.samples_per_hour:
            continue
        storm = row.storm_key
        ifile = a.ifs_dir / f'{storm}.npy'
        hfile = (a.era5_root / 'hour' / '2010' / f'{storm}.npy')
        if not hfile.is_file():
            hfile = a.era5_root / 'hour' / f'{storm}.npy'
        if not ifile.is_file() or not hfile.is_file():
            continue
        try:
            H = np.load(hfile, mmap_mode='r', allow_pickle=False)
            step = int(row.step_index)
            if step >= len(H):
                continue
            hframe = np.asarray(H[step])
            hour = float(hframe.flat[0])
            if not np.isfinite(hour) or not np.allclose(hframe, hour, atol=1e-6, rtol=0):
                continue
            hour = int(round(hour))
            if hour != (row.lead_h % 24):
                continue
            F = np.load(ifile, mmap_mode='r', allow_pickle=False)
            if step >= len(F) or F.ndim != 3 or not np.isfinite(F[step]).all():
                continue
            selected.append({'storm_key': storm, 'step_index': step,
                             'valid_hour_utc': hour, 'lead_h': int(row.lead_h),
                             'lon': float(row.lon), 'lat': float(row.lat),
                             'control_truth_hpa': float(row.true_mslp_corr),
                             'ifs_file': str(ifile)})
            seen_hour[hour] += 1
        except (OSError, ValueError, TypeError):
            continue
        if all(v >= a.samples_per_hour for v in seen_hour.values()):
            break
    if not selected:
        raise RuntimeError('No IFS and Control frames with consistent valid hours were found. '
                           'Check IFS/ERA5/hour paths and whether step_index matches frame number.')
    print('Samples selected by valid hour:', {h: seen_hour[h] for h in sorted(seen_hour)}, flush=True)
    return selected


def dates_for(storm, max_days):
    """Infer date candidates from the original creator's season start convention.

    No exact date exists in Control CSV or hour NPY. Therefore scan each possible
    day, accepting a provenance match only when the full spatial patch matches.
    DJF_2010 begins 2010-12-01 in the original creator.
    """
    m = re.match(r'^(DJF|MAM|JJA|SON)_(\d{4})_', storm)
    if not m:
        raise ValueError(f'Cannot identify season/year from storm: {storm}')
    season, year = m.group(1), int(m.group(2))
    start = dt.date(year, SEASON_START_MONTH[season], 1)
    # For DJF the following March is one year later.
    length = 90 if season == 'DJF' else 93
    return [start + dt.timedelta(days=i) for i in range(min(length, max_days))]


def exact_valid_time(ds, var, target):
    """Never use nearest-time selection: wrong lead time is the hazard being audited."""
    for name in ('valid_time', 'time'):
        if name not in var.dims or name not in ds.coords:
            continue
        coord = ds[name]
        if coord.ndim != 1:
            continue
        try:
            valid = np.asarray(coord.values).astype('datetime64[ns]')
        except (TypeError, ValueError):
            continue
        wanted = np.datetime64(target, 'ns')
        if not np.any(valid == wanted):
            continue
        selected = var.sel({name: wanted}).squeeze(drop=True)
        if selected.ndim == 2:
            return selected
    return None


def cut_same_as_original(field, center_lon, center_lat, a):
    """Reproduce cut_polygon spatial selection, including longitude periodicity.

    The original cut builds 28 lon and 24 lat points with np.arange, and
    xarray nearest-neighbor selection. For points crossing 0/360 it concatenates
    longitude-shifted copies so nearest can select across the dateline.
    """
    x = np.arange(center_lon-a.radius_deg_lon, center_lon+a.radius_deg_lon,
                  a.resolution_deg)
    y = np.arange(center_lat-a.radius_deg_lat, center_lat+a.radius_deg_lat,
                  a.resolution_deg)
    if not {'lat', 'lon'}.issubset(field.dims):
        raise ValueError(f'No lat/lon dims; found {field.dims}')
    if np.any(x < 0) or np.any(x >= 360):
        original = field
        minus = original.assign_coords(lon=original.lon-360)
        plus = original.assign_coords(lon=original.lon+360)
        field = xr.concat([minus, original, plus], dim='lon')
    result = field.sel(lon=x, lat=y, method='nearest')
    return np.asarray(result.values, dtype=np.float64)


def candidate_info(ds, key, day, valid_day):
    if key not in ds:
        return None, 'variable_absent', {}
    v = ds[key]
    if key == 'var151_2':
        frame = exact_valid_time(ds, v, valid_day)
        if frame is None:
            return None, 'valid_time_absent', metadata(ds, v)
    else:
        frame = v.squeeze(drop=True)
        if frame.ndim != 2:
            return None, 'not_static_2d', metadata(ds, v)
    return frame, 'available', metadata(ds, v)


def metadata(ds, var):
    out = {}
    for name in ('time', 'valid_time', 'step', 'forecast_reference_time', 'forecast_period'):
        if name in ds.coords or name in ds.variables:
            c = ds[name]
            vals = np.asarray(c.values).ravel()
            out[name] = str([str(v) for v in vals[:8]])
    out['variable_attrs'] = str({k: str(v) for k, v in var.attrs.items()
                                  if any(z in k.lower() for z in
                                         ('grib', 'step', 'forecast', 'time', 'type', 'unit'))})
    out['global_attrs'] = str({k: str(v) for k, v in ds.attrs.items()
                                if any(z in k.lower() for z in
                                       ('grib', 'step', 'forecast', 'time', 'type'))})
    return out


def eval_day(sample, valid_day, root, observed, a, record_metadata):
    """Compare the exact expected forecast and, for 00, static same-day var151."""
    prev = valid_day - dt.timedelta(days=1)
    init_day = prev if sample['valid_hour_utc'] == 0 else valid_day
    valid_dt = dt.datetime.combine(valid_day, dt.time(hour=sample['valid_hour_utc']))
    pf = nc_path(root, init_day)
    if not pf.is_file():
        return None
    # A valid 00 frame in a 00-initialized previous-day file is a +24 candidate.
    with xr.open_dataset(pf, decode_times=True) as ds:
        source, status, attrs = candidate_info(ds, 'var151_2', init_day, valid_dt)
        if source is None:
            return None
        try:
            cut = cut_same_as_original(source, sample['lon'], sample['lat'], a)
        except (KeyError, IndexError, ValueError) as exc:
            return {'valid_date': str(valid_day), 'error': f'crop_failed: {exc}'}
        if cut.shape != observed.shape:
            return {'valid_date': str(valid_day), 'error': f'shape: NC {cut.shape} vs NPY {observed.shape}'}
        residual = observed - cut
        if not np.isfinite(residual).all():
            return {'valid_date': str(valid_day), 'error': 'nonfinite_patch'}
        result = {
            'valid_date': str(valid_day), 'expected_nc_file': str(pf),
            'expected_source': f'previous_day_fc_24' if sample['valid_hour_utc'] == 0
                               else f'same_day_fc_{sample["lead_h"]}',
            'forecast_max_abs_diff_pa': float(np.max(np.abs(residual))),
            'forecast_mean_abs_diff_pa': float(np.mean(np.abs(residual))),
            'npy_min_pa': float(np.min(observed)),
            'nc_forecast_min_pa': float(np.min(cut)),
            'metadata': attrs,
        }
    if sample['valid_hour_utc'] == 0:
        static_path = nc_path(root, valid_day)
        if static_path.is_file():
            with xr.open_dataset(static_path, decode_times=True) as ds:
                static, status, mattrs = candidate_info(ds, 'var151', valid_day, valid_dt)
                if static is not None:
                    try:
                        cut_static = cut_same_as_original(static, sample['lon'], sample['lat'], a)
                        if cut_static.shape == observed.shape and np.isfinite(cut_static).all():
                            residual_static = observed-cut_static
                            result['static_max_abs_diff_pa'] = float(np.max(np.abs(residual_static)))
                            result['static_mean_abs_diff_pa'] = float(np.mean(np.abs(residual_static)))
                            if record_metadata:
                                result['static_metadata'] = mattrs
                    except (ValueError, KeyError):
                        pass
    return result


def verify_sample(sample, root, a):
    field = np.load(sample['ifs_file'], mmap_mode='r', allow_pickle=False)[sample['step_index']]
    observed = np.asarray(field, dtype=np.float64)
    if observed.ndim != 2:
        raise ValueError(f'Not a 2D field: {sample["storm_key"]}')

    best_fc = None
    best_static = None
    attempts = 0
    count_missing = 0
    candidate_days = dates_for(sample['storm_key'], a.limit_scan_days)
    for i, day in enumerate(candidate_days):
        path = nc_path(root, day - dt.timedelta(days=1) if sample['valid_hour_utc'] == 0 else day)
        if not path.is_file():
            count_missing += 1
            continue
        val = eval_day(sample, day, root, observed, a, record_metadata=False)
        if val is None or 'forecast_max_abs_diff_pa' not in val:
            continue
        attempts += 1
        if best_fc is None or val['forecast_max_abs_diff_pa'] < best_fc['forecast_max_abs_diff_pa']:
            best_fc = val
        if ('static_max_abs_diff_pa' in val and
            (best_static is None or val['static_max_abs_diff_pa'] < best_static['static_max_abs_diff_pa'])):
            best_static = val
        exact_forecast = best_fc['forecast_max_abs_diff_pa'] <= a.tolerance_pa
        exact_static = best_static is not None and best_static['static_max_abs_diff_pa'] <= a.tolerance_pa
        if exact_forecast or exact_static:
            break
        if (i + 1) % 30 == 0:
            print(f'    searched {i+1} days; best forecast max diff: '
                  f'{best_fc["forecast_max_abs_diff_pa"]:.4g} Pa', flush=True)

    result = {k: v for k, v in sample.items() if k != 'ifs_file'}
    result['dates_scanned'] = len(candidate_days)
    result['dates_with_usable_nc'] = attempts
    result['missing_source_nc_days'] = count_missing
    if best_fc is None:
        result.update(status='NO_EXACT_NC_FIELD_FOUND',
                      note='No converted NC files with expected forecast valid time')
        return result, None

    forecast_diff = best_fc['forecast_max_abs_diff_pa']
    static_diff = best_static['static_max_abs_diff_pa'] if best_static is not None else float('nan')
    good_fc = forecast_diff <= a.tolerance_pa
    good_static = np.isfinite(static_diff) and static_diff <= a.tolerance_pa
    if good_fc and not good_static:
        status = 'CONFIRMED_NPY_MATCHES_EXPECTED_FC_NC'
    elif good_static and not good_fc:
        status = 'NPY_MATCHES_SAME_DAY_STATIC_NOT_FORECAST'
    elif good_fc and good_static:
        status = 'AMBIGUOUS_IDENTICAL_STATIC_AND_FORECAST'
    else:
        status = 'UNVERIFIED_NO_CLOSE_FIELD_MATCH'

    chosen = best_fc if good_fc or not good_static else best_static
    # The chosen day's source fields also contain the relevant metadata.
    full = eval_day(sample, dt.date.fromisoformat(chosen['valid_date']), root,
                    observed, a, record_metadata=True)
    if full is None:
        full = chosen
    result.update({k: v for k, v in best_fc.items()
                   if k not in ('metadata', 'static_metadata', 'static_max_abs_diff_pa', 'static_mean_abs_diff_pa')})
    result.update(status=status,
                  static_best_valid_date=best_static['valid_date'] if best_static else '',
                  static_max_abs_diff_pa=static_diff,
                  static_mean_abs_diff_pa=(best_static.get('static_mean_abs_diff_pa', float('nan'))
                                           if best_static else float('nan')),
                  note='NPY spatial match; GRIB step still requires provenance metadata or original MARS request')
    if good_static and not good_fc:
        result['valid_date'] = chosen['valid_date']
        result['expected_nc_file'] = chosen['expected_nc_file']
    met = {'storm_key': sample['storm_key'], 'step_index': sample['step_index'],
           'lead_h': sample['lead_h'], 'valid_hour_utc': sample['valid_hour_utc'],
           'valid_date': chosen['valid_date'], 'expected_nc_file': chosen['expected_nc_file'],
           **{f'forecast_{k}':v for k,v in full.get('metadata',{}).items()},
           **{f'static_{k}':v for k,v in full.get('static_metadata',{}).items()}}
    return result, met


def main():
    a = cli()
    if a.samples_per_hour < 1 or a.tolerance_pa <= 0:
        raise ValueError('Need --samples-per-hour >=1 and --tolerance-pa > 0')
    a.out.mkdir(parents=True, exist_ok=True)
    root = detect_root(a)
    print('Converted NetCDF root:', root)
    print('IFS NPY directory:', a.ifs_dir)
    print('ERA5 hour NPY:', a.era5_root / 'hour')
    print('Control CSV:', a.control_dir)
    print('Output directory:', a.out, flush=True)
    samples = inputs_from_control(a)
    rows, metas = [], []
    for j, sample in enumerate(samples, start=1):
        print(f'\n[{j}/{len(samples)}] {sample["storm_key"]}, step {sample["step_index"]}, '
              f'valid {sample["valid_hour_utc"]:02d} UTC, lead {sample["lead_h"]}h', flush=True)
        r, met = verify_sample(sample, root, a)
        rows.append(r)
        if met is not None:
            metas.append(met)
        print(f'  RESULT: {r["status"]}; date={r.get("valid_date", "unknown")}; '
              f'forecast_max_abs_diff_pa={r.get("forecast_max_abs_diff_pa", float("nan")):.5g}; '
              f'static_max_abs_diff_pa={r.get("static_max_abs_diff_pa", float("nan")):.5g}', flush=True)
        pd.DataFrame(rows).to_csv(a.out/'ifs_npy_nc_audit.csv', index=False)
        if metas:
            pd.DataFrame(metas).to_csv(a.out/'nc_metadata_audit.csv', index=False)
    df = pd.DataFrame(rows)
    for col in ('forecast_max_abs_diff_pa', 'static_max_abs_diff_pa'):
        if col not in df:
            df[col] = np.nan
    summary = df.groupby(['lead_h', 'status'], as_index=False).agg(
        n=('storm_key','size'),
        median_forecast_max_abs_diff_pa=('forecast_max_abs_diff_pa','median'),
        median_static_max_abs_diff_pa=('static_max_abs_diff_pa','median')
    )
    summary.to_csv(a.out/'ifs_npy_nc_summary.csv', index=False)
    print('\n' + '='*70)
    print(summary.to_string(index=False))
    print('Written audit files to:', a.out)
    print('NOTE: A field equality does NOT establish init/step GRIB metadata if lost by conversion.')

if __name__ == '__main__':
    main()
