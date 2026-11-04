#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify EVERY 2010-labelled IFS cyclone-frame against the converted MARS NC.

Designed for the Ubuntu/WSL source machine. No Control CSVs, cyclone-center
coordinates, or per-storm absolute dates required. It finds the calendar date
of each storm using one exact spatial-field match to a valid-time forecast NC,
then checks EVERY 6-hour NPY frame against its corresponding NC field.

Forecast source expected from MARS request: init 00 UTC, steps 6/12/18/24.
Valid 00 -> +24 from previous day's NC. 06/12/18 -> same day's +6/+12/+18.

READ-ONLY. Never writes into the original NPY/NC directories. Output records
explicitly distinguish missing files, missing variable/time, spatial mismatch,
and storms whose absolute date could not be reconstructed.

Usage:
    python3 verify_ifs_full_year_2010_WSL.py \\
      --manifest source_all_valid_times.csv \\
      --nc-root data \\
      --ifs-dir data/forecast_0/var151_2/2010

The manifest was previously extracted independently from all IFS and ERA5/hour
NPY files and includes storm_key, step_index, valid_hour_utc.

A pixel-exact match to converted NetCDF confirms the NPY source FIELD. It does
not alone prove original GRIB initialization/step metadata if conversion lost it.
"""
from __future__ import annotations
import argparse
import calendar
import datetime as dt
from functools import lru_cache
from pathlib import Path
import re
import sys
import traceback

import numpy as np
import pandas as pd
import xarray as xr

HOUR_TO_LEAD = {0: 24, 6: 6, 12: 12, 18: 18}
START_MONTH = {'DJF': 12, 'MAM': 3, 'JJA': 6, 'SON': 9}


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', type=Path, default=Path(__file__).with_name('source_all_valid_times.csv'))
    p.add_argument('--nc-root', type=Path, default=Path('data'))
    p.add_argument('--ifs-dir', type=Path, default=Path('data/forecast_0/var151_2/2010'))
    p.add_argument('--out', type=Path, default=Path('data/IFS_full_2010_audit'))
    p.add_argument('--tolerance-pa', type=float, default=0.1)
    p.add_argument('--max-anchors', type=int, default=4,
                   help='Different NPY frames tried to identify each storm date (default 4)')
    p.add_argument('--storm', type=str, default=None, help='Optional single storm for troubleshooting')
    p.add_argument('--max-storms', type=int, default=None, help='For smoke tests only')
    return p.parse_args()


def path_for(root: Path, day: dt.date):
    return root / day.strftime('%Y_%m') / f'mslp_{day:%Y_%m_%d}_converted_2.nc'


def season_dates(key: str):
    m = re.match(r'^(DJF|MAM|JJA|SON)_(\d{4})_', key)
    if not m:
        raise ValueError(f'Cannot get season/year from {key!r}')
    season, year = m.group(1), int(m.group(2))
    start = dt.date(year, START_MONTH[season], 1)
    if season == 'DJF':
        end_exclusive = dt.date(year + 1, 3, 1)
    else:
        end_month = START_MONTH[season] + 3
        end_exclusive = dt.date(year, end_month, 1)
    days = (end_exclusive - start).days
    return season, [start + dt.timedelta(days=i) for i in range(days)]


def read_manifest(path: Path, restrict_storm=None, max_storms=None):
    if not path.is_file():
        raise FileNotFoundError(f'Missing manifest: {path}. Include source_all_valid_times.csv from package.')
    df = pd.read_csv(path, dtype={'storm_key':str})
    for required in ('storm_key', 'step_index', 'valid_hour_utc'):
        if required not in df:
            raise ValueError(f'Manifest missing {required!r}')
    df['storm_key'] = df['storm_key'].str.strip().str.replace(r'(?i)\.npy$', '', regex=True)
    df['step_index'] = pd.to_numeric(df['step_index'], errors='raise').astype(int)
    df['valid_hour_utc'] = pd.to_numeric(df['valid_hour_utc'], errors='raise').astype(int)
    df = df[df.storm_key.str.contains('_2010_', regex=False)].copy()
    if restrict_storm:
        df = df[df.storm_key == re.sub(r'(?i)\.npy$', '', restrict_storm)]
    if max_storms:
        names = sorted(df.storm_key.unique())[:max_storms]
        df = df[df.storm_key.isin(names)]
    if df.duplicated(['storm_key','step_index']).any():
        raise ValueError('Manifest contains duplicate storm_key/step_index')
    if not df.valid_hour_utc.isin(HOUR_TO_LEAD).all():
        raise ValueError('Manifest contains hours other than 00/06/12/18 UTC')
    if df.empty:
        raise ValueError('Manifest contains no selected 2010 data')
    return df.sort_values(['storm_key','step_index']).reset_index(drop=True)


def _coord_datetimes(coord):
    try:
        arr = np.asarray(coord.values).astype('datetime64[ns]').reshape(-1)
        if np.isnat(arr).any():
            return []
        return [pd.Timestamp(x).to_pydatetime().replace(tzinfo=None) for x in arr]
    except (TypeError, ValueError, OverflowError):
        return []


def latlon_matrix(frame):
    if not set(('lat','lon')).issubset(frame.dims):
        raise ValueError(f'Invalid spatial dimensions {frame.dims}')
    frame = frame.squeeze(drop=True).transpose('lat','lon')
    lat = np.asarray(frame.lat.values, dtype=float)
    lon = np.asarray(frame.lon.values, dtype=float)
    # The original xarray .sel(lat=[...],lon=[...],method='nearest') returned
    # rows and columns in the requested order: increasing latitude/longitude.
    mat = np.asarray(frame.values, dtype=np.float64)
    mat = mat[np.argsort(lat), :][:, np.argsort(lon)]
    dlat = float(np.median(np.abs(np.diff(np.sort(lat)))))
    dlon = float(np.median(np.abs(np.diff(np.sort(lon)))))
    # The old cutter sampled target latitude/longitude every 1.25 degrees.
    # The global NC may be finer, e.g. 0.25 degrees, so the patch samples every
    # fifth NC grid cell in that case. Exact integer strides are supported.
    ratio_y = 1.25/dlat
    ratio_x = 1.25/dlon
    sy, sx = int(round(ratio_y)), int(round(ratio_x))
    if sy < 1 or sx < 1 or not (np.isclose(ratio_y, sy, atol=1e-5)
                              and np.isclose(ratio_x, sx, atol=1e-5)):
        raise ValueError(f'NC grid resolution {dlat},{dlon} not an integer divisor '
                         'of 1.25 deg; requires explicit center-coordinate sampling')
    return (mat, sy, sx)


@lru_cache(maxsize=12)
def read_nc(root_string: str, day: dt.date):
    """Load once per daily NC; returns {valid datetime -> global 2D array}, static, metadata.

    Cache is bounded; datasets close immediately (no file-descriptor growth).
    """
    path = path_for(Path(root_string), day)
    if not path.is_file():
        return {}, None, 'NC_FILE_MISSING', {}, str(path)
    with xr.open_dataset(path, decode_times=True) as ds:
        fc = {}
        metadata = {}
        for key in ('forecast_reference_time','step','forecast_period','time','valid_time'):
            if key in ds:
                metadata[key] = str(np.asarray(ds[key].values).ravel()[:8].tolist())
        for varname in ('var151_2','var151'):
            if varname in ds:
                metadata[varname+'_attrs'] = str({k:str(v) for k,v in ds[varname].attrs.items()
                    if any(z in k.lower() for z in ('step','grib','forecast','type','time'))})
        if 'var151_2' in ds:
            da = ds['var151_2']
            for timename in ('valid_time','time'):
                if timename not in da.dims or timename not in ds.coords:
                    continue
                times = _coord_datetimes(ds[timename])
                if len(times) != da.sizes[timename]:
                    continue
                for i, valid in enumerate(times):
                    frame = da.isel({timename:i}).squeeze(drop=True)
                    if set(frame.dims) == {'lat','lon'}:
                        fc[valid] = latlon_matrix(frame)
                break
        static = None
        if 'var151' in ds:
            v = ds['var151'].squeeze(drop=True)
            if set(v.dims) == {'lat','lon'}:
                static = latlon_matrix(v)
        problem = 'OK' if fc else ('FORECAST_VARIABLE_MISSING' if 'var151_2' not in ds
                                      else 'FORECAST_VALID_TIME_MISSING')
        return fc, static, problem, metadata, str(path)


def forecast_field(root: Path, valid: dt.datetime):
    day = valid.date() - dt.timedelta(days=1) if valid.hour == 0 else valid.date()
    fc, _, status, _, path = read_nc(str(root), day)
    if status != 'OK':
        return None, status, path
    field = fc.get(valid)
    return field, ('OK' if field is not None else 'VALID_TIME_ABSENT'), path


def match_spatial(whole, patch, tolerance):
    """Find the exact cyclone patch within a global field, without lat/lon input.

    Check the centre, two distant corners, then all cells. The three-point
    vectorized filter avoids many slow per-grid-cell Python checks.
    """
    if whole is None or patch.ndim != 2 or not np.isfinite(patch).all():
        return False, float('nan'), -1, -1, 0
    whole, sy, sx = whole
    ny, nx = patch.shape
    gy, gx = whole.shape
    if (ny-1)*sy >= gy or (nx-1)*sx >= gx:
        return False, float('nan'), -1, -1, 0
    cy, cx = ny//2, nx//2
    yy, xx = np.nonzero(np.isfinite(whole) & (np.abs(whole-patch[cy,cx]) <= tolerance))
    if len(yy) == 0:
        return False, float('nan'), -1, -1, 0
    y0=yy-cy*sy
    x0=xx-cx*sx
    good=(y0 >= 0) & (y0+(ny-1)*sy < gy)
    if not np.any(good):
        return False, float('nan'), -1, -1, 0
    yy,xx,y0,x0=yy[good],xx[good],y0[good],x0[good]
    g0 = np.abs(whole[y0, x0 % gx] - patch[0,0]) <= tolerance
    g1 = np.abs(whole[y0+(ny-1)*sy, (x0+(nx-1)*sx) % gx] - patch[-1,-1]) <= tolerance
    good = g0 & g1
    if not np.any(good):
        return False, float('nan'), -1, -1, 0
    y0,x0,yy,xx=y0[good],x0[good],yy[good],xx[good]
    best=np.inf
    iy=sy*np.arange(ny)[:,None]
    ix=sx*np.arange(nx)[None,:]
    for y,x,ybase,xbase in zip(yy,xx,y0,x0):
        crop=whole[ybase+iy, (xbase+ix)%gx]
        diff=float(np.max(np.abs(crop-patch)))
        best=min(best,diff)
        if diff<=tolerance:
            return True,diff,int(y),int(x),len(yy)
    return False,(best if np.isfinite(best) else float('nan')),-1,-1,len(yy)


def anchors_for(group, arr, max_anchors):
    """Choose alternate frames across the storm, favoring 06/12/18 over 00."""
    good = []
    for r in group.itertuples(index=False):
        i = int(r.step_index)
        if 0 <= i < arr.shape[0] and np.isfinite(arr[i]).all():
            good.append((i, int(r.valid_hour_utc)))
    if not good:
        return []
    positions = sorted(good, key=lambda item: (item[1] == 0, abs(item[0]-arr.shape[0]//2)))
    # Prefer nearby but distinct valid hours to rule out midnight-specific issues.
    result = []
    hours = set()
    for item in positions:
        if item[1] not in hours:
            result.append(item)
            hours.add(item[1])
        if len(result) >= max_anchors:
            return result
    for item in positions:
        if item not in result:
            result.append(item)
        if len(result) >= max_anchors:
            return result
    return result


def find_anchors_one_round(root, unresolved, args, round_num, anchors, chosen):
    """Day-major seasonal scan: open each NC day only once per round, not once/storm."""
    grouped = {}
    for storm in unresolved:
        seq = anchors[storm]
        if round_num >= len(seq):
            continue
        step, hour = seq[round_num]
        season, dates = chosen[storm]['season'], chosen[storm]['dates']
        patch = np.asarray(chosen[storm]['npy'][step], dtype=np.float64)
        grouped.setdefault(season, {}).setdefault(hour, []).append((storm, step, patch, dates))
    located = {}
    for season in ('DJF','MAM','JJA','SON'):
        per_hour = grouped.get(season, {})
        if not per_hour:
            continue
        dates = next(iter(next(iter(per_hour.values()))))[3]
        for nday, day in enumerate(dates):
            if (nday+1) % 30 == 0:
                left = sum(len(t) for t in per_hour.values())
                print(f'    {season} anchor-round {round_num+1}: day {nday+1}/{len(dates)}; unresolved {left}', flush=True)
            if not any(per_hour.values()):
                break
            for hour in (0,6,12,18):
                if not per_hour.get(hour):
                    continue
                valid = dt.datetime.combine(day, dt.time(hour=hour))
                field, status, _ = forecast_field(root, valid)
                if field is None:
                    continue
                still = []
                for storm, step, patch, day_range in per_hour[hour]:
                    match, maxdiff, cy, cx, _ = match_spatial(field, patch, args.tolerance_pa)
                    if match:
                        located[storm] = {'anchor_step':step,'anchor_datetime':valid,
                            'anchor_max_diff_pa':maxdiff,'anchor_grid_y':cy,'anchor_grid_x':cx,
                            'anchor_round':round_num+1}
                    else:
                        still.append((storm,step,patch,day_range))
                per_hour[hour] = still
    return located


def classify_step(storm, step, valid, hour, arr, root, tol):
    row = dict(storm_key=storm, step_index=int(step), valid_hour_utc=int(hour),
               lead_h=HOUR_TO_LEAD[int(hour)], valid_datetime=str(valid),
               init_datetime=str(valid - dt.timedelta(hours=HOUR_TO_LEAD[int(hour)])))
    if step >= len(arr) or step < 0:
        return {**row, 'status':'STEP_OUTSIDE_NPY', 'forecast_max_abs_diff_pa':np.nan}
    patch = np.asarray(arr[step], dtype=np.float64)
    if not np.isfinite(patch).all() or patch.ndim!=2:
        return {**row, 'status':'NONFINITE_OR_INVALID_NPY', 'forecast_max_abs_diff_pa':np.nan}
    expected, status, path = forecast_field(root, valid)
    row['expected_nc_file'] = path
    if status != 'OK':
        return {**row, 'status':status, 'forecast_max_abs_diff_pa':np.nan}
    ok, diff, yi, xi, candidates = match_spatial(expected, patch, tol)
    row['forecast_max_abs_diff_pa'] = diff
    row['candidate_centers_tested'] = candidates
    if ok:
        row.update(status='CONFIRMED_NPY_MATCHES_EXPECTED_FC_NC',
                   matched_lat_index=yi, matched_lon_index=xi)
    # Static field is a secondary diagnostic. If forecast matches, compare the
    # static field at the SAME geographic patch, not at some unrelated location.
    if hour == 0:
        _, static, _, _, _ = read_nc(str(root), valid.date())
        if static is not None:
            whole_static, sy_static, sx_static = static
            whole_fc, sy_fc, sx_fc = expected
            if ok and (sy_static, sx_static) == (sy_fc, sx_fc) and whole_static.shape == whole_fc.shape:
                ny, nx = patch.shape
                cy, cx = ny//2, nx//2
                y0, x0 = yi - cy*sy_fc, xi - cx*sx_fc
                if 0 <= y0 and y0 + (ny-1)*sy_fc < whole_static.shape[0]:
                    crop = whole_static[y0 + sy_fc*np.arange(ny)[:,None],
                                        (x0 + sx_fc*np.arange(nx))[None,:] % whole_static.shape[1]]
                    row['static_same_location_max_abs_diff_pa'] = float(np.max(np.abs(crop-patch)))
                    row['static_same_location_mean_abs_diff_pa'] = float(np.mean(np.abs(crop-patch)))
                    if row['static_same_location_max_abs_diff_pa'] <= tol:
                        return {**row, 'status':'AMBIGUOUS_IDENTICAL_STATIC_AND_FORECAST'}
            elif not ok:
                yes, diff_static, _, _, _ = match_spatial(static, patch, tol)
                row['static_max_abs_diff_pa'] = diff_static
                if yes:
                    return {**row, 'status':'NPY_MATCHES_SAME_DAY_STATIC_NOT_FORECAST'}
    if ok:
        return row
    return {**row, 'status':'NPY_NOT_MATCHED_TO_EXPECTED_FORECAST'}


def save_reports(rows, storm_rows, args, expected_samples, expected_storms):
    args.out.mkdir(parents=True, exist_ok=True)
    details = pd.DataFrame(rows)
    if hasattr(args, 'manifest_data') and len(details):
        keep = ['storm_key', 'step_index', 'true_mslp_ifs_from_npy', 'pred_mslp_ifs_from_npy']
        if set(keep).issubset(args.manifest_data.columns):
            details = details.merge(args.manifest_data[keep], on=['storm_key','step_index'],
                                    how='left', validate='one_to_one')
    details.to_csv(args.out/'full_year_field_audit.csv', index=False, float_format='%.8f')
    sdf = pd.DataFrame(storm_rows)
    sdf.to_csv(args.out/'storm_anchor_audit.csv', index=False)
    if details.empty:
        return
    summary = details.groupby(['lead_h','status'], dropna=False, as_index=False).agg(
        n=('storm_key','size'), storms=('storm_key','nunique'),
        median_forecast_max_abs_diff_pa=('forecast_max_abs_diff_pa','median'))
    full_counts = details.groupby('lead_h', dropna=False).size().to_dict()
    summary['fraction_of_lead'] = summary.apply(lambda r: r['n'] / full_counts[r['lead_h']], axis=1)
    summary.to_csv(args.out/'full_year_summary.csv', index=False)
    if {'true_mslp_ifs_from_npy', 'pred_mslp_ifs_from_npy'}.issubset(details.columns):
        metric = details.copy()
        metric['abs_error_hpa'] = (metric['pred_mslp_ifs_from_npy'] - metric['true_mslp_ifs_from_npy']).abs()
        metrics = metric.groupby(['lead_h', 'status'], dropna=False).agg(
            N=('storm_key', 'size'), MAE_hPa=('abs_error_hpa', 'mean')).reset_index()
        metrics.to_csv(args.out / 'mae_by_provenance_status.csv', index=False)
    issues = details[details.status!='CONFIRMED_NPY_MATCHES_EXPECTED_FC_NC']
    issues.to_csv(args.out/'full_year_unverified.csv',index=False)
    print('\n'+'='*75+'\nFULL YEAR VERIFICATION\n'+'='*75)
    print(f'Manifest rows: {expected_samples:,}; reported rows: {len(details):,}')
    print(f'Manifest storms: {expected_storms:,}; reported storms: {len(sdf):,}')
    print(summary.to_string(index=False))
    print(f'Confirmed: {(details.status=="CONFIRMED_NPY_MATCHES_EXPECTED_FC_NC").sum():,} / {len(details):,}')
    print(f'Unverified: {len(issues):,}. See full_year_unverified.csv')
    print(f'Output: {args.out}')
    if len(details)!=expected_samples or len(sdf)!=expected_storms:
        raise RuntimeError('Internal audit coverage mismatch; investigate generated files')


def main():
    a=arguments()
    if a.tolerance_pa<=0 or a.max_anchors<1:
        raise ValueError('Provide --tolerance-pa > 0 and --max-anchors >= 1')
    if not a.nc_root.is_dir() or not a.ifs_dir.is_dir():
        raise FileNotFoundError(f'NC or IFS path missing: {a.nc_root}; {a.ifs_dir}')
    df=read_manifest(a.manifest,a.storm,a.max_storms)
    a.manifest_data = df
    groups=list(df.groupby('storm_key',sort=True))
    print(f'NC root: {a.nc_root}\nIFS NPY: {a.ifs_dir}\nManifest: {a.manifest}')
    print(f'Selected {len(df):,} NPY frames from {len(groups):,} storms (2010-labelled seasonal tracks).')
    print('Note: DJF_2010 in original code starts December 2010 and extends into 2011.',flush=True)
    chosen={}
    anchors={}
    unlocated=set()
    rows=[]
    storm_rows=[]
    for storm,group in groups:
        filepath = a.ifs_dir / (storm+'.npy')
        lead_counts={HOUR_TO_LEAD[h]:int((group.valid_hour_utc==h).sum()) for h in HOUR_TO_LEAD}
        if not filepath.is_file():
            storm_rows.append({'storm_key':storm,'status':'NPY_FILE_MISSING','n_manifest':len(group),
                'n_npy':np.nan, 'source_path':str(filepath)})
            rows.extend({'storm_key':storm,'step_index':int(r.step_index),'valid_hour_utc':int(r.valid_hour_utc),
                'lead_h':HOUR_TO_LEAD[int(r.valid_hour_utc)], 'status':'NPY_FILE_MISSING',
                'forecast_max_abs_diff_pa':np.nan} for r in group.itertuples(index=False))
            continue
        try:
            data=np.load(filepath,allow_pickle=False)
        except (ValueError,OSError) as exc:
            storm_rows.append({'storm_key':storm,'status':'NPY_READ_ERROR','source_path':str(filepath),'note':str(exc)})
            rows.extend({'storm_key':storm,'step_index':int(r.step_index),'valid_hour_utc':int(r.valid_hour_utc),
                'lead_h':HOUR_TO_LEAD[int(r.valid_hour_utc)], 'status':'NPY_READ_ERROR',
                'forecast_max_abs_diff_pa':np.nan} for r in group.itertuples(index=False))
            continue
        if data.ndim!=3 or data.shape[1:]!=(24,28):
            note=f'expected (T,24,28), got {data.shape}'
            storm_rows.append({'storm_key':storm,'status':'NPY_SHAPE_ERROR','source_path':str(filepath),'note':note})
            rows.extend({'storm_key':storm,'step_index':int(r.step_index),'valid_hour_utc':int(r.valid_hour_utc),
                'lead_h':HOUR_TO_LEAD[int(r.valid_hour_utc)], 'status':'NPY_SHAPE_ERROR',
                'forecast_max_abs_diff_pa':np.nan} for r in group.itertuples(index=False))
            continue
        season,dates = season_dates(storm)
        chosen[storm]={'npy':data,'group':group,'season':season,'dates':dates,'path':str(filepath)}
        anchors[storm]=anchors_for(group,data,a.max_anchors)
        unlocated.add(storm)
    print(f'Eligible storms: {len(unlocated):,}; immediate file/schema exceptions: {len(groups)-len(unlocated):,}',flush=True)
    matched={}
    for attempt in range(a.max_anchors):
        round_located=find_anchors_one_round(a.nc_root, unlocated, a, attempt, anchors, chosen)
        matched.update(round_located)
        unlocated-=set(round_located)
        print(f'Anchor pass {attempt+1}/{a.max_anchors}: found {len(round_located):,}; '
              f'remaining {len(unlocated):,}',flush=True)
        if not unlocated:
            break
    if unlocated:
        print('Examples with no recovered date:',sorted(unlocated)[:15],flush=True)
    # Construct verification jobs by initialisation date so each NC is loaded only
    # a few times, instead of repeatedly reopening dates for each storm.
    jobs=[]
    for storm,group in groups:
        if storm not in chosen:
            continue
        rec=chosen[storm]
        arr=rec['npy']
        anchor=matched.get(storm)
        if anchor is None:
            storm_rows.append({'storm_key':storm,'status':'UNRESOLVED_STORM_DATE',
                'n_manifest':len(group),'n_npy':len(arr),'source_path':rec['path']})
            rows.extend({'storm_key':storm,'step_index':int(r.step_index),
                'valid_hour_utc':int(r.valid_hour_utc),'lead_h':HOUR_TO_LEAD[int(r.valid_hour_utc)],
                'status':'UNRESOLVED_STORM_DATE','forecast_max_abs_diff_pa':np.nan}
                for r in group.itertuples(index=False))
            continue
        anchor_step=anchor['anchor_step']
        anchor_dt=anchor['anchor_datetime']
        storm_rows.append({'storm_key':storm,'status':'STORM_DATE_ANCHORED',
            'n_manifest':len(group),'n_npy':len(arr),'anchor_step':anchor_step,
            'anchor_valid_datetime':str(anchor_dt),'anchor_round':anchor['anchor_round'],
            'anchor_max_diff_pa':anchor['anchor_max_diff_pa'],'source_path':rec['path']})
        for r in group.itertuples(index=False):
            step=int(r.step_index)
            hour=int(r.valid_hour_utc)
            valid=anchor_dt+dt.timedelta(hours=6*(step-anchor_step))
            jobs.append((valid.date() - dt.timedelta(days=1) if valid.hour==0 else valid.date(),
                         storm,step,hour,valid))
    jobs.sort(key=lambda item:(item[0],item[4],item[1]))
    print(f'Validating {len(jobs):,} dated steps, sorted by NC file date',flush=True)
    for n,(_init_date, storm,step,hour,valid) in enumerate(jobs,1):
        if valid.hour!=hour:
            rows.append({'storm_key':storm,'step_index':step,'valid_hour_utc':hour,
                         'lead_h':HOUR_TO_LEAD[hour], 'valid_datetime':str(valid),
                         'status':'INCONSISTENT_HOUR_SEQUENCE',
                         'forecast_max_abs_diff_pa':np.nan})
        else:
            try:
                rows.append(classify_step(storm,step,valid,hour,chosen[storm]['npy'],a.nc_root,a.tolerance_pa))
            except Exception as exc:
                rows.append({'storm_key':storm,'step_index':step,'valid_hour_utc':hour,
                             'lead_h':HOUR_TO_LEAD[hour],'valid_datetime':str(valid),
                             'status':'CHECK_ERROR', 'note':repr(exc),
                             'forecast_max_abs_diff_pa':np.nan})
        if n%3000==0:
            print(f'  Validated {n:,}/{len(jobs):,} dated steps; total classified {len(rows):,}',flush=True)
            save_reports(rows,storm_rows,a,expected_samples=len(rows),expected_storms=len(storm_rows))
    save_reports(rows,storm_rows,a,expected_samples=len(df),expected_storms=len(groups))
    # Small independent metadata record from four representative verified rows.
    metadata_rows=[]
    verified=[r for r in rows if r.get('status') in
              ('CONFIRMED_NPY_MATCHES_EXPECTED_FC_NC','AMBIGUOUS_IDENTICAL_STATIC_AND_FORECAST')]
    for h in (6,12,18,24):
        example=next((r for r in verified if r.get('lead_h') == h and r.get('valid_datetime')),None)
        if example:
            init=dt.datetime.fromisoformat(example['init_datetime']).date()
            _,_,_,meta,nc_file=read_nc(str(a.nc_root),init)
            metadata_rows.append({'lead_h':h,'valid_datetime':example['valid_datetime'],
                                  'init_date':str(init),'path':nc_file,**meta})
    pd.DataFrame(metadata_rows).to_csv(a.out/'nc_metadata_examples.csv',index=False)
    print('Note: exact NC spatial matches demonstrate field provenance, not independent GRIB-step metadata.')

if __name__=='__main__':
    main()
