#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit IFS 2010 .npy files versus ERA5 and Control (6/12/18/24 h).

IMPORTANT SCIENTIFIC LIMITATION
-------------------------------
The hour array is assumed to contain VALID UTC hour (0,6,12,18), not lead time.
A forecast initialized at 00 UTC and verified at 06 UTC is a 6-h forecast;
the same verification hour could also be an 18-h forecast initialized at 12 UTC
on the previous day. A single 3-D forecast tensor (time, y, x) cannot hold both
forecasts at the same verification time. Thus the 'both' scenario is a diagnostic
of eligible timestamps, NOT verified availability of independent predictions.
Even 00-only and 12-only are *hypotheses* until the IFS source's initialization
metadata is checked. This script NEVER overwrites true_vs_pred_* files.

Run on Arctic:
    python audit_ifs_initialization_2010.py

Optional:
    python audit_ifs_initialization_2010.py \
        --old-ifs-dir '/path/to/old/CSV/files' --limit-storms 25

Uses paths below by default; all can be set via command-line flags.
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

LEADS = (6, 12, 18, 24)
LEAD_TO_CONTROL = {6: "control_31.csv", 12: "control_32.csv",
                   18: "control_33.csv", 24: "control_34.csv"}
INIT_SCENARIOS = {"only_00": (0,), "only_12": (12,), "both_00_12": (0, 12)}
OUTPUT_COLUMNS = [
    "scenario", "storm_key", "step_index", "horizon_h", "valid_hour_utc",
    "candidate_init_hour_utc", "in_control", "passes_control_start_rule",
    "true_mslp_ifs", "pred_mslp_ifs", "ifs_error_hpa",
    "ifs_min_error_vs_era5_center_hpa", "ifs_min_error_vs_control_truth_hpa",
    "era5_center_hpa", "control_true_mslp_corr", "control_pred_mslp_corr",
    "control_error_hpa", "center_vs_control_truth_absdiff_hpa",
    "forecast_lead_verified",
]


def parse_args():
    base = Path('data/steps_analysis')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--era5-root', type=Path, default=Path('data/era5'))
    p.add_argument('--ifs-dir', type=Path, default=base / 'Forecast 2010' / '2010')
    p.add_argument('--control-dir', type=Path, default=base / 'updated_data' / 'control')
    p.add_argument('--out', type=Path, default=base / 'IFS_2010_initialization_audit')
    p.add_argument('--year', type=int, default=2010)
    p.add_argument('--radius', type=int, default=8,
                   help='Radius 8 = (up to) 17x17 pixels, matching the old code.')
    p.add_argument('--old-ifs-dir', type=Path, default=None,
                   help='Optional folder holding old true_vs_pred_6h.csv etc.')
    p.add_argument('--limit-storms', type=int, default=None, help='For quick dry runs.')
    return p.parse_args()


def year_directory(root, name, year):
    folder = root / name
    with_year = folder / str(year)
    return with_year if with_year.is_dir() else folder


def normalize_storm(s):
    s = str(s).strip().strip('"\'')
    return re.sub(r'(?i)\.npy$', '', s).strip()


def load_controls(folder, year):
    lookups = {}
    source_info = []
    for lead in LEADS:
        path = folder / LEAD_TO_CONTROL[lead]
        if not path.is_file():
            raise FileNotFoundError('Missing Control file: {}'.format(path))
        df = pd.read_csv(path, usecols=['storm_key', 'step_index',
                                        'true_mslp_corr', 'pred_mslp_corr'],
                         dtype={'storm_key': str})
        df['storm_key'] = df['storm_key'].map(normalize_storm)
        df = df[df['storm_key'].str.contains('_{}_'.format(year), regex=False)].copy()
        df['step_index'] = pd.to_numeric(df['step_index'], errors='raise').astype(int)
        key_cols = ['storm_key', 'step_index']
        if df.duplicated(key_cols).any():
            raise ValueError('Duplicate storm/step keys in {}'.format(path))
        lookups[lead] = {
            (r[0], int(r[1])): (float(r[2]), float(r[3]))
            for r in df[key_cols + ['true_mslp_corr','pred_mslp_corr']].itertuples(index=False, name=None)
        }
        min_by_storm = df.groupby('storm_key')['step_index'].min()
        expected = 3 + lead // 6
        source_info.append({
            'horizon_h': lead, 'control_2010_samples': len(df),
            'control_2010_storms': df.storm_key.nunique(),
            'control_expected_min_step': expected,
            'observed_min_step': int(df.step_index.min()) if len(df) else None,
            'storm_count_first_step_not_expected': int((min_by_storm != expected).sum()),
        })
    return lookups, pd.DataFrame(source_info)


def read_hour(frame):
    """UTC hour from hour.npy time slice; every spatial cell must be identical."""
    a = np.asarray(frame)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError('Missing/nonfinite hour field')
    v = float(a.flat[0])
    if not np.allclose(a, v, atol=1e-6, rtol=0):
        raise ValueError('hour grid is NOT spatially constant')
    if not np.isclose(v, round(v), atol=1e-6):
        raise ValueError('Noninteger UTC hour: {}'.format(v))
    h = int(round(v))
    if h not in (0, 6, 12, 18):
        raise ValueError('Unexpected hour {}; expected 0,6,12,18'.format(h))
    return h


def pressure_factor(arr):
    """Detect Pa/hPa using a valid frame, even if early frames are missing."""
    trial = [4, 5, 6, 7, 0, 1, 2, 3, len(arr)//2, len(arr)-1]
    observed = []
    for t in trial:
        if t < 0 or t >= len(arr):
            continue
        probe = np.asarray(arr[t], dtype=float)
        finite = probe[np.isfinite(probe)]
        if finite.size == 0:
            continue
        median = float(np.median(finite))
        if 60000 < median < 120000:
            return 0.01, 'Pa'
        if 600 < median < 1200:
            return 1.0, 'hPa'
        observed.append(median)
    raise ValueError('No plausible pressure field: trial medians {}'.format(observed))


def local_min(a, index, radius, factor):
    field = np.asarray(a[index], dtype=float)
    ny, nx = field.shape
    cy, cx = ny // 2, nx // 2
    y0, y1 = max(0, cy - radius), min(ny, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(nx, cx + radius + 1)
    w = field[y0:y1, x0:x1]
    if not np.isfinite(w).any():
        raise ValueError('No finite values within center window')
    return float(np.nanmin(w) * factor), float(field[cy, cx] * factor)


def issue_row(info, status, detail=''):
    info['status'], info['detail'] = status, detail
    return info


def collect(args, control):
    r_dir = year_directory(args.era5_root, 'msl', args.year)
    h_dir = year_directory(args.era5_root, 'hour', args.year)
    f_dir = args.ifs_dir
    if f_dir.resolve() == r_dir.resolve():
        raise ValueError('IFS forecast directory equals ERA5 truth directory. '
                         'Comparing the same files would generate artificial zero errors.')
    for d in (r_dir, h_dir, f_dir):
        if not d.is_dir():
            raise FileNotFoundError('Directory not found: {}'.format(d))
    print('\nInput folders:')
    print('  IFS forecast MSL:', f_dir)
    print('  ERA5 truth MSL  :', r_dir)
    print('  Valid UTC hours :', h_dir)
    print('  Center min radius:', args.radius, '(radius 8 means a 17x17 window)')

    files = sorted(f_dir.glob('*.npy'))
    files = [p for p in files if '_{}_'.format(args.year) in p.stem]
    if args.limit_storms is not None:
        files = files[:args.limit_storms]
    if not files:
        raise ValueError('No {} storm .npy files in {}'.format(args.year, f_dir))

    available_r = {p.stem for p in r_dir.glob('*.npy')}
    available_h = {p.stem for p in h_dir.glob('*.npy')}
    available_f = {p.stem for p in files}
    audit = []
    output = []
    time_rows = []
    bad_hours = 0
    checked_hours = 0
    print('  Forecast storm files:', len(files))
    print('  Missing ERA5 files:', len(available_f - available_r))
    print('  Missing hour files:', len(available_f - available_h))

    for n_file, fp in enumerate(files, 1):
        storm = fp.stem
        info = {'storm_key': storm, 'ifs_file': fp.name,
                'has_era5': storm in available_r, 'has_hour': storm in available_h}
        if storm not in available_r or storm not in available_h:
            audit.append(issue_row(info, 'missing_input', 'Missing ERA5 and/or hour .npy'))
            continue
        try:
            F = np.load(str(fp), mmap_mode='r', allow_pickle=False)
            R = np.load(str(r_dir / fp.name), mmap_mode='r', allow_pickle=False)
            H = np.load(str(h_dir / fp.name), mmap_mode='r', allow_pickle=False)
            info.update(ifs_length=len(F), era5_length=len(R), hour_length=len(H),
                        ifs_shape=str(F.shape), era5_shape=str(R.shape),
                        hour_shape=str(H.shape))
            if F.ndim != 3 or R.ndim != 3 or H.ndim not in (1, 2, 3):
                raise ValueError('Unexpected shape: F={}, R={}, H={}'.format(F.shape, R.shape, H.shape))
            if F.shape != R.shape or len(H) != len(F):
                raise ValueError('Unaligned dimensions: F={}, R={}, H={}'.format(F.shape, R.shape, H.shape))
            f_factor, f_unit = pressure_factor(F)
            r_factor, r_unit = pressure_factor(R)
            info['ifs_pressure_unit'], info['era5_pressure_unit'] = f_unit, r_unit
            valid_hours = []
            for t in range(len(F)):
                try:
                    valid_hours.append(read_hour(H[t]))
                except ValueError as e:
                    raise ValueError('Invalid hour at step {}: {}'.format(t, e))
            unexpected_time_jumps = [t for t in range(1,len(F))
                                     if (valid_hours[t] - valid_hours[t-1]) % 24 != 6]
            info['n_bad_6h_steps'] = len(unexpected_time_jumps)
            info['first_hour'] = valid_hours[0]
            info['hour_counts'] = ';'.join('{}:{}'.format(h,valid_hours.count(h)) for h in (0,6,12,18))
            checked_hours += 1
            if unexpected_time_jumps:
                bad_hours += 1
            # Store the local-minimum value only once per valid step, then test
            # possible initialization/lead assignments without fabricating forecasts.
            invalid_pressure_steps = 0
            for t, v_hour in enumerate(valid_hours):
                try:
                    pred, fcenter = local_min(F, t, args.radius, f_factor)
                    truth, rcenter = local_min(R, t, args.radius, r_factor)
                    if not (600 < pred < 1200 and 600 < truth < 1200):
                        raise ValueError('Implausible minimum: forecast={}, ERA5={}'.format(pred, truth))
                except ValueError:
                    # Record, and do not create a potentially fabricated forecast.
                    invalid_pressure_steps += 1
                    continue
                time_rows.append({'storm_key':storm, 'step_index':t,
                                  'valid_hour_utc':v_hour,
                                  'true_mslp_ifs_from_npy':truth,
                                  'pred_mslp_ifs_from_npy':pred,
                                  'era5_center_hpa':rcenter})
                for horizon in LEADS:
                    if t < horizon // 6:
                        continue    # Need enough 6-hour steps to reach the valid time
                    init_hour = (v_hour - horizon) % 24
                    if init_hour not in (0, 12):
                        continue
                    cvals = control[horizon].get((storm, t))
                    has_control = cvals is not None
                    control_truth, control_pred = cvals if has_control else (np.nan, np.nan)
                    history_ok = t >= 3 + horizon // 6
                    ctrl_err = control_pred - control_truth if has_control else np.nan
                    base_row = {
                        'storm_key': storm, 'step_index': t, 'horizon_h': horizon,
                        'valid_hour_utc': v_hour,
                        'candidate_init_hour_utc': init_hour,
                        'in_control': has_control,
                        'passes_control_start_rule': history_ok,
                        'true_mslp_ifs': truth,
                        'pred_mslp_ifs': pred,
                        'ifs_error_hpa': pred - truth,
                        'ifs_min_error_vs_era5_center_hpa': pred - rcenter,
                        'ifs_min_error_vs_control_truth_hpa': (pred-control_truth) if has_control else np.nan,
                        'era5_center_hpa': rcenter,
                        'control_true_mslp_corr': control_truth,
                        'control_pred_mslp_corr': control_pred,
                        'control_error_hpa': ctrl_err,
                        'center_vs_control_truth_absdiff_hpa': abs(rcenter - control_truth)
                            if has_control else np.nan,
                        'forecast_lead_verified': False,
                    }
                    # A 3D IFS tensor holds one field per valid time; even if
                    # 00+12 cycles exist globally, which one this field belongs
                    # to is NOT known without init/step metadata.
                    for scenario, allowed in INIT_SCENARIOS.items():
                        if init_hour in allowed:
                            output.append(dict(base_row, scenario=scenario))
            info['invalid_pressure_steps'] = invalid_pressure_steps
            audit.append(issue_row(info, 'ok'))
        except Exception as e:
            audit.append(issue_row(info, 'invalid_input', '{}: {}'.format(type(e).__name__, e)))
        if n_file % 200 == 0:
            print('  Scanned {:,}/{:,} forecast files'.format(n_file, len(files)))
    print('  Files with non-6-hour jumps:', bad_hours, '/', checked_hours)
    return (pd.DataFrame(output, columns=OUTPUT_COLUMNS),
            pd.DataFrame(audit), pd.DataFrame(time_rows))


def make_summary(candidates, control_info, audit):
    rows = []
    for scenario in INIT_SCENARIOS:
        for horizon in LEADS:
            data = candidates.loc[(candidates.scenario == scenario) &
                                  (candidates.horizon_h == horizon)]
            hist = data.loc[data.passes_control_start_rule]
            matched = data.loc[data.in_control & data.passes_control_start_rule]
            rows.append({
                'scenario': scenario, 'horizon_h': horizon,
                'candidate_n_from_ifs': len(data),
                'candidate_storms': data.storm_key.nunique(),
                'candidate_n_after_control_start': len(hist),
                'matched_control_n': len(matched),
                'matched_control_storms': matched.storm_key.nunique(),
                'ifs_mae_local_min_on_matched_hpa': matched.ifs_error_hpa.abs().mean(),
                'ifs_bias_local_min_on_matched_hpa': matched.ifs_error_hpa.mean(),
                'ifs_min_vs_era5_center_mae_hpa': matched.ifs_min_error_vs_era5_center_hpa.abs().mean(),
                'ifs_min_vs_control_truth_mae_hpa': matched.ifs_min_error_vs_control_truth_hpa.abs().mean(),
                'control_mae_tracked_center_on_matched_hpa': matched.control_error_hpa.abs().mean(),
                'era5_center_vs_control_truth_mae_hpa':
                    matched.center_vs_control_truth_absdiff_hpa.mean(),
                'lead_identity_verified': False,
            })
    df = pd.DataFrame(rows).merge(control_info, on='horizon_h', how='left')
    print('\n============ SAMPLE-COUNT AND ERROR SUMMARY ============')
    print(df[['scenario','horizon_h','candidate_n_from_ifs',
              'candidate_n_after_control_start','matched_control_n',
              'ifs_mae_local_min_on_matched_hpa',
              'control_mae_tracked_center_on_matched_hpa']].to_string(index=False,
                                                    float_format=lambda v:'%.3f'%v))
    print('\nSource file statuses:')
    print(audit.status.value_counts(dropna=False).to_string())
    print('\nWARNING: All scenario forecasts are HYPOTHESES, not validated lead times.')
    print('The both_00_12 scenario reuses ONE IFS field at a given timestamp for')
    print('two distinct hypothetical leads. This is NOT a corrected IFS dataset.')
    print('The Control MAE is verified against tracked-center truth; IFS uses local minima.')
    return df


def compare_old(candidates, all_times, directory, out):
    if directory is None:
        return
    reports = []
    hour_reports = []
    for horizon in LEADS:
        p = directory / 'true_vs_pred_{}h.csv'.format(horizon)
        if not p.is_file():
            print('Old IFS CSV not found, skipping:', p)
            continue
        old = pd.read_csv(p)
        if not {'storm_key','step_index','true_mslp_ifs','pred_mslp_ifs'}.issubset(old.columns):
            print('Old IFS columns unrecognized, skipping:', p)
            continue
        old.storm_key = old.storm_key.map(normalize_storm)
        old.step_index = pd.to_numeric(old.step_index).astype(int)
        if old.duplicated(['storm_key','step_index']).any():
            print('WARNING: duplicate keys in old CSV',p)
        old = old.drop_duplicates(['storm_key','step_index'])
        # Critical check independent of any scenario: recover the UTC verification
        # hour for each row previously exported to a true_vs_pred CSV.
        with_hour = old.merge(all_times, on=['storm_key','step_index'], how='left')
        for valid_hour, z in with_hour.groupby('valid_hour_utc', dropna=False):
            hour_reports.append({
                'old_ifs_horizon_h':horizon,
                'valid_hour_utc':valid_hour,
                'number_of_old_rows':len(z),
                'candidate_init_hour_implied_by_valid_hour':
                    (int(valid_hour)-horizon)%24 if pd.notna(valid_hour) else np.nan,
                'mean_abs_pred_old_vs_npy_hpa':
                    (z.pred_mslp_ifs-z.pred_mslp_ifs_from_npy).abs().mean(),
                'mean_abs_truth_old_vs_npy_hpa':
                    (z.true_mslp_ifs-z.true_mslp_ifs_from_npy).abs().mean(),
            })
        base = candidates[candidates.horizon_h == horizon]
        for scenario in INIT_SCENARIOS:
            sub = base[base.scenario == scenario]
            paired = sub.merge(old[['storm_key','step_index','true_mslp_ifs','pred_mslp_ifs']],
                               on=['storm_key','step_index'], how='inner', suffixes=('_new','_old'))
            reports.append({
                'scenario': scenario, 'horizon_h': horizon,
                'old_n': len(old), 'candidate_n': len(sub),
                'old_keys_present_in_candidate': len(paired),
                'mean_abs_pred_diff_hpa': (paired.pred_mslp_ifs_new - paired.pred_mslp_ifs_old).abs().mean(),
                'mean_abs_truth_diff_hpa': (paired.true_mslp_ifs_new - paired.true_mslp_ifs_old).abs().mean(),
            })
    if hour_reports:
        hour_p = out / 'old_ifs_hour_distribution.csv'
        hr = pd.DataFrame(hour_reports)
        hr.to_csv(hour_p, index=False)
        print('\nOLD IFS CSV: VALID UTC HOURS (most informative diagnosis)')
        print(hr.to_string(index=False, float_format=lambda v: '%.4f' % v))
        print('Written:',hour_p)
    if reports:
        out_p = out / 'old_ifs_vs_candidates.csv'
        pd.DataFrame(reports).to_csv(out_p, index=False)
        print('\nComparison with old true_vs_pred CSVs:',out_p)
        print(pd.DataFrame(reports).to_string(index=False, float_format=lambda v: '%.5f' % v))


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    control, control_info = load_controls(args.control_dir, args.year)
    print('Control 2010 counts:')
    print(control_info.to_string(index=False))
    candidates, source_audit, all_times = collect(args, control)
    if candidates.empty:
        print('No candidates. Inspect source_file_audit.csv for reasons.')
        source_audit.to_csv(args.out / 'source_file_audit.csv', index=False)
        raise SystemExit(1)
    summary = make_summary(candidates, control_info, source_audit)
    summary.to_csv(args.out / 'scenario_summary.csv', index=False)
    source_audit.to_csv(args.out / 'source_file_audit.csv', index=False)
    all_times.to_csv(args.out / 'source_all_valid_times.csv', index=False)
    for scenario in INIT_SCENARIOS:
        for horizon in LEADS:
            z = candidates[(candidates.scenario == scenario) &
                           (candidates.horizon_h == horizon)]
            p = args.out / 'CANDIDATE_{}_{}h.csv'.format(scenario, horizon)
            z.to_csv(p, index=False, float_format='%.8f')
    compare_old(candidates, all_times, args.old_ifs_dir, args.out)
    # Per-storm counts show whether losses are whole storms or individual steps.
    grouped = (candidates.groupby(['scenario','horizon_h','storm_key'], dropna=False)
               .agg(n_candidate=('step_index','size'),
                    n_after_control_start=('passes_control_start_rule','sum'),
                    n_matched_control=('in_control','sum'),
                    first_step=('step_index','min'), last_step=('step_index','max'))
               .reset_index())
    grouped.to_csv(args.out/'storm_counts.csv', index=False)
    print('\nWritten to:',args.out)
    print('  scenario_summary.csv : 12 rows (3 scenarios x 4 lead times)')
    print('  source_file_audit.csv: missing inputs, mismatched shapes, invalid hours')
    print('  source_all_valid_times.csv: EVERY source step + hour + predicted/true minima')
    print('  storm_counts.csv     : per-storm time and coverage')
    print('  CANDIDATE_*.csv      : hypotheses, NOT corrected forecast tables')
    print('  old_ifs_hour_distribution.csv: old CSV hour/lead mismatch (optional)')
    print('  old_ifs_vs_candidates.csv: only if --old-ifs-dir is supplied')

if __name__ == '__main__':
    main()
