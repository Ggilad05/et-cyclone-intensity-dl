#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute and audit ONE full-year supplementary metrics table from original predictions.

Based on the user's original V9 manuscript code and original full-year SS script.
No 2010 IFS or local-minimum data are used here. Python 3.9+; numpy/pandas only.

The ORIGINAL table deliberately mixes these samples:
  MAE / Bias / STD: each model's own full-year available forecast steps;
  SS for Leakage / Control / Linear: the exact complete-case intersection of
    Control + Linear + Persistence + Leakage on (storm_key, step_index).
  SS confidence intervals: storm-cluster bootstrap; Bias confidence intervals:
    normal approximation, not cluster bootstrap (preserves original method).
  Simple NN SS: its own 6h storm/step intersection with Control persistence.
  MSL-only SS: calculated and audited, but left as '--' in the paper by default
    to preserve the previous supplementary table's reporting choice.

Important: This program does NOT simply read rounded numbers from LaTeX.
It recomputes everything from the source CSV files. Former table values are
used only for the separate comparison/audit report.

Run (Mac):
  python build_supplementary_full_dataset_table.py \\
      --base-dir data/updated_data

Outputs into outputs/supplementary_full_dataset/ by default (run from repository root).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEADS = (6, 12, 18, 24)
MODELS = ('Leakage', 'Control', 'MSL-only', 'Simple NN', 'Linear Regression', 'Persistence')
METRICS = ('MAE', 'Bias', 'STD', 'SS')
KEY = ['storm_key', 'step_index']
CONTROL_FILES = {6: 'control_31.csv', 12: 'control_32.csv',
                 18: 'control_33.csv', 24: 'control_34.csv'}
LEAKAGE_FILES = {h: f'leakage_{h}d.csv' for h in LEADS}
ORIGINAL_HEADER_N = {6: 639502, 12: 594124, 18: 548746, 24: 503368}
Z_95 = 1.959963984540054
TRUTH_TOL = 1e-5  # hPa, same as original V9 checks

# Populated below with historical rounded values for COMPARISON ONLY.
# This constant is not used by the numerical calculation or the generated table.
REFERENCE_JSON = r'''{"12": {"Control": {"Bias": [0.032, 0.026, 0.038], "MAE": [1.752], "SS": [0.575, 0.572, 0.577], "STD": [2.319]}, "Leakage": {"Bias": [0.057, 0.052, 0.063], "MAE": [1.562], "SS": [0.619, 0.617, 0.621], "STD": [2.073]}, "Linear Regression": {"Bias": [-0.054, -0.065, -0.043], "MAE": [3.192], "SS": [0.205, 0.203, 0.208], "STD": [4.309]}, "Persistence": {"Bias": [0.272, 0.258, 0.286], "MAE": [4.027], "STD": [5.441]}}, "18": {"Control": {"Bias": [-0.013, -0.021, -0.004], "MAE": [2.509], "SS": [0.574, 0.571, 0.576], "STD": [3.304]}, "Leakage": {"Bias": [0.021, 0.012, 0.031], "MAE": [2.259], "SS": [0.614, 0.612, 0.616], "STD": [2.969]}, "Linear Regression": {"Bias": [-0.099, -0.115, -0.082], "MAE": [4.709], "SS": [0.177, 0.175, 0.18], "STD": [6.274]}, "Persistence": {"Bias": [0.451, 0.431, 0.471], "MAE": [5.638], "STD": [7.592]}}, "24": {"Control": {"Bias": [0.154, 0.142, 0.166], "MAE": [3.314], "SS": [0.553, 0.55, 0.556], "STD": [4.372]}, "Leakage": {"Bias": [0.119, 0.106, 0.132], "MAE": [2.905], "SS": [0.607, 0.604, 0.61], "STD": [3.826]}, "Linear Regression": {"Bias": [-0.15, -0.172, -0.128], "MAE": [5.931], "SS": [0.175, 0.172, 0.177], "STD": [7.889]}, "Persistence": {"Bias": [0.667, 0.641, 0.693], "MAE": [7.022], "STD": [9.449]}}, "6": {"Control": {"Bias": [-0.045, -0.049, -0.041], "MAE": [1.118], "SS": [0.515, 0.513, 0.517], "STD": [1.501]}, "Leakage": {"Bias": [-0.015, -0.018, -0.011], "MAE": [0.959], "SS": [0.581, 0.579, 0.583], "STD": [1.297]}, "Linear Regression": {"Bias": [-0.019, -0.025, -0.013], "MAE": [1.903], "SS": [0.178, 0.176, 0.181], "STD": [2.539]}, "MSL-only": {"Bias": [-0.068, -0.071, -0.064], "MAE": [1.144], "STD": [1.511]}, "Persistence": {"Bias": [0.126, 0.118, 0.133], "MAE": [2.318], "STD": [3.103]}, "Simple NN": {"Bias": [-0.012, -0.018, -0.006], "MAE": [1.758], "SS": [0.228, 0.225, 0.23], "STD": [2.387]}}}'''
REFERENCE = json.loads(REFERENCE_JSON)


def normalize_keys(d: pd.DataFrame, source: Path) -> pd.DataFrame:
    d = d.copy()
    d['storm_key'] = (d['storm_key'].astype('string').str.strip().str.strip('\\\"\'')
                      .str.replace(r'(?i)\.npy$', '', regex=True).str.strip())
    if d.storm_key.isna().any() or d.storm_key.eq('').any():
        raise ValueError(f'{source}: empty storm_key')
    ind = pd.to_numeric(d.step_index, errors='coerce')
    if (ind.isna().any() or not np.isfinite(ind.to_numpy(float)).all()
            or not np.equal(ind.to_numpy(float), np.floor(ind.to_numpy(float))).all()):
        raise ValueError(f'{source}: non-integer or missing step_index')
    d['step_index'] = ind.astype('int64')
    if d.duplicated(KEY).any():
        example = d.loc[d.duplicated(KEY, keep=False), KEY].head(5).to_dict('records')
        raise ValueError(f'{source}: duplicate keys, e.g. {example}')
    return d


def read_predictions(path: Path, required: list[str], optional: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing required CSV: {path}')
    hdr = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    missing = [c for c in required if c not in hdr]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}; found {hdr}')
    columns = list(dict.fromkeys(required + [x for x in optional if x in hdr]))
    d = pd.read_csv(path, usecols=columns, dtype={'storm_key': 'string'})
    d.columns = d.columns.str.strip()
    if d.empty:
        raise ValueError(f'{path}: empty dataset')
    d = normalize_keys(d, path)
    for col in d.columns:
        if col in KEY:
            continue
        d[col] = pd.to_numeric(d[col], errors='coerce')
        if not np.isfinite(d[col].to_numpy(float)).all():
            raise ValueError(f'{path}: non-finite {col}; refusing silent row drops')
    return d


def column_choice(path: Path, options: tuple[str, ...]) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    hdr = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    return next((name for name in options if name in hdr), None) or _missing_col(path, hdr, options)


def _missing_col(path, header, choices):
    raise ValueError(f'{path}: expected one of {choices}; got {header}')


def describe_errors(error: np.ndarray) -> dict:
    e = np.asarray(error, dtype=float)
    if e.ndim != 1 or len(e) <= 1 or not np.isfinite(e).all():
        raise ValueError('Invalid signed errors, require 2+ finite samples')
    n = len(e)
    bias = float(e.mean())
    std = float(e.std(ddof=1))
    half_width = Z_95 * std / np.sqrt(n)
    rmse = float(np.sqrt(np.mean(e**2)))
    # Explicitly verify identity used by last discussion: population RMSE,
    # sample STD, and sample count N.
    if not np.isclose(rmse**2, bias**2 + (n-1)/n*std**2, rtol=1e-9, atol=1e-9):
        raise AssertionError('RMSE decomposition inconsistent')
    return dict(N=n, MAE=float(np.mean(np.abs(e))), Bias=bias, STD=std,
                Bias_CI_low=bias-half_width, Bias_CI_high=bias+half_width,
                RMSE=rmse)


def ss_cluster_bootstrap(storms: np.ndarray, error: np.ndarray,
                         base_error: np.ndarray, bootstrap: int, seed: int) -> dict:
    """Storm-resampling with equal *forecast-step* weights, like original SS code.

    Baseline and model errors must already be aligned on exactly the same keys.
    Counts, sum(squared errors) aggregated within storms; whole storms resampled.
    """
    e = np.asarray(error, float)
    b = np.asarray(base_error, float)
    keys = np.asarray(storms).astype(str)
    if len(e) != len(b) or len(e) != len(keys) or not len(e):
        raise ValueError('SS inputs have nonmatching lengths or zero rows')
    if not np.isfinite(e).all() or not np.isfinite(b).all():
        raise ValueError('Non-finite SS input')
    _, indices = np.unique(keys, return_inverse=True)
    n_storms = int(indices.max()) + 1
    counts = np.bincount(indices, minlength=n_storms)
    model_sq = np.bincount(indices, weights=e*e, minlength=n_storms)
    base_sq = np.bincount(indices, weights=b*b, minlength=n_storms)
    model_rmse = float(np.sqrt(model_sq.sum()/counts.sum()))
    base_rmse = float(np.sqrt(base_sq.sum()/counts.sum()))
    if base_rmse < 1e-10:
        raise ValueError('Persistence RMSE is zero: SS is undefined')
    actual = 1-model_rmse/base_rmse
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap)
    for k in range(bootstrap):
        selected = rng.integers(0, n_storms, size=n_storms)
        base_sum = base_sq[selected].sum()
        samples[k] = 1-np.sqrt(model_sq[selected].sum()/base_sum) if base_sum>0 else np.nan
    samples = samples[np.isfinite(samples)]
    if len(samples) < max(10, bootstrap*.95):
        raise ValueError('SS bootstrap had too many invalid replicates')
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return dict(SS=float(actual), SS_CI_low=float(lo), SS_CI_high=float(hi),
                SS_N=len(e), SS_Storms=n_storms, SS_Model_RMSE=model_rmse,
                SS_Persistence_RMSE=base_rmse)


def safe_tex_name(s):
    return s.replace('-', '--')  # hyphen acceptable in tabular text


def print_summary(label: str, d: dict):
    details = f"SS={d['SS']:.3f} [{d['SS_CI_low']:.3f}, {d['SS_CI_high']:.3f}]" if 'SS' in d else 'SS=--'
    print(f"  {label:18s} N={d['N']:>8,} MAE={d['MAE']:.3f} "
          f"Bias={d['Bias']:+.3f} STD={d['STD']:.3f} | {details}")


def table_text(stats: dict, n_header: dict, include_msl_ss: bool) -> str:
    """Return a single AMS/booktabs table, four horizon columns, no split tables."""
    lines = [
        r'% Auto-generated from raw CSVs by build_supplementary_full_dataset_table.py.',
        r'% DO NOT use historical values from the verification report as numeric inputs.',
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Performance metrics for the full multi-year ERA5 dataset (1979--2020)',
        r'at 6, 12, 18, and 24~h forecast lead times. The Mean Absolute Error (MAE),',
        r'mean error (Bias; predicted minus observed cyclone-center MSL), and',
        r'standard deviation of signed errors (STD) are evaluated over the available',
        r'samples of each model. Parentheses for Bias indicate normal-approximation',
        r'95\% confidence intervals. Skill Score (SS) is defined as',
        r'$1-\mathrm{RMSE}_{\mathrm{model}}/\mathrm{RMSE}_{\mathrm{Persistence}}$.',
        r'For Control, Leakage, and Linear Regression, SS is computed on the same',
        r'complete-case intersection of cyclone and time-step identifiers, which can',
        r'differ from the samples used for the other metrics. For Simple NN, the',
        r'SS comparison uses its available 6~h steps matched to Control persistence.',
        r'Parentheses for SS indicate 95\% cyclone-cluster bootstrap intervals.',
        r'The column-header $N$ gives the maximum number of available core-model',
        r'samples at each lead and is not necessarily the sample count for every',
        r'model and measure; see the machine-readable sample-count audit.',
        r'The Persistence SS is zero by definition and is not separately tabulated.',
        r'Dashes indicate unavailable or unreported model--metric combinations.}',
        r'\label{tab:supp_full_metrics}',
        r'\begingroup', r'\setlength{\tabcolsep}{4.5pt}',
        r'\renewcommand{\arraystretch}{1.10}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{llcccc}', r'\toprule',
        'Measure & Model & ' + ' & '.join(f'{h} h ($N{{=}}{n_header[h]:,}$)'.replace(',', r'{,}')
                                    for h in LEADS) + r' \\',
        r'\midrule'
    ]
    label = {'MAE':'MAE (hPa)', 'Bias':'Bias (hPa)', 'STD':'STD (hPa)', 'SS':'SS'}
    for metric in METRICS:
        displayed_models = tuple(m for m in MODELS if not (metric=='SS' and m=='Persistence'))
        for j, model in enumerate(displayed_models):
            cells = []
            for h in LEADS:
                r = stats.get((h, model))
                if (r is None or (metric=='SS' and model=='MSL-only' and not include_msl_ss)
                        or (metric=='SS' and 'SS' not in r)):
                    cells.append('--')
                elif metric in ('Bias','SS'):
                    cells.append((f"${r[metric]:.3f} "
                                  f"({r[metric+'_CI_low']:.3f}, "
                                  f"{r[metric+'_CI_high']:.3f})$"))
                else:
                    cells.append(f'${r[metric]:.3f}$')
            lines.append((label[metric] if j==0 else '')+' & '+model+' & '
                         +' & '.join(cells)+r' \\')
        if metric != 'SS':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}%', r'}', r'\endgroup', r'\end{table}']
    return '\n'.join(lines)+'\n'


def compare_previous(lead, model, measures: dict, include_msl_ss: bool):
    original = REFERENCE.get(str(lead), {}).get(model, {})
    out = []
    for metric in METRICS:
        if model=='MSL-only' and metric=='SS' and not include_msl_ss:
            continue
        vals = original.get(metric)
        if vals is None:
            if metric=='SS' and 'SS' in measures:
                out.append(dict(lead_h=lead, Model=model, Metric=metric,
                                Old_value=np.nan, New_value=measures[metric],
                                Difference=np.nan, Status='NOT_PREVIOUSLY_REPORTED'))
            continue
        cur = measures.get(metric)
        if cur is None:
            out.append(dict(lead_h=lead, Model=model, Metric=metric,
                            Old_value=vals[0], New_value=np.nan,
                            Difference=np.nan, Status='MISSING_NEW_RESULT'))
            continue
        # Point estimates have 3-digit historical rounding; allow at most 0.001.
        delta = cur-vals[0]
        out.append(dict(lead_h=lead, Model=model, Metric=metric,
                        Old_value=vals[0], New_value=cur, Difference=delta,
                        Status='OK' if abs(delta)<.0011 else 'CHECK_POINT_ESTIMATE'))
        if metric in ('Bias','SS') and len(vals)==3:
            # Bootstrap limits can change slightly depending on RNG/replicates.
            tolerance = .003 if metric=='Bias' else .006
            for pos, lim in [('low',vals[1]),('high',vals[2])]:
                name=f'{metric}_CI_{pos}'
                delta=measures[name]-lim
                out.append(dict(lead_h=lead, Model=model, Metric=name,
                                Old_value=lim, New_value=measures[name],Difference=delta,
                                Status='OK' if abs(delta)<tolerance else 'CHECK_CI'))
    return out


def build(args):
    base = args.base_dir.expanduser()
    control_dir = (args.control_dir or base/'control').expanduser()
    leakage_dir = (args.leakage_dir or base/'leakage').expanduser()
    output_dir = (args.output_dir or Path('outputs/supplementary_full_dataset')).expanduser()
    ablations = {
        'MSL-only': (args.msl_only_file or base/'msl_only'/'all_cyclone_data_full_31_version_156_157_158_159_6h.csv').expanduser(),
        'Simple NN': (args.simple_nn_file or base/'simple_nn'/'all_cyclone_data_full_31_version_0_1_2_3.csv').expanduser()
    }
    results = {}
    audit_rows, comparisons, input_paths, notes = [], [], [], []
    n_header = {}
    for lead in LEADS:
        cp, lp = control_dir/CONTROL_FILES[lead], leakage_dir/LEAKAGE_FILES[lead]
        linear = column_choice(cp, ('linear_pred', 'lin_pred'))
        c = read_predictions(cp, KEY+['true_mslp_corr','pred_mslp_corr','persistence_pred',linear])
        leakage_pred_col = column_choice(lp, ('pred_mslp_corr','dl_pred','pred'))
        l = read_predictions(lp, KEY+[leakage_pred_col], optional=('true_mslp_corr',))
        l = l.rename(columns={leakage_pred_col:'leakage_pred',
                              **({'true_mslp_corr':'leakage_truth'} if 'true_mslp_corr' in l else {})})
        joined = c.merge(l, on=KEY, how='inner', validate='one_to_one')
        if joined.empty:
            raise ValueError(f'{lead}h: no overlapping Control / Leakage cases')
        if 'leakage_truth' in joined:
            err_truth = float(np.max(np.abs(joined.leakage_truth-joined.true_mslp_corr)))
            if err_truth>TRUTH_TOL:
                raise ValueError(f'{lead}h: Leakage/Control ERA5 target mismatch {err_truth:.8g} hPa')
        else:
            err_truth = None
        # Original SS cohort equals the complete-case intersection of all
        # three models (and persistence). All columns are strictly finite.
        common = joined
        n_header[lead] = max(len(c), len(joined))
        if len(c) != ORIGINAL_HEADER_N[lead]:
            notes.append(f'WARNING {lead}h Control count {len(c):,} differs from old header {ORIGINAL_HEADER_N[lead]:,}')
        print(f'\n[{lead} h] Control N={len(c):,}; Leakage/Control common N={len(common):,}; '
              f'storms={common.storm_key.nunique():,}')
        input_paths += [dict(lead_h=lead,Model='Control/Linear/Persistence',File=str(cp),N=len(c)),
                        dict(lead_h=lead,Model='Leakage',File=str(lp),N=len(l),
                             N_matched=len(joined))]
        truth, basepred = c.true_mslp_corr, c.persistence_pred
        desc = {}
        desc['Control'] = describe_errors((c.pred_mslp_corr-truth).to_numpy(float))
        desc['Linear Regression'] = describe_errors((c[linear]-truth).to_numpy(float))
        desc['Persistence'] = describe_errors((basepred-truth).to_numpy(float))
        desc['Leakage'] = describe_errors((joined.leakage_pred-joined.true_mslp_corr).to_numpy(float))

        p_error = (common.persistence_pred-common.true_mslp_corr).to_numpy(float)
        ss_columns = [('Leakage','leakage_pred'),('Control','pred_mslp_corr'),
                      ('Linear Regression',linear)]
        for idx, (model,col) in enumerate(ss_columns):
            error = (common[col]-common.true_mslp_corr).to_numpy(float)
            ss = ss_cluster_bootstrap(common.storm_key.to_numpy(str),error,p_error,
                                      args.bootstraps,args.seed+lead*100+idx)
            desc[model].update(ss)
        desc['Persistence'].update(dict(SS=0.0, SS_CI_low=0., SS_CI_high=0.,
                                        SS_N=len(common), SS_Storms=common.storm_key.nunique(),
                                        SS_Model_RMSE=float(np.sqrt(np.mean(p_error**2))),
                                        SS_Persistence_RMSE=float(np.sqrt(np.mean(p_error**2)))))
        for model, r in desc.items():
            r['metric_cohort'] = 'available_model_rows' if model!='Leakage' else 'control_leakage_intersection'
            r['ss_cohort'] = 'control_leakage_complete_cases'
            results[(lead,model)] = r
            print_summary(model,r)
            audit_rows.append(dict(lead_h=lead,Model=model,Model_N=r['N'],
                    SS_N=r.get('SS_N',np.nan), SS_Storms=r.get('SS_Storms',np.nan),
                    MAE=r['MAE'],Bias=r['Bias'],Bias_CI_low=r['Bias_CI_low'],
                    Bias_CI_high=r['Bias_CI_high'],STD=r['STD'],RMSE=r['RMSE'],
                    Persistence_RMSE_on_SS_samples=r.get('SS_Persistence_RMSE',np.nan),
                    SS=r.get('SS',np.nan),SS_CI_low=r.get('SS_CI_low',np.nan),
                    SS_CI_high=r.get('SS_CI_high',np.nan),
                    metric_cohort=r['metric_cohort'],ss_cohort=r['ss_cohort'],
                    Header_N=n_header[lead]))
            comparisons.extend(compare_previous(lead,model,r,args.include_msl_ss))
        notes.append(f'{lead}h core cohort: Control/Linear/Persistence metrics N={len(c)}; '
                     f'Leakage metrics N={len(joined)}; common-model SS N={len(common)}; '
                     f'common storms={common.storm_key.nunique()}; '
                     f'Leakage source N={len(l)}; max truth difference={err_truth}')
        if lead != 6:
            continue
        # 6h reference configurations, on their individual prediction cohorts.
        for ix, (model, file) in enumerate(ablations.items()):
            if not file.is_file():
                if args.allow_missing_ablations:
                    notes.append(f'WARNING {model} 6h file missing: {file}; table shows dashes')
                    print(f'  [WARNING] Missing 6h ablation {model}: {file}')
                    continue
                raise FileNotFoundError(f'{model}: expected {file}; '
                                        'provide --msl-only-file/--simple-nn-file')
            pcol = column_choice(file, ('pred_mslp_corr','pred_mslp_hpa'))
            tcol = column_choice(file, ('true_mslp_corr','true_mslp_hpa'))
            ab = read_predictions(file, KEY+[pcol,tcol])
            r = describe_errors((ab[pcol]-ab[tcol]).to_numpy(float))
            matched = ab.merge(c[KEY+['true_mslp_corr','persistence_pred']],
                               on=KEY, how='inner', validate='one_to_one', suffixes=('_ab','_control'))
            if not matched.empty:
                ab_truth = tcol+'_ab' if tcol=='true_mslp_corr' else tcol
                base_truth = 'true_mslp_corr_control' if tcol=='true_mslp_corr' else 'true_mslp_corr'
                delta = np.abs(matched[ab_truth].to_numpy(float)-matched[base_truth].to_numpy(float))
                max_delta = float(np.max(delta))
                if max_delta>TRUTH_TOL:
                    notes.append(f'WARNING {model} matched target disagreement max {max_delta} hPa; SS omitted')
                    raise ValueError(f'{model} matched Control target differs, max {max_delta:.8g} hPa')
                pred = pcol+'_ab' if pcol in ['true_mslp_corr','persistence_pred'] else pcol
                e = (matched[pred]-matched[ab_truth]).to_numpy(float)
                pe = (matched.persistence_pred-matched[base_truth]).to_numpy(float)
                r.update(ss_cluster_bootstrap(matched.storm_key.to_numpy(str), e, pe,
                                              args.bootstraps, args.seed+600+ix))
                if model=='MSL-only' and not args.include_msl_ss:
                    r['ss_cohort'] = '6h matched to control (AUDIT ONLY; original table has dash)'
                else:
                    r['ss_cohort'] = '6h ablation/control keys'
                notes.append(f'{model}: full ablation N={len(ab)}, matched Control SS N={len(matched)}, '
                             f'storms={matched.storm_key.nunique()}, max target difference {max_delta}')
            else:
                notes.append(f'WARNING {model}: no matching 6h keys to Control; SS unavailable')
                if model=='Simple NN':
                    raise ValueError('Simple NN SS cannot be reproduced: no (storm_key,step_index) overlap')
            r['metric_cohort'] = 'full_6h_ablation_source'
            r.setdefault('ss_cohort', 'no SS keys')
            results[(6,model)] = r
            print_summary(model,r)
            input_paths.append(dict(lead_h=6,Model=model,File=str(file),N=len(ab),
                                    N_matched=len(matched)))
            audit_rows.append(dict(lead_h=6,Model=model,Model_N=r['N'],
                    SS_N=r.get('SS_N',np.nan),SS_Storms=r.get('SS_Storms',np.nan),
                    MAE=r['MAE'],Bias=r['Bias'],Bias_CI_low=r['Bias_CI_low'],
                    Bias_CI_high=r['Bias_CI_high'],STD=r['STD'],RMSE=r['RMSE'],
                    Persistence_RMSE_on_SS_samples=r.get('SS_Persistence_RMSE',np.nan),
                    SS=r.get('SS',np.nan),SS_CI_low=r.get('SS_CI_low',np.nan),
                    SS_CI_high=r.get('SS_CI_high',np.nan),
                    metric_cohort=r['metric_cohort'],ss_cohort=r['ss_cohort'],
                    Header_N=n_header[6]))
            comparisons.extend(compare_previous(6,model,r,args.include_msl_ss))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'Supplementary_Combined_Full_Dataset_Metrics_RECALCULATED.tex').write_text(
        table_text(results,n_header,args.include_msl_ss),encoding='utf-8')
    pd.DataFrame(audit_rows).sort_values(['lead_h','Model']).to_csv(
        output_dir/'FullDataset_Metrics_Audit.csv',index=False,float_format='%.12g')
    pd.DataFrame(comparisons).sort_values(['lead_h','Model','Metric']).to_csv(
        output_dir/'FullDataset_Comparison_with_old_table.csv',index=False,float_format='%.12g')
    pd.DataFrame(input_paths).to_csv(output_dir/'FullDataset_Input_Files.csv',index=False)
    (output_dir/'FullDataset_Methods_and_Warnings.txt').write_text(
        'FULL DATASET SUPPLEMENTARY METRICS — audit notes\n'
        +f'Bootstrap replicates={args.bootstraps}; independent seed base={args.seed}\n'
        +'SS uses one same-step ERA5 center Persistence denominator per matched SS cohort.\n'
        +'Core SS bootstrap resamples storms; all forecast steps have equal point-estimate weight.\n'
        +'Bias CI uses normal approximation, matching earlier S2/S3 methodology.\n'
        +'Original reference table is used ONLY for a comparison audit, never for calculation.\n'
        + '\n'.join(notes)+'\n',encoding='utf-8')
    problems = [x for x in comparisons if x['Status'].startswith('CHECK') or x['Status']=='MISSING_NEW_RESULT']
    print(f'\nOUTPUT: {output_dir}')
    print('  - Supplementary_Combined_Full_Dataset_Metrics_RECALCULATED.tex')
    print('  - FullDataset_Metrics_Audit.csv')
    print('  - FullDataset_Comparison_with_old_table.csv')
    print('  - FullDataset_Input_Files.csv')
    print('  - FullDataset_Methods_and_Warnings.txt')
    print(f'Previous-table numeric discrepancies needing review: {len(problems)}')
    for item in problems[:25]:
        print(f"  CHECK: {item['lead_h']}h {item['Model']} {item['Metric']} "
              f"old={item['Old_value']} new={item['New_value']:.5f}")
    if args.fail_on_difference and problems:
        raise SystemExit(f'{len(problems)} mismatches relative to historical table; see audit CSV')


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-dir', type=Path,
                    default=Path('data/updated_data'))
    ap.add_argument('--control-dir', type=Path)
    ap.add_argument('--leakage-dir', type=Path)
    ap.add_argument('--msl-only-file', type=Path, help='Original 6h MSL-only prediction CSV')
    ap.add_argument('--simple-nn-file', type=Path, help='Original 6h Simple NN prediction CSV')
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--bootstraps', type=int, default=1000,
                    help='1000 matches original full-dataset SS bootstrap script')
    ap.add_argument('--seed', type=int, default=42,
                    help='Reproducible RNG for storm bootstrap (historical intervals may differ slightly)')
    ap.add_argument('--include-msl-ss', action='store_true',
                    help='Show newly computed MSL-only SS instead of historical dash')
    ap.add_argument('--allow-missing-ablations', action='store_true',
                    help='Permit absent 6h reference CSVs (generates dashes and warnings)')
    ap.add_argument('--fail-on-difference', action='store_true',
                    help='Exit nonzero if new metrics differ from historical rounded table')
    a = ap.parse_args()
    if a.bootstraps < 100:
        ap.error('--bootstraps must be at least 100')
    build(a)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
