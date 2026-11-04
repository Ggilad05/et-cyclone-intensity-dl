#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible Table 5 + AMS paper / supplementary figures (consistent cohorts).

Python >= 3.9; requires numpy, pandas. No dependency on xarray/GRIB.

Input per lead (6/12/18/24 h):
  control/control_31.csv ... control_34.csv
  leakage/leakage_6d.csv ... leakage_24d.csv
  IFS/true_vs_pred_6h.csv ... true_vs_pred_24h.csv
  source_all_valid_times.csv (the COMPLETE ERA5 local-minimum time series)

Table 5 and 2010 skill use ONE strict three-way intersection of exact
(storm_key, step_index) pairs across Control, Leakage, and IFS.
Figure 5b: pair-specific intersections: only pairs that include IFS use
2010 cases; all other pairs use all available years.
Figure 5c: pair-specific IFS/Control or IFS/Persistence matches, not the
three-way IFS/Control/Leakage intersection.
Figure 5a intentionally reproduces the ORIGINAL mixed-cohort presentation:
IFS alone uses 2010; all other models and 6h ablations use ALL test years.
No paired sample comparison may be inferred from that mixed-cohort overview.
No index shifting, row-based alignment, or pressure-value matching is done.

Targets / physically consistent persistence baselines:
  CNN/Linear/Leakage: tracked-center ERA5 MSL at target step;
    persistence = Control's center MSL at model initialization.
  IFS: ERA5 local MSL minimum near cyclone at target step;
    persistence = ERA5 local MSL minimum at initialization step,
    obtained from the all-valid-times history CSV using step_index - lead/6.

Because spatial verification targets differ, SS values for IFS and CNN are
NOT strictly on an identical target, even though case IDs are matched.
See README provided with the bundle.

Outputs:
  Main_Paper/Figure5a_MAE_original_cohorts.pdf  original mixed cohorts
  Supplementary/Figure5a_2010_strict_MAE.pdf  optional matched check (no t-1 ablations)
  Table5_2010_consistent.tex    AMS-style LaTeX table
  Table5_2010_metrics.csv       point estimates and cyclone-bootstrap CIs
  Table5_2010_join_audit.csv    counts at each stage
  Table5_2010_diagnostics.csv   target differences and denominator tests
  matched_6h.csv ...          all fields and errors used by the table
  unmatched_*.csv             anti-join diagnostics
  table5_run_info.txt          inputs and settings

Example:
 python build_table5_consistent.py \
   --base-dir ~/Desktop/steps_analysis/updated_data \
   --ifs-dir ~/Desktop/forecast/Fixed_2010 \
   --ifs-history ./source_all_valid_times.csv \
   --out-dir ~/Desktop/steps_analysis/updated_data/Plots/Table5_consistent
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

LEADS = (6, 12, 18, 24)
CONTROL_NAMES = {6: 'control_31.csv', 12: 'control_32.csv',
                 18: 'control_33.csv', 24: 'control_34.csv'}
LEAKAGE_NAMES = {6: 'leakage_6d.csv', 12: 'leakage_12d.csv',
                 18: 'leakage_18d.csv', 24: 'leakage_24d.csv'}
MODELS = ('IFS 2010', 'Leakage', 'Control', 'Linear', 'Persistence')
KEY = ['storm_key', 'step_index']
NUMERIC = ['ifs_pred', 'ifs_truth', 'local_persistence', 'center_truth',
           'center_persistence', 'control_pred', 'leakage_pred', 'linear_pred']
CHECK_TOL = 1e-5  # hPa, for verifying actual ERA5 truths/persistence


def read_subset(path: Path, required: List[str], optional=()):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required file: {path}')
    hdr = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    absent = [c for c in required if c not in hdr]
    if absent:
        raise ValueError(f'{path} missing required columns: {absent}. Columns: {hdr}')
    cols = list(dict.fromkeys(required + [c for c in optional if c in hdr]))
    # Avoid loading other potentially very large CSV columns.
    d = pd.read_csv(path, usecols=cols, dtype={'storm_key': 'string'})
    d.columns = d.columns.str.strip()
    if d.empty:
        raise ValueError(f'Empty input file: {path}')
    return d


def normalize_keys(d: pd.DataFrame, source: str, require_2010=False):
    d = d.copy()
    if any(k not in d for k in KEY):
        raise ValueError(f'{source}: keys {KEY} are required')
    d['storm_key'] = (d['storm_key'].astype('string').str.strip()
                      .str.strip('\\\"\'')
                      .str.replace(r'(?i)\.npy$', '', regex=True).str.strip())
    if d['storm_key'].isna().any() or d['storm_key'].eq('').any():
        raise ValueError(f'{source}: blank storm keys')
    step = pd.to_numeric(d['step_index'], errors='coerce')
    if step.isna().any() or not np.isfinite(step.to_numpy(dtype=float)).all():
        raise ValueError(f'{source}: invalid step indices')
    if not np.equal(np.floor(step), step).all():
        raise ValueError(f'{source}: non-integer step_index')
    d['step_index'] = step.astype('int64')
    if (d.step_index < 0).any():
        raise ValueError(f'{source}: negative step_index')
    if require_2010:
        valid = d.storm_key.str.contains(r'(?<!\d)2010(?!\d)', regex=True).fillna(False)
        d = d.loc[valid].copy()
        if d.empty:
            raise ValueError(f'{source}: no 2010 storm tracks found')
    duplicate = d.duplicated(KEY, keep=False)
    if duplicate.any():
        ex = d.loc[duplicate, KEY].head(6).to_dict('records')
        raise ValueError(f'{source}: duplicate (storm_key, step_index) keys: {ex}')
    return d


def finite_or_raise(d, names, label):
    for name in names:
        d[name] = pd.to_numeric(d[name], errors='coerce')
        if not np.isfinite(d[name].to_numpy(dtype=float)).all():
            raise ValueError(f'{label}: non-finite values in {name}. '
                             'No silent row removal is allowed.')
        if (d[name].lt(800) | d[name].gt(1100)).any():
            # Typical meteorological center/local minimum pressure guard.
            print(f'  WARNING [{label}]: some {name} values outside 800..1100 hPa.')
    return d


def load_control(path):
    necessary = KEY + ['true_mslp_corr', 'pred_mslp_corr', 'persistence_pred']
    d = read_subset(path, necessary, optional=('linear_pred', 'lin_pred'))
    if 'linear_pred' not in d and 'lin_pred' not in d:
        raise ValueError(f'{path}: missing linear_pred / lin_pred')
    linear = 'linear_pred' if 'linear_pred' in d else 'lin_pred'
    d = normalize_keys(d, str(path), require_2010=True)
    d = d.rename(columns={'true_mslp_corr':'center_truth',
                          'pred_mslp_corr':'control_pred',
                          'persistence_pred':'center_persistence',
                          linear:'linear_pred'})
    return finite_or_raise(d[KEY + ['center_truth', 'control_pred',
                                    'center_persistence', 'linear_pred']].copy(),
                           ['center_truth', 'control_pred', 'center_persistence',
                            'linear_pred'], str(path))


def load_leakage(path):
    hdr = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    candidates = ['pred_mslp_corr', 'dl_pred', 'pred']
    pred = next((c for c in candidates if c in hdr), None)
    if pred is None:
        raise ValueError(f'{path}: missing Leakage prediction ({candidates})')
    d = read_subset(path, KEY + [pred], optional=('true_mslp_corr',))
    d = normalize_keys(d, str(path), require_2010=True)
    rename = {pred:'leakage_pred'}
    if 'true_mslp_corr' in d:
        rename['true_mslp_corr'] = 'leakage_truth_check'
    d = d.rename(columns=rename)
    finite = ['leakage_pred'] + (['leakage_truth_check'] if 'leakage_truth_check' in d else [])
    return finite_or_raise(d, finite, str(path))


def load_ifs(path, lead):
    cols = KEY + ['true_mslp_ifs', 'pred_mslp_ifs']
    d = normalize_keys(read_subset(path, cols, optional=('horizon_h',)),
                       str(path), require_2010=True)
    if 'horizon_h' in d:
        declared = pd.to_numeric(d.horizon_h, errors='coerce')
        if declared.isna().any() or not declared.eq(lead).all():
            raise ValueError(f'{path}: horizon_h inconsistent with {lead}h')
    d = d.rename(columns={'true_mslp_ifs':'ifs_truth',
                          'pred_mslp_ifs':'ifs_pred'})
    return finite_or_raise(d[KEY + ['ifs_truth','ifs_pred']].copy(),
                           ['ifs_truth','ifs_pred'], str(path))


def load_history(path):
    columns = KEY + ['true_mslp_ifs_from_npy', 'era5_center_hpa']
    d = normalize_keys(read_subset(path, columns), str(path), require_2010=True)
    d = d.rename(columns={'true_mslp_ifs_from_npy':'local_history_truth',
                          'era5_center_hpa':'center_history_truth'})
    return finite_or_raise(d, ['local_history_truth', 'center_history_truth'], str(path))


def anti_join(a, b):
    return a[KEY].merge(b[KEY], on=KEY, how='left', indicator=True,
                        validate='one_to_one').query('_merge == "left_only"')[KEY].copy()


def merge_for_lead(lead, control_dir, leakage_dir, ifs_dir, history, out, audit):
    control = load_control(control_dir / CONTROL_NAMES[lead])
    leakage = load_leakage(leakage_dir / LEAKAGE_NAMES[lead])
    ifs = load_ifs(ifs_dir / f'true_vs_pred_{lead}h.csv', lead)
    audit.append(dict(lead_h=lead, stage='control_2010', n=len(control)))
    audit.append(dict(lead_h=lead, stage='leakage_2010', n=len(leakage)))
    audit.append(dict(lead_h=lead, stage='ifs_2010', n=len(ifs)))
    # Control is the anchor. Never join on pressure or adjust step indices.
    anti_join(control, ifs).to_csv(out / f'unmatched_{lead}h_control_vs_ifs.csv', index=False)
    anti_join(ifs, control).to_csv(out / f'unmatched_{lead}h_ifs_vs_control.csv', index=False)
    joined = control.merge(ifs, on=KEY, how='inner', validate='one_to_one')
    audit.append(dict(lead_h=lead, stage='control_intersect_ifs', n=len(joined)))
    anti_join(joined, leakage).to_csv(out / f'unmatched_{lead}h_control_ifs_vs_leakage.csv', index=False)
    joined = joined.merge(leakage, on=KEY, how='inner', validate='one_to_one')
    audit.append(dict(lead_h=lead, stage='strict_three_way_intersection', n=len(joined)))
    if joined.empty:
        raise ValueError(f'{lead}h: zero common samples')

    if 'leakage_truth_check' in joined:
        delta = np.abs(joined['leakage_truth_check'] - joined['center_truth'])
        if (delta > CHECK_TOL).any():
            ix = delta.idxmax()
            raise ValueError(f'{lead}h: Leakage and Control verifying truths differ on same key! '
                             f'Max diff={delta.max():.6g} hPa, row={joined.loc[ix,KEY].to_dict()}')

    # Verify that the IFS ERA5 local-minimum truth matches the raw, all-time source.
    verify = history[KEY + ['local_history_truth','center_history_truth']].rename(
        columns={'local_history_truth':'source_target_local',
                 'center_history_truth':'source_target_center'})
    joined = joined.merge(verify, on=KEY, how='left', validate='one_to_one')
    if joined[['source_target_local', 'source_target_center']].isna().any().any():
        raise ValueError(f'{lead}h: missing target-step ERA5 record in history')
    diff_local = np.abs(joined.ifs_truth - joined.source_target_local)
    diff_center = np.abs(joined.center_truth - joined.source_target_center)
    if (diff_local > CHECK_TOL).any() or (diff_center > CHECK_TOL).any():
        raise ValueError(f'{lead}h: mismatch to ERA5 history at target. '
                         f'Max local={diff_local.max()}, center={diff_center.max()}. '
                         'Verify input provenance/step_index before proceeding.')

    # IMPORTANT: initial-state ERA5 local minimum must be evaluated at t0,
    # not borrowed from Control's different (center-pressure) target.
    joined['t0_step_index'] = joined.step_index - lead // 6
    if (joined.t0_step_index < 0).any():
        raise ValueError(f'{lead}h: negative initialization index')
    initial = history[KEY + ['local_history_truth','center_history_truth']].rename(columns={
        'step_index':'t0_step_index',
        'local_history_truth':'local_persistence',
        'center_history_truth':'center_source_persistence'})
    joined = joined.merge(initial, on=['storm_key','t0_step_index'],
                          how='left', validate='many_to_one')
    if joined[['local_persistence','center_source_persistence']].isna().any().any():
        rows = joined.loc[joined.local_persistence.isna(), KEY + ['t0_step_index']].head()
        raise ValueError(f'{lead}h: missing t0 ERA5 pressure from local history.\n{rows}')
    # Independent check that Control persistence really refers to the t0 step.
    delta_t0 = np.abs(joined.center_source_persistence - joined.center_persistence)
    if (delta_t0 > CHECK_TOL).any():
        ex = joined.loc[delta_t0.idxmax(),KEY].to_dict()
        raise ValueError(f'{lead}h: Control persistence predictor NOT equal to ERA5 '
                         f'center at t0 (max diff {delta_t0.max():.6g} hPa, {ex}). '
                         'May indicate wrong horizons/index alignment.')
    joined = joined.sort_values(KEY).reset_index(drop=True)
    for model, pred, truth in (
        ('IFS_2010','ifs_pred','ifs_truth'),
        ('Leakage','leakage_pred','center_truth'),
        ('Control','control_pred','center_truth'),
        ('Linear','linear_pred','center_truth'),
        ('Persistence','center_persistence','center_truth'),
    ):
        joined[f'error_{model}'] = joined[pred] - joined[truth]
    joined['persistence_error_IFS_2010'] = joined.local_persistence - joined.ifs_truth
    joined['persistence_error_center'] = joined.center_persistence - joined.center_truth
    joined.to_csv(out / f'matched_{lead}h.csv', index=False, float_format='%.12g')
    diagnostics = dict(lead_h=lead, matched_n=len(joined), storms=joined.storm_key.nunique(),
        max_local_target_diff_hpa=float(diff_local.max()),
        max_center_target_diff_hpa=float(diff_center.max()),
        max_center_t0_diff_hpa=float(delta_t0.max()),
        mean_abs_truth_difference_hpa=float(np.abs(joined.ifs_truth-joined.center_truth).mean()),
        ifs_local_persistence_rmse_hpa=float(np.sqrt(np.mean(joined.persistence_error_IFS_2010**2))),
        center_persistence_rmse_hpa=float(np.sqrt(np.mean(joined.persistence_error_center**2))))
    return joined, diagnostics


def compute_metrics(df, lead, n_boot, seed):
    n = len(df)
    err = np.column_stack([df[f'error_{m.replace(" ","_")}'].to_numpy(float) for m in MODELS])
    base_local = df.persistence_error_IFS_2010.to_numpy(float)
    base_center = df.persistence_error_center.to_numpy(float)
    bases = np.column_stack([base_local, base_center, base_center, base_center, base_center])
    point_mae = np.mean(np.abs(err), axis=0)
    point_bias = np.mean(err, axis=0)
    point_std = np.std(err, ddof=1, axis=0)
    point_rmse = np.sqrt(np.mean(err**2, axis=0))
    point_basermse = np.sqrt(np.mean(bases**2, axis=0))
    if np.any(point_basermse < 1e-10):
        raise ValueError(f'{lead}h: zero persistence RMSE denominator')
    point_ss = 1 - point_rmse/point_basermse
    if not np.allclose(point_rmse**2, point_bias**2 + point_std**2*(n-1)/n,
                       atol=1e-10, rtol=1e-10):
        raise AssertionError(f'{lead}h: RMSE/Bias/STD algebra failed')
    if not np.allclose(err[:,-1], bases[:,-1]):
        raise AssertionError(f'{lead}h: Persistence must have SS = 0')

    # Cyclone cluster bootstrap: resample entire tracks, but weight EACH
    # forecast step equally within the concatenated resampled tracks.
    # Same set of selected tracks is used for every model in each replicate.
    _, index = np.unique(df.storm_key.to_numpy(str), return_inverse=True)
    k = index.max() + 1
    count = np.bincount(index, minlength=k)
    sums_e = np.stack([np.bincount(index, weights=err[:,j], minlength=k)
                       for j in range(len(MODELS))], axis=1)
    sums_abs = np.stack([np.bincount(index, weights=np.abs(err[:,j]), minlength=k)
                       for j in range(len(MODELS))], axis=1)
    sums_sq = np.stack([np.bincount(index, weights=err[:,j]**2, minlength=k)
                       for j in range(len(MODELS))], axis=1)
    sums_bas = np.stack([np.bincount(index, weights=bases[:,j]**2, minlength=k)
                        for j in range(len(MODELS))], axis=1)
    rng = np.random.default_rng(seed + lead)
    boot_bias = np.empty((n_boot, len(MODELS)))
    boot_mae = np.empty_like(boot_bias)
    boot_ss = np.empty_like(boot_bias)
    for b in range(n_boot):
        drawn = rng.integers(0,k,size=k)
        denom = count[drawn].sum()
        boot_bias[b] = sums_e[drawn].sum(axis=0)/denom
        boot_mae[b] = sums_abs[drawn].sum(axis=0)/denom
        brmse = np.sqrt(sums_sq[drawn].sum(axis=0)/denom)
        bbase = np.sqrt(sums_bas[drawn].sum(axis=0)/denom)
        if (bbase <= 0).any():
            raise ValueError(f'{lead}h: invalid bootstrap persistence denominator')
        boot_ss[b] = 1-brmse/bbase
    records=[]
    for j, model in enumerate(MODELS):
        rec={'Lead_h':lead,'Model':model,'N':n,'Storms':k,
             'MAE':point_mae[j], 'Bias':point_bias[j],
             'STD':point_std[j], 'RMSE':point_rmse[j],
             'Persistence_RMSE':point_basermse[j], 'SS':point_ss[j],
             'Verification_target':'ERA5 local minimum' if j==0 else 'ERA5 tracked center',
             'Persistence_reference':'ERA5 local min at t0' if j==0 else 'ERA5 center at t0'}
        for metric, boot in [('MAE',boot_mae),('Bias',boot_bias),('SS',boot_ss)]:
            rec[f'{metric}_CI_low'],rec[f'{metric}_CI_high'] = np.percentile(boot[:,j],[2.5,97.5])
        records.append(rec)
    return records


def make_latex(stats):
    lookup={(int(row.Lead_h),row.Model):row for row in stats.itertuples(index=False)}
    counts={lead:int(lookup[lead,'IFS 2010'].N) for lead in LEADS}
    def cell(lead, model, metric):
        r=lookup[lead,model]
        if metric in ('Bias','SS'):
            return (f'${getattr(r,metric):.3f} '
                    f'({getattr(r,metric+"_CI_low"):.3f}, '
                    f'{getattr(r,metric+"_CI_high"):.3f})$')
        return f'${getattr(r,metric):.3f}$'
    lines=[
      r'% GENERATED by build_table5_consistent.py — do not manually edit statistics.',
      r'% Strict 3-way case intersection; t0-local ERA5 min for IFS persistence.',
      r'\begin{table}[t]',
      r'\caption{Performance metrics for IFS, Leakage, Control, Linear Regression, and',
      r'Persistence for the matched 2010 test set. At each lead time, all models',
      r'are evaluated on identical cyclone identifiers and forecast-step indices.',
      r'The IFS forecast is verified against the ERA5 local minimum MSL near the',
      r'cyclone, whereas the remaining models are verified against the ERA5 MSL',
      r'at the tracked cyclone center. MAE, Bias, and the standard deviation',
      r'(STD) of signed errors are expressed in hPa. Skill Score (SS) is defined',
      r'as $1-\mathrm{RMSE}_{\mathrm{model}}/\mathrm{RMSE}_{\mathrm{persistence}}$.',
      r'The persistence reference is the ERA5 local minimum at initialization',
      r'for IFS and the ERA5 center pressure at initialization for other models.',
      r'Therefore, SS values compare improvement relative to target-specific',
      r'persistence and are not strictly on an identical spatial verification',
      r'target across IFS and CNNs. Parentheses give 95\% confidence intervals',
      r'from cyclone-cluster bootstrap resampling for Bias and SS.}',
      r'\label{tab:metrics_summary}',r'\centering',r'\resizebox{\textwidth}{!}{%',
      r'\begin{tabular}{llcccc}',r'\topline',
      'Measure & Model & '+' & '.join(f'{h} h ($N={counts[h]:,}$)' for h in LEADS)+r' \\',
      r'\midline']
    labels={'MAE':'MAE (hPa)','Bias':'Bias (hPa)','STD':'STD (hPa)','SS':'SS'}
    for metric in ['MAE','Bias','STD','SS']:
        for j,model in enumerate(MODELS):
            lines.append((labels[metric] if j==0 else '')+' & '+model+' & '+
                         ' & '.join(cell(lead,model,metric) for lead in LEADS)+r' \\')
        if metric!='SS':lines.append(r'\midline')
    lines.extend([r'\botline',r'\end{tabular}%',r'}',r'\end{table}'])
    return '\n'.join(lines)+'\n'




# ---- Unified figure/diagnostics layer (same strictly joined observations) ----

COLORS = {'IFS 2010':'#7b3294', 'Leakage':'#e39b1b', 'Control':'#265cb5',
          'Linear':'#278153', 'Persistence':'#bb3c40',
          'MSL-only':'#1c1c1c', 'Simple NN':'#8c564b'}
FIG_MODEL_ORDER = ('IFS 2010','Leakage','Control','Linear','Persistence')
# The 6h KS heatmap may also include up to four ablation models.
# Each optional model needs the same canonical signed-error column as the
# core five models. Previously these names were missing here, causing a
# KeyError: 'MSL-only' when an optional prediction file existed.
OPTIONAL_6H_MODELS = ('MSL-only', 'Simple NN', 'MSL-only (t-1)', 'Control (t-1)')
ERROR_COLUMNS = {name: 'error_' + name.replace(' ', '_')
                 for name in (*FIG_MODEL_ORDER, *OPTIONAL_6H_MODELS)}


def save_figure(fig, folder, name):
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder / (name + '.pdf'), bbox_inches='tight', pad_inches=0.12)
    fig.savefig(folder / (name + '.png'), dpi=300, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)




def _paired_ks(abs_a,abs_b):
    # Descriptive two-sample KS D on matched pairs, without an independence p-value.
    a=np.sort(np.asarray(abs_a,float)); b=np.sort(np.asarray(abs_b,float))
    all_values=np.unique(np.concatenate((a,b)))
    d=np.max(np.abs(np.searchsorted(a,all_values,side='right')/len(a) -
                    np.searchsorted(b,all_values,side='right')/len(b)))
    win=100*np.mean(np.asarray(abs_a) < np.asarray(abs_b))
    tie=100*np.mean(np.asarray(abs_a) == np.asarray(abs_b))
    return float(d),float(win),float(tie)


def pair_stats(df, lead, pairs=FIG_MODEL_ORDER):
    rows=[]
    for first in pairs:
        for second in pairs:
            if first==second:continue
            a=np.abs(df[ERROR_COLUMNS[first]].to_numpy(float))
            b=np.abs(df[ERROR_COLUMNS[second]].to_numpy(float))
            ks,win,tie=_paired_ks(a,b)
            rows.append({'Lead_h':lead,'First':first,'Second':second,
                         'N':len(df),'KS_D':ks,'Win_pct':win,'Ties_pct':tie})
    return rows








def _bin_moments(df, x, error_names, boot, seed, min_count, n_boundaries=25):
    """Original saturation bins, with cluster-bootstrap CIs.

    The user's supplied source (Pasted markdown (5).md) uses exactly:
        max_x = np.percentile(x, 99.5)
        shared_bins = np.linspace(0, max_x, 25)
        np.digitize(x, shared_bins)
    and plots only bins with >=20 steps. We preserve that bin geometry
    instead of V3's ad-hoc unequal-width bins and hard-coded x cutoff.
    For CI ONLY, retain cyclone-cluster bootstrap rather than the source's
    non-cluster normal approximation.
    """
    x=np.asarray(x,dtype=float)
    if len(x)==0 or not np.isfinite(x).all() or np.any(x<0):
        raise ValueError('Invalid intensity change values')
    bound=np.percentile(x,99.5)
    if bound<=0: raise ValueError('Nonpositive 99.5th percentile intensity change')
    edges=np.linspace(0,float(bound),n_boundaries)
    bs=len(edges)-1
    inds=np.digitize(x,edges)-1
    # Precisely like the original loop over bins 1..len(shared_bins)-1:
    # values >= upper edge are excluded from displayed binned curves.
    valid_sample=(inds>=0)&(inds<bs)
    group=df.storm_key.to_numpy(str)
    err=np.column_stack([np.abs(df[ERROR_COLUMNS[m]].to_numpy(float)) for m in error_names])
    ids=inds[valid_sample]
    e=err[valid_sample]
    counts=np.bincount(ids,minlength=bs)
    sums=np.stack([np.bincount(ids,weights=e[:,j],minlength=bs)
                   for j in range(e.shape[1])],axis=1)
    means=np.divide(sums,counts[:,None],out=np.full_like(sums,np.nan),where=counts[:,None]>0)
    _,gid=np.unique(group,return_inverse=True)
    k=gid.max()+1
    offsets=gid[valid_sample]*bs+ids
    cluster_count=np.bincount(offsets,minlength=k*bs).reshape(k,bs)
    cluster_sums=np.stack([np.bincount(offsets,weights=e[:,j],minlength=k*bs).reshape(k,bs)
                           for j in range(e.shape[1])],axis=2)
    rng=np.random.default_rng(seed)
    curves=np.full((boot,bs,len(error_names)),np.nan)
    for i in range(boot):
        drawn=rng.integers(0,k,size=k)
        n=cluster_count[drawn].sum(axis=0)
        summed=cluster_sums[drawn].sum(axis=0)
        curves[i]=np.divide(summed,n[:,None],out=np.full((bs,len(error_names)),np.nan),where=n[:,None]>0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',category=RuntimeWarning)
        lo=np.nanpercentile(curves,2.5,axis=0)
        hi=np.nanpercentile(curves,97.5,axis=0)
    centers=(edges[:-1]+edges[1:])/2
    return centers,counts,means,lo,hi,(counts>=min_count),edges


def add_simple_nn_errors(base: pd.DataFrame, simple_file: Path, purpose: str):
    """Join the original Simple NN predictions on exact step keys.

    Inner join reduces *every* model in the requested saturation figure to
    the same observations. Table 5 is not touched by this function.
    """
    if simple_file is None:
        print(f'  [SIMPLE NN MISSING] {purpose}: no source file detected. '
              'Pass --simple-nn-file PATH. Chart is not silently labeled as including Simple NN.')
        return None
    if not simple_file.is_file():
        raise FileNotFoundError(f'{purpose}: Simple NN file not found: {simple_file}')
    header=pd.read_csv(simple_file,nrows=0).columns.str.strip().tolist()
    pred=next((c for c in ('pred_mslp_corr','pred_mslp_hpa') if c in header),None)
    tru=next((c for c in ('true_mslp_corr','true_mslp_hpa') if c in header),None)
    if pred is None or tru is None:
        raise ValueError(f'Simple NN source missing truth/pred columns: {header}')
    nn=normalize_keys(read_subset(simple_file,KEY+[pred,tru]),str(simple_file))
    nn=nn.rename(columns={pred:'_simple_prediction',tru:'_simple_truth'})
    # The base target can be called center_truth (strict) or center_truth_full.
    truth_col='center_truth' if 'center_truth' in base else 'center_truth_full'
    n_before=len(base)
    common=base.merge(nn,on=KEY,how='inner',validate='one_to_one')
    if common.empty:
        raise ValueError(f'{purpose}: Simple NN has zero common samples with base; '
                         'verify storm_key and step_index (not pressure)')
    diff=np.abs(common['_simple_truth'].to_numpy(float)-common[truth_col].to_numpy(float))
    if not np.isfinite(diff).all() or np.max(diff)>CHECK_TOL:
        raise ValueError(f'{purpose}: Simple NN target mismatch. max={np.max(diff):.6g} hPa')
    common[ERROR_COLUMNS['Simple NN']]=common['_simple_prediction']-common[truth_col]
    common=common.drop(columns=['_simple_prediction','_simple_truth'])
    print(f'  [SIMPLE NN] {purpose}: source={simple_file}; matched {len(common):,}/{n_before:,} '
          f'steps, {common.storm_key.nunique():,} storms; plotted with all models on the same steps.')
    return common


def load_full_lead(lead, control_dir, leakage_dir):
    # Full-year results are deliberately separate from strict 2010 cohort.
    cp=control_dir/CONTROL_NAMES[lead]
    lp=leakage_dir/LEAKAGE_NAMES[lead]
    control=normalize_keys(read_subset(cp,KEY+['true_mslp_corr','pred_mslp_corr','persistence_pred'],
                          optional=('linear_pred','lin_pred')),str(cp))
    linear='linear_pred' if 'linear_pred' in control else 'lin_pred'
    if linear not in control:raise ValueError(f'{cp}: missing linear pred')
    lc=load_leakage_all_years(lp)
    for col in ['true_mslp_corr','pred_mslp_corr','persistence_pred',linear]:
        control[col]=pd.to_numeric(control[col],errors='raise')
    if not np.isfinite(control[['true_mslp_corr','pred_mslp_corr','persistence_pred',linear]].to_numpy(float)).all():
        raise ValueError(f'{cp}: invalid numeric values')
    c=control
    c['error_Control']=c.pred_mslp_corr-c.true_mslp_corr
    c['error_Linear']=c[linear]-c.true_mslp_corr
    c['error_Persistence']=c.persistence_pred-c.true_mslp_corr
    c=c.rename(columns={'true_mslp_corr':'center_truth_full'})
    joined=c.merge(lc,on=KEY,how='inner',validate='one_to_one')
    if 'leakage_truth_check' in joined:
        diff=np.abs(joined.leakage_truth_check-joined.center_truth_full)
        if diff.gt(CHECK_TOL).any():raise ValueError(f'{lead}h: full-data Leakage truth mismatches Control by {diff.max()}')
    joined['error_Leakage']=joined.leakage_pred-joined.center_truth_full
    joined['persistence_error_center']=joined['error_Persistence']
    metrics=[]
    for name,col in [('Control','error_Control'),('Linear','error_Linear'),('Persistence','error_Persistence')]:
        er=c[col].to_numpy(float)
        metrics.append({'Lead_h':lead,'Model':name,'N':len(c),
            'MAE':np.mean(np.abs(er)),'Bias':np.mean(er),'STD':np.std(er,ddof=1)})
    er=joined.error_Leakage.to_numpy(float)
    metrics.append({'Lead_h':lead,'Model':'Leakage','N':len(joined),
        'MAE':np.mean(np.abs(er)),'Bias':np.mean(er),'STD':np.std(er,ddof=1)})
    if joined.empty:raise ValueError(f'{lead}h: zero shared Control and Leakage rows')
    return metrics,joined


def load_leakage_all_years(path):
    h=pd.read_csv(path,nrows=0).columns.str.strip().tolist()
    pred=next((c for c in ('pred_mslp_corr','dl_pred','pred') if c in h),None)
    if pred is None:raise ValueError(f'{path}: missing Leakage prediction')
    d=normalize_keys(read_subset(path,KEY+[pred],optional=('true_mslp_corr',)),str(path))
    d=d.rename(columns={pred:'leakage_pred', 'true_mslp_corr':'leakage_truth_check'})
    for col in ['leakage_pred']+(['leakage_truth_check'] if 'leakage_truth_check' in d else []):
        d[col]=pd.to_numeric(d[col],errors='raise')
        if not np.isfinite(d[col].to_numpy(float)).all():raise ValueError(f'{path}: nonfinite {col}')
    return d




def _full_pairs(lead, d):
    rows=[]
    models=['Leakage','Control','Linear','Persistence']
    for a in models:
        for b in models:
            if a==b:continue
            D,win,tie=_paired_ks(np.abs(d[ERROR_COLUMNS[a]].to_numpy(float)),
                                 np.abs(d[ERROR_COLUMNS[b]].to_numpy(float)))
            rows.append({'Lead_h':lead,'First':a,'Second':b,'N':len(d),
                        'KS_D':D,'Win_pct':win,'Ties_pct':tie})
    return rows


def load_pairwise_error_frames(lead, control_dir, leakage_dir, ifs_dir,
                               optional_6h=None):
    """Independent prediction samples for each model: no implicit 3-way filter.

    IFS is restricted to the supplied 2010 IFS CSV. Each other model includes
    every available forecast step across all years. The exact pairwise
    intersection is performed ONLY when two models are compared.
    """
    cp = control_dir / CONTROL_NAMES[lead]
    c = normalize_keys(read_subset(cp, KEY +
        ['true_mslp_corr', 'pred_mslp_corr', 'persistence_pred'],
        optional=('linear_pred', 'lin_pred')), str(cp))
    lcol = 'linear_pred' if 'linear_pred' in c else 'lin_pred'
    if lcol not in c:
        raise ValueError(f'{cp}: missing linear prediction')
    truth = pd.to_numeric(c.true_mslp_corr, errors='raise').to_numpy(float)
    frames = {}
    for model, pred in [('Control', 'pred_mslp_corr'),
                        ('Persistence', 'persistence_pred'), ('Linear', lcol)]:
        diff = pd.to_numeric(c[pred], errors='raise').to_numpy(float) - truth
        if not np.isfinite(diff).all():
            raise ValueError(f'{cp}: nonfinite {model} errors')
        frames[model] = c[KEY].assign(abs_err=np.abs(diff))

    lp = leakage_dir / LEAKAGE_NAMES[lead]
    leakage = load_leakage_all_years(lp)
    if 'leakage_truth_check' in leakage:
        truth_l = pd.to_numeric(leakage.leakage_truth_check,
                                errors='raise').to_numpy(float)
        diff_l = leakage.leakage_pred.to_numpy(float) - truth_l
        frames['Leakage'] = leakage[KEY].assign(abs_err=np.abs(diff_l))
    else:
        pair = leakage[KEY+['leakage_pred']].merge(
            c[KEY+['true_mslp_corr']], on=KEY, how='inner',
            validate='one_to_one')
        if pair.empty:raise ValueError(f'{lead}h: no Leakage/Control shared rows')
        frames['Leakage'] = pair[KEY].assign(
            abs_err=np.abs(pair.leakage_pred - pair.true_mslp_corr))

    ifs = load_ifs(ifs_dir / f'true_vs_pred_{lead}h.csv', lead)
    frames['IFS 2010'] = ifs[KEY].assign(
        abs_err=np.abs(ifs.ifs_pred.to_numpy(float)-ifs.ifs_truth.to_numpy(float)))

    if lead == 6 and optional_6h:
        for model, path in optional_6h.items():
            if path is None:continue
            if not Path(path).is_file():raise FileNotFoundError(f'{model}: {path}')
            header = pd.read_csv(path,nrows=0).columns.str.strip().tolist()
            pc = next((k for k in ('pred_mslp_corr','pred_mslp_hpa') if k in header),None)
            tc = next((k for k in ('true_mslp_corr','true_mslp_hpa') if k in header),None)
            if pc is None or tc is None:raise ValueError(f'{model}: missing prediction/verification')
            ab = normalize_keys(read_subset(Path(path),KEY+[pc,tc]), str(path))
            err = pd.to_numeric(ab[pc],errors='raise').to_numpy(float) - \
                  pd.to_numeric(ab[tc],errors='raise').to_numpy(float)
            if not np.isfinite(err).all():raise ValueError(f'{model}: nonfinite errors')
            frames[model] = ab[KEY].assign(abs_err=np.abs(err))
            # A valid path and CSV schema do NOT guarantee overlapping cases.
            # Diagnose the overlap without silently shifting step indices.
            overlap_ctrl = len(frames[model][KEY].merge(frames['Control'][KEY],on=KEY,how='inner',validate='one_to_one'))
            overlap_ifs = len(frames[model][KEY].merge(frames['IFS 2010'][KEY],on=KEY,how='inner',validate='one_to_one'))
            ids = ab.storm_key.astype(str)
            print(f'  [6h ABLATION] {model}: {Path(path).resolve()} | rows={len(ab):,} | '
                  f'storms={ids.nunique():,} | Control pair N={overlap_ctrl:,} | IFS 2010 pair N={overlap_ifs:,}',flush=True)
            if not overlap_ifs:
                print(f'  [NO MATCH] {model} has no (storm_key,step_index) common with IFS 2010. '
                      'This does not establish a wrong path; inspect the storm-year coverage. '
                      f'Example file keys: {ab[KEY].head(3).to_dict("records")}',flush=True)
    return frames


def mixed_pairwise_ks(frames, lead):
    """KS and win rates on pair-specific cases, with original KS p-value stars.

    Statistical significance is the conventional 2-sample KS approximation,
    which does NOT adjust for temporal dependence of forecasts in each storm.
    """
    from scipy.stats import ks_2samp
    core_names=['Persistence','Linear','Control','Leakage','MSL-only','Simple NN',
                'MSL-only (t-1)','Control (t-1)']
    extras=sorted(n for n in frames if n not in core_names and n!='IFS 2010')
    names=[n for n in core_names if n in frames]+extras
    if 'IFS 2010' in frames: names.append('IFS 2010')
    rows=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            merged = frames[a].merge(frames[b],on=KEY,how='inner',
                                     suffixes=('_a','_b'),validate='one_to_one')
            if merged.empty:
                # No statistical comparison exists. Record N=0 explicitly;
                # never substitute another population or manufacture a KS value.
                scope='2010 pairwise matched' if 'IFS 2010' in (a,b) else 'All years pairwise matched'
                print(f'  [KS N/A] {lead}h {a} vs {b}: no shared (storm_key, step_index). '
                      'Reporting N/A, not stopping the whole paper build.', flush=True)
                for first,second in ((a,b),(b,a)):
                    rows.append({'Lead_h':lead,'First':first,'Second':second,
                                 'N':0,'KS_D':np.nan,'KS_p':np.nan,
                                 'Significant':False,'Win_pct':np.nan,'Ties_pct':np.nan,
                                 'Sample_scope':scope,'Status':'NO_SHARED_STEPS'})
                continue
            aa=merged.abs_err_a.to_numpy(float)
            bb=merged.abs_err_b.to_numpy(float)
            if not np.isfinite(aa).all() or not np.isfinite(bb).all():
                raise ValueError(f'Non-finite errors: {a} vs {b}')
            test=ks_2samp(aa,bb,alternative='two-sided',method='asymp')
            scope='2010 pairwise matched' if 'IFS 2010' in (a,b) else 'All years pairwise matched'
            for first,second,x,y in ((a,b,aa,bb),(b,a,bb,aa)):
                rows.append({'Lead_h':lead,'First':first,'Second':second,
                             'N':len(x),'KS_D':float(test.statistic),
                             'KS_p':float(test.pvalue),
                             'Significant':bool(test.pvalue<0.05),
                             'Win_pct':float(100*np.mean(x<y)),
                             'Ties_pct':float(100*np.mean(x==y)),
                             'Sample_scope':scope,'Status':'OK'})
    return pd.DataFrame(rows)


def _original_make_ks_tex(pairs, out, label, title, leads=(12,18,24)):
    pairs=pairs.copy()
    def format_cell(lead,a,b):
        r=pairs[(pairs.Lead_h==lead)&(pairs.First==a)&(pairs.Second==b)]
        if len(r)!=1:raise ValueError(f'KS missing {lead} {a}/{b}')
        r=r.iloc[0]
        return ('--' if int(r.N)==0 else f'{r.Win_pct:.1f} / {r.KS_D:.3f}' +
                (r'$^*$' if bool(r.get('Significant', False)) else ''))
    if label == 'tab:supp_ks_ifs_2010_12_24':
        # Preserve the row order and the direction of model comparisons in
        # the user's existing manuscript Supplementary Table S4.
        combos=[('Linear','Persistence'),('Control','Persistence'),
                ('Leakage','Persistence'),('Control','Linear'),
                ('Leakage','Linear'),('Leakage','Control'),
                ('Control','IFS 2010'),('Leakage','IFS 2010'),
                ('Linear','IFS 2010'),('Persistence','IFS 2010')]
    else:
        group=list(FIG_MODEL_ORDER) if '2010' in label else ['Leakage','Control','Linear','Persistence']
        combos=[(a,b) for i,a in enumerate(group) for b in group[i+1:]]
    body=[r'\begin{table}[htbp]',r'\centering',
          r'\caption{'+title+r'}', r'\label{'+label+r'}',
          r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{l'+'c'*len(leads)+'}',r'\toprule',
          'Model pair (first vs second) & '+' & '.join(f'{h} h' for h in leads)+r' \\',r'\midrule']
    for a,b in combos:
        body.append(f'{a} vs. {b} & '+' & '.join(format_cell(h,a,b) for h in leads)+r' \\')
    body.extend([r'\bottomrule',r'\end{tabular}%',r'}',r'\end{table}'])
    (out/(label.replace(':','_')+'.tex')).write_text('\n'.join(body)+'\n',encoding='utf-8')


def optional_6h_models(df, path, model):
    # A 6h ablation is plotted on strict 2010 only if it covers EVERY case.
    if path is None:return None
    if not path.is_file():
        raise FileNotFoundError(f'{model}: {path}')
    header=pd.read_csv(path,nrows=0).columns.tolist()
    pred='pred_mslp_corr' if 'pred_mslp_corr' in header else 'pred_mslp_hpa'
    truth='true_mslp_corr' if 'true_mslp_corr' in header else 'true_mslp_hpa'
    ab=normalize_keys(read_subset(path,KEY+[pred,truth]),str(path))
    joined=df[KEY+['center_truth']].merge(ab,on=KEY,how='left',validate='one_to_one')
    if joined[pred].isna().any():
        print(f'  NOTE: {model} absent for {joined[pred].isna().sum()} 2010 strict rows; '
              'not adding its point to the 2010 comparison (no mixed subsets).')
        return None
    if np.abs(joined[truth]-joined.center_truth).max()>CHECK_TOL:
        raise ValueError(f'{model} verification target mismatches Control')
    return float(np.abs(joined[pred]-joined.center_truth).mean())


def optional_full_years_ablation(path, model):
    if path is None:return None
    if not path.is_file():raise FileNotFoundError(f'{model}: missing {path}')
    header=pd.read_csv(path,nrows=0).columns.tolist()
    t=next((c for c in ('true_mslp_corr','true_mslp_hpa') if c in header),None)
    p=next((c for c in ('pred_mslp_corr','pred_mslp_hpa') if c in header),None)
    if not t or not p:raise ValueError(f'{path}: ablation requires truth/pred MSL columns; found {header}')
    d=pd.read_csv(path,usecols=[t,p]);e=pd.to_numeric(d[p],errors='raise')-pd.to_numeric(d[t],errors='raise')
    if e.empty or not np.isfinite(e.to_numpy(float)).all():raise ValueError(f'{path}: nonfinite 6h ablation error')
    return {'Lead_h':6,'Model':model,'N':len(e),'MAE':np.abs(e).mean(),
            'Bias':e.mean(),'STD':e.std(ddof=1)}




def resolve_ifs_dir(directory):
    root=Path(directory)
    flat=all((root/f'true_vs_pred_{lead}h.csv').is_file() for lead in LEADS)
    sub=root/'matched_control'
    nested=all((sub/f'true_vs_pred_{lead}h.csv').is_file() for lead in LEADS)
    if flat and nested:
        raise ValueError(f'Ambiguous IFS input: complete CSV sets exist in {root} AND {sub}. '
                         'Point --ifs-dir to the intended one explicitly.')
    if nested:return sub
    if flat:return root
    missing=[str(root/f'true_vs_pred_{lead}h.csv') for lead in LEADS
             if not (root/f'true_vs_pred_{lead}h.csv').is_file()]
    raise FileNotFoundError('Could not find a complete four-lead IFS CSV set. '
                            f'Tried {root} and {sub}; missing direct files: {missing}')


SCRIPT_VERSION = "V9-SUPP-2010-BIAS-ORIGINAL-STYLE-20261029"

def parse_cli():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    parser.add_argument('--base-dir',type=Path, default=Path('~/Desktop/steps_analysis/updated_data').expanduser())
    parser.add_argument('--control-dir',type=Path,default=None)
    parser.add_argument('--leakage-dir',type=Path,default=None)
    parser.add_argument('--ifs-dir',type=Path,default=Path('~/Desktop/forecast/Fixed_2010').expanduser())
    parser.add_argument('--ifs-history',type=Path,default=Path(__file__).resolve().with_name('source_all_valid_times.csv'))
    parser.add_argument('--out-dir',type=Path,default=None)
    parser.add_argument('--bootstraps',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--no-plot',action='store_true',help='Skip all figures (tables/data only)')
    parser.add_argument('--check-inputs',action='store_true',help='Only print actual file paths and 6h pairwise overlaps, then exit')
    parser.add_argument('--no-full-years',action='store_true',help='Skip supplementary full-year curves and 6h whole-dataset intensity analysis')
    parser.add_argument('--curve-bootstraps',type=int,default=350,help='Whole-track bootstrap draws for binned intensity curves')
    parser.add_argument('--intensity-bin-boundaries', type=int, default=25,
        help='Number of equally spaced bin boundaries up to the 99.5th percentile; original source uses 25 (24 possible points). Use 13 for fewer points (12 bins).')
    parser.add_argument('--msl-only-file',type=Path,default=None,help='Optional 6h ablation predictions with storm_key and step_index')
    parser.add_argument('--simple-nn-file',type=Path,default=None,help='Optional 6h ablation predictions with storm_key and step_index')
    parser.add_argument('--msl-t1-file',type=Path,default=None,help='Optional MSL-only (t-1) 6h ablation')
    parser.add_argument('--control-t1-file',type=Path,default=None,help='Optional Control (t-1) 6h ablation')
    parser.add_argument('--include-folder-variants',action='store_true',
        help='Include control_0/1/2 and msl_only_0/1/2 as six separately labeled models in 6h KS only; never assume suffix means t-1.')
    args=parser.parse_args()
    if args.curve_bootstraps<100: parser.error('--curve-bootstraps must be >=100')
    if args.intensity_bin_boundaries<3: parser.error('--intensity-bin-boundaries must be >=3')
    if args.bootstraps<100:
        parser.error('--bootstraps must be >= 100')
    args.base_dir=args.base_dir.expanduser()
    args.control_dir=(args.control_dir or args.base_dir/'control').expanduser()
    args.leakage_dir=(args.leakage_dir or args.base_dir/'leakage').expanduser()
    args.ifs_dir=args.ifs_dir.expanduser()
    args.ifs_history=args.ifs_history.expanduser()
    args.out_dir=(args.out_dir or args.base_dir/'Plots'/'Paper_Figures_V9_Bias_2010').expanduser()
    args.msl_only_file=args.msl_only_file.expanduser() if args.msl_only_file else None
    args.simple_nn_file=args.simple_nn_file.expanduser() if args.simple_nn_file else None
    args.msl_t1_file=args.msl_t1_file.expanduser() if args.msl_t1_file else None
    args.control_t1_file=args.control_t1_file.expanduser() if args.control_t1_file else None
    # Explicit, user-verified filenames under updated_data. Never fall back to
    # old ~/Downloads paths: those may be stale or unrelated experiments.
    verified_defaults = {
       'msl_only_file': args.base_dir / 'msl_only' / 'all_cyclone_data_full_31_version_156_157_158_159_6h.csv',
       'simple_nn_file': args.base_dir / 'simple_nn' / 'all_cyclone_data_full_31_version_0_1_2_3.csv',
    }
    for field,path in verified_defaults.items():
        if getattr(args,field) is None:
            if not path.is_file():
                raise FileNotFoundError(f'Expected {field} at the USER-VERIFIED location: {path}. '
                   'Supply the corresponding --*-file option if it was moved.')
            setattr(args,field,path)
    # These older labels refer to specific forecast-history experiments; the
    # directory suffixes _0/_1/_2 have not been independently defined. Never
    # relabel a folder variant as t-1 just because the number looks similar.
    global _STYLE_ABLATION_FILES
    _STYLE_ABLATION_FILES = {
        'MSL-only': args.msl_only_file,
        'Simple NN': args.simple_nn_file,
    }
    if args.msl_t1_file is not None:
        _STYLE_ABLATION_FILES['MSL-only (t-1)'] = args.msl_t1_file
    if args.control_t1_file is not None:
        _STYLE_ABLATION_FILES['Control (t-1)'] = args.control_t1_file
    if args.include_folder_variants:
        variant_files = {
            'Control [control_0]': ('control_0','all_cyclone_data_full_31_version_0_10_7_8_all_01.csv'),
            'Control [control_1]': ('control_1','all_cyclone_data_full_31_version_1_8_6_7_all_11.csv'),
            'Control [control_2]': ('control_2','all_cyclone_data_full_31_version_4_8_10_6_all_21.csv'),
            'MSL-only [msl_only_0]': ('msl_only_0','all_cyclone_data_full_31_version_2_4_9_11_msl_01.csv'),
            'MSL-only [msl_only_1]': ('msl_only_1','all_cyclone_data_full_31_version_0_4_5_8_msl_11.csv'),
            'MSL-only [msl_only_2]': ('msl_only_2','all_cyclone_data_full_31_version_1_5_3_7_msl_21.csv'),
        }
        print('[VARIANTS] Extra folder variants will be compared as distinct models on Figure 5b.', flush=True)
        print('[VARIANTS] Folder suffixes are NOT interpreted as forecast-history offsets.', flush=True)
        for label,(folder,filename) in variant_files.items():
            path = args.base_dir / folder / filename
            if not path.is_file():
                raise FileNotFoundError(f'{label} missing in user-provided folder: {path}')
            _STYLE_ABLATION_FILES[label]=path
    return args


def main():
    args=parse_cli()
    print(f"[SCRIPT VERSION] {SCRIPT_VERSION}", flush=True)
    print(f"[SCRIPT PATH] {Path(__file__).resolve()}", flush=True)
    # Pre-flight check: every model name used in the optional 6h
    # KS heatmap must be present in the signed-error mapping.
    assert all(model in ERROR_COLUMNS for model in (*FIG_MODEL_ORDER, *OPTIONAL_6H_MODELS)), \
        "Incomplete ERROR_COLUMNS mapping"
    args.ifs_dir=resolve_ifs_dir(args.ifs_dir)
    # Fail preflight before touching any previous output.
    for lead in LEADS:
        for p in (args.control_dir/CONTROL_NAMES[lead],
                  args.leakage_dir/LEAKAGE_NAMES[lead],
                  args.ifs_dir/f'true_vs_pred_{lead}h.csv'):
            if not p.is_file():raise FileNotFoundError(p)
    if not args.check_inputs and not args.ifs_history.is_file():
        raise FileNotFoundError(f'IFS local-minimum ERA5 history missing: {args.ifs_history}. '
              'Download the bundled source_all_valid_times.csv and keep it next to this script, '
              'or use --ifs-history PATH. This is REQUIRED for correctly defined IFS SS.')
    if args.check_inputs:
        print('\n6h input and optional model path audit (NO charts will be created):')
        print('  base_dir:', args.base_dir)
        print('  NOTE: t-1 sources are included ONLY when explicitly supplied. Suffixes _0/_1/_2 are not interpreted.')
        for model,path in globals().get('_STYLE_ABLATION_FILES',{}).items():
            print(f'  {model}: ' + (f'{Path(path).resolve()} [EXISTS]' if path and Path(path).is_file() else 'NOT DETECTED'))
        frames=load_pairwise_error_frames(6,args.control_dir,args.leakage_dir,args.ifs_dir,
                                          optional_6h=globals().get('_STYLE_ABLATION_FILES',{}))
        for model in frames:
            if model == 'IFS 2010': continue
            common=len(frames[model][KEY].merge(frames['IFS 2010'][KEY],on=KEY,how='inner',validate='one_to_one'))
            print(f'  {model}: total={len(frames[model]):,}; overlap with IFS 2010={common:,}')
        return
    history=load_history(args.ifs_history)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    print('Matched 2010 Table 5 — STRICT, target-consistent persistence')
    print('Control directory:',args.control_dir)
    print('Leakage directory:',args.leakage_dir)
    print('IFS directory:',args.ifs_dir)
    print('ERA5 history:',args.ifs_history)
    print('Output directory:',args.out_dir)
    audit=[]; diag=[]; metrics=[]; paired_by_lead={}
    for lead in LEADS:
        print(f'\n[{lead}h]')
        paired,d=merge_for_lead(lead,args.control_dir,args.leakage_dir,
                                 args.ifs_dir,history,args.out_dir,audit)
        paired_by_lead[lead]=paired
        rows=compute_metrics(paired,lead,args.bootstraps,args.seed)
        metrics.extend(rows);diag.append(d)
        print(f'  N={len(paired):,}; storms={d["storms"]:,}; '
              f'center/local t0/max target discrepancy={d["max_center_t0_diff_hpa"]:.3g}/'
              f'{d["max_local_target_diff_hpa"]:.3g} hPa')
        for row in rows:
            print(f'  {row["Model"]:12s} MAE={row["MAE"]:.3f} Bias={row["Bias"]:.3f} '
                  f'STD={row["STD"]:.3f} SS={row["SS"]:.3f} '
                  f'[{row["SS_CI_low"]:.3f},{row["SS_CI_high"]:.3f}]')
    metrics_df=pd.DataFrame(metrics)
    pd.DataFrame(audit).to_csv(args.out_dir/'Table5_2010_join_audit.csv',index=False)
    pd.DataFrame(diag).to_csv(args.out_dir/'Table5_2010_diagnostics.csv',index=False)
    metrics_df.to_csv(args.out_dir/'Table5_2010_metrics.csv',index=False,float_format='%.10g')
    (args.out_dir/'Table5_2010_consistent.tex').write_text(make_latex(metrics_df),encoding='utf-8')
    if not args.no_plot:
        set_ams_style()
        print('\nGenerating strict 2010 Table 5 and supplementary diagnostics (all models same cases)...')
        main_fig_dir=args.out_dir/'Main_Paper'
        supp_fig_dir=args.out_dir/'Supplementary'
        main_fig_dir.mkdir(parents=True,exist_ok=True)
        supp_fig_dir.mkdir(parents=True,exist_ok=True)
        plot_metrics_2010(metrics_df,supp_fig_dir)
        make_ss_figure(metrics_df,supp_fig_dir)
        plot_error_advantage(args.control_dir,args.ifs_dir,main_fig_dir)
        plot_error_distribution(paired_by_lead,supp_fig_dir)
        paired_rows=[]
        for h in LEADS:
            paired_rows.extend(pair_stats(paired_by_lead[h],h))
        pair_df=pd.DataFrame(paired_rows)
        pair_df.to_csv(supp_fig_dir/'IFS_matched_KS_Win_all_leads.csv',index=False)
        # Figure 5b: pairs with IFS use 2010; all other pairs use all years.
        # Build 6h pairs independently of the 2010 strict 3-way Table 5.
        all_frames_6h=load_pairwise_error_frames(
            6,args.control_dir,args.leakage_dir,args.ifs_dir,
            optional_6h=globals().get('_STYLE_ABLATION_FILES',{}))
        mixed_6h=mixed_pairwise_ks(all_frames_6h,6)
        print('  Figure 5b: only IFS pairs use 2010; all non-IFS pairs use all years')
        heatmap_6h(mixed_6h,all_frames_6h,main_fig_dir)
        # Preserve a separate strictly matched 2010-only diagnostic table.
        pair_df.to_csv(supp_fig_dir/'IFS_strict_2010_KS_Win_all_leads.csv',index=False)
        make_ks_tex(pair_df,supp_fig_dir,'tab:supp_ks_ifs_2010_STRICT_12_24',
            'Pairwise absolute-error comparisons on strictly matched 2010 cyclone steps. '
            'Entries are first-model win percentage / two-sample KS distance. '
            'All pairs at a given horizon use identical forecast steps; '
            'IFS is verified against ERA5 local minimum and other models against ERA5 tracked-center MSL. '
            'KS distances are descriptive; no independent-sample significance claims are made.')
        intensity_2010=add_simple_nn_errors(paired_by_lead[6],args.simple_nn_file,'2010 strict 6h intensity')
        if intensity_2010 is not None:
            plot_intensity_curve(intensity_2010, list(FIG_MODEL_ORDER)+['Simple NN'], supp_fig_dir,
                 'S3_2010_intensity_change_6h_strict', args.curve_bootstraps,
                 args.seed+611, min_count=20, n_boundaries=args.intensity_bin_boundaries)
        else:
            print('  SKIP 2010 intensity chart: cannot reproduce original six-model figure without Simple NN.')
        add_optional_ablation_points(paired_by_lead[6],args,metrics_df,supp_fig_dir)
        plot_reference_supplementary_bias_2010(metrics_df, paired_by_lead[6], args, supp_fig_dir)
        if not args.no_full_years:
            print('Generating separate whole-dataset supplementary charts and tables...')
            full_rows=[]; full_ks=[]
            for h in LEADS:
                full_metrics,paired_full=load_full_lead(h,args.control_dir,args.leakage_dir)
                full_rows.extend(full_metrics)
                full_ks.extend(_full_pairs(h,paired_full))
                print(f'  {h}h: full Control={next(x["N"] for x in full_metrics if x["Model"]=="Control"):,}; '
                      f'Control-Leakage paired={len(paired_full):,}')
                if h==6:
                    intensity_full=add_simple_nn_errors(paired_full,args.simple_nn_file,'full-dataset 6h intensity')
                    if intensity_full is not None:
                        plot_intensity_curve(intensity_full,
                            ['Leakage','Control','Linear','Persistence','Simple NN'],
                            main_fig_dir,'Figure6_candidate_full_dataset_intensity_change_6h',
                            args.curve_bootstraps,args.seed+666,min_count=20,
                            n_boundaries=args.intensity_bin_boundaries)
                    else:
                        print('  SKIP full-data intensity chart: original included Simple NN.')
            for name,path in [('MSL-only',args.msl_only_file),('Simple NN',args.simple_nn_file),
                              ('MSL-only (t-1)',args.msl_t1_file),('Control (t-1)',args.control_t1_file)]:
                r=optional_full_years_ablation(path,name)
                if r is not None:full_rows.append(r)
            full_stats=pd.DataFrame(full_rows)
            full_stats.to_csv(supp_fig_dir/'S3_FullDataset_metrics.csv',index=False)
            full_ks_df=pd.DataFrame(full_ks)
            full_ks_df.to_csv(supp_fig_dir/'S3_FullDataset_KS_Win_paired_STRICT_LK.csv',index=False)
            # Mixed-cohort tables follow the ORIGINAL Figure 5b methodology:
            # with IFS: pairwise 2010; without IFS: pairwise all years.
            mixed_by_lead=[mixed_6h]
            for h in (12,18,24):
                frames=load_pairwise_error_frames(h,args.control_dir,args.leakage_dir,args.ifs_dir)
                mixed_by_lead.append(mixed_pairwise_ks(frames,h))
            mixed_all=pd.concat(mixed_by_lead,ignore_index=True)
            mixed_all.to_csv(supp_fig_dir/'KS_mixed_cohorts_ALL_leads.csv',index=False)
            nonifs=mixed_all.loc[(mixed_all.First!='IFS 2010') &
                                 (mixed_all.Second!='IFS 2010')].copy()
            nonifs.to_csv(supp_fig_dir/'S3_FullDataset_KS_Win_paired.csv',index=False)
            make_ks_tex(mixed_all,supp_fig_dir,'tab:supp_ks_ifs_2010_12_24',
              'Pairwise absolute-error comparisons. Pairs involving IFS are '
              'matched by cyclone and step from 2010; all other model pairs '
              'use pair-specific common samples from all available years. '
              'Entries show first-model win percentage / two-sample KS D. '
              'Asterisks denote conventional KS p<0.05; within-cyclone '
              'temporal dependence is not accounted for.')
            make_full_dataset_graphs(full_stats,None,supp_fig_dir)
            plot_original_figure5a_cohorts(full_stats, args.ifs_dir, main_fig_dir, supp_fig_dir)
            make_ks_tex(nonifs,supp_fig_dir,'tab:supp_ks_all_leads',
              'Pairwise absolute-error comparisons across the full multi-year dataset, '
              'restricted to pairwise common steps at each lead. Each cell is '
              'first-model win percentage / KS distance. Asterisks denote conventional KS p<0.05.',
              leads=LEADS)

    info=(f'Figure5a: ORIGINAL mixed cohorts (IFS 2010 only; other models all years).\n'
          f'Control directory: {args.control_dir}\nLeakage directory: {args.leakage_dir}\n'
          f'IFS directory: {args.ifs_dir}\nERA5 all-times: {args.ifs_history}\n'
          f'Bootstrap replicates: {args.bootstraps}, seed: {args.seed}\n'
          'Definition: strict 3-way intersection, exact storm/step.\n'
          'IFS persistence = local ERA5 minimum at t0 (NOT Control center at t0).\n'
          'Other persistence = tracked ERA5 center at t0.\n'
          'Cases identical per lead; spatial verification targets different across models.\n')
    (args.out_dir/'table5_run_info.txt').write_text(info,encoding='utf-8')
    print('\nSUCCESS. Copy LaTeX from:',args.out_dir/'Table5_2010_consistent.tex')
    print('Figures in Main_Paper/ and Supplementary/ under:',args.out_dir)
    if args.no_full_years and not args.no_plot:
        print('NOTE: --no-full-years skips Figure5a original mixed-cohort plot and mixed KS tables for 12/18/24h.')


# ==========================================================================
# EXACT STYLE ADAPTERS FROM USER'S ORIGINAL PLOTTING SCRIPTS (26 Oct 2026)
# Numerical data and cohort definitions above this section are unchanged.
# The only deliberate presentation deviations: NO TITLES, NO SUBFIGURES.
# ==========================================================================
from contextlib import contextmanager

# Original source: 'Pasted markdown (2)(3).md' (MAE and Bias).
STYLE_MAE_BIAS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 13.5,
    'legend.title_fontsize': 14,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
}
# Original source: 'Pasted markdown (3).md' (KS and KDE).
STYLE_KS_KDE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'figure.figsize': (9, 6),
    'figure.dpi': 300,
    'font.size': 18,
    'axes.labelsize': 22,
    'axes.titlesize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18,
    'lines.linewidth': 2.5,
    'lines.markersize': 9,
}
# Original source: 'Pasted markdown (5).md' (Saturation curves).
STYLE_SATURATION = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'figure.dpi': 300,
    'font.size': 12,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 7,
}
# Original source: 'Pasted markdown (4).md' (SS). It does NOT set
# global rcParams, so keep Matplotlib's defaults and copy every explicit
# fontsize/size/color/grid setting from the original plotting block.
REFERENCE_COLORS = {'IFS 2010': 'purple', 'Leakage': 'orange',
                    'Control': 'blue', 'Linear': 'green',
                    'Persistence': 'red', 'MSL-only': 'black',
                    'Simple NN': 'tab:brown',
                    'MSL-only (t-1)': 'tab:pink', 'Control (t-1)': 'tab:cyan'}

@contextmanager
def reference_style(settings):
    import matplotlib as mpl
    # Isolated style per Figure: prevent the larger KS fonts from leaking
    # into MAE or the explicit SS plot.
    with mpl.rc_context(rc=settings):
        yield


def set_ams_style():
    """Original scripts had DIFFERENT style dicts; do not set one globally."""
    import matplotlib
    matplotlib.use('Agg')


def save_reference_figure(fig, out, name, dpi=300, skip_tight_layout=False):
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # On large KS matrices, Matplotlib may report that tight_layout cannot
    # fit all the original 18/16/15 pt labels at the original 12x10 figsize.
    # Do not shrink fonts or change the requested original figure dimensions:
    # bbox_inches='tight' in savefig captures those labels on export instead.
    if not skip_tight_layout:
        fig.tight_layout()
    fig.savefig(out/(name+'.pdf'),format='pdf',bbox_inches='tight')
    fig.savefig(out/(name+'.png'),dpi=dpi,bbox_inches='tight')
    plt.close(fig)


def _model_stats(stats, model):
    return stats[stats.Model.eq(model)].set_index('Lead_h').loc[list(LEADS)]


def _reference_mae_plot(stats, out, name, include_ifs=True, extras=None):
    import matplotlib.pyplot as plt
    extras=extras or {}
    if not include_ifs and not extras:
        for model in ('MSL-only','Simple NN','MSL-only (t-1)','Control (t-1)'):
            r=stats.loc[stats.Model.eq(model)&stats.Lead_h.eq(6)]
            if len(r): extras[model]=(float(r.MAE.iloc[0]),0.)
    labels={'IFS 2010':'IFS 2010','Leakage':'Leakage','Control':'Control',
            'Linear':'Linear Regression','Persistence':'Persistence'}
    with reference_style(STYLE_MAE_BIAS):
        fig,ax=plt.subplots(figsize=(9,6))  # EXACT original figsize
        models=(['IFS 2010'] if include_ifs else [])+['Leakage','Control','Linear','Persistence']
        for model in models:
            if model not in set(stats.Model):continue
            s=_model_stats(stats,model)
            ax.plot(LEADS,s.MAE.to_numpy(float),marker='o',color=REFERENCE_COLORS[model],label=labels[model])
        for model,(val,_) in extras.items():
            ax.plot(6,val,marker='x',color=REFERENCE_COLORS[model],
                    markeredgewidth=2.5,markersize=10,linestyle='None',label=({'MSL-only':'MSL Only','MSL-only (t-1)':'MSL Only (t-1)'}.get(model,model)))
        ax.set_xlim(4,27)
        ax.set_xlabel('Lead Time (h)')
        ax.set_ylabel('MAE (hPa)')
        ax.set_xticks(LEADS)
        ax.grid(True,linestyle=':',alpha=0.7)
        ax.legend(loc='upper left',bbox_to_anchor=(0.105,1.0),frameon=False)
        # EXACT original left 1D zoom strip; this is a decorative inset axis,
        # not a second panel/subfigure. We intentionally remove its old '6h' title.
        strip_models=['Leakage','Control']+[m for m in ('MSL-only','MSL-only (t-1)','Control (t-1)') if m in extras]
        vals=[]
        for model in strip_models:
            vals.append((model,float(extras[model][0]) if model in extras else float(_model_stats(stats,model).MAE.iloc[0])))
        if len(vals)>1:
            axins=ax.inset_axes([0.075,0.38,0.025,0.54])
            axins.get_xaxis().set_visible(False)
            axins.set_frame_on(False)
            for spine in axins.spines.values():spine.set_visible(False)
            axins.axvline(x=0.,color='black',linewidth=1.5,zorder=0)
            for model,val in vals:
                if model.startswith('MSL') or model.startswith('Control (t-1)'):
                    axins.plot(0,val,marker='x',color=REFERENCE_COLORS[model],
                               markeredgewidth=2.5,markersize=10,clip_on=False,zorder=3)
                else:
                    axins.plot(0,val,marker='o',color=REFERENCE_COLORS[model],
                               markersize=9,clip_on=False,zorder=3)
            sorted_vals=sorted(v for _,v in vals)
            margin=(max(sorted_vals)-min(sorted_vals))*0.15
            if margin==0:margin=0.01
            axins.set_ylim(min(sorted_vals)-margin,max(sorted_vals)+margin)
            axins.set_xlim(-1,1)
            axins.yaxis.tick_left()
            axins.set_yticks(sorted_vals)
            axins.set_yticklabels([f'{v:.3f}' for v in sorted_vals],fontsize=11)
            axins.tick_params(axis='y',which='both',length=0,pad=8)
        save_reference_figure(fig,out,name,dpi=600)  # original PNG dpi=600


def _reference_bias_plot(stats,out,name,include_ifs=True,extras=None):
    import matplotlib.pyplot as plt
    extras=extras or {}
    if not include_ifs and not extras:
        for model in ('MSL-only','Simple NN','MSL-only (t-1)','Control (t-1)'):
            r=stats.loc[stats.Model.eq(model)&stats.Lead_h.eq(6)]
            if len(r): extras[model]=(float(r.Bias.iloc[0]),float(r.STD.iloc[0]))
    labels={'IFS 2010':'IFS 2010','Leakage':'Leakage','Control':'Control',
            'Linear':'Linear Regression','Persistence':'Persistence'}
    offsets={'IFS 2010':-0.60,'Leakage':-0.30,'Control':0,
             'Linear':0.30,'Persistence':0.60}
    with reference_style(STYLE_MAE_BIAS):
        fig,ax=plt.subplots(figsize=(9,6))
        models=(['IFS 2010'] if include_ifs else [])+['Leakage','Control','Linear','Persistence']
        for model in models:
            if model not in set(stats.Model):continue
            s=_model_stats(stats,model)
            ax.errorbar(np.asarray(LEADS,dtype=float)+offsets[model],
                        s.Bias.to_numpy(float),yerr=s.STD.to_numpy(float),
                        fmt='o',color=REFERENCE_COLORS[model],label=labels[model],
                        capsize=4,elinewidth=1.5,alpha=0.8)
        extras_offsets={'MSL-only':6.15,'Simple NN':6.30,
                        'MSL-only (t-1)':6.20,'Control (t-1)':6.10}
        for model,(bias,std) in extras.items():
            ax.errorbar(extras_offsets[model],bias,yerr=std,fmt='x',
                        color=REFERENCE_COLORS[model],label=({'MSL-only':'MSL Only','MSL-only (t-1)':'MSL Only (t-1)'}.get(model,model)),
                        markersize=10,markeredgewidth=2.5,capsize=4,elinewidth=1.5)
        ax.set_xlim(4,26.5)
        ax.axhline(0,color='black',linestyle='-',linewidth=1,alpha=0.5)
        ax.set_xlabel('Lead Time (h)')
        ax.set_ylabel('Bias (hPa)')
        ax.set_xticks(LEADS)
        ax.grid(True,linestyle=':',alpha=0.7)
        ax.legend(loc='upper left',bbox_to_anchor=(0.105,1.0),frameon=False)
        # Original narrow strip with original capsize, text size and position.
        vals=[('Leakage',float(_model_stats(stats,'Leakage').Bias.iloc[0]),float(_model_stats(stats,'Leakage').STD.iloc[0])),
              ('Control',float(_model_stats(stats,'Control').Bias.iloc[0]),float(_model_stats(stats,'Control').STD.iloc[0]))]
        for model in ('MSL-only','MSL-only (t-1)','Control (t-1)'):
            if model in extras:vals.append((model,*extras[model]))
        axins=ax.inset_axes([0.075,0.38,0.025,0.54]);axins.get_xaxis().set_visible(False)
        axins.set_frame_on(False)
        for spine in axins.spines.values():spine.set_visible(False)
        axins.axvline(x=0.,color='black',linewidth=1.5,zorder=0)
        for model,bias,std in vals:
            axins.errorbar(0.,bias,yerr=std,fmt='x' if model in extras else 'o',
                           color=REFERENCE_COLORS[model],capsize=4,elinewidth=1.5,
                           markersize=10 if model in extras else 8,
                           markeredgewidth=2.5 if model in extras else 0,
                           clip_on=False,zorder=3)
        sorted_vals=sorted(v for _,v,_ in vals)
        margin=(max(sorted_vals)-min(sorted_vals))*1.5
        if margin==0:margin=1.0
        axins.set_ylim(min(sorted_vals)-margin,max(sorted_vals)+margin)
        axins.set_xlim(-1,1)
        axins.yaxis.tick_left()
        axins.set_yticks(sorted_vals)
        axins.set_yticklabels([f'{v:.3f}' for v in sorted_vals],fontsize=11)
        axins.tick_params(axis='y',which='both',length=0,pad=8)
        save_reference_figure(fig,out,name,dpi=600)


# ---------------------------------------------------------------------------
# Supplementary Bias vs lead time — STRICT 2010 + IFS.
# Style is copied from the user's supplementary_bias_full_dataset script,
# not from the older reference MAE/Bias plot used elsewhere in this file.
# Error bars represent +/- SAMPLE STD of signed errors, NOT 95% confidence
# intervals of the mean. Sorting and 0.55h spacing follow the original source.
# ---------------------------------------------------------------------------
STYLE_SUPPLEMENTARY_BIAS_REFERENCE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'figure.dpi': 300,
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'lines.linewidth': 2.3,
    'lines.markersize': 7,
}

SUPP_BIAS_COLOR = {
    'IFS 2010': 'purple',
    'Leakage': 'orange',
    'Control': 'blue',
    'MSL-only': 'black',
    'Simple NN': 'tab:brown',
    'Linear': 'green',
    'Persistence': 'red',
}
SUPP_BIAS_LABEL = {'MSL-only': 'MSL Only',
                   'Linear': 'Linear Regression'}


def _strict_6h_optional_bias(paired6: pd.DataFrame, path: Path, model: str):
    """Compute optional ablation bias/STD on EXACT Table-5 6h cases.

    Never silently substitute an all-years bias for a strictly matched 2010
    bias.  No row-index, pressure, or storm-name-only matching is allowed.
    """
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f'{model}: expected 6h model CSV at {path}')
    hdr = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    pred = next((c for c in ('pred_mslp_corr', 'pred_mslp_hpa') if c in hdr), None)
    true = next((c for c in ('true_mslp_corr', 'true_mslp_hpa') if c in hdr), None)
    if pred is None or true is None:
        raise ValueError(f'{model}: missing prediction or truth in {path}, columns={hdr}')
    ab = normalize_keys(read_subset(Path(path), KEY+[pred,true]), str(path))
    matched = paired6[KEY+['center_truth']].merge(
        ab[KEY+[pred,true]], on=KEY, how='left', validate='one_to_one', indicator=True)
    missing = matched['_merge'].ne('both').sum()
    if missing:
        raise ValueError(f'{model}: only {len(matched)-missing:,} / {len(matched):,} '
                         'strict 2010 steps present; cannot plot on same cohort')
    if not np.isfinite(matched[[pred,true]].to_numpy(float)).all():
        raise ValueError(f'{model}: nonfinite values in strict 2010 rows')
    if np.abs(matched[true] - matched.center_truth).max() > CHECK_TOL:
        raise ValueError(f'{model}: validating ERA5 truth differs from Control on identical keys')
    err = (matched[pred]-matched.center_truth).to_numpy(float)
    return dict(Lead_h=6, Model=model, N=len(err),
                Storms=paired6.storm_key.nunique(),
                MAE=np.abs(err).mean(), Bias=err.mean(),
                STD=np.std(err, ddof=1),
                Verification_target='ERA5 tracked center')


def plot_reference_supplementary_bias_2010(metrics_df: pd.DataFrame,
                                           paired6: pd.DataFrame, args, out):
    """Original Bias/STD visual layout, with IFS purple and farthest left.

    2010 strict Table 5 cases for *every* model, including MSL-only & Simple
    NN at 6h. N = 3583, 3030, 2489, 1999 on the full user dataset.
    Main Table 5 and Figure 5a (mixed cohorts) remain untouched.
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    names = ('IFS 2010','Leakage','Control','Linear','Persistence')
    cols = ('Lead_h','Model','N','Storms','MAE','Bias','STD','Verification_target')
    d = metrics_df.loc[metrics_df.Model.isin(names), list(cols)].copy()
    for mod,path in [('MSL-only',args.msl_only_file), ('Simple NN',args.simple_nn_file)]:
        optional = _strict_6h_optional_bias(paired6, path, mod)
        d = pd.concat([d,pd.DataFrame([optional])],ignore_index=True)
    d = d.sort_values(['Lead_h','STD'],kind='stable').reset_index(drop=True)

    # Guard against accidentally blending 2010 with any full-dataset cohort.
    expected = metrics_df.loc[metrics_df.Model.eq('IFS 2010')].set_index('Lead_h')['N']
    for r in d.itertuples(index=False):
        if int(r.N) != int(expected.loc[int(r.Lead_h)]):
            raise AssertionError(f'{r.Model} {r.Lead_h}h: wrong cohort size N={r.N}')
    if d[['Bias','STD','MAE']].isna().any().any():
        raise ValueError('2010 supplementary bias: missing error statistics')

    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    # Figure-specific cohort & metric audit (no changes to Table 5).
    d.to_csv(out/'supplementary_bias_2010_with_IFS_metrics.csv',index=False,
             float_format='%.12g')

    with reference_style(STYLE_SUPPLEMENTARY_BIAS_REFERENCE):
        fig, ax = plt.subplots(figsize=(9,6))  # exactly the original bias fig
        spacing = 0.55  # precisely the original within-lead offset
        for lead in LEADS:
            sub = d.loc[d.Lead_h.eq(lead)].sort_values('STD',kind='stable').copy()
            # User requires IFS farthest left in each group; it is also the
            # smallest-STD model in the supplied 2010 dataset.
            sub = pd.concat([sub.loc[sub.Model.eq('IFS 2010')],
                             sub.loc[~sub.Model.eq('IFS 2010')]])
            if sub.empty or not sub.Model.eq('IFS 2010').any():
                raise ValueError(f'2010 bias: no IFS point for {lead}h')
            offsets = (np.arange(len(sub))-(len(sub)-1)/2.)*spacing
            for offset,row in zip(offsets,sub.itertuples(index=False)):
                ax.errorbar(float(lead)+float(offset), float(row.Bias),
                            yerr=float(row.STD), fmt='o',
                            color=SUPP_BIAS_COLOR[row.Model], capsize=4,
                            elinewidth=1.4, markersize=7.5, alpha=0.9,
                            zorder=3)
        ax.axhline(0.0,color='black',linewidth=1.0,alpha=0.6,zorder=1)
        ax.set_xlabel('Lead Time (h)')
        ax.set_ylabel('Bias (hPa)')
        ax.set_xticks(LEADS)
        ax.set_xlim(3.8,25.5)
        ax.grid(True,linestyle=':',alpha=0.6)
        order = d.loc[d.Lead_h.eq(6)].sort_values('STD',kind='stable').Model.tolist()
        # The original puts all 6h models first and then any additional models.
        for model in d.Model.unique():
            if model not in order: order.append(model)
        order.remove('IFS 2010')
        order.insert(0,'IFS 2010')
        handles = [mlines.Line2D([],[],color=SUPP_BIAS_COLOR[m],marker='o',
                   linestyle='None',markersize=7.5,
                   label=SUPP_BIAS_LABEL.get(m,m)) for m in order]
        ax.legend(handles=handles,loc='best',frameon=False)
        # No title and no subfigures, as requested.
        fig.tight_layout()
        fig.savefig(out/'supplementary_bias_2010_with_IFS.pdf',
                    format='pdf',bbox_inches='tight')
        fig.savefig(out/'supplementary_bias_2010_with_IFS.png',
                    dpi=600,bbox_inches='tight')
        plt.close(fig)
    print('  Saved supplementary_bias_2010_with_IFS.pdf/.png and matching metrics CSV')
    print('  All models shown on the SAME 2010 strict Table 5 samples per lead')


def plot_metrics_2010(stats,out):
    _reference_mae_plot(stats,out,'Figure5a_2010_strict_MAE',include_ifs=True)
    _reference_bias_plot(stats,out,'IFS_matched_2010_Bias_STD',include_ifs=True)
    # The original two-panel MAE/Bias graphic is deliberately not generated.


def make_full_dataset_graphs(full_metrics, full_matched_six, out):
    _reference_mae_plot(full_metrics,out,'S3_FullDataset_MAE',include_ifs=False)
    _reference_bias_plot(full_metrics,out,'S3_FullDataset_Bias_STD',include_ifs=False)


def add_optional_ablation_points(paired6,args,main_fig,out):
    optional={}
    for model,pathname in [
        ('MSL-only',args.msl_only_file),('Simple NN',args.simple_nn_file),
        ('MSL-only (t-1)',getattr(args,'msl_t1_file',None)),
        ('Control (t-1)',getattr(args,'control_t1_file',None))]:
        if pathname is None:continue
        v=optional_6h_models(paired6,pathname,model)
        if v is None:continue
        header=pd.read_csv(pathname,nrows=0).columns.tolist()
        tr='true_mslp_corr' if 'true_mslp_corr' in header else 'true_mslp_hpa'
        pr='pred_mslp_corr' if 'pred_mslp_corr' in header else 'pred_mslp_hpa'
        data=normalize_keys(read_subset(pathname,KEY+[tr,pr]),str(pathname))
        sub=paired6[KEY].merge(data,on=KEY,how='inner',validate='one_to_one')
        signed=(sub[pr]-sub[tr]).to_numpy(float)
        optional[model]=(v,(float(signed.mean()),float(signed.std(ddof=1))))
    if not optional:return
    _reference_mae_plot(main_fig,out,'Figure5a_2010_strict_MAE',include_ifs=True,
                        extras={k:(v[0],0) for k,v in optional.items() if k in ('MSL-only','Simple NN')})
    _reference_bias_plot(main_fig,out,'IFS_matched_2010_Bias_STD',include_ifs=True,
                         extras={k:v[1] for k,v in optional.items()})
    pd.DataFrame([{'Model':m,'Lead_h':6,'N':len(paired6),'MAE':v[0],
                   'Bias':v[1][0],'STD':v[1][1]} for m,v in optional.items()]).to_csv(
                   Path(out)/'Optional_6h_ablations_2010_strict.csv',index=False)


def plot_original_figure5a_cohorts(full_metrics, ifs_dir, main_out, supp_out):
    """Recreate original Figure 5a's cohort semantics, NOT a strict comparison.

    In the user's original script get_df_metrics(c_path) and
    get_df_metrics(lek_path) were called on the complete cross-validation
    dataset; only get_forecast2010_metrics was restricted to 2010.
    The strict cohort is still used elsewhere for Table 5 / 2010 KS / SS.
    """
    ifs_rows=[]
    for h in LEADS:
        input_path=Path(ifs_dir)/f'true_vs_pred_{h}h.csv'
        d=load_ifs(input_path,h)
        error=(d['ifs_pred']-d['ifs_truth']).to_numpy(dtype=float)
        ifs_rows.append({'Lead_h':h,'Model':'IFS 2010','N':len(d),
            'MAE':float(np.abs(error).mean()),'Bias':float(error.mean()),
            'STD':float(error.std(ddof=1))})
    mixed=pd.concat([full_metrics,pd.DataFrame(ifs_rows)],ignore_index=True)
    # Display 6h ablation points on THEIR original all-year populations,
    # not on the strict 2010 subpopulation.
    optional={}
    for model in ('MSL-only','Simple NN','MSL-only (t-1)','Control (t-1)'):
        rows=mixed.loc[(mixed.Model==model)&(mixed.Lead_h==6)]
        if len(rows):
            optional[model]=(float(rows.MAE.iloc[0]),
                             (float(rows.Bias.iloc[0]),float(rows.STD.iloc[0])))
    _reference_mae_plot(mixed,main_out,'Figure5a_MAE_original_cohorts',
        include_ifs=True,extras={model:(v[0],0) for model,v in optional.items()})
    _reference_bias_plot(mixed,supp_out,'Bias_STD_original_cohorts',
        include_ifs=True,extras={model:v[1] for model,v in optional.items()})
    mixed.to_csv(Path(main_out)/'Figure5a_original_cohort_metrics.csv',index=False)
    print('\nFigure 5a uses ORIGINAL cohort definitions:')
    for h in LEADS:
        print(f'  {h:2d}h | ' + ' | '.join(
            f'{model} N={int(mixed.loc[(mixed.Lead_h==h)&(mixed.Model==model),"N"].iloc[0]):,}, '
            f'MAE={float(mixed.loc[(mixed.Lead_h==h)&(mixed.Model==model),"MAE"].iloc[0]):.3f}'
            for model in ('IFS 2010','Leakage','Control','Linear','Persistence')))
    print('Figure 5a is a MIXED-cohort descriptive overview, NOT a paired comparison.')
    print('2010 strictly matched performance stays in Table 5 and Supplementary.')


def make_ss_figure(stats,outdir):
    """Copy the user's SS plotting block from Pasted markdown (4).md."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    with mpl.rc_context(rc=mpl.rcParamsDefault):
        x=np.asarray(LEADS,dtype=float)
        fig,ax=plt.subplots(figsize=(8,6))
        for model,offset,color,label in [('IFS 2010',-0.3,'purple','IFS 2010'),
                                         ('Leakage',-0.1,'orange','Leakage'),
                                         ('Control',+0.1,'blue','Control'),
                                         ('Linear',+0.3,'green','Linear Regression')]:
            v=_model_stats(stats,model)
            y=v.SS.to_numpy(float)
            ci=np.vstack([y-v.SS_CI_low.to_numpy(float),v.SS_CI_high.to_numpy(float)-y])
            ax.errorbar(x+offset,y,yerr=ci,fmt='-o',color=color,label=label,
                        linewidth=2,markersize=8,capsize=4,elinewidth=1.5)
        ax.axhline(0,color='black',linestyle='--',linewidth=1.5,alpha=0.7,
                   label='Persistence Baseline')
        ax.set_xlabel('Lead Time (h)',fontsize=20)
        ax.set_ylabel('Skill Score (SS)',fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{h}h' for h in LEADS])
        ax.set_ylim(-0.1,1.05)
        ax.tick_params(axis='both',labelsize=20)
        ax.grid(True,linestyle='--',alpha=0.6)
        for spine in ax.spines.values():spine.set_color('black');spine.set_linewidth(1.2)
        save_reference_figure(fig,outdir,'Skill_Score_vs_Persistence_2010_consistent',dpi=300)


def _ks_heatmap_reference(df,lead,out,name,model_names,draw_divider=True):
    """KS styling from supplied seaborn original; numerical values from paired rows."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    with reference_style(STYLE_KS_KDE):
        n=len(model_names);D=np.full((n,n),np.nan);win=np.full((n,n),np.nan)
        annotation=np.empty((n,n),dtype=object)
        source=df.loc[df.Lead_h.eq(lead)]
        for i,a in enumerate(model_names):
            for j,b in enumerate(model_names):
                if i==j:D[i,j]=0.0;annotation[i,j]='-';continue
                s=source.loc[source.First.eq(a)&source.Second.eq(b)]
                if len(s)!=1:raise ValueError(f'Missing KS entry: {lead}h {a} / {b}')
                r=s.iloc[0]
                if int(r.N)==0 or not np.isfinite(r.KS_D):
                    annotation[i,j]='N/A'
                    continue
                D[i,j]=float(r.KS_D);win[i,j]=float(r.Win_pct)
                annotation[i,j]=f'{r.KS_D:.3f}' + ('*' if bool(r.get('Significant',False)) else '')
        fig_size=(12,10) if n>5 else (10,8) # EXACT from original
        fig,ax=plt.subplots(figsize=fig_size)
        sns.heatmap(D,annot=annotation,fmt='',cmap='Blues',
                    xticklabels=model_names,yticklabels=model_names,
                    linewidths=0.5,linecolor='white',ax=ax,
                    vmin=0.,vmax=1.,square=True,
                    annot_kws={'size':16,'weight':'normal'},
                    cbar_kws={'label':'KS Distance (D)'})
        # seaborn masks NaN cells, so explicitly write N/A on white cells.
        for i in range(n):
            for j in range(n):
                if i!=j and annotation[i,j]=='N/A':
                    ax.text(j+0.5,i+0.5,'N/A',ha='center',va='center',
                            color='black',fontsize=16)
        cbar=ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('KS Distance (D)',size=22)
        for i in range(n):
            for j in range(n):
                if i!=j and np.isfinite(win[i,j]):
                    ax.text(j+0.96,i+0.04,f'{win[i,j]:.1f}%',ha='right',va='top',
                            color='black',fontsize=15,alpha=0.8,fontweight='bold')
        if draw_divider and 'IFS 2010' in model_names:
            idx=model_names.index('IFS 2010')
            ax.axhline(idx,color='black',linewidth=3)
            ax.axvline(idx,color='black',linewidth=3)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right',fontsize=18)
        ax.set_yticklabels(ax.get_yticklabels(),rotation=0,fontsize=18)
        # Preserve original heatmap font/size; avoid tight_layout's impossible-
        # margins warning when four optional ablations enlarge the matrix.
        save_reference_figure(fig,out,name,dpi=300,skip_tight_layout=True)


def heatmap_6h(mixed_pairs, frames_6h, out):
    """Original Figure 5b: IFS comparisons 2010; all other pairs all years.

    Optional ablations are pair-matched individually; they do not reduce any
    other model pair's sample population.
    """
    out=Path(out)
    names=list(frames_6h)
    ordered=sorted((m for m in names if m!='IFS 2010'),
                   key=lambda m:float(frames_6h[m].abs_err.mean()),reverse=True)
    ordered.append('IFS 2010')
    _ks_heatmap_reference(mixed_pairs,6,out,
                          'Figure5b_2010_strict_KS_Win_6h',ordered)
    # Legacy filename is retained for Overleaf compatibility, though its
    # cohort semantics are deliberately MIXED (not a strict 2010 comparison).
    mixed_pairs.to_csv(out/'Figure5b_mixed_cohort_KS_Win_6h.csv',index=False)
    (out/'Figure5b_KS_caption_note.txt').write_text(
        'Pairwise absolute-error comparisons at 6 h. Comparisons involving '
        'IFS use cyclone/step-matched forecasts from 2010 only; all other '
        'model pairs use their pairwise common steps from all available years. '
        'Percentages are the win ratio of the row model; D is the two-sample '
        'KS distance. Asterisks denote statistically significant differences '
        '(p < 0.05; conventional KS test). Conventional p-values do not '
        'account for dependence among steps of the same cyclone.\n', encoding='utf-8')




def plot_error_advantage(control_dir, ifs_dir, out):
    """Original 5c KDE: each line paired with IFS alone (2010 only).

    No implicit requirement for Leakage to exist on each Control/Persistence
    step: 6h and 24h comparisons use the IFS-Control pairwise intersection.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    series={}
    audit=[]
    for lead in (6,24):
        c=load_control(control_dir/CONTROL_NAMES[lead])
        i=load_ifs(ifs_dir/f'true_vs_pred_{lead}h.csv',lead)
        pair=c.merge(i,on=KEY,how='inner',validate='one_to_one')
        if pair.empty:raise ValueError(f'{lead}h: IFS/Control KDE pairing empty')
        ifs_err=np.abs(pair.ifs_pred-pair.ifs_truth).to_numpy(float)
        for model,col in [('Control','control_pred'),('Persistence','center_persistence')]:
            err=np.abs(pair[col]-pair.center_truth).to_numpy(float)
            vals=ifs_err-err
            series[(lead,model)]=vals
            audit.append({'Lead_h':lead,'Model':model,'N':len(vals),
                          'Win_pct':100*float(np.mean(vals>0)),
                          'Sample_scope':'IFS matched to model, 2010'})
    out=Path(out)
    pd.DataFrame(audit).to_csv(out/'Figure5c_IFS_pairwise_2010_counts.csv',index=False)
    with reference_style(STYLE_KS_KDE):
        fig,ax=plt.subplots(figsize=(10,7))
        for lead,model,label,color,ls in [
            (6,'Control','Control vs IFS (6h)','blue','-'),
            (24,'Control','Control vs IFS (24h)','blue','--'),
            (6,'Persistence','Persistence vs IFS (6h)','red','-'),
            (24,'Persistence','Persistence vs IFS (24h)','red','--')]:
            val=series[(lead,model)]
            win=100*np.mean(val>0)
            sns.kdeplot(data=val,fill=False,linewidth=2.5,
                        color=color,linestyle=ls,alpha=1.0,
                        label=f'{label} [Win: {win:.1f}%]',zorder=3,ax=ax)
        ax.axvline(0,color='black',linestyle='-',linewidth=1.2,alpha=0.5,zorder=2)
        ax.set_xlabel('Difference in Absolute Error (hPa)')
        ax.set_ylabel('Density')
        ax.grid(True,linestyle=':',alpha=0.7)
        ax.set_xlim(left=-10,right=4)
        ymax=ax.get_ylim()[1]
        ax.text(-0.5,ymax*1.02,r'$\leftarrow$ IFS Better',color='black',ha='right',fontsize=18)
        ax.text(0.5,ymax*1.02,r'Model Better $\rightarrow$',color='black',ha='left',fontsize=18)
        ax.legend(loc='upper left',frameon=False,fontsize=16)
        save_reference_figure(fig,out,'Figure5c_2010_strict_advantage_vs_IFS',dpi=300)




def plot_error_distribution(paired_by_lead,out):
    import matplotlib.pyplot as plt
    import seaborn as sns
    with reference_style(STYLE_KS_KDE):
        fig,ax=plt.subplots(figsize=(10,7))
        for lead,ls in zip(LEADS,['-','--','-.',':']):
            er=paired_by_lead[lead].error_IFS_2010.to_numpy(float)
            sns.kdeplot(data=er,fill=False,linewidth=2.5,color='purple',
                        linestyle=ls,alpha=1.0,label=f'{lead}h',ax=ax)
        ax.axvline(0,color='black',linestyle='-',linewidth=1.2,alpha=0.5)
        ax.set_xlabel('IFS Signed Error (hPa)')
        ax.set_ylabel('Density')
        ax.grid(True,linestyle=':',alpha=0.7)
        ax.legend(loc='upper left',frameon=False,fontsize=16)
        save_reference_figure(fig,out,'IFS_2010_signed_error_distributions',dpi=300)


def plot_intensity_curve(df,models,out,name,boot,seed,min_count,n_boundaries=25):
    """Original 6h saturation aesthetics and equal-width percentile bins.

    Cluster bootstrap remains in place for more robust confidence intervals.
    """
    import matplotlib.pyplot as plt
    with reference_style(STYLE_SATURATION):
        x=np.abs(df.persistence_error_center.to_numpy(float))
        centers,counts,avg,low,high,mask,edges=_bin_moments(
            df,x,models,boot,seed,min_count,n_boundaries=n_boundaries)
        visible=mask
        fig,ax=plt.subplots(figsize=(9,6))
        max_x_plot=0.;max_y_plot=0.
        for j,model in enumerate(models):
            if not visible.any():continue
            c=REFERENCE_COLORS[model]
            xx=centers[visible];yy=avg[visible,j]
            ax.fill_between(xx,low[visible,j],high[visible,j],color=c,alpha=0.15,edgecolor='none')
            if model=='Persistence':
                ax.plot(xx,yy,color=c,linestyle='-',linewidth=1.2,
                        marker=None,label='Persistence')
            else:
                ax.plot(xx,yy,color=c,linestyle='-',linewidth=2.5,
                        marker='o',markersize=8,markeredgecolor='white',
                        markeredgewidth=1,label='Linear Regression' if model=='Linear' else model)
            max_x_plot=max(max_x_plot,float(max(xx)))
            max_y_plot=max(max_y_plot,float(np.nanmax(high[visible,j])))
        ax.set_xlabel('Intensity change (hPa)')
        ax.set_ylabel('MAE (hPa)')
        ax.grid(True,linestyle=':',alpha=0.7)
        for spine in ax.spines.values():spine.set_color('black');spine.set_linewidth(1.2)
        ax.legend(loc='upper left',frameon=False)
        if max_x_plot>0:
            ax.set_xlim(0,max_x_plot*1.05)
            ax.set_ylim(0,max_y_plot*1.05)
        save_reference_figure(fig,out,name,dpi=300)
        records=[]
        for i in range(len(counts)):
            for j,model in enumerate(models):
                records.append({'Bin_low':edges[i],'Bin_high':edges[i+1],
                  'Center_x':centers[i],'N':int(counts[i]),'Model':model,
                  'MAE':avg[i,j],'MAE_CI_low':low[i,j], 'MAE_CI_high':high[i,j],
                  'plotted':bool(visible[i]),'Bin_definition':f'{n_boundaries} equal-width boundaries to P99.5',
                  'Population_N':len(df)})
        pd.DataFrame(records).to_csv(Path(out)/(name+'_bins.csv'),index=False)


def make_ks_tex(pairs,out,label,title,leads=(12,18,24)):
    # Preserve the old LaTeX generation and add the ORIGINAL separate heatmaps
    # for full-year non-IFS samples (one lead time per PDF/PNG).
    _original_make_ks_tex(pairs,out,label,title,leads)
    if '2010' not in label:
        models=['Persistence','Linear','Control','Leakage']
        for h in leads:
            _ks_heatmap_reference(pairs,h,out,f'KS_full_dataset_{h}h_no_IFS',models,draw_divider=False)


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print(f'\nFATAL: {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(1)
