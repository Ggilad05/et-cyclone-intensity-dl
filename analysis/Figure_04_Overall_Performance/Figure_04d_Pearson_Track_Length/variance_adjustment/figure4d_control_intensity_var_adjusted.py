#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4d — per-storm Pearson r of predicted vs observed cyclone intensity.

Only the Control models at 6 h and 24 h are drawn (no MSL-only series).

CAUTION ABOUT THE HORIZONTAL AXIS
---------------------------------
The older script used L = number of *available prediction/verification rows*,
truncated to at most 32 (6 h) or 29 (24 h), and labelled x = 6 * L hours.
That is NOT independently established full cyclone lifetime. This script
makes that distinction explicit:

  --length-mode available (default): reproduce the old grouping, plotting
      6 * count(available pairs) on x. A sample-count-based proxy, NOT
      physical lifetime or the exact elapsed time from first to last sample.

  --length-mode lifetime --lifetime-csv PATH:
      group by independently supplied FULL storm lifetime in hours. The CSV
      must have storm_key and lifetime_hours (or track_length_hours or
      storm_duration_hours), covering BOTH 6 h and 24 h Control files.
      Lifetime is binned to nearest 6 h to obtain a usable group size.
      Within each lifetime group, only storms with equal numbers of valid
      paired observations are randomized against each other. Singleton
      observation-count subgroups are excluded from both the real and null
      series for fair comparison and reported as an audit count.

Statistics:
  * One within-storm Pearson r per time series, then mean r across storms.
  * 95% storm-bootstrap percentile CI around mean r (2.5%–97.5%).
  * 5th–95th percentile of NULL group-mean r from independent whole-storm
    derangements (no self-match), within equal-length verification sequences.
  * No KS-test / significance symbols: those referred to MSL-only and are
    intentionally removed.
  * By default at least 8 valid 6-hourly PAIRS are required. This filter
    is not a verification of original cyclone lifetime >=48 hours.

Paths default to prior macOS data layout, but CLI arguments can override.
NEW DIAGNOSTIC: per-storm temporal variance of observed intensity and of the
forecast residual (pred_int - true_int), using EXACTLY the same kept storms,
prediction steps, and grouping as the main correlation figure. A separate
variability plot, group CSV, per-storm CSV and plain-English summary are saved.
This is a descriptive check, NOT a causal test of why r increases with length.

Requires numpy, pandas, matplotlib. No scipy needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

import matplotlib
# Use a noninteractive backend by default; --show can select the user's backend.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_DIR = Path('data/intensity_space_files/files/new')
DEFAULT_OUTPUT = Path('outputs/figure_04d')
NAME_CANDIDATES = {
    6: ['intensity_control_31.csv', 'intensity_all_cyclone_data_full_31.csv'],
    24: ['intensity_control_34.csv', 'intensity_all_cyclone_data_full_34.csv'],
}
MAX_SAMPLES = {6: 32, 24: 29}  # Matching the original analysis.
HOURS_PER_SAMPLE = 6


def file_for_lead(folder: Path, lead: int, specified: Path | None) -> Path:
    if specified is not None:
        p = specified.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f'{lead} h Control CSV not found: {p}')
        return p
    for name in NAME_CANDIDATES[lead]:
        p = folder / name
        if p.is_file():
            return p
    exp = '31' if lead == 6 else '34'
    candidates = sorted(folder.glob(f'intensity_*{exp}*.csv'))
    candidates = [p for p in candidates if 'msl_only' not in p.name.lower() and 'leakage' not in p.name.lower()]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f'Cannot uniquely identify Control {lead}h intensity file in {folder}. '
        f'Candidates: {[p.name for p in candidates]}. Specify --control-{lead}.'
    )


def clean_keys(series: pd.Series) -> pd.Series:
    return (series.astype('string').str.strip()
            .str.replace(r'''^[\"']+|[\"']+$''', '', regex=True)
            .str.replace(r'(?i)\.npy$', '', regex=True))


def read_control(path: Path, lead: int) -> pd.DataFrame:
    needed = ['storm_key', 'step_index', 'true_int', 'pred_int']
    present = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    absent = sorted(set(needed) - set(present))
    if absent:
        raise ValueError(f'{path.name}: missing columns {absent}; expected true_int and pred_int.')
    d = pd.read_csv(path, usecols=needed, low_memory=False)
    d.columns = d.columns.str.strip()
    d['storm_key'] = clean_keys(d['storm_key'])
    for col in needed[1:]:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    before = len(d)
    d = d.dropna(subset=needed)
    d = d[np.isfinite(d[['step_index', 'true_int', 'pred_int']].to_numpy()).all(axis=1)]
    if len(d) != before:
        print(f'[WARN] {path.name}: dropped {before-len(d):,} invalid rows')
    dupes = d.duplicated(['storm_key', 'step_index']).sum()
    if dupes:
        raise ValueError(f'{path.name}: {dupes:,} duplicate storm/step keys; not safe to calculate per-storm r')
    if d.empty:
        raise ValueError(f'{path.name}: no valid observations')
    print(f'[INPUT] Control {lead}h: {len(d):,} samples / {d.storm_key.nunique():,} storms — {path}')
    return d


def read_lifetimes(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = frame.columns.str.strip()
    if 'storm_key' not in frame:
        raise ValueError('Lifetime CSV requires storm_key column')
    candidates = ['lifetime_hours', 'track_length_hours', 'storm_duration_hours', 'duration_hours']
    col = next((s for s in candidates if s in frame.columns), None)
    if col is None:
        raise ValueError('Lifetime CSV needs a FULL cyclone-lifetime-in-hours column: ' + ', '.join(candidates))
    frame['storm_key'] = clean_keys(frame['storm_key'])
    frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame = frame.dropna(subset=['storm_key', col])
    if (frame[col] <= 0).any():
        raise ValueError('Storm lifetime must be positive')
    conflicting = frame.groupby('storm_key')[col].nunique()
    if (conflicting > 1).any():
        raise ValueError(f'Conflicting lifetime values for {(conflicting > 1).sum()} storms')
    values = frame.groupby('storm_key')[col].first().to_dict()
    print(f'[LIFETIME] Loaded full cyclone lifetimes for {len(values):,} storms from {path.name}')
    return values


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = mat - mat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError('Constant time series passed prefilter')
    return mat / norm


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a uniform random derangement using rejection sampling, with a safe fallback."""
    original = np.arange(n)
    for _ in range(50):
        perm = rng.permutation(n)
        if np.all(perm != original):
            return perm
    order = rng.permutation(n)
    p = np.empty(n, dtype=int)
    p[order] = np.roll(order, 1)
    return p


def make_groups(data: pd.DataFrame, lead: int, mode: str,
                lifetimes: dict[str, float] | None, min_pairs: int,
                minimum_lifetime_h: float):
    grouped: dict[float, list[dict]] = {}
    stats = dict(short=0, flat=0, missing_lifetime=0, below_lifetime=0,
                 nonconsecutive=0, beyond_cap=0, included=0)
    for key, frame in data.groupby('storm_key', sort=False):
        frame = frame.sort_values('step_index')
        full_sample_count = len(frame)
        frame = frame.iloc[:MAX_SAMPLES[lead]]
        if len(frame) < min_pairs:
            stats['short'] += 1
            continue
        vals = frame[['true_int', 'pred_int']].to_numpy(dtype=float)
        if (not np.isfinite(vals).all()
                or np.std(vals[:, 0]) < 1e-10
                or np.std(vals[:, 1]) < 1e-10):
            stats['flat'] += 1
            continue
        n_pairs = len(frame)
        if np.any(np.diff(frame.step_index.to_numpy(float)) != 1):
            stats['nonconsecutive'] += 1
        if full_sample_count > MAX_SAMPLES[lead]:
            stats['beyond_cap'] += 1

        if mode == 'available':
            # Legacy x=6*number of paired forecast-verification samples.
            x = n_pairs * HOURS_PER_SAMPLE
        else:
            lifetime = lifetimes.get(str(key)) if lifetimes is not None else None
            if lifetime is None:
                stats['missing_lifetime'] += 1
                continue
            if lifetime < minimum_lifetime_h:
                stats['below_lifetime'] += 1
                continue
            x = int(np.floor(float(lifetime) / HOURS_PER_SAMPLE + 0.5)) * HOURS_PER_SAMPLE
        grouped.setdefault(float(x), []).append({
            'key': str(key), 'n_pairs': n_pairs,
            'true': vals[:, 0], 'pred': vals[:, 1],
        })
        stats['included'] += 1
    print(f'[GROUP] Control {lead}h: {stats} | x positions: {len(grouped)}')
    return grouped, stats


def analyze_groups(groups: dict[float, list[dict]], lead: int, rng: np.random.Generator,
                   boots: int, nulls: int, min_storms: int, mode: str):
    records = []
    omitted_total = 0
    for x, items in sorted(groups.items()):
        # For the random null, partner time series must have the SAME number
        # of samples. Exclude singleton subgroups from BOTH null and observed.
        by_n: dict[int, list[dict]] = {}
        for item in items:
            by_n.setdefault(item['n_pairs'], []).append(item)
        eligible = {n: sub for n, sub in by_n.items() if len(sub) >= 2}
        excluded = sum(len(sub) for n, sub in by_n.items() if len(sub) < 2)
        omitted_total += excluded
        n = sum(map(len, eligible.values()))
        if n < min_storms:
            continue

        subarrays = []
        r_per_storm = []
        for sample_len, sub in eligible.items():
            t = normalize_rows(np.stack([item['true'] for item in sub]))
            p = normalize_rows(np.stack([item['pred'] for item in sub]))
            rs = np.einsum('ij,ij->i', t, p).clip(-1, 1)
            subarrays.append((t, p, len(sub)))
            r_per_storm.append(rs)
        observed = np.concatenate(r_per_storm)

        boot_mean = np.empty(boots, dtype=float)
        for j in range(0, boots, 128):
            take = min(128, boots - j)
            draw = rng.integers(0, n, size=(take, n))
            boot_mean[j:j + take] = observed[draw].mean(axis=1)

        null_mean = np.empty(nulls, dtype=float)
        for i in range(nulls):
            total = 0.0
            for t, p, k in subarrays:
                perm = derangement(k, rng)
                total += float(np.einsum('ij,ij->', t, p[perm]))
            null_mean[i] = total / n
        blo, bhi = np.quantile(boot_mean, [0.025, 0.975])
        nlo, nhi = np.quantile(null_mean, [0.05, 0.95])
        records.append(dict(
            lead_h=lead, length_mode=mode, x_hours=x, n_storms=n,
            n_omitted_singleton_count_groups=excluded,
            n_storms_raw_group=len(items), mean_r=float(observed.mean()),
            bootstrap_95_low=float(blo), bootstrap_95_high=float(bhi),
            random_mean_r=float(null_mean.mean()), random_5_low=float(nlo),
            random_95_high=float(nhi),
            n_sample_count_subgroups=len(eligible),
        ))
    print(f'[STATS] Control {lead}h: plotted {len(records)} x positions; '
          f'dropped {omitted_total:,} storms from singleton observation-count subgroups')
    if not records:
        raise RuntimeError(
            f'Control {lead}h: no bins with at least {min_storms} valid storms. '
            'Try smaller --min-storms or verify lifetime metadata.'
        )
    return records



def compute_variability(groups: dict[float, list[dict]], lead: int,
                        min_storms: int, mode: str) -> pd.DataFrame:
    """Per-storm variability on the SAME samples used for Pearson and random pairing.

    Both temporal variances are sample variances (ddof=1), within each storm:
      var_true  = Var_t[true_int(t)]
      var_error = Var_t[pred_int(t) - true_int(t)]
    We also report their SD and ratio; note that the error SD ignores any
    constant forecast bias, whereas RMSE includes it.
    """
    rows = []
    for x, items in sorted(groups.items()):
        by_count = {}
        for item in items:
            by_count.setdefault(item['n_pairs'], []).append(item)
        eligible = [item for sub in by_count.values() if len(sub) >= 2 for item in sub]
        if len(eligible) < min_storms:
            continue
        for item in eligible:
            true = item['true']
            pred = item['pred']
            err = pred - true
            var_true = float(np.var(true, ddof=1))
            var_error = float(np.var(err, ddof=1))
            sd_true = float(np.sqrt(var_true))
            sd_error = float(np.sqrt(var_error))
            # A zero residual variation makes a variance ratio undefined; keep
            # it missing, rather than generating infinite plotted values.
            ratio = var_true / var_error if var_error > 1e-20 else np.nan
            r = float(np.corrcoef(true, pred)[0, 1])
            rows.append({
                'lead_h': lead, 'length_mode': mode, 'storm_key': item['key'],
                'x_hours': x, 'n_samples': item['n_pairs'],
                'true_int_variance': var_true, 'true_int_sd': sd_true,
                'error_variance': var_error, 'error_sd': sd_error,
                'signal_to_error_variance_ratio': ratio,
                'error_rmse': float(np.sqrt(np.mean(err**2))),
                'pearson_r': r,
                'error_vs_truth_corr': float(np.corrcoef(true, err)[0, 1])
                    if sd_error > 1e-10 else np.nan,
            })
    if not rows:
        raise RuntimeError(f'No eligible storms for variability analysis at {lead} h')
    return pd.DataFrame(rows)


def summarize_variability(per_storm: pd.DataFrame) -> pd.DataFrame:
    """One row per position, using storms identical to the main Pearson curve."""
    records = []
    for (lead, x), d in per_storm.groupby(['lead_h', 'x_hours'], sort=True):
        records.append({
            'lead_h': int(lead), 'x_hours': float(x), 'n_storms': len(d),
            'mean_r': d['pearson_r'].mean(),
            'mean_true_variance': d['true_int_variance'].mean(),
            'median_true_variance': d['true_int_variance'].median(),
            'mean_error_variance': d['error_variance'].mean(),
            'median_error_variance': d['error_variance'].median(),
            'mean_true_sd': d['true_int_sd'].mean(),
            'median_true_sd': d['true_int_sd'].median(),
            'mean_error_sd': d['error_sd'].mean(),
            'median_error_sd': d['error_sd'].median(),
            'median_signal_to_error_variance_ratio':
                d['signal_to_error_variance_ratio'].median(),
            'median_error_rmse': d['error_rmse'].median(),
        })
    return pd.DataFrame(records).sort_values(['lead_h', 'x_hours']).reset_index(drop=True)


def plot_variability(groups_df: pd.DataFrame, out_dir: Path, mode: str,
                     show: bool) -> Path:
    """Separate diagnostic figure; DOES NOT touch Figure 4d main panel."""
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(9, 8),
                                     sharex=True, layout='constrained')
    # One color identifies the measured quantity, one line style the lead.
    for lead, ls in [(6, '-'), (24, '--')]:
        d = groups_df[groups_df['lead_h'] == lead].sort_values('x_hours')
        if d.empty:
            continue
        x = d.x_hours.to_numpy(float)
        ax.plot(x, d['median_true_variance'].to_numpy(float),
                color='tab:blue', ls=ls, marker='o', markersize=4.5,
                label=f'True intensity (Control {lead} h)')
        ax.plot(x, d['median_error_variance'].to_numpy(float),
                color='tab:red', ls=ls, marker='s', markersize=4.0,
                label=f'Forecast error (Control {lead} h)')
        ax_ratio.plot(x,
                      d['median_signal_to_error_variance_ratio'].to_numpy(float),
                      color='tab:green', ls=ls, marker='o', markersize=4.5,
                      label=f'Control {lead} h')
    ax.set_ylabel('Median within-storm variance\n(intensity units squared)')
    ax_ratio.set_ylabel('Median Var(true intensity) /\nVar(forecast error)')
    ax_ratio.set_xlabel('Cyclone lifetime (h)' if mode == 'lifetime'
                        else 'Available forecast samples × 6 h')
    for a in (ax, ax_ratio):
        a.grid(True, alpha=.35, ls=':')
        a.legend(loc='best', frameon=False, fontsize=11)
    stem = f'Figure4d_Intensity_Variability_{mode}'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / (stem + '.pdf'), bbox_inches='tight')
    fig.savefig(out_dir / (stem + '.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f'[SAVED] Variability diagnostic: {stem}.pdf/.png')
    return out_dir / stem


def write_variability_summary(per_storm: pd.DataFrame, by_length: pd.DataFrame,
                              out_dir: Path, mode: str) -> None:
    """Produce descriptive Spearman trend checks without p-value claims."""
    lines = [
        'FIGURE 4d: TRUE-INTENSITY VARIABILITY / CORRELATION DIAGNOSTIC',
        'Variances are calculated within each storm on the same forecast/verification',
        'samples used by the Pearson-correlation figure (including sample caps).',
        'The second variance is for pred_int - true_int, NOT for prediction itself.',
        'These associations are descriptive, not evidence of causation.',
        'A longer series does not, by itself, guarantee higher Pearson r.',
        f'Grouping mode: {mode}', '',
    ]
    for lead in (6, 24):
        d = per_storm[per_storm['lead_h'] == lead]
        g = by_length[by_length['lead_h'] == lead]
        def rho(a, b):
            val = d[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
            return float(val[a].corr(val[b], method='spearman')) if len(val) >= 3 else np.nan
        def grho(a, b):
            val = g[[a,b]].replace([np.inf, -np.inf], np.nan).dropna()
            return float(val[a].corr(val[b], method='spearman')) if len(val) >= 3 else np.nan
        xcol = 'x_hours'
        lines += [
            f'Control {lead} h: {len(d):,} storms in {len(g)} plotted x groups',
            f'  Spearman rho, across storms: x vs Var(true intensity) = '
            f'{rho(xcol, "true_int_variance"):+.3f}',
            f'  Spearman rho, across storms: x vs Var(forecast error) = '
            f'{rho(xcol, "error_variance"):+.3f}',
            f'  Spearman rho, across storms: x vs signal/error variance ratio = '
            f'{rho(xcol, "signal_to_error_variance_ratio"):+.3f}',
            f'  Spearman rho, across storms: x vs within-storm Pearson r = '
            f'{rho(xcol, "pearson_r"):+.3f}',
            f'  Spearman rho, across storms: Var(true intensity) vs Pearson r = '
            f'{rho("true_int_variance", "pearson_r"):+.3f}',
            f'  Spearman rho, across x-group summaries: x vs median Var(true) = '
            f'{grho(xcol, "median_true_variance"):+.3f}',
            f'  Spearman rho, across x-group summaries: x vs median signal/error ratio = '
            f'{grho(xcol, "median_signal_to_error_variance_ratio"):+.3f}',
            '',
        ]
    lines += [
        'INTERPRETATION:',
        '* If Var(true intensity) and the signal/error ratio both rise across x,',
        '  this is consistent with an increasing intensity signal relative to',
        '  temporally varying forecast errors in longer observed sequences.',
        '* If Var(true) increases but Var(error) increases as much or more, the',
        '  signal-to-error explanation is not established by that observation.',
        '* Storm length, temporal autocorrelation, different storm populations,',
        '  and the non-independent overlapping time windows can affect trends.',
        '* A high signal/error variance ratio does not PROVE error independence:',
        '  check error_vs_truth_corr in the per-storm CSV. This simple check',
        '  is not a causal attribution or a universal identity for Pearson r.',
    ]
    path = out_dir / f'Figure4d_Intensity_Variability_{mode}_summary.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'[SAVED] {path}')


def plot_figure(frame: pd.DataFrame, out_dir: Path, mode: str,
                show: bool, width: float = 9, height: float = 6):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 15,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(width, height), layout='constrained')
    blue = '#1f77b4'
    for lead, color, style in [(6, '#aaaaaa', '--'), (24, '#747474', ':')]:
        d = frame[frame.lead_h == lead].sort_values('x_hours')
        if d.empty:
            continue
        x = d.x_hours.to_numpy(float)
        ax.fill_between(x, d.random_5_low.to_numpy(float), d.random_95_high.to_numpy(float),
                        color=color, alpha=0.17, lw=0, zorder=1)
        ax.plot(x, d.random_mean_r.to_numpy(float), color=color, linestyle=style,
                lw=2, zorder=2)
    for lead, style in [(6, '-'), (24, '--')]:
        d = frame[frame.lead_h == lead].sort_values('x_hours')
        if d.empty:
            continue
        x = d.x_hours.to_numpy(float)
        ax.fill_between(x, d.bootstrap_95_low.to_numpy(float), d.bootstrap_95_high.to_numpy(float),
                        color=blue, alpha=0.20 if lead == 6 else 0.14, lw=0, zorder=3)
        ax.plot(x, d.mean_r.to_numpy(float), color=blue, linestyle=style,
                marker='o', markersize=5.7, lw=2.5, zorder=4)
    ax.set_ylabel(r'Mean Pearson $r$')
    if mode == 'lifetime':
        ax.set_xlabel('Cyclone lifetime (h)')
    else:
        ax.set_xlabel('Available forecast samples × 6 h')
    xmax = float(frame.x_hours.max())
    xmin = float(frame.x_hours.min())
    ax.set_xlim(xmin - 7, xmax + 7)
    ticks = np.arange(np.ceil(xmin / 24) * 24, xmax + 1, 24)
    if len(ticks) > 1:
        ax.set_xticks(ticks)
    ax.set_ylim(-.45, 1.05)
    ax.grid(True, linestyle=':', alpha=0.60)
    for spine in ax.spines.values():
        spine.set_color('lightgray')
        spine.set_linewidth(1.5)
    handles = [
        Line2D([], [], color=blue, linestyle='-', marker='o', lw=2.5, label='Control (6 h)'),
        Line2D([], [], color=blue, linestyle='--', marker='o', lw=2.5, label='Control (24 h)'),
        Line2D([], [], color='#aaaaaa', linestyle='--', lw=2, label='Random pairing (6 h)'),
        Line2D([], [], color='#747474', linestyle=':', lw=2, label='Random pairing (24 h)'),
    ]
    ax.legend(handles=handles, loc='lower right', frameon=False)
    stem = f'Figure4d_Control_Intensity_{mode}'
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / (stem + '.png')
    pdf = out_dir / (stem + '.pdf')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f'[SAVED] {png}\n[SAVED] {pdf}')
    return stem


def write_caption(out_dir: Path, mode: str, stem: str):
    if mode == 'lifetime':
        first = ('Mean per-storm Pearson correlation between predicted and observed cyclone '
                 'intensity as a function of full cyclone lifetime (binned into 6-h groups)')
        extra = ('The Pearson correlation is calculated from the available forecast–verification '
                 'pairs for each storm; the lifetime on the horizontal axis is obtained independently '
                 'of the number of such pairs.')
    else:
        first = ('Mean per-storm Pearson correlation between predicted and observed cyclone '
                 'intensity as a function of the number of available 6-hourly forecast–verification '
                 'pairs (shown in equivalent 6-h units)')
        extra = ('The horizontal coordinate reflects the number of available paired samples, '
                 'not independently verified full cyclone lifetime.')
    caption = (
        first + ' for the Control model at 6-h and 24-h lead times, alongside random-pairing '
        'baselines. Blue shading shows 95% storm-bootstrap confidence intervals for mean '
        'correlation. Gray shading shows the 5th–95th percentile range of the group-mean '
        'correlations from random pairings of different cyclones with matching numbers of '
        'available forecast–verification pairs. ' + extra
    )
    (out_dir / (stem + '_caption.txt')).write_text(caption + '\n', encoding='utf-8')
    print('[CAPTION] ' + caption)



def analyze_variance_standardization(per_storm: pd.DataFrame,
                                     output_dir: Path,
                                     mode: str,
                                     *,
                                     quantile_bins: int = 5,
                                     min_storms_per_stratum: int = 15,
                                     bootstrap_reps: int = 1500,
                                     seed: int = 1234,
                                     show: bool = False) -> pd.DataFrame:
    """Separate the *composition* effect of true-intensity variance from length.

    Each storm's r is left untouched. For each lead separately, define GLOBAL
    quintiles of within-storm Var(true_intensity). For a given available-sample
    length, estimate E[r | length, quintile], then reweight those means to the
    same pooled quintile proportions at EVERY length. This directly standardizes
    the distribution of observed intensity variance across x groups.

    It is a descriptive variance-adjustment, NOT a causal estimate or a new
    definition of Pearson correlation. Longer/shorter series can still differ
    in other ways, including finite-sample uncertainty and storm characteristics.
    """
    required = {'lead_h', 'x_hours', 'true_int_variance', 'pearson_r', 'storm_key'}
    missing = required - set(per_storm.columns)
    if missing:
        raise ValueError('Per-storm CSV is missing columns: ' + ', '.join(sorted(missing)))
    if not (2 <= quantile_bins <= 10 and min_storms_per_stratum >= 2
            and bootstrap_reps >= 100):
        raise ValueError('Require 2 <= quantile_bins <= 10, min >= 2, boots >= 100')
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = per_storm.copy()
    for col in ('lead_h', 'x_hours', 'true_int_variance', 'pearson_r'):
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=['lead_h', 'x_hours', 'true_int_variance', 'pearson_r'])
    data = data[(data.true_int_variance > 0) & data.pearson_r.between(-1, 1)]
    if data.duplicated(['lead_h', 'storm_key']).any():
        raise ValueError('Each storm should contribute once per lead; duplicate keys found')
    rng = np.random.default_rng(seed)
    results = []
    strata = []
    edges_rows = []
    references = {}
    for lead, lead_df in data.groupby('lead_h', sort=True):
        d = lead_df.copy()
        # Quantiles are based on pooled storm variance for THIS lead, not on each
        # x group separately (that would fail to adjust length-group composition).
        q, bins = pd.qcut(d.true_int_variance,
                          q=quantile_bins,
                          labels=False,
                          retbins=True,
                          duplicates='drop')
        k = len(bins) - 1
        if k < 2:
            raise ValueError(f'Control {lead}h: cannot form >=2 distinct variance groups')
        d['variance_stratum'] = q.astype(int)
        ref_counts = d.variance_stratum.value_counts().reindex(range(k), fill_value=0)
        weights = (ref_counts / ref_counts.sum()).to_numpy(float)
        references[lead] = weights
        for i in range(k):
            edges_rows.append(dict(lead_h=lead, stratum=i + 1,
                                   min_true_variance=bins[i],
                                   max_true_variance=bins[i+1],
                                   reference_weight=weights[i]))
        for x, g in d.groupby('x_hours', sort=True):
            raw = g.pearson_r.to_numpy(float)
            strata_arrays = []
            for idx in range(k):
                v = g.loc[g.variance_stratum == idx, 'pearson_r'].to_numpy(float)
                strata_arrays.append(v)
                strata.append(dict(lead_h=lead, x_hours=x, stratum=idx + 1,
                                   n_storms=len(v), mean_r=v.mean() if len(v) else np.nan,
                                   median_true_variance=float(g.loc[g.variance_stratum == idx,
                                                                     'true_int_variance'].median())
                                   if len(v) else np.nan))
            overlap = all(len(v) >= min_storms_per_stratum for v in strata_arrays)
            rec = dict(lead_h=lead, x_hours=x, n_storms=len(g),
                       raw_mean_r=float(raw.mean()),
                       min_stratum_n=min(map(len, strata_arrays)),
                       common_support=overlap,
                       adjusted_mean_r=np.nan, adjusted_ci_low=np.nan,
                       adjusted_ci_high=np.nan, raw_ci_low=np.nan, raw_ci_high=np.nan)
            if overlap:
                means = np.array([v.mean() for v in strata_arrays], dtype=float)
                rec['adjusted_mean_r'] = float(weights @ means)
                boot_adjust = np.zeros(bootstrap_reps, dtype=float)
                for j, values in enumerate(strata_arrays):
                    # Bootstrap storms WITHIN each variance stratum; reference
                    # weights remain fixed, so uncertainty is conditional on
                    # the pooled variance distribution.
                    n = len(values)
                    means_boot = np.empty(bootstrap_reps)
                    for start in range(0, bootstrap_reps, 100):
                        end = min(start + 100, bootstrap_reps)
                        idx = rng.integers(0, n, size=(end-start, n))
                        means_boot[start:end] = values[idx].mean(axis=1)
                    boot_adjust += weights[j] * means_boot
                rec['adjusted_ci_low'], rec['adjusted_ci_high'] = map(
                    float, np.quantile(boot_adjust, [0.025, 0.975]))
                raw_boot = np.empty(bootstrap_reps)
                for start in range(0, bootstrap_reps, 100):
                    end = min(start + 100, bootstrap_reps)
                    idx = rng.integers(0, len(raw), size=(end-start, len(raw)))
                    raw_boot[start:end] = raw[idx].mean(axis=1)
                rec['raw_ci_low'], rec['raw_ci_high'] = map(
                    float, np.quantile(raw_boot, [0.025, 0.975]))
            results.append(rec)

    result = pd.DataFrame(results).sort_values(['lead_h', 'x_hours'])
    strata_df = pd.DataFrame(strata)
    edges_df = pd.DataFrame(edges_rows)
    stem = f'Figure4d_Intensity_Variance_Adjusted_{mode}'
    result.to_csv(output_dir / f'{stem}_by_length.csv', index=False, float_format='%.9g')
    strata_df.to_csv(output_dir / f'{stem}_within_quintiles.csv', index=False, float_format='%.9g')
    edges_df.to_csv(output_dir / f'{stem}_reference_weights.csv', index=False, float_format='%.9g')

    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'font.size': 13, 'axes.labelsize': 14, 'pdf.fonttype': 42})
    leads = sorted(result.lead_h.unique())
    fig, axes = plt.subplots(1, len(leads), figsize=(7.2 * len(leads), 5.6),
                             sharey=True, layout='constrained')
    if len(leads) == 1:
        axes = [axes]
    summary = [
        'Figure 4d variance-adjusted (directly standardized) Pearson analysis',
        'Per-storm Pearson correlations are unchanged: only their aggregation is reweighted.',
        'Variance strata defined separately per lead from pooled within-storm true-intensity variances.',
        f'{quantile_bins} quantile bins; min {min_storms_per_stratum} storms per variance stratum at each x.',
        f'Stratified bootstrap {bootstrap_reps} replicates for pointwise 95% CI.',
        'No causal inference; effects of forecast sample count / other storm properties remain.',
        '',
    ]
    for ax, lead in zip(axes, leads):
        d = result[(result.lead_h == lead) & (result.common_support)].copy()
        all_lead = result[result.lead_h == lead]
        xall = all_lead.x_hours.to_numpy(float)
        ax.plot(xall, all_lead.raw_mean_r, color='#a5a5a5', lw=1.3,
                linestyle=':', label='Raw mean (all lengths)')
        if d.empty:
            ax.set_title(f'Control {int(lead)} h - no common variance support')
            continue
        x = d.x_hours.to_numpy(float)
        raw = d.raw_mean_r.to_numpy(float)
        adj = d.adjusted_mean_r.to_numpy(float)
        ax.plot(x, raw, color='#1f77b4', lw=2.2, marker='o', markersize=3.8,
                label='Original mean Pearson r')
        ax.plot(x, adj, color='#d95f02', lw=2.2, marker='s', markersize=3.7,
                label='Variance-standardized mean r')
        ax.fill_between(x, d.adjusted_ci_low.to_numpy(float),
                        d.adjusted_ci_high.to_numpy(float), color='#d95f02', alpha=.17,
                        label='Adjusted 95% bootstrap CI')
        ax.set_title(f'Control ({int(lead)} h)')
        ax.set_xlabel('Available forecast samples × 6 h' if mode == 'available'
                      else 'Cyclone lifetime (h)')
        ax.grid(True, linestyle=':', alpha=.5)
        if len(d) >= 3:
            # Descriptive slopes per 6h increase, not causal effects.
            raw_slope = np.polyfit(x / 6., raw, 1)[0]
            adj_slope = np.polyfit(x / 6., adj, 1)[0]
            relative = (1-abs(adj_slope/raw_slope))*100 if abs(raw_slope)>1e-12 else np.nan
            summary += [f'Control {int(lead)} h | common-support lengths: {len(d)}/{len(all_lead)}',
                        f'  Raw slope in r per 6-h sample: {raw_slope:+.5f}',
                        f'  Adjusted slope in r per 6-h sample: {adj_slope:+.5f}',
                        f'  Relative slope-magnitude reduction (descriptive): {relative:+.1f}%',
                        f'  Mean raw r on common support: {raw.mean():.4f}',
                        f'  Mean variance-adjusted r on common support: {adj.mean():.4f}', '']
            print(f'[ADJUSTMENT] Control {lead}h: raw slope={raw_slope:+.5f}, '
                  f'adjusted slope={adj_slope:+.5f}, common length groups={len(d)}')
        else:
            summary.append(f'Control {int(lead)} h: fewer than 3 common-support x groups.')
    axes[0].set_ylabel('Mean per-storm Pearson r')
    axes[0].legend(frameon=False, fontsize=10, loc='lower right')
    for ax in axes:
        ax.set_ylim(-.05, 1.05)
    fig.savefig(output_dir / f'{stem}.png', dpi=250, bbox_inches='tight')
    fig.savefig(output_dir / f'{stem}.pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    (output_dir / f'{stem}_summary.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print('[VARIANCE-ADJUSTED] Results saved in:', output_dir)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_DIR, help='Folder with intensity_control_31/34.csv')
    parser.add_argument('--control-6', type=Path, default=None, help='Exact intensity Control 6h CSV')
    parser.add_argument('--control-24', type=Path, default=None, help='Exact intensity Control 24h CSV')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--length-mode', choices=['available', 'lifetime'], default='available',
                        help='Available paired sample count (legacy) or independent FULL storm lifetime')
    parser.add_argument('--lifetime-csv', type=Path, default=None,
                        help='Required with --length-mode lifetime: columns storm_key, lifetime_hours')
    parser.add_argument('--min-lifetime-hours', type=float, default=48.,
                        help='Filter independently recorded lifetimes below this value (lifetime mode)')
    parser.add_argument('--min-pairs', type=int, default=8,
                        help='Minimum number of paired verification steps per storm')
    parser.add_argument('--min-storms', type=int, default=8,
                        help='Minimum number of eligible storms per x position')
    parser.add_argument('--bootstraps', type=int, default=3000,
                        help='Number of whole-storm bootstrap resamples')
    parser.add_argument('--random-pairings', type=int, default=5000,
                        help='Number of random whole-storm derangements for each x position')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--show', action='store_true', help='Open matplotlib display after saving')
    parser.add_argument('--variance-bins', type=int, default=5, help='Number of quantiles for signal-variance adjustment')
    parser.add_argument('--variance-min-storms', type=int, default=15, help='Minimum storms in each variance quantile at each duration')
    parser.add_argument('--variance-bootstraps', type=int, default=1500, help='Stratified bootstrap replications')
    args = parser.parse_args(argv)

    if args.length_mode == 'lifetime' and args.lifetime_csv is None:
        parser.error('--length-mode lifetime requires --lifetime-csv /path/to/full_storm_lifetimes.csv')
    if args.min_pairs < 3 or args.min_storms < 2 or args.bootstraps < 100 or args.random_pairings < 100:
        parser.error('Use min-pairs >=3, min-storms >=2, bootstraps >=100, random-pairings >=100')
    paths = {lead: file_for_lead(args.input_dir.expanduser(), lead, getattr(args, f'control_{lead}'))
             for lead in (6, 24)}
    lifetimes = read_lifetimes(args.lifetime_csv.expanduser()) if args.lifetime_csv else None
    rows = []
    variability_rows = []
    for lead in (6, 24):
        d = read_control(paths[lead], lead)
        groups, _ = make_groups(d, lead, args.length_mode, lifetimes, args.min_pairs, args.min_lifetime_hours)
        variability_rows.append(compute_variability(groups, lead, args.min_storms, args.length_mode))
        # Different deterministic seed per lead avoids coupling the null distributions.
        rng = np.random.default_rng(args.seed + lead * 1009)
        rows.extend(analyze_groups(groups, lead, rng, args.bootstraps, args.random_pairings,
                                   args.min_storms, args.length_mode))
    frame = pd.DataFrame(rows)
    out = args.output_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stem = plot_figure(frame, out, args.length_mode, args.show)
    frame.to_csv(out / (stem + '_stats.csv'), index=False, float_format='%.8g')
    # Additional research check: we preserve the original figure and write a
    # separate variance diagnostic calculated on the SAME included storms.
    storm_variability = pd.concat(variability_rows, ignore_index=True)
    length_variability = summarize_variability(storm_variability)
    vstem = f'Figure4d_Intensity_Variability_{args.length_mode}'
    storm_variability.to_csv(out / (vstem + '_per_storm.csv'), index=False,
                            float_format='%.8g')
    length_variability.to_csv(out / (vstem + '_by_length.csv'), index=False,
                             float_format='%.8g')
    # Scientific consistency: mean Pearson correlations must match Figure 4d.
    merged = frame.merge(length_variability, on=['lead_h', 'x_hours'],
                         suffixes=('_main','_diagnostic'), validate='one_to_one')
    if len(merged) != len(frame) or not np.allclose(
        merged['mean_r_main'], merged['mean_r_diagnostic'], atol=1e-10
    ):
        raise RuntimeError('Diagnostic sample mismatch with Figure 4d Pearson r')
    plot_variability(length_variability, out, args.length_mode, args.show)
    write_variability_summary(storm_variability, length_variability,
                              out, args.length_mode)
    analyze_variance_standardization(
        storm_variability, out, args.length_mode,
        quantile_bins=args.variance_bins,
        min_storms_per_stratum=args.variance_min_storms,
        bootstrap_reps=args.variance_bootstraps,
        seed=args.seed + 101, show=args.show)
    write_caption(out, args.length_mode, stem)
    (out / (stem + '_sources.txt')).write_text(
        '\n'.join([f'Control {lead}h: {paths[lead]}' for lead in (6, 24)]) +
        f'\nlength_mode={args.length_mode}\nmin_pairs={args.min_pairs}' +
        f'\nlifetime_csv={args.lifetime_csv}\n', encoding='utf-8')
    print('[SAVED] ' + str(out / (stem + '_stats.csv')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
