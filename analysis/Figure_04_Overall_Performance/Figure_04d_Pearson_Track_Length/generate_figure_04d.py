"""Mean per-storm Pearson r vs storm-track duration (hours), AMS figure.


Run this ENTIRE script as a Jupyter cell, or via `%run /path/to/file.py`.

No argparse is used.


Scientific definitions:

* Every r is computed across time within ONE storm; lines are means across storms.

* Model shading: 95% storm-level bootstrap confidence intervals of the mean.

* Gray curves: random whole-storm pairing null, computed separately for

  Control 6h and Control 24h (95% intervals of permuted group MEAN r).

* Filled/open BLACK MSL-only markers: KS comparison with Control 6h

  at that duration (BH-FDR significant / not significant). The KS statistic

  compares marginal distributions, not whether one model is better.

* KS p-value mode defaults to paired label-swap permutation because the SAME

  storms are evaluated by Control and MSL-only; classical two-sample KS

  p-values (which presume independent samples) are also saved for reference.

* Random-pairing significance is calculated and saved separately in the CSV;

  marker fill does NOT indicate this test.

* One time step corresponds to six hours. The earliest duration is 48 h.


NOTE: "not significant" does NOT mean models are equivalent.

"""


from pathlib import Path
import os

import re


import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import numpy as np

import pandas as pd

from scipy.stats import ks_2samp


# =====================================================================

# 1. USER CONFIGURATION — EDIT HERE, NO COMMAND-LINE ARGUMENTS

# =====================================================================

BASE_DIR = Path(os.environ.get('CYCLONE_ANALYSIS_DIR', './data')).expanduser()

INPUT_DIR = Path(os.environ.get('CYCLONE_INTENSITY_INPUT_DIR', str(BASE_DIR / 'intensity_space_files' / 'files' / 'new'))).expanduser()

OUTPUT_DIR = Path(os.environ.get('CYCLONE_OUTPUT_DIR', './outputs/figure_04d')).expanduser()


CONTROL_6H_FILE = 'intensity_control_31.csv'

CONTROL_24H_FILE = None  # Auto-discover experiment 34; or set exact filename.

MSL_ONLY_FILE = None     # Auto-discover version_156_157_158_159.

MSL_ONLY_GLOB = 'intensity_*version_156_157_158_159*.csv'


RANDOM_SEED = 42

N_BOOTSTRAP = 3000

N_RANDOM_PAIRINGS = 5000

N_KS_PERMUTATIONS = 1999

KS_P_MODE = 'paired'   # 'paired' (recommended) or 'classic' (original ks_2samp p)

KS_FDR = True          # Apply BH-FDR across tested 6h track lengths.

MIN_OVERLAP = 8

MIN_STORMS_PER_LENGTH = 8

MAX_MATRIX_N = 3000

MAX_LEN_BY_LEAD = {6: 32, 24: 29}  # Same truncation as original analysis.

HOURS_PER_STEP = 6

Y_AXIS_LOWER = -0.45  # Extend below 0 to give the legend room.

OUTPUT_STEM = 'Pearson_vs_Track_Length_KS_Bootstrap_Random'


MODEL_KEYS = [(6, 'Control'), (6, 'MSL Only'), (24, 'Control')]

PLOT_STYLE = {

    (6, 'Control'): {'color': 'tab:blue', 'linestyle': '-', 'label': 'Control (6h)'},

    (6, 'MSL Only'): {'color': 'black', 'linestyle': '-', 'label': 'MSL Only (6h)'},

    (24, 'Control'): {'color': 'tab:blue', 'linestyle': '--', 'label': 'Control (24h)'},

}


plt.rcParams.update({

    'font.family': 'serif',

    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],

    'figure.dpi': 300,

    'font.size': 16,

    'axes.labelsize': 18,

    'xtick.labelsize': 15,

    'ytick.labelsize': 15,

    'legend.fontsize': 12.5,

    'legend.title_fontsize': 14,

    'lines.linewidth': 2.5,

    'pdf.fonttype': 42,

})


# =====================================================================

# 2. SAFE SOURCE-FILE RESOLUTION

# =====================================================================

def _resolve_user_path(path):

    p = Path(path).expanduser()

    return p if p.is_absolute() else INPUT_DIR / p


def find_control_file(lead, name=None):

    if name is not None:

        path = _resolve_user_path(name)

        if not path.is_file():

            raise FileNotFoundError(f'Control {lead}h not found: {path}')

        return path


    experiment_id = {6: '31', 24: '34'}[lead]

    regex = re.compile(

        rf'^intensity_(?:control|all_cyclone_data_(?:full|combined))_{experiment_id}(?:_|\.csv$)',

        re.IGNORECASE,

    )

    matches = [p for p in sorted(INPUT_DIR.glob('intensity_*.csv'))

               if regex.match(p.name) and not re.search(

                   r'msl[_-]?only|leakage|version_156_157_158_159', p.name, re.I)]

    if len(matches) == 1:

        return matches[0]

    if len(matches) > 1:

        raise ValueError(

            f'Multiple Control {lead}h candidate files:\n' +

            '\n'.join(f'  {p.name}' for p in matches) +

            f'\nSet CONTROL_{lead}H_FILE to the correct exact filename.'

        )

    nearby = [p for p in INPUT_DIR.glob('intensity_*.csv')

              if re.search(rf'(?<!\d){experiment_id}(?!\d)', p.name)]

    raise FileNotFoundError(

        f'Could not resolve Control {lead}h in {INPUT_DIR}.\n' +

        'Nearby file names:\n' +

        ('\n'.join(f'  {p.name}' for p in sorted(nearby)[:30]) or '  (none)') +

        f'\nSet CONTROL_{lead}H_FILE explicitly.'

    )


def resolve_files():

    if not INPUT_DIR.is_dir():

        raise FileNotFoundError(f'Missing input folder: {INPUT_DIR}')

    if MSL_ONLY_FILE is not None:

        msl = _resolve_user_path(MSL_ONLY_FILE)

    else:

        matches = sorted(INPUT_DIR.glob(MSL_ONLY_GLOB))

        if len(matches) != 1:

            raise ValueError(

                f'Expected one MSL-only file, found {len(matches)}:\n' +

                ('\n'.join(f'  {p.name}' for p in matches) or '  (none)') +

                '\nSet MSL_ONLY_FILE to an exact CSV filename.'

            )

        msl = matches[0]


    files = {

        (6, 'Control'): find_control_file(6, CONTROL_6H_FILE),

        (6, 'MSL Only'): msl,

        (24, 'Control'): find_control_file(24, CONTROL_24H_FILE),

    }

    for key, p in files.items():

        if not p.is_file():

            raise FileNotFoundError(f'Missing file for {key}: {p}')

        print(f'  {key[1]} ({key[0]}h): {p.name}', flush=True)

    if len(set(files.values())) != len(files):

        raise ValueError('A single CSV was assigned to multiple models.')

    return files


# =====================================================================

# 3. STORM-LEVEL DATA

# =====================================================================

def clean_storm_keys(series):

    return (series.astype('string').str.strip()

            .str.strip(chr(34) + chr(39))

            .str.replace(r'\.(npy|npz)$', '', regex=True))


def load_tracks_by_length(filename, max_len):

    required = ['storm_key', 'step_index', 'true_int', 'pred_int']

    df = pd.read_csv(filename)

    df.columns = df.columns.str.strip()

    missing = sorted(set(required) - set(df.columns))

    if missing:

        raise ValueError(f'{filename}: missing columns {missing}')

    df = df[required].copy()

    df['storm_key'] = clean_storm_keys(df['storm_key'])

    for col in ['step_index', 'true_int', 'pred_int']:

        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['storm_key', 'step_index'])


    duplicates = df.duplicated(['storm_key', 'step_index']).sum()

    if duplicates:

        raise ValueError(f'{filename.name}: {duplicates} duplicated storm/step rows')


    grouped = {}

    skipped = 0

    for key, storm in df.groupby('storm_key', sort=False):

        storm = storm.sort_values('step_index').iloc[:max_len]

        t = storm['true_int'].to_numpy(dtype=float)

        p = storm['pred_int'].to_numpy(dtype=float)

        n = len(t)

        if (n < MIN_OVERLAP or not np.isfinite(t).all()

                or not np.isfinite(p).all()

                or np.std(t) <= 1e-12 or np.std(p) <= 1e-12):

            skipped += 1

            continue

        grouped.setdefault(n, []).append((

            str(key), t, p, storm['step_index'].to_numpy(dtype=float)

        ))


    result = {}

    for length, entries in grouped.items():

        result[length] = {

            'keys': np.array([e[0] for e in entries]),

            'true': np.stack([e[1] for e in entries]),

            'pred': np.stack([e[2] for e in entries]),

            'steps': np.stack([e[3] for e in entries]),

        }

    print(f'    usable={sum(len(v["keys"]) for v in result.values()):,}; '

          f'skipped={skipped:,}; lengths={len(result)}', flush=True)

    return result


def match_6h_groups(control, msl):

    """Use the SAME storms in both 6h series, for a fair KS comparison."""

    aligned = {(6, 'Control'): {}, (6, 'MSL Only'): {}}

    for L in sorted(set(control) & set(msl)):

        a, b = control[L], msl[L]

        a_map = {k: i for i, k in enumerate(a['keys'])}

        b_map = {k: i for i, k in enumerate(b['keys'])}

        common = sorted(

            k for k in (a_map.keys() & b_map.keys())

            if np.array_equal(a['steps'][a_map[k]], b['steps'][b_map[k]])

        )

        if len(common) < MIN_STORMS_PER_LENGTH:

            continue

        for key, group, mapping in [

            ((6, 'Control'), a, a_map),

            ((6, 'MSL Only'), b, b_map),

        ]:

            rows = np.array([mapping[k] for k in common], dtype=int)

            aligned[key][L] = {

                'keys': np.asarray(common),

                'true': group['true'][rows],

                'pred': group['pred'][rows],

                'steps': group['steps'][rows],

            }

    return aligned


def normalise_rows(x):

    x = x - x.mean(axis=1, keepdims=True)

    n = np.linalg.norm(x, axis=1, keepdims=True)

    return x / n


def storm_correlations(group):

    a = normalise_rows(group['true'])

    b = normalise_rows(group['pred'])

    return np.clip(np.einsum('ij,ij->i', a, b), -1, 1), a, b


# =====================================================================

# 4. STATISTICS

# =====================================================================

def bootstrap_mean_ci(values, rng):

    n = len(values)

    result = np.empty(N_BOOTSTRAP, dtype=float)

    for start in range(0, N_BOOTSTRAP, 128):

        count = min(128, N_BOOTSTRAP - start)

        indices = rng.integers(0, n, size=(count, n))

        result[start:start + count] = values[indices].mean(axis=1)

    return np.quantile(result, [0.025, 0.975])


def random_pairing_means(true_u, pred_u, rng):

    n = len(true_u)

    result = np.empty(N_RANDOM_PAIRINGS, dtype=float)

    indices = np.arange(n)

    if n <= MAX_MATRIX_N:

        all_pairs = np.clip(true_u @ pred_u.T, -1, 1)

        for b in range(N_RANDOM_PAIRINGS):

            result[b] = all_pairs[indices, rng.permutation(n)].mean()

    else:

        for b in range(N_RANDOM_PAIRINGS):

            result[b] = np.einsum(

                'ij,ij->', true_u, pred_u[rng.permutation(n)]

            ) / n

    return result


def analyze_model_group(group, rng):

    r, true_u, pred_u = storm_correlations(group)

    lo, hi = bootstrap_mean_ci(r, rng)

    null = random_pairing_means(true_u, pred_u, rng)

    null_lo, null_hi = np.quantile(null, [0.025, 0.975])

    observed = r.mean()

    p = (1 + np.count_nonzero(null >= observed - 1e-12)) / (N_RANDOM_PAIRINGS + 1)

    return {

        'n_storms': len(r),

        'mean_r': float(observed),

        'bootstrap_ci_low': float(lo),

        'bootstrap_ci_high': float(hi),

        'random_mean_r': float(null.mean()),

        'random_ci_low': float(null_lo),

        'random_ci_high': float(null_hi),

        'p_random_pairing': float(p),

    }


def paired_permutation_ks_pvalue(control_r, msl_r, rng):

    """Test KS distance by swapping labels independently WITHIN each storm.


    The statistic is the ordinary two-sample KS distance. Paired label swaps

    preserve the pairing and are valid under within-storm label exchangeability.

    """

    observed = ks_2samp(control_r, msl_r).statistic

    n = len(control_r)

    extreme = 0

    for _ in range(N_KS_PERMUTATIONS):

        swap = rng.random(n) < 0.5

        a = np.where(swap, msl_r, control_r)

        b = np.where(swap, control_r, msl_r)

        d = ks_2samp(a, b).statistic

        extreme += (d >= observed - 1e-12)

    return (1 + extreme) / (N_KS_PERMUTATIONS + 1)


def adjust_bh(p):

    p = np.asarray(p, dtype=float)

    order = np.argsort(p)

    rank_adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)

    backward = np.minimum.accumulate(rank_adjusted[::-1])[::-1]

    result = np.empty(len(p), dtype=float)

    result[order] = np.clip(backward, 0, 1)

    return result


def analyze(files):

    rng = np.random.default_rng(RANDOM_SEED)

    all_groups = {}

    for key in MODEL_KEYS:

        lead, model = key

        print(f'\nLoading {model} ({lead}h)', flush=True)

        all_groups[key] = load_tracks_by_length(files[key], MAX_LEN_BY_LEAD[lead])


    matched = match_6h_groups(all_groups[(6, 'Control')], all_groups[(6, 'MSL Only')])

    all_groups[(6, 'Control')] = matched[(6, 'Control')]

    all_groups[(6, 'MSL Only')] = matched[(6, 'MSL Only')]

    print('\n6h Control / MSL Only restricted to their matched storm keys.', flush=True)


    rows = []

    r_by_model_and_length = {}

    for key in MODEL_KEYS:

        lead, model = key

        print(f'\nAnalyzing {model} ({lead}h)', flush=True)

        for length, group in sorted(all_groups[key].items()):

            if len(group['keys']) < MIN_STORMS_PER_LENGTH:

                continue

            r, _, _ = storm_correlations(group)

            r_by_model_and_length[(key, length)] = r

            result = analyze_model_group(group, rng)

            rows.append({

                'lead_h': lead, 'model': model,

                'length_steps': length,

                'track_hours': length * HOURS_PER_STEP,

                'source_csv': files[key].name,

                **result,

            })

            print(f'    L={length:2d} ({length * HOURS_PER_STEP:3d} h), '

                  f'n={len(r):5d}, r={result["mean_r"]:.3f}, '

                  f'p_random={result["p_random_pairing"]:.4g}', flush=True)


    if not rows:

        raise RuntimeError('No track groups satisfy MIN_STORMS_PER_LENGTH')

    table = pd.DataFrame(rows)

    table['p_random_pairing_fdr'] = adjust_bh(table['p_random_pairing'])

    table['random_pairing_significant_fdr'] = table['p_random_pairing_fdr'] < 0.05


    ks_rows = []

    for length in sorted(all_groups[(6, 'Control')]):

        key_a, key_b = ((6, 'Control'), length), ((6, 'MSL Only'), length)

        if key_a not in r_by_model_and_length or key_b not in r_by_model_and_length:

            continue

        r_control = r_by_model_and_length[key_a]

        r_msl = r_by_model_and_length[key_b]

        assert len(r_control) == len(r_msl)

        ks_result = ks_2samp(r_control, r_msl, alternative='two-sided')

        if KS_P_MODE == 'paired':

            p_used = paired_permutation_ks_pvalue(r_control, r_msl, rng)

        elif KS_P_MODE == 'classic':

            p_used = float(ks_result.pvalue)

        else:

            raise ValueError('KS_P_MODE must be paired or classic')

        ks_rows.append({

            'length_steps': length,

            'ks_D_control_vs_msl': float(ks_result.statistic),

            'ks_p_classic': float(ks_result.pvalue),

            'ks_p_used': float(p_used),

            'ks_p_method': KS_P_MODE,

            'ks_n_pairs': len(r_control),

        })

    if not ks_rows:

        raise RuntimeError('No matched 6h groups for KS comparisons')

    ks = pd.DataFrame(ks_rows)

    ks['ks_p_fdr'] = adjust_bh(ks['ks_p_used']) if KS_FDR else ks['ks_p_used']

    ks['ks_significant_0p05'] = ks['ks_p_fdr'] < 0.05

    table = table.merge(ks, on='length_steps', how='left', validate='many_to_one')

    # KS is only defined for the 6h Control-versus-MSL comparison.

    mask_24h = table['lead_h'] != 6

    ks_numeric = ['ks_D_control_vs_msl', 'ks_p_classic', 'ks_p_used',

                  'ks_p_fdr', 'ks_n_pairs']

    table.loc[mask_24h, ks_numeric] = np.nan

    table.loc[mask_24h, 'ks_p_method'] = None

    table['ks_significant_0p05'] = table['ks_significant_0p05'].astype('boolean')

    table.loc[mask_24h, 'ks_significant_0p05'] = pd.NA

    return table.sort_values(['lead_h', 'model', 'length_steps']).reset_index(drop=True), ks


# =====================================================================

# 5. PLOT

# =====================================================================

def plot_results(table, stem):

    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')


    # Two control-derived random-pairing baselines and their null intervals.

    for lead, color, ls in [(6, '#a8a8a8', '--'), (24, '#777777', ':')]:

        subset = table[(table['lead_h'] == lead) & (table['model'] == 'Control')]

        subset = subset.sort_values('track_hours')

        if subset.empty:

            continue

        x = subset['track_hours'].to_numpy(float)

        ax.fill_between(x, subset['random_ci_low'].to_numpy(float),

                        subset['random_ci_high'].to_numpy(float),

                        color=color, alpha=0.18, linewidth=0, zorder=1)

        ax.plot(x, subset['random_mean_r'].to_numpy(float),

                color=color, linestyle=ls, lw=2, zorder=2)


    for key in MODEL_KEYS:

        lead, model = key

        subset = table[(table['lead_h'] == lead) & (table['model'] == model)]

        subset = subset.sort_values('track_hours')

        if subset.empty:

            continue

        style = PLOT_STYLE[key]

        x = subset['track_hours'].to_numpy(float)

        r = subset['mean_r'].to_numpy(float)

        ax.fill_between(x, subset['bootstrap_ci_low'].to_numpy(float),

                        subset['bootstrap_ci_high'].to_numpy(float),

                        color=style['color'], alpha=0.18, linewidth=0, zorder=3)

        ax.plot(x, r, ls=style['linestyle'], color=style['color'], lw=2.4, zorder=4)


        if key == (6, 'MSL Only'):

            # ONLY the MSL 6h markers encode KS comparison with Control 6h.

            sig = subset['ks_significant_0p05'].fillna(False).to_numpy(bool)

            ax.scatter(x[sig], r[sig], s=55, facecolors=style['color'],

                       edgecolors=style['color'], zorder=5)

            ax.scatter(x[~sig], r[~sig], s=55, facecolors='none',

                       edgecolors=style['color'], linewidths=1.6, zorder=5)

        else:

            # These are reference curves; filled markers have NO KS meaning.

            ax.scatter(x, r, s=55, facecolors=style['color'],

                       edgecolors=style['color'], zorder=5)


    handles = [

        Line2D([], [], color='tab:blue', marker='o', label='Control (6h)', lw=2.4),

        Line2D([], [], color='black', marker='o', label='MSL Only (6h)', lw=2.4),

        Line2D([], [], color='tab:blue', marker='o', ls='--',

               label='Control (24h)', lw=2.4),

        Line2D([], [], color='#a8a8a8', ls='--',

               label='Random Pairing (6h)', lw=2),

        Line2D([], [], color='#777777', ls=':',

               label='Random Pairing (24h)', lw=2),

    ]

    ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.99, 0.02),

              frameon=False)


    ax.set_xlabel('Track Length (h)')

    ax.set_ylabel(r'Mean Pearson $r$')

    ax.set_ylim(Y_AXIS_LOWER, 1.05)

    ax.set_xlim(MIN_OVERLAP * HOURS_PER_STEP - 7,

MAX_LEN_BY_LEAD[6] * HOURS_PER_STEP + 7)

    ax.set_xticks(np.arange(MIN_OVERLAP * HOURS_PER_STEP,

MAX_LEN_BY_LEAD[6] * HOURS_PER_STEP + 1, 12))

    ax.grid(True, linestyle=':', alpha=0.6)

    for spine in ax.spines.values():

        spine.set_color('lightgray')

        spine.set_linewidth(1.5)


    # No text annotation in lower-left. Significance definition belongs in caption.

    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')

    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')

    plt.show()

    plt.close(fig)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if N_BOOTSTRAP < 100 or N_RANDOM_PAIRINGS < 99 or N_KS_PERMUTATIONS < 99:

        raise ValueError('Too few repetitions for stable intervals / p-values.')

    print('Resolved model input files:', flush=True)

    files = resolve_files()

    table, ks_table = analyze(files)

    stem = OUTPUT_DIR / OUTPUT_STEM

    table.to_csv(stem.with_suffix('.csv'), index=False)

    ks_table.to_csv(OUTPUT_DIR / f'{OUTPUT_STEM}_KS_comparison.csv', index=False)

    plot_results(table, stem)

    print('\nSaved:', stem.with_suffix('.pdf'), stem.with_suffix('.png'),

          stem.with_suffix('.csv'), sep='\n  ')

    print(f'KS comparisons significantly DIFFERENT after {"BH-FDR" if KS_FDR else "raw p"}: '

          f'{ks_table["ks_significant_0p05"].sum()}/{len(ks_table)} lengths.')

    print('Marker fill is KS significance for MSL Only (6h) vs Control (6h), '

          'NOT significance vs random pairing.')

    return table, ks_table


if __name__ == '__main__':

    results_df, ks_df = main()


