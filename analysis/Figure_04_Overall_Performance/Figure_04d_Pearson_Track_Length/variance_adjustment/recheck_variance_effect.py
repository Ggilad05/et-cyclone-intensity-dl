#!/usr/bin/env python3
"""Recalculate Figure 4d correlation after holding the distribution of true-intensity variance fixed.

Runs from the per-storm CSV already produced by figure4d_control_intensity.
No original model/track CSVs required.
"""
from pathlib import Path
import argparse
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure4d_control_intensity_var_adjusted import analyze_variance_standardization

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=Path('outputs/figure_04d/Figure4d_Intensity_Variability_available_per_storm.csv'))
parser.add_argument('--output-dir', type=Path, default=Path('outputs/figure_04d_variance_adjusted'))
parser.add_argument('--mode', choices=['available','lifetime'], default='available')
parser.add_argument('--bins', type=int, default=5)
parser.add_argument('--min-storms', type=int, default=15)
parser.add_argument('--bootstraps', type=int, default=1500)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--show', action='store_true')
args = parser.parse_args()
source=args.input.expanduser()
if not source.is_file():
    raise SystemExit(f'Missing per-storm CSV: {source}\nUse --input /path/to/Figure4d_Intensity_Variability_available_per_storm.csv')
data=pd.read_csv(source)
found=set(data['length_mode'].dropna().astype(str).unique()) if 'length_mode' in data else set()
if found and found != {args.mode}:
    raise SystemExit(f'Source length_mode={found}, but --mode={args.mode}; use consistent mode.')
analyze_variance_standardization(data, args.output_dir.expanduser(), args.mode,
                                quantile_bins=args.bins,
                                min_storms_per_stratum=args.min_storms,
                                bootstrap_reps=args.bootstraps,
                                seed=args.seed,
                                show=args.show)
