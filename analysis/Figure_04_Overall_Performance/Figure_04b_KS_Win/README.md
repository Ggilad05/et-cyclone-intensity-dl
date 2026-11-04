# Figure 4b — Pairwise KS distances and winning ratios

Standalone implementation from the November 4 analysis. It compares **forecast-step-matched pairs**: IFS comparisons on 2010, all non-IFS pairs on all available years. It also saves the pairwise audit CSV.

## Run

```bash
CYCLONE_ANALYSIS_DIR="$PWD/data" IFS_2010_DIR="$PWD/data/ifs_2010" \
CYCLONE_OUTPUT_DIR="$PWD/outputs/figure_04b" \
python analysis/Figure_04_Overall_Performance/Figure_04b_KS_Win/generate_figure_04b.py
```

The input layout expected under `CYCLONE_ANALYSIS_DIR` is `updated_data/control/`, `updated_data/leakage/`, `updated_data/msl_only/`, and `updated_data/simple_nn/`. Adjust constants if necessary.

This was converted from a user-supplied rendered Markdown code block into executable Python; the numerical method is unchanged. Only personal absolute paths were replaced by environment-based locations.

**Caution:** the reported `scipy.stats.ks_2samp` p-values assume independent observations and do not correct for multiple time steps per storm. The KS distances themselves are descriptive.
