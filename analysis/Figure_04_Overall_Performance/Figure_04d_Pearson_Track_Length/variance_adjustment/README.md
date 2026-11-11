# Figure 4d — True-intensity variance adjustment (8 Nov 2026)

Checks how much of the rise in mean per-storm Pearson r with available sample count is associated with storms that have more observed-intensity variance.

Pearson r is invariant to rescaling, so z-scoring intensity does not remove this effect. Instead the mean r at each x is **directly standardized** for the distribution of within-storm `Var(true_int)`:

1. For each lead separately, split storms into 5 pooled quantile groups of `Var(true_int)` (same cut points at all x).
2. At each x and group, take the mean per-storm r.
3. Re-weight with fixed pooled group weights at every x. An x is adjusted only if each group has ≥ 15 storms.
4. 95% CIs: bootstrap of storms within each group (1500 replicates).

The x axis stays **available forecast samples × 6 h**, not cyclone lifetime. Per-storm r values are unchanged; only their aggregation is reweighted.

## Files

- `figure4d_control_intensity_var_adjusted.py` — full pipeline from the raw intensity CSVs: original Figure 4d, variance diagnostic, and the variance-adjusted analysis.
- `recheck_variance_effect.py` — runs only the adjustment, from the per-storm CSV written by `../figure4d_control_intensity.py`.
- `results/` — outputs from the real Control data run (figure + summary).

## Run

From the repository root, either the full pipeline:

```bash
python analysis/Figure_04_Overall_Performance/Figure_04d_Pearson_Track_Length/variance_adjustment/figure4d_control_intensity_var_adjusted.py \
  --input-dir data/intensity_space_files/files/new --output-dir outputs/figure_04d
```

or only the adjustment from an existing per-storm CSV:

```bash
python analysis/Figure_04_Overall_Performance/Figure_04d_Pearson_Track_Length/variance_adjustment/recheck_variance_effect.py \
  --input outputs/figure_04d/Figure4d_Intensity_Variability_available_per_storm.csv \
  --output-dir outputs/figure_04d_variance_adjusted
```

## Result (real data)

Descriptive slope of mean r per extra 6-hourly sample, over common-support x positions:

| Control | Raw slope | Adjusted slope | Reduction |
| --- | ---: | ---: | ---: |
| 6 h | 0.00692 | 0.00144 | 79.2% |
| 24 h | 0.01025 | 0.00215 | 79.0% |

These are descriptive linear-fit changes, not causal estimates or confidence intervals. The raw curves reproduce the original Figure 4d statistics to within 5e-9.
