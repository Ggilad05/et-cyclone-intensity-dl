# Figure 4d — Pearson correlation vs track duration

For each storm and lead time, compute the correlation between predicted and observed intensity time series; average these coefficients by **available forecast-series length**. Includes storm bootstrap intervals, random whole-storm-pairing null, and a paired label-swap KS comparison (Control 6h vs MSL-only 6h) with BH-FDR.

## Run

```bash
CYCLONE_INTENSITY_INPUT_DIR="$PWD/data/intensity_space_files/files/new" \
CYCLONE_OUTPUT_DIR="$PWD/outputs/figure_04d" \
python analysis/Figure_04_Overall_Performance/Figure_04d_Pearson_Track_Length/generate_figure_04d.py
```

Expected inputs: `intensity_control_31.csv`, a uniquely detectable Control 24h experiment-34 CSV, and a uniquely detectable MSL-only 6h CSV. Each needs `storm_key, step_index, true_int, pred_int`. Set `CONTROL_24H_FILE` and `MSL_ONLY_FILE` inside the script when auto-discovery is ambiguous.

**Interpretation:** `track_hours = 6 * number of available forecast steps`, truncated to at most 32 steps for 6h and 29 for 24h. This is **not necessarily the cyclone's full physical lifetime**. The script makes all Control/MSL-only 6h comparisons on identical storm/step sequences. The gray bands are permutation null intervals; model bands are storm bootstrap 95% CIs.

Only local personal output paths were removed from the user-supplied source. The calculations and statistical method have not been intentionally changed.

## Updated scripts (7–8 Nov 2026)

- `figure4d_control_intensity.py` — Control-only intensity version (6 h and 24 h + random-pairing baselines; no MSL-only curve). The x axis is labelled **Available forecast samples × 6 h**. `--length-mode lifetime --lifetime-csv <csv>` uses independently measured track lifetimes instead. Each run also writes a variance diagnostic (`Figure4d_Intensity_Variability_available_*`): per-storm `Var(true_int)`, error variance and their ratio.
- `variance_diagnostic/` — summary of that diagnostic on the real data (descriptive Spearman associations, e.g. 6 h: x vs `Var(true)` +0.40, `Var(true)` vs r +0.84).
- `variance_adjustment/` — direct standardization of mean r by true-intensity variance quintiles; see its README. On the real data the slope of r vs sample count drops by ~79% (6 h and 24 h). Descriptive, not causal.

```bash
python analysis/Figure_04_Overall_Performance/Figure_04d_Pearson_Track_Length/figure4d_control_intensity.py \
  --input-dir data/intensity_space_files/files/new --output-dir outputs/figure_04d
```

`generate_figure_04d.py` (Nov 4) is kept for provenance.
