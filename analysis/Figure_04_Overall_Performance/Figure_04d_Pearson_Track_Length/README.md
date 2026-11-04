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
