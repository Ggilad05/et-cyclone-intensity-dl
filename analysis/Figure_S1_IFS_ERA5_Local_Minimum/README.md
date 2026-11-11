# Figure S1 — IFS vs ERA5 local MSL minimum (single cyclone, four times)

ERA5 (solid blue isobars, blue ×) and operational IFS (dashed red isobars, red ×) around one summer North Pacific cyclone, with shading for IFS − ERA5 MSLP (Pa) on a common color scale.

## Identified case

- Storm: **`JJA_2010_NH_neg_135`** — the same cyclone as in the original notebook figure, identified manually from the original plotting order. Do not substitute a similar storm.
- Raw NPY indices **4, 5, 6, 7** (i.e. `np.load(...)[4:]`, steps 0–3).
- Valid times: (a) 15 Jul 2010 18 UTC, (b) 16 Jul 00 UTC, (c) 16 Jul 06 UTC, (d) 16 Jul 12 UTC. These are valid times, not lead times.

## Files

| File | Purpose |
| --- | --- |
| `figure_s1_paper_separate_panels.py` | Publication figure: four separate panel PDFs/PNGs (a–d, legend inside each), a 2×2 overview, and one shared colorbar. Also writes a CSV with ERA5 local-minimum and tracked-center pressures. |
| `figure_s1_original_order_first100.py` | Identification tool: reproduces the old notebook's unsorted `os.listdir` order and builds a contact sheet of the first 100 frames, using the original ±8 grid-cell square minimum search. |
| `FigureS1_Overleaf_4_subfigures.tex` | Overleaf snippet arranging the four panels and colorbar, with the caption. |

## Run

From the repository root:

```bash
python analysis/Figure_S1_IFS_ERA5_Local_Minimum/figure_s1_paper_separate_panels.py \
  --era5-dir data/era5/msl/2010 \
  --ifs-dir data/forecast/2010 \
  --hour-dir data/era5/hour/2010 \
  --output-dir outputs/figure_s1
```

Outputs: `FigureS1_a_18UTC.pdf`, `FigureS1_b_00UTC.pdf`, `FigureS1_c_06UTC.pdf`, `FigureS1_d_12UTC.pdf`, `FigureS1_shared_colorbar.pdf` (+ PNGs and the overview).

## Minimum-search note

By default the IFS local minimum is searched within a **circular 8° radius** of the domain center (1.25° grid). The original notebook used a **square of ±8 grid cells**; pass `--legacy-square-8` to reproduce it exactly. The choice can change which minimum is marked, so state the method used in the caption/methods.

Input NPY arrays are not committed.
