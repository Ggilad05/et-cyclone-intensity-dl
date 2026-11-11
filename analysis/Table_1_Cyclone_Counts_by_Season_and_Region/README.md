# Table 1 — Cyclone Counts by Season and Region

Number of cyclones and 6-hourly track steps across seasons and geographic regions in the Northern and Southern Hemispheres (1979–2020).

The table uses the Control storm-ID cohort and the region-classification lists. Six-hourly positions are selected from 3-hourly cyclone tracks using odd `t` values.

**Files:**
- `Table_1_Cyclone_Counts_by_Season_and_Region.tex` — manuscript-ready LaTeX table.
- `Table_1_Cyclone_Counts_by_Season_and_Region.csv` — numerical table.
- `generate_table_1.py` — Python reconstruction script.
- `reference/control_34_storm_keys.csv` — unique storm IDs from the Control dataset.

**Results:** 45,378 cyclones and 821,014 six-hourly track steps.

Run from the repository root:

```bash
python analysis/Table_1_Cyclone_Counts_by_Season_and_Region/generate_table_1.py \
  --data-dir /arctic2/shreibshtein/MasterProject/OrCyclones/ERA_5_Track
```

The script reads the existing `classifications/` directory and writes a CSV to `outputs/table_1/`. The original tracking NetCDF files are required to regenerate the counts.

## `table1_fig1.py` (6 Nov 2026)

Builds Table 1 **and** a track-density map in one pass, both from the Control storm cohort (Control CSV defines which storms; original NetCDF tracks define where/when; odd `t = 1,3,5,...` for 6-hourly UTC). It searches `classifications/revised/` then `classifications/original/`, so the corrected SA and exclusive SO lists are used, and it stops if SA and SO overlap.

```bash
python analysis/Table_1_Cyclone_Counts_by_Season_and_Region/table1_fig1.py \
  --data-dir data/ERA_5_Track --out-dir outputs/table1_figure1
```

Note: the manuscript Figure 1 currently uses the broader storm-track sample (`../Figure_1_Global_Cyclone_Track_Density/generate_figure_1.py`). The map from this script is a Control-cohort alternative; `generate_table_1.py` remains the reference Table 1 script.
