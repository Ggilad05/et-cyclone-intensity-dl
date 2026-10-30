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
