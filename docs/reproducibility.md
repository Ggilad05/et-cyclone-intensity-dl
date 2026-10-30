# Reproducibility and Git workflow

1. Add the current `src/` code without modifying its behavior; make an initial commit.
2. Add redacted configs and document how machine-local `DATA_PATH` is supplied.
3. Add generation scripts for Table 1 and Figure 1; document which tracks and filters are used.
4. Validate every region's storm IDs and compare against the actual test split before generating results.
5. Generate figures/tables into ignored `outputs/` first, then commit reviewed final exports selectively if journal requirements call for it.
6. Commit each coherent change (e.g., `fix: make regional storm lists disjoint`, `docs: explain genesis-latitude filter`).

Avoid committing raw ERA5 files, saved checkpoints, credentials, unpublished large data tables, and file paths that identify private servers or user directories. Keep research outputs under version control only when provenance and sharing permission are confirmed.
