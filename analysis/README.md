# Manuscript analyses

Suggested structure once the scripts are gathered:

- `analysis/figures/` — create Figure 1 (track map), and all other manuscript plots
- `analysis/tables/` — regenerate Table 1 and performance tables
- `analysis/metrics/` — common sample pairing, MAE, bias, SS, winning ratios, KS distances, bootstrap routines

Each script should declare inputs and exact outputs, work from a path supplied on the command line, and record the git commit in the run log. Keep `control_31.csv`–`control_34.csv` in untracked `data/`, not in GitHub.
