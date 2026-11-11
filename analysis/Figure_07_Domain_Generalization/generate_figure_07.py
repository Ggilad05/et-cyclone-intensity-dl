#!/usr/bin/env python3
"""Figure 7 — MAE vs lead time, Control vs domain-generalization (DG) models.

(a) seasons, (b) regions. For each DG category (held-out season or region),
DG forecasts are inner-joined to Control on (storm_key, step_index, lead_h), so
both curves use exactly the same storms and steps. Solid = Control, dashed = DG.

Run from the repository root:
  python analysis/Figure_07_Domain_Generalization/generate_figure_07.py \
      --base-dir data/updated_data --out-dir outputs/figure_07

Expected inputs (not committed):
  <base-dir>/control/control_31.csv ... control_34.csv   (6/12/18/24 h)
  <base-dir>/dg/<lead>/<category>.csv, lead in 6,12,18,24 and category in
      sa, so, np, na, med, winter, spring, summer, autumn
Each CSV needs: storm_key, step_index, dl_abs_error.

Only personal paths were replaced by command-line options; the calculation is
unchanged from the original script.
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================
# 0. AMS PUBLICATION STYLE SETTINGS
# ==============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "figure.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "lines.linewidth": 2.5,
    "lines.markersize": 7
})

# ==============================
# 1. CONFIGURATION
# ==============================
HEMISPHERES = ["NH", "SH"]
REGIONS = ["NA", "NP", "MED", "SA", "SO"]
SEASONS = ["WINTER", "SPRING", "SUMMER", "AUTUMN"]

MODEL_TO_LEAD = {"31": 6, "32": 12, "33": 18, "34": 24}
DG_LEADS = ["6", "12", "18", "24"]
DG_CATEGORIES = ["sa", "so", "np", "na", "med", "winter", "summer", "spring", "autumn"]

USECOLS = ["storm_key", "step_index", "dl_abs_error"]


# ==============================
# 2. DATA LOADING FUNCTIONS
# ==============================
def load_control_df(control_dir: str) -> pd.DataFrame:
    dfs = []
    for mk, lead in MODEL_TO_LEAD.items():
        path = os.path.join(control_dir, f"control_{mk}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, usecols=lambda c: c in USECOLS, dtype={'storm_key': str})
                df = df.dropna(subset=["step_index"])
                df["storm_key"] = df["storm_key"].str.strip()
                df["lead_h"] = lead
                dfs.append(df)
            except Exception as e:
                print(f"[ERROR] Failed to load Control {mk}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_dg_df(dg_dir: str) -> pd.DataFrame:
    dfs = []
    for lead in DG_LEADS:
        lead_folder = os.path.join(dg_dir, lead)
        if not os.path.exists(lead_folder):
            continue

        for category in DG_CATEGORIES:
            path = os.path.join(lead_folder, f"{category}.csv")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, usecols=lambda c: c in USECOLS, dtype={'storm_key': str})
                    df = df.dropna(subset=["step_index"])
                    df["storm_key"] = df["storm_key"].str.strip()
                    df["lead_h"] = int(lead)
                    df["dg_category"] = category.upper()
                    dfs.append(df)
                except Exception as e:
                    print(f"[ERROR] Failed to load DG file {category}.csv in lead {lead}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ==============================
# 3. PLOTTING FUNCTIONS
# ==============================
def plot_mae_vs_lead_seasons(merged_df: pd.DataFrame, out_dir: str):
    print("Generating MAE vs Lead Time for Seasons...")
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'WINTER': 'tab:blue', 'SPRING': 'tab:orange', 'SUMMER': 'tab:green', 'AUTUMN': 'tab:red'}
    leads = [6, 12, 18, 24]
    legend_elements = []

    for season in SEASONS:
        subset = merged_df[merged_df['dg_category'] == season]
        if subset.empty:
            continue

        ctrl_mean = subset.groupby('lead_h')['dl_abs_error_Control'].mean().reindex(leads)
        dg_mean = subset.groupby('lead_h')['dl_abs_error_DG'].mean().reindex(leads)

        col = colors[season]

        # Control solid, DG dashed
        ax.plot(leads, ctrl_mean, color=col, linestyle='-', marker='o', label='_nolegend_')
        ax.plot(leads, dg_mean, color=col, linestyle='--', marker='o', label='_nolegend_')

        # season color in legend
        legend_elements.append(mlines.Line2D([0], [0], color=col, lw=2.5, label=season.capitalize()))

    # line-style key (Control vs DG)
    legend_elements.append(mlines.Line2D([], [], color='none', label=''))  # spacer
    legend_elements.append(mlines.Line2D([0], [0], color='black', linestyle='-', lw=2.5, label='Control'))
    legend_elements.append(mlines.Line2D([0], [0], color='black', linestyle='--', lw=2.5, label='DG'))

    ax.set_xticks(leads)
    ax.set_xlabel("Lead Time (h)")
    ax.set_ylabel("MAE (hPa)")

    ax.grid(True, linestyle=":", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    ax.legend(handles=legend_elements, loc='upper left', frameon=False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "MAE_vs_Lead_Seasons.pdf")
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_mae_vs_lead_regions(merged_df: pd.DataFrame, out_dir: str):
    print("Generating MAE vs Lead Time for Regions...")
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {'NA': 'tab:blue', 'NP': 'tab:orange', 'MED': 'tab:green', 'SA': 'tab:red', 'SO': 'tab:purple'}
    leads = [6, 12, 18, 24]
    legend_elements = []

    for region in REGIONS:
        subset = merged_df[merged_df['dg_category'] == region]
        if subset.empty:
            continue

        ctrl_mean = subset.groupby('lead_h')['dl_abs_error_Control'].mean().reindex(leads)
        dg_mean = subset.groupby('lead_h')['dl_abs_error_DG'].mean().reindex(leads)

        col = colors[region]

        ax.plot(leads, ctrl_mean, color=col, linestyle='-', marker='o', label='_nolegend_')
        ax.plot(leads, dg_mean, color=col, linestyle='--', marker='o', label='_nolegend_')

        legend_elements.append(mlines.Line2D([0], [0], color=col, lw=2.5, label=region))

    legend_elements.append(mlines.Line2D([], [], color='none', label=''))
    legend_elements.append(mlines.Line2D([0], [0], color='black', linestyle='-', lw=2.5, label='Control'))
    legend_elements.append(mlines.Line2D([0], [0], color='black', linestyle='--', lw=2.5, label='DG'))

    ax.set_xticks(leads)
    ax.set_xlabel("Lead Time (h)")
    ax.set_ylabel("MAE (hPa)")

    ax.grid(True, linestyle=":", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    ax.legend(handles=legend_elements, loc='upper left', frameon=False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "MAE_vs_Lead_Regions.pdf")
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# ==============================
# 4. MAIN EXECUTION
# ==============================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default="data/updated_data",
                    help="Folder containing control/ and dg/")
    ap.add_argument("--out-dir", default="outputs/figure_07")
    args = ap.parse_args()

    base_dir = os.path.expanduser(args.base_dir)
    out_dir = os.path.expanduser(args.out_dir)

    print("Loading Control data...")
    ctrl_df = load_control_df(os.path.join(base_dir, "control"))

    print("Loading Domain Generalization (DG) data...")
    dg_df = load_dg_df(os.path.join(base_dir, "dg"))

    if ctrl_df.empty or dg_df.empty:
        print("[ERROR] Datasets are missing. Please verify paths.")
        return

    print("Merging datasets (Strict pairing)...")
    # Inner join: Control and DG compared on exactly the same storms and time steps
    merged_df = pd.merge(dg_df, ctrl_df, on=['storm_key', 'step_index', 'lead_h'], suffixes=('_DG', '_Control'))

    os.makedirs(out_dir, exist_ok=True)

    plot_mae_vs_lead_seasons(merged_df, out_dir)
    plot_mae_vs_lead_regions(merged_df, out_dir)

    print("\nProcess completed successfully!")


if __name__ == "__main__":
    main()
