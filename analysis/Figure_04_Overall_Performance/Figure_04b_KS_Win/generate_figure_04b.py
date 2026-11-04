#!/usr/bin/env python3

"""Figure 5b, caption-consistent pairwise comparisons.


Non-IFS model pairs: matched forecast cases from all available years.

Pairs involving IFS 2010: matched forecast cases from the IFS 2010 dataset only.

KS statistics and win rates use exactly the same paired samples in each cell.

The 6 h heatmap is saved as figure5b_6h_mixed_scope.pdf / .png.

"""


from __future__ import annotations


import glob

import os

import warnings

from pathlib import Path


import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from scipy.stats import ks_2samp


warnings.filterwarnings("ignore", category=FutureWarning)


plt.rcParams.update({

    "font.family": "serif",

    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    "figure.dpi": 300,

    "font.size": 18,

    "axes.labelsize": 22,

    "xtick.labelsize": 18,

    "ytick.labelsize": 18,

    "legend.fontsize": 16,

    "lines.linewidth": 2.5,

})


# ------------------------------------------------------------

# 1. PATHS: retained from the code supplied by the author

# ------------------------------------------------------------

BASE_DIR = os.path.expanduser(os.environ.get("CYCLONE_ANALYSIS_DIR", "./data"))

SAVE_DIR = os.path.expanduser(os.environ.get("CYCLONE_OUTPUT_DIR", "./outputs/figure_04b"))

CONTROL_DIR = os.path.join(BASE_DIR, "updated_data", "control")

LEAKAGE_DIR = os.path.join(BASE_DIR, "updated_data", "leakage")

FORECAST2010_DIR = os.path.expanduser(os.environ.get("IFS_2010_DIR", "./data/ifs_2010"))


FILES = {

    6: os.path.join(CONTROL_DIR, "control_31.csv"),

    12: os.path.join(CONTROL_DIR, "control_32.csv"),

    18: os.path.join(CONTROL_DIR, "control_33.csv"),

    24: os.path.join(CONTROL_DIR, "control_34.csv"),

}

LEAKAGE_KEYS = {6: "6d", 12: "12d", 18: "18d", 24: "24d"}

ADDITIONAL_MODELS_6H = {

    "MSL Only": os.path.join(BASE_DIR, "updated_data", "msl_only", "all_cyclone_data_full_31_version_156_157_158_159_6h.csv"),

    "Simple NN": os.path.join(BASE_DIR, "updated_data", "simple_nn", "all_cyclone_data_full_31_version_0_1_2_3.csv"),

    "MSL Only (t-1)": os.path.join(BASE_DIR, "updated_data", "msl_only", "all_cyclone_data_full_31_version_0_msl_only.csv"),

    "Control (t-1)": os.path.join(BASE_DIR, "updated_data", "control", "all_cyclone_data_full_31_version_1_all.csv"),

}


IFS_NAME = "IFS 2010"

# Preserve the order in the supplied Figure 5b; additional models (t-1) are

# used in panel (a) and intentionally not included in panel (b).

FIG5B_ORDER = [

    "Persistence", "Linear Regression", "Simple NN",

    "Control", "MSL Only", "Leakage", IFS_NAME,

]


def _as_clean_key(frame: pd.DataFrame, source: str) -> pd.DataFrame:

    """Never fabricate storm IDs: doing so creates false forecast matches."""

    frame = frame.copy()

    frame.columns = frame.columns.str.strip()

    if "storm_key" not in frame.columns:

        raise ValueError(f"Missing 'storm_key' column in {source}. Cannot pair cases safely.")

    key = frame["storm_key"].astype("string").str.strip()

    key = key.str.replace(r"\.(?:npy|npz)$", "", regex=True)

    if key.isna().any() or key.eq("").any():

        raise ValueError(f"Empty/missing storm_key entries in {source}")

    frame["storm_key"] = key.astype(str)

    if "step_index" in frame.columns:

        # Match integer/floating CSV representations, e.g. 5 and 5.0.

        step = pd.to_numeric(frame["step_index"], errors="coerce")

        if step.isna().any():

            raise ValueError(f"Non-numeric/missing step_index entries in {source}")

        frame["step_index"] = step

    return frame


def _errors_from_df(

    df: pd.DataFrame, model_name: str, pred_col: str,

    true_col: str, source: str,

) -> pd.DataFrame:

    if pred_col not in df.columns or true_col not in df.columns:

        raise ValueError(

            f"{source}: model {model_name}: missing columns "

            f"{pred_col!r} or {true_col!r}; available={list(df.columns)}"

        )

    pred = pd.to_numeric(df[pred_col], errors="coerce")

    truth = pd.to_numeric(df[true_col], errors="coerce")

    keep = np.isfinite(pred.to_numpy()) & np.isfinite(truth.to_numpy())

    keys = ["storm_key"] + (["step_index"] if "step_index" in df.columns else [])

    result = df.loc[keep, keys].copy()

    result["abs_err"] = np.abs(pred.loc[keep].to_numpy() - truth.loc[keep].to_numpy())

    result["true_mslp"] = truth.loc[keep].to_numpy()

    if result.empty:

        raise ValueError(f"{source}: {model_name} has zero finite forecasts")

    # Duplicate keys would turn an inner join into an artificial many-to-many

    # expansion and distort both KS and win percentages.

    duplicates = result.duplicated(subset=keys, keep=False)

    if duplicates.any():

        sample = result.loc[duplicates, keys].head(5).to_dict("records")

        raise ValueError(

            f"{source}: {model_name} has {duplicates.sum()} duplicate forecast "

            f"rows for matching keys {keys}. Examples: {sample}. "

            "Please identify a unique forecast-case ID before computing Figure 5b."

        )

    print(f"LOADED {model_name:22} rows={len(result):>8,} source={Path(source).name}")

    return result


def load_control_models(lead: int) -> dict[str, pd.DataFrame]:

    path = FILES[lead]

    if not os.path.isfile(path):

        raise FileNotFoundError(path)

    df = _as_clean_key(pd.read_csv(path), path)

    truth = next((c for c in ["true_mslp_corr", "true_mslp_hpa", "true_mslp"] if c in df), None)

    if truth is None:

        raise ValueError(f"No true pressure column in {path}")

    pred_cols = {

        "Control": ["pred_mslp_corr"],

        "Persistence": ["persistence_pred"],

        "Linear Regression": ["linear_pred", "lin_pred"],

    }

    errors: dict[str, pd.DataFrame] = {}

    for name, choices in pred_cols.items():

        chosen = next((c for c in choices if c in df), None)

        if chosen is None:

            print(f"WARNING: {name} missing from {path}; skipped")

            continue

        errors[name] = _errors_from_df(df, name, chosen, truth, path)

    return errors


def load_leakage(lead: int) -> pd.DataFrame:

    path = os.path.join(LEAKAGE_DIR, f"leakage_{LEAKAGE_KEYS[lead]}.csv")

    if not os.path.isfile(path):

        raise FileNotFoundError(path)

    df = _as_clean_key(pd.read_csv(path), path)

    for pred, truth in [

        ("pred_mslp_corr", "true_mslp_corr"),

        ("pred_mslp_hpa", "true_mslp_hpa"),

    ]:

        if pred in df and truth in df:

            return _errors_from_df(df, "Leakage", pred, truth, path)

    raise ValueError(f"Leakage prediction/truth columns not found in {path}")


def load_extra_models_6h() -> dict[str, pd.DataFrame]:

    errors: dict[str, pd.DataFrame] = {}

    for name, path in ADDITIONAL_MODELS_6H.items():

        # The figure caption's panel (b) includes these two but not t-1 cases.

        if name not in FIG5B_ORDER:

            continue

        if not os.path.isfile(path):

            raise FileNotFoundError(path)

        df = _as_clean_key(pd.read_csv(path), path)

        truth = next((c for c in ["true_mslp_corr", "true_mslp_hpa", "true_mslp"] if c in df), None)

        pred = next((c for c in ["pred_mslp_corr", "pred_mslp_hpa"] if c in df), None)

        if truth is None or pred is None:

            raise ValueError(f"Prediction/truth column missing in {path}")

        errors[name] = _errors_from_df(df, name, pred, truth, path)

    return errors


def load_ifs(lead: int) -> pd.DataFrame:

    path = os.path.join(FORECAST2010_DIR, f"true_vs_pred_{lead}h.csv")
    paths = [path] if os.path.isfile(path) else []

    if not paths:

        raise FileNotFoundError(f"No IFS 2010 files found for {lead}h in {FORECAST2010_DIR}")

    df = _as_clean_key(pd.concat([pd.read_csv(p) for p in paths], ignore_index=True),

                       f"IFS_{lead}h_files")

    truth = next((c for c in ["true_mslp_ifs", "true_mslp_hpa"] if c in df), None)

    pred = next((c for c in ["pred_mslp_ifs", "pred_mslp_hpa"] if c in df), None)

    if truth is None or pred is None:

        raise ValueError(f"IFS truth/prediction columns not found. Available: {list(df.columns)}")

    print(f"IFS files matched: {len(paths)}")

    return _errors_from_df(df, IFS_NAME, pred, truth, paths[0])


def matched_pair(left: pd.DataFrame, right: pd.DataFrame,

                 name_left: str, name_right: str) -> pd.DataFrame:

    left_step = "step_index" in left.columns

    right_step = "step_index" in right.columns

    if left_step != right_step:

        # Matching on storm_key alone is valid ONLY if both datasets have one

        # forecast per storm. Otherwise it silently matches wrong lead steps.

        if left.duplicated("storm_key").any() or right.duplicated("storm_key").any():

            raise ValueError(

                f"Cannot safely pair {name_left} with {name_right}: step_index "

                f"exists for one model only, and at least one storm has "

                f"multiple forecasts. Original code silently fell back "

                f"to storm_key, causing incorrect many-to-many pairing."

            )

    keys = ["storm_key", "step_index"] if left_step and right_step else ["storm_key"]

    merged = pd.merge(left, right, on=keys, how="inner", validate="one_to_one",

                      suffixes=("_row", "_col"), sort=False)

    if merged.empty:

        raise ValueError(

            f"No shared forecast cases for {name_left} vs {name_right} "

            f"using keys={keys}. Check storm identifiers and time indices."

        )

    return merged


def compute_pairwise(errors: dict[str, pd.DataFrame], order: list[str]) -> tuple:

    """Non-IFS pairs use all years; IFS pairs use matched 2010 forecast cases."""

    n = len(order)

    D = np.zeros((n, n), dtype=float)

    win = np.full((n, n), np.nan, dtype=float)

    annot = np.full((n, n), "-", dtype=object)

    records = []


    for i in range(n):

        for j in range(i + 1, n):

            row_name, col_name = order[i], order[j]

            # IFS was loaded exclusively from its 2010 source. Thus the inner

            # intersection *automatically* restricts only IFS pairs to 2010.

            is_ifs_pair = IFS_NAME in (row_name, col_name)

            scope = "IFS 2010 matched cases" if is_ifs_pair else "All-year matched cases"

            m = matched_pair(errors[row_name], errors[col_name], row_name, col_name)

            er = m["abs_err_row"].to_numpy()

            ec = m["abs_err_col"].to_numpy()

            stat, pval = ks_2samp(er, ec, alternative="two-sided", method="auto")

            row_wins = np.count_nonzero(er < ec)

            col_wins = np.count_nonzero(ec < er)

            ties = len(m) - row_wins - col_wins

            target_diff = np.abs(m["true_mslp_row"].to_numpy() - m["true_mslp_col"].to_numpy())

            mean_target_diff = float(np.mean(target_diff))

            max_target_diff = float(np.max(target_diff))

            if max_target_diff > 0.1:

                print(f"WARNING targets differ for {row_name} vs {col_name}: "

                      f"mean={mean_target_diff:.4g}, max={max_target_diff:.4g} hPa; "

                      "verify the two files represent the same forecast target")

            D[i, j] = D[j, i] = stat

            annot[i, j] = annot[j, i] = f"{stat:.3f}{'*' if pval < 0.05 else ''}"

            win[i, j] = row_wins / len(m) * 100

            win[j, i] = col_wins / len(m) * 100

            records.append({

                "row_model": row_name,

                "column_model": col_name,

                "sample_scope": scope,

                "n_paired_forecasts": len(m),

                "n_storms": m["storm_key"].nunique(),

                "ks_D": stat,

                "ks_pvalue": pval,

                "significant_p_lt_0_05": pval < 0.05,

                "row_wins_percent": win[i, j],

                "column_wins_percent": win[j, i],

                "ties_percent": 100 * ties / len(m),

                "mae_row_on_pair_hpa": np.mean(er),

                "mae_column_on_pair_hpa": np.mean(ec),

                "mean_abs_true_mslp_difference_hpa": mean_target_diff,

                "max_abs_true_mslp_difference_hpa": max_target_diff,

            })

            print(f"PAIR {row_name:20} vs {col_name:20} "

                  f"scope={scope:23} n={len(m):>8,} "

                  f"KS={stat:.4f} p={pval:.2e} "

                  f"win={win[i,j]:.1f}%/{win[j,i]:.1f}% "

                  f"ties={100*ties/len(m):.2f}%")

    return D, annot, win, pd.DataFrame.from_records(records)


def draw_heatmap(order: list[str], D: np.ndarray, annot: np.ndarray,

                 win: np.ndarray, output_stem: str) -> None:

    n = len(order)

    fig, ax = plt.subplots(figsize=(12, 10) if n > 5 else (10, 8))

    sns.heatmap(

        D, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1,

        xticklabels=order, yticklabels=order, linewidths=.5,

        linecolor="white", square=True, ax=ax,

        annot_kws={"size": 16, "weight": "normal"},

        cbar_kws={"label": "KS Distance (D)"},

    )

    for i in range(n):

        for j in range(n):

            if i != j and np.isfinite(win[i, j]):

                ax.text(j + .96, i + .04, f"{win[i,j]:.1f}%",

                        ha="right", va="top", color="black", fontsize=15,

                        alpha=.85, fontweight="bold")

    if IFS_NAME in order:

        idx = order.index(IFS_NAME)

        ax.axhline(idx, color="black", linewidth=3)

        ax.axvline(idx, color="black", linewidth=3)

    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=18)

    ax.set_yticklabels(order, rotation=0, fontsize=18)

    fig.tight_layout()

    for ext in ("pdf", "png"):

        p = f"{output_stem}.{ext}"

        fig.savefig(p, format=ext, bbox_inches="tight",

                    **({"dpi": 300} if ext == "png" else {}))

        print(f"SAVED: {p}")

    plt.close(fig)


def main() -> None:

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Figure 5b: caption-consistent mixed-year pairwise comparisons")

    print("Non-IFS vs non-IFS: all years; IFS vs anything: IFS 2010 matched cases")

    errors = load_control_models(6)

    errors["Leakage"] = load_leakage(6)

    errors.update(load_extra_models_6h())

    errors[IFS_NAME] = load_ifs(6)

    missing = set(FIG5B_ORDER) - set(errors)

    if missing:

        raise ValueError(f"Cannot produce caption-consistent 5b; missing {missing}")

    D, annot, win, report = compute_pairwise(errors, FIG5B_ORDER)

    stem = os.path.join(SAVE_DIR, "figure5b_6h_mixed_scope")

    draw_heatmap(FIG5B_ORDER, D, annot, win, stem)

    report.to_csv(f"{stem}_pairwise_audit.csv", index=False)

    print(f"SAVED: {stem}_pairwise_audit.csv")

    print("Done: Figure 5b uses caption-specified pairwise evaluation scopes.")


if __name__ == "__main__":

    main()


