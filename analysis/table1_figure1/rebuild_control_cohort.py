#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild Table 1 and Figure 1 using CONTROL storm membership and ORIGINAL
NetCDF cyclone tracks sampled by their t coordinates at 6-hour UTC cadence.

The Control CSV(s) define WHICH storms are included. The original NetCDF
defines WHERE and WHEN each storm travels. No genesis/motion/max70 filtering
is reapplied; those filters were already applied upstream of the Control cohort.

Crucially, select t = 1,3,5,..., not array positions 0,2,4,... .
This matches the user's ERA5 extraction script:
  odd_time_indexes = time_indexes[time_indexes % 2 != 0]
  time = season_start + timedelta(hours=3) * (i-1)
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
import numpy as np
import xarray as xr

SEASONS = ("DJF", "MAM", "JJA", "SON")
REGIONS = ("North Atlantic", "North Pacific", "Mediterranean", "South Atlantic", "Southern Ocean")
CATEGORIES = SEASONS + REGIONS
BOUNDS = {
    "North Atlantic": (-80, -10, 35, 70),
    "North Pacific": (120, 240, 30, 60),
    "Mediterranean": (-10, 50, 30, 45),
    "South Atlantic": (-65, 0, -70, -20),  # Original Figure 1 SA rectangle (lon W65–0, lat S70–S20)
    "Southern Ocean": (0, 360, -70, -30),
}
CLASSIFICATION_FILE_ALIASES = {
    "North Atlantic": ("classification_NH_NA.txt",),
    "North Pacific": ("classification_NH_NP.txt",),
    "Mediterranean": ("classification_NH_MED.txt",),
    "South Atlantic": ("classification_SH_SA_corrected.txt", "classification_SH_SA.txt"),
    "Southern Ocean": ("classification_SH_SO_exclusive.txt",),
}


def normalize_storm_name(value: str) -> str:
    """Normalize case-preserving foo, foo.npy and foo.npy.npy to foo.npy."""
    s = str(value).strip().strip("\"'")
    while s.lower().endswith('.npy'):
        s = s[:-4]
    return s + '.npy' if s else ''


def load_control_keys(csv_paths: list[Path]) -> set[str]:
    result = set()
    for path in csv_paths:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or 'storm_key' not in reader.fieldnames:
                raise ValueError(f"No storm_key column in Control CSV: {path}")
            seen = set()
            rows = 0
            for row in reader:
                rows += 1
                val = normalize_storm_name(row.get('storm_key',''))
                if val:
                    seen.add(val)
        print(f"Control {path.name}: {rows:,} rows, {len(seen):,} unique storm IDs")
        result.update(seen)
    print(f"Union of Control storm IDs: {len(result):,}")
    return result


def load_classifications(folder: Path) -> dict[str, set[str]]:
    members = {}
    for region, aliases in CLASSIFICATION_FILE_ALIASES.items():
        # The starter repository keeps NH lists under original/ and
        # disjoint SH lists under revised/. A standalone flat directory
        # is supported as well (the format of the original script package).
        candidates = [folder / n for n in aliases]
        candidates += [folder / 'revised' / n for n in aliases]
        candidates += [folder / 'original' / n for n in aliases]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing classification for {region} in {folder} (expected {aliases})")
        names = {normalize_storm_name(n) for n in re.split(r'[,\s]+', path.read_text(encoding='utf-8-sig'))}
        names.discard('')
        members[region] = names
        print(f"Classification {region}: {len(names):,} IDs ({path.name})")
    overlap = members['South Atlantic'] & members['Southern Ocean']
    if overlap:
        raise RuntimeError(f"SA and SO classification lists overlap: {len(overlap)} IDs")
    return members

def new_stats():
    return {hemi: {cat: {"storms": 0, "steps": 0} for cat in CATEGORIES}
            for hemi in ("NH", "SH")}


def bump(stats, hemi, category, steps):
    stats[hemi][category]["storms"] += 1
    stats[hemi][category]["steps"] += steps


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_table(stats, out_dir, total, step_count_3h):
    rows = []
    for cat in CATEGORIES:
        rows.append({
            "category": cat,
            "NH_storms": stats["NH"][cat]["storms"],
            "NH_steps_6h": stats["NH"][cat]["steps"],
            "SH_storms": stats["SH"][cat]["storms"],
            "SH_steps_6h": stats["SH"][cat]["steps"],
        })
    nh_n = sum(stats["NH"][s]["storms"] for s in SEASONS)
    sh_n = sum(stats["SH"][s]["storms"] for s in SEASONS)
    nh_t = sum(stats["NH"][s]["steps"] for s in SEASONS)
    sh_t = sum(stats["SH"][s]["steps"] for s in SEASONS)
    rows.append({"category": "Total", "NH_storms": nh_n, "NH_steps_6h": nh_t,
                 "SH_storms": sh_n, "SH_steps_6h": sh_t})
    write_csv(out_dir / "Table1_updated.csv", list(rows[0]), rows)

    with (out_dir / "Table1_updated_rows.tex").open("w", encoding="utf-8") as f:
        f.write("% Table 1 rows. Storms and 6-hour track steps; 1979--2020 inclusive.\n")
        for r in rows:
            f.write(f"{r['category']} & {r['NH_storms']:,} & {r['NH_steps_6h']:,} & "
                    f"{r['SH_storms']:,} & {r['SH_steps_6h']:,} " + "\\\\\n")

    print("\n===== UPDATED TABLE 1 (STEPS EVERY 6 h) =====")
    print(f"{'CATEGORY':<20} | {'NH STORMS':>11} | {'NH STEPS':>11} | {'SH STORMS':>11} | {'SH STEPS':>11}")
    for r in rows:
        print(f"{r['category']:<20} | {r['NH_storms']:>11,} | {r['NH_steps_6h']:>11,} | "
              f"{r['SH_storms']:>11,} | {r['SH_steps_6h']:>11,}")
    print(f"\nControl-matched storms: {total:,} storms; {nh_t+sh_t:,} sampled 6h steps "
          f"(vs {step_count_3h:,} valid 3h tracking positions)")
    if nh_n + sh_n != total:
        raise AssertionError("Global total disagrees with hemisphere-season totals")


def draw_figure(density_per_year, out_dir: Path, start_year: int, end_year: int):
    """Original AMS-style map. Cartopy is imported only if plotting is requested."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        import matplotlib.ticker as mticker
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.feature import ShapelyFeature
        from shapely.geometry import Polygon
    except ImportError as e:
        raise SystemExit(f"Plot requires matplotlib, cartopy and shapely: {e}. "
                         "Install cartopy or rerun with --no-plot.") from e

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })
    lon_edges = np.arange(0, 365, 5)
    lat_edges = np.arange(-90, 95, 5)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={"projection": ccrs.PlateCarree(central_longitude=60)})
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="black", linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=2)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 15}
    gl.ylabel_style = {"size": 15}
    gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120, 180, 240])

    im = ax.pcolormesh(lon_centers, lat_centers,
                       np.ma.masked_where(density_per_year == 0, density_per_year),
                       transform=ccrs.PlateCarree(), cmap="Greens", shading="auto")
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    cbar.set_label("Track Density (systems per year)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    labels = {
        "Southern Ocean": (65, -25), "North Pacific": (180, 65),
        "North Atlantic": (-45, 75), "Mediterranean": (20, 50),
        "South Atlantic": (-32.5, -15),  # Original label position, just north of SA rectangle
    }
    for name, (west, east, south, north) in BOUNDS.items():
        if name == "Southern Ocean":
            # Match original plotting solution for a globally zonal region.
            for lat in (south, north):
                ax.plot([-120, 240], [lat, lat], transform=ccrs.PlateCarree(),
                        color="black", linewidth=2, zorder=3)
        else:
            poly = Polygon([(west, south), (east, south), (east, north), (west, north)])
            feat = ShapelyFeature([poly], ccrs.PlateCarree(), edgecolor="black",
                                  facecolor="none", linewidth=2, zorder=3)
            ax.add_feature(feat)
        cx, cy = labels[name]
        ax.text(cx, cy, name, fontsize=12, fontweight="bold", color="black",
                transform=ccrs.PlateCarree(), ha="center", va="center", zorder=10,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    stem = f"Figure1_track_density_{start_year}_{end_year}_6h"
    fig.savefig(out_dir / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 saved: {out_dir / (stem + '.pdf')} and .png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--control-csv', required=True, nargs='+', type=Path,
                   help='Full all-year Control CSV(s); use one 24h file or several for union across leads.')
    p.add_argument('--data-dir', type=Path, default=Path('/arctic2/shreibshtein/MasterProject/OrCyclones/ERA_5_Track'))
    p.add_argument('--classifications-dir', type=Path, default=Path(__file__).resolve().parents[2]/'classifications')
    p.add_argument('--out-dir', type=Path, default=Path('./Table1_Figure1_from_Control'))
    p.add_argument('--start-year', type=int, default=1979)
    p.add_argument('--end-year', type=int, default=2020)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--allow-missing', action='store_true',
                   help='Permit missing storm IDs in source tracks; NOT recommended for manuscript outputs.')
    args = p.parse_args()
    if args.start_year > args.end_year:
        p.error('--start-year must be <= --end-year')
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ids = load_control_keys([x.expanduser() for x in args.control_csv])
    members = load_classifications(args.classifications_dir.expanduser())
    data_dir = args.data_dir.expanduser()
    files = sorted(data_dir.glob('*.nc'))
    if not files:
        raise SystemExit(f'No NetCDF files in {data_dir}')

    # Control keys already identify selected storms. Do not apply pre_tracks()
    # or the max70 filter again: either would change the model's actual cohort.
    target_ids = set()
    for storm_id in ids:
        match = re.search(r'_(\d{4})_', storm_id)
        if not match:
            raise ValueError(f'Cannot extract year from Control storm ID: {storm_id}')
        if args.start_year <= int(match.group(1)) <= args.end_year:
            target_ids.add(storm_id)
    stats = new_stats()
    lat_edges = np.arange(-90, 95, 5)
    lon_edges = np.arange(0, 365, 5)
    density = np.zeros((len(lat_edges)-1, len(lon_edges)-1), dtype=float)
    included = []
    selected = set()
    missing_valid_time = []
    diagnostics = []
    total_3h = 0
    processed_files = 0
    for file in files:
        m = re.search(r'_(\d{4})_', file.name)
        if not m:
            continue
        year = int(m.group(1))
        season = file.name.split('_')[0]
        if not (args.start_year <= year <= args.end_year and season in SEASONS):
            continue
        processed_files += 1
        if processed_files % 20 == 1:
            print(f'NetCDF {processed_files}: {file.name}', flush=True)
        with xr.open_dataset(file) as ds:
            arr = ds['data'].to_numpy()
            trackids = ds['trackid'].to_numpy()
        if arr.ndim != 3 or arr.shape[0] < 4:
            raise ValueError(f'Unexpected NetCDF data dimensions in {file}: {arr.shape}')
        for j, trackid in enumerate(trackids):
            storm_id = normalize_storm_name(f'{file.stem}_{int(trackid)}.npy')
            if storm_id not in target_ids:
                continue
            if storm_id in selected:
                raise ValueError(f'Duplicate storm ID in NetCDF sources: {storm_id}')
            selected.add(storm_id)
            t = arr[0, :, j]
            lon = arr[1, :, j]
            lat = arr[2, :, j]
            valid_t = np.isfinite(t) & (t != 0)
            time6 = valid_t & (np.mod(t, 2) == 1)
            valid6 = time6 & np.isfinite(lon) & np.isfinite(lat)
            if not np.any(valid6):
                missing_valid_time.append({'storm_id': storm_id, 'file': file.name})
                continue
            if int(np.count_nonzero(valid6)) != int(np.count_nonzero(time6)):
                diagnostics.append({'storm_id': storm_id, 'issue': 'nonfinite_coordinate_at_6h',
                                    'count': int(np.count_nonzero(time6)-np.count_nonzero(valid6))})
            t6 = t[valid6]
            if len(t6) != len(np.unique(t6)):
                diagnostics.append({'storm_id': storm_id, 'issue': 'duplicate_t_values', 'count': len(t6)-len(np.unique(t6))})
            lat6 = lat[valid6]
            lon6 = lon[valid6] % 360.0
            steps6 = len(t6)
            steps3 = int(np.count_nonzero(valid_t))
            total_3h += steps3
            hemi = 'NH' if ('_NH_' in storm_id) else 'SH' if ('_SH_' in storm_id) else ('NH' if lat6[0] >= 0 else 'SH')
            bump(stats, hemi, season, steps6)
            regions = [reg for reg in REGIONS if storm_id in members[reg]]
            for reg in regions:
                bump(stats, hemi, reg, steps6)
            included.append({'storm_id': storm_id, 'season': season, 'year': year,
                             'hemisphere': hemi, 'steps_3h': steps3, 'steps_6h': steps6,
                             'first_t_6h': int(t6[0]), 'last_t_6h': int(t6[-1]),
                             'max_abs_lat_6h': float(np.max(np.abs(lat6))),
                             'regions': ';'.join(regions)})
            # Figure 1: storms counted at most once in each 5x5 degree grid cell.
            hist, _, _ = np.histogram2d(lat6, lon6, bins=[lat_edges, lon_edges])
            density += (hist > 0)

    not_found = sorted(target_ids - selected)
    print(f'\nControl cohort IDs within date range: {len(target_ids):,}')
    print(f'Found in NetCDF: {len(selected):,}; missing: {len(not_found):,}')
    write_csv(out/'missing_control_storms.csv', ['storm_id'], [{'storm_id': x} for x in not_found])
    write_csv(out/'control_storms_without_6h_steps.csv', ['storm_id','file'], missing_valid_time)
    write_csv(out/'time_quality_flags.csv', ['storm_id','issue','count'], diagnostics)
    write_csv(out/'included_storms.csv',
              ['storm_id','season','year','hemisphere','steps_3h','steps_6h','first_t_6h',
               'last_t_6h','max_abs_lat_6h','regions'], included)
    if (not_found or missing_valid_time) and not args.allow_missing:
        raise SystemExit(f'ERROR: {len(not_found):,} unmatched and {len(missing_valid_time):,} without 6h '
                         f'positions. Check {out} audit files. Use --allow-missing only for debugging.')
    if not included:
        raise SystemExit('No Control storms matched; check NetCDF filename and storm_key conventions.')

    save_table(stats, out, len(included), total_3h)
    cover_rows = []
    for region in REGIONS:
        control_region = members[region] & target_ids
        matched_region = members[region] & (selected - {x['storm_id'] for x in missing_valid_time})
        cover_rows.append({'region':region, 'classification_ids':len(members[region]),
                          'control_in_region':len(control_region), 'found_6h_in_region':len(matched_region)})
    write_csv(out/'classification_coverage.csv', list(cover_rows[0]), cover_rows)
    print('\nClassification coverage (control IDs -> matched 6h tracks):')
    for row in cover_rows:
        print(f"  {row['region']:<17} {row['control_in_region']:,} -> {row['found_6h_in_region']:,}")
    print('t is filtered using t % 2 == 1 (matches original ERA5 extraction), NOT [::2].')
    print('Note: steps are COMPLETE 6h tracking positions for included storms, not forecast-target row counts.')
    density /= (args.end_year - args.start_year + 1)
    np.save(out/'Figure1_density_systems_per_year.npy', density)
    if not args.no_plot:
        draw_figure(density, out, args.start_year, args.end_year)
    print(f'Finished. Outputs in {out}')


if __name__ == '__main__':
    main()
