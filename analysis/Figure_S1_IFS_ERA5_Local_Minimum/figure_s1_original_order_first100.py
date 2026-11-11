#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay the FIRST plotted frames in the original Figure S1 notebook.

IMPORTANT: this intentionally reproduces the ORIGINAL DISCOVERY order, not the
revised science/verification selection: os.listdir (unsorted) -> each NPY[4:]
-> t=0,1,... The original marked BOTH local minima using an 8-GRID-CELL SQUARE
search, not 8 geographic degrees. This is for IDENTIFYING the historical frame.
Once found, use a verified storm/step and the newer method for publication.

Outputs: numbered PDF contact sheet and numbered CSV; no filter by lead/date,
no lookup in forecast tables, no sorting of storm names.
"""
import argparse
import csv
import os
from pathlib import Path
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--era5', type=Path, default=Path('data/era5/msl/2010'))
    p.add_argument('--forecast', type=Path, default=Path('data/forecast/2010'))
    p.add_argument('--hour', type=Path, default=Path('data/era5/hour/2010'))
    p.add_argument('--output', type=Path, default=Path('outputs/figure_s1_original_order'))
    p.add_argument('--limit', type=int, default=100, help='Number of individual displayed frames')
    p.add_argument('--per-page', type=int, default=12, help='Panels per PDF page (12 or 16 is a good choice)')
    p.add_argument('--radius-grid', type=int, default=8, help='Original square search halfwidth in grid cells')
    p.add_argument('--grid-deg', type=float, default=1.25)
    p.add_argument('--sort-files', action='store_true', help='Do NOT use for original order! Only for diagnostic alternate order.')
    return p.parse_args()


def extract_minimum(field, radius):
    ny, nx = field.shape
    cy, cx = ny // 2, nx // 2
    y0, y1 = max(0, cy-radius), min(ny, cy+radius+1)
    x0, x1 = max(0, cx-radius), min(nx, cx+radius+1)
    win = field[y0:y1, x0:x1]
    if not np.any(np.isfinite(win)):
        raise ValueError('All non-finite local pressure values')
    idx = np.unravel_index(np.nanargmin(win), win.shape)
    return float(win[idx]), (int(y0+idx[0]), int(x0+idx[1]))


def get_raw_hour(hour_arr, raw_index):
    if hour_arr is None or raw_index >= len(hour_arr):
        return ''
    val = np.asarray(hour_arr[raw_index]).flat[0]
    return int(np.rint(val)) if np.isfinite(val) else ''


def degrees(row_col, shape, grid_deg):
    y, x = row_col
    ny, nx = shape
    return ((x - (nx-1)/2.0)*grid_deg,
            (y - (ny-1)/2.0)*grid_deg)


def plot_panel(ax, F, R, v1, pos1, v2, pos2, grid_deg, panel_title):
    ny, nx = R.shape
    extent = (-nx*grid_deg/2, nx*grid_deg/2, -ny*grid_deg/2, ny*grid_deg/2)
    delta = F-R
    vmax = max(20.0, float(np.nanmax(np.abs(delta))))
    ax.imshow(delta, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
              extent=extent, interpolation='nearest')
    # Follow original plotting conventions for the contours.
    try:
        ax.contour(R, colors='blue', linestyles='solid', linewidths=.85, extent=extent)
        ax.contour(F, colors='red', linestyles='dashed', linewidths=.85, extent=extent)
    except ValueError:
        pass
    x_e, y_e = degrees(pos2, R.shape, grid_deg)
    x_f, y_f = degrees(pos1, R.shape, grid_deg)
    ax.plot(x_e, y_e, 'bx', ms=7, mew=1.4)
    ax.plot(x_f, y_f, 'rx', ms=7, mew=1.4)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xticks((-10, -5, 0, 5, 10))
    ax.set_yticks((-10, -5, 0, 5, 10))
    ax.tick_params(labelsize=6, length=2)
    ax.set_title(panel_title, fontsize=8, pad=3)
    ax.set_aspect('equal', adjustable='box')


def main():
    a = get_args()
    if a.limit < 1 or a.per_page < 1:
        raise SystemExit('--limit and --per-page must be >=1')
    a.output.mkdir(parents=True, exist_ok=True)
    if not a.forecast.is_dir() or not a.era5.is_dir():
        raise SystemExit(f'Missing input folder. forecast={a.forecast} era5={a.era5}')

    # CRUCIAL: no sorting; exactly the original notebook's directory traversal.
    files = [f for f in os.listdir(a.forecast) if f.endswith('.npy')]
    if a.sort_files:
        files.sort()
    print(f'[FILES] {len(files)}. First 10 in original directory enumeration: {files[:10]}', flush=True)

    rows = []
    page_entries = []
    pdf_path = a.output/'FigureS1_original_first100_contact_sheet.pdf'
    csv_path = a.output/'FigureS1_original_first100_index.csv'

    def export_page(pdf):
        if not page_entries:
            return
        n = len(page_entries)
        cols = 4 if a.per_page >= 8 else 2
        nrows = math.ceil(n/cols)
        fig, axs = plt.subplots(nrows, cols, figsize=(cols*3.20, nrows*3.00), squeeze=False)
        for ax in axs.flat:
            ax.axis('off')
        for entry, ax in zip(page_entries, axs.flat):
            ax.axis('on')
            plot_panel(ax, entry['F'], entry['R'], entry['fmin'], entry['fpos'],
                       entry['rmin'], entry['rpos'], a.grid_deg, entry['title'])
        first_id = page_entries[0]['id']
        last_id = page_entries[-1]['id']
        fig.suptitle(f'Original notebook enumeration — images #{first_id:03d}–#{last_id:03d}  '
                     f'(no date / lead filter)', fontsize=13, y=.999)
        fig.tight_layout(rect=(0, 0, 1, .97), pad=1.1, h_pad=1.4)
        pdf.savefig(fig, dpi=160, bbox_inches='tight')
        plt.close(fig)
        page_entries.clear()

    with PdfPages(pdf_path) as pdf:
        for filename in files:
            era_path = a.era5/filename
            fpath = a.forecast/filename
            if not era_path.is_file():
                print(f'[SKIP] no ERA5 counterpart for {filename}', flush=True)
                continue
            F_all = np.load(fpath, mmap_mode='r')
            R_all = np.load(era_path, mmap_mode='r')
            if F_all.ndim != 3 or R_all.ndim != 3 or F_all.shape[1:] != R_all.shape[1:]:
                print(f'[SKIP] unexpected shape: {filename}, IFS={F_all.shape}, ERA5={R_all.shape}', flush=True)
                continue
            hour_path = a.hour/filename
            H_all = np.load(hour_path, mmap_mode='r') if hour_path.exists() else None
            n = min(len(F_all), len(R_all))
            # CRUCIAL: raw frame 4 is first displayed: [4:][t]
            for raw in range(4, n):
                t = raw-4
                F = np.asarray(F_all[raw], dtype=float)
                R = np.asarray(R_all[raw], dtype=float)
                fmin, fpos = extract_minimum(F, a.radius_grid)
                rmin, rpos = extract_minimum(R, a.radius_grid)
                hour = get_raw_hour(H_all, raw)
                lead = {0:6, 6:12, 12:18, 18:24}.get(hour, '')
                img_id = len(rows)+1
                title = f'#{img_id:03d}  {Path(filename).stem}\nt={t}, raw={raw}, hour={hour}, lead={lead}'
                rows.append({
                    'image_number': img_id,
                    'file_name': filename,
                    'storm_key': Path(filename).stem,
                    'original_t': t,
                    'raw_index': raw,
                    'raw_hour': hour,
                    'lead_h_from_prior_mapping': lead,
                    'ifs_min_pa': fmin,
                    'era5_min_pa': rmin,
                    'ifs_minus_era5_min_hpa': (fmin-rmin)/100,
                    'ifs_min_grid_y': fpos[0], 'ifs_min_grid_x': fpos[1],
                    'era5_min_grid_y': rpos[0], 'era5_min_grid_x': rpos[1],
                })
                page_entries.append({'id':img_id,'F':F,'R':R,'fmin':fmin,'rmin':rmin,
                                     'fpos':fpos,'rpos':rpos,'title':title})
                if len(page_entries) == a.per_page:
                    export_page(pdf)
                    print(f'[PDF] Exported up to #{img_id}', flush=True)
                if len(rows) >= a.limit:
                    break
            if len(rows) >= a.limit:
                break
        export_page(pdf)
    if not rows:
        raise SystemExit('No frames found!')
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'[DONE] {len(rows)} original-order frames in {pdf_path}')
    print(f'[DONE] Lookup table: {csv_path}')
    print('Use the image_number in the PDF to identify the storm and original raw index.')
    print('Note: os.listdir order is filesystem dependent, so it can differ if files were moved.')


if __name__ == '__main__':
    main()
