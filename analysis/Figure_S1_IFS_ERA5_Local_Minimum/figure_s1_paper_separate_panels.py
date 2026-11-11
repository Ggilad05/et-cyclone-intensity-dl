#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera-ready four-panel ERA5/IFS comparison for the identified Figure S1 storm.

The identified original notebook shows the first four raw entries (indices 4,5,6,7)
of JJA_2010_NH_neg_135, with hour metadata 18,00,06,12 in that exact order.

Outputs a 2x2 overview plus FOUR separate PDF/PNG panel files (a-d), and
one shared standalone colorbar PDF/PNG, all with identical color limits and
contour levels. It does not select storms via a forecast table or date.

The original blue X marked the ERA5 *local minimum*. That is the default for
faithful Figure S1 reproduction. Both the ERA5 local minimum and the ERA5
tracked-center pressure are saved to a CSV, so quantitative comparison against
ERA5 center does not silently become a minimum-pressure comparison.

The IFS local minimum is searched within a circular 8-degree radius of the
storm-centered domain center on the 1.25-degree grid. Pass --legacy-square-8
if you need the exact original notebook selection (square +/-8 grid cells).

Requires numpy, matplotlib. Uses your local NPY arrays only; no web access.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

STORM = 'JJA_2010_NH_neg_135'
EXPECTED_RAW_HOURS = (18, 0, 6, 12)
LEAD_BY_RAW_HOUR = {0: 6, 6: 12, 12: 18, 18: 24}


def cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--storm', default=STORM, help='Identified original Figure S1 cyclone')
    ap.add_argument('--era5-dir', type=Path, default=Path('data/era5/msl/2010'))
    ap.add_argument('--ifs-dir', type=Path, default=Path('data/forecast/2010'))
    ap.add_argument('--hour-dir', type=Path, default=Path('data/era5/hour/2010'))
    ap.add_argument('--output-dir', type=Path, default=Path('outputs/figure_s1'))
    ap.add_argument('--raw-indices', nargs=4, type=int, default=[4, 5, 6, 7])
    ap.add_argument('--hour-order', nargs=4, type=int, default=list(EXPECTED_RAW_HOURS))
    ap.add_argument('--radius-deg', type=float, default=8.0)
    ap.add_argument('--legacy-square-8', action='store_true', help='Use original +/-8 grid-cell square minimum search')
    ap.add_argument('--grid-deg', type=float, default=1.25)
    ap.add_argument('--era5-marker', choices=['minimum', 'center'], default='minimum',
                    help='Original Figure S1 uses the local ERA5 minimum (blue X)')
    ap.add_argument('--contour-hpa', type=float, default=4.0,
                    help='Shared pressure contour interval')
    ap.add_argument('--colorbar-pa', type=float, default=None,
                    help='Optional explicit symmetric color scale +/- value (Pa)')
    ap.add_argument('--dpi', type=int, default=300)
    return ap.parse_args()


def load_storm_array(folder: Path, stem: str):
    candidates = [folder/(stem+'.npy'), folder/stem]
    for path in candidates:
        if path.is_file():
            return np.load(path, mmap_mode='r'), path
    raise FileNotFoundError(f'Not found: {stem}.npy in {folder}')


def hourly_value(H, raw):
    if raw >= len(H):
        raise IndexError(f'Hour metadata has only {len(H)} frames, requested raw index {raw}')
    v = np.asarray(H[raw]).flat[0]
    if not np.isfinite(v) or abs(v-round(float(v)))>1e-4:
        raise ValueError(f'Invalid hour at raw {raw}: {v}')
    return int(round(float(v)))


def to_pa(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    v = float(np.nanmedian(arr))
    if 800 < v < 1200:  # hPa -> Pa, if needed
        return 100*arr
    if 8e4 < v < 1.2e5:
        return arr
    raise ValueError(f'Pressure units unclear: median={v}; expected Pa (~100000) or hPa (~1000)')


def grid_coords(shape, cell):
    ny,nx = shape
    x = (np.arange(nx)-(nx-1)/2)*cell
    y = (np.arange(ny)-(ny-1)/2)*cell
    return x,y


def find_min(field, grid_deg, radius_deg, legacy_square):
    ny,nx=field.shape
    cy,cx=ny//2,nx//2
    # The same grid-index center convention as the notebook, not a new tracked position.
    dy,dx=np.ogrid[:ny,:nx]
    if legacy_square:
        mask=(np.abs(dy-cy)<=8)&(np.abs(dx-cx)<=8)
    else:
        mask=((dy-cy)*grid_deg)**2 + ((dx-cx)*grid_deg)**2 <= radius_deg**2 + 1e-8
    masked=np.where(mask & np.isfinite(field), field, np.inf)
    if not np.isfinite(masked).any():
        raise ValueError('No finite pressure within minimum search region')
    iy,ix=np.unravel_index(np.argmin(masked), field.shape)
    return (int(iy),int(ix)), float(field[iy,ix])


def nearest_ceil(n, unit):
    return float(unit*math.ceil(n/unit))


def main():
    a=cli()
    if a.contour_hpa <= 0 or a.grid_deg<=0 or a.radius_deg<=0:
        raise SystemExit('Grid spacing, radius and contour interval must be positive')
    if len(set(a.hour_order))!=4:
        raise SystemExit('Expected four distinct raw hours')
    F_all, ifs_path=load_storm_array(a.ifs_dir,a.storm)
    R_all, era_path=load_storm_array(a.era5_dir,a.storm)
    H_all, hour_path=load_storm_array(a.hour_dir,a.storm)
    if F_all.ndim!=3 or R_all.ndim!=3 or F_all.shape!=R_all.shape:
        raise ValueError(f'IFS/ERA5 must be aligned 3D matrices, got {F_all.shape}, {R_all.shape}')
    if max(a.raw_indices)>=len(F_all) or min(a.raw_indices)<0:
        raise IndexError(f'Requested {a.raw_indices}, available 0..{len(F_all)-1}')
    print('[DATA] IFS:', ifs_path)
    print('[DATA] ERA5:', era_path)
    print('[DATA] hours:', hour_path)
    frames=[]
    for raw in a.raw_indices:
        hour=hourly_value(H_all,raw)
        F=to_pa(F_all[raw]); R=to_pa(R_all[raw])
        if not np.all(np.isfinite(F)) or not np.all(np.isfinite(R)):
            raise ValueError(f'Non-finite pressure in raw frame {raw}')
        fpos, fmin=find_min(F,a.grid_deg,a.radius_deg,a.legacy_square_8)
        rpos, rmin=find_min(R,a.grid_deg,a.radius_deg,a.legacy_square_8)
        cy,cx=R.shape[0]//2,R.shape[1]//2
        frames.append(dict(raw=raw, hour=hour, lead=LEAD_BY_RAW_HOUR.get(hour),
                           F=F, R=R, delta=F-R, fpos=fpos, rpos=rpos,
                           fmin=fmin, rmin=rmin, era5_center_pa=float(R[cy,cx]),
                           era5_center_pos=(cy,cx)))
        print(f'[FRAME] raw={raw:2d} hour={hour:02d} UTC lead={LEAD_BY_RAW_HOUR.get(hour)} h  '
              f'ERA5 center={R[cy,cx]/100:.2f} hPa  '
              f'ERA5 min={rmin/100:.2f} hPa  IFS min={fmin/100:.2f} hPa')

    hours=[f['hour'] for f in frames]
    if len(set(hours))!=4:
        raise ValueError(f'Hour metadata unexpectedly repeated: {hours}')
    if set(hours)!=set(a.hour_order):
        raise ValueError(f'The requested hours {a.hour_order} are not in selected raw indices {a.raw_indices}: found {hours}')
    by_hour={f['hour']:f for f in frames}
    panels=[by_hour[h] for h in a.hour_order]
    if a.hour_order==list(EXPECTED_RAW_HOURS):
        assert [p['raw'] for p in panels]==[4,5,6,7] or a.raw_indices != [4,5,6,7]
    a.output_dir.mkdir(parents=True,exist_ok=True)

    xmax=max(np.max(np.abs(p['delta'])) for p in panels)
    vmax= a.colorbar_pa if a.colorbar_pa is not None else nearest_ceil(xmax,25)
    if vmax<=0: vmax=25.
    # Use the same absolute MSLP isobars (not per-panel auto contours), so
    # changes in the location of a contour really represent pressure changes.
    vmin_abs=min(np.min(p['F']) for p in panels)
    vmin_abs=min(vmin_abs,*(np.min(p['R']) for p in panels))
    vmax_abs=max(np.max(p['F']) for p in panels)
    vmax_abs=max(vmax_abs,*(np.max(p['R']) for p in panels))
    step=a.contour_hpa*100
    lo=math.floor(vmin_abs/step)*step
    hi=math.ceil(vmax_abs/step)*step
    levels=np.arange(lo,hi+step*.5,step)
    if len(levels)<2: levels=np.linspace(vmin_abs-1,vmax_abs+1,3)

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
                         'axes.titlesize':15, 'axes.labelsize':13,
                         'xtick.labelsize':11, 'ytick.labelsize':11,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes=plt.subplots(2,2,figsize=(11.8,10.8),
                           sharex=True,sharey=True)
    fig.subplots_adjust(left=.082,right=.875,bottom=.072,top=.875,hspace=.16,wspace=.12)
    tags='abcd'; im=None
    for i,(ax,p) in enumerate(zip(axes.flat,panels)):
        R,F=p['R'],p['F']
        xx,yy=grid_coords(R.shape,a.grid_deg)
        ext=(xx[0]-a.grid_deg/2,xx[-1]+a.grid_deg/2,
             yy[0]-a.grid_deg/2,yy[-1]+a.grid_deg/2)
        im=ax.imshow(p['delta'],cmap='RdBu_r',origin='lower',interpolation='nearest',
                     extent=ext,vmin=-vmax,vmax=vmax,rasterized=True)
        ax.contour(xx,yy,R,levels=levels,colors='#1547ff',linestyles='solid',linewidths=1.35)
        ax.contour(xx,yy,F,levels=levels,colors='#f33b4a',linestyles='dashed',linewidths=1.30)
        era_pos=p['rpos'] if a.era5_marker=='minimum' else p['era5_center_pos']
        bx,by=xx[era_pos[1]],yy[era_pos[0]]
        rx,ry=xx[p['fpos'][1]],yy[p['fpos'][0]]
        # Marker order reproduces Figure S1 (red over blue when coincident).
        ax.scatter([bx],[by],marker='x',s=115,linewidths=2.5,c='#1547ff',zorder=7)
        ax.scatter([rx],[ry],marker='x',s=115,linewidths=2.5,c='#fb1625',zorder=8)
        ax.set_xlim(-10,10); ax.set_ylim(-10,10)
        ax.set_aspect('equal',adjustable='box')
        ax.set_xticks([-10,-5,0,5,10]); ax.set_yticks([-10,-5,0,5,10])
        ax.tick_params(direction='out',length=4,width=.85)
        lead=f"{p['lead']}-h lead" if p['lead'] is not None else 'lead unknown'
        ax.set_title(f"({tags[i]})  {p['hour']:02d} UTC  ({lead})",pad=7)
    axes[0,0].set_ylabel('Degrees'); axes[1,0].set_ylabel('Degrees')
    axes[1,0].set_xlabel('Degrees'); axes[1,1].set_xlabel('Degrees')
    fig.suptitle('ERA5 and IFS sea-level pressure',fontsize=17,y=.970)

    # Original Figure S1 style: a legend INSIDE the upper-right corner
    # of each panel (blue solid ERA5 and red dashed IFS).
    elements = [
        Line2D([0], [0], color='#1547ff', ls='-', lw=1.9, label='ERA5'),
        Line2D([0], [0], color='#f33b4a', ls='--', lw=1.9, label='IFS'),
    ]
    for ax in axes.flat:
        ax.legend(handles=elements, loc='upper right', frameon=True,
                  fontsize=12, handlelength=2.5, framealpha=0.87)

    cax=fig.add_axes([.902,.155,.022,.635])
    cb=fig.colorbar(im,cax=cax)
    cb.set_label(r'$\Delta\mathrm{MSL}$ (Pa): IFS $-$ ERA5',fontsize=13,labelpad=12)
    cb.ax.tick_params(labelsize=11)
    if vmax<=300: cb.set_ticks(np.linspace(-vmax,vmax,5))

    stem='FigureS1_ERA5_IFS_18_00_06_12'
    png=a.output_dir/(stem+'.png'); pdf=a.output_dir/(stem+'.pdf')
    fig.savefig(png,dpi=a.dpi,bbox_inches='tight',pad_inches=.09)
    fig.savefig(pdf,bbox_inches='tight',pad_inches=.09)
    plt.close(fig)

    # Save each subfigure INDEPENDENTLY for LaTeX / Overleaf. A single colorbar
    # is exported separately so the scale remains shared after assembly.
    for i, p in enumerate(panels):
        R, F = p['R'], p['F']
        xx, yy = grid_coords(R.shape, a.grid_deg)
        ext = (xx[0]-a.grid_deg/2, xx[-1]+a.grid_deg/2,
               yy[0]-a.grid_deg/2, yy[-1]+a.grid_deg/2)
        subfig, ax = plt.subplots(figsize=(5.4, 5.35))
        subfig.subplots_adjust(left=.16, right=.985, bottom=.13, top=.88)
        ax.imshow(p['delta'], cmap='RdBu_r', origin='lower',
                  interpolation='nearest', extent=ext,
                  vmin=-vmax, vmax=vmax, rasterized=True)
        ax.contour(xx, yy, R, levels=levels, colors='#1547ff',
                   linestyles='solid', linewidths=1.35)
        ax.contour(xx, yy, F, levels=levels, colors='#f33b4a',
                   linestyles='dashed', linewidths=1.30)
        era_pos = p['rpos'] if a.era5_marker == 'minimum' else p['era5_center_pos']
        bx, by = xx[era_pos[1]], yy[era_pos[0]]
        rx, ry = xx[p['fpos'][1]], yy[p['fpos'][0]]
        ax.scatter([bx], [by], marker='x', s=115, linewidths=2.5,
                   c='#1547ff', zorder=7)
        ax.scatter([rx], [ry], marker='x', s=115, linewidths=2.5,
                   c='#fb1625', zorder=8)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_yticks([-10, -5, 0, 5, 10])
        ax.tick_params(direction='out', length=4, width=.85)
        ax.set_xlabel('Degrees')
        ax.set_ylabel('Degrees')
        lead = f"{p['lead']}-h lead" if p['lead'] is not None else 'lead unknown'
        ax.set_title(f"({tags[i]})  {p['hour']:02d} UTC  ({lead})", pad=7)
        ax.legend(handles=elements, loc='upper right', frameon=True,
                  fontsize=12, handlelength=2.5, framealpha=0.87)
        panel_stem = f'FigureS1_{tags[i]}_{p["hour"]:02d}UTC'
        for extension in ('pdf', 'png'):
            outfile = a.output_dir / f'{panel_stem}.{extension}'
            kwargs = {'dpi': a.dpi} if extension == 'png' else {}
            subfig.savefig(outfile, bbox_inches='tight', pad_inches=.06, **kwargs)
            print('[SAVED]', outfile)
        plt.close(subfig)

    # Standalone horizontal colorbar, to place under the four panels in Overleaf.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    colorfig = plt.figure(figsize=(9, 1.25))
    cax_single = colorfig.add_axes([.075, .53, .85, .27])
    sm = ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap='RdBu_r')
    sm.set_array([])
    cb_single = colorfig.colorbar(sm, cax=cax_single, orientation='horizontal')
    cb_single.ax.tick_params(labelsize=13)
    if vmax <= 300:
        cb_single.set_ticks(np.linspace(-vmax, vmax, 5))
    cb_single.set_label(r'$\Delta\mathrm{MSL}$ (Pa): IFS $-$ ERA5', fontsize=14, labelpad=4)
    for extension in ('pdf', 'png'):
        outfile = a.output_dir / f'FigureS1_shared_colorbar.{extension}'
        kwargs = {'dpi': a.dpi} if extension == 'png' else {}
        colorfig.savefig(outfile, bbox_inches='tight', pad_inches=.04, **kwargs)
        print('[SAVED]', outfile)
    plt.close(colorfig)

    csv_path=a.output_dir/(stem+'_values.csv')
    with csv_path.open('w',newline='',encoding='utf-8') as fp:
        wr=csv.DictWriter(fp,fieldnames=['panel','storm_key','raw_index','raw_hour_utc',
           'lead_hours_per_previous_mapping','era5_center_hpa','era5_local_min_hpa',
           'ifs_local_min_hpa','ifs_min_minus_era5_center_hpa',
           'ifs_min_minus_era5_local_min_hpa','era5_min_y','era5_min_x','ifs_min_y','ifs_min_x'])
        wr.writeheader()
        for i,p in enumerate(panels):
            wr.writerow(dict(panel=tags[i],storm_key=a.storm,raw_index=p['raw'],
             raw_hour_utc=p['hour'],lead_hours_per_previous_mapping=p['lead'],
             era5_center_hpa=p['era5_center_pa']/100,
             era5_local_min_hpa=p['rmin']/100,ifs_local_min_hpa=p['fmin']/100,
             ifs_min_minus_era5_center_hpa=(p['fmin']-p['era5_center_pa'])/100,
             ifs_min_minus_era5_local_min_hpa=(p['fmin']-p['rmin'])/100,
             era5_min_y=p['rpos'][0],era5_min_x=p['rpos'][1],
             ifs_min_y=p['fpos'][0],ifs_min_x=p['fpos'][1]))
    print('[SAVED]',png)
    print('[SAVED]',pdf)
    print('[SAVED]',csv_path)
    print('[ORDER]',[(p['raw'],p['hour'],p['lead']) for p in panels])


if __name__=='__main__':
    main()
