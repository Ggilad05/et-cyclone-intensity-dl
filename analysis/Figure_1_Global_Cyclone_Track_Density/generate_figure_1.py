# Figure 1 — Global Cyclone Track Density
# Source adapted from the provided plotting script.

import matplotlib.pyplot as plt

import matplotlib.patheffects as pe

import matplotlib.ticker as mticker

import sys

import xarray as xr

import numpy as np

import glob

import re

import cartopy.crs as ccrs

import cartopy.feature as cfeature

from cartopy.feature import ShapelyFeature

from shapely.geometry import Polygon, LineString



# --- AMS Journal Formatting Settings ---

plt.rcParams.update({

    "font.family": "serif",                # שימוש בגופן בעל עוקצים (Serif)

    "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"], # התאמה לתבנית של AMS

    "text.usetex": False,                  # שנה ל-True רק אם מותקנת לך סביבת LaTeX מקומית שעובדת עם פייתון

    "mathtext.fontset": "stix",            # עיצוב סמלים מתמטיים שייראה כמו LaTeX

    "axes.labelsize": 14,

    "xtick.labelsize": 12,

    "ytick.labelsize": 12,

    "legend.fontsize": 12,

})



# --- 1. EXISTING FUNCTIONS (UNTOUCHED) ---



def mean_longitude(lon_list):

    lon_radians = np.radians(lon_list)

    mean_sin = np.mean(np.sin(lon_radians))

    mean_cos = np.mean(np.cos(lon_radians))

    mean_angle_rad = np.arctan2(mean_sin, mean_cos)

    mean_lon = np.degrees(mean_angle_rad) % 360

    return mean_lon





def prepare_track_density(lat, lon):

    tapx = 9

    tapy = 3

    ysize = int((180 / tapy)) + 1

    xsize = int((360 / tapx)) + 1

    density = np.zeros([ysize, xsize])

    for i in range(len(lat)):

        x = (lon[i] / tapx).round().astype('int')

        y = ((lat[i] + 90) / tapy).round().astype('int')

        density[y, x] += 1

    return density





def pre_tracks(file_path):

    def filter_tracks(ds):

        nonz = (ds.t.data != 0).sum(axis=0)

        E = ds.intensity.data

        b = np.argmax(E, axis=0)

        lati = np.abs(ds.lat.data[0])

        loni = ds.lon.data[0]

        lonm = ds.lon.data[b, np.arange(len(b))]

        dlon = np.mod(lonm - loni, 360)

        dldt = dlon / (b + 1)

        ind = (lati < 60) & (lati > 20) & (nonz > 16) & (dldt > 0.3) & (dlon < 200)

        return ds.isel(trackid=ind)



    ds = xr.open_dataset(file_path).load().data

    names = ['t', 'lon', 'lat', 'intensity']

    datasets = []

    for i in range(4):

        datasets.append(ds[i, :, :].to_dataset(name=names[i]).drop('variables'))

    ds = xr.merge(datasets)



    return filter_tracks(ds)





# --- 2. MAIN EXECUTION ---



if __name__ == "__main__":

    # --- Settings ---

    folder_path = '/arctic2/shreibshtein/MasterProject/OrCyclones/ERA_5_Track'

    tapx = 5

    tapy = 5



    # Setup global grid

    lon_edges = np.arange(0, 360 + tapx, tapx)

    lat_edges = np.arange(-90, 90 + tapy, tapy)

    ysize = len(lat_edges) - 1

    xsize = len(lon_edges) - 1

    global_density = np.zeros([ysize, xsize])



    # --- DEFINE POLYGONS ---

    # 1. North Pacific: Defined as 120 to 240 (which wraps to 120E-120W)

    # 2. Southern Ocean: Defined 0-360 for statistics logic

    polygons = {

        "North Atlantic": Polygon([(-80, 35), (-10, 35), (-10, 70), (-80, 70)]),



        "North Pacific": Polygon([(120, 30), (240, 30), (240, 60), (120, 60)]),



        "Mediterranean": Polygon([(-10, 30), (50, 30), (50, 45), (-10, 45)]),

        "South Atlantic": Polygon([(-65, -70), (0, -70), (0, -20), (-65, -20)]),



        "Southern Ocean": Polygon([(0, -70), (360, -70), (360, -30), (0, -30)]),

    }



    # Map for Table Rows

    table_regions = {

        "North Atlantic": "North Atlantic",

        "North Pacific": "North Pacific",

        "Mediterranean": "Mediterranean",

        "South Atlantic": "South Atlantic",

        "South Ocean": "Southern Ocean"

    }



    # Initialize Stats Dictionary

    stats = {

        'NH': {k: {'storms': 0, 'steps': 0} for k in ['DJF', 'MAM', 'JJA', 'SON'] + list(table_regions.keys())},

        'SH': {k: {'storms': 0, 'steps': 0} for k in ['DJF', 'MAM', 'JJA', 'SON'] + list(table_regions.keys())}

    }



    # Global Totals

    global_total_storms = 0

    global_total_steps = 0



    file_list = glob.glob(folder_path + "/*.nc")

    year_pattern = re.compile(r'_(\d{4})_')

    counter = 0



    print("Starting processing...")



    for file in file_list:

        filename = file.split('/')[-1]



        # 1. Extract Year

        match = year_pattern.search(filename)

        if match:

            year = int(match.group(1))

        else:

            continue



        if (year > 2020) or (year <= 1979):

            continue



        # 2. Extract Season

        file_season = filename.split("_")[0]

        if file_season not in ['DJF', 'MAM', 'JJA', 'SON']:

            continue



        print(f"Processing: {year} - {file_season}")



        data = pre_tracks(file)

        num_tracks = data["trackid"].values

        unique_tracks = np.unique(num_tracks)



        for num in unique_tracks:

            counter += 1



            track = data.sel(trackid=num)

            lons = track['lon'].to_numpy()

            lats = track['lat'].to_numpy()



            valid_mask = (lons != 0) & (lats != 0) & (~np.isnan(lons))

            lons = lons[valid_mask]

            lats = lats[valid_mask]



            if len(lons) == 0:

                continue



            steps_count = len(lons)

            global_total_storms += 1

            global_total_steps += steps_count



            # --- 1. GLOBAL DENSITY MAP CALC ---

            storm_density, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])

            storm_footprint = (storm_density > 0).astype(int)

            global_density += storm_footprint



            # --- 2. STATISTICS CALCULATION ---

            hemi = 'NH' if lats[0] >= 0 else 'SH'



            stats[hemi][file_season]['storms'] += 1

            stats[hemi][file_season]['steps'] += steps_count



            # --- C. Determine Region Intersection ---

            # 1. wrapped -180 to 180 (For Atlantic, Med)

            lons_180 = (lons + 180) % 360 - 180

            track_line_180 = LineString(zip(lons_180, lats))



            # 2. wrapped 0 to 360 (For NP & SO)

            lons_360 = lons % 360

            track_line_360 = LineString(zip(lons_360, lats))



            for table_name, poly_key in table_regions.items():

                poly = polygons[poly_key]



                if poly_key in ["North Pacific", "Southern Ocean"]:

                    if poly.intersects(track_line_360):

                        stats[hemi][table_name]['storms'] += 1

                        stats[hemi][table_name]['steps'] += steps_count

                else:

                    if poly.intersects(track_line_180):

                        stats[hemi][table_name]['storms'] += 1

                        stats[hemi][table_name]['steps'] += steps_count



    # --- FINAL OUTPUTS ---



    num_years = 2020 - 1979

    global_density_map = global_density / num_years



    # --- PRINT TABLE ---

    print("\n" + "=" * 80)

    print(f"{'CATEGORY':<20} | {'NH (STORMS)':<12} | {'NH (STEPS)':<12} | {'SH (STORMS)':<12} | {'SH (STEPS)':<12}")

    print("-" * 80)



    categories = ['DJF', 'MAM', 'JJA', 'SON', 'North Atlantic', 'North Pacific', 'Mediterranean', 'South Atlantic',

                  'South Ocean']



    for cat in categories:

        nh_storms = stats['NH'][cat]['storms']

        nh_steps = stats['NH'][cat]['steps']

        sh_storms = stats['SH'][cat]['storms']

        sh_steps = stats['SH'][cat]['steps']

        print(f"{cat:<20} | {nh_storms:<12} | {nh_steps:<12} | {sh_storms:<12} | {sh_steps:<12}")



    print("-" * 80)

    print(f"{'GLOBAL TOTAL':<20} | {global_total_storms:<27} | {global_total_steps:<27}")

    print("=" * 80 + "\n")



    # Save

    npy_filename = f'mean_1979_2020_{tapx}x{tapy}_trajectory.npy'

    np.save(npy_filename, global_density_map)

    print(f"Saved density data to {npy_filename}")



    # --- 3. PLOTTING ---



    # Center on 60E

    proj = ccrs.PlateCarree(central_longitude=60)



    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': proj})



    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='black', linewidth=0.5, zorder=1)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=2)

    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5, zorder=2)



    # Gridlines with explicit ticks

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,

                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    gl.top_labels = False

    gl.right_labels = False

    gl.xlabel_style = {"size": 15}

    gl.ylabel_style = {"size": 15}



    # Ticks covering the view [-120 to 240 relative to 0]

    gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120, 180, 240])



    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

    plot_density = np.ma.masked_where(global_density_map == 0, global_density_map)



    im = ax.pcolormesh(lon_centers, lat_centers, plot_density,

                       transform=ccrs.PlateCarree(), cmap='Greens', shading='auto')



    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)

    cbar.set_label('Track Density (systems per year)', fontsize=14)

    cbar.ax.tick_params(labelsize=12)



    # Label Overrides

    label_overrides = {

        "Southern Ocean": (65, -25),

        "North Pacific": (180, 65),

        "North Atlantic": (-45, 75),

        "Mediterranean": (20, 50),

        "South Atlantic": (327.5, -15)

    }



    for name, poly in polygons.items():

        # --- FIX: SPECIAL HANDLING FOR SOUTHERN OCEAN ---

        if name == "Southern Ocean":

            # Don't use the Polygon feature (which creates vertical lines).

            # Manually draw Top (-30) and Bottom (-70) lines across the world.

            # Using -180 to 360 ensures it covers the wrap.

            ax.plot([-120, 240], [-30, -30], transform=ccrs.PlateCarree(),

                    color='black', linewidth=2.0, zorder=3)

            ax.plot([-120, 240], [-70, -70], transform=ccrs.PlateCarree(),

                    color='black', linewidth=2.0, zorder=3)



        else:

            # Standard Polygon drawing for other regions

            feat = ShapelyFeature([poly], crs=ccrs.PlateCarree(), edgecolor="black", facecolor="none", linewidth=2.0,

                                  zorder=3)

            ax.add_feature(feat)



        # Labels

        if name in label_overrides:

            cx, cy = label_overrides[name]

        else:

            cx, cy = poly.centroid.x, poly.centroid.y



        ax.text(cx, cy, name,

                fontsize=12, fontweight='bold', color='black',

                transform=ccrs.PlateCarree(),

                ha="center", va="center", zorder=10,

                path_effects=[pe.withStroke(linewidth=3, foreground="white")])



    out_png = f'mean_1979_2020_{tapx}x{tapy}_trajectory_60E_fixed.png'

    out_pdf = f'mean_1979_2020_{tapx}x{tapy}_trajectory_60E_fixed.pdf'

    plt.savefig(out_pdf, format='pdf', bbox_inches='tight') # שמירה וקטורית

    plt.savefig(out_png, dpi=600, bbox_inches='tight')



    print(f"Figure saved to {out_png}")

    plt.show()