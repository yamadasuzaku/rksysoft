#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Constants
FILE_PATH = "savenpz/"
FILES = [
    "qlff__EVENTS__PHA.npz",
    "qlff__EVENTS__AC_ITYPE.npz",
    "qlff__ORBIT__LAT.npz",
    "qlff__ORBIT__LON.npz"
]
TIME_50MK = 150526800.0
LON_MIN = 100
LON_MAX = 200
LON_CUT = False

def load_data(filename):
    ldata = np.load(filename, allow_pickle=True)
    return ldata["time"], ldata["data"]

def convert_longitude(longitude):
    return longitude - 360 if longitude > 180 else longitude

def plot_data(pha_b50mk, pha_a50mk, xscale, yscale, bin_count, range_min, range_max, outfname="output.png"):
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel("PHA")
    plt.grid(alpha=0.8)
    plt.figtext(0.1, 0.95, "time50mK = 150526800.0 (2023-10-09T05:00:00)")
    if LON_CUT:
        plt.figtext(0.1, 0.90, f"{LON_MIN} < longitude < {LON_MAX}")
    else:
        plt.figtext(0.1, 0.90, "no longitude cut")
    plt.hist(pha_b50mk, bins=bin_count, range=(range_min, range_max), histtype='step', label="TIME < time50mK")
    plt.hist(pha_a50mk, bins=bin_count, range=(range_min, range_max), histtype='step', label="TIME > time50mK")
    plt.legend(loc="lower left")
    plt.savefig(outfname)
    plt.show()

def setup_heatmap_plot(ax, title):
    ax.set_extent([-180, 180, -60, 60])
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    plt.title(title)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.tight_layout()

def plot_heatmap_data(lon_data, lat_data, title_prefix, outfname):
    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    setup_heatmap_plot(ax, f"Antico counts : {title_prefix} time50mK = {TIME_50MK} (2023-10-09T05:00:00), no PHA cut, AC_ITYPE==0")
    converted_lons = [convert_longitude(lon) for lon in lon_data]
    H, xedges, yedges = np.histogram2d(converted_lons, lat_data, bins=[120, 29], range=[[-180, 180], [-31.15, 31.15]])
    pcm = plt.pcolormesh(xedges, yedges, H.T, norm=LogNorm(vmin=1, vmax=np.amax(H)), transform=ccrs.PlateCarree())
    plt.colorbar(pcm, ax=ax, shrink=0.4, label='Number of events')
    plt.tight_layout()
    plt.savefig(title_prefix + "_" + outfname)
    plt.show()

def main():
    # Load Data
    time_pha, pha = load_data(FILE_PATH + FILES[0])
    time_acitype, acitype = load_data(FILE_PATH + FILES[1])
    time_lat, lat = load_data(FILE_PATH + FILES[2])
    time_lon, lon = load_data(FILE_PATH + FILES[3])

    valid_indices = np.where((acitype == 0) & (time_pha > 0) & (pha > 0))[0]
    time_pha, pha = time_pha[valid_indices], pha[valid_indices]
    lon_interp, lat_interp = np.interp(time_pha, time_lon, lon), np.interp(time_pha, time_lat, lat)

    if LON_CUT:
        valid_lons = np.where((lon_interp > LON_MIN) & (lon_interp < LON_MAX))[0]
        time_pha, pha, lon_interp, lat_interp = time_pha[valid_lons], pha[valid_lons], lon_interp[valid_lons], lat_interp[valid_lons]

    before_cut, after_cut = np.where(time_pha < TIME_50MK)[0], np.where(time_pha > TIME_50MK)[0]
    pha_before, pha_after = pha[before_cut], pha[after_cut]
    lon_before, lon_after = lon_interp[before_cut], lon_interp[after_cut]
    lat_before, lat_after = lat_interp[before_cut], lat_interp[after_cut]

    # Plot
    plot_data(pha_before, pha_after, "linear", "linear", 100, 1, 1000, outfname="antico_spec_linlin.png")
    plot_data(pha_before, pha_after, "log", "linear", 1000, 1, 10000, outfname="antico_spec_loglin.png")
    plot_heatmap_data(lon_before, lat_before, "before", "antico_heatmap.png")
    plot_heatmap_data(lon_after, lat_after, "after", "antico_heatmap.png")

if __name__ == "__main__":
    main()
