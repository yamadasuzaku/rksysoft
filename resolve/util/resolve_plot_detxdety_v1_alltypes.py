#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import astropy.io.fits
import sys
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

# Observation Type Information
itypename = [0, 1, 2, 3, 4]
typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]

def load_pixel_mapping_from_csv(filename="detx_dety_pixel.csv"):
    """
    CSVファイルからピクセルマッピングを読み込む関数
    """
    pixel_fromdetxdety = [[0 for _ in range(6)] for _ in range(6)]
    with open(filename, 'r') as csvfile:
        for line in csvfile:
            i, j, k = map(int, line.split(","))
            pixel_fromdetxdety[i-1][j-1] = k
    return pixel_fromdetxdety


def plot_data(counts, pixel_fromdetxdety, fname, date_obs, itype):
    """
    データをプロットする関数
    """
    xbins = np.linspace(0.5, 6.5, 7)
    ybins = np.linspace(0.5, 6.5, 7)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{fname} DATE-OBS : {date_obs} TYPE =" + str(typename[itype]))

    for ax, norm in zip(axs, [LogNorm(vmin=1, vmax=counts.max()), None]):
        pcm = ax.pcolormesh(xbins, ybins, counts.T, norm=norm, cmap="plasma")
        plt.colorbar(pcm, ax=ax)

        for one_detx in np.arange(1, 7):
            for one_dety in np.arange(1, 7):
                ax.text(one_detx - 0.3, one_dety + 0.3, pixel_fromdetxdety[one_detx-1][one_dety-1],
                        ha="center", va="center", color="0.9", size=8)

                ax.text(one_detx, one_dety, f'{int(counts[one_detx-1][one_dety-1])}',
                        ha="center", va="center", color="k", size=9)

        ax.set_xlabel("DETX")
        ax.set_ylabel("DETY")

    plt.savefig(str(typename[itype] + "_" + fname.replace(".evt", ".png").replace(".gz", "")))
    plt.show()

def main():

    if len(sys.argv) != 2:
        print(f'Usage: # python {sys.argv[0]} filename')
        sys.exit()

    fname = sys.argv[1]
    pixel_fromdetxdety = load_pixel_mapping_from_csv()

    with astropy.io.fits.open(fname) as fits:
        date_obs = fits[1].header["DATE-OBS"]
        data = fits[1].data

        times, pixel, detx, dety, pha, itype = data["TIME"], data["PIXEL"], data["DETX"], data["DETY"], data["PHA"], data["ITYPE"]

        counts = np.zeros([6, 6])

        print("#, detx, dety, pixel, count)")
        for oneitype in itypename:
            for one_detx in np.arange(1, 7):
                for one_dety in np.arange(1, 7):
                    cutid = np.where((detx == one_detx) & (dety == one_dety) & (itype == oneitype) )[0]
    #                cutid = np.where((detx == one_detx) & (dety == one_dety))[0]

                    one_pixel = pixel_fromdetxdety[one_detx-1][one_dety-1]
                    one_count = len(pha[cutid])
                    counts[one_detx-1][one_dety-1] = int(one_count)
                    print(one_detx-1, one_detx, one_dety, one_pixel, one_count)

            plot_data(counts, pixel_fromdetxdety, fname, date_obs, oneitype)

if __name__ == "__main__":
    main()
