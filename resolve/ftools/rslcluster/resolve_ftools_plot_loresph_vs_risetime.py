#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# 定数の設定
g_typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]
itype_interest = [0, 1, 2, 3, 4]
npix = 36

# ピクセルごとのマーカーと色（36種）
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * 3
colors = plt.cm.tab20(np.linspace(0, 1, npix))

def load_fits_data(filename, xcol, ycol):
    with fits.open(filename) as hdul:
        data = hdul[1].data
        itype = data['ITYPE']
        pixel = data['PIXEL']
        x_data = data[xcol]
        y_data = data[ycol]
    return itype, pixel, x_data, y_data

def plot_by_itype(itype_data, pixel_data, x_data, y_data, xcol, ycol, xrange):
    for itype_val in itype_interest:
        mask_itype = (itype_data == itype_val)
        plt.figure(figsize=(12, 8))
        for pix in range(npix):
            mask_pix = (pixel_data == pix)
            mask = mask_itype & mask_pix
            if np.sum(mask) == 0:
                continue
            x = x_data[mask]
            y = y_data[mask]
            plt.scatter(x, y, s=8, alpha=0.6, label=f'Pixel {pix}', marker=markers[pix], color=colors[pix])

            # x軸の範囲内にあるデータの平均を算出して表示
            if xrange:
                mask_range = mask & (x_data >= xrange[0]) & (x_data <= xrange[1])
                if np.sum(mask_range) > 0:
                    avg_val = np.mean(x_data[mask_range])
                    plt.text(xrange[1] + 10 + (pix // 4) * 20, 5 + (pix % 4) * 10,
                             f"{pix}: {avg_val:.1f}", fontsize=8, color=colors[pix])

        plt.title(f"{xcol} vs. {ycol} for ITYPE={itype_val} ({g_typename[itype_val]})")
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        if xrange:
            plt.xlim(xrange)
        plt.grid(True)
        plt.legend(ncol=4, fontsize=7, loc='upper right', bbox_to_anchor=(1.0, 1))
        plt.tight_layout()
        output_name = f"plot_{xcol}_vs_{ycol}_itype{itype_val}.png"
        plt.savefig(output_name)
        plt.show()
        print(f"Saved: {output_name}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Scatter plot of xcol vs. ycol by ITYPE and PIXEL.")
    parser.add_argument("fits_file", help="Input FITS file")
    parser.add_argument("--xcol", default="LO_RES_PH", help="Column name for x-axis (default: LO_RES_PH)")
    parser.add_argument("--ycol", default="RISE_TIME", help="Column name for y-axis (default: RISE_TIME)")
    parser.add_argument("--xrange", nargs=2, type=int, metavar=('XMIN', 'XMAX'),
                        help="X-axis range (e.g. --xrange 12100 12300)")
    args = parser.parse_args()

    itype, pixel, x_data, y_data = load_fits_data(args.fits_file, args.xcol, args.ycol)
    plot_by_itype(itype, pixel, x_data, y_data, args.xcol, args.ycol, args.xrange)

if __name__ == "__main__":
    main()
