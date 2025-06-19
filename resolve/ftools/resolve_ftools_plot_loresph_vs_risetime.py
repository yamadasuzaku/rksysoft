#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# 定数の設定
g_typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]
itype_interest = [0, 1, 2, 3, 4]
npix = 36
LO_RES_PH_range = (12220, 12300)

# ピクセルごとのマーカーと色（36種）
markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'] * 3
colors = plt.cm.tab20(np.linspace(0, 1, npix))

def load_fits_data(filename, rise_time_col):
    with fits.open(filename) as hdul:
        data = hdul[1].data
        itype = data['ITYPE']
        pixel = data['PIXEL']
        lo_res_ph = data['LO_RES_PH']
        rise_time = data[rise_time_col]
    return itype, pixel, lo_res_ph, rise_time

def plot_by_itype(itype_data, pixel_data, lo_res_ph_data, rise_time_data, rise_time_col):
    for itype_val in itype_interest:
        mask_itype = (itype_data == itype_val)
        plt.figure(figsize=(12, 8))
        for pix in range(npix):
            mask_pix = (pixel_data == pix)
            mask = mask_itype & mask_pix
            if np.sum(mask) == 0:
                continue
            y = rise_time_data[mask]
            x = lo_res_ph_data[mask]
            plt.scatter(x, y, s=8, alpha=0.6, label=f'Pixel {pix}', marker=markers[pix], color=colors[pix])

            # LO_RES_PH が指定範囲内にあるデータの平均を算出して表示
            mask_range = mask & (lo_res_ph_data >= LO_RES_PH_range[0]) & (lo_res_ph_data <= LO_RES_PH_range[1])
            if np.sum(mask_range) > 0:
                avg_val = np.mean(lo_res_ph_data[mask_range])
                plt.text(LO_RES_PH_range[1] + 10 + (pix // 4) * 20, 5 + (pix % 4) * 10, 
                         f"{pix}: {avg_val:.1f}", fontsize=8, color=colors[pix])

        plt.title(f"LO_RES_PH vs. {rise_time_col} for ITYPE={itype_val} ({g_typename[itype_val]})")
        plt.xlabel(rise_time_col)
        plt.xlabel("LO_RES_PH")
        plt.ylabel("RISE_TIME")
        plt.xlim(LO_RES_PH_range)
        plt.grid(True)
        plt.legend(ncol=4, fontsize=7, loc='upper right', bbox_to_anchor=(0.9, 1))
        plt.tight_layout()
        output_name = f"plot_loresph_vs_{rise_time_col}_itype{itype_val}.png"
        plt.savefig(output_name)
        print(f"Saved: {output_name}")
        plt.show()
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Scatter plot of LO_RES_PH vs. RISE_TIME by ITYPE and PIXEL.")
    parser.add_argument("fits_file", help="Input FITS file")
    parser.add_argument("--rise_time_col", default="RISE_TIME", help="Column name for rise time (default: RISE_TIME)")
    args = parser.parse_args()

    itype, pixel, lo_res_ph, rise_time = load_fits_data(args.fits_file, args.rise_time_col)
    plot_by_itype(itype, pixel, lo_res_ph, rise_time, args.rise_time_col)

if __name__ == "__main__":
    main()
