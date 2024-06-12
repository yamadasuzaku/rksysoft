#!/usr/bin/env python

import argparse
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description='Process FITS file and generate a PNG plot.')
parser.add_argument('filename', type=str, help='FITS file name')
parser.add_argument('--binnum', '-b', type=int, default=200, help='Number of bins for histogram')
parser.add_argument('--margin', '-m', type=int, default=10, help='Margin for histogram range')
args = parser.parse_args()

# 入力引数を変数に割り当て
filename = args.filename
outfname = filename.split(".")[0] + ".png"
binnum = args.binnum
margin = args.margin
xmin = - margin 
xmax = 65535 + margin

# プロットのパラメータ設定
params = {'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 7}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

itypename = [0, 1, 2, 3, 4]
typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]

# FITS ファイルを開く
print(f"Opening FITS file: {filename}")
hdulist = fits.open(filename)

# データを読み込む
print("Reading data from FITS file")
data = hdulist[1].data
pha = hdulist[1].data["PHA"]
pha2 = hdulist[1].data["PHA2"]
itype = hdulist[1].data["ITYPE"]

# 'PHA2' カラムで NULL 値 (2147483647) の位置を検索する
print("Filtering data with PHA2 == 2147483647 (NULL)")
null_positions = np.where(pha2 == 2147483647)
pha = pha[null_positions]
itype = itype[null_positions]

plt.figure(figsize=(11, 7))
plt.subplots_adjust(right=0.8) # make the right space bigger
plt.xscale("linear")
plt.yscale("log")
plt.ylabel("Counts/bin")
plt.xlabel("PHA")
plt.grid(alpha=0.8)
plt.title(f"{filename} filtered with pha2 == 2147483647 (NULL)")

for itype_ in itypename:
    print(f"Processing itype {typename[itype_]} (ITYPE={itype_})")
    # Filter data by itype
    typecut = (itype == itype_)
    subpha = pha[typecut]
    event_number = len(subpha)
    print(f"Number of events: {event_number}")
    # Compute histogram for all pixels of current itype
    hist, binedges = np.histogram(subpha, bins=binnum, range=(xmin, xmax))
    bincenters = 0.5 * (binedges[1:] + binedges[0:-1])
    plt.errorbar(bincenters, hist, yerr=np.sqrt(hist), fmt='-', label=typename[itype_] + "("+str(event_number)+ "c)", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=6)

ofname = f"fig_{typename[itype_]}_{outfname}"
print(f"Saving plot to {ofname}")
plt.savefig(ofname)
plt.show()
print(f"..... {ofname} is created.")
