#!/usr/bin/env python

import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob 
import astropy.io.fits
import sys

argvs = sys.argv  
argc = len(argvs)

if (argc != 2): 
    print('Usage: # python %s filename' % argvs[0])
    quit()  

import matplotlib.cm as cm
from matplotlib.colors import LogNorm

fname = argvs[1]
#fname = "xa00340356Arsl_p0px0000_cl.evt"

csvfilename="detx_dety_pixel.csv"
csvfile = open(csvfilename)

# detx_dety_pixel_list = []
# for line in csvfile:
#     x, y, z = map(int, line.split(","))
#     detx_dety_pixel_list.append([x, y, z])

# 2次元リストの初期化
pixel_fromdetxdety = [[0 for _ in range(6)] for _ in range(6)]  # 6x6のリストを0で初期化 

# 各行を処理
#for line in data_str.strip().split("\n"):
for line in csvfile:	
    i, j, k = map(int, line.split(","))
    pixel_fromdetxdety[i-1][j-1] = k

fits = astropy.io.fits.open(fname)
date_obs = fits[1].header["DATE-OBS"]

data = fits[1].data

times = data["TIME"]
pixel = data["PIXEL"]
detx = data["DETX"]
dety = data["DETY"]
pha = data["PHA"]

detx_list = set(detx)
dety_list = set(dety)

num = 0
counts = np.zeros([6,6]) 

print("#, detx, dety, pixel, count)")  
for one_detx in np.arange(1,7):
    for one_dety in np.arange(1,7):

        cutid = np.where( (detx == one_detx) & (dety == one_dety))[0]

        one_pixel = pixel_fromdetxdety[one_detx-1][one_dety-1]
        one_count = len(pha[cutid]) 
        counts[one_detx-1][one_dety-1] = int(one_count)
        print(num, one_detx, one_dety, one_pixel, one_count) 
        num += 1 

xbins = np.linspace(0.5,6.5,7)
ybins = np.linspace(0.5,6.5,7)

F = plt.figure(figsize=(10,4))

ax = plt.subplot(1,2,1)

plt.figtext(0.05, 0.92, fname + " DATE-OBS : " + date_obs)

pcm = plt.pcolormesh(xbins, ybins, counts, norm=LogNorm(vmin=1, vmax=counts.max()), cmap="plasma")

for one_detx in np.arange(1,7):
    for one_dety in np.arange(1,7):
        text = ax.text(one_detx -0.3, one_dety + 0.3, pixel_fromdetxdety[one_detx-1][one_dety-1],
                       ha="center", va="center", color="0.9", size = 8)

for one_detx in np.arange(1,7):
    for one_dety in np.arange(1,7):
        text = ax.text(one_detx, one_dety, '%d' % counts[one_detx-1][one_dety-1],
                       ha="center", va="center", color="k")


plt.colorbar(pcm)
plt.xlabel("DETX")
plt.ylabel("DETY")

ax = plt.subplot(1,2,2)
pcm = plt.pcolormesh(xbins, ybins, counts, cmap="plasma")
plt.colorbar(pcm)

for one_detx in np.arange(1,7):
    for one_dety in np.arange(1,7):
        text = ax.text(one_detx -0.3, one_dety + 0.3, pixel_fromdetxdety[one_detx-1][one_dety-1],
                       ha="center", va="center", color="0.9", size = 8)

for one_detx in np.arange(1,7):
    for one_dety in np.arange(1,7):
        text = ax.text(one_detx, one_dety, '%d' % counts[one_detx-1][one_dety-1],
                       ha="center", va="center", color="k")


plt.xlabel("DETX")
plt.ylabel("DETY")
outfname = fname.replace(".evt",".png").replace(".gz","")
plt.savefig(outfname)
plt.show()
