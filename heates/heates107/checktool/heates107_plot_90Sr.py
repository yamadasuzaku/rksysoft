#!/usr/bin/env python 

import sys,os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import pickle
import itertools
from datetime import datetime
from scipy.ndimage import median_filter

import sys

if len(sys.argv) < 2:
	print("usage : python script.py pklfilename")
	sys.exit()
else:
	fname = sys.argv[1]

with open(fname,'rb') as f:
    arr=pickle.load(f)

grid1, grid2 = 2, 1 
xfigsize, yfigsize = 10, 8  
fig, axs = plt.subplots(grid1, grid2, figsize=(xfigsize, yfigsize), sharex=True, sharey=False)

counter=0
sampleperiod=16.38e-6
recordlen=3000
xtime = np.array([one*sampleperiod for one in range(recordlen)])
print(xtime)

ymin1, ymax1, ymin2, ymax2 = 1, 35000, -0.1, 12.3

ppeaks = []
pchans = []

for i, data in enumerate(arr):
	time = data[0]
	dtime = datetime.fromtimestamp(time)

	ch = data[1]
	rawpulse = data[2]
	peak_value = np.amax(rawpulse) - np.amin(rawpulse)

	if np.amax(rawpulse) > 0 :
		filt_pulse = median_filter(rawpulse, size=3)
		if np.amax(filt_pulse)-np.min(filt_pulse) > 0:
			normed_pulse = (filt_pulse - np.min(filt_pulse))/(np.amax(filt_pulse)-np.min(filt_pulse))  + ch * 0.1
		else:
			print("skip error : pulse <= 0, ", normed_pulse)
	else:
		print("skip error : pulse <= 0, ", rawpulse)
		continue 
	
	abs_pulse = rawpulse - np.min(rawpulse) + 5 * ch

	if i == 0: counter = time
	if time == counter:
		ax = axs[0]
		plt.figtext(0.1,0.9, fname + " : " + str(dtime))

		ax.plot(xtime, abs_pulse, alpha=0.8, lw=1)
		ax.set_ylim(ymin1, ymax1)		
		ax.set_ylabel("pulse")
		ax.set_yscale("log")

		ax = axs[1] 
		ax.plot(xtime, normed_pulse, alpha=0.8, lw=1)
		ax.set_ylim(ymin2, ymax2)		
		ax.set_xlabel("Time(s) from " + str(time))
		ax.set_ylabel("nomalized pulse")
		ppeaks.append(peak_value)
		pchans.append(ch)

	else:
		plt.show()
		plt.close()

		ppeaks = np.array(ppeaks)
		pchans = np.array(pchans)

		pcut = np.where(ppeaks > 1000)
		high_chan = pchans[pcut]
		high_chan_str = ",".join(map(str, high_chan))

		pcut = np.where(ppeaks <= 50)
		low_chan = pchans[pcut]
		low_chan_str = ",".join(map(str, low_chan))

		plt.figure(figsize=(10, 6))
		plt.figtext(0.1,0.95, fname + " : " + str(dtime))
		plt.figtext(0.1,0.92, "high ch (>1e3) : " + high_chan_str)
		plt.figtext(0.1,0.90, " low ch (<50)  : " + low_chan_str)

		plt.plot(pchans, ppeaks, "o-")		
		plt.grid(alpha=0.5)
		plt.yscale("log")
		plt.ylabel("peak_value")
		plt.xlabel("ch")
		plt.show()
		plt.close()
		ppeaks = []
		pchans = []

		counter = time
		fig, axs = plt.subplots(grid1, grid2, figsize=(xfigsize, yfigsize), sharex=True, sharey=False)
		ax = axs[0]
		plt.figtext(0.1,0.9, fname + " : " + str(dtime))

		ax.plot(xtime, abs_pulse, alpha=0.8, lw=1)
		ax.set_ylim(ymin1, ymax1)		
		ax.set_ylabel("pulse")

		ax = axs[1] 
		ax.plot(xtime, normed_pulse, alpha=0.8, lw=1)
		ax.set_ylim(ymin2, ymax2)		
		dtime = datetime.fromtimestamp(time)
		ax.set_xlabel("Time(s) from " + str(time))
		ax.set_ylabel("nomalized pulse")



