#!/usr/bin/env python 

import numpy as np
import matplotlib.pylab as plt
plt.rcParams['font.family'] = 'serif'
import glob

obsid_list=[]

fname="obsid.list"
outfname="nicer_cygx1_allspec_Fe.png"

xmin, xmax = 6, 8
ymin, ymax = 2, 200.0

for one in open(fname):
	obsid_list.append(one.strip())
print(obsid_list)	

def get_arrays_from_qdp(qdpfilename):
	header = 3
	qdpfile = open(qdpfilename)
	print(qdpfilename)
	x_list = []
	xe_list = []
	y_list = []
	ye_list = []

	for i, one in enumerate(qdpfile):
		if i < header:
			continue # skip header
		one = one.strip().split()
		print(i,one)
		x_list.append(float(one[0]))
		xe_list.append(float(one[1]))
		y_list.append(float(one[2]))
		ye_list.append(float(one[3]))

	x_list  = np.array(x_list)
	xe_list  = np.array(xe_list)
	y_list  = np.array(y_list)
	ye_list = np.array(ye_list)
	qdpfile.close()		

	return x_list, xe_list, y_list, ye_list


plt.figure(figsize=(7,12))

cmap = plt.get_cmap("tab10") 

ax1 = plt.subplot2grid((3,1), (0,0), rowspan=1)

ax1.grid(alpha=0.5,linestyle='dotted')
ax1.set_yscale("log")
ax1.set_xscale("linear")
ax1.set_ylabel(r"Counts s$^{-1}$ keV$^{-1}$ ")
#ax1.set_xlabel("Energy (keV)")
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)

ax1.axvline(x = 6.4, color = 'g', alpha=0.6, lw=1, label = 'Neutral Fe',linestyle='dotted')
ax1.axvline(x = 6.698, color = 'r', alpha=0.6, lw=1, label = 'He-like Fe (6.698 keV)',linestyle='dotted')
ax1.axvline(x = 6.966, color = 'r', alpha=0.6, lw=1, label = 'H-like Fe (6.966 keV)',linestyle='dotted')


ax2 = plt.subplot2grid((3,1), (1,0), rowspan=2)   

ax2.grid(alpha=0.5,linestyle='dotted')
ax2.set_yscale("linear")
ax2.set_xscale("linear")
ax2.set_ylabel(r"Ratio + offset")
ax2.set_xlabel("Energy (keV)")
ax2.set_xlim(xmin, xmax)
#ax2.set_ylim(0.8,1.2)

ax2.axvline(x = 6.4, color = 'g', alpha=0.6, lw=1,   label = None,linestyle='dotted')  
ax2.axvline(x = 6.698, color = 'r', alpha=0.6, lw=1, label = None,linestyle='dotted')
ax2.axvline(x = 6.966, color = 'r', alpha=0.6, lw=1, label = None,linestyle='dotted')


for i, obsid in enumerate(obsid_list):
	qdpfile = glob.glob("../" + obsid + "/*_plld.qdp")
	if not len(qdpfile) == 1:
		print("ERROR : more than one qdp file found", qdpfile)
	
	x_list, xe_list, y_list, ye_list = get_arrays_from_qdp(qdpfile[0])
	ecut = np.where( ((x_list) > xmin ) & ((x_list) < xmax )) 
	x_list  = x_list[ecut]
	xe_list = xe_list[ecut]
	y_list  = y_list[ecut] 
	ye_list = ye_list[ecut] 

	func4 = np.poly1d(np.polyfit(x_list, y_list, 4))


	ax1.errorbar(x_list, y_list, xerr=xe_list, yerr=ye_list, fmt=".", color=cmap(i), label=obsid)
	ax1.errorbar(x_list, func4(x_list), fmt="-", color=cmap(i), alpha=0.6, label=None) 
	ax1.legend(bbox_to_anchor=(0., 1.01, 1., 0.01), loc='lower left',ncol=3, borderaxespad=0.,fontsize=8)

	ax2.errorbar(x_list, 0.15*i + y_list/func4(x_list), fmt="-", alpha=0.3, color=cmap(i), label=None)
	ax2.errorbar(x_list, 0.15*i + y_list/func4(x_list), xerr=xe_list, yerr=ye_list/func4(x_list), fmt=".", color=cmap(i), label=None)


plt.savefig(outfname)
print(outfname + " is created. ")
plt.show()

