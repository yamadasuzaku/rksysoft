#!/usr/bin/env python 

import numpy as np
import matplotlib.pylab as plt
import glob

obsid_list=[]

fname="obsid.list"
outfname="nicer_cygx1_allspec.png"

xmin, xmax = 0.5, 10.0
ymin, ymax = 0.1, 1.0e5 

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


fig, ax = plt.subplots(figsize=(10.0, 6.0))

plt.subplot(111)
plt.yscale("log")
plt.xscale("log")
plt.ylabel(r"Counts s$^{-1}$ keV$^{-1}$ ")
plt.xlabel("Energy (keV)")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

for obsid in obsid_list:
	qdpfile = glob.glob("../" + obsid + "/*_plld.qdp")
	if not len(qdpfile) == 1:
		print("ERROR : more than one qdp file found", qdpfile)
	
	x_list, xe_list, y_list, ye_list = get_arrays_from_qdp(qdpfile[0])
	plt.errorbar(x_list, y_list, xerr=xe_list, yerr=ye_list, fmt=".", label=obsid)
	plt.legend()

plt.savefig(outfname)
print(outfname + " is created. ")
plt.show()

