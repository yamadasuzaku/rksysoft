#!/usr/bin/env python 

import numpy as np
import uproot
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import sys
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.colors as colors

if len(sys.argv) < 2:
	print("usage : python script.py rootfilename")
else:
	fname = sys.argv[1]

def convert_to_datetime(unix_time):
    return datetime.fromtimestamp(unix_time)
vectorized_convert = np.vectorize(convert_to_datetime)

file = uproot.open(fname) # root file name 
fnametag = fname.replace(".root","")
tree = file["tree"]
set_channum = set(tree["channum"].array())
channum = tree["channum"].array()

num_alive = 0
num_dead = 0

for i in range(0,115,1):
	if i in set_channum:
		print(i, "1")
		num_alive +=1
	else:
		print(i, "0")
		num_dead +=1

print("number of alive", num_alive)
print("number of dead", num_dead)

# for i, chan in enumerate(set_channum):
# 	print(i, chan)