#!/usr/bin/env python 
"""
History
2022.2.6 ver1. S.Y. 
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.cm as cm
import argparse 
import pandas as pd
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go 
import os

parser = argparse.ArgumentParser(description='This program is to create hists and plot them with lines.', 
                                 epilog='Example of use : python heates_plotly_mkspec_wxraylib.py run0014_mass_noi12.hdf5')
parser.add_argument('h5name', help='Name of HDF5 including cname key')
parser.add_argument('-b', '--bins', help='number of bins (default: %(default)d)"', default=2000, type=int)
parser.add_argument('-l', '--lox', help='lower limit of x (default: %(default)f)', default=0, type=float)
parser.add_argument('-u', '--upx', help='upper limit of x (default: %(default)f)', default=12000, type=float)
parser.add_argument('-c', '--cname', help='name of columns used for histogram (default: %(default)s)', default="energy", type=str)
parser.add_argument('-p', '--pixelnum', help='number of pixels (default: %(default)d)', default=240, type=int)
parser.add_argument('-d', '--debug', help='debug (default: %(default)s)', action='store_true')
parser.add_argument('-g', '--goodflag', help='good only (default: %(default)s)', action='store_false')
parser.add_argument('-m', '--pbeamflag', help='pbeam good (default: %(default)s)', action='store_false')

args = parser.parse_args()
h5name = args.h5name # just shorten names
bins = args.bins
lox = args.lox
upx = args.upx
cname = args.cname
pixelnum = args.pixelnum
debug = args.debug
goodflag = args.goodflag
pbeamflag = args.pbeamflag

h5dir=os.path.dirname(h5name)

if debug:
    print('..................................................')
    print('h5name                  = ',h5name)
    print('cname                   = ',cname)
    print('bins,lox, upx           = ',bins,lox,upx)
    print('pixelnum                = ', pixelnum)
    print('goodflag                = ', goodflag)
    print('pbeamflag                = ', pbeamflag)
    print('debug = ', debug)
    print('..................................................')

h5=h5py.File(h5name,'r') # this is used for global variables
sname=os.path.basename(h5name).replace(".hdf5","")

def save_npz(npzname="test.npz", sname="run_test",cname="ene",bins=1e3,lox=0,upx=1e4,pixelnum=10,debug=False,goodflag=True):
    """
    from HDF5 to npz
    """
    all_info={}
    not_found_num = 0

    for i, ch in enumerate(np.arange(pixelnum)*2+1):
    #for ch in np.arange(240)*2+1:
        chn='chan%d'%(ch)
        if chn in h5:
            chan=h5[chn]
            if debug:
                print(chan, ch)
            energy=np.array(chan[cname])
            good=np.array(chan['good'])                        
            pbeam=np.array(chan['pbeam'])                        
            if goodflag:
                if pbeamflag:
                    cutid = np.where( (good==1) & (pbeam==1) )[0]
                    y, bins = np.histogram(energy[cutid], bins=bins, range=(lox,upx)) # create hist for one chan
                    print("data length = ", len(cutid))
                else:
                    y, bins = np.histogram(energy[good], bins=bins, range=(lox,upx)) # create hist for one chan
            else:
                y, bins = np.histogram(energy, bins=bins, range=(lox,upx)) # create hist for one chan	        		
            x = 0.5 * (bins[1:] + bins[:-1]) # obtain the middle of bins 
            all_info[chn+"_y"] = y
        else:
            if not_found_num == 0:
                print("[Not found in HDF5] ch = ", ch, end=", ")
                not_found_num += 1
            else:
                print(ch, end=", ")
                not_found_num += 1                
            continue
        if i == 0:
            sumy = y
        else:
            sumy = sumy + y
    print(" done. ("+str(not_found_num)+" pixels discarded.)" )
        
    all_info["x"] = x 
    all_info["sum_y"] = sumy

    np.savez(npzname, **all_info)
    print("[Save]", npzname, " is saved.")
    print('.........................................................................................')

def plotly_hists(npzname="test.npz",sname="run_test",cname="ene",bins=1e3,lox=0,upx=1e4, debug=False):
    """
    plot all and each pixels using plotly 
    """
    htmlname = npzname.replace(".npz","_plotly.html")

    npdata = np.load(npzname, allow_pickle=True)
    dlist = list(npdata.keys())
    chanlist = [s for s in dlist if 'chan' in s]
    channum = [s.replace("chan","").replace("_y","") for s in dlist if 'chan' in s]
    x = npdata['x']
    binsize=str("%3.1f" % (x[1]-x[0]))

    data = []
    # add sum of all pixels 
    evenum = np.sum(npdata["sum_y"])
    trace = go.Scatter(x=x, y=npdata["sum_y"], name="sum (" + str(evenum) + "c)")
    data.append(trace)

    #loop through data to create plotly trace 
    for i, (chan, cnum) in enumerate(zip(chanlist,channum)):
        evenum = np.sum(npdata[chan])
        trace = go.Scatter(x=x, y=npdata[chan], name=cnum + "(" + str(evenum) + "c)")
        data.append(trace)
        if debug:
            print("in plotly_hists : ", i, chan, cnum, evenum)

    layout = go.Layout(title=npzname,xaxis=dict(title=cname),yaxis=dict(title="counts / " + binsize + " eV") ) 


    fig=go.Figure(layout=layout,data=data) 
    fig.update_yaxes(type="log")
    fig.update_yaxes(exponentformat='none')
    fig.update_xaxes(exponentformat='none')

    # Add dropdown
    fig.update_layout(
    updatemenus=[
            dict(
                buttons=list([
                        dict(
                            args=[{"yaxis.type": "log"}],
                            label="log",
                            method="relayout"
                            ),
                        dict(
                            args=[{"yaxis.type": "linear"}],
                            label="linear",
                            method="relayout"
                            )
                        ]),
            direction="down")
            ])

    fig.write_html(htmlname)
    print("[Save]", "save as ", htmlname)
    print('.........................................................................................')



#---- run from here
# save data into npz

#---- run from here
# save data into npz
if h5dir == "":
        npzname= cname + "_" + sname + "_" + str(lox) + "-" + str(upx) + "_pnum" + str(pixelnum) + "_b" + str(bins) + "_good" + str(goodflag)  + "_pbeam" + str(pbeamflag)+ ".npz"
else:
        npzname= h5dir + "/" + cname + "_" + sname + "_" + str(lox) + "-" + str(upx) + "_pnum" + str(pixelnum) + "_b" + str(bins) + "_good" + str(goodflag)  + "_pbeam" + str(pbeamflag)+ ".npz"

print('     (1) save_npz ', npzname)
save_npz(npzname=npzname, sname=sname,cname=cname,bins=bins,lox=lox,upx=upx, pixelnum=pixelnum,debug=debug,goodflag=goodflag)

# plot data using plotly and generate html 
print('     (2) plotly_hists')
plotly_hists(npzname=npzname,sname=sname,cname=cname,bins=bins,lox=lox,upx=upx,debug=debug)
