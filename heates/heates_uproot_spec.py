#!/usr/bin/env python 
"""
History
2022.3.19 ver1. S.Y. 
"""

import sys
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
import uproot
import datetime


parser = argparse.ArgumentParser(description='This program is to create hists and plot them with lines.', 
                                 epilog='Example of use : python heates_uproot_spec.py filenames.list')
parser.add_argument('flist', help='A file that containds a file name of rootfile at one name per one line')
parser.add_argument('-d', '--debug', help='debug (default: %(default)s)', action='store_true')
parser.add_argument('-n', '--normed', help='debug (default: %(default)s)', action='store_true')

parser.add_argument('-b', '--bins', help='number of bins (default: %(default)d)"', default=2000, type=int)
parser.add_argument('-l', '--lox', help='lower limit of x (default: %(default)f)', default=500, type=float)
parser.add_argument('-u', '--upx', help='upper limit of x (default: %(default)f)', default=10500, type=float)
parser.add_argument('--ts', help='lower limit of dt (default: %(default)f)', default=4090, type=float)
parser.add_argument('--te', help='upper limit of dt (default: %(default)f)', default=4120, type=float)

args = parser.parse_args()
flist = args.flist # just shorten names
debug = args.debug

ts = args.ts
te = args.te
lox = args.lox
upx = args.upx
bins = args.bins
normed = args.normed

if debug:
    print('..................................................')
    print('flist              = ',flist)
    print('debug = ', debug)
    print('..................................................')

filelistfile=open(flist, 'r')
""" Create the list of input files """
filelist = []

for onefile in filelistfile:
    onef = onefile.strip()
    filelist.append(str(onef))

#h5dir=os.path.dirname(h5name)
data = []

for froot in filelist:

    tlist = []
    datelist = []
    rate = []
    rate_err = []


    file = uproot.open(froot)

    dt = np.array(file['tree']['dt'].array())
    pbeam = np.array(file['tree']['pbeam'].array())
    good = np.array(file['tree']['good'].array())
    energy=np.array(file['tree']['energy'].array())
    cutid = np.where((dt > ts) & (dt <= te) & (good==1) ) [0]
    cutene = energy[cutid]

    y, xedge = np.histogram(cutene, bins=bins, range=(lox,upx))
    x = 0.5 * (xedge[1:] + xedge[:-1]) # obtain the middle of bins 
    if normed:
        y = y/np.sum(y)
    trace = go.Scatter(x=x, y=y, name=os.path.basename(froot))
    data.append(trace)

binsize=int(x[1] - x[0])

cutcond = str(ts) + " < dt < "  + str(te)
if normed:
    layout = go.Layout(title="Hist1D (normed by total counts) : "  + cutcond,xaxis=dict(title="Energy (eV)"),yaxis=dict(title="counts/" + str(binsize)+"eV" ))     
else:
    layout = go.Layout(title="Hist1D"  + cutcond,xaxis=dict(title="Energy (eV)"),yaxis=dict(title="counts/" + str(binsize)+"eV" )) 


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

if normed:
    htmlname = "hist_" + flist.split(".")[0] + "_" + str(ts) + "_" + str(te) + "_" + str(bins) + "_normed.html"
else:
    htmlname = "hist_" + flist.split(".")[0] + "_" + str(ts) + "_" + str(te) + "_" + str(bins) + ".html"
fig.write_html(htmlname)
print("[Save]", "save as ", htmlname)
print('.........................................................................................')


