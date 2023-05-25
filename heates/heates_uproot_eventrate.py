#!/usr/bin/env python 
"""
History
2022.2.6 ver1. S.Y. 
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
                                 epilog='Example of use : python heates_plotly_mkspec_wxraylib.py run0014_mass_noi12.hdf5')
parser.add_argument('flist', help='A file that containds a file name of rootfile at one name per one line')
parser.add_argument('-d', '--debug', help='debug (default: %(default)s)', action='store_true')

args = parser.parse_args()
flist = args.flist # just shorten names
debug = args.debug

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

    tstamp = np.array(file['tree']['timestamp'].array())
    pbeam = np.array(file['tree']['pbeam'].array())
    energy=np.array(file['tree']['energy'].array())

    tstart = tstamp[0]
    tstop = tstamp[-1]
    ta = np.linspace(tstart, tstop, 10)
    ts = ta[:-1]
    te = ta[1:]

    for (ts, te) in zip(ts,te):
        dt = te - ts 
        tc = (te + ts) * 0.5
        datetc = datetime.datetime.fromtimestamp(tc)
        cutid = np.where((tstamp > ts) & (tstamp <= te) & (pbeam==1) ) [0]

        if dt > 0:
            everate = len(tstamp[cutid])/dt
            everate_err = np.sqrt(len(tstamp[cutid]))/dt
        else:
            print("ERROR, dt <=0 ", dt)
            sys.exit()

        tlist.append(tc)
        datelist.append(datetc)
        rate.append(everate)
        rate_err.append(everate_err)

        print(tc, datetc, everate)

    trace = go.Scatter(x=datelist, y=rate, 
                        error_y = dict(type="data", array=rate_err, visible=True), name=froot)
    data.append(trace)


layout = go.Layout(title="Event Rates : pbeam == 1",xaxis=dict(title="Time"),yaxis=dict(title="counts/s") ) 
fig=go.Figure(layout=layout,data=data) 
fig.update_yaxes(type="linear")
fig.update_yaxes(exponentformat='none')
fig.update_xaxes(exponentformat='none')

# Add dropdown
fig.update_layout(
updatemenus=[
        dict(
            buttons=list([
                    dict(
                        args=[{"yaxis.type": "linear"}],
                        label="linear",
                        method="relayout"
                        ),
                    dict(
                        args=[{"yaxis.type": "log"}],
                        label="log",
                        method="relayout"
                        )
                    ]),
        direction="down")
        ])

htmlname = "eventrate.html"
fig.write_html(htmlname)
print("[Save]", "save as ", htmlname)
print('.........................................................................................')


