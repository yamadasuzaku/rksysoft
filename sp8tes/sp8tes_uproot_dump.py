#!/usr/bin/env python 
"""
History
2022.4.7 ver1. S.Y. 
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

parser = argparse.ArgumentParser(description='This program is to dump txt from sp8 root file', 
                                 epilog='Example of use : python heates_sp8_uproot_dumptxt.py roofile')
parser.add_argument('flist', help='A file that containds a file name of rootfile at one name per one line')
parser.add_argument('-d', '--debug', help='debug to plot using plotly (default: %(default)s)', action='store_true')
parser.add_argument('-l', '--emin', help='lower limit of x (default: %(default)f)', default=1000.5, type=float)
parser.add_argument('-m', '--emax', help='upper limit of x (default: %(default)f)', default=18000.5, type=float)
parser.add_argument('-b', '--binsize', help='bin of x (default: %(default)f)', default=5.0, type=float)
parser.add_argument('-s', '--skipnum', help='to skip the number of run when debug if true (default: %(default)f)', default=300, type=int)

args = parser.parse_args()
flist = args.flist # just shorten names
debug = args.debug
emin = args.emin
emax = args.emax
binsize = args.binsize
skipnum = args.skipnum

binnum=int((emax-emin)/binsize)
ebins=np.linspace(emin,emax,binnum)

OUTPUTDIR="./output/"

args = parser.parse_args()
print("--- input params ----")
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
print("---------------------")

# if debug:
#     print('..................................................')
#     print('flist              = ',flist)
#     print('debug = ', debug)
#     print('..................................................')

filelistfile=open(flist, 'r')
""" Create the list of input files """
filelist = []

for onefile in filelistfile:
    onef = onefile.strip()
    filelist.append(str(onef))

#h5dir=os.path.dirname(h5name)
data = []

outputdata = {}

for froot in filelist:
    frootbase = os.path.basename(froot)
    outputhtml = frootbase.replace(".root","_spec_1d.html")
    outputcsv = frootbase.replace(".root","_spec_1d.csv")
    outputtag = frootbase.replace(".root","_spec_1d")

    data = []

    file = uproot.open(froot)

    channum = np.array(file['tree']['channum'].array())
    energy = np.array(file['tree']['energy'].array())
    good = np.array(file['tree']['good'].array())
    index_last_external_trigger = np.array(file['tree']["index_last_external_trigger"].array())    
    setindex = list(set(index_last_external_trigger))

    if debug:
	    setindex = setindex[::skipnum]

    nspec = 0

    for i, index in enumerate(setindex):
        cutid = np.where((index_last_external_trigger == index) & (good==1) ) [0]
        cutid_bad = np.where((index_last_external_trigger == index) & (good==0) ) [0]

        cutlen = len(cutid)
        cutlen_bad = len(cutid_bad)
        print("..... process ", i, index, " good=", cutlen, " bad=", cutlen_bad)

        if cutlen == 0:
            print("Warning (skip) event number is 0 ", i, index)
            continue 

        ienergy = energy[cutid]

        y, bins = np.histogram(ienergy, bins=ebins) # create hist for one chan   
        x = 0.5 * (bins[1:] + bins[:-1]) # obtain the middle of bins 

        if i ==0:
            outputdata["energy"] = x 
        outputdata[str(index)] = y 

        if debug:
            trace = go.Scatter(x=x, y=y, error_y = dict(type="data", array=np.sqrt(y), visible=True), name=str(index))
            data.append(trace)	
            nspec = nspec + 1
            if nspec > 100: 
                break

    np.savez(OUTPUTDIR + outputtag, **outputdata) 
    print("[Save]", "save as ", OUTPUTDIR + outputtag + ".npz")

    df = pd.DataFrame(outputdata.values(), index=outputdata.keys()).T
    df.to_csv(OUTPUTDIR + outputcsv, index_label="row")
    print("[Save]", "save as ", OUTPUTDIR +  outputcsv)


    if debug:

        layout = go.Layout(title="SP8TES from " + froot,xaxis=dict(title="Energy (eV)"),yaxis=dict(title="counts/" + str(binsize) + " eV") ) 
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
    
        fig.write_html(OUTPUTDIR + outputhtml)
        print("[Save]", "save as ", OUTPUTDIR + outputhtml)
        print('.........................................................................................')


