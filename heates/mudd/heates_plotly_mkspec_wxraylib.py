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
import os 
import plotly.graph_objects as go 

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

parser.add_argument('--lz', help='atomic number of low Z (default: %(default)d)', default=13, type=int)
parser.add_argument('--mz', help='atomic number of middle Z (default: %(default)d)', default=43, type=int)
parser.add_argument('--hz', help='atomic number of high Z (default: %(default)d)', default=90, type=int)
parser.add_argument('--iene', help='input energy for xraylib (default: %(default)f)', default=11.0, type=float)


args = parser.parse_args()
h5name = args.h5name # just shorten names
bins = args.bins
lox = args.lox
upx = args.upx
cname = args.cname
pixelnum = args.pixelnum
debug = args.debug
goodflag = args.goodflag

lz = args.lz 
mz = args.mz
hz = args.hz
iene = args.iene

if debug:
    print('..................................................')
    print('h5name                  = ',h5name)
    print('cname                   = ',cname)
    print('bins,lox, upx           = ',bins,lox,upx)
    print('pixelnum                = ', pixelnum)
    print('goodflag                = ', goodflag)
    print('(xraylib) lz,mz,hz,iene = ', lz,mz,hz,iene)    
    print('debug = ', debug)
    print('..................................................')

h5dir=os.path.dirname(h5name)

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
            if goodflag:
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


#---- run for xraylib to get line lists 

def get_lines_xraylib(debug=debug):

    print('          (start making line lists using xraylib)         ')

    import xraylib

    xraylib.XRayInit()
    print("xraylib version: {}".format(xraylib.__version__))

    atnum, symbol, elists, ilists, elist, ilist = [],[],[],[],[],[]

    excluded_line = []
    # light elements (lz <= Z < mz)
    print('          ', lz, ' <= Z < ', mz, '  --> Ka1,Ka2,Kb1')    
    for i in np.arange(lz,mz):
        i = int(i) # in case numpy create i as float. 
        asymbol = xraylib.AtomicNumberToSymbol(i)

        try:
            elist = [ #energy 
                xraylib.LineEnergy(i, xraylib.KL2_LINE),  # Ka2
                xraylib.LineEnergy(i, xraylib.KL3_LINE),  # Ka1
                xraylib.LineEnergy(i, xraylib.KM2_LINE),  # Kb3 
                xraylib.LineEnergy(i, xraylib.KM3_LINE)   # Kb1   
                    ]
            ilist = [ #energy 
                xraylib.CS_FluorLine_Kissel_Cascade(i, xraylib.KL2_LINE, iene), # Ka2
                xraylib.CS_FluorLine_Kissel_Cascade(i, xraylib.KL3_LINE, iene), # Ka1
                xraylib.CS_FluorLine_Kissel_Cascade(i, xraylib.KM2_LINE, iene), # Kb3
                xraylib.CS_FluorLine_Kissel_Cascade(i, xraylib.KM3_LINE, iene), # Kb1
                    ]

            if debug:
                print(i, str(asymbol), elist, ilist)

            atnum.append(i) 
            symbol.append(asymbol)
            elists.append(elist)
            ilists.append(ilist)

        except:
            excluded_line.append(asymbol)
            if debug:
                print(asymbol, sys.exc_info())

    # heavy elements (mz <= Z < hz)
    print('          ', mz, ' <= Z < ', hz, '  --> La1,La2,Lb1,Lg1,Lg2,Lg3')    
    for i in np.arange(mz,hz):

        i = int(i) # in case numpy create i as float.         
        asymbol = xraylib.AtomicNumberToSymbol(i)

        try:
            elist = [ # energy 
                xraylib.LineEnergy(i, xraylib.L3M4_LINE), # La2
                xraylib.LineEnergy(i, xraylib.L3M5_LINE), # La1
                xraylib.LineEnergy(i, xraylib.L2M4_LINE), # Lb1
                xraylib.LineEnergy(i, xraylib.L2N4_LINE), # Lg1
                xraylib.LineEnergy(i, xraylib.L1N2_LINE), # Lg2
                xraylib.LineEnergy(i, xraylib.L1N3_LINE)  # Lg3
                ]

            ilist = [ # intensity 
                xraylib.CS_FluorLine_Kissel_Cascade(i, xraylib.L3M4_LINE, iene), # La2
                xraylib.CS_FluorLine_Kissel_Cascade(i, xraylib.L3M5_LINE, iene), # La1
                xraylib.CS_FluorLine_Kissel_Cascade(i, xraylib.L2M4_LINE, iene), # Lb1
                xraylib.CS_FluorLine_Kissel_Cascade(i, xraylib.L2N4_LINE, iene), # Lg1
                xraylib.CS_FluorLine_Kissel_Cascade(i, xraylib.L1N2_LINE, iene), # Lg2
                xraylib.CS_FluorLine_Kissel_Cascade(i, xraylib.L1N3_LINE, iene)  # Lg3
                ]

            if debug:
                print(i, str(asymbol), elist, ilist)

            atnum.append(i) 
            symbol.append(asymbol)
            elists.append(elist)
            ilists.append(ilist)

        except:
            excluded_line.append(asymbol)
            if debug:
                print(asymbol, sys.exc_info())

    print("[not calculated b/c input energy = ", iene, " keV] elements =  ", ", ".join(excluded_line))


    datadict = {'atnum': atnum, 'symbol': symbol, 'ene': elists, 'intensity': ilists} 
    df = pd.DataFrame(datadict)
    csvname = 'linelist_fromxraylib_' + str(lz) + "_"  + str(mz) + "_"  + str(hz) + "_input_"  + str(iene)  + "keV.csv"
    pngname = 'linelist_fromxraylib_' + str(lz) + "_"  + str(mz) + "_"  + str(hz) + "_input_"  + str(iene)  + "keV.png"
    htmlname = 'linelist_fromxraylib_' + str(lz) + "_"  + str(mz) + "_"  + str(hz) + "_input_"  + str(iene)  + "keV.html"

    df.to_csv(csvname) # save
    df_line = pd.read_csv(csvname) #open 
    print('         (finish making line lists using xraylib)         ')

    return df_line


def plotly_hists_wlines(df_line, npzname="test.npz",sname="run_test",cname="ene",bins=1e3,lox=0,upx=1e4, debug=False):
    """
    plot all and each pixels using plotly 
    """
    htmlname = npzname.replace(".npz","_plotly_wlines.html")

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
            print("in plotly_hists_wlines: ", i, chan, cnum, evenum)

    layout = go.Layout(title="input energy = " + str(iene) + " keV created from " + npzname,xaxis=dict(title=cname),yaxis=dict(title="counts / " + binsize + " eV") ) 

    # add lines into data
    scale = np.amax(npdata["sum_y"]) * 0.1
    eV2keV = 1.0e3
    for index, row in df_line.iterrows():
        x =  np.array(ast.literal_eval(row["ene"])) * eV2keV
        y =  np.array(ast.literal_eval(row["intensity"])) * scale
        trace = go.Scatter(x=x, y=y, 
                           name=row["symbol"]+"("+str(row["atnum"])+")",
                           opacity=.8,
                           mode = 'lines+markers',line = dict(shape = 'linear', dash = 'dash', width = 2))
        data.append(trace)
        if debug:
            print("in plot_plotly_csv : ", index, row["symbol"]+"("+str(row["atnum"])+")")


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
if h5dir == "":
	npzname= cname + "_" + sname + "_" + str(lox) + "-" + str(upx) + "_pnum" + str(pixelnum) + "_b" + str(bins) + "_good" + str(goodflag) + ".npz"
else:
	npzname= h5dir + "/" + cname + "_" + sname + "_" + str(lox) + "-" + str(upx) + "_pnum" + str(pixelnum) + "_b" + str(bins) + "_good" + str(goodflag) + ".npz"

print('     (1) save_npz ')
save_npz(npzname=npzname, sname=sname,cname=cname,bins=bins,lox=lox,upx=upx, pixelnum=pixelnum,debug=debug,goodflag=goodflag)

# plot data using plotly and generate html 
print('     (2) plotly_hists')
plotly_hists(npzname=npzname,sname=sname,cname=cname,bins=bins,lox=lox,upx=upx,debug=debug)

# create line lists using xraylib
print('     (3) get_lines_xraylib')
df_line = get_lines_xraylib(debug=debug)

# plot data with lines 
print('     (4) plotly_hists_wlines')
plotly_hists_wlines(df_line,npzname=npzname,sname=sname,cname=cname,bins=bins,lox=lox,upx=upx,debug=debug)



