#!/usr/bin/env python

import sys, os, math, commands
import numpy as np
import matplotlib.pyplot as plt
#import ROOT
import pyfits

import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator
#import pyfits
import optparse
from matplotlib.font_manager import fontManager, FontProperties
import matplotlib.pylab as plab
#import yaml
import re
import sys
from numpy import linalg as LA
import csv
import pytz
import datetime
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, HourLocator, DateFormatter, WeekdayLocator, MONDAY, AutoDateLocator

#from pyPdf import PdfFileWriter, PdfFileReader

# argvs = sys.argv
# if len(argvs)-1 < 1:
#     print 'usage : %s [arg0]'%os.path.basename(argvs[0])
#     quit()

#### variable
#rootfiletag = 'run0002_0000_mass_BB'
rootfiletag  = 'run0004_0000_mass_BC'
rootfilename =  rootfiletag + '.root'
channel = 1
plotpulse = False
valgood = 1
#### const
LEN_PULSE_RECORD = 1024

#### global settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 'small'
plt.rcParams['ytick.labelsize'] = 'small'
plt.rcParams['figure.subplot.left'] = 0.15
plt.rcParams['figure.subplot.right'] = 0.95
plt.rcParams['figure.subplot.bottom'] = 0.10
plt.rcParams['figure.subplot.top'] = 0.90
plt.rcParams['figure.subplot.wspace'] = 0.2
plt.rcParams['figure.subplot.hspace'] = 0.2
#plt.rcParams['axes.formatter.limits'] = -3, 3

plt_font = {
    'family' : 'serif',
    'color'  : 'black',
    'weight' : 'normal',
    'size'   : 16,
}

ftag="run0004_0000_mass_BCsub2_chan1_good1"
#ftag="run0002_0000_mass_BBsub2_chan1_good1"
sub = np.load( ftag + ".npy" )
print ftag, "sub = ", np.mean(sub), " std =", np.std(sub)

leng = len(sub.T)
jet = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=leng)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

outputfiguredir = "figpulse"
commands.getoutput('mkdir -p ' + outputfiguredir)

F = plt.figure(figsize=(12,8))
xvec = range(LEN_PULSE_RECORD)
dt = 240e-9 * 30
powspec = []


for i, (onetmpadc) in enumerate(sub):
    
#    print i, np.mean(onetmpadc), np.std(onetmpadc) 
    c = scalarMap.to_rgba(i)
    ax = plt.subplot(1,1,1)
    plt.errorbar(xvec, onetmpadc, fmt='-', color = c, label="CH = " + str(channel), alpha=0.2)  

    ay2, freq = mlab.psd(onetmpadc, LEN_PULSE_RECORD, 1./dt, window=mlab.window_hanning, sides='onesided', scale_by_freq=True)
    powspec.append(ay2)

plt.savefig(outputfiguredir + "/" + ftag + ".png") 


# plot powerspectra
xvec = range(LEN_PULSE_RECORD)
dt = 240e-9 * 30

powspec = np.mean(powspec, axis=0)
F = plt.figure(figsize=(12,8))
ax = plt.subplot(2,1,1)
plt.yscale('log')
plt.errorbar(freq, powspec, fmt='r-', label="CH = " + str(channel), alpha=0.9)  

ax = plt.subplot(2,1,2)
plt.xscale('log')
plt.yscale('log')
plt.errorbar(freq, powspec, fmt='r-', label="CH = " + str(channel), alpha=0.9)  

plt.savefig(outputfiguredir + "/fft_" + ftag + ".png") 





