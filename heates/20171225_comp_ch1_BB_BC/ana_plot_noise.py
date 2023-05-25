#!/usr/bin/env python
""" ana_plot_noise.py is to just plot noise

History:
2017-02-17 ; 1.0; plot noise from scrach 

"""

__author__ =  'Shinya Yamada (syamada(at)tmu.ac.jp'
__version__=  '1.0'

import os
import sys
import math
import cmath
import commands
import numpy as np
import scipy as sp
import scipy.fftpack as sf
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator
import optparse
from matplotlib.font_manager import fontManager, FontProperties
import matplotlib.pylab as plab
import re
import sys
from numpy import linalg as LA
import csv
import random
import h5py

params = {'xtick.labelsize': 10, # x ticks
          'ytick.labelsize': 10, # y ticks
          'legend.fontsize': 10
                    }

plt.rcParams.update(params)


class eventfile():

    def __init__ (self, filename, chan, debug):

        self.chan     = chan
        self.debug    = debug
        self.filename = str(filename).strip()

        self.h5file = h5py.File(self.filename,"r")
        chan = "chan" + str(chan)
        self.psd = self.h5file[chan]["noise_psd"].value

        
def plot_psd(evelist, listname, plotflag):
    """
    plot fft
    """
    F = plt.figure(figsize=(12,8.)) #        plt.subplots_adjust(wspace = 0.1, hspace = 0.3, top=0.9, bottom = 0.08,right=0.92, left = 0.1)

    ax = plt.subplot(2,1,1)
    plt.title("compare noise spectra")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel(r'Frequency(Hz)')
    plt.ylabel(r'Power ($rms/\sqrt{Hz}$)')

    freq = np.linspace(0, 0.5 / (240e-9 * 30), 513)

    for eve in evelist:
        plt.errorbar(freq, eve.psd, fmt='-', label=str(eve.filename), alpha = 0.8) 
        plt.legend(numpoints=1, frameon=False, loc="best")


    ax = plt.subplot(2,1,2)
#    plt.title("compare noise spectra")
    plt.xscale('linear')
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel(r'Frequency(Hz)')
    plt.ylabel(r'Power ($rms/\sqrt{Hz}$)')

    freq = np.linspace(0, 0.5 / (240e-9 * 30), 513)

    for eve in evelist:
        plt.errorbar(freq, eve.psd, fmt='-', label=str(eve.filename), alpha = 0.8) 
        plt.legend(numpoints=1, frameon=False, loc="best")

    outputfiguredir = "fig_fft"
    commands.getoutput('mkdir -p ' + outputfiguredir)
    outfile = outputfiguredir + "/" + listname.replace(".","") + "_compfft.png"
    if plotflag:
        plt.show()
    plt.savefig(outfile)
    
    

def main():

    print "+++++++++++++++++++++++++++++++++++++++++++++"
    print "[START] ", __file__
    print "+++++++++++++++++++++++++++++++++++++++++++++"

    usage = u'%prog fileList [-c 423]'    
    version = __version__
    parser = optparse.OptionParser(usage=usage, version=version)
    parser.add_option('-c', '--chan', action='store', type='int', help='channel number', metavar='CHAN', default=1)
    parser.add_option('-d', '--debug', action='store_true', help='debug flag', metavar='DBEUG', default=False)
    parser.add_option('-p', '--plotflag', action='store_true', help='debug flag', metavar='PLOTFLAG', default=False)

    options, args = parser.parse_args() 
    
    chan     = options.chan
    debug    = options.debug
    plotflag = options.plotflag    

    print "+++++++++++++++++++++++++++++++++++++++++++++"
    print "   (OPTIONS)"
    print "-- chan                            ", chan
    print "-- debug                           ", debug
    print "-- plotflag                        ", plotflag
    print "+++++++++++++++++++++++++++++++++++++++++++++"

    argc = len(args)
    if (argc < 1):
        print '| ERROR, file list file need specified  '
        print parser.print_help()
        quit()

    listname = args[0] # get a file name which contains file names
    filelistfile=open(listname, 'r')

    # create file list
    filelist = []
    for i, filename in enumerate(filelistfile):
        print "...... reading a file (", i, ") ", filename 
        filelist.append(filename)

    # create event file classes 
    evelist = []
    for i, filename in enumerate(filelist):
        print "...... creating a class for a file (", i, ") ", filename
        eve = eventfile(filename, chan, debug)
        evelist.append(eve)

    print "=================== plot psd =========================="

    plot_psd(evelist, listname, plotflag)
    
    print "=================== Finish    =========================="

if __name__ == '__main__':
    main()
