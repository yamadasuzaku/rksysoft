#!/usr/bin/env python

""" XMM-Newton_plot_pointsource.py 

This is a python script to plot image of XMM Newton 
History: 
2018-05-04 ; ver 1.0; First version made by S.Y.
"""

__author__ =  'Shinya Yamada (syamada(at)tmu.ac.jp)'
__version__=  '1.0'

import os
import sys
import math
import commands 
import re
import sys

# for I/O, optparse
import optparse

# numpy 
import numpy as np
from numpy import linalg as LA

# conver time into date
import datetime

# matplotlib http://matplotlib.org
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator
import matplotlib.colors as cr
from matplotlib.font_manager import fontManager, FontProperties
import matplotlib.pylab as plab

## for FITS IMAGE 
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename

params = {'xtick.labelsize': 10, # x ticks
          'ytick.labelsize': 10, # y ticks
          'legend.fontsize': 8
                    }
plt.rcParams['font.family'] = 'serif' 
plt.rcParams.update(params)

import ROOT
ROOT.gROOT.SetBatch(1)

# Fits I.O class
class Fits():
    # member variables 
    debug = False

    def __init__ (self, eventfile, debug):

        self.eventfilename = eventfile
        self.debug = debug

        #extract header information 
        if os.path.isfile(eventfile):
            self.filename = get_pkg_data_filename(eventfile)
            self.hdu = fits.open(self.filename)[0]
            self.dirname = eventfile.replace(".fits","").replace(".gz","")
            print "..... __init__ "
            self.data = self.hdu.data
            self.data[np.isnan(self.data)] = 0 # nan ==> 0, note that radio data include negative values
            self.data = np.abs(self.data).T # data is forced to be positive (Is it right?)

            if self.debug: 
            	print "... in __init__ Class Fits ()"
                print self.filename
                print self.hdu

        else:
            print "ERROR: cat't find the fits file : ", self.eventfilename
            quit()


    def fitsimage2root_all(self, h2dname="h2d", ftag = "auto"):

        """
        convert image to root 
        """

        if ftag == "auto":
            ftag = self.dirname
        else:
            ftag = ftag 
        

        ROOT.gStyle.SetOptLogz(1)

        rootfilename=ftag + ".root"
        pngfilename=ftag + ".png"
        TFile = ROOT.TFile.Open(rootfilename,"RECREATE")

        x, y = self.data.shape
        print "...   fits2root()"
        print "..... x, y = ", x, y  

        detx_binnum = x
        detx_min = -0.5
        detx_max = x -0.5        
        dety_binnum = y
        dety_min = -0.5
        dety_max = y -0.5

        h2d = ROOT.TH2D(h2dname,"", detx_binnum, detx_min, detx_max, dety_binnum, dety_min, dety_max)

        can = ROOT.TCanvas("can", "", 800+4, 600+26)
#        h2d = ROOT.TH2D(h2dname,"", detx_binnum, detx_min, detx_max, dety_binnum, dety_min, dety_max)

        for i in np.arange(x):
            for j in np.arange(y):
                h2d.SetBinContent(i, j, self.data[i][j])
        
        h2d.GetYaxis().SetTitle("Y")
        h2d.GetXaxis().SetTitle("X")
        h2d.Draw("COLZ")

        can.SaveAs(pngfilename)

        TFile.Write()
        TFile.Close()


    def fitsimage2root_part(self, detxmin, detxmax, detymin, detymax, h2dname="h2d", ftag = "auto"):

        """
        convert image to root 
        """

        if ftag == "auto":
            ftag = self.dirname
        else:
            ftag = ftag 
        

        ROOT.gStyle.SetOptLogz(1)

        rootfilename=ftag + ".root"
        pngfilename=ftag + ".png"
        TFile = ROOT.TFile.Open(rootfilename,"RECREATE")

        x, y = self.data.shape

        print "...   fits2root()"
        print "..... x, y = ", x, y

        detxbinnum = int(detxmax - detxmin)
        detybinnum = int(detymax - detymin)

        print detxbinnum, detxmin, detxmax, detybinnum, detymin, detymax
        h2d = ROOT.TH2D(h2dname,"", detxbinnum, detxmin -0.5, detxmax -0.5, detybinnum, detymin -0.5, detymax -0.5)

        can = ROOT.TCanvas("can", "", 800+4, 600+26)
#        h2d = ROOT.TH2D(h2dname,"", detx_binnum, detx_min, detx_max, dety_binnum, dety_min, dety_max)

        for i in np.arange(detxbinnum):
            for j in np.arange(detybinnum):
                h2d.SetBinContent(i, j, self.data[i + int(detxmin)][j + int(detymin)])
        
        h2d.GetYaxis().SetTitle("Y")
        h2d.GetXaxis().SetTitle("X")
        h2d.Draw("COLZ")

        can.SaveAs(pngfilename)

        TFile.Write()
        TFile.Close()


def main():

    """ start main loop """
    curdir = os.getcwd()

    """ Setting for options """
    usage = u'%prog FileList '    
    version = __version__
    parser = optparse.OptionParser(usage=usage, version=version)

    parser.add_option('-d', '--debug', action='store_true', help='The flag to show detailed information', metavar='DEBUG', default=False)
    parser.add_option('-j', '--justdumpall', action='store_false', help='The flag just to dump all', metavar='JUSTDUMPALL', default=True)
    parser.add_option('-x', '--detxmin', action='store', type='int', help='det x min', metavar='DETXMIN', default=1100)
    parser.add_option('-b', '--detxmax', action='store', type='int', help='det x max', metavar='DETXMAX', default=1500)
    parser.add_option('-y', '--detymin', action='store', type='int', help='det y min', metavar='DETYMIN', default=1200)
    parser.add_option('-z', '--detymax', action='store', type='int', help='det y max', metavar='DETYMAX', default=1600)
    parser.add_option('-f', '--ftag', action='store', type='str', help='file tag', metavar='FTAG', default="auto")

    options, args = parser.parse_args()

    print "---------------------------------------------"
    print "| START  :  " + __file__

    debug =  options.debug
    justdumpall =  options.justdumpall
    detxmin =  options.detxmin
    detxmax =  options.detxmax
    detymin =  options.detymin
    detymax =  options.detymax
    ftag =  options.ftag

    print "[image selection]"
    print "..... detxmin    ", detxmin
    print "..... detxmax    ", detxmax
    print "..... detymin    ", detymin
    print "..... detymax    ", detymax
    print "[output]"
    print "..... ftag       ", ftag
    print "[debug]"
    print "..... debug ", debug
    print "[plot all or selected region]"
    print "..... justdumpall ", justdumpall
    
    argc = len(args)
    if (argc < 1):
        print '| STATUS : ERROR, need to specity input file list '
        print parser.print_help()
        quit()

    print "\n| STEP1  : open the list of the files"    
    filelistfile=open(args[0], 'r')

    """ Create the list of input files """
    filelist = []
    for onefile in filelistfile:
        if debug: print onefile.split()
        filelist.append(str(onefile.split()[0]))
    
    # strip() works in the same way as chop() in perl or ruby
    print "..... filelist = " + str(filelist)
    if debug: print filelist

    print "\n| STEP2  : Read each file and do process " + "\n"
    evelist = []

    for i, (filename) in enumerate(zip(filelist)):
        filename = filename[0]
        filenametag = filename.replace(".txt","")
        print "START : No." + str(i)  + " " + filename
        eve = Fits(filename, debug)
        if justdumpall:
            print "----- just dump all image into ROOT"
            eve.fitsimage2root_all(h2dname="h2d", ftag = ftag)
        else:
            print "----- just dump a part of image into ROOT"
            eve.fitsimage2root_part(detxmin, detxmax, detymin, detymax, h2dname="h2dpart", ftag = ftag)
        print "..... finish \n"

if __name__ == '__main__':
    main()
