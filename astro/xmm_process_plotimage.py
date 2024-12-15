#!/usr/bin/env python

""" xmm_process_plotimage.py

This is a python script to plot images for publication 
History: 
2018-08-19 ; ver 1.0; made by S.Y. xmm_process_compimage.py

input : assume that three X-ray images + radio image

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
import matplotlib
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


# Fits I.O class
class Fits():
    # member variables 
    debug = False

    def __init__ (self, eventfile, det, debug):

        self.eventfilename = eventfile
        self.debug = debug
        self.det = det

        #extract header information 
        if os.path.isfile(eventfile):
            self.filename = get_pkg_data_filename(eventfile)
            self.hdu = fits.open(self.filename)[0]

#            self.dateobs = self.hdu.header['DATE-OBS']            
#            self.object = self.hdu.header['OBJECT']                        

            self.wcs = WCS(self.hdu.header)
            self.cdelt = self.wcs.wcs.cdelt
            self.p2deg = np.abs(self.cdelt[0]) # [deg/pixel]
            self.p2arcsec = 3600. * np.abs(self.cdelt[0]) # [arcsec/pixel]

            self.dirname = eventfile.replace(".fits","").replace(".gz","")

            print "..... __init__ "
#            print "      self.dateobs    = ", self.dateobs
#            print "      self.object     = ", self.object
            print "      self.p2deg      = ", self.p2deg, " [deg/pixel]"
            print "      self.p2arcsec   = ", self.p2arcsec, " [arcsec/pix]"

            self.data = self.hdu.data
            self.data[np.isnan(self.data)] = 0 # nan ==> 0, note that radio data include negative values
            self.data = np.abs(self.data) # data is forced to be positive (Is it right?)

            if self.debug: 
            	print "... in __init__ Class Fits ()"
                print self.filename
                print self.hdu
                self.wcs.printwcs()

        else:
            print "ERROR: cat't find the fits file : ", self.eventfilename
            quit()


    def plotwcs_astropy(self, vmin = 1e-1, vmax = 20, debug = False):
        """
        just plot the entire image
        """
        print "\n..... plotwcs_astropy ...... "
    
        F = plt.figure(figsize=(12,8))
        ax = plt.subplot(111,projection=self.wcs)

#        plt.figtext(0.1,0.97, "OBJECT = " + self.object + " DATE = " + self.dateobs)
        plt.figtext(0.1,0.95, "Unit = " + str(self.p2arcsec) + " [arcsec/pix]" )

        plt.imshow(self.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=vmin,vmax=np.amax(self.hdu.data)))
        plt.colorbar()
        plt.xlabel('RA')
        plt.ylabel('Dec')
        ax.grid(color='black', ls='dotted')

        outputfiguredir = "fig_image_" + self.dirname
        commands.getoutput('mkdir -p ' + outputfiguredir)
        plt.savefig(outputfiguredir + "/" + "plotwcs.png")
        if debug:
            plt.show()

    def init_aplpy(self):

        import aplpy
        self.aplpyfig = aplpy.FITSFigure(self.eventfilename)        


    def plotwcs_aplpy(self, vmin = 1e-1, vmax = 20, debug = False, ra = 289.0, dec = 4.727, width = 1.0, height = 0.5):
        """
        just plot the entire image
        """
        print "\n..... plotwcs_aplpy ...... "
    
        self.init_aplpy()

#        F = plt.figure(figsize=(12,8))
#        ax = plt.subplot(111,projection=self.wcs)

#        plt.figtext(0.1,0.97, "OBJECT = " + self.object + " DATE = " + self.dateobs)
#        plt.figtext(0.1,0.95, "Unit = " + str(self.p2arcsec) + " [arcsec/pix]" )
#        self.aplpyfig.show_colorscale()
        self.aplpyfig.show_colorscale()
        self.aplpyfig.recenter(ra, dec, width=width, height = height)
#        self.aplpyfig.scalebar.show(0.2)


#        self.aplpyfig.add_colorbar()        

        outputfiguredir = "fig_image_aplpy_" + self.dirname
        commands.getoutput('mkdir -p ' + outputfiguredir)
        self.aplpyfig.savefig(outputfiguredir + "/" + "plotwcs.png")
        if debug:
            matplotlib.pyplot.show()



    def plotwcs_ps(self, detx, dety, ds = 40, vmin = 1e-1, vmax = 20):
        """
        just plot the enlarged image around (detx,dety)
        """

        print "\n..... plotwcs_ps ...... "

        self.detx = detx
        self.dety = dety

        gpixcrd = np.array([[ self.detx, self.dety]], np.float_)
        gwrdcrd = self.wcs.all_pix2world(gpixcrd,1)
        ra = gwrdcrd[0][0]
        dec = gwrdcrd[0][1]        
        self.ra = ra
        self.dec = dec        

        print "detx, dety = ", detx, dety
        print "ra, dec    = ", ra,  dec

        F = plt.figure(figsize=(12,8))

#        plt.figtext(0.1,0.97, "OBJECT = " + self.object + " DATE = " + self.dateobs)
        plt.figtext(0.1,0.95, "Unit = " + str(self.p2arcsec) + " [arcsec/pix]" )

        ax = plt.subplot(111, projection=self.wcs)

        plt.imshow(self.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=vmin,vmax=vmax))

        print " self.hdu.data[detx][dety] = ", self.hdu.data[detx][dety] , " (detx,dety) = (", detx, ",", dety, ")"

        ax.set_xlim(detx - ds, detx + ds)
        ax.set_ylim(dety - ds, dety + ds)		
        plt.colorbar()
        plt.xlabel('RA')
        plt.ylabel('Dec')

        outputfiguredir = "fig_image_ps_" + self.dirname
        commands.getoutput('mkdir -p ' + outputfiguredir)
        plt.savefig(outputfiguredir + "/" + "plotwcs_ps_detx" + str("%d" % detx) + "_dety" + str("%d" % dety) + ".png")

    def mk_radialprofile(self, detx, dety, ds = 40, ndiv = 20, vmin = 1e-1, vmax = 20):
        """
        create the radial profiles
        """

        print "\n..... mk_radialprofile ...... "

        self.detx = detx
        self.dety = dety
        
        gpixcrd = np.array([[ self.detx, self.dety]], np.float_)
        gwrdcrd = self.wcs.all_pix2world(gpixcrd,1)
        ra = gwrdcrd[0][0]
        dec = gwrdcrd[0][1]        
        self.ra = ra
        self.dec = dec        
        print "detx, dety = ", detx, dety
        print "ra, dec    = ", ra,  dec

        # radial profiles around the input (ra, dec)
        rc, rp = calc_radialprofile(self.data, detx, dety, ds = ds, ndiv = ndiv) 
        
        # plot images and radial profiles
        F = plt.figure(figsize=(10,12))
        F.subplots_adjust(left=0.1, bottom = 0.1, right = 0.9, top = 0.87, wspace = 0.3, hspace=0.3)
#        plt.figtext(0.1,0.97, "OBJECT = " + self.object + " DATE = " + self.dateobs)
        plt.figtext(0.1,0.95, "Unit = " + str(self.p2arcsec) + " [arcsec/pix]" )
        plt.figtext(0.1,0.93, "input center [deg] (x,c,ra,dec) = (" + str("%3.4f" % detx) + ", " + str("%3.4f" % dety) + ", "+ str("%3.4f" % ra) + ", " + str("%3.4f" % dec) + ")")

        ax = plt.subplot(3,2,1)
        plt.title("(1) SKY image")
        plt.imshow(self.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=vmin,vmax=vmax))
#        ax.set_xlim(detx - ds, detx + ds)
#        ax.set_ylim(dety - ds, dety + ds)		
#        plt.imshow(self.hdu.data, origin='lower', cmap=plt.cm.viridis)
        plt.colorbar()
        plt.scatter(detx, dety, c="k", s=300, marker="x")
        plt.xlabel('X')
        plt.ylabel('Y')

        ax = plt.subplot(3,2,2, projection=self.wcs)
        plt.title("(2) Ra Dec image (FK5)")
        plt.imshow(self.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=vmin,vmax=vmax))
        plt.colorbar()
        plt.scatter(detx, dety, c="k", s=300, marker="x")
        plt.xlabel('X')
        plt.ylabel('Y')

        ax = plt.subplot(3,2,3)
        plt.title("(3) SKY image")
        plt.imshow(self.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=vmin,vmax=vmax))
        ax.set_xlim(detx - ds, detx + ds)
        ax.set_ylim(dety - ds, dety + ds)       
        plt.colorbar()
        plt.scatter(detx, dety, c="k", s=300, marker="x")
        plt.xlabel('X')
        plt.ylabel('Y')


        ax = plt.subplot(3,2,4, projection=self.wcs)
        plt.title("(4) Ra Dec image (FK5)")
        plt.imshow(self.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=vmin,vmax=vmax))
        ax.set_xlim(detx - ds, detx + ds)
        ax.set_ylim(dety - ds, dety + ds)       
        plt.colorbar()
        plt.scatter(detx, dety, c="k", s=300, marker="x")
        plt.xlabel('X')
        plt.ylabel('Y')

        ax = plt.subplot(3,2,5)
        plt.title("(5) radial profile (pix)")
        plt.errorbar(rc, rp, fmt="ko-", label="input center")
        plt.xlabel('radial distance (pixel)')
        plt.ylabel(r'c s$^{-1}$ deg$^{-2}$')
        plt.grid(True)            
        plt.legend(numpoints=1, frameon=False)

        ax = plt.subplot(3,2,6)
        plt.title("(6) radial profile (arcsec)")
        plt.errorbar(rc * self.p2arcsec, rp, fmt="ko-", label="input center")
        plt.xlabel('radial distance (arcsec)')
        plt.ylabel(r'c s$^{-1}$ deg$^{-2}$')
        plt.legend(numpoints=1, frameon=False)
        plt.grid(True)

        outputfiguredir = "fig_image_xy_" + self.dirname
        commands.getoutput('mkdir -p ' + outputfiguredir)
        plt.savefig(outputfiguredir + "/" + "mkrad_detx" + str("%d" % detx) + "_dety" + str("%d" % dety) + ".png")


        # dump the radial profiles into the text file            
        outputfiguredir = "txt_image_xy_" + self.dirname
        commands.getoutput('mkdir -p ' + outputfiguredir)

        fout = open(outputfiguredir + "/" + "mkrad_detx" + str("%d" % detx) + "_dety" + str("%d" % dety) + ".txt", "w")
        for onex, oney1 in zip(rc * self.p2arcsec, rp):
            outstr=str(onex) + " " + str(oney1) + " \n"
            fout.write(outstr) 
        fout.close()        


def calc_radialprofile(data, xg, yg, ds = 10, ndiv = 10, debug = False):
    """
    calc simple peak (just for consistendety check) and baricentric peak. 
    """    
    tmp = data.T
    nr  = np.linspace(0, ds, ndiv)
    rc = (nr[:-1] + nr[1:]) * 0.5
    rp = np.zeros(len(nr)-1)
    nrp = np.zeros(len(nr)-1)

    for i in range(ds*2):
        for j in range(ds*2):
            itx = int(xg - ds + i)
            ity = int(yg - ds + j)
            val = tmp[itx][ity]
            dist = dist2d(itx, ity, xg, yg)
            for k, (rin,rout) in enumerate(zip(nr[:-1], nr[1:])):
                if dist >=  rin and dist < rout:
                    rp[k] = rp[k] + val

    # normalize rp
    for m, (rin,rout) in enumerate(zip(nr[:-1], nr[1:])):
        darea = np.pi *  ( np.power(rout,2) - np.power(rin,2) )
        nrp[m] = rp[m] / darea
        if debug:
            print m, rp[m], nrp[m], rin, rout, darea
                    
    return rc, nrp
            
            
def dist2d(x, y, xg, yg):
    return np.sqrt( np.power( x - xg  ,2) + np.power( y - yg  ,2) )

def calc_center(data, idetx, idety, ds = 10):
    """
    calc simple peak (just for consistendety check) and baricentric peak. 
    """    
    tmp = data.T

    xmax = -1.
    ymax = -1.
    zmax = -1.
    
    xg = 0.
    yg = 0.
    tc = 0.
    
    for i in range(ds*2):
        for j in range(ds*2):
            tx = idetx - ds + i
            ty = idety - ds + j
            val = tmp[tx][ty]
            tc = tc + val
            xg = xg + val * tx
            yg = yg + val * ty
            
            if  val > zmax:
                zmax = val
                xmax = idetx - ds + i
                ymax = idety - ds + i
                
    if tc > 0:
        xg = xg / tc
        yg = yg / tc
    else:
        print "[ERROR] in calc_center : tc is negaive. Something wrong."
        sys.exit()
        
    print "..... in calc_center : [simple peak] xmax, ymax, zmax = ", xmax, ymax, zmax
    print "..... in calc_center : [total counts in ds] tc = ", tc
        
    if zmax < 0:
        print "[ERROR] in calc_center : zmax is not found. Something wrong."
        sys.exit()
        
    return xg, yg, tc


def plotwcs_all_old(clist, ra, dec, ds = 80, vmin = 1e-1, vmax = 20):
    """
    just plot the enlarged image around (ra,dec)
    xmm1, xmm2, xmm3, vla
    """

    print "\n..... plotwcs_all ...... "

    F = plt.figure(figsize=(14,4))
    F.subplots_adjust(left = 0.05, right = 0.96, top = 0.93, bottom = 0.05, wspace=0.3, hspace = 0.3)

    nmax = len(clist)
    for i, oneimage in enumerate(clist):

        # convert (ra,dec) [degrees] to pixel (X, Y)
        wrdcrd = np.array([[ ra, dec]], np.float_)
        pixcrd = oneimage.wcs.all_world2pix(wrdcrd,1)
        cx = pixcrd[0][0]
        cy = pixcrd[0][1]
        print "......... ", oneimage.filename
        print "wrdcrd = ", wrdcrd
        print "pixcrd = ", pixcrd, " cx, cy = ", cx, cy

        ax = plt.subplot(1, nmax, i + 1, projection=oneimage.wcs)

        if i < 3: # for XMM 
            plt.imshow(oneimage.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=vmin,vmax=vmax))
        else: # for VLA
            vmin = 1e-3
            vmax = 0.10
            plt.imshow(oneimage.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=vmin,vmax=vmax))

        ax.set_xlim(cx - ds, cx + ds)
        ax.set_ylim(cy - ds, cy + ds)

        cbar = plt.colorbar(orientation="horizontal",shrink=0.8)
        if i < 3: # for XMM 
            cbar.set_label(r'c/s/deg$^2$',size=12)
        else: # for VLA
            cbar.set_label(r'Jy/Beam',size=12)                
        plt.title(oneimage.dirname)        
        plt.xlabel('RA')
        plt.ylabel('Dec')

    outputfiguredir = "fig_image_all"
    commands.getoutput('mkdir -p ' + outputfiguredir)
    plt.savefig(outputfiguredir + "/" + "plotwcs_all_ra" + str("%d" % ra) + "_dec" + str("%d" % dec) + "_ds" + str("%d" % ds) + ".png")


def plotwcs_fourimages_astropy(infname, clist, ra, dec, dx = 80, dy = 80, vmin = 1e-1, vmax = 40, debug = False):
    """
    just plot the enlarged image around (detx,dety)
    xmm1, xmm2, xmm3, vla
    """

    print "\n..... plotwcs_fourimages_astropy ...... all in one figure"

    otag = infname.replace(".csv","").replace(".txt","").replace(".dat","")

    F = plt.figure(figsize=(14,4))
    F.subplots_adjust(left = 0.05, right = 0.96, top = 0.93, bottom = 0.05, wspace=0.3, hspace = 0.3)

    nmax = len(clist)

    for i, oneimage in enumerate(clist):

        # convert (ra,dec) [degrees] to pixel (X, Y)
        wrdcrd = np.array([[ ra, dec]], np.float_)
        pixcrd = oneimage.wcs.all_world2pix(wrdcrd,1)
        cx = pixcrd[0][0]
        cy = pixcrd[0][1]
        print "......... ", oneimage.filename
        print "wrdcrd = ", wrdcrd
        print "pixcrd = ", pixcrd, " cx, cy = ", cx, cy

        if i == 0: # XMM 0.4-0.75 keV
            ax = plt.subplot(1, 4, 1, projection=oneimage.wcs)

        elif i == 1: # XMM 0.75-1.3 keV
            ax = plt.subplot(1, 4, 2, projection=oneimage.wcs)

        elif i == 2: # XMM 2-7.2 keV
            ax = plt.subplot(1, 4, 3, projection=oneimage.wcs)
        elif i == 3: # VLA
            ax = plt.subplot(1, 4, 4, projection=oneimage.wcs)
        else:
            pass # continue to use same ax 

        if i < 3: # for XMM 
            plt.imshow(oneimage.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=vmin,vmax=vmax))
        else: # for VLA
            tvmin = 1e-3
            tvmax = 0.10
            plt.imshow(oneimage.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=tvmin,vmax=tvmax))

        ax.set_xlim(cx - dx, cx + dx)
        ax.set_ylim(cy - dy, cy + dy)
        ax.grid(color='black', ls='dotted')
#        ax.grid(color='white', ls='solid')

        if i < 3: # for XMM 
            cbar = plt.colorbar(orientation="horizontal",shrink=0.8)
            cbar.set_label(r'c/s/deg$^2$',size=12)
        else: # for VLA
            cbar = plt.colorbar(orientation="horizontal",shrink=0.8)
            cbar.set_label(r'Jy/Beam',size=12)                

        plt.title(oneimage.dirname)        
        plt.xlabel('RA')
        plt.ylabel('Dec')

    outputfiguredir = "plotwcs_fourimages_astropy"
    commands.getoutput('mkdir -p ' + outputfiguredir)
    outfig = outputfiguredir + "/" + "plotwcs_all_ra" + str("%d" % ra) + "_dec" + str("%d" % dec) \
                             + "_dx" + str("%d" % dx) + "_dy" + str("%d" % dy) + "_" + otag + ".png"
    plt.savefig(outfig)
    print "..... ", outfig, " is created. "
    if debug:
        plt.show()




def plotwcs_oneimages_astropy(infname, clist, ra, dec, dx = 80, dy = 80, vmin = 1e-1, vmax = 300, debug = False):
    """
    just plot one figure and save it. 
    """

    print "\n..... plotwcs_fourimages_astropy ...... all in one figure"

    otag = infname.replace(".csv","").replace(".txt","").replace(".dat","")

    nmax = len(clist)

    for i, oneimage in enumerate(clist):

        F = plt.figure(figsize=(14,8))
        F.subplots_adjust(left = 0.1, right = 0.96, top = 0.93, bottom = 0.05, wspace=0.3, hspace = 0.3)

        # convert (ra,dec) [degrees] to pixel (X, Y)
        wrdcrd = np.array([[ ra, dec]], np.float_)
        pixcrd = oneimage.wcs.all_world2pix(wrdcrd,1)
        cx = pixcrd[0][0]
        cy = pixcrd[0][1]
        print "......... ", oneimage.filename
        print "wrdcrd = ", wrdcrd
        print "pixcrd = ", pixcrd, " cx, cy = ", cx, cy

        if i == 0: # XMM 0.4-0.75 keV
            ax = plt.subplot(1, 1, 1, projection=oneimage.wcs)

        elif i == 1: # XMM 0.75-1.3 keV
            ax = plt.subplot(1, 1, 1, projection=oneimage.wcs)

        elif i == 2: # XMM 2-7.2 keV
            ax = plt.subplot(1, 1, 1, projection=oneimage.wcs)
        elif i == 3: # VLA
            ax = plt.subplot(1, 1, 1, projection=oneimage.wcs)
        else:
            pass # continue to use same ax 

        if i < 3: # for XMM 
            plt.imshow(oneimage.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=vmin,vmax=vmax))
        else: # for VLA
            tvmin = 1e-3
            tvmax = 0.10
            plt.imshow(oneimage.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=tvmin,vmax=tvmax))

        ax.set_xlim(cx - dx, cx + dx)
        ax.set_ylim(cy - dy, cy + dy)
        ax.grid(color='black', ls='dotted')
#        ax.grid(color='white', ls='solid')

        if i < 3: # for XMM 
            cbar = plt.colorbar(orientation="vertical",shrink=0.7)
            cbar.set_label(r'c/s/deg$^2$',size=8)
        else: # for VLA
            cbar = plt.colorbar(orientation="vertical",shrink=0.7)
            cbar.set_label(r'Jy/Beam',size=8)                

        plt.title(oneimage.dirname)        
        plt.xlabel('RA')
        plt.ylabel('Dec')

        outputfiguredir = "plotwcs_oneimages_astropy"
        commands.getoutput('mkdir -p ' + outputfiguredir)
        outfig = outputfiguredir + "/" + "plotwcs_all_ra" + str("%d" % ra) + "_dec" + str("%d" % dec) \
                             + "_dx" + str("%d" % dx) + "_dy" + str("%d" % dy) + "_" + otag + "_" + str("%02d" % i) + ".png"
    plt.savefig(outfig)
    print "..... ", outfig, " is created. "
    if debug:
        plt.show()



def plotwcs_oneimageswithcont_astropy(infname, clist, ra, dec, dx = 80, dy = 80, vmin = 1e-1, vmax = 300, debug = False, contnum = 20):
    """
    plot one figure overlaid with radio contour. 
    """

    print "\n..... plotwcs_fourimageswithcont_astropy ...... all in one figure"

    otag = infname.replace(".csv","").replace(".txt","").replace(".dat","")

    nmax = len(clist)

    radio = clist[-1] # the last one is regarded as radio image

    for i, oneimage in enumerate(clist):

        F = plt.figure(figsize=(14,8))
        F.subplots_adjust(left = 0.1, right = 0.96, top = 0.93, bottom = 0.05, wspace=0.3, hspace = 0.3)

        # convert (ra,dec) [degrees] to pixel (X, Y)
        wrdcrd = np.array([[ ra, dec]], np.float_)
        pixcrd = oneimage.wcs.all_world2pix(wrdcrd,1)
        cx = pixcrd[0][0]
        cy = pixcrd[0][1]
        print "......... ", oneimage.filename
        print "wrdcrd = ", wrdcrd
        print "pixcrd = ", pixcrd, " cx, cy = ", cx, cy

        if i == 0: # XMM 0.4-0.75 keV
            ax = plt.subplot(1, 1, 1, projection=oneimage.wcs)

        elif i == 1: # XMM 0.75-1.3 keV
            ax = plt.subplot(1, 1, 1, projection=oneimage.wcs)

        elif i == 2: # XMM 2-7.2 keV
            ax = plt.subplot(1, 1, 1, projection=oneimage.wcs)
        elif i == 3: # VLA
            ax = plt.subplot(1, 1, 1, projection=oneimage.wcs)
        else:
            pass # continue to use same ax 

        if i < 3: # for XMM 
            plt.imshow(oneimage.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=vmin,vmax=vmax))
#            ax.contour(radio.data, transform=ax.get_transform(WCS(oneimage.hdu.header)), levels=np.logspace(-3, -1., 20), colors='black', alpha=0.3)

        else: # for VLA
            tvmin = 1e-3
            tvmax = 0.10
            plt.imshow(oneimage.data, origin='lower', cmap=plt.cm.jet, norm=cr.LogNorm(vmin=tvmin,vmax=tvmax))
            ax.contour(oneimage.data, levels=np.logspace(-3, -1., contnum), colors='black', alpha=0.3)


        ax.set_xlim(cx - dx, cx + dx)
        ax.set_ylim(cy - dy, cy + dy)
        ax.grid(color='black', ls='dotted')
#        ax.grid(color='white', ls='solid')

        if i < 3: # for XMM 
            cbar = plt.colorbar(orientation="vertical",shrink=0.7)
            cbar.set_label(r'c/s/deg$^2$',size=8)
        else: # for VLA
            cbar = plt.colorbar(orientation="vertical",shrink=0.7)
            cbar.set_label(r'Jy/Beam',size=8)                

        plt.title(oneimage.dirname)        
        plt.xlabel('RA')
        plt.ylabel('Dec')

        outputfiguredir = "plotwcs_oneimageswithcont_astropy"
        commands.getoutput('mkdir -p ' + outputfiguredir)
        outfig = outputfiguredir + "/" + "plotwcs_all_ra" + str("%d" % ra) + "_dec" + str("%d" % dec) \
                             + "_dx" + str("%d" % dx) + "_dy" + str("%d" % dy) + "_" + otag + "_" + str("%02d" % i) + ".png"
        plt.savefig(outfig)
        print "..... ", outfig, " is created. "
        if debug:
            plt.show()





def main():

    """ start main loop """
    curdir = os.getcwd()

    """ Setting for options """
    usage = u'%prog FileList '    
    version = __version__
    parser = optparse.OptionParser(usage=usage, version=version)

    parser.add_option('-d', '--debug', action='store_true', help='The flag to show detailed information', metavar='DEBUG', default=False)     
    parser.add_option('-x', '--detx', action='store', type='int', help='det x', metavar='DETX', default=1239)
    parser.add_option('-y', '--dety', action='store', type='int', help='det y', metavar='DETY', default=1348)
    parser.add_option('-s', '--dataExtractSizeX', action='store', type='int', help='data size to be extracted from image in X', metavar='DS', default=80)
    parser.add_option('-u', '--dataExtractSizeY', action='store', type='int', help='data size to be extracted from image in Y', metavar='DS', default=80)

    parser.add_option('-n', '--numberOfDivision', action='store', type='int', help='number of division of annulus', metavar='NDIV', default=20)

    parser.add_option('-r', '--rightascension', action='store', type='float', help='right ascension', metavar='RIGHTASCENSION', default=289.005)
    parser.add_option('-c', '--declination', action='store', type='float', help='declination', metavar='DECLINATION', default=4.727)


    options, args = parser.parse_args()

    print "---------------------------------------------"
    print "| START  :  " + __file__

    debug =  options.debug
    detx =  options.detx
    dety =  options.dety
    dx =  options.dataExtractSizeX
    dy =  options.dataExtractSizeY

    ndiv =  options.numberOfDivision
    ra =  options.rightascension
    dec =  options.declination

    print "..... detx    ", detx
    print "..... dety    ", dety 
    print "..... dx    ", dx,  " (bin of the input image in X)"
    print "..... dy    ", dy,  " (bin of the input image in Y)"
    print "..... ndiv  ", ndiv
    print "..... debug ", debug
    print "..... ra    ", ra,  " (deg)"
    print "..... dec   ", dec, " (deg)"
    
    argc = len(args)

    if (argc < 1):
        print '| STATUS : ERROR '
        print parser.print_help()
        quit()


    print "\n| STEP1  : open the list of the files"    
    infname=args[0]
    filelistfile=open(infname, 'r')


    """ Create the list of input files """
    filelist = []
    detlist = []

    for onefile in filelistfile:
        if debug: print onefile.split()

        filelist.append(str(onefile.split(",")[0]))
        detlist.append(str(onefile.split(",")[1]))

    # strip() works in the same way as chop() in perl or ruby
    print "..... filelist = " + str(filelist)
    if debug: print filelist

    print "\n| STEP2  : Read each file and do process " + "\n"

    evelist = []
    for i, (filename,det) in enumerate(zip(filelist, detlist)):
        filenametag = filename.replace(".txt","")
        print "START : No." + str(i)  + " " + filename
        eve = Fits(filename, det, debug)
        eve.plotwcs_astropy(debug = debug) # plot the entire  image        

        evelist.append(eve)
        if debug:
            plt.show(debug)
        print "..... finish \n"

    # plot four images in one figure
    plotwcs_fourimages_astropy(infname, evelist, 288.55, 4.94, dx = 700, dy = 350, vmin = 1e-1, vmax = 300)

    # xmm each image in W50 with radio contour only on radio 
    plotwcs_oneimageswithcont_astropy(infname, evelist, 288.55, 4.94, dx = 700, dy = 350, vmin = 1e-1, vmax = 300)

    # entire image of W50
    plotwcs_oneimageswithcont_astropy(infname, evelist, 288.0, 4.98, dx = 1000, dy = 400, vmin = 1e-1, vmax = 300, contnum = 10)    


if __name__ == '__main__':
    main()
