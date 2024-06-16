#!/usr/bin/env python

""" xtend_pileup_gen_plfraction.py

This is a python script to plot the radial profile of the image.
History: 
2024-05-20 ; ver 1; S.Yamada
2024-05-21 ; ver 1.1; Y.Sakai, update pileup fraction image
2024-06-16 ; ver 2; S.Yamada, found a bug in pfrac thanks to Nobukawa-kun 
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cr
from astropy.io import fits
from astropy.wcs import WCS
from scipy.signal import convolve


params = {'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

def calc_pileupfraction(r):
    r = np.asarray(r)  # convert numpy.array
    result = np.where(r < 0, 0.0, (1. - (1. + r) * np.exp(-r)) / (1. - np.exp(-r)))
    return result

def interpolate_centers(radial_centers, radial_pileupfraction, target_values = [0.03, 0.01]):

    result = []
    
    for target in target_values:
        interp_center = -1
        for i in range(1, len(radial_pileupfraction)):
            if (radial_pileupfraction[i-1] >= target >= radial_pileupfraction[i] or
                radial_pileupfraction[i-1] <= target <= radial_pileupfraction[i]):
                interp_center = np.interp(target, 
                                          [radial_pileupfraction[i-1], radial_pileupfraction[i]], 
                                          [radial_centers[i-1], radial_centers[i]])
                break
        result.append(interp_center)
    
    return result    

class Fits:
    """Class for handling FITS files and plotting radial profiles."""
    
    def __init__(self, eventfile, debug=False):
        self.eventfilename = eventfile
        self.debug = debug

        if os.path.isfile(eventfile):
            self._initialize_fits_data(eventfile)
            self._print_initialization_info()
        else:
            print(f"ERROR: Can't find the fits file: {self.eventfilename}")
            quit()

    def _initialize_fits_data(self, eventfile):
        """Initialize FITS data and extract header information."""
        self.filename = eventfile
        self.hdu = fits.open(self.filename)[0]
        self.wcs = WCS(self.hdu.header)
        self.cdelt = self.wcs.wcs.cdelt
        self.p2deg = np.abs(self.cdelt[0])  # [deg/pixel]
        self.p2arcsec = 3600.0 * np.abs(self.cdelt[0])  # [arcsec/pixel]
        self.dirname = f"{eventfile.rsplit('.', 1)[0]}"
        self.data = self.hdu.data
        self.dateobs = self.hdu.header["DATE-OBS"]
        self.obsid = self.hdu.header["OBS_ID"]
        self.object = self.hdu.header["OBJECT"]
        self.datamode = self.hdu.header["DATAMODE"]
        self.exposure = self.hdu.header["EXPOSURE"]
        self.lastdel = self.hdu.header["LASTDEL"]
        print(f"[init] ... dateobs={self.dateobs}, obsid={self.obsid}, object={self.object}, datamode={self.datamode}")
        print(f"[init] ... exposure={self.exposure}, lastdel={self.lastdel} ....")

        self._process_data()

    def _process_data(self):
        """Process FITS data to handle NaNs and ensure positivity."""
        self.data[np.isnan(self.data)] = 0
        self.data = np.abs(self.data)

    def _print_initialization_info(self):
        """Print initialization info for debugging purposes."""
        print("[def _print_initialization_info]")
        print(f"      self.p2deg      = {self.p2deg} [deg/pixel]")
        print(f"      self.p2arcsec   = {self.p2arcsec} [arcsec/pixel]")

        if self.debug:
            print("... in __init__ Class Fits()")
            print(f"filename = {self.filename}")
            print(f"hdu = {self.hdu}")
            print(f"data.shape = {self.data.shape}")
            self.wcs.printwcs()

    def plotwcs(self, x_center, y_center, vmin=1e-1, vmax=20, manual=False):
        """Plot the entire image with WCS."""
        print("\n[def plotwcs]")
        if not manual:
            vmin = np.amin(self.hdu.data[self.hdu.data > 0]) + 0.1  # Avoid vmin being too close to 0 for LogNorm
            vmax = np.amax(self.hdu.data)
        print("     vmin = ", vmin, "vmax = ", vmax)

        print("vmin",vmin, "vmax",vmax)
        plt.figure(figsize=(12, 8))
        plt.figtext(0.1, 0.95, f"OBSID={self.obsid} {self.object} mode={self.datamode} exp={self.exposure} unit={self.p2arcsec}[arcsec/pixel] lastdel={self.lastdel}")
        plt.figtext(0.05, 0.025, f"fname={self.filename}")

        img = plt.imshow(self.data, origin='lower', cmap=plt.cm.CMRmap, norm=cr.LogNorm(vmin=vmin, vmax=vmax))        
        plt.scatter(x_center, y_center, c="k", s=300, marker="x", alpha=0.3)

        cbar = plt.colorbar(img)
        cbar.set_label('Intensity (c/bin)', rotation=270, labelpad=15)

        plt.xlabel('RA')
        plt.ylabel('Dec')

        outputfigname = f"{self.dirname}_plotwcs.png"
        plt.savefig(outputfigname)
        print(f"..... {outputfigname} is created.")

    def plotwcs_ps(self, x_center, y_center, search_radius=40, vmin=1e-1, vmax=20, manual=False):
        """Plot the enlarged image around the given center."""
        print("\n[def plotwcs_ps]")
        if not manual:
            vmin = np.amin(self.hdu.data[self.hdu.data > 0]) + 0.1  # Avoid vmin being too close to 0 for LogNorm
            vmax = np.amax(self.hdu.data)
        print("     vmin = ", vmin, "vmax = ", vmax)

        pix_coords = np.array([[x_center, y_center]], np.float_)
        world_coords = self.wcs.all_pix2world(pix_coords, 1)
        ra, dec = world_coords[0]

        self.ra = ra
        self.dec = dec

        print(f"x_center, y_center = {x_center}, {y_center}")
        print(f"ra, dec    = {ra}, {dec}")

        plt.figure(figsize=(12, 8))
        plt.figtext(0.1, 0.95, f"OBSID={self.obsid} {self.object} mode={self.datamode} exp={self.exposure} unit={self.p2arcsec}[arcsec/pixel] lastdel={self.lastdel}")
        plt.figtext(0.05, 0.025, f"fname={self.filename}")

        ax = plt.subplot(111, projection=self.wcs)
        img = plt.imshow(self.data, origin='lower', cmap=plt.cm.CMRmap, norm=cr.LogNorm(vmin=vmin, vmax=vmax))
        ax.scatter(x_center, y_center, c="k", s=300, marker="x", alpha=0.3)
        print(f"self.hdu.data[x_center][y_center] = {self.hdu.data[x_center][y_center]} (x_center, y_center) = ({x_center}, {y_center})")

        ax.set_xlim(x_center - search_radius, x_center + search_radius)
        ax.set_ylim(y_center - search_radius, y_center + search_radius)
        cbar = plt.colorbar(img)
        cbar.set_label('Intensity (c/bin)', rotation=270, labelpad=15)

        plt.xlabel('RA')
        plt.ylabel('Dec')

        outputfigname = f"{self.dirname}_plotwcs_ps_{x_center}_ycenter{y_center}.png"
        plt.savefig(outputfigname)
        print(f"..... {outputfigname} is created.")

    def mk_radialprofile(self, x_center, y_center, search_radius=40, ndiv=20, vmin=1e-1, vmax=20, manual=False, cellsize=3*3):
        """Create the radial profiles."""
        print("\n[def mk_radialprofile]")
        if not manual:
            vmin = np.amin(self.hdu.data[self.hdu.data > 0]) + 0.1  # Avoid vmin being too close to 0 for LogNorm
            vmax = np.amax(self.hdu.data)
        print("     vmin = ", vmin, "vmax = ", vmax)

        pix_coords = np.array([[x_center, y_center]], np.float_)
        world_coords = self.wcs.all_pix2world(pix_coords, 1)
        ra, dec = world_coords[0]

        self.ra = ra
        self.dec = dec
        print(f"x_center, y_center = {x_center}, {y_center}")
        print(f"ra, dec    = {ra}, {dec}")

        rc, rp = calc_radial_profile(self.data, x_center, y_center, search_radius=search_radius, ndiv=ndiv, exposure=self.exposure)
        radial_centers, radial_pileupfraction = calc_radial_pileupfraction(self.data, x_center, y_center, \
                           exposure=self.exposure, search_radius=search_radius, ndiv=ndiv, cellsize=cellsize, lastdel = self.lastdel)
        pileupfraction_img = calc_2d_pileupfraction(self.data, exposure=self.exposure, cellsize=cellsize, lastdel = self.lastdel)
        
        target_values = [0.03, 0.01] # pileup fraction 3%, 1%
        target_lss = ["-", "--"] # pileup linestyle
        pileup_result = interpolate_centers(radial_centers, radial_pileupfraction, target_values = target_values) 
        # Plot images and radial profiles
        fig = plt.figure(figsize=(16, 8))
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.87, wspace=0.5, hspace=0.4)
        plt.figtext(0.1, 0.95, f"OBSID={self.obsid} {self.object} mode={self.datamode} exp={self.exposure} unit={self.p2arcsec}[arcsec/pixel] lastdel={self.lastdel}")
        plt.figtext(0.1, 0.93, f"Input center [deg] (x, y, ra, dec) = ({x_center:.4f}, {y_center:.4f}, {ra:.4f}, {dec:.4f}) cellsize for pileup = {cellsize}")
        plt.figtext(0.05, 0.025, f"fname={self.filename}")

        ax1 = fig.add_subplot(2, 4, 1)
        self._plot_image(ax1, self.data, "(1) SKY image", x_center, y_center, vmin, vmax)

        ax2 = fig.add_subplot(2, 4, 5, projection=self.wcs)
        self._plot_image(ax2, self.data, "(2) Ra Dec image (FK5)", x_center, y_center, vmin, vmax, wcs=self.wcs)

        ax3 = fig.add_subplot(2, 4, 2)
        self._plot_image(ax3, self.data, "(3) SKY image", x_center, y_center, vmin, vmax, search_radius)

        ax4 = fig.add_subplot(2, 4, 6, projection=self.wcs)
        self._plot_image(ax4, self.data, "(4) Ra Dec image (FK5)", x_center, y_center, vmin, vmax, search_radius, wcs=self.wcs)

        ax5 = fig.add_subplot(2, 4, 3)
        self._plot_radial_profile(ax5, rc, rp, "(5) Radial profile (pix)", 'radial distance (pixel)', 'c s$^{-1}$ pixel$^{-2}$')

        ax6 = fig.add_subplot(2, 4, 7)
        self._plot_radial_profile(ax6, radial_centers, radial_pileupfraction, "(6) Pileup Fraction", 'radial distance (pixel)', 'fraction')
        for _target, _r, _ls in zip(target_values,pileup_result, target_lss):
            if _r > 0:
                print(f"***** pileup fraction ={_target*100:.2f} % at {_r:.2f} pixel *****")
                ax6.axhline(y=_target, color='g', ls=_ls, alpha=0.5, label=f"plfrac={_target*100:.2f} % at {_r:.2f} pixel")
                plt.legend(numpoints=1, frameon=True)

        ax7 = fig.add_subplot(2, 4, 4)
        pileup_vmin, pileup_vmax = self._pileup_image_vmin_vmax(pileupfraction_img, x_center, y_center, search_radius)
        self._plot_image(ax7, pileupfraction_img, "(7) Pileup Fraction image", x_center, y_center, pileup_vmin, pileup_vmax, search_radius, label='fraction')
        self._plot_contour(ax7, pileupfraction_img, x_center, y_center, search_radius, target_values, target_lss)

        ax8 = fig.add_subplot(2, 4, 8)
        zoom_in_range = int(search_radius/5)
        pileup_vmin, pileup_vmax = self._pileup_image_vmin_vmax(pileupfraction_img, x_center, y_center, zoom_in_range)
        self._plot_image(ax8, pileupfraction_img, "(8) Pileup Fraction image (zoom in)", x_center, y_center, pileup_vmin, pileup_vmax, zoom_in_range, label='fraction')
        self._plot_contour(ax8, pileupfraction_img, x_center, y_center, zoom_in_range, target_values, target_lss)


        outputfigname = f"{self.dirname}_radialprofile_xcenter{x_center}_ycenter{y_center}.png"
        plt.savefig(outputfigname)
        print(f"..... {outputfigname} is created.")

        self._save_radial_profile(rc, rp, f"{self.dirname}_radialprofile_xcenter{x_center}_ycenter{y_center}.txt")
        self._save_radial_profile(radial_centers, radial_pileupfraction, f"{self.dirname}_pileupfraction_xcenter{x_center}_ycenter{y_center}.txt")
        self._save_pileup_oneline(pileup_result, f"{self.dirname}_pileupfraction_xcenter{x_center}_ycenter{y_center}.csv")

    def _pileup_image_vmin_vmax(self, data, x_center, y_center, search_radius=None):
        if search_radius is None:
            data_copy = data.copy()
        else:
            data_copy = data[y_center-search_radius:y_center+search_radius+1, x_center-search_radius:x_center+search_radius+1].copy()
        
        clean_data = data_copy[~np.isnan(data_copy)]
        finite_data = clean_data[np.isfinite(clean_data)]
        positive_data = finite_data[finite_data > 0]
        vmin = np.min(positive_data)
        vmax = np.max(positive_data)

        return vmin, vmax

    def _plot_image(self, ax, data, title, x_center, y_center, vmin, vmax, search_radius=None, wcs=None, label=None):
        ax.set_title(title)
        im = ax.imshow(data, origin='lower', cmap=plt.cm.CMRmap, norm=cr.LogNorm(vmin=vmin, vmax=vmax), aspect='auto', extent=[0, data.shape[1], 0, data.shape[0]])
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, rotation=270, labelpad=15)
        ax.scatter(x_center, y_center, c="k", s=300, marker="x",alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if search_radius is not None:
            ax.set_xlim(x_center - search_radius, x_center + search_radius)
            ax.set_ylim(y_center - search_radius, y_center + search_radius)
        if wcs is not None:
            ax.coords.grid(True, color='gray', ls='solid')
        ax.set_aspect('equal', adjustable='box')

    def _plot_contour(self, ax, data, x_center, y_center, search_radius=None, levels=None, linestyles=None):
        sorted_indices = np.argsort(levels)
        sorted_levels = np.array(levels)[sorted_indices]
        sorted_linestyles = np.array(linestyles)[sorted_indices]

        cs = ax.contour(data, levels=sorted_levels, colors='g', origin='lower', linestyles=sorted_linestyles, extent=[0, data.shape[1], 0, data.shape[0]])

        if search_radius is not None:
            ax.set_xlim(x_center - search_radius, x_center + search_radius)
            ax.set_ylim(y_center - search_radius, y_center + search_radius)
        ax.set_aspect('equal', adjustable='box')

    def _plot_radial_profile(self, ax, rc, rp, title, xlabel, ylabel):
        ax.set_title(title)
        ax.errorbar(rc, rp, fmt="ko-", label="data")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(numpoints=1, frameon=True)

    def _save_radial_profile(self, rc, rp, outputcsv):
        with open(outputcsv, "w") as fout:
            for _rc, _rcsec, _rp in zip(rc, rc * self.p2arcsec, rp):
                fout.write(f"{_rc},{_rcsec},{_rp}\n")
        print(f"..... {outputcsv} is created.")

    def _save_pileup_oneline(self, pileup_result, outputcsv):
        rough_rate = np.sum(self.data)/self.exposure
        with open(outputcsv, "w") as fout:
            fout.write(f"{self.obsid},{pileup_result[0]},{pileup_result[1]},{self.object},{self.datamode},{self.exposure},{self.lastdel},{rough_rate}\n")
        print(f"..... {outputcsv} is created.")

def calc_radial_pileupfraction(data, x_center, y_center, search_radius=10, ndiv=10, exposure = 1e4, cellsize=3*3, debug=False, lastdel=4.0):
    """
    Calculate the radial pileup fraction around a given center.

    Parameters:
    data (2D array): The input data array.
    x_center (int): The x-coordinate of the center.
    y_center (int): The y-coordinate of the center.
    search_radius (int): The radius around the center to search for the profile. Default is 10.
    ndiv (int): The number of divisions in the radial profile. Default is 10.
    exposure (float): The exposure time to calculate pileup fraction. Default is 1e4.
    cellsize (float): The cell size for single phtoon. Default is 3x3, though need to check. 
    debug (bool): If True, print debug information. Default is False.
    lastdel (float): readout time for a given count rate

    Returns:
    tuple: The radial coordinates and the normalized radial profile.
    """
    transposed_data = data.T
    radial_bins = np.linspace(0, search_radius, ndiv)
    radial_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    radial_profile = np.zeros(len(radial_bins) - 1)

    x_indices, y_indices = np.meshgrid(np.arange(2 * search_radius), np.arange(2 * search_radius), indexing='ij')
    x_indices = x_center - search_radius + x_indices
    y_indices = y_center - search_radius + y_indices

    distances = dist2d(x_indices, y_indices, x_center, y_center)
    values = transposed_data[x_indices, y_indices]

    for k in range(len(radial_bins) - 1):
        mask = (distances >= radial_bins[k]) & (distances < radial_bins[k + 1])
        radial_profile[k] = np.sum(values[mask])

    # Normalize the radial profile and calculate pileup fraction 
    areas = np.pi * (radial_bins[1:]**2 - radial_bins[:-1]**2)
    # pile fraction is c/s/cellsize x lastdel ~ counts/1 readout. 
    radial_profile_per_areas_exposure = radial_profile * cellsize * lastdel / (areas * exposure) 
    radial_pileupfraction = calc_pileupfraction(radial_profile_per_areas_exposure)

    if debug:
        for m in range(len(radial_bins) - 1):
            print(m, radial_profile[m], radial_pileupfraction[m], radial_bins[m], radial_bins[m+1], areas[m])

    return radial_centers, radial_pileupfraction

def calc_2d_pileupfraction(data, exposure=1e4, cellsize=3*3, debug=False, lastdel = 4.0):
    """
    Calculate the pileup from image information.

    Parameters:
    data (2D array): The input data array.
    exposure (float): The exposure time to calculate pileup fraction. Default is 1e4.
    cellsize (float): The cell size for single phtoon. Default is 3x3, though need to check. 
    debug (bool): If True, print debug information. Default is False.
    lastdel (float): readout time for a given count rate

    Returns:
    2D array: The pileup 2d image.
    """
    cellsize_kernel = np.ones((3,3), dtype=data.dtype)
    cellsize_per_exposure = convolve(data, cellsize_kernel, mode='same') * lastdel / exposure
    pileupfraction_img = calc_pileupfraction(cellsize_per_exposure)

    if debug:
        print("vmin = ", np.min(pileupfraction_img), "vmax = ", np.max(pileupfraction_img), "mean = ", np.mean(pileupfraction_img))

    return pileupfraction_img

def calc_radial_profile(data, x_center, y_center, search_radius=10, ndiv=10, exposure = 1e4, debug=False):
    """
    Calculate the radial profile of the data around a given center.

    Parameters:
    data (2D array): The input data array.
    x_center (int): The x-coordinate of the center.
    y_center (int): The y-coordinate of the center.
    search_radius (int): The radius around the center to search for the profile. Default is 10.
    ndiv (int): The number of divisions in the radial profile. Default is 10.
    debug (bool): If True, print debug information. Default is False.

    Returns:
    tuple: The radial coordinates and the normalized radial profile.
    """
    transposed_data = data.T
    radial_bins = np.linspace(0, search_radius, ndiv)
    radial_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    radial_profile = np.zeros(len(radial_bins) - 1)

    x_indices, y_indices = np.meshgrid(np.arange(2 * search_radius), np.arange(2 * search_radius), indexing='ij')
    x_indices = x_center - search_radius + x_indices
    y_indices = y_center - search_radius + y_indices

    distances = dist2d(x_indices, y_indices, x_center, y_center)
    values = transposed_data[x_indices, y_indices]

    for k in range(len(radial_bins) - 1):
        mask = (distances >= radial_bins[k]) & (distances < radial_bins[k + 1])
        radial_profile[k] = np.sum(values[mask])

    # Normalize the radial profile
    areas = np.pi * (radial_bins[1:]**2 - radial_bins[:-1]**2)
    normalized_radial_profile = radial_profile / (areas * exposure) 

    if debug:
        for m in range(len(radial_bins) - 1):
            print(m, radial_profile[m], normalized_radial_profile[m], radial_bins[m], radial_bins[m+1], areas[m])

    return radial_centers, normalized_radial_profile

def dist2d(x, y, x_center, y_center):
    """Calculate the Euclidean distance between points (x, y) and (x_center, y_center)."""
    return np.hypot(x - x_center, y - y_center)

def calc_center(data, x_center, y_center, search_radius=10):
    """
    Calculate the simple peak (just for consistency check) and barycentric peak.
    
    Parameters:
    data (2D array): The input data array.
    x_center (int): The x-coordinate of the initial guess for the peak.
    y_center (int): The y-coordinate of the initial guess for the peak.
    search_radius (int): The radius around the initial guess to search for the peak. Default is 10.
    
    Returns:
    tuple: The coordinates of the barycentric peak (barycenter_x, barycenter_y) and the total counts (total_counts).
    """
    transposed_data = data.T
    x_indices, y_indices = np.meshgrid(np.arange(-search_radius, search_radius), np.arange(-search_radius, search_radius), indexing='ij')
    x_indices += x_center
    y_indices += y_center
    
    values = transposed_data[x_indices, y_indices]
    total_counts = np.sum(values)
    
    if total_counts > 0:
        barycenter_x = np.sum(values * x_indices) / total_counts
        barycenter_y = np.sum(values * y_indices) / total_counts
    else:
        print("[ERROR] in calc_center: Total counts are non-positive. Something went wrong.")
        sys.exit(1)

    peak_idx = np.unravel_index(np.argmax(values), values.shape)
    peak_x, peak_y = x_indices[peak_idx], y_indices[peak_idx]
    peak_value = values[peak_idx]
    
    print(f"..... in calc_center: [simple peak] x_max, y_max, z_max = {peak_x}, {peak_y}, {peak_value}")
    print(f"..... in calc_center: [total counts in search radius] total_counts = {total_counts}")

    if peak_value < 0:
        print("[ERROR] in calc_center: Peak value not found. Something went wrong.")
        sys.exit(1)
        
    return barycenter_x, barycenter_y, total_counts

def main():
    """Start main loop"""
    curdir = os.getcwd()

    """Setting for options"""
    usage = '%(prog)s fits.img'
    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument('-o', '--auto_updated_center', action='store_false', help='Flag to automatically calculate center', default=True)
    parser.add_argument('-d', '--debug', action='store_true', help='Flag to show detailed information', default=False)
    parser.add_argument('-m', '--manual', action='store_true', help='Flag to use vmax, vmin', default=False)
    parser.add_argument('-x', '--x_center', type=int, help='x coordinate of center', default=1215)
    parser.add_argument('-y', '--y_center', type=int, help='y coordinate of center', default=1215)
#   parser.add_argument('-c', '--cellsize', type=int, help='cell size for a single event', default=3*3)
    parser.add_argument('-c', '--cellsize', type=int, help='cell size for a single event', default=1 + 4 )
    parser.add_argument('-a', '--vmax', type=float, help='VMAX', default=3e3)
    parser.add_argument('-i', '--vmin', type=float, help='VMIN', default=1)
    parser.add_argument('-s', '--search_radius', type=int, help='Search radius for data extraction', default=80)
    parser.add_argument('-n', '--numberOfDivision', type=int, help='Number of divisions for annulus', default=20)
    parser.add_argument('filename', help='Input FITS file')

    options = parser.parse_args()

    print(f"---------------------------------------------\n| START  :  {__file__}")
    print(f"..... auto_updated_center {options.auto_updated_center}")
    print(f"..... x_center {options.x_center} y_center {options.y_center}")
    print(f"..... search_radius {options.search_radius} (bin of image) ndiv {options.numberOfDivision} cellsize {options.cellsize}")
    print(f"..... debug {options.debug} manual {options.manual}")
    print(f"..... vmax {options.vmax} vmin {options.vmin}")

    filename = options.filename

    print(f"\n| Read each file and do process \n")
    print(f"START : {filename}")
    eve = Fits(filename, options.debug)

    if options.auto_updated_center:
        print(".....(update calc_center).....")
        x_center, y_center, _ = calc_center(eve.data, options.x_center, options.y_center, search_radius=options.search_radius)
        x_center, y_center = int(x_center), int(y_center)
        print(f"(x_center, y_center) is updated from ({options.x_center}, {options.y_center}) to ({x_center}, {y_center}).")

    eve.plotwcs(x_center, y_center, vmax=options.vmax, vmin=options.vmin, manual=options.manual)
    eve.plotwcs_ps(x_center, y_center, search_radius=options.search_radius, vmax=options.vmax, vmin=options.vmin, manual=options.manual)
    eve.mk_radialprofile(x_center, y_center, search_radius=options.search_radius, \
                           ndiv=options.numberOfDivision, vmax=options.vmax, vmin=options.vmin, \
                           manual=options.manual, cellsize=options.cellsize)

    print("| finish \n")

if __name__ == '__main__':
    main()