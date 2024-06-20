#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
from astropy.io import fits
from pathlib import Path

params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

pixel_map = np.array([
    [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],  # DETY
    [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],  # DETX
    [12, 11, 9, 19, 21, 23, 14, 13, 10, 20, 22, 24, 16, 15, 17, 18, 25, 26, 8, 7, 0, 35, 33, 34, 6, 4, 2, 28, 31, 32, 5, 3, 1, 27, 29, 30]  # PIXEL
])

def plot_data_6x6(prevt, itypes, dumptext=False):
    data = fits.open(prevt)
    pulse = data[1].data['PULSEREC'].copy()
    pix_num_list = data[1].data['PIXEL'].copy()
    itype_list = data[1].data['ITYPE'].copy()
    data.close()

    dt = 80.0e-6  # hres fix
    x_time = np.arange(0, pulse.shape[-1], 1) * dt

    for itype in itypes:
        itype = int(itype)
        itype_str = get_itype_str(itype)

        path = Path(prevt)
        ofile = f'pulserecode_{path.stem}_{itype_str}.png'

        print(f'Load file name = {prevt}')
        print(f'Output file name = {ofile}')
        print(f'hres = {dt} s')
        print(f'ITYPE = {itype}:{itype_str}')

        fig, ax = plt.subplots(6, 6, figsize=(16, 9), sharex=True, sharey=True)
        for e in range(36):
            dety = 6 - pixel_map.T[e][0]
            detx = pixel_map.T[e][1] - 1
            pixel = pixel_map.T[e][2]
            pix_mask = pix_num_list == pixel
            if itype == -1:
                itype_mask = itype_list[pix_mask] <= 7
                ph_max = pulse.max()
            else:
                itype_mask = itype_list[pix_mask] == itype
                if pulse[itype_list == itype].size > 0:
                    ph_max = pulse[itype_list == itype].max()
                else:
                    ph_max = 0
            num_of_evt = len(pulse[pix_mask][itype_mask])
            print(f'DETY={dety}, DETX={detx}, PIXEL={pixel}')

            for j, p in enumerate(pulse[pix_mask][itype_mask]):
                if j == 0:
                    ax[dety, detx].text(0.07, ph_max - 500, fr'$\rm Pix = {pixel}$', ha='center', va='center')
                    ax[dety, detx].text(0.07, ph_max - 2500, fr'$\rm N   = {num_of_evt}$', ha='center', va='center')

                ax[dety, detx].plot(x_time, p)

            if dumptext and num_of_evt > 0:
                dump_to_npz(f"{path.stem}_{itype_str}_dety{dety}_detx{detx}_pixel{pixel}.npz", x_time, pulse[pix_mask][itype_mask])

        for i in range(6):
            ax[5, i].set_xlabel(r'$\rm time\ (s)$', fontsize=10)
            ax[i, 0].set_ylabel(r'$\rm PulseRec$', fontsize=10)

        plt.suptitle(f"Pulse Recode: {prevt}, ITYPE={itype}")
        plt.tight_layout()
        plt.savefig(ofile)

def get_itype_str(itype):
    itype_dict = {-1: 'all', 0: 'Hp', 1: 'Mp', 2: 'Ms', 3: 'Lp', 4: 'Ls', 5: 'BL', 6: 'EL', 7: '--'}
    return itype_dict.get(itype, 'unknown')

def dump_to_npz(filename, x_time, pulses):
    np.savez(filename, x_time=x_time, pulses=pulses)

def load_from_npz(filename):
    data = np.load(filename)
    return data['x_time'], data['pulses']

def main():
    parser = argparse.ArgumentParser(description="Process FITS files and plot data.")
    parser.add_argument('prevt', type=str, help='Input FITS file')
    parser.add_argument('--itypelist', type=str, default='0,1,2,3,4', help='Comma-separated list of itype values (default: 0,1,2,3,4)')
    parser.add_argument('--dumptext', action='store_true', help='Flag to dump x_time and pulse data to text files')

    args = parser.parse_args()
    itypes = [int(itype) for itype in args.itypelist.split(',')]
    
    plot_data_6x6(args.prevt, itypes, args.dumptext)

if __name__ == "__main__":
    main()
