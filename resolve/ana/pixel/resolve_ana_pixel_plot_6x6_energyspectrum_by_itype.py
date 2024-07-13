#!/usr/bin/env python 

import argparse
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os 

params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'

def get_filename_without_extension(filepath):
    """
    ファイルパスからディレクトリと拡張子を削除したファイル名を取得する関数
    """
    basename = os.path.basename(filepath)
    filename_without_extension = os.path.splitext(basename)[0]
    return filename_without_extension

def pi2e(pi):
    """
    Convert PI to energy using the formula:
    energy = PI * 0.5 + 0.5
    """
    return pi * 0.5 + 0.5 

def gen_energy_hist(pi, bin_width):
    bins = np.arange(0, 40e3, bin_width)
    ncount, bin_edges = np.histogram(pi, bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_half_width = bin_width / 2
    ncount_sqrt = np.sqrt(ncount)
    energy = pi2e(bin_centers)
    return energy, ncount, bin_half_width, ncount_sqrt

def plot_spec_6x6(ifile, bin_width, ene_min, ene_max, itypemax = 5, commonymax = True):
    pixel_map = np.array([
        [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], # DETY
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6], # DETX
        [12, 11, 9, 19, 21, 23, 14, 13, 10, 20, 22, 24, 16, 15, 17, 18, 25, 26, 8, 7, 0, 35, 33, 34, 6, 4, 2, 28, 31, 32, 5, 3, 1, 27, 29, 30] # PIXEL
    ])
    
    itype_name = ['Hp', 'Mp', 'Ms', 'Lp', 'Ls']
    data = fits.open(ifile)
    itype_list = data[1].data['ITYPE']
    pix_num_list = data[1].data['PIXEL']
    pi = data[1].data['PI']

    fig, ax = plt.subplots(6, 6, figsize=(21.3, 12), sharex=True)
    fig.suptitle(f'{ifile}: Energy range {ene_min}-{ene_max} eV')

    g_ymax = 0

    for i in range(36):
        dety = 6 - pixel_map.T[i][0]
        detx = pixel_map.T[i][1] - 1
        pixel = pixel_map.T[i][2]
        mask_pix = pix_num_list == pixel
        pi_pix = pi[mask_pix]
        itype_pix = itype_list[mask_pix]

        y_min, y_max = np.inf, -np.inf

        for itype in range(itypemax):
            print(f"PIXEL={pixel}: ITYPE={itype}, {itype_name[itype]}")
            mask_itype = itype_pix == itype
            pi_pix_itype = pi_pix[mask_itype]

            energy, ncount, xerr, yerr = gen_energy_hist(pi_pix_itype, bin_width)
            mask = (ene_min < energy) & (energy < ene_max)

            if mask.any():
                current_ymin = ncount[mask].min()
                current_ymax = ncount[mask].max()
                if g_ymax < current_ymax:
                    g_ymax = current_ymax

                y_min = min(y_min, current_ymin)
                y_max = max(y_max, current_ymax)
                ax[dety, detx].set_title(rf"PIXEL={pixel}")
                ax[dety, detx].errorbar(energy, ncount, xerr=xerr, yerr=yerr, ls='-')
                ax[dety, detx].set_xlim(ene_min, ene_max)
                if y_min > 0 and y_max > 0:
                    ax[dety, detx].set_ylim(y_min, y_max + 5)
                if np.any(ncount > 0):
                    ax[dety, detx].semilogy()                

        if dety == 5 and detx == 0:  # 左下のパネルにのみ表示
            if itypemax < 3:
                ax[dety, detx].legend([r'$\rm Hp$', r'$\rm Mp$', r'$\rm Ms$'], loc='lower left')
            else:
                ax[dety, detx].legend([r'$\rm Hp$', r'$\rm Mp$', r'$\rm Ms$',r'$\rm Lp$', r'$\rm Ls$'], loc='lower left')

    for i in range(6):
        ax[i, 0].set_ylabel(rf'$\rm Counts/{bin_width}eV$')
        ax[5, i].set_xlabel(rf'$\rm Energy\ (eV)$')

    if commonymax:
        for i in range(36):
            dety = 6 - pixel_map.T[i][0]
            detx = pixel_map.T[i][1] - 1
            ax[dety, detx].set_ylim(0.1,g_ymax)
            ax[dety, detx].semilogy()                

    plt.tight_layout()
    ftag = get_filename_without_extension(ifile)
    outfile = f'resolve_spec_plot6x6_{ftag}_EneRan{ene_min}-{ene_max}eV.png'
    print(f'..... {outfile} is created.')
    plt.savefig(outfile)

def main():
    parser = argparse.ArgumentParser(description='Plot 6x6 energy spectrum.')
    parser.add_argument('filename', type=str, help='Input file name')
    parser.add_argument('--bin_width', '-b', type=float, default=4, help='Bin width for histogram')
    parser.add_argument('--ene_min', '-l', type=float, default=6300, help='Minimum energy')
    parser.add_argument('--ene_max', '-x', type=float, default=6900, help='Maximum energy')
    parser.add_argument('--itypemax', '-i', type=int, default=5, help='Max of ITYPE')
    parser.add_argument('--commonymax', '-c', action='store_false', help='Flag to set global ymax')
    args = parser.parse_args()

    plot_spec_6x6(ifile=args.filename, bin_width=args.bin_width, ene_min=args.ene_min, ene_max=args.ene_max, \
                  itypemax = args.itypemax, commonymax = args.commonymax)

if __name__ == "__main__":
    main()
