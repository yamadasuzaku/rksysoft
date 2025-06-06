#!/usr/bin/env python 

import argparse
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os 

params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'

g_types = [r'$\rm Hp$', r'$\rm Mp$', r'$\rm Ms$', r'$\rm Lp$', r'$\rm Ls$']

def filter_strings_by_indices(strings, indices):
    """
    指定されたインデックスを元に、文字列の配列をフィルタリングする。

    :param strings: フィルタリング対象の文字列配列
    :param indices: 取得したいインデックスの整数配列
    :return: フィルタリングされた文字列の配列
    """
    return [strings[i] for i in indices]

def propagate_division_error(y1, yerr1, y2, yerr2):
    """
    Calculate the propagated error for the division y1 / y2.
    
    Parameters:
    y1 (float or np.array): Numerator value(s)
    yerr1 (float or np.array): Error in the numerator value(s)
    y2 (float or np.array): Denominator value(s)
    yerr2 (float or np.array): Error in the denominator value(s)
    
    Returns:
    result (float or np.array): Result of the division y1 / y2
    propagated_error (float or np.array): Propagated error of the result
    """
    # Initialize result and propagated_error with zeros
    result = np.zeros_like(y1, dtype=float)
    propagated_error = np.zeros_like(y1, dtype=float)

    # Find indices where neither the numerator nor the denominator is zero
    valid_indices = (y1 > 0) & (y2 > 0)

    # Calculate division and propagated error only for valid indices
    result[valid_indices] = y1[valid_indices] / y2[valid_indices]
    propagated_error[valid_indices] = result[valid_indices] * np.sqrt(
        (yerr1[valid_indices] / y1[valid_indices]) ** 2 +
        (yerr2[valid_indices] / y2[valid_indices]) ** 2
    )

    return result, propagated_error

def get_filename_without_extension(filepath):
    """
    ファイルパスからディレクトリと拡張子を削除したファイル名を取得する関数
    """
    basename = os.path.basename(filepath)
    filename_without_extension = os.path.splitext(basename)[0]
    return filename_without_extension

def pi_to_ev(pi):
    """Convert PI units to energy in eV."""
    return pi * 0.5 + 0.25

def ev_to_pi(ev):
    """Convert energy in eV to PI units."""
    return 2 * ev - 0.5

def gen_energy_hist(epi2, bin_width):
    bins = np.arange(0, 40e3, bin_width)
    ncount, bin_edges = np.histogram(epi2, bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_half_width = bin_width / 2
    ncount_sqrt = np.sqrt(ncount)
#    energy = pi_to_ev(bin_centers)
    energy = bin_centers
    return energy, ncount, bin_half_width, ncount_sqrt

def plot_spec_6x6(ifile, bin_width, ene_min, ene_max, itypenames = [0], commonymax = True, ratioflag=False, \
                  outtag="auto", ylogflag=False, plotflag=False, commonymaxvalue = None):
    pixel_map = np.array([
        [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], # DETY
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6], # DETX
        [12, 11, 9, 19, 21, 23, 14, 13, 10, 20, 22, 24, 16, 15, 17, 18, 25, 26, 8, 7, 0, 35, 33, 34, 6, 4, 2, 28, 31, 32, 5, 3, 1, 27, 29, 30] # PIXEL
    ])
    
    itype_name = ['Hp', 'Mp', 'Ms', 'Lp', 'Ls']
    data = fits.open(ifile)
    objectname = data[1].header["OBJECT"]

    itype_list = data[1].data['ITYPE']
    pix_num_list = data[1].data['PIXEL']
    epi2 = data[1].data['EPI2']

    fig, ax = plt.subplots(6, 6, figsize=(21.3, 12), sharex=True)
    fig.suptitle(f'{ifile}: {objectname}, energy {ene_min}-{ene_max} eV, itype {outtag}')

    g_ymax = 0

    if ratioflag:
        # Initialize the dictionary
        stored_hist_dic = {}
        for itype in itypenames:
            cutid = np.where( (itype_list==itype)&((pix_num_list==0)|(pix_num_list==17)|(pix_num_list==18)|(pix_num_list==35)) )[0]
            epi2_itype_allpixel = epi2[cutid]
            energy, ncount, xerr, yerr = gen_energy_hist(epi2_itype_allpixel, bin_width)
            stored_hist_dic[itype] = (energy, ncount, xerr, yerr)

    for i in range(36):
        dety = 6 - pixel_map.T[i][0]
        detx = pixel_map.T[i][1] - 1
        pixel = pixel_map.T[i][2]
        mask_pix = pix_num_list == pixel
        epi2_pix = epi2[mask_pix]
        itype_pix = itype_list[mask_pix]

        y_min, y_max = np.inf, -np.inf

        for itype in itypenames:
            print(f"{objectname} PIXEL={pixel}: ITYPE={itype}, {itype_name[itype]}")
            mask_itype = itype_pix == itype
            epi2_pix_itype = epi2_pix[mask_itype]

            energy, ncount, xerr, yerr = gen_energy_hist(epi2_pix_itype, bin_width)
            mask = (ene_min < energy) & (energy < ene_max)

            energy, ncount, yerr = energy[mask], ncount[mask], yerr[mask]

            if mask.any():
                current_ymin = np.amin(ncount)
                current_ymax = np.amax(ncount)
                if g_ymax < current_ymax:
                    g_ymax = current_ymax

                y_min = min(y_min, current_ymin)
                y_max = max(y_max, current_ymax)
                ax[dety, detx].set_title(rf"PIXEL={pixel}")
                if ratioflag:
                    ave_energy, ave_ncount, ave_xerr, ave_yerr = stored_hist_dic[itype]          
                    ave_energy, ave_ncount, ave_yerr = ave_energy[mask], ave_ncount[mask], ave_yerr[mask]
                    ratio, ratio_err = propagate_division_error(ncount, yerr, ave_ncount, ave_yerr)      
                    # ax[dety, detx].errorbar(energy, ncount/ave_ncount, \
                    #       yerr=ncount/ave_ncount * np.sqrt( (yerr/ncount)**2 + (ave_yerr/ave_ncount)**2),ls='-')
                    handle = ax[dety, detx].errorbar(energy, ratio, xerr=xerr, yerr=ratio_err, ls='-')
                    # Set error bar transparency
                    for bar in handle[2]:
                        bar.set_alpha(0.2)            

                else:
                    handle = ax[dety, detx].errorbar(energy, ncount, xerr=xerr, yerr=yerr, ls='-')
                    # Set error bar transparency
                    for bar in handle[2]:
                        bar.set_alpha(0.2)            


                ax[dety, detx].set_xlim(ene_min, ene_max)

                if ratioflag: # autoscale when commonymax is false
                    if ylogflag:
                        ax[dety, detx].semilogy()                
                else:                      
                    if y_min > 0 and y_max > 0:
                        ax[dety, detx].set_ylim(y_min, y_max + 5)
                    if np.any(ncount > 0):
                        if ylogflag:
                            ax[dety, detx].semilogy()                

        if dety == 5 and detx == 0:  # 左下のパネルにのみ表示
            filtered_g_types = filter_strings_by_indices(g_types, itypenames)
            ax[dety, detx].legend(filtered_g_types, loc='lower left')

    for i in range(6):
        if ratioflag:
            ax[i, 0].set_ylabel('ratio to cen4')
            ax[5, i].set_xlabel(rf'$\rm Energy\ (eV)$')            
        else:
            ax[i, 0].set_ylabel(rf'$\rm Counts/{bin_width}eV$')
            ax[5, i].set_xlabel(rf'$\rm Energy\ (eV)$')

    if commonymax:
        if commonymaxvalue:
            g_ymax = commonymaxvalue
        print("..... enter commonymax with g_ymax = ", g_ymax)                
        for i in range(36):
            dety = 6 - pixel_map.T[i][0]
            detx = pixel_map.T[i][1] - 1
            if ratioflag:
                ax[dety, detx].set_ylim(0.01,10)
                if ylogflag:
                    ax[dety, detx].semilogy()                
            else:
                ax[dety, detx].set_ylim(0.1,g_ymax)
                if ylogflag:
                    ax[dety, detx].semilogy()                

    plt.tight_layout()
    ftag = get_filename_without_extension(ifile)
    if ratioflag:
        outfile = f'resolve_spec_plot6x6_{ftag}_EneRan{ene_min}-{ene_max}eV_itype{outtag}_bin{bin_width}_ratio.png'
    else:
        outfile = f'resolve_spec_plot6x6_{ftag}_EneRan{ene_min}-{ene_max}eV_itype{outtag}_bin{bin_width}.png'        
    print(f'..... {outfile} is created.')    
    plt.savefig(outfile)

    if ratioflag:
        fig, ax = plt.subplots(figsize=(8,10), sharex=True)
        fig.suptitle(f'Check all spec, {ifile}: Energy range {ene_min}-{ene_max} eV')

        for itype in itypenames:
            ave_energy, ave_ncount, ave_xerr, ave_yerr = stored_hist_dic[itype]          
            ax.errorbar(ave_energy, ave_ncount, xerr=ave_xerr, yerr=ave_yerr, fmt='o-', label=itype_name[itype])
        plt.legend()
        plt.tight_layout()  
        plt.savefig("checkall_" + outfile)

    if plotflag:
        plt.show()        

def main():
    parser = argparse.ArgumentParser(description='Plot 6x6 energy spectrum.')
    parser = argparse.ArgumentParser(
             description='This program is used to plot 6x6 energy spectrum.',
              epilog='''
Example (1): plot spectra from 2keV to 20 keV witn 400 eV bin, Hp only 
  python resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py xa300049010rsl_p0px3000_cl.evt -b 400 -l 2000 -x 20000 -y 0
Example (2): plot spectral ratios from 2keV to 20 keV witn 400 eV bin, Hp only 
  python resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py xa300049010rsl_p0px3000_cl.evt -r -b 400 -l 2000 -x 20000 -y 0 -c -g
  ''',
  formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('filename', type=str, help='Input file name')
    parser.add_argument('--bin_width', '-b', type=float, default=4, help='Bin width for histogram')
    parser.add_argument('--ene_min', '-l', type=float, default=6300, help='Minimum energy')
    parser.add_argument('--ene_max', '-x', type=float, default=6900, help='Maximum energy')
    parser.add_argument('--commonymax', '-c', action='store_false', help='Flag to set global ymax')
    parser.add_argument('--commonymaxvalue', '-cval', type=float, default=None, help='max value used when commonymax is true.')
    parser.add_argument('--ratioflag', '-r', action='store_true', help='Flag to make a spectral ratio to average of central four pixels')
    parser.add_argument('--ylogflag', '-g', action='store_false', help='Flag not to make y log (linear when used.)')
    parser.add_argument('--itypenames', '-y', type=str, help='Comma-separated list of itype', default='0,1,2,3,4')
    parser.add_argument('--plotflag', '-p', action='store_true', help='Flag to do show')

    args = parser.parse_args()
    itypenames = list(map(int, args.itypenames.split(',')))
    itypeinfo = args.itypenames.replace(",","_")

    plot_spec_6x6(ifile=args.filename, bin_width=args.bin_width, ene_min=args.ene_min, ene_max=args.ene_max, \
                  itypenames = itypenames, commonymax = args.commonymax, ratioflag=args.ratioflag, outtag=itypeinfo,\
                  ylogflag=args.ylogflag, plotflag=args.plotflag, commonymaxvalue = args.commonymaxvalue)

if __name__ == "__main__":
    main()
