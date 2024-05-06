#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import datetime
import argparse
import sys
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import scipy.optimize as so
import scipy.special

# Constants
MJD_REFERENCE_DAY = 58484
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')
TIME_50MK = 150526800.0

# Plotting Configuration
plt.rcParams['font.family'] = 'serif'
usercmap = plt.get_cmap('jet')
cNorm = Normalize(vmin=0, vmax=35)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=usercmap)

# Observation Type Information
itypename = [0, 1, 2, 3, 4]
typename = ["Hp", "Mp", "Ms", "Lp", "Ls"]

def ev_to_pi(ev):
    """Convert energy in eV to PI units."""
    return (ev - 0.5) * 2

def pi_to_ev(pi):
    """Convert PI units to energy in eV."""
    return pi * 0.5 + 0.5

# Define the energy range
emin, emax = 5870, 5920  # Energy range in eV
pimin, pimax = ev_to_pi(emin), ev_to_pi(emax)
binnum = int(pimax - pimin)

def open_fits_data(filename):
    """Open a FITS file and return its data, or exit if the file is not found."""
    try:
        return fits.open(filename)[1].data
    except FileNotFoundError:
        print(f"ERROR: File not found {filename}")
        sys.exit()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze MnKa spectra from a FITS file.',
    	usage='''
    python resolve_ana_pixel_ql_fit_MnKa_v1.py xa900001010rsl_p0px5000_cl.evt
    python resolve_ana_pixel_ql_fit_MnKa_v1.py xa900001010rsl_p0px5000_cl.evt  -s 161805543 --name after 
    python resolve_ana_pixel_ql_fit_MnKa_v1.py xa900001010rsl_p0px5000_cl.evt  -e 161801172 --name before
    ''')
    	
    parser.add_argument('filename', help='The filename of the FITS file to process.')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-s','--start', type=float, help='starttime', default=-1)
    parser.add_argument('-e','--end', type=float, help='endtime', default=9e9)
    parser.add_argument('-n', '--name',type=str, help='file key name', default="cl")

    return parser.parse_args()

def calcchi(params,consts,model_func,xvalues,yvalues,yerrors, debug=False):
    model = model_func(xvalues,params,consts)
    # avoid zero errors
    valid_indices = yerrors > 0
    # filter data
    filtered_yvalues = yvalues[valid_indices]
    filtered_model = model[valid_indices]
    filtered_yerrors = yerrors[valid_indices]
    # calc chi
    chi = (filtered_yvalues - filtered_model) / filtered_yerrors
    chi2 = np.sum(np.power(chi,2.0))
    if debug: print("chi2 = ", chi2)   
    return(chi)

# optimizer
def solve_least_squares(xvalues,yvalues,yerrors,param_init,consts,model_func):
    print("..... do fit in solve_least_squares")
    result = so.least_squares(
    calcchi, 
    param_init, 
    args=(consts, model_func, xvalues, yvalues, yerrors),
    )    

    param_result = result.x
    # ヤコビアンを取得
    J = result.jac
    n = len(yvalues)  # データ点の数
    p = len(result.x)  # パラメータの数
    # 残差の平方和を自由度で割って分散を計算
    residuals = result.fun
    sigma2 = np.sum(residuals**2) / (n - p) 
    # 共分散行列を計算
    covariance_matrix = np.linalg.inv(J.T.dot(J)) * sigma2    
    # パラメータの標準誤差を抽出
    error_result = np.sqrt(np.diag(covariance_matrix))    
    # 結果表示
    print("Optimized Parameters:", param_result)
    print("Parameter Errors:", error_result)

    dof = len(xvalues) - 1 - len(param_init)
    chi2 = np.sum(np.power(calcchi(param_result,consts,model_func,xvalues,yvalues,yerrors),2.0))

    return([param_result, error_result, chi2, dof])

def mymodel(x,params, consts, tailonly = False):
    norm,gw,gain,bkg1,bkg2 = params    
    # norm : nomarlizaion
    # gw : sigma of gaussian
    # gain : gain of the spectrum 
    # bkg1 : constant of background
    # bkg2 : linearity of background    
    initparams = [norm,gw,gain,bkg1,bkg2]
    def rawfunc(x): # local function, updated when mymodel is called 
        return MnKalpha(x,initparams,consts=consts)               
    return rawfunc(x)

def MnKalpha(xval,params,consts=[]):
    norm,gw,gain,bkg1,bkg2 = params
    # norm : normalization 
    # gw : sigma of the gaussian 
    # gain : if gain changes
    # consttant facter if needed 
    # Mn K alpha lines, Holzer, et al., 1997, Phys. Rev. A, 56, 4554, + an emperical addition
    energy = np.array([ 5898.853, 5897.867, 5894.829, 5896.532, 5899.417, 5902.712, 5887.743, 5886.495])
    lgamma =  np.array([    1.715,    2.043,    4.499,    2.663,    0.969,   1.5528,    2.361,    4.216]) # full width at half maximum
    amp =    np.array([    0.790,    0.264,    0.068,    0.096,   0.0714,   0.0106,    0.372,      0.1])

    prob = (amp * lgamma) / np.sum(amp * lgamma) # probabilites for each lines. 

    model_y = 0 
    if len(consts) == 0:
        consts = np.ones(len(energy))
    else:
        consts = consts

    for i, (ene,lg,pr,con) in enumerate(zip(energy,lgamma,prob,consts)):
        voi = voigt(xval,[ene*gain,lg*0.5,gw])
        model_y += norm * con * pr * voi

    background = bkg1 * np.ones(len(xval)) + (xval - np.mean(xval)) * bkg2
    model_y = model_y + background
    return model_y

def voigt(xval,params):
    center,lw,gw = params
    # center : center of Lorentzian line
    # lw : HWFM of Lorentzian (half-width at half-maximum (HWHM))
    # gw : sigma of the gaussian 
    z = (xval - center + 1j*lw)/(gw * np.sqrt(2.0))
    w = scipy.special.wofz(z)
    model_y = (w.real)/(gw * np.sqrt(2.0*np.pi))
    return model_y

def process_data(data, TRIGTIME_FLAG=False, AC_FLAG=False):
    time = data["TRIGTIME"] if TRIGTIME_FLAG else data["TIME"]
    itype = data["AC_ITYPE"] if AC_FLAG else data["ITYPE"]
    if len(time) == 0:
        print("ERROR: data is empty", time)
        sys.exit()
    additional_columns = [data[col] for col in ["PI", "RISE_TIME", "DERIV_MAX", "PIXEL"]]
    sortid = np.argsort(time)
    sorted_columns = [col[sortid] for col in [time, itype] + additional_columns]
    dtime = np.array([REFERENCE_TIME.datetime + datetime.timedelta(seconds=float(t)) for t in sorted_columns[0]])
    print(f"data from {dtime[0]} --> {dtime[-1]}")
    dt = np.diff(sorted_columns[0])
    return [column[:-1] for column in sorted_columns], dt, dtime[:-1]

def plot_histogram(pi_filtered, color, itype_, filename, label_suffix, alpha=0.9):

    outfname=f"ql_plotspec_MnK_{filename.replace('.evt', '').replace('.gz', '')}.png"
    title=f"Spectra of {filename}"

    plt.figure(figsize=(11, 7))
    plt.subplots_adjust(right=0.8) # make the right space bigger
    plt.xscale("linear")
    plt.yscale("linear")
    plt.ylabel("Counts/bin")
    plt.xlabel("PI (eV)")
    plt.grid(alpha=0.8)
    plt.title(title + " TYPE = " + typename[itype_])    
    hist, binedges = np.histogram(pi_filtered, bins=binnum, range=(pimin, pimax))
    bincenters = 0.5 * (binedges[1:] + binedges[:-1])
    ene = bincenters * 0.5 + 0.5  # Converting bin centers to energy
    event_number = len(pi_filtered)
    label = f"{label_suffix} ({event_number}c)"
    plt.errorbar(ene, hist, yerr=np.sqrt(hist), color=color, fmt='-', label=label, alpha=alpha)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=6)
    plt.xlim(emin, emax)
    ofname = f"fig_{typename[itype_]}_pixel{label_suffix}_{outfname}"
    plt.savefig(ofname)
    plt.show()
    plt.close()
    print(f"..... {ofname} is created.")

def fit_histogram(pi_filtered, color, itype_, filename, label_suffix, alpha=0.9, debug = False):

    outfname=f"ql_plotspec_fitMnK_{filename.replace('.evt', '').replace('.gz', '')}.png"    

    hist, binedges = np.histogram(pi_filtered, bins=binnum, range=(pimin, pimax))
    bincenters = 0.5 * (binedges[1:] + binedges[:-1])
    ene = bincenters * 0.5 + 0.5  # Converting bin centers to energy
    event_number = len(pi_filtered)
    label = f"{label_suffix} ({event_number}c)"

    gfwhm = 4
    gw = gfwhm / 2.35
    norm = np.sum(hist)/2
    gain = 1.0001
    bkg1 = 0.0
    bkg2 = 0.0
    init_params=[norm,gw,gain,bkg1,bkg2]
    consts = [1,1,1,1,1,1,1,1]

    model_y = mymodel(ene,init_params,consts)

    plt.figure(figsize=(12,8))

    plt.title("Mn Kalpha fit (initial values)")
    plt.xlabel("Energy (eV)")
    plt.errorbar(ene, hist, yerr=np.sqrt(hist), fmt='ko', label = "data")
    plt.plot(ene, model_y, 'r-', label = "model")
    plt.legend(numpoints=1, frameon=False, loc="upper left")
    plt.grid(linestyle='dotted',alpha=0.5)
    plt.savefig("fit_MnKalpha_init.png")
    if debug:
	    plt.show()

    # do fit
    result, error, chi2, dof = solve_least_squares(ene, hist, np.sqrt(hist), init_params, consts, mymodel)

    # get results 
    norm, norme = np.abs(result[0]), np.abs(error[0])
    gw, gwe = np.abs(result[1]), np.abs(error[1])
    gain, gaine = np.abs(result[2]), np.abs(error[2])
    bkg1, bkg1e = np.abs(result[3]), np.abs(error[3])
    bkg2, bkg2e = np.abs(result[4]), np.abs(error[4])
    fwhm, fwhme = 2.35 * gw, 2.35 * gwe

    label1 = "N = " + str("%4.2f(+/-%4.2f)" % (norm,norme)) + " g = " + str("%4.5f(+/-%4.5f)" % (gain,gaine)) + " dE = " + str("%4.2f(+/-%4.2f)" % (fwhm,fwhme) + " (FWHM)")
    label2 = "chi/dof = " + str("%4.2f"%chi2) + "/" + str(dof) + " = " + str("%4.2f"%  (chi2/dof))
    label3 = "bkg1 = " + str("%4.2f(+/-%4.2f)" % (bkg1,bkg1e)) + " bkg2 = " + str("%4.2f(+/-%4.2f)" % (bkg2,bkg2e))

    strlog = "N," + str("%4.4f,%4.4f" % (norm,norme)) + ",g," + str("%4.5f,%4.5f" % (gain,gaine)) + ",dE," + str("%4.4f,%4.4f" % (fwhm,fwhme) + ",") \
            + ",chidof," + str("%4.4f"%chi2) + "," + str(dof) \
            + ",bkg1," + str("%4.4f,%4.4f)" % (bkg1,bkg1e)) + ",bkg2," + str("%4.4f,%4.4f" % (bkg2,bkg2e))

    print(label1, label2, label3)
    print(strlog)

    fitmodel = mymodel(ene,result,consts)
    plt.figure(figsize=(10,7))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    plt.figtext(0.05,0.03, f"fname={filename}")
    plt.title(f"pixel={label_suffix} Type={typename[itype_]} : fit MnKa " + "\n" + label1 + "\n" + label2 + ", " + label3)
    plt.errorbar(ene, hist, yerr=np.sqrt(hist), fmt='ko', label = "data", ms=2)
    plt.plot(ene, fitmodel, 'r-', label = "model")
    background = bkg1 * np.ones(len(ene)) + (ene - np.mean(ene)) * bkg2
    plt.plot(ene, background, 'b-', label = "background", alpha = 0.9, lw = 1)

    eye = np.eye(len(consts))
    for i, oneeye in enumerate(eye):
        plt.plot(ene, mymodel(ene,result,consts=oneeye), alpha = 0.7, lw = 1, linestyle="--", label = str(i+1))
    plt.grid(linestyle='dotted',alpha=0.5)
    plt.legend(numpoints=1, frameon=False, loc="upper left")
        
    ax2 = plt.subplot2grid((3,1), (2,0))    
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel(r'Energy (eV)')
    plt.ylabel(r'Resisual')
    resi = fitmodel - hist 
    plt.errorbar(ene, resi, yerr = np.sqrt(hist), fmt='ko', ms=2)
    plt.grid(linestyle='dotted',alpha=0.5)    

    ofname = f"fig_{typename[itype_]}_{label_suffix}_{outfname}"
    plt.savefig(ofname)
    if debug: 
        plt.show()

    return fwhm, fwhme, gain, gaine, strlog

def plot_MnKa(pi, itype, pixel, filename, debug, name, fit=True):
    for itype_ in itypename[:1]: # loop for Hp only
#    for itype_ in itypename[1:2]: # loop for Mp
#    for itype_ in itypename[:2]: # loop for Hp, Mp only 
        csvfilename = "fit_summary_" + typename[itype_] + "_" + str(name) + ".csv"
        csvf = open(csvfilename, 'w')

        xfwhm, xfwhme, xgain, xgaine = [],[],[],[]
        fit_pixels = []

        # Filter data by itype
        typecut = (itype == itype_) & (pi >= pimin) & (pi < pimax) 
        pi_filtered = pi[typecut]

        print("\n ===== all pixels ===== ")
        # Compute and plot histogram for all pixels of current itype
        if fit:
            fwhm, fwhme, gain, gaine, strlog = fit_histogram(pi_filtered, "k", itype_, filename, f"all_{name}", alpha=0.9, debug = debug)
            xfwhm.append(fwhm)
            xfwhme.append(fwhme)
            xgain.append(gain)
            xgaine.append(gaine)
            fit_pixels.append(-1) # -1 means all pixels 
            csvf.write("-1," + strlog+"\n")

        else:
            plot_histogram(pi_filtered, "k", itype_, filename, "all", alpha=0.9)

        # Plot histograms for each pixel
        for pixel_ in np.arange(36):
            print("\n ===== pixel =", pixel_, " =====")
            pixelcut = (pixel == pixel_) & typecut & (pi >= pimin) & (pi < pimax)
            pi_pixel_filtered = pi[pixelcut]
            
            if len(pi_pixel_filtered) == 0:
                print("warning: data is empty for pixel =", pixel_)
                continue

            color = scalarMap.to_rgba(pixel_)
            if fit:
                fwhm, fwhme, gain, gaine, strlog = fit_histogram(pi_pixel_filtered, color, itype_, filename, f"P{pixel_}_{name}", alpha=0.9, debug = debug)
                xfwhm.append(fwhm)
                xfwhme.append(fwhme)
                xgain.append(gain)
                xgaine.append(gaine)
                fit_pixels.append(pixel_)
                csvf.write(str(pixel_) + "," + strlog+"\n")

            else:
                plot_histogram(pi_pixel_filtered, color, itype_, filename, f"P{pixel_}_{name}", alpha=0.9)

        csvf.close()
        plt.figure(figsize=(12,8))
        plt.title("Results of Mn Kalpha fit")

        plt.subplot(211)
        plt.ylabel("Energy resolution (eV)")
        plt.xlabel("Pixels")
        plt.errorbar(fit_pixels, xfwhm, yerr=xfwhme, fmt='ko', label = typename[itype_])
        plt.legend(numpoints=1, frameon=False, loc="upper left")	   
        plt.grid(linestyle='dotted',alpha=0.1)

        plt.subplot(212)
        plt.ylabel("Energy Scale")
        plt.xlabel("Pixels")
        plt.errorbar(fit_pixels, xgain, yerr=xgaine, fmt='ko', label = typename[itype_])
        plt.legend(numpoints=1, frameon=False, loc="upper left")	   
        plt.grid(linestyle='dotted',alpha=0.1)
        plt.savefig("fit_summary_" + typename[itype_] + "_" + str(name) + ".png")
        plt.show()

def main():
    args = parse_arguments()
    # print out all arguments 
    args_dict = vars(args)
    for arg in args_dict:
        print(f"{arg}: {args_dict[arg]}")
    data = open_fits_data(args.filename)
    processed_data, dt, dtime, = process_data(data)
    time, itype, pi, rise_time, deriv_max, pixel = processed_data  # data unpack
    cutid = np.where((time > args.start) & (time < args.end))[0]
    time = time[cutid]
    itype = itype[cutid]
    pi = pi[cutid]
    rise_time = rise_time[cutid]
    deriv_max = deriv_max[cutid]
    pixel = pixel[cutid]
    dtime = dtime[cutid]
    print(f"UPDATED : data from {dtime[0]} --> {dtime[-1]}")
    plot_MnKa(pi, itype, pixel, args.filename, args.debug, args.name)

if __name__ == "__main__":
    main()