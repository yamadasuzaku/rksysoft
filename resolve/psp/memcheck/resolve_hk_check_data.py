#!/usr/bin/env python

from astropy.io import fits
from astropy.time import Time

import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt

def str2date_2raw(tstart, str):
    time = tstart + datetime.timedelta(seconds=float(str))
    return time
vstr2date=np.vectorize(str2date_2raw)

flist = glob.glob("xa*rsl_a0.hk1*")
for fname in flist:
    fnametag = fname.split("/")[-1].split(".")[0]
    hdul = fits.open(fname)
    hdul_hk_psp_ess = hdul["HK_PSP_ESS"]
    for pspid in ["A0","A1","B0","B1"]:
        key="PSP" + pspid + "_ESS_CPU_ACC_FEMI_ERR_FLG"
        data = hdul_hk_psp_ess.data[key]
        time = hdul_hk_psp_ess.data["TIME"]
        unique_arr = np.unique(data, return_counts=True)
        dateobs = hdul[0].header["DATE-OBS"]
        dateend = hdul[0].header["DATE-END"]
#        print("NOT-FOUND", dateobs, dateend, fname, key, unique_arr)                        
        if len(unique_arr[0]) == 2:
            print("DETECTED", dateobs, dateend, fname, key, unique_arr)                        
            cutid = np.where(data>0)[0]
            start = time[cutid[0]]
            stop = time[cutid[-1]]      

            mjdrefi = hdul_hk_psp_ess.header["MJDREFI"]
            reftime = Time(mjdrefi, format='mjd')
#            tstart = datetime.datetime( 2014, 1, 1, 0, 0, 0) 
            dstart = reftime.datetime + datetime.timedelta(seconds=float(start))      
            dstop = reftime.datetime + datetime.timedelta(seconds=float(stop))                  
            print("...... time = ", start, stop)
            print("...... time = ", dstart, dstop)

            # check contents 

            F = plt.figure(figsize=(14,7))
            ax = plt.subplot(1,1,1)
            plt.title(str(fname) + " ERR : " + key + " at " + dstart.strftime("%Y-%m-%dT%H:%M:%S"))

            time_margin = 30 # sec
            timecutid = np.where( ((start - time_margin) < time) &  ((stop + time_margin) > time) )[0]

            for i, pid in enumerate(["A0","A1","B0","B1"]):
                colname = ["_ESS_CPU_ACC_FEMI_ERR_FLG","_ESS_CPU_TO_FEMI_ACC_FLG"]
                for j, col in enumerate(colname): 
                    ckey="PSP" + pid + col
                    print("ckey =", ckey)
                    cutdata = hdul_hk_psp_ess.data[ckey]
                    tcutdata = cutdata[timecutid]
                    tcuttime = time[timecutid]
                    tcutdatetime  = vstr2date(reftime.datetime, tcuttime)

                    plt.xlabel('Time')    
                    plt.grid(True, alpha = 0.4)
                    if j == 0:
                        fmt = "o-"
                    else:
                        fmt = "o"                        
                    plt.errorbar(tcutdatetime, tcutdata + 3*i + 0.1*j, fmt=fmt, alpha=0.8, ms=3, label=ckey)

                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])                    
                    # Put a legend to the right of the current axis
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#                    plt.legend(numpoints=1, frameon=False, bbox_to_anchor=(1.001, 1), loc='upper left', borderaxespad=0)

            plt.savefig("check_data_" + fnametag + "_" + key + ".png")

    hdul.close()
