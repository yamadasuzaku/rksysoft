#!/usr/bin/env python

from astropy.io import fits
from astropy.time import Time

import glob
import numpy as np
import datetime

flist = glob.glob("*.hk")
for fname in flist:
    hdul = fits.open(fname)
    hdul_hk_psp_ess = hdul["HK_PSP_ESS"]
    for pspid in ["A0","A1","B0","B1"]:
        key="PSP" + pspid + "_ESS_CPU_ACC_FEMI_ERR_FLG"
        data = hdul_hk_psp_ess.data[key]
        time = hdul_hk_psp_ess.data["TIME"]
        unique_arr = np.unique(data, return_counts=True)
#        dateobs = hdul[0].header["DATE-OBS"]
#        dateend = hdul[0].header["DATE-END"]
#        print("NOT-FOUND", dateobs, dateend, fname, key, unique_arr)                        
        if len(unique_arr[0]) == 2:
            print("DETECTED", fname, key, unique_arr)                        
#            print("DETECTED", dateobs, dateend, fname, key, unique_arr)                        

            cutid = np.where(data>0)[0]
            start = time[cutid[0]]
            stop = time[cutid[-1]]      

#            mjdrefi = hdul_hk_psp_ess.header["MJDREFI"]
#            reftime = Time(mjdrefi, format='mjd')
            tstart = datetime.datetime( 2014, 1, 1, 0, 0, 0) 
            dstart = tstart + datetime.timedelta(seconds=float(start))      
            dstop = tstart + datetime.timedelta(seconds=float(stop))                  
#            dstart = reftime.datetime + datetime.timedelta(seconds=float(start))      
#            dstop = reftime.datetime + datetime.timedelta(seconds=float(stop))                  
            print("...... time = ", start, stop)
            print("...... time = ", dstart, dstop)


    hdul.close()
