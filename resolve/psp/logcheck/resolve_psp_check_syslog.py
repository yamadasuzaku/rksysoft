#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import datetime
import astropy.io.fits

# MJD reference day 01 Jan 2019 00:00:00
MJD_REFERENCE_DAY = 58484
reference_time = Time(MJD_REFERENCE_DAY, format='mjd')

def process_data(target_fname):
    data = astropy.io.fits.open(target_fname)[14].data
    time, msg, psp_id = data["TIME"], data["MSG"], data["PSP_ID"]
    sortid = np.argsort(time)
    time = time[sortid]
    msg = msg[sortid]
    psp_id = psp_id[sortid]
    print(f"Number of events are {len(time)} in {target_fname}")
    return time, msg, psp_id

def search_in_msg(msg, dtime, psp_id, search_term = "psp_task_wfrb_watch"):
    for onemsg, onedtime, onepsp_id in zip(msg, dtime, psp_id):
        if search_term in onemsg:
            print(onedtime, "PSP_ID=", onepsp_id, onemsg)

def main(target_fname):

    time, msg, psp_id = process_data(target_fname)
    dtime = [reference_time.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in time]
    search_in_msg(msg, dtime, psp_id)

    fname_tag = target_fname.replace(".evt","").replace(".gz","").replace(".hk1","")
    outfname = "pspcheck_" + fname_tag + ".txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some pixel data')
    parser.add_argument('fname', type=str, help='Target fits file name')

    args = parser.parse_args()
    main(args.fname)
