#!/usr/bin/env python 

import argparse
import subprocess
import sys
import os

def e2ch(ene):
    """Convert energy to channel."""
    return int(ene/6.0)

def run_xselect_commands(evt_file, pi_filter, grade_filter, gti_filter, key):
    print(key, " : ", pi_filter, " & ", grade_filter, " & ", gti_filter)
    outfile = f"{evt_file.rsplit('.', 1)[0]}_{key}.img"
    logfile = f"{evt_file.rsplit('.', 1)[0]}_{key}.log"

    if os.path.exists(outfile):
        os.remove(outfile)
    if os.path.exists(logfile):
        os.remove(logfile)

    xselect_commands = f"""
no

read event {evt_file} ./
yes

PI_FILTER
GRADE_FILTER
GTI_FILTER

extract image

save image {outfile}
exit
no
    """
    
	# update xselect commands     
    xselect_commands = xselect_commands.replace("PI_FILTER",pi_filter).replace("GRADE_FILTER",grade_filter).replace("GTI_FILTER",gti_filter)  

    try:
        with open(logfile, "w") as log_file:
            process = subprocess.run(
                ["xselect"],
                input=xselect_commands,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            print(process.stdout)
            log_file.write(process.stdout)
        print(f"Output written to {outfile}")
    except subprocess.CalledProcessError as e:
        print(f"Error during xselect execution: {e}")

# Argument parser setup

parser = argparse.ArgumentParser(
  description='This program is used to perform selection on an event file using the xselect command.',
  epilog='''
Example:
  python xtend_pileup_genevt_v0.py xa300041010xtd_p0300000a0_uf.evt xa300041010xtd_p0300000a0_cl.evt xa300041010xtd_p0300000a0_cl.gti
  ''',
  formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("ufevt", help="Path to the unfiltered event file.")
parser.add_argument("clevt", help="Path to the cleaned event file.")
parser.add_argument("clgti", help="Path to the GTI file of the cleaned event file.")
args = parser.parse_args()
ufevt = args.ufevt
clevt = args.clevt
clgti = args.clgti

# Energy cuts
e1 = e2ch(500.)   # 0.5 keV
e2 = e2ch(3000.)   # 3   keV
e3 = e2ch(7000.)  # 7   keV
e4 = e2ch(10000.)  # 10  keV
pi_filters = {
    "ene1": f"filter pha_cutoff {e1} {e2}",
    "ene2": f"filter pha_cutoff {e2} {e3}",
    "ene3": f"filter pha_cutoff {e3} {e4}",
    "enea": f"filter pha_cutoff {e1} {e4}"
}

# Grade cuts
grade_filters = {
    "grade0": 'filter grade "0"',
    "grade1": 'filter grade "1"',
    "gradeg": 'filter grade "0,2-4,6"'
}

# GTI cut
gti_filters = {
    "clgti": f"filter time file {clgti}" 
}

# Generate and apply filters
for k1, pi_filter in pi_filters.items():
    for k2, grade_filter in grade_filters.items():
        for k3, gti_filter in gti_filters.items():
            key = f"{k1}_{k2}_{k3}"
            run_xselect_commands(ufevt, pi_filter, grade_filter, gti_filter, key)
