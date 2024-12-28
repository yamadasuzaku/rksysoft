#!/usr/bin/env python 

import argparse
import subprocess
import sys
import os

def e2ch(ene):
    """Convert energy to channel."""
    return int(ene / 6.0)

def run_xselect_commands(evt_file, pi_filter, key, binsize, identifier):
    print(key, " : ", pi_filter)
    outfile = f"{evt_file.rsplit('.')[0]}_{key}_{identifier}.lc"
    logfile = f"{evt_file.rsplit('.')[0]}_{key}_{identifier}.log"

    if os.path.exists(outfile):
        os.remove(outfile)
    if os.path.exists(logfile):
        os.remove(logfile)

    xselect_commands = f"""
no

read event {evt_file} ./
yes

set binsize {binsize}

PI_FILTER 

show filter 

extract curve offset=no  exposure=0.0

save curve {outfile}
exit
no
    """
    
    # Update xselect commands     
    xselect_commands = xselect_commands.replace("PI_FILTER", pi_filter)

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
  python xtend_genlc.py xtend_cl.evt --timebinsize 64 --identifier test
  ''',
  formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("clevt", help="Path to the cleaned event file.")
parser.add_argument("--timebinsize", "-t", type=int, default=128, help="Time bin size in seconds (default: 128).")
parser.add_argument("--identifier", "-i", type=str, default="default", help="Identifier for output filenames (default: 'default').")

args = parser.parse_args()
clevt = args.clevt
timebinsize = args.timebinsize
identifier = args.identifier

# Energy cuts
e1 = e2ch(500.)   # 0.5 keV
e2 = e2ch(3000.)   # 3   keV
e3 = e2ch(10000.)  # 10  keV
pi_filters = {
    "ene1": f"filter pha_cutoff {e1} {e2}",
    "ene2": f"filter pha_cutoff {e2} {e3}",
    "enea": f"filter pha_cutoff {e1} {e3}"
}

# Generate and apply filters
for k1, pi_filter in pi_filters.items():
    key = f"{k1}"
    run_xselect_commands(clevt, pi_filter, key, timebinsize, identifier)
