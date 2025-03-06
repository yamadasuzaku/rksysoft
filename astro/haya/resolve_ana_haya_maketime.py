#!/usr/bin/env python

import os
import subprocess
import argparse
from astropy.io import fits
import sys 

def color_text(text, color):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    return f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(color_text(f"Error: {filepath} does not exist.", "red"))
        sys.exit(1)

def check_command_exists(command):
    if subprocess.call(f"type {command}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        print(color_text(f"Error: {command} is not available in the PATH.", "red"))
        sys.exit(1)

def run_shell_script(script_content, script_name):
    with open(script_name, "w") as script_file:
        script_file.write(script_content)
    os.chmod(script_name, 0o755)
    subprocess.run(f"./{script_name}", shell=True)
    os.remove(script_name)

def main():
    parser = argparse.ArgumentParser(description="Process XRISM data files.")
    parser.add_argument('lcfile', help="Path to the input FITS file (lightcurve)")    
    parser.add_argument('--debug', '-d', action='store_true', default=False, help='The debug flag')
    parser.add_argument('--cut_rate', '-c', type=float, default=50, help='The cut for the count rate')
    parser.add_argument('--sign', '-s', type=str, default=">", help='sign for the cut')

    args = parser.parse_args()

    lcfile = args.lcfile
    debug = args.debug
    cut_rate = args.cut_rate
    sign = args.sign

    # Check if necessary files and commands exist
    check_file_exists(lcfile)
    check_command_exists("xselect")

    lcfilename = os.path.basename(lcfile).replace(".lc", "").replace(".gz", "")

    primary_header = fits.open(lcfile)[0].header
    itypelist = [0, 1, 2, 3, 4]
    typenamelist = ["Hp", "Mp", "Ms", "Lp", "Ls"]
    gtifile = f"{lcfilename}_cut{str(cut_rate)}.gti"

    subprocess.run(f"rm -f {gtifile}", shell=True, check=True)
    subprocess.run(f"maketime {lcfile} {gtifile} \"RATE{sign}{cut_rate}\" anything anything TIME no", shell=True, check=True)
    check_file_exists(gtifile)
    print(f"{gtifile} is created.")

#     print(color_text(f"Stage 1: Running xselect for {args.eventfile}", "blue"))
#     xselect_script = f"""#!/bin/sh
# xselect << EOF
# xsel
# no
# read event
# ./
# {eventfile}
# yes

# set binsize {binsize}

# show filter

# extract all
# save curve {lcfile} 


# no
# exit
# no
# EOF
#         """

#     run_shell_script(xselect_script, f"run_xselect_mklc.sh")
#     check_file_exists(f"{lcfile}")
        
       
if __name__ == "__main__":
    main()
