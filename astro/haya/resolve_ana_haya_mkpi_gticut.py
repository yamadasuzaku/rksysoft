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


    itypelist = [0, 1, 2, 3, 4]
    typenamelist = ["Hp", "Mp", "Ms", "Lp", "Ls"]


def main():
    parser = argparse.ArgumentParser(description="Process XRISM data files.")
    parser.add_argument('eventfile', help="Path to the input FITS file")    
    parser.add_argument('gtifile', help="Path to the gti file")
    parser.add_argument('--debug', '-d', action='store_true', default=False, help='The debug flag')

    args = parser.parse_args()

    eventfile = args.eventfile
    gtifile = args.gtifile
    debug = args.debug

    # Check if necessary files and commands exist
    check_file_exists(eventfile)
    check_file_exists(gtifile)
    
    check_command_exists("xselect")

    eventfilename = os.path.basename(eventfile).replace(".evt", "").replace(".gz", "")
    gtifilename = os.path.basename(gtifile).replace(".gti", "").replace(".gz", "")

    primary_header = fits.open(eventfile)[0].header
    ra_source = primary_header["RA_NOM"]
    dec_source = primary_header["DEC_NOM"]
    itypelist = [0]
    typenamelist = ["Hp"]
#    itypelist = [0, 1, 2, 3, 4]
#    typenamelist = ["Hp", "Mp", "Ms", "Lp", "Ls"]

    pifile = eventfilename + "_" + gtifilename + ".pi"
    outevtfile = eventfilename + "_" + gtifilename + ".evt"

    subprocess.run(f"rm -f {pifile}", shell=True, check=True)
    subprocess.run(f"rm -f {outevtfile}", shell=True, check=True)

    for n, (itype, typename) in enumerate(zip(itypelist, typenamelist)):
        # xselect commands

        print(color_text(f"Stage 1: Running xselect for {typename}", "blue"))
        xselect_script = f"""#!/bin/sh
rm -rf rsl_source_{typename}_{gtifilename}.lc rsl_source_{typename}_{gtifilename}.img rsl_source_{typename}_{gtifilename}.pha rsl_source_{typename}_{gtifilename}.evt
xselect << EOF
xsel
no
read event
./
{eventfile}
yes

filter time file {gtifile}
show filter

extract all
set image det
extract all
filter column "PIXEL=0:11,13:35"
filter GRADE {itype}
extract all
save all
rsl_source_{typename}_{gtifilename}
extract events
save events
rsl_source_{typename}_{gtifilename}.evt
no
exit
no
EOF
        """
        run_shell_script(xselect_script, f"run_xselect_{typename}_{gtifilename}.sh")
        check_file_exists(f"rsl_source_{typename}_{gtifilename}.evt")

#     print(color_text(f"Stage 1: Running xselect for {args.eventfile}", "blue"))
#     xselect_script = f"""#!/bin/sh
# xselect << EOF
# xsel
# no
# read event
# ./
# {eventfile}
# yes

# filter time file {gtifile}

# show filter

# extract event 
# save event {outevtfile} 

# extract all

# save spec {pifile} 


# no
# exit
# no
# EOF
#         """

#     run_shell_script(xselect_script, f"run_xselect_mkevt.sh")
#     check_file_exists(f"{pifile}")
#     check_file_exists(f"{outevtfile}")
        
       
if __name__ == "__main__":
    main()
