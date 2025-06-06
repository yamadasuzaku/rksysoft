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
    parser.add_argument("--eventfile", default="xa300049010rsl_p0px3000_cl.evt", help="Path to the event file")
    parser.add_argument("--ehkfile", default="xa300049010.ehk", help="Path to the EHK file")
    parser.add_argument("--regfile", default="region_RSL_det.reg", help="Path to the region file")
    parser.add_argument("--pix_gti_file", default="xa300049010rsl_px3000_exp.gti", help="Path to the pixel GTI file")
    parser.add_argument("--bad_pixel_image_file", default="NONE", help="Path to the bad pixel image file")
    parser.add_argument("--flickering_pixel_file", default="NONE", help="Path to the flickering pixel file")
    parser.add_argument('--clobber', '-c', default="yes", help='Flag to skip when the output already exist (yes or no)')

    args = parser.parse_args()
    clobber=args.clobber

    # Check if necessary files and commands exist
    check_file_exists(args.eventfile)
    check_file_exists(args.ehkfile)
    check_file_exists(args.regfile)
    check_file_exists(args.pix_gti_file)
    check_command_exists("ftcopy")
    check_command_exists("xselect")
    check_command_exists("rslmkrmf")
    check_command_exists("xaexpmap")
    check_command_exists("xaarfgen")
    check_command_exists("fparkey")
    check_command_exists("grppha")

    eventfilename = os.path.basename(args.eventfile).replace(".evt", "").replace(".gz", "")

    primary_header = fits.open(args.eventfile)[0].header
    ra_source = primary_header["RA_NOM"]
    dec_source = primary_header["DEC_NOM"]

    itypelist = [0, 1, 2, 3, 4]
    typenamelist = ["Hp", "Mp", "Ms", "Lp", "Ls"]

    # Apply ftcopy command
    print(color_text("Stage 1: Filtering events using ftcopy", "blue"))
    ftcopy_cmd = f'ftcopy infile="{args.eventfile}[EVENTS][(PI>=600) && ' \
                 f'(((((RISE_TIME+0.00075*DERIV_MAX)>46)&&((RISE_TIME+0.00075*DERIV_MAX)<58))&&ITYPE<4)||(ITYPE==4))&&STATUS[4]==b0]" ' \
                 f'outfile={eventfilename}2.evt copyall=yes clobber=yes history=yes'
    subprocess.run(ftcopy_cmd, shell=True, check=True)

    eventfile = f"{eventfilename}2.evt"
    eventfilename = f"{eventfilename}2"

    for n, (itype, typename) in enumerate(zip(itypelist, typenamelist)):
        # xselect commands
        print(color_text(f"Stage 2: Running xselect for {typename}", "blue"))
        xselect_script = f"""#!/bin/sh
rm -rf rsl_source_{typename}.lc rsl_source_{typename}.img rsl_source_{typename}.pha rsl_source_{typename}.evt
xselect << EOF
xsel
no
read event
./
{eventfile}
yes
extract all
set image det
extract all
filter region {args.regfile}
filter column "PIXEL=0:11,13:35"
filter GRADE {itype}
extract all
save all
rsl_source_{typename}
extract events
save events
rsl_source_{typename}.evt
no
exit
no
EOF
        """
        run_shell_script(xselect_script, f"run_xselect_{typename}.sh")

        check_file_exists(f"rsl_source_{typename}.evt")
        
        # rslmkrmf command
        print(color_text(f"Stage 3: Creating RMF file for {typename}", "blue"))
        rmf_file = f"rsl_source_{typename}.rmf"
        rslmkrmf_cmd = f'rslmkrmf infile=rsl_source_{typename}.evt outfileroot=rsl_source_{typename} resolist={itypelist[n]} ' \
                       f'regmode=DET regionfile={args.regfile} clobber={clobber} logfile="rslmkrmf_{typename}.log"'
        subprocess.run(rslmkrmf_cmd, shell=True, check=True)

        # xaexpmap command
        print(color_text(f"Stage 4: Creating exposure map for {typename}", "blue"))
        expomap = "rsl_cl_evt.expo"
        xaexpmap_cmd = f'xaexpmap ehkfile={args.ehkfile} gtifile=rsl_source_{typename}.evt instrume=RESOLVE badimgfile=None ' \
                       f'pixgtifile={args.pix_gti_file} outfile={expomap} outmaptype=EXPOSURE delta=20.0 numphi=1 stopsys=SKY instmap=CALDB ' \
                       f'qefile=CALDB contamifile=CALDB vigfile=CALDB obffile=CALDB fwfile=CALDB gvfile=CALDB maskcalsrc=yes ' \
                       f'fwtype=FILE specmode=MONO specfile=spec.fits specform=FITS evperchan=DEFAULT abund=1 cols=0 covfac=1 clobber=yes ' \
                       f'chatter=1 logfile=xaexpmap.log'
        subprocess.run(xaexpmap_cmd, shell=True, check=True)

        # xaarfgen command
        print(color_text(f"Stage 5: Creating ARF file for {typename}", "blue"))
        arf_file = f"rsl_source_{typename}.arf"
        xaarfgen_cmd = f'xaarfgen xrtevtfile=raytrace_ptsrc_{typename}.fits source_ra={ra_source} source_dec={dec_source} ' \
                       f'telescop=XRISM instrume=RESOLVE emapfile={expomap} regmode=DET regionfile={args.regfile} sourcetype=POINT ' \
                       f'rmffile={rmf_file} erange="0.3 18.0 0 0" outfile={arf_file} numphoton=300000 minphoton=100 teldeffile=CALDB ' \
                       f'qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB ' \
                       f'mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB ' \
                       f'mode=h clobber=yes seed=7 imgfile=NONE logfile="xaarfgen_{typename}.log"'
        subprocess.run(xaarfgen_cmd, shell=True, check=True)

        # Update header keywords
        print(color_text(f"Stage 6: Updating header keywords for {typename}", "blue"))
        subprocess.run(f"fparkey {rmf_file} rsl_source_{typename}.pha RESPFILE", shell=True, check=True)
        subprocess.run(f"fparkey {arf_file} rsl_source_{typename}.pha ANCRFILE", shell=True, check=True)
        subprocess.run(f"mv rsl_source_{typename}.pha rsl_source_{typename}_v0.pha", shell=True, check=True)

        # grppha commands
        print(color_text(f"Stage 7: Running grppha for {typename}", "blue"))
        grppha_script = f"""
        #!/bin/sh
        rm -rf rsl_source_{typename}_v1.pha
        grppha <<EOF
        rsl_source_{typename}_v0.pha
        rsl_source_{typename}_v1.pha
        group min 30
        exit
        EOF
        """
        run_shell_script(grppha_script, f"run_grppha_{typename}.sh")
        
if __name__ == "__main__":
    main()
