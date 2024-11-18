#!/usr/bin/env python 
import os
import sys
import subprocess
import argparse
from astropy.io import fits
import numpy as np
import shutil

# 色付きの標準出力用
def print_status(message, status_type="info"):
    color_codes = {
        "info": "\033[94m",  # Blue
        "success": "\033[92m",  # Green
        "warning": "\033[93m",  # Yellow
        "error": "\033[91m"  # Red
    }
    reset_code = "\033[0m"
    print(f"{color_codes.get(status_type, '')}{message}{reset_code}")

# ファイル存在確認
def check_file_existance(file):
    if not os.path.isfile(file):
        print_status(f'ERROR: File "{file}" is not found', "error")
        sys.exit()
    else:
        print_status(f'File "{file}" exists', "success")

# PATH確認
def check_command_in_path(command):
    if shutil.which(command) is None:
        print_status(f'ERROR: Command "{command}" is not in PATH', "error")
        sys.exit()
    else:
        print_status(f'Command "{command}" is in PATH', "success")

# Main処理
def main(args):
    check_file_existance(args.eventfile)
    check_file_existance(args.ehkfile)
    check_file_existance(args.bad_pixel_image_file)
    if args.flickering_pixel_file != "NONE":
        check_file_existance(args.flickering_pixel_file)
    # 初期化
    if len(args.regfiles) == 0:
        print("ERROR : no region files. ", args.regfiles)
    for regfile in args.regfiles:
        check_file_existance(file=regfile)

    check_command_in_path("xselect")
    check_command_in_path("xtdrmf")
    check_command_in_path("xaexpmap")
    check_command_in_path("xaarfgen")
    check_command_in_path("fparkey")
    check_command_in_path("ftgrouppha")

    primary_header = fits.open(args.eventfile)[0].header
    ra_sorce = primary_header["RA_NOM"]
    dec_sorce = primary_header["DEC_NOM"]

    # delete old files if any
    for regfile in args.regfiles:
        fitsfname = os.path.basename(regfile).replace(".reg", "")        
        subprocess.run(f"rm -rf {fitsfname}.img {fitsfname}.pha {fitsfname}.lc {fitsfname}.evt", shell=True)

    # xselectで書くregionのimg、lc、phaを作成
    with open("run_extract.sh", "w") as fout:
        fout.write(f"""#!/bin/sh
rm -rf xsel*
xselect << EOF
xsel
read ev
./
! read event file
{args.eventfile}
yes
extract all
! then apply region filters and save all events. 
""")
        for regfile in args.regfiles:
            fitsfname = os.path.basename(regfile).replace(".reg", "")
            fout.write(f"""filter region {regfile}
extract all
save all
{fitsfname}
extract events
save events
{fitsfname}.evt
no
clear region
clear events
extract all
""")
        fout.write("""exit
no
EOF
""")
    
    subprocess.run("bash run_extract.sh", shell=True)
    subprocess.run("rm -rf run_extract.sh", shell=True)

    # xtdrmfによるrmf fileの作成
    print_status("===== Run xtdrmf =====", "info")
    rmf_file = "xtd_source.rmf"
    pha_file = f"{fitsfname}.pha"
    if args.clobber == "no" and os.path.exists(f"{rmf_file}"):
        print_status(f"Skipping: {rmf_file} already exists.","info")
    else:        
        check_file_existance(file=pha_file)
        subprocess.run(f'xtdrmf infile={pha_file} outfile={rmf_file} clobber=yes rmfparam=$CALDB/data/xrism/xtend/bcf/response/xa_xtd_rmfparam_20190101v004.fits', shell=True)
        check_file_existance(file=rmf_file)

    # xaexpmapによるexposure mapの作成
    print_status("===== Run xaexpmap =====", "info")
    expomap = "xtd_cl_evt.expo"
    subprocess.run(f'xaexpmap ehkfile={args.ehkfile} gtifile={pha_file} instrume=XTEND badimgfile={args.bad_pixel_image_file} pixgtifile={args.flickering_pixel_file} outfile={expomap} outmaptype=EXPOSURE delta=20.0 numphi=1 clobber=yes logfile=xtd_xaexpmap.log', shell=True)
    check_file_existance(file=expomap)

    # xaarfgenによる各region fileのarf fileの作成
    for regfile in args.regfiles:
        fitsfname = os.path.basename(regfile).replace(".reg", "")
        print_status(f"===== Run xaarfgen for {fitsfname} =====", "info")
        arf_file = f"{fitsfname}.arf"
        subprocess.run(f'xaarfgen xrtevtfile=xtd_raytrace_ptsrc.fits source_ra={ra_sorce} source_dec={dec_sorce} telescop=XRISM instrume=XTEND emapfile={expomap} regmode=RADEC regionfile={regfile} sourcetype=POINT rmffile={rmf_file} erange="0.3 18.0 0 0" outfile={arf_file} numphoton=1000000 minphoton=100 teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB gatevalvefile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB imgfile=NONE seed=7 clobber=yes logfile="xaarfgen_{fitsfname}.log"', shell=True)
        check_file_existance(file=arf_file)

        # header keywordの変更
        subprocess.run(f"fparkey {rmf_file} {fitsfname}.pha RESPFILE", shell=True)
        subprocess.run(f"fparkey {arf_file} {fitsfname}.pha ANCRFILE", shell=True)

        ftgrouppha_cmd = f'ftgrouppha infile={fitsfname}.pha outfile={fitsfname}_gopt.pha ' \
                         f'grouptype=opt respfile={rmf_file} clobber=True'
        print(f"run : {ftgrouppha_cmd}")
        subprocess.run(ftgrouppha_cmd, shell=True, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process region files and generate related data.")
    parser.add_argument('eventfile', help='Input FITS file (e.g., a300049010xtd_p031300010_cl.evt)')    
    parser.add_argument("--ehkfile", "-e", default="xa300049010.ehk", help="EHK file")
    parser.add_argument('--clobber', '-c', choices=["yes", "no"], default="no",help='Flag to skip if the output exists (yes or no)')    
    parser.add_argument("--bad_pixel_image_file", "-b", default="xa300049010xtd_p031300010.bimg", help="Bad pixel image file")
    parser.add_argument("--regfiles", nargs='+', default=[f"./regions/{f}" for f in os.listdir("./regions") if f.endswith(".reg")], help="List of region files to process")
    parser.add_argument("--flickering_pixel_file", default="NONE", help="Flickering pixel file")

    args = parser.parse_args()
    main(args)

