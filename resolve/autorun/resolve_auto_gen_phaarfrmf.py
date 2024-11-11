#!/usr/bin/env python

import os
import subprocess
import argparse
from astropy.io import fits
import sys
import numpy as np

# ピクセルマッピング
pixel_fromdetxdety = [
    [12, 11,  9, 19, 21, 23],
    [14, 13, 10, 20, 22, 24],
    [16, 15, 17, 18, 25, 26],
    [ 8,  7,  0, 35, 33, 34],
    [ 6,  4,  2, 28, 31, 32],
    [ 5,  3,  1, 27, 29, 30]
]

itypenamelist = ["Hp", "Mp", "Ms", "Lp", "Ls"]

def generate_region_file(pixels, regfile):
    pixel_list = [int(p) for p in pixels.split(",")]
    box_lines = []

    for one_detx in np.arange(1, 7):
        for one_dety in np.arange(1, 7):
            pixel = pixel_fromdetxdety[one_detx - 1][one_dety - 1]
            if pixel in pixel_list:
                box_lines.append(f"+box({one_detx},{one_dety},1,1)")

    with open(regfile, 'w') as fout:
        fout.write("physical\n")
        fout.write("\n".join(box_lines))
    print(f"Generated: {regfile}")

def generate_region_file_all(filename):
    """region_RSL_det.reg ファイルを自動生成する"""
    content = """physical
+box(4,1,5,1.00000000)
+box(3.5,2,6,1.00000000)
+box(3.5,3,6,1.00000000)
+box(3.5,4,6,1.00000000)
+box(3.5,5,6,1.00000000)
+box(3.5,6,6,1.00000000)
"""
    with open(filename, "w") as f:
        f.write(content)

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

def generate_pixel_string_xselectformat(pixels):
    """
    指定されたピクセル番号を "PIXEL=" 形式で生成する関数。
    同じ数字で "start:end" 形式にし、カンマで区切る。

    Args:
        pixels (str): 含めるピクセル番号のカンマ区切りの文字列 (例: '0,1,2,3,35')。

    Returns:
        str: "PIXEL="形式の文字列 (例: 'PIXEL=0:0,1:1,2:2,3:3,35:35')。
    """
    # ピクセル番号を整数のリストに変換し、ソートする
    included_pixels = sorted(int(p) for p in pixels.split(","))

    # ピクセル番号を "start:end" 形式に変換して結合
    pixel_str = ",".join(f"{p}:{p}" for p in included_pixels)

    return f"PIXEL={pixel_str}"



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
#    os.remove(script_name)

def main():
    parser = argparse.ArgumentParser(description="")

    parser = argparse.ArgumentParser(
      description='Generate Resolve pha, rmf, arf',
      epilog='''
        Example 
       (1) standard   : resolve_auto_gen_phaarfrmf.py -eve xa000114000rsl_p0px1000_cl.evt -ehk xa000114000.ehk -gti xa000114000rsl_px1000_exp.gti
       (2) other      :   
      ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    

    parser.add_argument("--eventfile", "-eve", default="xa300049010rsl_p0px3000_cl.evt", help="Path to the event file")
    parser.add_argument("--ehkfile", "-ehk", default="xa300049010.ehk", help="Path to the EHK file")
    parser.add_argument("--pix_gti_file", "-gti", default="xa300049010rsl_px3000_exp.gti", help="Path to the pixel GTI file")
    parser.add_argument('--itypenames', '-y', type=str, help='カンマ区切りのitypeリスト (e.g., -y 0,1)', default='0')
    parser.add_argument('--clobber', '-c', choices=["yes", "no"], default="no",help='Flag to skip if the output exists (yes or no)')
    parser.add_argument('--gmin', '-g', type=int, help='grppha min value', default=30)
    parser.add_argument('--numphoton', '-np', type=int, help='numphoton for arfgen', default=300000)
    parser.add_argument("--pixels", type=str, default=None, help="Comma-separated list of pixel numbers (e.g., '0,17,18,35').")
    parser.add_argument("--pname", type=str, default="all", help="Short name for the output files (e.g., 'inner').")    

    args = parser.parse_args()
    itypenames = list(map(int, args.itypenames.split(',')))
    gmin = args.gmin
    pixels = args.pixels
    pname = args.pname
    numphoton = args.numphoton

    if pname == "all":
        regfile="region_RSL_det.reg"
        xselect_sel_pixel="PIXEL=0:11,13:35"
        # 自動生成された region ファイルを確認・生成
        if not os.path.exists(regfile):
            print(color_text(f"Generating region file: {regfile}", "green"))
            generate_region_file_all(regfile)
    else:
        if pixels == None:
            print(color_text(f"pixels should be set when pname is not all. ", "red"))
        regfile=f"region_RSL_det_{pname}.reg"
        generate_region_file(pixels, regfile)
        xselect_sel_pixel=generate_pixel_string_xselectformat(pixels)

    # 必要なファイルとコマンドの存在確認
    check_file_exists(args.eventfile)
    check_file_exists(args.ehkfile)
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

    # Apply ftcopy command
    print(color_text("Stage 1: Filtering events using ftcopy", "blue"))
    ftcopy_cmd = f'ftcopy infile="{args.eventfile}[EVENTS][(PI>=600) && ' \
                 f'(((((RISE_TIME+0.00075*DERIV_MAX)>46)&&((RISE_TIME+0.00075*DERIV_MAX)<58))&&ITYPE<4)||(ITYPE==4))&&STATUS[4]==b0]" ' \
                 f'outfile={eventfilename}2.evt copyall=yes clobber=yes history=yes'
    subprocess.run(ftcopy_cmd, shell=True, check=True)

    eventfile = f"{eventfilename}2.evt"
    eventfilename = f"{eventfilename}2"

    for i, itype in enumerate(itypenames):
        typename=itypenamelist[i] + "_" + pname
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

set image det
filter region {regfile}
filter column "{xselect_sel_pixel}"
filter GRADE {itype}
show filter
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
        if args.clobber == "no" and os.path.exists(rmf_file):
            print(color_text(f"Skipping: {rmf_file} already exists.", "yellow"))
        else:        
            rslmkrmf_cmd = f'rslmkrmf infile=rsl_source_{typename}.evt outfileroot=rsl_source_{typename} resolist={itype} ' \
                           f'regmode=DET regionfile={regfile} clobber=yes logfile="rslmkrmf_{typename}.log"'
            subprocess.run(rslmkrmf_cmd, shell=True, check=True)

        # xaexpmap command
        print(color_text(f"Stage 4: Creating exposure map for {typename}", "blue"))
        expomap = "rsl_cl_evt.expo"
        # clobber の判定と処理
        if args.clobber == "no" and os.path.exists(expomap):
            print(color_text(f"Skipping: {expomap} already exists.", "yellow"))
        else:        
            xaexpmap_cmd = f'xaexpmap ehkfile={args.ehkfile} gtifile=rsl_source_{typename}.evt instrume=RESOLVE badimgfile=None ' \
                           f'pixgtifile={args.pix_gti_file} outfile={expomap} outmaptype=EXPOSURE delta=20.0 numphi=1 stopsys=SKY instmap=CALDB ' \
                           f'qefile=CALDB contamifile=CALDB vigfile=CALDB obffile=CALDB fwfile=CALDB gvfile=CALDB maskcalsrc=yes ' \
                           f'fwtype=FILE specmode=MONO specfile=spec.fits specform=FITS evperchan=DEFAULT abund=1 cols=0 covfac=1 clobber=yes ' \
                           f'chatter=1 logfile=xaexpmap.log'
            subprocess.run(xaexpmap_cmd, shell=True, check=True)

        # xaarfgen command
        print(color_text(f"Stage 5: Creating ARF file for {typename}", "blue"))
        arf_file = f"rsl_source_{typename}.arf"
        if args.clobber == "no" and os.path.exists(arf_file):
            print(color_text(f"Skipping: {arf_file} already exists.", "yellow"))
        else:        
            xrtevtfile="raytrace_pointsource.fits" # this should be used for any types, any pixels. 
            if os.path.exists(xrtevtfile):
                print(color_text(f"Skip raytracing: {xrtevtfile} is used instead of creating a new one.", "yellow"))
            else:
                print(color_text(f"Do raytracing: {xrtevtfile} is created", "yellow"))                                
            xaarfgen_cmd = f'xaarfgen xrtevtfile={xrtevtfile} source_ra={ra_source} source_dec={dec_source} ' \
                           f'telescop=XRISM instrume=RESOLVE emapfile={expomap} regmode=DET regionfile={regfile} sourcetype=POINT ' \
                           f'rmffile={rmf_file} erange="0.3 18.0 0 0" outfile={arf_file} numphoton={numphoton} minphoton=100 teldeffile=CALDB ' \
                           f'qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB ' \
                           f'mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB ' \
                           f'mode=h clobber=yes seed=7 imgfile=NONE logfile="xaarfgen_{typename}.log"'
            subprocess.run(xaarfgen_cmd, shell=True, check=True)

        # Update header keywords
        print(color_text(f"Stage 6: Updating header keywords for {typename}", "blue"))
        subprocess.run(f"fparkey {rmf_file} rsl_source_{typename}.pha RESPFILE", shell=True, check=True)
        subprocess.run(f"fparkey {arf_file} rsl_source_{typename}.pha ANCRFILE", shell=True, check=True)
#        subprocess.run(f"mv rsl_source_{typename}.pha rsl_source_{typename}_v0.pha", shell=True, check=True)

        # grppha commands
        print(color_text(f"Stage 7: Running grppha for {typename}", "blue"))
        grppha_script = f"""
        #!/bin/sh
        rm -rf rsl_source_{typename}_gmin{gmin}.pha
        grppha <<EOF
        rsl_source_{typename}.pha
        rsl_source_{typename}_gmin{gmin}.pha
        group min {gmin}
        exit
        EOF
        """
        run_shell_script(grppha_script, f"run_grppha_{typename}.sh")

        ftgrouppha_cmd = f'ftgrouppha infile=rsl_source_{typename}.pha outfile=rsl_source_{typename}_gopt.pha ' \
                         f'grouptype=opt respfile={rmf_file} clobber=True'
        print(f"run : {ftgrouppha_cmd}")
        subprocess.run(ftgrouppha_cmd, shell=True, check=True)

        check_command_exists("resolve_spec_rebin_30_10.sh")
        subprocess.run(f"resolve_spec_rebin_30_10.sh rsl_source_{typename}.pha rsl_source_{typename}_r3010.pha", shell=True, check=True)


if __name__ == "__main__":
    main()