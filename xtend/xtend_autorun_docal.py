#!/usr/bin/env python 

import glob
import re
import argparse
import os
import subprocess
import shutil
import argparse
import time
import sys

topdir = os.getcwd()

def write_to_file(filename, content):
    with open(filename, 'w') as f:
        if isinstance(content, list): 
            for one in content:
                f.write(one + '\n')
        else:
            f.write(content + '\n')

def check_program_in_path(program_name):
    program_path = shutil.which(program_name)
    
    if program_path is None:
        print(f"Error: {program_name} not found in $PATH.")
        sys.exit(1)
    else:
        print(f"{program_name} is found at {program_path}")

def strip_gz_extension(filename):
    """
    .gz を除いたファイル名を返す。
    """
    if filename.endswith(".gz"):
        return filename[:-3]
    return filename


def resolve_gz_filename(filepath):
    """
    .gz の有無を確認し、存在するならそのファイルを返す。
    なければ、元のファイルパスを返す。
    """
    if os.path.exists(filepath + ".gz"):
        return filepath + ".gz"
    elif os.path.exists(filepath):
        return filepath
    else:
        raise FileNotFoundError(f"File not found: {filepath} or {filepath}.gz")    

class ConsoleColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def do_pileupcheck(evtfile, runprog='xtend_pileup_check_quick.sh'):

    print(f"[START:{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> {runprog} <<<")        
    # check if runprog exist in PATH
    check_program_in_path(runprog)

    # Run the program with the necessary arguments
    try:
        print(f"Executing command: {runprog} {evtfile}")        
        subprocess.run([runprog,evtfile], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

    print(f"[END:{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> {runprog} <<<\n")

def dojob(obsid, runprog, arguments=None, \
          subdir=None, linkfiles=None, timebinsize=100, use_flist=False, gdir=f"000108000/resolve/event_cl/"):

    print(f"[START:{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> {runprog} <<<")        
    # Define directory and file names based on obsid and other parameters

    # check if runprog exist in PATH
    check_program_in_path(runprog)
    
    gotodir = os.path.join(topdir, gdir)
    # ディレクトリが存在するか確認
    if not os.path.exists(gotodir):
        print(f"Error: The directory '{gotodir}' does not exist.", file=sys.stderr)
        sys.exit(1)  # エラーステータス1で終了
    print(f"'{gotodir}' exists. Proceeding with the script.")

    if subdir == None:
        os.chdir(gotodir)
        pass
    else:
        # Change to the processing directory
        os.makedirs(os.path.join(gotodir, subdir), exist_ok=True)
        os.chdir(os.path.join(gotodir, subdir))

    # Create symbolic links for the necessary files
    if linkfiles == None:
        pass
    else:
        for fname in linkfiles:

            link_fname = os.path.basename(fname)
            # 既存のリンクがある場合は削除
            if os.path.islink(link_fname):
                os.remove(link_fname)
                print(f"Removed existing link: {link_fname}")

            try:
                os.symlink(fname, os.path.basename(fname))
            except FileExistsError:
                print(f"Link {fname} already exists, skipping.")

    # Optionally, create a file list
    if use_flist:
        with open("f.list", "w") as flist:
            flist.write(clevt + "\n")

    # Run the program with the necessary arguments
    try:
        # Print the command that will be executed
        if arguments == None:
            print(f"Executing command: {runprog}")        
            subprocess.run([runprog], check=True)
        else:            
            print(f"Executing command: {runprog} {' '.join(arguments.split())}")        
            subprocess.run([runprog] + arguments.split(), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

    os.chdir(topdir)

    print(f"[END:{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> {runprog} <<<\n")


def color_print(message, color):
    """Print a message with the specified color."""
    print(f"{color}{message}{ConsoleColors.ENDC}")

def extract_data_class_and_ccd(filename):
    """
    Extract data class and determine included CCDs from a given filename.

    Parameters:
        filename (str): Event file name (e.g., xa300046010xtd_p0300000a0_cl.evt)

    Returns:
        tuple: (dataClass, included_ccds)
            dataClass (str): Extracted data class (e.g., '300000a0')
            included_ccds (str): CCD information
                'CCD1,2,3,4' : 2nd digit is '0'
                'CCD1,2'     : 2nd digit is '1'
                'CCD3,4'     : 2nd digit is '2'
                'Unknown'    : Otherwise
    """
    match = re.search(r'_p.([A-Za-z0-9]{8})', filename)
    if not match:
        color_print(f"Invalid filename format: {filename}", ConsoleColors.FAIL)
        return None, None

    data_class = match.group(1)
    second_digit = data_class[1]

    ccd_mapping = {
        '0': 'CCD1,2,3,4',
        '1': 'CCD1,2',
        '2': 'CCD3,4'
    }
    included_ccds = ccd_mapping.get(second_digit, 'Unknown')

    return data_class, included_ccds

def get_sorted_pha_files(gotodir, nametag="gopt"):
    # 指定ディレクトリのパス
    # glob でファイルを取得
    files = glob.glob(os.path.join(gotodir, f"*_{nametag}.pha"))

    # ファイルが見つからない場合、エラーを吐く
    if not files:
        raise FileNotFoundError(f"No matching files found in directory: {gotodir}")

    # 生成日時でソート
    sorted_files = sorted(files, key=os.path.getctime)
    file_names = [os.path.basename(file) for file in sorted_files]
    return file_names

# メイン関数
def main():    
    parser = argparse.ArgumentParser(description="Process event files.")
    parser.add_argument('obsid', help='OBSID')    
    parser.add_argument(
        '--steps', '-s', 
        nargs='*', 
        type=int, 
        choices=[1, 2, 3, 4], 
        default=[1, 2, 3, 4],
        help="Specify which steps to execute (1: Check pileup, 2: Create region, 3: Create PHA, RMF, ARF), Step 4: Check PHA, RMF, ARF. Default is all steps."
    )
    parser.add_argument('-n', '--numphoton', type=int, help='number of photon for arfgen', default=1000000)    
    parser.add_argument('--genhtml', '-html', action='store_false', help='stop generate html')
    parser.add_argument('--clobber', '-c', choices=["yes", "no"], default="no",help='Flag to skip if the output exists (yes or no)')    

    args = parser.parse_args()
    obsid = args.obsid
    steps = args.steps 
    genhtml = args.genhtml

    gdir=f"{obsid}/xtend/event_cl/"
    filenames = glob.glob(f"{gdir}/xa{obsid}*_cl.evt*")    
    if not filenames:
        color_print(f"No .evt files found. in {filenames}", ConsoleColors.WARNING)
        return

    for filename in filenames:
        clevt = os.path.basename(filename)
        ehk = os.path.basename(resolve_gz_filename(os.path.join(f"{obsid}/auxil", f"xa{obsid}.ehk")))
        bimg_unzip_name = strip_gz_extension(clevt).replace("_cl.evt", ".bimg")
        bimg = os.path.basename(resolve_gz_filename(os.path.join(f"{obsid}/xtend/event_uf", f"{bimg_unzip_name}")))
        
        print(f"Output file will be: {bimg}")

        data_class, ccd_info = extract_data_class_and_ccd(clevt)

        if data_class is None or ccd_info is None:
            continue

        color_print(f"Processing file: {filename}", ConsoleColors.OKCYAN)
        color_print(f"  Data Class: {data_class}", ConsoleColors.OKGREEN)
        color_print(f"  Included CCDs: {ccd_info}", ConsoleColors.OKGREEN)

        if ccd_info in {'CCD1,2,3,4', 'CCD1,2'}:
            color_print("  Performing analysis...", ConsoleColors.OKBLUE)
            if 1 in steps:
                color_print("    Step 1: Check pileup", ConsoleColors.OKCYAN)
                runprog="xtend_pileup_check_quick.sh"        
                arguments=f"{clevt}"
                dojob(obsid, runprog, arguments = arguments, subdir="checkpileup_std", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/xtend/event_cl/")        
            if 2 in steps:
                color_print("    Step 2: Create region", ConsoleColors.OKCYAN)                
                runprog="xtend_util_genregion.py"        
                arguments=f"{clevt}"
                dojob(obsid, runprog, arguments = arguments, subdir="checkpileup_std", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/xtend/event_cl/")        

            if 3 in steps:
                color_print("    Step 3: Create PHA, RMF, ARF", ConsoleColors.OKCYAN)
                runprog="xtend_auto_gen_phaarfrmf.py"        
                arguments=f"{clevt} -e {ehk} -b {bimg} -n {args.numphoton} -c {args.clobber}"
                dojob(obsid, runprog, arguments = arguments, subdir="checkpileup_std", linkfiles=[f"../{clevt}",f"../../../auxil/{ehk}",f"../../event_uf/{bimg}"], gdir=f"{obsid}/xtend/event_cl/")        

            if 4 in steps:
                color_print("    Step 4: Check PHA, RMF, ARF", ConsoleColors.OKCYAN)
                runprog="xrism_util_plot_arf.py"        
                arguments=""
                dojob(obsid, runprog, arguments = arguments, subdir="checkpileup_std", gdir=f"{obsid}/xtend/event_cl/")        

                # plot all, inner, and outer spec 
                runprog="xrism_spec_qlfit_many.py"        
                gotodir = f"{obsid}/xtend/event_cl/checkpileup_std"
                arguments=f"f_gopt.list --fname {obsid}_xtend_comp_all_in_out" 
                write_to_file(f"{gotodir}/f_gopt.list", get_sorted_pha_files(gotodir))
                dojob(obsid, runprog, arguments = arguments, gdir=gotodir)        

                arguments=f"f_gopt.list --fname {obsid}_xtend_comp_all_in_out_narrow --emin 6.0 --emax 7.5 --xscale off --progflags 1,1,1" 
                dojob(obsid, runprog, arguments = arguments, gdir=gotodir)        

        else:
            color_print("  Skipping analysis.", ConsoleColors.WARNING)

    ################### create HTML ###################################################################

    if genhtml:
        print("..... create html file")
        runprog="xrism_autorun_png2html.py"                
        check_program_in_path(runprog)
        subprocess.run([runprog] + [obsid] + ["--keyword"] + ["checkpileup_"] + ["--ver"] +  ["v0"], check=True)

if __name__ == "__main__":
    main()
