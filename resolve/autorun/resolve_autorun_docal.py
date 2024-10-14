#!/usr/bin/env python

import os
import subprocess
import shutil
import argparse
import time

topdir = os.getcwd()

def check_program_in_path(program_name):
    # $PATH内でプログラムを探す
    program_path = shutil.which(program_name)
    
    # 見つからない場合はエラーメッセージを表示して終了
    if program_path is None:
        print(f"Error: {program_name} not found in $PATH.")
        sys.exit(1)
    else:
        print(f"{program_name} is found at {program_path}")

# コマンドライン引数を解析する関数
def parse_args():
    """
    コマンドライン引数を解析する。
    """
    parser = argparse.ArgumentParser(description='',
                                     usage='python resolve_ana_pixel_mklc_branch.py f.list -y 0 -p 0 -g')

    parser = argparse.ArgumentParser(
      description='Resolve calibration run. ',
      epilog='''
        Example 1) Be filter:
        
      ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument('obsid', help='OBSID')
    parser.add_argument('--timebinsize', '-t', type=float, help='光度曲線の時間ビンサイズ', default=100.0)
    parser.add_argument('--itypenames', '-y', type=str, help='カンマ区切りのitypeリスト', default='0,1,2,3,4')
    parser.add_argument('--plotpixels', '-p', type=str, help='プロットするピクセルのカンマ区切りリスト', default=','.join(map(str, range(36))))
    parser.add_argument('--output', '-o', type=str, help='出力ファイル名のプレフィックス', default='mklc')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード')
    parser.add_argument('--show', '-s', action='store_true', help='plt.show()を実行するかどうか。defaultはplotしない。')    
    # Define the fwe option with choices OPEN or ND
    parser.add_argument('--fwe', choices=['OPEN', 'ND'], default="OPEN", help='Choose OPEN for 1000 or ND for 3000')
    args = parser.parse_args()
    
    # Set fwe based on the chosen option
    if args.fwe == 'OPEN':
        fwe_value = 1000
    elif args.fwe == 'ND':
        fwe_value = 3000
    else:
        raise ValueError("Invalid option for fwe.")
    
    # 引数の確認をプリント
    print("----- 設定 -----")
    # Print the command-line arguments to help with debugging
    args_dict = vars(args)
    print("Command-line arguments:")
    for arg, value in args_dict.items():
        print(f"{arg}: {value}")    
    print("-----------------")

    return args, fwe_value

def dojob(obsid, runprog, arguments="cl.evt", fwe=3000, \
          subdir="check_lc", linkfiles=["cl.evt"], timebinsize=100, use_flist=False, gdir=f"000108000/resolve/event_cl/"):
    # Define directory and file names based on obsid and other parameters

    # check if runprog exist in PATH
    check_program_in_path(runprog)
    
    gotodir = os.path.join(topdir, gdir)

    # Change to the processing directory
    os.makedirs(os.path.join(gotodir, subdir), exist_ok=True)
    os.chdir(os.path.join(gotodir, subdir))

    # Create symbolic links for the necessary files
    for fname in linkfiles:
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
        print(f"Executing command: {runprog} {' '.join(arguments.split())}")        
        subprocess.run([runprog] + arguments.split(), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

    os.chdir(topdir)
                
# メイン関数
def main():
    """
    スクリプトを実行するメイン関数。
    """
    args, fwe_value = parse_args()
    obsid = args.obsid
    timebinsize=args.timebinsize
    fwe = args.fwe

    # program list 
    procdic = {"qlmklc":True,"qlmkspec":True,"spec6x6":True}
        
    clname = f"xa{obsid}rsl_p0px{fwe_value}_cl"
    clevt = f"{clname}.evt"    

    if procdic["qlmklc"]:
        runprog="resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py"        
        print(f"[START:{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> {runprog} <<<")        
        if args.show:
            arguments=f"{clevt} --timebinsize {timebinsize} -d"
        else:
            arguments=f"{clevt} --timebinsize {timebinsize}"        
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_qlmklc", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")
        print(f"[END:{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> {runprog} <<<\n")

    if procdic["qlmkspec"]:
        runprog="resolve_ana_pixel_ql_plotspec.py"        
        print(f"[START:{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> {runprog} <<<")        
        if args.show:
            arguments=f"{clevt} --rebin 4 --emin 0 --emax 20000 -d"
        else:
            arguments=f"{clevt} --rebin 4 --emin 0 --emax 20000"        

        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_qlmkspec", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")
        print(f"[END:{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> {runprog} <<<\n")


    if procdic["spec6x6"]:
        runprog="resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py"        
        print(f"[START:{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> {runprog} <<<")        
        if args.show:
            arguments=f"{clevt} -l 2000 -x 10000 -b 20 -c -p"
        else:
            arguments=f"{clevt} -l 2000 -x 10000 -b 20 -c"        

        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_spec6x6", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")
        print(f"[END:{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> {runprog} <<<\n")
        
        
        
if __name__ == "__main__":
    main()
        

    
