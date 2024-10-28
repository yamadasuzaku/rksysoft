#!/usr/bin/env python

import os
import subprocess
import shutil
import argparse
import time
import sys

topdir = os.getcwd()

def write_to_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content + '\n')

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
        Example 
       (1) ND filter   : resolve_autorun_docal.py 300049010 --fwe ND -b 20 -l 2000 -x 9000 --progflags 0,0,1,1
       (2) Open filter : resolve_autorun_docal.py 000109000          -b 20 -l 2000 -x 9000 --progflags 1,0,1,1
      ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument('obsid', help='OBSID')
    # カンマ区切りの数値列を受け取る
    parser.add_argument('--progflags', type=str, help='Comma-separated flags for qlmklc, qlmkspec, spec6x6 (e.g. 0,1,0)')
    parser.add_argument('--calflags', type=str, help='Comma-separated flags for cal operations (e.g. 0,1,0)')    
    parser.add_argument('--anaflags', type=str, help='Comma-separated flags for ana operations (e.g. 1,0,0)')    
    parser.add_argument('--timebinsize', '-t', type=float, help='光度曲線の時間ビンサイズ', default=100.0)
    parser.add_argument('--itypenames', '-y', type=str, help='カンマ区切りのitypeリスト', default='0,1,2,3,4')
    parser.add_argument('--plotpixels', '-p', type=str, help='プロットするピクセルのカンマ区切りリスト', default=','.join(map(str, range(36))))
    parser.add_argument('--output', '-o', type=str, help='出力ファイル名のプレフィックス', default='mklc')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード')
    parser.add_argument('--show', '-s', action='store_true', help='plt.show()を実行するかどうか。defaultはplotしない。')    
    parser.add_argument('--bin_width', '-b', type=float, default=4, help='Bin width for histogram')
    parser.add_argument('--ene_min', '-l', type=float, default=6300, help='Minimum energy')
    parser.add_argument('--ene_max', '-x', type=float, default=6900, help='Maximum energy')
    parser.add_argument('--genhtml', '-html', action='store_false', help='stop generate html')
    parser.add_argument('--gmin', type=int, help='grppha min value', default=30)

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

    # Change to the processing directory
    os.makedirs(os.path.join(gotodir, subdir), exist_ok=True)
    os.chdir(os.path.join(gotodir, subdir))



    # Create symbolic links for the necessary files
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
        print(f"Executing command: {runprog} {' '.join(arguments.split())}")        
        subprocess.run([runprog] + arguments.split(), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

    os.chdir(topdir)

    print(f"[END:{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> {runprog} <<<\n")

# メイン関数
def main():

################### setting for arguments ###################################################################
    args, fwe_value = parse_args()

    progflags = args.progflags
    calflags = args.calflags
    anaflags = args.anaflags

    obsid = args.obsid
    timebinsize=args.timebinsize
    fwe = args.fwe

    bin_width = int(args.bin_width)
    ene_min = args.ene_min
    ene_max = args.ene_max
    genhtml = args.genhtml
    itypenames = list(map(int, args.itypenames.split(',')))
    plotpixels = list(map(int, args.plotpixels.split(',')))
    gmin = args.gmin    

    # カンマで分割して、数値に変換
    # ユーザーの入力をパースし、整数に変換
    progflags = progflags or ""
    flag_values = [int(x) for x in progflags.split(',')]
    # 12個未満の場合は、0 で埋める
    if len(flag_values) < 12:
        flag_values += [0] * (12 - len(flag_values))    

    calflags = calflags or ""
    cal_values = [int(x) for x in calflags.split(',')]
    # 3個未満の場合は、0 で埋める
    if len(flag_values) < 3:
        flag_values += [0] * (3 - len(flag_values))    

    anaflags = anaflags or ""
    ana_values = [int(x) for x in anaflags.split(',')]
    # 1個未満の場合は、0 で埋める
    if len(ana_values) < 1:
        ana_values += [0] * (1 - len(flag_values))    

    # 数値列をTrue/Falseに変換し、flagが1の時だけ実行
    procdic = {
        "qlmklc": bool(flag_values[0]),
        "qlmkspec": bool(flag_values[1]),
        "spec6x6": bool(flag_values[2]),
        "deltat": bool(flag_values[3]),
        "deltat-rt-pha": bool(flag_values[4]),
        "detxdety": bool(flag_values[5]),
        "temptrend": bool(flag_values[6]),
        "plotghf": bool(flag_values[7]),        
        "plotgti": bool(flag_values[8]),        
        "spec-eachgti": bool(flag_values[9]),                
        "lc-eachgti": bool(flag_values[10]),                        
        "mkbratio": bool(flag_values[11]),
    }
    print(f"procdic = {procdic}")    

    caldic = {
        "lsdist": bool(cal_values[0]),
        "lsdetail": bool(cal_values[1]),
        "specratio6x6": bool(cal_values[2]),        
    }
    print(f"caldic = {caldic}")    

    anadic = {
        "genpharmfarf": bool(ana_values[0]),
    }
    print(f"anadic = {anadic}")    

################### setting for input files ###################################################################

    clname = f"xa{obsid}rsl_p0px{fwe_value}_cl"
    clevt = f"{clname}.evt"    
    ufname = f"xa{obsid}rsl_p0px{fwe_value}_uf"
    ufevt = f"{ufname}.evt"
    rsla0hk1 = f"xa{obsid}rsl_a0.hk1"
    ghf = f"xa{obsid}rsl_000_fe55.ghf"
    telgti = f"xa{obsid}rsl_tel.gti"
    uf50evt = f"xa{obsid}rsl_p0px5000_uf.evt"
    cl50evt = f"xa{obsid}rsl_p0px5000_cl.evt"
    ufacevt = f"xa{obsid}rsl_a0ac_uf.evt"
    elgti= f"xa{obsid}rsl_el.gti"
    expgti = f"xa{obsid}rsl_px{fwe_value}_exp.gti"
    ehk = f"xa{obsid}.ehk"
################### standard process ###################################################################

    if procdic["qlmklc"]:
        runprog="resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py"        
        if args.show:
            arguments=f"{clevt} --timebinsize {timebinsize} -d"
        else:
            arguments=f"{clevt} --timebinsize {timebinsize}"        
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_qlmklc", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")

    if procdic["qlmkspec"]:
        runprog="resolve_ana_pixel_ql_plotspec.py"        
        if args.show:
            arguments=f"{clevt} --rebin {bin_width} --emin {ene_min} --emax {ene_max} -d"
        else:
            arguments=f"{clevt} --rebin {bin_width} --emin {ene_min} --emax {ene_max}"

        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_qlmkspec", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")

        # Fe check
        arguments=f"{clevt} --rebin 4 --emin 6100 --emax 7100"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_qlmkspec", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

    if procdic["spec6x6"]:
        runprog="resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py"        
        if args.show:
            arguments=f"{clevt} -l {ene_min} -x {ene_max} -b {bin_width} -c -p"
        else:
            arguments=f"{clevt} -l {ene_min} -x {ene_max} -b {bin_width} -c"        

        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_spec6x6", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")
        # Fe check
        arguments=f"{clevt} -l 6100 -x 7100 -b 4 -c"        
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_spec6x6", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")
        
    if procdic["deltat"]:
        runprog="resolve_ana_pixel_deltat_distribution.py"        
        arguments=f"{clevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_deltat", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

    if procdic["deltat-rt-pha"]:
        runprog="resolve_ana_pixel_deltat_risetime_distribution.py"        
        arguments=f"{clevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_deltat-rt-pha", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

    if procdic["detxdety"]:
        runprog="resolve_plot_detxdety.py"        
        arguments=f"{clevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_detxdety", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        
        arguments=f"{clevt}  -min 60000 -max 65537"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_detxdety", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        
        arguments=f"{clevt}  -min 0 -max 59999"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_detxdety", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

    if procdic["temptrend"]:
        runprog="resolve_hk_plot_temptrend.sh"
        arguments=f"{rsla0hk1}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_temptrend", linkfiles=[f"../{rsla0hk1}"], gdir=f"{obsid}/resolve/hk/")        

    if procdic["plotghf"]:
        runprog="resolve_ecal_plot_ghf_detail.py"
        arguments=f"{ghf}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_plotghf", linkfiles=[f"../{ghf}"], gdir=f"{obsid}/resolve/event_uf/")        
        runprog="resolve_ecal_plot_ghf_with_FWE.py"
        arguments=f"{ghf} --hk1 {rsla0hk1}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_plotghf", linkfiles=[f"../{ghf}",f"../../hk/{rsla0hk1}"], gdir=f"{obsid}/resolve/event_uf/")        

    if procdic["plotgti"]:
        runprog="resolve_util_gtiplot.py"

        arguments=f"{uf50evt},{ufacevt} -e {uf50evt},{ufacevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_plotgti", linkfiles=[f"../{uf50evt}",f"../{ufacevt}"], gdir=f"{obsid}/resolve/event_uf/")        

        arguments=f"{uf50evt} -e {cl50evt},{uf50evt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_plotgti", linkfiles=[f"../{uf50evt}",f"../../event_cl/{cl50evt}"], gdir=f"{obsid}/resolve/event_uf/")        

        arguments=f"{telgti},{ufevt} -e {clevt},{ufevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_plotgti", linkfiles=[f"../{ufevt}",f"../{telgti}",f"../../event_cl/{clevt}"], gdir=f"{obsid}/resolve/event_uf/")        

        arguments=f"{clevt},{uf50evt} -e {clevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_plotgti", linkfiles=[f"../{uf50evt}",f"../../event_cl/{clevt}"], gdir=f"{obsid}/resolve/event_uf/")        

        arguments=f"{clevt},{elgti} -e {ufacevt} -c r -l -" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_plotgti", linkfiles=[f"../{ufacevt}",f"../{elgti}",f"../../event_cl/{clevt}"], gdir=f"{obsid}/resolve/event_uf/")        

    if procdic["spec-eachgti"]:
        runprog="resolve_ana_pixel_mkspec_eachgti.py"

        arguments=f"{clevt} -i 6000 -x 9000 -y 0 -m 5 -r 5 -t -v 0.002" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_spec-eachgti", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

        arguments=f"{clevt} -i 2000 -x 10000 -y 0 -m 5 -r 10 -t -v 0.002" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_spec-eachgti", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

        arguments=f"{clevt} -i 3000 -x 5000 -y 0 -m 5 -r 10 -t -v 0.002" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_spec-eachgti", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

    if procdic["lc-eachgti"]:
        runprog="resolve_ana_pixel_mklc_branch.py"
        gotodir = f"{obsid}/resolve/event_cl/"
        subdir = "check_lc-eachgti"
        arguments=f"f.list -l -u -y 0 -p 0,17,18,35 -t 256 -o p0_17_18_35" 
        os.makedirs(os.path.join(gotodir, subdir), exist_ok=True)
        write_to_file(f"{gotodir}/{subdir}/f.list", clevt)
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir=subdir, linkfiles=[f"../{clevt}"], gdir=gotodir)        

    if procdic["mkbratio"]:
        runprog="resolve_ana_pixel_mklc_branch.py"
        gotodir = f"{obsid}/resolve/event_cl/"
        subdir = "check_mkbratio"
        arguments=f"f.list -g -u -t 256" 
        os.makedirs(os.path.join(gotodir, subdir), exist_ok=True)
        write_to_file(f"{gotodir}/{subdir}/f.list", clevt)
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir=subdir, linkfiles=[f"../{clevt}"], gdir=gotodir)        

################### calibration ###################################################################

    if caldic["lsdist"]:
        runprog="resolve_ana_run_addprevnext_Lscheck.sh"        
        arguments=f"{ufevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkcal_lsdist", linkfiles=[f"../{ufevt}",f"../../event_cl/{clevt}"], gdir=f"{obsid}/resolve/event_uf/")        

    if caldic["lsdetail"]:
        runprog="resolve_run_ana_pixel_Ls_mksubgroup_using_saturatedflags.sh"        
        ufclgtievt=f"{ufname}_noBL_prevnext_cutclgti.evt"   
        arguments=f"{ufclgtievt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkcal_lsdetail", linkfiles=[f"../{ufclgtievt}"], gdir=f"{obsid}/resolve/event_uf/checkcal_lsdist")        

    if caldic["specratio6x6"]:
        runprog="resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py"        
        arguments=f"{clevt} -r -y 0 -l 2000 -x 12000 -b 250 -c -g"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkcal_specratio6x6", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        


################### analysis ###################################################################

    if anadic["genpharmfarf"]:
        runprog="resolve_auto_gen_phaarfrmf.py"        
        arguments=f"-eve {clevt} -ehk {ehk} -gti {expgti} -g {gmin}" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkana_genpharmfarf", linkfiles=[f"../{clevt}",f"../../../auxil/{ehk}",f"../../event_uf/{expgti}"], gdir=f"{obsid}/resolve/event_cl/")        

################### create HTML ###################################################################

    if genhtml:
        print("..... create html file")
        runprog="resolve_autorun_png2html.py"                
        check_program_in_path(runprog)
        subprocess.run([runprog] + [obsid] + ["--keyword"] + ["check_"] + ["--ver"] +  ["v0"], check=True)
        subprocess.run([runprog] + [obsid] + ["--keyword"] + ["checkcal_"] + ["--ver"] +  ["v0"], check=True)
        subprocess.run([runprog] + [obsid] + ["--keyword"] + ["checkana_"] + ["--ver"] +  ["v0"], check=True)
                        
if __name__ == "__main__":
    main()
        

    
