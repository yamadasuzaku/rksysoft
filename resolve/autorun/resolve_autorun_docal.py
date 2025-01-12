#!/usr/bin/env python

import os
import subprocess
import shutil
import argparse
import time
import sys

# set global variables
topdir = os.getcwd()

# フラグの設定情報を辞書で管理
flag_configs = {
    "progflags": [
        "qlmklc", "qlmkspec", "spec6x6", "deltat", "deltat-rt-pha", "detxdety",
        "temptrend", "plotghf", "plotgti", "spec-eachgti", "lc-eachgti", "mkbratio"
    ],
    "calflags": [
        "lsdist", "lsdetail", "specratio6x6", "statusitype", "statitype", "antico", "fe55fit"
    ],
    "deeplsflags": ["addprevnext", "defcluster"],
    "anaflags": ["genpharmfarf", "qlfit", "compcluf"]
}

def write_to_file(filename, content):
    with open(filename, 'w') as f:
        if isinstance(content, list): 
            for one in content:
                f.write(one + '\n')
        else:
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

def parse_flags(flag_input, keys):
    """フラグ文字列をリストに変換し、不足分を埋める"""
    # 空文字列の場合は空リストを返す
    values = [int(x) for x in flag_input.split(",") if x.strip()] if flag_input else []
    return values + [0] * (len(keys) - len(values))

def generate_flag_dict(flag_values, keys):
    """フラグ値リストを辞書に変換"""
    return {key: bool(value) for key, value in zip(keys, flag_values)}

def get_cordir(base_gdir, is_special_case=False, debug=True):
    """
    Generate the directory path based on whether it is a special case.
    
    Args:
        base_gdir (str): Base directory path.
        is_special_case (bool): Whether the special '_rslgain' suffix is required.
        
    Returns:
        str: Generated directory path.
    """
    generated_dir_path = base_gdir.replace("event_cl","event_cl_rslgain") if is_special_case else f"{base_gdir}"
    if debug: 
        print(f"in get_cordir: generated_dir_path = {generated_dir_path}")
    return generated_dir_path


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
       (1) ND filter (Cyg X-1)  : resolve_autorun_docal.py 300049010 -b 20 -l 2000 -x 9000 --progflags 1,1,1,1,1,1,1,1,1,1,1,1 --calflags 1,1,1,1,1 --anaflags 1,1,1 --fwe ND
       (2) Open filter (Keper)  : 
      ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument('obsid', help='OBSID')
    for flag_name, keys in flag_configs.items():
        example_flags = ",".join(["1"] * len(keys))  # keysの長さに基づいて例を作成
        parser.add_argument(
            f"--{flag_name}",
            type=str,
            help=f"Comma-separated flags for {flag_name} (e.g. --{flag_name} {example_flags})"
        )        
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
    print("----- input arguments -----")
    # Print the command-line arguments to help with debugging
    args_dict = vars(args)
    print("Command-line arguments:")
    for arg, value in args_dict.items():
        print(f"{arg}: {value}")    
    print("-----------------")

    return args, fwe_value

def dojob(obsid, runprog, arguments=None, fwe=3000, \
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

# メイン関数
def main():

################### setting for arguments ###################################################################
    args, fwe_value = parse_args()

    # 各フラグを辞書形式に変換
    flag_dicts = {}
    print(f"*********** (start) flag settting ***************")
    for flag_name, keys in flag_configs.items():
        flag_input = getattr(args, flag_name, "")
        flag_values = parse_flags(flag_input, keys)
        flag_dicts[flag_name] = generate_flag_dict(flag_values, keys)

    # フラグの設定結果の出力
    for name, flag_dict in flag_dicts.items():
        print(f"{name} = {flag_dict}")
    print(f"*********** (end) flag settting *****************")

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


################### setting for input files ###################################################################

    clname = f"xa{obsid}rsl_p0px{fwe_value}_cl"
    clevt = f"{clname}.evt"    
    clgcorevt = f"{clname}gcor.evt"  # so far this is only used for Cyg X-1
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

    if flag_dicts["progflags"]["qlmklc"]:
        runprog="resolve_ana_pixel_ql_mklc_binned_sorted_itype_v1.py"        
        if args.show:
            arguments=f"{clevt} --timebinsize {timebinsize} -d"
        else:
            arguments=f"{clevt} --timebinsize {timebinsize}"        
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_qlmklc", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")

    if flag_dicts["progflags"]["qlmkspec"]:
        runprog="resolve_ana_pixel_ql_plotspec.py"        
        if args.show:
            arguments=f"{clevt} --rebin {bin_width} --emin {ene_min} --emax {ene_max} -d"
        else:
            arguments=f"{clevt} --rebin {bin_width} --emin {ene_min} --emax {ene_max}"

        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_qlmkspec", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")

        # Fe check
        arguments=f"{clevt} --rebin 4 --emin 6100 --emax 7100"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_qlmkspec", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

    if flag_dicts["progflags"]["spec6x6"]:
        runprog="resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py"        
        if args.show:
            arguments=f"{clevt} -l {ene_min} -x {ene_max} -b {bin_width} -c -p"
        else:
            arguments=f"{clevt} -l {ene_min} -x {ene_max} -b {bin_width} -c"        

        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_spec6x6", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")
        # Fe check
        arguments=f"{clevt} -l 6100 -x 7100 -b 4 -c"        
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_spec6x6", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")
        
    if flag_dicts["progflags"]["deltat"]:
        runprog="resolve_ana_pixel_deltat_distribution.py"        
        arguments=f"{clevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_deltat", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

    if flag_dicts["progflags"]["deltat-rt-pha"]:
        runprog="resolve_ana_pixel_deltat_risetime_distribution.py"        
        arguments=f"{clevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_deltat-rt-pha", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

    if flag_dicts["progflags"]["detxdety"]:
        runprog="resolve_plot_detxdety.py"        
        arguments=f"{clevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_detxdety", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        
        arguments=f"{clevt}  -min 60000 -max 65537"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_detxdety", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        
        arguments=f"{clevt}  -min 0 -max 59999"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_detxdety", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

    if flag_dicts["progflags"]["temptrend"]:
        runprog="resolve_hk_plot_temptrend.sh"
        arguments=f"{rsla0hk1}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_temptrend", linkfiles=[f"../{rsla0hk1}"], gdir=f"{obsid}/resolve/hk/")        

    if flag_dicts["progflags"]["plotghf"]:
        runprog="resolve_ecal_plot_ghf_detail.py"
        arguments=f"{ghf}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_plotghf", linkfiles=[f"../{ghf}"], gdir=f"{obsid}/resolve/event_uf/")        
        runprog="resolve_ecal_plot_ghf_with_FWE.py"
        arguments=f"{ghf} --hk1 {rsla0hk1}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_plotghf", linkfiles=[f"../{ghf}",f"../../hk/{rsla0hk1}"], gdir=f"{obsid}/resolve/event_uf/")        

    if flag_dicts["progflags"]["plotgti"]:
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

    if flag_dicts["progflags"]["spec-eachgti"]:
        runprog="resolve_ana_pixel_mkspec_eachgti.py"

        arguments=f"{clevt} -i 6000 -x 9000 -y 0 -m 5 -r 5 -t -v 0.002" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_spec-eachgti", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

        arguments=f"{clevt} -i 2000 -x 10000 -y 0 -m 5 -r 10 -t -v 0.002" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_spec-eachgti", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

        arguments=f"{clevt} -i 3000 -x 5000 -y 0 -m 5 -r 10 -t -v 0.002" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="check_spec-eachgti", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

    if flag_dicts["progflags"]["lc-eachgti"]:
        runprog="resolve_ana_pixel_mklc_branch.py"
        gotodir = f"{obsid}/resolve/event_cl/"
        subdir = "check_lc-eachgti"
        arguments=f"f.list -l -u -y 0 -p 0,17,18,35 -t 256 -o p0_17_18_35" 
        os.makedirs(os.path.join(gotodir, subdir), exist_ok=True)
        write_to_file(f"{gotodir}/{subdir}/f.list", clevt)
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir=subdir, linkfiles=[f"../{clevt}"], gdir=gotodir)        

    if flag_dicts["progflags"]["mkbratio"]:
        runprog="resolve_ana_pixel_mklc_branch.py"
        gotodir = f"{obsid}/resolve/event_cl/"
        subdir = "check_mkbratio"
        arguments=f"f.list -g -u -t 256" 
        os.makedirs(os.path.join(gotodir, subdir), exist_ok=True)
        write_to_file(f"{gotodir}/{subdir}/f.list", clevt)
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir=subdir, linkfiles=[f"../{clevt}"], gdir=gotodir)        

################### calibration ###################################################################

    if flag_dicts["calflags"]["lsdist"]:
        runprog="resolve_ana_run_addprevnext_Lscheck.sh"        
        arguments=f"{ufevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkcal_lsdist", linkfiles=[f"../{ufevt}",f"../../event_cl/{clevt}"], gdir=f"{obsid}/resolve/event_uf/")        

    if flag_dicts["calflags"]["lsdetail"]:
        runprog="resolve_run_ana_pixel_Ls_mksubgroup_using_saturatedflags.sh"        
        ufclgtievt=f"{ufname}_noBL_prevnext_cutclgti.evt"   
        arguments=f"{ufclgtievt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkcal_lsdetail", linkfiles=[f"../{ufclgtievt}"], gdir=f"{obsid}/resolve/event_uf/checkcal_lsdist")        

    if flag_dicts["calflags"]["specratio6x6"]:
        runprog="resolve_ana_pixel_plot_6x6_energyspectrum_by_itype.py"        
        arguments=f"{clevt} -r -y 0 -l 2000 -x 12000 -b 250 -c -g"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkcal_specratio6x6", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

    if flag_dicts["calflags"]["statusitype"]:
        runprog="resolve_util_stat_status_itype_fast.py"        
        arguments=f"{clevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkcal_statusitype", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        
        arguments=f"{ufevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkcal_statusitype", linkfiles=[f"../{ufevt}"], gdir=f"{obsid}/resolve/event_uf/")        

    if flag_dicts["calflags"]["statitype"]:
        runprog="resolve_util_stat_itype.py"        
        arguments=f"{clevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkcal_statitype", linkfiles=[f"../{clevt}"], gdir=f"{obsid}/resolve/event_cl/")        

    if flag_dicts["calflags"]["antico"]:
        runprog="resolve_ana_antico_comp_ELVhilo.sh"        
        arguments=f"{obsid}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkcal_antico", linkfiles=[f"../{ufacevt}",f"../../../auxil/{ehk}"], gdir=f"{obsid}/resolve/event_uf/")

    if flag_dicts["calflags"]["fe55fit"]:
        runprog="resolve_ana_pixel_ql_fit_MnKa_v2_EPI2.py"
        arguments=f"{uf50evt} --paper -n timeave_epi2"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkcal_fe55fit", linkfiles=[f"../{uf50evt}"], gdir=f"{obsid}/resolve/event_uf/")

################### deepls ###################################################################
    if flag_dicts["deeplsflags"]["addprevnext"]:
        runprog="resolve_ana_run_addprevnext_Lscheck_for_deepls.sh"        
        arguments=f"{ufevt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkdeepls_addprevnext", linkfiles=[f"../{ufevt}",f"../../event_cl/{clevt}"], gdir=f"{obsid}/resolve/event_uf/")        

    if flag_dicts["deeplsflags"]["defcluster"]:
        runprog="resolve_ana_pixel_Ls_define_cluster.py"        
        ufclgtievt=f"{ufname}_noBL_prevnext_cutclgti.evt"   
        arguments=f"{ufclgtievt}"
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkdeepls_defcluster", linkfiles=[f"../{ufclgtievt}"], gdir=f"{obsid}/resolve/event_uf/checkdeepls_addprevnext")        

################### analysis ###################################################################

    if flag_dicts["anaflags"]["genpharmfarf"]:
        runprog="resolve_auto_gen_phaarfrmf.py"        
        special_case = (obsid == "300049010") # Cyg X-1 SWG
        cldir=get_cordir(f"{obsid}/resolve/event_cl/",is_special_case=special_case)
        if special_case:
            clevt = clgcorevt

        # all pixel 
        arguments=f"-eve {clevt} -ehk {ehk} -gti {expgti} --gmin {gmin} --numphoton 300000 --clobber no --pname all" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkana_genpharmfarf", linkfiles=[f"../{clevt}",f"../../../auxil/{ehk}",f"../../event_uf/{expgti}"], gdir=cldir)        
        # centeral 4 pixels 
        arguments=f"-eve {clevt} -ehk {ehk} -gti {expgti} --gmin {gmin} --numphoton 300000 --clobber no --pname inner --pixels 0,17,18,35" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkana_genpharmfarf", linkfiles=[f"../{clevt}",f"../../../auxil/{ehk}",f"../../event_uf/{expgti}"], gdir=cldir)        
        # outer 31 pixels
        arguments=f"-eve {clevt} -ehk {ehk} -gti {expgti} --gmin {gmin} --numphoton 300000 --clobber no --pname outer --pixels 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkana_genpharmfarf", linkfiles=[f"../{clevt}",f"../../../auxil/{ehk}",f"../../event_uf/{expgti}"], gdir=cldir)        

        # # only for Cyg X-1
        # arguments=f"-eve {clgcorevt} -ehk {ehk} -gti {expgti} --gmin {gmin} --numphoton 300000 --clobber no" 
        # dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkana_genpharmfarf", linkfiles=[f"../{clgcorevt}",f"../../../auxil/{ehk}",f"../../event_uf/{expgti}"], gdir=f"{obsid}/resolve/event_cl_rslgain/")        

    if flag_dicts["anaflags"]["qlfit"]:
        special_case = (obsid == "300049010") # Cyg X-1 SWG
        gdir=get_cordir(f"{obsid}/resolve/event_cl/checkana_genpharmfarf",is_special_case=special_case)

        # all pixel 
        runprog="resolve_spec_qlfit.py"        
        phafile="rsl_source_Hp_all_gopt.pha"
        rmffile="rsl_source_Hp_all.rmf"
        arffile="rsl_source_Hp_all.arf"

        arguments=f"{phafile}" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkana_qlfit", linkfiles=[f"../{phafile}",f"../{rmffile}",f"../{arffile}"], gdir=gdir)        

        arguments=f"{phafile} --emin 6.0 --emax 7.5 --xscale off --progflags 1,1,1" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkana_qlfit", linkfiles=[f"../{phafile}",f"../{rmffile}",f"../{arffile}"], gdir=gdir)        

        # inner
        phafile="rsl_source_Hp_inner_gopt.pha"
        rmffile="rsl_source_Hp_inner.rmf"
        arffile="rsl_source_Hp_inner.arf"

        arguments=f"{phafile}" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkana_qlfit", linkfiles=[f"../{phafile}",f"../{rmffile}",f"../{arffile}"], gdir=gdir)        

        arguments=f"{phafile} --emin 6.0 --emax 7.5 --xscale off --progflags 1,1,1" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkana_qlfit", linkfiles=[f"../{phafile}",f"../{rmffile}",f"../{arffile}"], gdir=gdir)        

        # outer
        phafile="rsl_source_Hp_outer_gopt.pha"
        rmffile="rsl_source_Hp_outer.rmf"
        arffile="rsl_source_Hp_outer.arf"

        arguments=f"{phafile}" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkana_qlfit", linkfiles=[f"../{phafile}",f"../{rmffile}",f"../{arffile}"], gdir=gdir)        

        arguments=f"{phafile} --emin 6.0 --emax 7.5 --xscale off --progflags 1,1,1" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkana_qlfit", linkfiles=[f"../{phafile}",f"../{rmffile}",f"../{arffile}"], gdir=gdir)        

        # plot all, inner, and outer spec 
        runprog="xrism_spec_qlfit_many.py"        
        gotodir = f"{obsid}/resolve/event_cl/checkana_genpharmfarf/checkana_qlfit"
        gotodir=get_cordir(gotodir,is_special_case=special_case)

        arguments=f"f_gopt.list --fname {obsid}_resolve_comp_all_in_out" 
        write_to_file(f"{gotodir}/f_gopt.list", ["rsl_source_Hp_all_gopt.pha","rsl_source_Hp_inner_gopt.pha","rsl_source_Hp_outer_gopt.pha"])
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, gdir=gotodir)        

        arguments=f"f_gopt.list --fname {obsid}_resolve_comp_all_in_out_narrow --emin 6.0 --emax 7.5 --xscale off --progflags 1,1,1" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, gdir=gotodir)        

        runprog="xrism_util_plot_arf.py"
        arguments=""
        dojob(obsid, runprog, arguments = arguments, subdir="checkana_qlfit", gdir=gdir)        

    if flag_dicts["anaflags"]["compcluf"]:

        # # all pixel 
        runprog="resolve_util_screen_ufcl_std.sh"
        arguments=f"{ufevt}" 
        dojob(obsid, runprog, arguments = arguments, fwe = fwe, subdir="checkana_compcluf", linkfiles=[f"../{ufevt}",f"../../event_cl/{clevt}"], gdir=f"{obsid}/resolve/event_uf/")        

        # all pixel 
        runprog="run_resolve_ana_pixel_hist1d_many_eventfiles.sh"
        arguments=f"" 
        dojob(obsid, runprog, fwe = fwe, subdir="checkana_compcluf", gdir=f"{obsid}/resolve/event_uf/")        

################### create HTML ###################################################################

    if genhtml:
        print("..... create html file")
        runprog="xrism_autorun_png2html.py"                
        check_program_in_path(runprog)
        subprocess.run([runprog] + [obsid] + ["--keyword"] + ["check_"] + ["--ver"] +  ["v0"], check=True)
        subprocess.run([runprog] + [obsid] + ["--keyword"] + ["checkcal_"] + ["--ver"] +  ["v0"], check=True)
        subprocess.run([runprog] + [obsid] + ["--keyword"] + ["checkdeepls_"] + ["--ver"] +  ["v0"], check=True)
        subprocess.run([runprog] + [obsid] + ["--keyword"] + ["checkana_"] + ["--ver"] +  ["v0"], check=True)
                        
if __name__ == "__main__":
    main()
        

    
