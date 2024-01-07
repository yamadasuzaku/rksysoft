#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time

import glob
import numpy as np
import datetime
import os

os.makedirs('./logfile', exist_ok = True)
file_path = './logfile/output_syslog.txt'

#target_date = datetime.datetime.strptime("20231013", '%Y%m%d')
#target_date = datetime.datetime.strptime("20231009", '%Y%m%d')
target_date = datetime.datetime.strptime("20231213", '%Y%m%d')
#target_date = datetime.datetime.strptime("20240105", '%Y%m%d')

def cut_flist(flist):
    filtered_files = []

    for file in flist:
        # ファイル名の日付部分だけを取得
        file_date_str = file[12:20]
        file_date = datetime.datetime.strptime(file_date_str, '%Y%m%d')
#        try:
#            file_date = datetime.strptime(file_date_str, '%Y%m%d')
#        except ValueError:
#            continue  # ファイル名が日付形式でない場合はスキップ
#    
        # 3. globで取得したファイルリストから、条件に合致するファイル名のみを抽出
        if file_date > target_date:
            filtered_files.append(file)
    return filtered_files

# MJD reference day 01 Jan 2019 00:00:00
MJD_REFERENCE_DAY = 58484
reference_time = Time(MJD_REFERENCE_DAY, format='mjd')

def write_to_file(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text)

def process_data(target_fname):
    data = fits.open(target_fname)[14].data
    time, msg, psp_id = data["TIME"], data["MSG"], data["PSP_ID"]
    sortid = np.argsort(time)
    time = time[sortid]
    msg = msg[sortid]
    psp_id = psp_id[sortid]
    print(f"Number of events are {len(time)} in {target_fname}")
    return time, msg, psp_id

def search_in_msg(msg, dtime, psp_id, search_term = "psp_task_wfrb_watch"):
    for onemsg, onedtime, onepsp_id in zip(msg, dtime, psp_id):
        if search_term in onemsg:
            outstr=str(onedtime) + " PSP_ID= " + str(onepsp_id) + " msg= " + onemsg
            print(outstr)
            write_to_file(file_path, outstr+"\n")

# ディレクトリパス
directory_path = "/data0/qlff/"
# globを使用してディレクトリ内のすべてのファイルを取得
file_paths = glob.glob(os.path.join(directory_path, '**/xa*rsl_*.hk1.gz'), recursive=True)

# ファイル名から日付部分を取り出してソート
sorted_file_paths = sorted(file_paths, key=lambda x: x.split('/')[3])

flist = cut_flist(sorted_file_paths)
#flist = sorted_file_paths # no cut 

for i, onef in enumerate(flist):
    try:
        time, msg, psp_id = process_data(onef)
        dtime = [reference_time.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in time]
        search_in_msg(msg, dtime, psp_id)        
    except Exception as e:
        print(e)
    
