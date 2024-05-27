#!/usr/bin/env python

#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.cm as cm

params = {'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

import pandas as pd
import re
import subprocess
import glob

def interpret_mode(s):
    if len(s) != 9:
        raise ValueError("Input string must be 9 characters long.")
    
    m_value = int(s[3])
    m_mapping = {
        0: "full",
        1: "1/8win",
        2: "full0.1s",
        3: "1/8win0.1s"
    }

    m_marker = {
        0: "o",
        1: "s",
        2: "v",
        3: "D"
    }
    
    if m_value not in m_mapping:
        raise ValueError("Invalid value for 'm'. Must be 0, 1, 2, or 3.")
    
    return m_mapping[m_value], m_marker[m_value]

def get_object_name(number):
    try:
        result = subprocess.run(
            ['resolve_util_findobs.py', number, 'observation_id', '--output_columns', 'object_name', '-n'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running external command: {e}")
        return 'unknown'

def filter_radius_and_rate(df):
    filtered_df = df[df['rate'] > 1e-9]
    if filtered_df.empty:
        raise ValueError("All data points are discarded because all rate values are <= 1e-9.")
    
    filtered_radius = filtered_df['radius']
    filtered_rate = filtered_df['rate']
    
    return filtered_radius, filtered_rate

def align_filtered_data(radius1, rate1, radius2, rate2):
    df1 = pd.DataFrame({'radius': radius1, 'rate': rate1})
    df2 = pd.DataFrame({'radius': radius2, 'rate': rate2})
    
    merged_df = pd.merge(df1, df2, on='radius', suffixes=('_1', '_2'))
    
    aligned_rate1 = merged_df['rate_1']
    aligned_rate2 = merged_df['rate_2']
    
    return aligned_rate1, aligned_rate2

def plot_scat(file_pattern_a, file_pattern_b, output_file, key1="Rate", key2="Rate (c/s)", debug=False):
    files_grade_a = sorted(glob.glob(file_pattern_a))
    print(f"files_grade_a = {files_grade_a}")
    files_grade_b = sorted(glob.glob(file_pattern_b))
    print(f"files_grade_b = {files_grade_b}")

    usercmap = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=len(files_grade_a))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=usercmap)

    plt.figure(figsize=(14, 8))

    for i, (file_path1, file_path2) in enumerate(zip(files_grade_a, files_grade_b)):
        c = scalarMap.to_rgba(i)
        print("")
        print(f"file_path1 = {file_path1} file_path2 = {file_path2} ")
        match = re.search(r'xa(\d+)xtd', file_path1)
        if match:
            number = match.group(1)
        else:
            number = 'unknown'

        match = re.search(r'_p([a-zA-Z0-9]+)_uf', file_path1)
        if match:
            mode = match.group(1)
            mode, marker = interpret_mode(mode)
        else:
            mode = 'unknown'
            marker = "*"

        object_name = get_object_name(number)

        print(f"object_name = {object_name} mode = {mode} ")

        df1 = pd.read_csv(file_path1, header=None, names=['radius', 'ignored', 'rate'])
        radius1, rate1 = filter_radius_and_rate(df1)

        df2 = pd.read_csv(file_path2, header=None, names=['radius', 'ignored', 'rate'])
        radius2, rate2 = filter_radius_and_rate(df2)

        aligned_rate1, aligned_rate2 = align_filtered_data(radius1, rate1, radius2, rate2)

        linestyle = "--" if i > 9 else "-"

        plt.plot(aligned_rate2, aligned_rate1, color=c, marker=marker, linestyle=linestyle, alpha=0.8, label=f'{object_name} ({number} {mode})')

    plt.ylabel(f'{key1} of grade 1')
    plt.xlabel(f'{key2} of grade 0,2-4,6')
    plt.title('Xtend : Comparison of Rates')
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, alpha=0.4)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1),)
    plt.tight_layout()
    plt.savefig(output_file)
    if debug:
        plt.show()

def plot_data_one(file_pattern_a, output_file, key="Grade 1", label="Rate (c/s)", debug=False):
    files_grade_a = sorted(glob.glob(file_pattern_a))
    print(f"files_grade_a = {files_grade_a}")

    plt.figure(figsize=(14, 8))

    usercmap = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=len(files_grade_a))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=usercmap)

    for i, file_path in enumerate(files_grade_a):
        c = scalarMap.to_rgba(i)
        print("")        
        print(f"file_path = {file_path}")
        match = re.search(r'xa(\d+)xtd', file_path)
        if match:
            number = match.group(1)
        else:
            number = 'unknown'

        match = re.search(r'_p([a-zA-Z0-9]+)_uf', file_path)
        if match:
            mode = match.group(1)
            mode, marker = interpret_mode(mode)
        else:
            mode = 'unknown'
            marker = "*"

        object_name = get_object_name(number)
        df = pd.read_csv(file_path, header=None, names=['radius', 'ignored', 'rate'])
        radius, rate = filter_radius_and_rate(df)

        linestyle = "--" if i > 9 else "-"

        plt.plot(radius, rate, color=c, marker=marker, linestyle=linestyle, alpha=0.8, label=f'{object_name} ({number} {mode})')

    plt.xlabel('Radius (pixel)')
    plt.ylabel(label)
    plt.title(f'Xtend : {label} vs. Radius from {key}')
    plt.yscale("log")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1),)
    plt.tight_layout()

    plt.savefig(output_file)
    if debug:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='This program is used to plot radial profiles from texts created by pilep uf tool. ',
      epilog='''
        Example:
        python xtend_pileup_plot_radial_fromtext_comp2.py (please chech the default setting.)
        python xtend_pileup_plot_radial_fromtext_comp2.py --file_pattern_a "*ene1*txt" --file_pattern_b "*ene3*txt" --key1 "0.5-3 keV Rate" --key2 "7-10keV Rate" --output_file "hardness_xtend_pileup_comp2.png" --gkey1 "Grade 0,2-4,6" --gkey2 "Grade 0,2-4,6"
      ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)    
    parser.add_argument('--file_pattern_a', type=str, default='*grade1*txt', help='File pattern for the first set of files (default: *grade1*)')
    parser.add_argument('--file_pattern_b', type=str, default='*gradeg*txt', help='File pattern for the second set of files (default: *gradeg*)')
    parser.add_argument('--output_file', type=str, default="xtend_pileup_comp2.png", help='Output PNG file name')
    parser.add_argument('--gkey1', type=str, default='Grade 1', help='Label for file_pattern_a (default: Grade 1)')
    parser.add_argument('--gkey2', type=str, default='Grade 0,2-4,6', help='Label for file_pattern_b (default: Grade 0,2-4,6)')
    parser.add_argument('--key1', type=str, default='Rate', help='Label for y-axis (default: Rate)')
    parser.add_argument('--key2', type=str, default='Rate (c/s)', help='Label for x-axis (default: Rate (c/s))')
    parser.add_argument('-d', '--debug', action='store_true', help='Flag to show detailed information', default=False)
    args = parser.parse_args()

    plot_scat(args.file_pattern_a, args.file_pattern_b, "scat_grade1_gradeg_" + args.output_file, key1=args.key1, key2=args.key2, debug=args.debug)
    plot_data_one(args.file_pattern_a, "grade1_" + args.output_file, key=args.gkey1, label=args.key1, debug=args.debug)
    plot_data_one(args.file_pattern_b, "gradeg_" + args.output_file, key=args.gkey2, label=args.key2, debug=args.debug)
