#!/usr/bin/env python

import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)
import numpy as np
import argparse
import sys

# Set up argument parser to allow user input from command line
parser = argparse.ArgumentParser(description='Plot spectra from a QDP file.')
parser.add_argument('qdpfile', help='The name of the QDP file to process.')
parser.add_argument('-r', '--ratio', action='store_true', help='Display the ratio subplot.')
parser.add_argument('--output', type=str, help='Output file name', default='spex_fit.png')
args = parser.parse_args()

# Ensure only one argument is passed
if len(sys.argv) not in [2, 3]:
    parser.error("Invalid number of arguments.")

# Print the command-line arguments to help with debugging
args_dict = vars(args)
print("Command-line arguments:")
for arg, value in args_dict.items():
    print(f"{arg}: {value}")

# Extract arguments
qdpfile = args.qdpfile
show_ratio = args.ratio

def read_data(file_path):
    data = np.loadtxt(file_path, skiprows=1)
    x = data[:, 0]
    xerr = [data[:, 1], -data[:, 2]]
    y = data[:, 3]
    yerr = [data[:, 4], -data[:, 5]]
    model = data[:, 6]
    model_err = data[:, 7]
    return x, xerr, y, yerr, model, model_err

def plot_data(file_path, show_ratio):
    x, xerr, y, yerr, model, model_err = read_data(file_path)

    if show_ratio:
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(10, 8), sharex=True)
        ax1.errorbar(x, y, yerr=yerr, fmt='.', label='y vs x', capsize=0)
        ax1.errorbar(x, model, fmt='-', label='model vs x', capsize=0, color='red',alpha=0.9)
        ax1.set_ylabel('Counts/s/keV')
        ax1.legend()
        ax1.grid(True,alpha=0.3)

        ratio = y / model
        ratio_err = np.sqrt((yerr[0] / model) ** 2 + (y * model_err / model ** 2) ** 2)
        ax2.errorbar(x, ratio, yerr=ratio_err, fmt='.', label='ratio', capsize=0)
        ax2.set_xlabel('Energy (keV)')
        ax2.set_ylabel('data/model')
        ax2.legend()
        ax2.grid(True,alpha=0.3)
        ax2.axhline(1.0, color='black', linestyle='--', linewidth=2, label='ratio = 1.0')
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 6))
        plt.errorbar(x, y, yerr=yerr, fmt='.', label='data', capsize=0)
        plt.errorbar(x, model, fmt='-', label='model vs x', capsize=0, color='red',alpha=0.9)
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts/s/keV')
        plt.title(qdpfile)
        plt.legend()
        plt.grid(True,alpha=0.3)
    
    plt.savefig(args.output)
    print(f"Output file {args.output} is created.")
    plt.show()

plot_data(qdpfile, show_ratio)
