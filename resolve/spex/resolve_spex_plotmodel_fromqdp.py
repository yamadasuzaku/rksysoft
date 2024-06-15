#!/usr/bin/env python

import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)
import numpy as np
import argparse
import sys

# Set up argument parser to allow user input from command line
parser = argparse.ArgumentParser(description='Plot spectra from a model QDP file.')
parser.add_argument('qdpfile', help='The name of the QDP file to process.')
parser.add_argument('-l', '--ylin', action='store_true', help='Plot yscale in linear scale (log in default)')
parser.add_argument('-m','--emin', type=float, help='emin', default=1.5)
parser.add_argument('-x','--emax', type=float, help='emax', default=9.0)
parser.add_argument('--output', type=str, help='Output file name', default='spex_fitmodel.png')
args = parser.parse_args()

# Ensure only one argument is passed
if len(sys.argv) < 1:
    parser.error("Invalid number of arguments.")

# Print the command-line arguments to help with debugging
args_dict = vars(args)
print("Command-line arguments:")
for arg, value in args_dict.items():
    print(f"{arg}: {value}")

# Extract arguments
qdpfile = args.qdpfile
ylin = args.ylin
emin = args.emin
emax = args.emax

def read_data(file_path, emin, emax):
    data = np.loadtxt(file_path, skiprows=1)
    xval = data[:, 0]
    xerr_1 = data[:, 1]
    xerr_2 = data[:, 2]
    model = data[:, 3]
    ecut = np.where( (xval > emin) & (xval <= emax))[0]
    print(xval, type(xval))
    print(ecut)
    xval = xval[ecut]
    xerr_1 = xerr_1[ecut]
    xerr_2 = xerr_2[ecut]
    xerr = [xerr_1, xerr_2]
    model = model[ecut]
    return xval, xerr, model

def plot_data(file_path):
    x, xerr, model = read_data(file_path, emin, emax)

    plt.figure(figsize=(10, 6))
    if ylin:
        plt.yscale('linear')
    else:
        plt.yscale('log')			        

    plt.errorbar(x, model, fmt='-', label='model', capsize=0, color='red',alpha=0.9)
    plt.ylabel(r'Photons/m$^2$/s/keV')
    plt.xlabel('Energy (keV)')
    plt.title(qdpfile)
    plt.legend()
    plt.grid(True,alpha=0.3)
    plt.savefig(args.output)
    print(f"Output file {args.output} is created.")
    plt.show()

plot_data(qdpfile)
