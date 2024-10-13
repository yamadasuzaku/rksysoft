#!/usr/bin/env python

import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)
import numpy as np
import pandas as pd
import argparse
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def extract_elements(parts):
    
    if '-' in parts:
        dash_index = parts.index('-')
        # '-' の前2つと、後ろの全てを抽出
        result = parts[dash_index-2:dash_index] + parts[dash_index+1:]
        return result
    else:
        return None

def read_data(file_path, emin, emax):
    data = np.loadtxt(file_path, skiprows=1)
    xval = data[:, 0]
    xerr_1 = data[:, 1]
    xerr_2 = data[:, 2]
    model = data[:, 3]
    ecut = np.where((xval > emin) & (xval <= emax))[0]
    xval = xval[ecut]
    xerr_1 = xerr_1[ecut]
    xerr_2 = xerr_2[ecut]
    xerr = [xerr_1, xerr_2]
    model = model[ecut]
    return xval, xerr, model

def read_secondary_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header_index = 0
    data_lines = lines[header_index + 2:]  # 2行分のヘッダーを飛ばす

    data = []
    for line in data_lines:
        if line.strip():
            parts = line.split()
            if len(parts) >= 14:
                try:
                    energy_keV = float(parts[10])
                    el = parts[2]
                    stage = parts[3]
                    ion = parts[5]
                    line = parts[6]
                    tau = float(parts[12])
                    ew_keV = float(parts[13])
                    lineinfo = extract_elements(parts)
                    data.append((energy_keV, ew_keV, tau, el, stage, ion, line, lineinfo))
                except ValueError:
                    continue

    df = pd.DataFrame(data, columns=['Energy keV', 'EW keV', 'tau','el','stage','ion','line', 'lineinfo'])
    return df

def plot_data(qdpfile, secondary_file, ylin, emin, emax, y1, y2, plotflag, marker_size):
    x, xerr, model = read_data(qdpfile, emin, emax)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    if ylin:
        ax1.set_yscale('linear')
    else:
        ax1.set_yscale('log')

    if y1 is not None:
        ax1.set_ylim(y1, y2)
        
    ax1.errorbar(x, model, fmt='-', label='model', capsize=0, color='red', alpha=0.9)
    ax1.set_ylabel(r'Photons/m$^2$/s/keV')
    ax1.set_xlabel('Energy (keV)')
    ax1.set_title(qdpfile)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if secondary_file:
        df = read_secondary_data(secondary_file)
        ax2 = ax1.twinx()
        ax2.scatter(df['Energy keV'], df['tau'], color='blue', alpha=0.8, s=marker_size, label="tau")
        ax2.set_ylabel('Optical depth at line center')
        ax2.legend(loc="lower right")
        print(f"Secondary data plotted from {secondary_file}")

    output_name = os.path.basename(qdpfile).replace('.qdp', '.png')
    plt.savefig(output_name)
    print(f"Output file {output_name} is created.")
    if plotflag:
        plt.show()

def plotly_data(qdpfile, secondary_file, ylin, emin, emax, y1, y2, marker_size):
    x, xerr, model = read_data(qdpfile, emin, emax)
    
    # Create a figure with Plotly
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add the model line plot
    fig.add_trace(
        go.Scatter(x=x, y=model, mode='lines', name='Model', line=dict(color='red'), opacity=0.5),
        secondary_y=False,
    )
    
    # Adjust y-axis scale
    if ylin:
        fig.update_yaxes(type='linear', secondary_y=False)
    else:
        fig.update_yaxes(type='log', secondary_y=False)

    if y1 is not None:
        fig.update_yaxes(range=[y1, y2], secondary_y=False)

    # Add secondary data scatter plot
    if secondary_file:
        df = read_secondary_data(secondary_file)
        for _ene, _tau, el_, stage_, ion_, line_, lineinfo_ in zip(df['Energy keV'], df['tau'], df['el'],df['stage'],df['ion'],df['line'], df['lineinfo']):
            linfo = " ".join(lineinfo_)
            fig.add_trace( 
                go.Scatter(x=[_ene], y=[_tau], mode='markers', 
                           marker=dict(size=marker_size, opacity=0.8), name=f"{el_}{stage_} {_ene} {linfo}"),
                secondary_y=True,
            )
        print(f"Secondary data plotted from {secondary_file}")

    # Update layout
    fig.update_layout(
        title=qdpfile,
        xaxis_title='Energy (keV)',
        yaxis_title=r'Photons/m^2/s/keV',
        legend=dict(
        x=1.05,  # Move legend further right (beyond the plot)
        y=1,     # Align to the top
        xanchor='left',  # Align the legend to the left of x=1.05
        yanchor='top',   # Align to the top
        orientation='v'  # Vertical orientation
        ),
        width=1000, height=600,
        font=dict(
            family="Times New Roman",  # Set the font to Times New Roman
            size=12,                   # Set the default font size (optional)
            color="black"              # Set the font color (optional)
        ),        
    )
    fig.update_yaxes(title_text="Optical depth at line center", secondary_y=True)
    
    # Save to HTML
    output_html = os.path.basename(qdpfile).replace('.qdp', '.html')
    fig.write_html(output_html)
    print(f"HTML file {output_html} is created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot spectra from a model QDP file.')
    parser.add_argument('qdpfile', help='The name of the QDP file to process.')
    parser.add_argument('-s', '--secondary_file', type=str, default=None, help='Secondary data file for scatter plot')
    parser.add_argument('-l', '--ylin', action='store_true', help='Plot yscale in linear scale (log in default)')
    parser.add_argument('-m', '--emin', type=float, help='emin', default=1.5)
    parser.add_argument('-x', '--emax', type=float, help='emax', default=10.0)
    parser.add_argument('--y1', type=float, help='y1', default=None)
    parser.add_argument('--y2', type=float, help='y2', default=None)
    parser.add_argument('--plot', '-p', action='store_true', default=False, help='Flag to display plot')
    parser.add_argument('--marker_size', type=float, help='Size of the scatter plot markers', default=6)
    parser.add_argument('--plotly', action='store_true', help='Use Plotly instead of Matplotlib')

    args = parser.parse_args()

    if args.plotly:
        plotly_data(args.qdpfile, args.secondary_file, args.ylin, args.emin, args.emax, args.y1, args.y2, args.marker_size)
    else:
        plot_data(args.qdpfile, args.secondary_file, args.ylin, args.emin, args.emax, args.y1, args.y2, args.plot, args.marker_size)
