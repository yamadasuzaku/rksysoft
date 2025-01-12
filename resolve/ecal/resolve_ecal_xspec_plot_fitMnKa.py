#!/usr/bin/env python

import argparse

import os
import pandas as pd
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)

# ディレクトリ内のCSVファイルを処理
def read_fit_results(directory):
    data = {"PIXEL": [], "Sigma": [], "Sigma_NegErr": [], "Sigma_PosErr": [],
            "Const": [], "Const_NegErr": [], "Const_PosErr": [],
            "Gain": [], "Gain_NegErr": [], "Gain_PosErr": []}

    for filename in os.listdir(directory):
        if filename.startswith("55Fe_PIXEL") and filename.endswith("_fitresult.csv"):
            pixel = int(filename.split("_")[1][5:])  # PIXEL番号を抽出
            df = pd.read_csv(os.path.join(directory, filename))
            
            data["PIXEL"].append(pixel)
            for param in ["Sigma", "Const", "Gain"]:
                row = df[df["Parameter"] == param].iloc[0]
                data[param].append(row["Value"])
                data[f"{param}_NegErr"].append(row["Negative Error"])
                data[f"{param}_PosErr"].append(row["Positive Error"])

    return pd.DataFrame(data)

# データをプロット
def plot_fit_results(df, outpng="default_fitMnKa.png", debug=False):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    parameters = ["Sigma", "Const", "Gain"]
    colors = ["blue", "green", "red"]

    for i, param in enumerate(parameters):
        if param == "Sigma":
            sigma2fwhm2eV = 2.35 * 1e3
            ax[i].errorbar(df["PIXEL"], df[param] * sigma2fwhm2eV, 
                           yerr=[df[f"{param}_NegErr"]*sigma2fwhm2eV, df[f"{param}_PosErr"]*sigma2fwhm2eV],
                           fmt='o', color=colors[i], label=param, capsize=1)
            ax[i].set_ylabel(r"$\Delta E$ (eV)")
            ax[i].set_title(f"dE vs PIXEL")
        elif param == "Gain":
            keV2eV=1e3
            ax[i].errorbar(df["PIXEL"], df[param]*keV2eV, 
                           yerr=[df[f"{param}_NegErr"]*keV2eV, df[f"{param}_PosErr"]*keV2eV],
                           fmt='o', color=colors[i], label=param, capsize=1)
            ax[i].set_ylabel(r"Energy Shift (eV)")
            ax[i].set_title(f"Energy Shift vs PIXEL")
        else:
            ax[i].errorbar(df["PIXEL"], df[param], 
                           yerr=[df[f"{param}_NegErr"], df[f"{param}_PosErr"]],
                           fmt='o', color=colors[i], label=param, capsize=1)
            ax[i].set_ylabel(f"{param}")
            ax[i].set_title(f"{param} vs PIXEL")

        ax[i].grid(True, linestyle='--', alpha=0.6)
        ax[i].legend()

    ax[-1].set_xlabel("PIXEL")
    plt.tight_layout()
    plt.savefig(outpng)
    print(f"{outpng} is saved.")
    if debug:
        plt.show()

# メイン処理
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract values from a file based on search text.")
    parser.add_argument("--directory", type=str, default="./", help="directory where 55Fe_PIXELdd_fitresult.csv exist.")
    parser.add_argument('--outpng', '-o', type=str, default='resolve_xspec_fitMnKa.png', help='output file name')
    parser.add_argument('--debug', '-d', action='store_true', default=False, help='The debug flag')

    args = parser.parse_args()

    directory = args.directory
    outpng = args.outpng
    debug = args.debug

    fit_results = read_fit_results(directory)
    fit_results.sort_values("PIXEL", inplace=True)  # PIXEL順にソート
    plot_fit_results(fit_results, outpng=outpng, debug=debug)

