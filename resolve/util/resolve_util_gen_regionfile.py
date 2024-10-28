#!/usr/bin/env python

import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

# ピクセルマッピング
pixel_fromdetxdety = [
    [12, 11,  9, 19, 21, 23],
    [14, 13, 10, 20, 22, 24],
    [16, 15, 17, 18, 25, 26],
    [ 8,  7,  0, 35, 33, 34],
    [ 6,  4,  2, 28, 31, 32],
    [ 5,  3,  1, 27, 29, 30]
]

def generate_region_file(pixels: str, name: str):
    pixel_list = [int(p) for p in pixels.split(",")]
    box_lines = []

    for one_detx in np.arange(1, 7):
        for one_dety in np.arange(1, 7):
            pixel = pixel_fromdetxdety[one_detx - 1][one_dety - 1]
            if pixel in pixel_list:
                box_lines.append(f"+box({one_detx},{one_dety},1,1)")

    filename = f"{name}.reg"

    with open(filename, 'w') as fout:
        fout.write("physical\n")
        fout.write("\n".join(box_lines))

    print(f"Generated: {filename}")
    return filename

def parse_and_plot_region_file(filepath, show=False):
    x_coords, y_coords = [], []

    with open(filepath, 'r') as fin:
        lines = fin.readlines()

    for line in lines[1:]:
        if line.startswith("+box"):
            _, coord_str = line.strip().split('(')
            detx, dety, _, _ = map(int, coord_str[:-1].split(','))
            x_coords.append(detx)
            y_coords.append(dety)

    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, s=200, c='red', marker='s', edgecolors='black')
    plt.xlim(0.5, 6.5)
    plt.ylim(0.5, 6.5)
    plt.xticks(np.arange(1, 7))
    plt.yticks(np.arange(1, 7))
    plt.grid(True)
    plt.title(f"Region: {os.path.basename(filepath)}", fontsize=9)
    plt.savefig(filepath.replace(".reg", ".png"))
    print(f"Saved plot as: {filepath.replace('.reg', '.png')}")

    if show:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Generate DS9 region files and plot them using matplotlib.",
        epilog=(
            "Examples:\n"
            "  resolve_util_gen_regionfile.py 0,17,18,35 inner\n"
            "  resolve_util_gen_regionfile.py 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34 outer\n"
            "Use --show to display the plot in a GUI window."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "pixels", type=str, help="Comma-separated list of pixel numbers (e.g., '0,17,18,35')."
    )
    parser.add_argument(
        "name", type=str, help="Short name for the output files (e.g., 'inner')."
    )    
    parser.add_argument(
        "--show", "-s", action="store_true", help="Display the plot in a GUI window."
    )

    args = parser.parse_args()
    region_file = generate_region_file(args.pixels, args.name)
    parse_and_plot_region_file(region_file, args.show)

if __name__ == "__main__":
    main()
