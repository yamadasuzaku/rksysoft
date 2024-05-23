#!/usr/bin/env python 

import argparse
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'

import numpy as np

def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # skip lines of COUNTS PIXEL ITYPE
    lines = [line.strip() for line in lines if line.strip() and not line.startswith('OBJECT') and not line.startswith('COUNTS')]
    
    counts = []
    pixels = []
    itypes = []
    
    for line in lines:
        count, pixel, itype = map(int, line.split())
        counts.append(count)
        pixels.append(pixel)
        itypes.append(itype)
    
    return np.array(counts), np.array(pixels), np.array(itypes)

def plot_data(counts, pixels, itypes, pngfile, evtfile, debug=False):
    heatmap_data = np.zeros((max(pixels) + 1, max(itypes) + 1))

    for count, pixel, itype in zip(counts, pixels, itypes):
        heatmap_data[pixel, itype] = count

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Counts')
    plt.xlabel('ITYPE')
    plt.ylabel('PIXEL')
    plt.title('Counts per Pixel and ITYPE : ' + evtfile)    

    # plot counts in the celles 
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            plt.text(j, i, f'{int(heatmap_data[i, j])}', ha='center', va='center', color='white', fontsize=8)

    plt.savefig(pngfile)

    if debug:    
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot data from rate.txt file')
    parser.add_argument('file_path', type=str, help='Path to the rate.txt file')
    parser.add_argument('debug', action='store_true', help='debug')
    
    args = parser.parse_args()
    
    counts, pixels, itypes = read_data(args.file_path)
    pngfile = args.file_path.replace(".txt",".png")
    evtfile = args.file_path.replace("_pixelrate.txt",".evt")
    plot_data(counts, pixels, itypes, pngfile, evtfile, debug=args.debug)

if __name__ == '__main__':
    main()