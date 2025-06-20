#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from astropy.io import fits

# Constants
itypemax = 8
g_typename = ["Hp", "Mp", "Ms", "Lp", "Ls", "BL", "EL", "NA"]
itype_interest = [0, 1, 2, 3, 4]  # Hp, Mp, Ms, Lp, Ls
npix = 36
LO_RES_PH_RANGE = (-1000 + 0, 16383 + 1000)
NEXT_INTERVAL_RANGE = (0, 255)

def add_bar_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height, f"{int(height)}",
                    ha='center', va='bottom', fontsize=8)

def analyze_and_plot(fits_file, output_dir="diagnostic_plots"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Opening FITS file: {fits_file}")

    with fits.open(fits_file) as hdul:
        data = hdul[1].data

        for pixel in range(npix):
            print(f"\nProcessing pixel {pixel}...")

            mask_pixel = data['PIXEL'] == pixel
            icl_l = data['ICLUSTERL'][mask_pixel]
            icl_s = data['ICLUSTERS'][mask_pixel]
            itype_vals = data['ITYPE'][mask_pixel]

            n_l = np.sum(icl_l > 0)
            n_s = np.sum(icl_s > 0)
            n_both = np.sum((icl_l > 0) | (icl_s > 0))
            n_both_zero = np.sum((icl_l == 0) | (icl_s == 0))

            print(f"  Events with ICLUSTERL > 0 : {n_l}")
            print(f"  Events with ICLUSTERS > 0 : {n_s}")
            print(f"  Events with both > 0      : {n_both}")
            print(f"  Events with both == 0     : {n_both_zero}")

            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            fig.suptitle(f"Pixel {pixel} - Diagnostic Summary")

            # 1. ITYPE distribution

            counts = [np.sum(itype_vals == t) for t in itype_interest]
            bars = axes[0, 0].bar(g_typename[:5], counts, color='skyblue')
            add_bar_labels(axes[0, 0], bars)
            axes[0, 0].set_title("ITYPE Distribution")
            axes[0, 0].set_ylabel("Counts")

            # 2. LO_RES_PH vs NEXT_INTERVAL
            # ITYPEラベルと色の対応
            itype_labels = g_typename[:5]
            itype_colors = ['blue', 'green', 'purple', 'cyan', 'magenta']

            # クラスタタイプに応じたマーカー形式
            cluster_types = {
                'ICLUSTERL>0': ((icl_l > 0) & (icl_s == 0), 'o'),
                'ICLUSTERS>0': ((icl_s > 0) & (icl_l == 0), '^'),
                'Both>0':      ((icl_l > 0) & (icl_s > 0),   's'),
                'ICLUSTERL/S==0': ((icl_l == 0) & (icl_s == 0), 'x'),
            }

            # 描画データ抽出
            x_all = data['NEXT_INTERVAL'][mask_pixel]
            y_all = data['LO_RES_PH'][mask_pixel]
            itype_all = data['ITYPE'][mask_pixel]

            # 散布図描画
            for cluster_label, (cluster_mask, marker) in cluster_types.items():
                for i, itype_val in enumerate([3,4]): # Lp, Ls
                    mask = (itype_all == itype_val) & cluster_mask
                    count = np.sum(mask)                    
                    if count > 0:
                        axes[0, 1].scatter(
                            x_all[mask],
                            y_all[mask],
                            s=20,
                            alpha=0.5,
                            marker=marker,
                            label=f"{itype_labels[itype_val]} ({cluster_label}) : {count}"
                        )

            axes[0, 1].set_xlim(*NEXT_INTERVAL_RANGE)
            axes[0, 1].set_ylim(*LO_RES_PH_RANGE)
            axes[0, 1].set_title("LO_RES_PH vs NEXT_INTERVAL\n(ITYPE × Cluster Type)")
            axes[0, 1].set_xlabel("NEXT_INTERVAL")
            axes[0, 1].set_ylabel("LO_RES_PH")
            axes[0, 1].grid(alpha=0.3)
            axes[0, 1].legend(loc='upper right')

            # 3. Pseudo Event Counts (separate bars)
            only_l = np.sum((icl_l > 0) & (icl_s == 0))
            only_s = np.sum((icl_s > 0) & (icl_l == 0))
            both = np.sum((icl_l > 0) | (icl_s > 0))

            bar_labels = ['ICLUSTERL>0', 'ICLUSTERS>0', 'Both>0']
            bar_values = [only_l, only_s, both]
            bar_colors = ['red', 'orange', 'green']

            bars = axes[1, 0].bar(bar_labels, bar_values, color=bar_colors)
            axes[1, 0].set_title("Pseudo Events by Type (Independent)")
            axes[1, 0].set_ylabel("Counts")
            add_bar_labels(axes[1, 0], bars)

            # 4. ITYPEs of Clustered Events (separate by cluster type)
            itypes_only_l = itype_vals[(icl_l > 0) & (icl_s == 0)]
            itypes_only_s = itype_vals[(icl_s > 0) & (icl_l == 0)]
            itypes_both = itype_vals[(icl_l > 0) | (icl_s > 0)]

            counts_only_l = [np.sum(itypes_only_l == t) for t in itype_interest]
            counts_only_s = [np.sum(itypes_only_s == t) for t in itype_interest]
            counts_both   = [np.sum(itypes_both   == t) for t in itype_interest]

            bar_width = 0.25
            x = np.arange(len(itype_interest))

            bars1 = axes[1, 1].bar(x - bar_width, counts_only_l, width=bar_width, color='red', label='ICLUSTERL>0')
            bars2 = axes[1, 1].bar(x,             counts_only_s, width=bar_width, color='orange', label='ICLUSTERS>0')
            bars3 = axes[1, 1].bar(x + bar_width, counts_both,   width=bar_width, color='green', label='Both>0')

            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(g_typename[:5])
            axes[1, 1].set_title("ITYPEs of Clustered Events (by Cluster Type)")
            axes[1, 1].set_ylabel("Counts")
            axes[1, 1].legend()

            add_bar_labels(axes[1, 1], bars1 + bars2 + bars3)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            outpath = os.path.join(output_dir, f"pixel_{pixel:02d}_diagnostic.png")
#            plt.show()
            fig.savefig(outpath)
            print(f"  Saved plot to: {outpath}")
            plt.close(fig)

        # Summary view: Discarded Events per Pixel by Cluster Type
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharey=True)
        all_pixels = data['PIXEL']
        all_itype = data['ITYPE']
        all_icl_l = data['ICLUSTERL']
        all_icl_s = data['ICLUSTERS']

        # 1. ICLUSTERL > 0 & ICLUSTERS == 0 (Only L)
        only_l_counts_lp = [np.sum((all_pixels == pix) & (all_icl_l > 0) & (all_icl_s == 0) & (all_itype == 3)) for pix in range(npix)]
        only_l_counts_ls = [np.sum((all_pixels == pix) & (all_icl_l > 0) & (all_icl_s == 0) & (all_itype == 4)) for pix in range(npix)]

        # 2. ICLUSTERS > 0 & ICLUSTERL == 0 (Only S)
        only_s_counts_lp = [np.sum((all_pixels == pix) & (all_icl_s > 0) & (all_icl_l == 0) & (all_itype == 3)) for pix in range(npix)]
        only_s_counts_ls = [np.sum((all_pixels == pix) & (all_icl_s > 0) & (all_icl_l == 0) & (all_itype == 4)) for pix in range(npix)]

        # 3. Both > 0
        both_counts_lp = [np.sum((all_pixels == pix) & ((all_icl_l > 0) | (all_icl_s > 0)) & (all_itype == 3)) for pix in range(npix)]
        both_counts_ls = [np.sum((all_pixels == pix) & ((all_icl_l > 0) | (all_icl_s > 0)) & (all_itype == 4)) for pix in range(npix)]

        # 4. All
        all_lp = [np.sum((all_pixels == pix) & (all_itype == 3)) for pix in range(npix)]
        all_ls = [np.sum((all_pixels == pix) & (all_itype == 4)) for pix in range(npix)]

        all_lpls_except_for_pixel12 = [np.sum((all_pixels == pix) & ((all_itype == 3) | (all_itype == 4))) for pix in range(npix) if pix != 12]
        ymax = np.amax(all_lpls_except_for_pixel12)

        def plot_cluster_summary(ax, lp_counts, ls_counts, title):
            x = np.arange(npix)
            width = 0.4
            bars1 = ax.bar(x - width/2, lp_counts, width=width, color="cyan", label="Lp")
            bars2 = ax.bar(x + width/2, ls_counts, width=width, color="magenta", label="Ls")
#            ax.set_title(title)
#            ax.set_xlabel("Pixel")
            ax.set_ylabel(f"Event Count \n {title}")
            ax.set_xticks(x)
            ax.set_xlim(-1,37)            
            ax.set_ylim(0,ymax)            
            ax.set_xticklabels(x)
            add_bar_labels(ax, bars1 + bars2)
            ax.legend()

        plot_cluster_summary(axes[0], all_lp, all_ls, "All Lp Ls")
        plot_cluster_summary(axes[1], only_l_counts_lp, only_l_counts_ls, "ICLUSTERL > 0")
        plot_cluster_summary(axes[2], only_s_counts_lp, only_s_counts_ls, "ICLUSTERS > 0")
        plot_cluster_summary(axes[3], np.array(all_lp) - np.array(both_counts_lp), np.array(all_ls) - np.array(both_counts_ls), "ALL - (L+S)")

        plt.figtext(0.05,0.02,f"{fits_file}", fontsize=8,color="gray")
  
        axes[-1].set_xlabel("Pixel")
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "summary_discarded_per_pixel_by_type.png")
        fig.savefig(summary_path)
        print(f"\nSaved cluster-type-separated summary plot to: {summary_path}")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pseudo clusters in FITS file")
    parser.add_argument("fits_file", help="Input FITS file with ICLUSTERL and ICLUSTERS columns")
    parser.add_argument("--output_dir", default="diagnostic_plots", help="Output directory for plots")
    args = parser.parse_args()
    analyze_and_plot(args.fits_file, args.output_dir)
