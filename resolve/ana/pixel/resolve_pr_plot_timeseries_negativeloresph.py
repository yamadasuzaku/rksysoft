#!/usr/bin/env python

import os
import astropy.io.fits
import matplotlib.pylab as plt
import argparse
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
from astropy.time import Time
import datetime

# Constants
MJD_REFERENCE_DAY = 58484
ADC_LENGTH = 1040
XADC_TIME_STEP = 80e-6  # sec

# Compute time array in seconds
XADC_TIME = np.arange(0, ADC_LENGTH, 1) * XADC_TIME_STEP

# Initialize reference time for MJD calculation
REFERENCE_TIME = Time(MJD_REFERENCE_DAY, format='mjd')

# Plotting configuration
params = {'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 7}
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update(params)


icolor =    ["r", "b", "c", "g", "m", "k", "y", "0.8"]

def compute_date_from_reference(date_sec, reference_time):
    """Compute date from a reference time for a given seconds value."""
    return reference_time.datetime + datetime.timedelta(seconds=float(date_sec))


def get_derivative(y_data, step=8, search_i_min=100, search_i_max=200):
    """
    Calculate the derivative of the given data.
    
    Args:
    - y_data (array): Input data for derivative calculation.
    - step (int): Number of elements for step calculation.
    - search_i_min (int): Minimum index for search range.
    - search_i_max (int): Maximum index for search range.
    
    Returns:
    - tuple: (derivative array, max value, max index, min value, min index, peak, peak index)
    """
    derivative_values = []
    index_array = np.arange(0, len(y_data), 1)

    # Calculate the derivatives
    for i, y_value in enumerate(y_data):
        prev_values = y_data[i - step:i] if i >= step else []
        post_values = y_data[i:i + step] if i + step < len(y_data) else []

        if prev_values and post_values:
            long_derivative = (np.mean(post_values) - np.mean(prev_values)) * 8
        else:
            long_derivative = 0
        
        derivative = np.floor((long_derivative + 2.) / 4.)
        derivative_values.append(derivative)

    # Convert the list to a numpy array
    np_derivatives = np.array(derivative_values)

    # Search for min/max within the defined range
    deriv_max = np.amax(np_derivatives[(index_array > search_i_min) & (index_array < search_i_max)])
    deriv_min = np.amin(np_derivatives[(index_array > search_i_min) & (index_array < search_i_max)])

    # Get indices of max/min values
    deriv_max_i = index_array[np.where(np_derivatives == deriv_max)][0]
    deriv_min_i = index_array[np.where(np_derivatives == deriv_min)][0]

    # Determine the peak derivative
    peak = deriv_max if np.abs(deriv_max) > np.abs(deriv_min) else deriv_min
    peak_index = index_array[np.where(np_derivatives == peak)][0]

    return np_derivatives, deriv_max, deriv_max_i, deriv_min, deriv_min_i, peak, peak_index


def get_itype_str(itype):
    """
    Returns a string representation for a given itype.

    Args:
    - itype (int): The itype value.

    Returns:
    - str: The string representation of the itype.
    """
    itype_dict = {-1: 'all', 0: 'Hp', 1: 'Mp', 2: 'Ms', 3: 'Lp', 4: 'Ls', 5: 'BL', 6: 'EL', 7: '--'}
    return itype_dict.get(itype, 'unknown')

def plot_fits(fname, pixel_list, itype_list, target_pulserec_mode, plot_flag, color_auto, offset1 = 10, offset2 = 10):
    """
    Process a FITS file and generate plots for the given pixels and itypes.
    If pixel_list is None, plot all pixels without filtering by pixel.
    """
    # Prepare output directory
    output_directory = fname.replace(".evt", "").replace(".gz", "") + "_plots"
    os.makedirs(output_directory, exist_ok=True)

    # Load FITS data
    hdu = astropy.io.fits.open(fname)[1]
    data = hdu.data
    date_obs, date_end = hdu.header['DATE-OBS'], hdu.header['DATE-END']
    obsid = hdu.header["OBS_ID"]
    target = hdu.header["OBJECT"]

    # Extract relevant columns
    time_array = data["TIME"]
    triglp_array = data["TRIG_LP"]
    itype_array = data["ITYPE"]
    pixel_array = data["PIXEL"]
    pha_array = data["PHA"]
    lo_array = data["LO_RES_PH"]
    derivmax_array = data["DERIV_MAX"]
    nt_array = data["NEXT_INTERVAL"]
    trigtime_array = data["TRIGTIME"]
    pulserec_array = data["PULSEREC"]
    if "PULSEREC_MODE" in data.columns.names:
        pulserec_mode_array = data["PULSEREC_MODE"]
    else:
        pulserec_mode_array = None
        print("PULSEREC_MODE key does not exist in the FITS file. Proceeding with default handling.")

    # If pixel_list is None, skip filtering by pixel and plot all pixels in one figure
    if pixel_list is None:
        print("[ERROR] Plotting for all pixels (not implemented)")
        sys.exit()
    else:
        npulse = 0
        for k, target_pixel in enumerate(pixel_list):
            print("------------------------------------")
            print(f"Processing target_pixel = {target_pixel}")

            fig, axes = plt.subplots(len(itype_array), 1, figsize=(10, 8), sharex=True, sharey=False)
#            plt.subplots_adjust(right=0.9, top=0.85)
#            plt.figure(figsize=(11, 8))
            plt.subplots_adjust(right=0.6)  # make space for the legend

            t0 = time_array[0]
            trigtime0 = trigtime_array[0]

            triglp0 = triglp_array[0]

            event_count_per_itype = []
            for itype in itype_list:
                print(f"Processing itype = {itype}")
                itype_str = get_itype_str(itype)

                filter_condition = (pixel_array == target_pixel) & (itype_array == itype) & (pulserec_mode_array == target_pulserec_mode)
                selected_indices = np.where(filter_condition)[0]

                selected_triglp = triglp_array[selected_indices] - triglp0
                selected_pulserec = pulserec_array[selected_indices]
                selected_times = time_array[selected_indices] - t0
                selected_trig_times = trigtime_array[selected_indices] - trigtime0
                selected_itype = itype_array[selected_indices]
                selected_pixel = pixel_array[selected_indices]
                selected_nt = nt_array[selected_indices]
                selected_lo = lo_array[selected_indices]
                selected_pha = pha_array[selected_indices]
                selected_derivmax = derivmax_array[selected_indices]
                selected_pulserec_mode = pulserec_mode_array[selected_indices]

                event_count_per_itype.append(len(selected_times))

                for i, (pulserec, onertime, oneitype, onepixel, onetrigtime, onelo, onederivmax, onepha, onent, onetriglp) in enumerate(
                    zip(selected_pulserec, selected_times, selected_itype, selected_pixel, selected_trig_times,
                        selected_lo, selected_derivmax, selected_pha, selected_nt, selected_triglp)):
                    print(i, onertime, oneitype, onepixel, onetrigtime)
                    color = icolor[oneitype] if not color_auto else None
                    if onepha == onelo or oneitype < 3:
                        axes[npulse].errorbar(onetriglp + np.arange(len(pulserec)), pulserec, fmt='-', color=color, alpha=0.8,
                                     label=f"it={oneitype} nt={onent} lo={onelo} triglp={onetriglp} \n pha={onepha} der={onederivmax} rtime={onertime:.4f}")
#                        axes[npulse].errorbar(XADC_TIME + onertime, pulserec, fmt='-', color=color, alpha=0.8,
#                                     label=f"it={oneitype} nt={onent} lo={onelo} triglp={onetriglp} \n pha={onepha} der={onederivmax} rtime={onertime:.4f}")

                    else:
                        axes[npulse].errorbar(onetriglp + np.arange(len(pulserec)), pulserec, fmt='-', color=color, alpha=0.8,
                                     label=f"[negative] it={oneitype} nt={onent} triglp={onetriglp} \n lo={onelo} pha={onepha} der={onederivmax} rtime={onertime:.4f}")
#                        axes[npulse].errorbar(XADC_TIME + onertime, pulserec, fmt='-', color=color, alpha=0.8,
#                                     label=f"[negative] it={oneitype} nt={onent} triglp={onetriglp} \n lo={onelo} pha={onepha} der={onederivmax} rtime={onertime:.4f}")

                    print(f"npulse = {npulse}")
                    axes[npulse].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8)
                    axes[npulse].set_ylabel('ADC(pulse)')
                    axes[npulse].grid(alpha=0.3, ls=':')
                    npulse = npulse + 1

            plt.figtext(0.1, 0.9, f"{obsid} {target} {fname} PIXEL={target_pixel}  pmode = {target_pulserec_mode}")
            plt.figtext(0.1, 0.94, f"{date_obs} to {date_end}  # of events = {np.sum(event_count_per_itype)} ({','.join(map(str, event_count_per_itype))})")
            time0_datetime = compute_date_from_reference(time_array[0], REFERENCE_TIME)
            plt.xlabel(f'TRIGLP from {triglp0} \n Time (s) from {time_array[0]} \n ({time0_datetime})')
            plt.savefig(os.path.join(output_directory, f"{fname}_pixel{target_pixel:02d}_all.png"))

            if plot_flag:
                plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process FITS file and generate plots.")
    parser.add_argument("fname", type=str, help="File name to be processed.")
    parser.add_argument('--itypelist', '-i', type=str, default='0,1,2,3,4', help='Comma-separated list of itype values (default: 0,1,2,3,4)')
    parser.add_argument('--plotpixels', '-p', type=str, help='Comma-separated list of pixels to plot', default=','.join(map(str, range(36))))
    parser.add_argument("--target_pulserec_mode", type=int, default=0, choices=[0, 1], help="Target pulserec_mode value.")
    parser.add_argument("--plot_flag", "-s", action="store_true", help="If set, will display plots.")
    parser.add_argument("--color_auto", "-c", action="store_true", help="If set, automatic color.")
    parser.add_argument('--offset1', type=float, help='offset1 used for ITYPE', default=10)
    parser.add_argument('--offset2', type=float, help='offset2 used for num pulse', default=100)

    args = parser.parse_args()
    itype_list = [int(itype) for itype in args.itypelist.split(',')]
    print(args.plotpixels)
    if args.plotpixels == "None":
        pixel_list = None
    else:
        pixel_list = list(map(int, args.plotpixels.split(',')))

    plot_fits(args.fname, pixel_list, itype_list, args.target_pulserec_mode, args.plot_flag, args.color_auto, offset1 = args.offset1, offset2 = args.offset2)
