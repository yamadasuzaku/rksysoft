import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load and visualize HDF5 data with verbose control.")
    parser.add_argument('filename', type=str, help='Input HDF5 filename')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed output')
    return parser.parse_args()


def load_data(filename, verbose=False):
    """
    Load data from an HDF5 file.

    Parameters:
        filename (str): Path to the HDF5 file.
        verbose (bool): Flag to enable verbose output.

    Returns:
        dict: Dictionary containing arrays of loaded data.
    """
    with h5py.File(filename, 'r') as f:
        # Get and sort channel numbers
        intchs = sorted([int(key[4:]) for key in f.keys() if key.startswith('chan')])

        # Initialize lists to store all data
        all_intchs, tss, enes, dtuss, rowcs, filtphases, peakvalues = ([] for _ in range(7))

        # Iterate through channels and load data
        for intch in intchs:
            ch_key = f'chan{intch}'
            try:
                _tss = f[ch_key]['timestamp'][:]
                _enes = f[ch_key]['energy'][:]
                _dtuss = f[ch_key]['dt_us'][:]
                _rowcs = f[ch_key]['rowcount'][:]
                _filtphases = f[ch_key]['filt_phase'][:]
                _peakvalues = f[ch_key]['peak_value'][:]

                all_intchs.extend([intch] * len(_tss))
                tss.extend(_tss)
                enes.extend(_enes)
                dtuss.extend(_dtuss)
                rowcs.extend(_rowcs)
                filtphases.extend(_filtphases)
                peakvalues.extend(_peakvalues)

                if verbose:
                    print(f"Loaded channel {ch_key} with {len(_tss)} events.")

            except KeyError:
                if verbose:
                    print(f"Warning: Missing data in channel {ch_key}. Skipping.")

    # Convert lists to numpy arrays
    all_intchs = np.array(all_intchs)
    tss = np.array(tss)
    enes = np.array(enes)
    dtuss = np.array(dtuss)
    rowcs = np.array(rowcs)
    filtphases = np.array(filtphases)
    peakvalues = np.array(peakvalues)

    # Sort all arrays by timestamp
    sort_id = np.argsort(tss)
    all_intchs = all_intchs[sort_id]
    tss = tss[sort_id]
    enes = enes[sort_id]
    dtuss = dtuss[sort_id]
    rowcs = rowcs[sort_id]
    filtphases = filtphases[sort_id]
    peakvalues = peakvalues[sort_id]

    if verbose:
        print(f"Total {len(tss)} events loaded and sorted.")

    return {
        'all_intchs': all_intchs,
        'tss': tss,
        'enes': enes,
        'dtuss': dtuss,
        'rowcs': rowcs,
        'filtphases': filtphases,
        'peakvalues': peakvalues
    }


def plot_event(event_data, idx, verbose=False):
    """
    Plot the event data for each trigger using subplots with shared x-axis.

    Parameters:
        event_data (dict): Dictionary containing arrays of event data.
        idx (int): Index of the current trigger.
        verbose (bool): Flag to enable detailed output.
    """
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)

    scatter_kwargs = dict(c=event_data['peakvalues'], cmap='viridis', s=10)

    axes[0].scatter(event_data['all_intchs'], event_data['tss'], **scatter_kwargs)
    axes[0].set_ylabel('Timestamp')
    axes[0].grid(True)

    axes[1].scatter(event_data['all_intchs'], event_data['enes'], **scatter_kwargs)
    axes[1].set_ylabel('Energy')
    axes[1].grid(True)

    axes[2].scatter(event_data['all_intchs'], event_data['dtuss'], **scatter_kwargs)
    axes[2].set_ylabel('dt_us')
    axes[2].grid(True)

    axes[3].scatter(event_data['all_intchs'], event_data['filtphases'], **scatter_kwargs)
    axes[3].set_ylabel('Filt Phase')
    axes[3].grid(True)

    axes[4].scatter(event_data['all_intchs'], event_data['peakvalues'], **scatter_kwargs)
    axes[4].set_ylabel('Peak Value')
    axes[4].set_xlabel('Channel')
    axes[4].grid(True)

    fig.suptitle(f'Events for Trigger {idx}', fontsize=16)
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes, orientation='vertical', label='peakvalues',)
#    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    if verbose:
        print(f"Plotted {len(event_data['tss'])} events for trigger {idx}.")

        
def process_data(data, verbose=False):
    """
    Process loaded data and visualize each trigger separately.

    Parameters:
        data (dict): Dictionary containing arrays of event data.
        verbose (bool): Flag to enable detailed output.
    """
    uniq_rowcs = np.unique(data['rowcs'])

    for idx, u_rowc in enumerate(uniq_rowcs):
        # Select events for each trigger
        trigger_mask = data['rowcs'] == u_rowc

        event_data = {
            'all_intchs': data['all_intchs'][trigger_mask],
            'tss': data['tss'][trigger_mask],
            'enes': data['enes'][trigger_mask],
            'dtuss': data['dtuss'][trigger_mask],
            'rowcs': data['rowcs'][trigger_mask],
            'filtphases': data['filtphases'][trigger_mask],
            'peakvalues': data['peakvalues'][trigger_mask]
        }

        if verbose:
            print(f"Processing trigger {idx}: rowcount = {u_rowc}")
            for key, value in event_data.items():
                print(f"{key}: {value}")
            input("Press Enter to continue...")

        # Plot the current event
        plot_event(event_data, idx, verbose)


if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.filename):
        print(f"Error: File {args.filename} does not exist.")
        exit(1)

    print("Loading data...")
    data = load_data(args.filename, verbose=args.verbose)

    print("Processing data...")
    process_data(data, verbose=args.verbose)

    print("All done.")
