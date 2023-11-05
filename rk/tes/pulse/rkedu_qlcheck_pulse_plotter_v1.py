#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse

# obtain file name and two sample numbers
parser = argparse.ArgumentParser(description='Plot waveforms from a FITS file.')
parser.add_argument('filename', help='The name of the FITS file to process.')
parser.add_argument('index_a', type=int, help='The index of the first pulse to plot.')
parser.add_argument('index_b', type=int, help='The index of the second pulse to plot.')
parser.add_argument('filter_size', type=int, help='The size of the uniform filter')
args = parser.parse_args()

fname = args.filename
index_a = args.index_a
index_b = args.index_b
filter_size = args.filter_size # Filter size for smoothing the derivative

def read_pulses(filename):
    """
    Reads pulse data from a FITS file.
    
    Parameters:
    filename (str): The name of the FITS file to read from.
    
    Returns:
    numpy.ndarray: An array of pulse data.
    """
    return fits.open(filename)[1].data["PulseRec"]

def plot_waveforms(pulse1, pulse2, indices, normalize=False):
    """
    Plots two waveforms on the same figure.
    
    Parameters:
    pulse1, pulse2 (numpy.ndarray): The waveforms to be plotted.
    indices (tuple): The indices of the pulses for labeling.
    normalize (bool): Whether to normalize the waveforms.
    """
    if normalize:
        pulse1 = pulse1 / np.amax(pulse1)
        pulse2 = pulse2 / np.amax(pulse2)
    plt.plot(pulse1, label=str(indices[0]))
    plt.plot(pulse2, label=str(indices[1]))
    plt.legend()
    plt.show()

def plot_derivatives(pulse1, pulse2, indices):
    """
    Plots the derivatives of two waveforms.
    
    Parameters:
    pulse1, pulse2 (numpy.ndarray): The waveforms whose derivatives are to be plotted.
    indices (tuple): The indices of the pulses for labeling.
    """
    diff_pulse1 = np.diff(pulse1)
    diff_pulse2 = np.diff(pulse2)
    plot_waveforms(diff_pulse1, diff_pulse2, indices)

def apply_filter(pulse, filter_size):
    """
    Applies a uniform filter to a pulse waveform.
    
    Parameters:
    pulse (numpy.ndarray): The waveform to filter.
    filter_size (int): The size of the filter.
    
    Returns:
    numpy.ndarray: The filtered waveform.
    """
    flatfilt = np.ones(filter_size) / filter_size
    return np.convolve(pulse, flatfilt, mode="same")

# Read pulse data from FITS file
pulses = read_pulses(fname)

# Extract pulses for the given indices
pulse_a = pulses[index_a]
pulse_b = pulses[index_b]

# Plot original waveforms
plot_waveforms(pulse_a, pulse_b, (index_a, index_b))

# Plot derivatives of waveforms
plot_derivatives(pulse_a, pulse_b, (index_a, index_b))

# Normalize and plot original waveforms and their derivatives
plot_waveforms(pulse_a, pulse_b, (index_a, index_b), normalize=True)
plot_derivatives(pulse_a, pulse_b, (index_a, index_b))

# Apply the filter to the derivatives
filt_diff_pulse_a = apply_filter(np.diff(pulse_a), filter_size)
filt_diff_pulse_b = apply_filter(np.diff(pulse_b), filter_size)

# Normalize and plot filtered derivatives
plot_waveforms(filt_diff_pulse_a, filt_diff_pulse_b, (index_a, index_b), normalize=True)
