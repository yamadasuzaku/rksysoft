#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
import glob
import datetime
import os
import argparse

# Define color text function for highlighting messages
def color_text(text, color):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    return f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process FITS log files and search for specific messages.")
parser.add_argument("--file_path", type=str, default="./logfile/output_syslog.txt", help="Output log file path")
parser.add_argument("--mjd_reference", type=int, default=58484, help="MJD reference day")
parser.add_argument("--base_path", type=str, default="/nasA_xarm1/sot/qlff/", help="Base path for file search")
parser.add_argument("--date", type=str, default="20250210", help="Date for filtering file search")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(os.path.dirname(args.file_path), exist_ok=True)

# Initialize MJD reference time
reference_time = Time(args.mjd_reference, format='mjd')

# Generate file paths dynamically based on arguments
search_pattern = f"{args.base_path}{args.date}*/*/*/xa*rsl_*.hk1.gz"
file_paths = glob.glob(search_pattern, recursive=True)

# Check if file_paths is empty
if not file_paths:
    print(color_text("Error: No files found matching the search pattern.", "red"))
    exit(1)

if args.debug:
    print("Search pattern:", search_pattern)
    print("Files found:", file_paths)

# Function to write output to file
def write_to_file(file_path, text):
    with open(file_path, 'a') as file:
        file.write(text)

# Function to process FITS data
def process_data(target_fname):
    """Extracts time, message, and PSP_ID from the FITS file."""
    data = fits.open(target_fname)[14].data
    time, msg, psp_id = data["TIME"], data["MSG"], data["PSP_ID"]
    
    # Sort by time
    sortid = np.argsort(time)
    time = time[sortid]
    msg = msg[sortid]
    psp_id = psp_id[sortid]
    
    if args.debug:
        print(f"Number of events: {len(time)} in {target_fname}")
    
    return time, msg, psp_id

# Global list to store unique output strings
outstr_list = []

# Function to search messages in FITS data
def search_in_msg(msg, dtime, psp_id, search_term="psp_task_wfrb_watch"):
    """Search for a specific term in messages and log matches."""
    found = False
    for onemsg, onedtime, onepsp_id in zip(msg, dtime, psp_id):
        if search_term in onemsg:
            outstr = f"{onedtime} PSP_ID= {onepsp_id} msg= {onemsg}"
            
            # Print with color for better visibility
            print(color_text(outstr, "cyan"))
            
            # Store unique messages globally
            outstr_list.append(outstr)
            
            # Mark that we found at least one occurrence
            found = True
    
    if not found and args.debug:
        print(f"No occurrences of '{search_term}' found in messages.")

# Process each file
for onef in file_paths:
    try:
        time, msg, psp_id = process_data(onef)
        
        # Convert time to datetime format
        dtime = [reference_time.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in time]
        
        # Search for specific terms
        search_in_msg(msg, dtime, psp_id)
        search_in_msg(msg, dtime, psp_id, search_term="process_samplrec")        
    except Exception as e:
        print(color_text(f"Error processing {onef}: {e}", "red"))

# Remove duplicates from outstr_list
unique_outstr_list = list(set(outstr_list))

# Write unique messages to file
for outstr in unique_outstr_list:
    write_to_file(args.file_path, outstr + "\n")
