#!/usr/bin/env python

from astropy.io import fits
import re

# File name example
file_name = '022625.fits'

# Use regular expressions to capture the different parts of the file name
match = re.match(r'(\d{2})(\d{2})(\d{2})\.fits', file_name)

if match:
    iso_electronic_sequence = match.group(1)
    atomic_number_z = match.group(2)
    spectroscopic_charge = match.group(3)

    # Display the extracted components
    print(f"Iso-electronic sequence: {iso_electronic_sequence}")
    print(f"Atomic number Z: {atomic_number_z}")
    print(f"Spectroscopic charge: {spectroscopic_charge}")
else:
    print("File name does not match the expected pattern.")

# Open the FITS file
with fits.open(file_name) as hdul:
    # Access the LEV and OSC BINTABLE extensions
    lev_data = hdul[1].data  # LEV table (extension 1)
    osc_data = hdul[2].data  # OSC table (extension 2)

    # Iterate through each row of the OSC table
    for row in osc_data:
        il = row['il']  # Corresponds to 1 BINTABLE LEV row number
        iu = row['iu']  # Corresponds to 1 BINTABLE LEV row number

        # Ensure il and iu are within the bounds of LEV table rows
        if 0 <= il < len(lev_data) and 0 <= iu < len(lev_data):
            # Retrieve the corresponding LEV data for il and iu
            lev_data_il = lev_data[il]
            lev_data_iu = lev_data[iu]

            # Print out the data from OSC and corresponding LEV rows
            print(f"OSC row: il = {il}, iu = {iu}, f = {row['f']}, a = {row['a']}, aclass = {row['aclass']}, abib = {row['abib']}")
            print(f"Corresponding LEV data (il): energy = {lev_data_il['energy']}, level = {lev_data_il['level']}")
            print(f"Corresponding LEV data (iu): energy = {lev_data_iu['energy']}, level = {lev_data_iu['level']}")
            print("-" * 80)

