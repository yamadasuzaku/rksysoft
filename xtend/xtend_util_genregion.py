#!/usr/bin/env python

import os
import argparse
from astropy.io import fits

def generate_region_file(c_ra, c_dec, rin, rout, output_dir="."):
    """
    Generate a .reg file for DS9.

    Parameters:
        c_ra (float): Central RA coordinate.
        c_dec (float): Central DEC coordinate.
        rin (int): Inner radius.
        rout (int): Outer radius.
        output_dir (str): Directory to save the output file.
    """
    # Automatically generate the filename
    filename = f"ds9_region_{rin}_{rout}.reg"
    filepath = os.path.join(output_dir, filename)

    # Template for DS9 format
    if rin == 0:
        content = (
            "# Region file format: DS9 version 4.1\n"
            "global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
            "fk5\n"
            f"circle({c_ra},{c_dec},{rout}\")\n"
        )
    else:
        content = (
            "# Region file format: DS9 version 4.1\n"
            "global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
            "fk5\n"
            f"annulus({c_ra},{c_dec},{rin}\",{rout}\")\n"
        )

    # Write to file
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Generated: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate DS9 region files based on FITS file headers.")
    parser.add_argument("fitsfile", help="Input FITS file to extract RA_NOM and DEC_NOM.")
    parser.add_argument("--output-dir", "-o", default="./regions", help="Directory to save region files.")
    parser.add_argument("--inout-list", "-i", nargs="*", default=[(0, 60), (0, 30), (30, 60)], type=lambda s: tuple(map(int, s.split(","))),
                        help="List of (rin,rout) pairs for region generation (e.g., '0,60 0,30 30,60').")
    args = parser.parse_args()

    # Read the FITS file
    fitsfile = args.fitsfile
    hdu = fits.open(fitsfile)[0]
    c_ra = hdu.header["RA_NOM"]
    c_dec = hdu.header["DEC_NOM"]

    # Create the output directory
    output_directory = args.output_dir
    os.makedirs(output_directory, exist_ok=True)

    # Generate files for each rin, rout pair
    for rin, rout in args.inout_list:
        generate_region_file(c_ra, c_dec, rin, rout, output_dir=output_directory)

if __name__ == "__main__":
    main()
