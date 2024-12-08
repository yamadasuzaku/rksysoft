#!/usr/bin/env python 

from astropy.io import fits
import numpy as np
import pandas as pd
import argparse
import os

def load_fits_data(fits_file):
    """
    Load the necessary columns from a FITS file and convert them to a DataFrame.
    
    Parameters:
    - fits_file (str): Path to the FITS file.
    
    Returns:
    - DataFrame: A DataFrame containing the selected columns from the FITS file.
    """
    print(f"Loading data from {fits_file}...")
    with fits.open(fits_file) as hdul:
        data = hdul[1].data  # Extract data from the second HDU (extension)

        # Convert columns to Little-endian format and store them in a DataFrame
        df = pd.DataFrame({
            'TIME': data['TIME'].byteswap().newbyteorder(),
            'ITYPE': data['ITYPE'].byteswap().newbyteorder(),
            'PIXEL': data['PIXEL'].byteswap().newbyteorder(),
            'TRIG_LP': data['TRIG_LP'].byteswap().newbyteorder(),
            'DERIV_MAX': data['DERIV_MAX'].byteswap().newbyteorder()
        })
    print("Data successfully loaded.")
    return df

def find_duplicates_by_pixel(df):
    """
    Identify duplicate 'TIME' rows within each 'PIXEL' group.
    
    Parameters:
    - df (DataFrame): A DataFrame containing event data.
    
    Returns:
    - DataFrame: A DataFrame with duplicated rows based on 'TIME' within each 'PIXEL' group.
    """
    print("Checking for duplicate rows based on the 'TIME' column within each 'PIXEL' group...")
    # Find duplicates within each PIXEL group
    duplicate_df = df.groupby('PIXEL').apply(lambda x: x[x.duplicated('TIME', keep=False)]).reset_index(drop=True)
    print(f"Found {len(duplicate_df)} duplicated rows." if not duplicate_df.empty else "No duplicates found.")
    return duplicate_df

def save_fits_from_dataframe(df, output_file):
    """
    Save a DataFrame as a new FITS file.
    
    Parameters:
    - df (DataFrame): A DataFrame containing the data to save.
    - output_file (str): Path to the output FITS file.
    """
    print(f"Saving duplicated rows to {output_file}...")
    # Create a new FITS table with the DataFrame's columns
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='TIME', format='D', array=df['TIME']),
        fits.Column(name='ITYPE', format='B', array=df['ITYPE']),
        fits.Column(name='PIXEL', format='B', array=df['PIXEL']),
        fits.Column(name='TRIG_LP', format='J', array=df['TRIG_LP']),
        fits.Column(name='DERIV_MAX', format='I', array=df['DERIV_MAX'])
    ])
    hdu.writeto(output_file, overwrite=True)
    print("Duplicated rows saved successfully.")

def analyze_duplicates(duplicate_df):
    """
    Analyze the distribution of the 'ITYPE' column within the duplicated rows,
    grouped by 'PIXEL'.
    
    Parameters:
    - duplicate_df (DataFrame): A DataFrame containing the duplicated rows.
    """
    print("Analyzing the distribution of 'ITYPE' values in duplicated rows by PIXEL...")

    # Group by 'PIXEL' and calculate the distribution of 'ITYPE' within each group
    grouped_distribution = duplicate_df.groupby('PIXEL')['ITYPE'].value_counts()

    for pixel, distribution in grouped_distribution.groupby(level=0):
        print(f"\nPIXEL {pixel}:")
        print(distribution)

def main(fits_file, output_file):
    """
    Main function to handle the process of detecting and analyzing duplicate rows.
    
    Parameters:
    - fits_file (str): Path to the input FITS file.
    - output_file (str): Path to save the duplicated rows as a new FITS file.
    """
    if not os.path.exists(fits_file):
        print(f"Error: {fits_file} not found.")
        return

    # Load data from the FITS file
    df = load_fits_data(fits_file)

    # Find duplicate rows within each PIXEL group based on 'TIME'
    duplicate_df = find_duplicates_by_pixel(df)

    if not duplicate_df.empty:
        # Analyze the distribution of 'ITYPE' values in the duplicated rows
        analyze_duplicates(duplicate_df)

        # Save the duplicated rows to a new FITS file
        save_fits_from_dataframe(duplicate_df, output_file)
    else:
        print("No duplicates to save. Exiting.")

if __name__ == '__main__':
    # Set up argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description='Find and analyze duplicate rows in a FITS file by PIXEL group.')
    parser.add_argument('fits_file', help='Path to the input FITS file')
    parser.add_argument('output_file', help='Path to save the duplicated rows as a FITS file')
    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.fits_file, args.output_file)
