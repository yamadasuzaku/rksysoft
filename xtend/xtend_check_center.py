import argparse
import numpy as np
from astropy.io import fits

def calculate_centroid(image_data):
    # Create a grid of X, Y coordinates
    y, x = np.indices(image_data.shape)
    
    # Sum of pixel values
    total = image_data.sum()
    
    # Calculate the centroid coordinates
    x_center = (x * image_data).sum() / total
    y_center = (y * image_data).sum() / total
    
    return x_center, y_center

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Calculate the centroid of a FITS image")
    
    # Add the filename argument
    parser.add_argument('fits_file', type=str, help="Path to the FITS file")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Open the FITS file
    with fits.open(args.fits_file) as hdul:
        image_data = hdul[0].data

    # Calculate the centroid coordinates
    x_centroid, y_centroid = calculate_centroid(image_data)

    print(f"Centroid coordinates: (X: {x_centroid}, Y: {y_centroid})")

if __name__ == "__main__":
    main()
