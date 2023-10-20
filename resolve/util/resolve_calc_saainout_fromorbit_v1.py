#!/usr/bin/env python

import numpy as np
import pandas as pd
import astropy.io.fits
from astropy.time import Time
import math
import datetime
import argparse

# Setting up a reference time based on MJD
MJD_REFERENCE_DAY = 58484
reference_time = Time(MJD_REFERENCE_DAY, format='mjd')

def convert_longitude(longitude):
    """Converts longitudes greater than 180 degrees to their equivalent negative values."""
    return longitude - 360 if longitude > 180 else longitude

def calculate_centroid(x, y):
    """Calculates the centroid (average) of a set of points."""
    return sum(x) / len(x), sum(y) / len(y)

def sort_points_based_on_centroid(x, y):
    """Sorts a list of points in counter-clockwise order based on their angles from the centroid."""
    centroid_x, centroid_y = calculate_centroid(x, y)

    def angle_from_centroid(point):
        """Computes the angle between a point and the centroid."""
        return (math.atan2(point[1] - centroid_y, point[0] - centroid_x) + 2 * math.pi) % (2*math.pi)

    sorted_points = sorted(zip(x, y), key=angle_from_centroid)
    sorted_x, sorted_y = zip(*sorted_points)
    return list(sorted_x), list(sorted_y)

def is_point_inside_polygon(x, y, poly_x, poly_y):
    """Determines if a point lies inside a given polygon."""
    num = len(poly_x)
    inside = np.full(x.shape, False, dtype=bool)

    p1x, p1y = poly_x[0], poly_y[0]
    for i in range(1, num + 1):
        p2x, p2y = poly_x[i % num], poly_y[i % num]
        mask = ((y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x)) & (p1y != p2y))
        xinters = (y[mask] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
        inside[mask] = np.logical_xor(inside[mask], x[mask] < xinters)
        p1x, p1y = p2x, p2y

    return inside

def save_to_file(filename, times, datetimes):
    """Saves times and their corresponding datetime values to a file."""
    with open(filename, 'w') as f:
        for t, dt in zip(times, datetimes):
            f.write(f"{t}, {dt}\n")

def process_orbit_data(orbitfile, saa_file):
    """Main function to process the orbit data and determine points inside a polygon."""
    # Loading data from csv and fits files
    df = pd.read_csv(saa_file, header=3, names=["alt", "lon", "lat"])
    data = astropy.io.fits.open(orbitfile)[2].data
    time, lon, lat = data["TIME"], data["LON"], data["LAT"]

    # Convert longitudes to the range [-180, 180]
    clon = np.array([convert_longitude(alon) for alon in lon])
    
    # Sort points based on their centroid
    saa_lon, saa_lat = df["lon"].to_numpy(), df["lat"].to_numpy()
    saa_lon, saa_lat = sort_points_based_on_centroid(saa_lon, saa_lat)

    # Close the polygon by adding the first point at the end
    saa_lon = np.append(saa_lon, saa_lon[0])
    saa_lat = np.append(saa_lat, saa_lat[0])

    # Check which points are inside the polygon
    inside = is_point_inside_polygon(clon, lat, saa_lon, saa_lat)

    tstart, tstop = [], []
    prev_one_inside = inside[0]
    if prev_one_inside:
        tstop.append(time[0])
    else:
        tstart.append(time[0])
    
    # If the state changes (from inside to outside or vice versa), record the time
    for i in range(1, len(inside)):
        if inside[i] != prev_one_inside:
            if inside[i]:
                tstop.append(time[i])
            else:
                tstart.append(time[i])
        prev_one_inside = inside[i]

    # Convert the times to datetime format
    tstart_dtime = [reference_time.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in tstart]
    tstop_dtime = [reference_time.datetime + datetime.timedelta(seconds=float(date_sec)) for date_sec in tstop]

    # Save the results to files
    save_to_file("saa_tstart.txt", tstart, tstart_dtime)
    save_to_file("saa_tstop.txt", tstop, tstop_dtime)

    # print results
    print(" TSTART =============================")
    for t, dt in zip(tstart,tstart_dtime):
        print(t, dt)
    print(" TSTOP ==============================")
    for t, dt in zip(tstop,tstop_dtime):
        print(t, dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process orbit data.')
    parser.add_argument('orbitfile', type=str, help='Path to the orbit file.')
    parser.add_argument('--saa-file', default="saa_sxs.conf.20160324a", type=str, help='Path to the SAA configuration file.')    
    args = parser.parse_args()

    process_orbit_data(args.orbitfile, args.saa_file)
