#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy import coordinates, units as u, constants as const
from astropy.coordinates import SkyCoord, FK5

# Set global plotting parameters
plt.rcParams.update({'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 8, 'font.family': 'serif'})

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Calculate the line-of-sight velocity of an astrophysical object.")
    parser.add_argument('-t', '--target', type=str, default="Cygnus X-1", help='Target name used for SkyCoord')
    parser.add_argument('-o', '--orb_file', type=str, default="xa300049010.orb", help='Path to the orbit FITS file')
    parser.add_argument('-d', '--distance', type=float, default=2.2, help='Distance to the target (kpc)')
    return parser.parse_args()

def radec_to_vector(radec, degrees=False):
    """Convert RA/DEC (or LON/LAT) to a 3D unit vector."""
    ra, dec = (np.deg2rad(radec) if degrees else radec)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])

def get_radec_from_xyz(x, y, z):
    """Convert Cartesian coordinates to right ascension and declination."""
    return np.arctan2(y, x), np.arctan2(z, np.sqrt(x**2 + y**2))

def load_orbit_data(orb_file):
    """Load XRISM orbital data from a FITS file."""
    try:
        with fits.open(orb_file) as hdul:
            data = hdul[2].data
        return pd.DataFrame({col: data[col] for col in ["TIME", "x", "y", "z", "vx", "vy", "vz"]})
    except Exception as e:
        print(f"Error loading FITS file: {e}")
        return None

def calculate_los_velocity(df, target_coords):
    """Compute the line-of-sight velocity of XRISM relative to the target."""
    ra_TEME, dec_TEME = get_radec_from_xyz(df["vx"].values, df["vy"].values, df["vz"].values)
    c = SkyCoord(ra_TEME, dec_TEME, unit="rad", frame=FK5, equinox='J2024.0')
    
    vel_XRISM = np.sqrt(df["vx"].values**2 + df["vy"].values**2 + df["vz"].values**2)
    print(f"max of velocity of XRISM = {np.amax(vel_XRISM):.2f}, min={np.amin(vel_XRISM):.2f}")

    v_XRISM = radec_to_vector([c.icrs.ra.deg, c.icrs.dec.deg], degrees=True)
    v_target = radec_to_vector([target_coords.ra.deg, target_coords.dec.deg], degrees=True)
    los_velocity_of_XRISM = np.dot((vel_XRISM * v_XRISM).T, v_target)
    return df["TIME"], los_velocity_of_XRISM

def plot_velocity(time, projected_velocity, df):
    """Plot velocity components and projected velocity."""
    plt.figure(figsize=(12, 6))
    for component in ["vx", "vy", "vz"]:
        plt.plot(df["TIME"], df[component], label=component)
    plt.plot(time, projected_velocity, label="Projected Velocity", c="k")
#    plt.xlim(df["TIME"].values[-200], df["TIME"].values[-1])
    print(f"max of projected_velocity = {np.amax(projected_velocity):.2f}, min={np.amin(projected_velocity):.2f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (km/s)")
    plt.legend()
    plt.title("Line-of-Sight Velocity Projection")
    plt.savefig("rv.png")
    print("rv.png is created.")
    plt.show()

def calc_vr_wrt_LSR(distance, lon, R0=8, Theta0=220*u.km/u.s):
    """Calculate the LSR velocity projected to the target."""
    DoverR0 = distance / R0
    R0overR = 1 / np.sqrt(DoverR0**2 - 2 * DoverR0 * np.cos(np.deg2rad(lon)) + 1)
    return (R0overR - 1) * Theta0 * np.sin(np.deg2rad(lon))

def calc_vr_sun_proper(OBJ_gal):
    """Compute the Sun's motion with respect to LSR projected onto the target direction."""
    sun_dir = SkyCoord("18h03m50.280s +30d00m16.83s", frame='fk5').galactic
    sun_vec = radec_to_vector([sun_dir.l.deg, sun_dir.b.deg], degrees=True)
    return np.dot(20 * u.km / u.s * sun_vec, radec_to_vector(OBJ_gal, degrees=True))

def calc_vr_earth_orbit(OBJ_radec):
    """Compute Earth's motion velocity with respect to the target."""
    v_amplitude = (2 * np.pi / (1.0 * u.yr) * const.au).to("km/s")
    sun = coordinates.get_sun(Time('2024-02-15T00:00:00', format='isot', scale='utc'))
    v_SUN = radec_to_vector([sun.ra.value, sun.dec.value], degrees=True)
    v_NEP = radec_to_vector([270, 66.56], degrees=True)  # North Ecliptic Pole
    v_OBJ = radec_to_vector(OBJ_radec, degrees=True)
    return np.dot(v_amplitude * np.cross(v_SUN, v_NEP), v_OBJ)

def main():
    args = parse_arguments()
    target_coords = SkyCoord.from_name(args.target)
    df = load_orbit_data(args.orb_file)
    if df is None:
        return
    
    time, projected_velocity = calculate_los_velocity(df, target_coords)
    plot_velocity(time, projected_velocity, df)
    
    gal_lon = target_coords.galactic.l.deg
    vr_lsr = calc_vr_wrt_LSR(args.distance, lon=gal_lon)
    vr_sun = calc_vr_sun_proper([target_coords.galactic.l.deg, target_coords.galactic.b.deg])
    vr_earth = calc_vr_earth_orbit([target_coords.ra.deg, target_coords.dec.deg])
    
    print(f"LSR velocity projection: {vr_lsr:.2f}")
    print(f"Solar motion projection: {vr_sun:.2f}")
    print(f"Earth orbit velocity projection: {vr_earth:.2f}")

if __name__ == "__main__":
    main()
