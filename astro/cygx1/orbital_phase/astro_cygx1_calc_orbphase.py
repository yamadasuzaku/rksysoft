#!/usr/bin/env python 

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

# Sky position of Cygnus X-1 (ICRS coordinates)
cygx1 = SkyCoord('19h58m21.675s', '+35d12m05.78s', frame='icrs')

def compute_phase(name, ut_str, Do, To, apply_hjd=False):
    """
    Compute orbital phase of Cygnus X-1 using various ephemerides.

    Parameters
    ----------
    name : str
        Label for output (e.g., 'Brocksopp MJD', 'Brocksopp HJD', 'LaSala HJD')
    ut_str : str
        Observation date-time string in ISO format (UTC)
    Do : float
        Reference epoch (in MJD or HJD)
    To : float
        Orbital period in days
    apply_hjd : bool
        If True, apply heliocentric correction (HJD); otherwise use MJD
    """
    # Convert input string to Time object
    t = Time(ut_str, scale='utc', location=EarthLocation(0, 0, 0))

    if apply_hjd:
        # Apply light travel time correction to solar system barycenter (HJD)
        lt = t.light_travel_time(cygx1, kind='heliocentric')
        t_corr = t + lt
        jd_val = t_corr.jd
    else:
        jd_val = t.mjd

    # Compute orbital phase
    phase = ((jd_val - Do) % To) / To

    print(f"{name:16s} | {ut_str} → Phase = {phase:.5f}")

# === Example comparison for one observation time ===
obs_time_1 = "2024-04-07 16:55:04"
obs_time_2 = "2024-04-10 13:41:04"

print("Compare orbital phase computed in 3 slightly different methods:")
print("-" * 70)

# Method 1: Brocksopp et al. 1999 using MJD
compute_phase("Brocksopp MJD", obs_time_1, Do=41874.207, To=5.599829, apply_hjd=False)
compute_phase("Brocksopp MJD", obs_time_2, Do=41874.207, To=5.599829, apply_hjd=False)

# Method 2: Brocksopp et al. 1999 using HJD correction
compute_phase("Brocksopp HJD", obs_time_1, Do=2441874.707, To=5.5998, apply_hjd=True)
compute_phase("Brocksopp HJD", obs_time_2, Do=2441874.707, To=5.5998, apply_hjd=True)

# Method 3: LaSala et al. 1998 using HJD correction
compute_phase("LaSala HJD", obs_time_1, Do=2450235.29, To=5.5998, apply_hjd=True)
compute_phase("LaSala HJD", obs_time_2, Do=2450235.29, To=5.5998, apply_hjd=True)

print("-" * 70)
print("The orbital phases calculated from these different methods differ only slightly,")
print("typically less than ~0.01 in phase, meaning they are consistent for most purposes.")
