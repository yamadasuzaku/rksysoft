#!/usr/bin/env python 

import argparse
import subprocess

def e2ch(ene):
    """Convert energy to channel."""
    return int(ene/6.0)

def select_event(eventfile, expr, output_suffix=None):
    """
    Select events from an event file using the ftselect command.

    Parameters:
    eventfile (str): Path to the event file.
    expr (str): Selection expression.
    output_suffix (str, optional): Suffix for the output file name. If provided,
                                   the output file name will be eventfile_{output_suffix}.fits.
                                   If not provided, the default suffix is '_cut'.
    """
    outfile_suffix = f"_{output_suffix}" if output_suffix else "_cut"
    outfile = f"{eventfile.rsplit('.', 1)[0]}{outfile_suffix}.evt"

    print(f"Selected event file: {eventfile}")
    print(f"Selection expression: {expr}")
    print(f"Output file: {outfile}")

    try:
        subprocess.run([
            "ftselect",
            f"infile={eventfile}",
            f"outfile={outfile}",
            f"expr={expr}",
            "clobber=yes",
            "chatter=5"
        ], check=True)
        print(f"Output written to {outfile}")
    except subprocess.CalledProcessError as e:
        print(f"Error during ftselect execution: {e}")

# Argument parser setup

parser = argparse.ArgumentParser(
  description='This program is used to perform selection on an event file using the ftselect command.',
  epilog='''
Example:
  python xtend_pileup_genevt_v0.py xa300041010xtd_p0300000a0_uf.evt xa300041010xtd_p0300000a0_cl.evt
  ''',
  formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("ufevt", help="Path to the unfiltered event file.")
parser.add_argument("clevt", help="Path to the cleaned event file.")
args = parser.parse_args()
ufevt = args.ufevt
clevt = args.clevt

# Energy cuts
e1 = e2ch(500.)   # 0.5 keV
e2 = e2ch(3000.)   # 3   keV
e3 = e2ch(7000.)  # 7   keV
e4 = e2ch(10000.)  # 10  keV
pi_filters = {
    "ene1": f"PI>={e1}&&PI<{e2}",
    "ene2": f"PI>={e2}&&PI<{e3}",
    "ene3": f"PI>={e3}&&PI<{e4}",
    "enea": f"PI>={e1}&&PI<{e4}"
}

# Grade cuts
grade_filters = {
    "grade0": "GRADE==0",
    "grade1": "GRADE==1",
    "gradeg": "GRADE==0||GRADE==2||GRADE==3||GRADE==4||GRADE==6"
}

# GTI cut
gti_filters = {
    "clgti": f"gtifilter('{clevt}')"
}

# Generate and apply filters
for k1, pi_filter in pi_filters.items():
    for k2, grade_filter in grade_filters.items():
        for k3, gti_filter in gti_filters.items():
            key = f"{k1}_{k2}_{k3}"
            cutfilter = f"({pi_filter})&&({grade_filter})&&({gti_filter})"
            print(key, cutfilter)
            select_event(ufevt, cutfilter, key)
