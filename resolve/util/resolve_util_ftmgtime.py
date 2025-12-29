#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
resolve_util_ftmgtime.py

Robust GTI extractor:
- Reads a FITS event file (*.evt or *.evt.gz)
- Finds a GTI-like binary table (prefer EXTNAME="GTI", fallback: first table with START/STOP)
- Writes <basename>.gti next to the input file

Designed to run on:
- older Linux Python (3.7+ recommended; avoids newer-only features)
- newer macOS Python
- environments where HEASOFT variables may or may not be set

Notes:
- Does NOT use dataclasses (to be safe for older envs)
- Avoids shell commands (no os.system rm)
- Uses pathlib if available, but works with os.path as well
"""

import argparse
import os
import sys

from astropy.io import fits
from astropy.table import Table, Column


def eprint(msg):
    sys.stderr.write(str(msg) + "\n")


def normalize_output_basename(input_evt_file):
    """
    xa....evt     -> xa....gti
    xa....evt.gz  -> xa....gti
    other.fits    -> other.gti (fallback)
    """
    base = os.path.basename(input_evt_file)

    # Strip .gz first if present
    if base.endswith(".gz"):
        base = base[:-3]

    # Strip .evt if present
    if base.endswith(".evt"):
        base = base[:-4]
    else:
        # Generic: strip one extension if any
        root, _ext = os.path.splitext(base)
        if root:
            base = root

    outname = base + ".gti"
    return os.path.join(os.path.dirname(os.path.abspath(input_evt_file)), outname)


def is_bintable_hdu(hdu):
    return isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))


def hdu_has_start_stop(hdu):
    try:
        data = hdu.data
        if data is None:
            return False
        names = getattr(data, "names", None)
        if not names:
            return False
        return ("START" in names) and ("STOP" in names)
    except Exception:
        return False


def find_gti_hdu(hdul, prefer_extname="GTI", user_index=None):
    """
    Return (index, hdu) for the GTI HDU.

    Priority:
      1) If user_index is given and valid and has START/STOP -> use it
      2) EXTNAME matches prefer_extname and has START/STOP
      3) First table HDU that has START/STOP
    """
    if user_index is not None:
        if user_index < 0 or user_index >= len(hdul):
            raise IndexError("Requested --gti-hdu-index={} out of range (0..{})".format(
                user_index, len(hdul) - 1
            ))
        hdu = hdul[user_index]
        if not is_bintable_hdu(hdu):
            raise ValueError("HDU[{}] is not a table HDU".format(user_index))
        if not hdu_has_start_stop(hdu):
            raise ValueError("HDU[{}] does not contain START/STOP columns".format(user_index))
        return user_index, hdu

    # Prefer EXTNAME=GTI
    for i, hdu in enumerate(hdul):
        if not is_bintable_hdu(hdu):
            continue
        extname = str(hdu.header.get("EXTNAME", "")).strip()
        if extname.upper() == str(prefer_extname).upper() and hdu_has_start_stop(hdu):
            return i, hdu

    # Fallback: first HDU with START/STOP
    for i, hdu in enumerate(hdul):
        if not is_bintable_hdu(hdu):
            continue
        if hdu_has_start_stop(hdu):
            return i, hdu

    raise RuntimeError("Could not find a GTI-like table HDU with START/STOP columns.")


def build_gti_table(start, stop, units=None):
    units = units or {}
    t = Table()
    t.add_column(Column(data=start, name="START", unit=units.get("START", None)))
    t.add_column(Column(data=stop, name="STOP", unit=units.get("STOP", None)))
    return t


def write_gti_file(output_path, primary_header, gti_header, gti_table, extname="GTI", overwrite=True):
    # Copy headers to avoid mutating originals
    prim_hdr = fits.Header(primary_header)
    gti_hdr = fits.Header(gti_header)

    primary_hdu = fits.PrimaryHDU(header=prim_hdr)
    gti_hdu = fits.BinTableHDU(gti_table, header=gti_hdr, name=extname)

    hdul_out = fits.HDUList([primary_hdu, gti_hdu])
    hdul_out.writeto(output_path, overwrite=overwrite)
    print("------> Created {}".format(output_path))


def extract_gti(input_evt_file, output_gti_file=None, gti_hdu_index=None, prefer_extname="GTI"):
    if not os.path.exists(input_evt_file):
        raise FileNotFoundError("Input file not found: {}".format(input_evt_file))

    if output_gti_file is None:
        output_gti_file = normalize_output_basename(input_evt_file)

    # Units (typical for GTI)
    units = {"START": "s", "STOP": "s"}

    with fits.open(input_evt_file) as hdul:
        primary_header = hdul[0].header

        idx, gti_hdu = find_gti_hdu(hdul, prefer_extname=prefer_extname, user_index=gti_hdu_index)

        gti_header = gti_hdu.header
        data = gti_hdu.data
        start = data["START"]
        stop = data["STOP"]

    gti_table = build_gti_table(start, stop, units=units)

    # Ensure directory exists
    outdir = os.path.dirname(os.path.abspath(output_gti_file))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)

    write_gti_file(
        output_path=output_gti_file,
        primary_header=primary_header,
        gti_header=gti_header,
        gti_table=gti_table,
        extname="GTI",
        overwrite=True,
    )

    return output_gti_file


def build_parser():
    p = argparse.ArgumentParser(
        description="Create a GTI FITS file from an event file (copy START/STOP from a GTI-like HDU)."
    )
    p.add_argument("input_evt_file", help="Input event file (*.evt or *.evt.gz)")
    p.add_argument(
        "-o", "--output",
        dest="output_gti_file",
        default=None,
        help="Output GTI file path (default: <input_basename>.gti next to input)",
    )
    p.add_argument(
        "--gti-hdu-index",
        type=int,
        default=None,
        help="Optional: specify the HDU index to use for GTI (0-based).",
    )
    p.add_argument(
        "--prefer-extname",
        default="GTI",
        help="Prefer this EXTNAME when searching for GTI (default: GTI).",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        out = extract_gti(
            input_evt_file=args.input_evt_file,
            output_gti_file=args.output_gti_file,
            gti_hdu_index=args.gti_hdu_index,
            prefer_extname=args.prefer_extname,
        )
        return 0
    except Exception as exc:
        eprint("ERROR: {}".format(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
