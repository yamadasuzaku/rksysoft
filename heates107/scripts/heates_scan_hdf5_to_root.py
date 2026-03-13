#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_hdf5_to_root.py

Purpose
-------
- Ensure that the script runs inside a ROOT-capable conda environment.
- Scan HDF5 files produced by the uMUX pipeline.
- Check whether the corresponding ROOT files already exist in:
    /home/heates107/output/umux/root
- If not, convert each HDF5 file into a ROOT TTree.
- Optionally keep watching the directory tree at a fixed interval.

Typical usage
-------------
    python scan_hdf5_to_root.py
    python scan_hdf5_to_root.py --watch
    python scan_hdf5_to_root.py --watch --interval 60
    python scan_hdf5_to_root.py --attrs timestamp energy rise_time postpeak_deriv
"""

from __future__ import print_function

import argparse
import glob
import os
import re
import sys
import time

try:
    from typing import Tuple, Optional, List
except ImportError:
    # Fallback for very old Python environments
    Tuple = tuple
    Optional = object
    List = list


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
ROOT_ENV_NAME = "root_env"
ROOT_ENV_PYTHON = "/home/heates107/miniforge3/envs/root_env/bin/python"
DEFAULT_SEARCH_GLOB = "/home/heates107/output/umux/202603*_run*/202603*_run*_mass.hdf5"
ROOT_OUTPUT_DIR = "/home/heates107/output/umux/root"

# These modules are imported only after relaunching inside root_env.
np = None
h5py = None
ROOT = None

TYPELIST = None
TAGLIST = None


# ----------------------------------------------------------------------
# Environment handling
# ----------------------------------------------------------------------
def ensure_root_environment():
    """
    Relaunch this script with the Python interpreter inside root_env
    unless it is already running there.

    This is more reliable than trying to run 'conda activate' inside
    a Python process.
    """
    current_prefix = os.environ.get("CONDA_DEFAULT_ENV", "")
    current_python = sys.executable

    if current_prefix == ROOT_ENV_NAME:
        return

    if current_python.startswith("/home/heates107/miniforge3/envs/root_env/"):
        return

    if not os.path.exists(ROOT_ENV_PYTHON):
        print("[ERROR] root_env Python was not found: {0}".format(ROOT_ENV_PYTHON), file=sys.stderr)
        sys.exit(1)

    print("[INFO] Not running inside root_env; relaunching with ROOT-capable Python.")
    print("[INFO] current python = {0}".format(current_python))
    print("[INFO] target  python = {0}".format(ROOT_ENV_PYTHON))

    os.execv(ROOT_ENV_PYTHON, [ROOT_ENV_PYTHON] + sys.argv)


def import_runtime_modules():
    """
    Import runtime modules only after switching to the ROOT-capable environment.
    """
    global np, h5py, ROOT, TYPELIST, TAGLIST

    import numpy as _np
    import h5py as _h5py
    import ROOT as _ROOT

    np = _np
    h5py = _h5py
    ROOT = _ROOT

    ROOT.gROOT.SetBatch(1)

    TYPELIST = [
        np.int16,
        np.uint16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        np.bool_,
    ]
    TAGLIST = [
        "/S",
        "/s",
        "/I",
        "/L",
        "/F",
        "/D",
        "/O",
    ]


# ----------------------------------------------------------------------
# HDF5 -> ROOT conversion
# ----------------------------------------------------------------------
def mktree(hf, attrs=None):
    """
    Build a ROOT TTree from an opened HDF5 file object.

    Notes
    -----
    - The implementation largely follows the original logic.
    - 'channum' is stored as int32 to match ROOT branch type '/I'.
    - Empty datasets are skipped safely.
    - Some additional warning messages are included for debugging.
    """
    print("[INFO] start making tree for attrs = {0}".format(attrs))
    tree = ROOT.TTree("tree", "tree")

    remove_names = ["cuts", "calibration", "filters"]
    vals = {}

    for i, (group_name, ds) in enumerate(hf.items()):
        print("[INFO] processing group: {0}".format(group_name), flush=True)

        if "timestamp" not in ds:
            print("[WARN] group '{0}' does not contain 'timestamp'; skipping.".format(group_name))
            continue

        ref_array = ds["timestamp"][:]
        refsize = ref_array.size
        reftype = type(ref_array)

        if refsize == 0:
            print("[WARN] group '{0}' has an empty 'timestamp' dataset; skipping.".format(group_name))
            continue

        data = {}

        if re.search(r"\d", group_name):
            channum = int(re.sub(r"\D", "", group_name))
        else:
            channum = -1

        data["channum"] = channum * np.ones(refsize, dtype=np.int32)

        for leaf_name, leaf in ds.items():
            if leaf_name in remove_names:
                continue

            if attrs is not None and leaf_name not in attrs:
                continue

            try:
                arr = leaf[:]
            except Exception as exc:
                print("[WARN] failed to read dataset {0}/{1}: {2}".format(group_name, leaf_name, exc))
                continue

            if not isinstance(arr, reftype):
                continue

            if arr.size != refsize:
                continue

            if arr.size == 0:
                continue

            registered = False
            for t, tag in zip(TYPELIST, TAGLIST):
                if isinstance(arr[0], t):
                    if i == 0 and leaf_name not in vals:
                        vals[leaf_name] = np.zeros(1, dtype=t)
                        tree.Branch(leaf_name, vals[leaf_name], leaf_name + tag)
                    data[leaf_name] = arr
                    registered = True
                    break

            if not registered:
                print("[WARN] unsupported type: {0}, type={1}".format(leaf_name, type(arr[0])))

        if i == 0 and "channum" not in vals:
            vals["channum"] = np.zeros(1, dtype=np.int32)
            tree.Branch("channum", vals["channum"], "channum/I")

        if "timestamp" not in data:
            print("[WARN] group '{0}' did not produce a usable 'timestamp'; skipping.".format(group_name))
            continue

        for j in range(data["timestamp"].size):
            for name, aaa in vals.items():
                if name in data:
                    aaa[0] = data[name][j]
                else:
                    if aaa.dtype == np.bool_:
                        aaa[0] = False
                    else:
                        aaa[0] = -1
            tree.Fill()

    print("[INFO] mktree done")
    return tree


def dump_hdf5_to_tree(hdf_filename, root_filename, attrs=None):
    """
    Convert one HDF5 file into one ROOT file containing a TTree.
    """
    print("")
    print("[INFO] start dumping to ROOT tree")
    print("[INFO] [hdf5] {0}".format(hdf_filename))
    print("[INFO] [root] {0}".format(root_filename))

    root_dir = os.path.dirname(root_filename)
    if root_dir and not os.path.exists(root_dir):
        os.makedirs(root_dir)

    hf = None
    rf = None

    try:
        hf = h5py.File(hdf_filename, "r")
        rf = ROOT.TFile(str(root_filename), "RECREATE")

        tree = mktree(hf, attrs=attrs)
        tree.Write()
        rf.Close()

        print("[INFO] finished")

    finally:
        try:
            if hf is not None:
                hf.close()
        except Exception:
            pass

        try:
            if rf is not None and rf.IsOpen():
                rf.Close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# File naming and file stability check
# ----------------------------------------------------------------------
def root_filename_from_hdf5(hdf5_path):
    """
    Map:
        /path/to/20260304_run0004_mass.hdf5
    to:
        /home/heates107/output/umux/root/20260304_run0004_mass.root
    """
    base = os.path.basename(hdf5_path)
    if base.endswith(".hdf5"):
        base = base[:-5] + ".root"
    else:
        base = base + ".root"
    return os.path.join(ROOT_OUTPUT_DIR, base)


def is_file_stable(filepath, wait_seconds=3):
    """
    Check whether the file size remains unchanged for a short period.

    This helps avoid converting a file that is still being written.
    """
    try:
        size1 = os.path.getsize(filepath)
        time.sleep(wait_seconds)
        size2 = os.path.getsize(filepath)
        return size1 == size2
    except OSError:
        return False


# ----------------------------------------------------------------------
# Scan and convert
# ----------------------------------------------------------------------
def scan_and_convert(search_glob, attrs=None, force=False, stable_check=True):
    """
    Scan HDF5 files matching the glob pattern and create missing ROOT files.

    Returns
    -------
    (found, converted, skipped)
    """
    hdf5_files = sorted(glob.glob(search_glob))
    found = len(hdf5_files)
    converted = 0
    skipped = 0

    print("[INFO] scan glob = {0}".format(search_glob))
    print("[INFO] found {0} hdf5 files".format(found))
    print("[INFO] root output dir = {0}".format(ROOT_OUTPUT_DIR))

    if not os.path.exists(ROOT_OUTPUT_DIR):
        os.makedirs(ROOT_OUTPUT_DIR)

    for hdf5_path in hdf5_files:
        root_path = root_filename_from_hdf5(hdf5_path)

        if os.path.exists(root_path) and not force:
            print("[SKIP] already exists: {0}".format(root_path))
            skipped += 1
            continue

        if stable_check and not is_file_stable(hdf5_path, wait_seconds=2):
            print("[SKIP] file still changing: {0}".format(hdf5_path))
            skipped += 1
            continue

        try:
            dump_hdf5_to_tree(hdf5_path, root_path, attrs=attrs)
            converted += 1
        except Exception as exc:
            print("[ERROR] failed to convert: {0}".format(hdf5_path))
            print("[ERROR] {0}".format(exc))

    return found, converted, skipped


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan *_mass.hdf5 files and create corresponding ROOT files if missing."
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_SEARCH_GLOB,
        help="Search glob for HDF5 files (default: {0})".format(DEFAULT_SEARCH_GLOB),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep watching periodically.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Watch interval in seconds (default: 60).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ROOT files.",
    )
    parser.add_argument(
        "--no-stable-check",
        action="store_true",
        help="Disable the file-size stability check.",
    )
    parser.add_argument(
        "--attrs",
        nargs="*",
        default=None,
        help="Branch names to keep (default: keep all supported datasets).",
    )
    return parser.parse_args()


def main():
    ensure_root_environment()
    import_runtime_modules()

    args = parse_args()

    print("[INFO] --------------------------------------------------")
    print("[INFO] python            = {0}".format(sys.executable))
    print("[INFO] CONDA_DEFAULT_ENV = {0}".format(os.environ.get("CONDA_DEFAULT_ENV", "")))
    print("[INFO] ROOT version      = {0}".format(ROOT.gROOT.GetVersion()))
    print("[INFO] search glob       = {0}".format(args.glob))
    print("[INFO] root output dir   = {0}".format(ROOT_OUTPUT_DIR))
    print("[INFO] watch mode        = {0}".format(args.watch))
    print("[INFO] interval [sec]    = {0}".format(args.interval))
    print("[INFO] attrs             = {0}".format(args.attrs))
    print("[INFO] --------------------------------------------------")

    if not args.watch:
        found, converted, skipped = scan_and_convert(
            search_glob=args.glob,
            attrs=args.attrs,
            force=args.force,
            stable_check=(not args.no_stable_check),
        )
        print("[INFO] done: found={0}, converted={1}, skipped={2}".format(found, converted, skipped))
        return

    while True:
        print("")
        print("[INFO] watch scan started at {0}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        found, converted, skipped = scan_and_convert(
            search_glob=args.glob,
            attrs=args.attrs,
            force=args.force,
            stable_check=(not args.no_stable_check),
        )
        print("[INFO] watch cycle done: found={0}, converted={1}, skipped={2}".format(found, converted, skipped))
        print("[INFO] sleep {0} sec".format(args.interval))
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
