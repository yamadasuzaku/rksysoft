#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
heates_scan_hdf5_to_root.py

Purpose
-------
- Ensure that the script runs inside a ROOT-capable conda environment.
- Scan HDF5 files produced by the uMUX pipeline.
- Check whether the corresponding ROOT files already exist in:
    /home/heates107/output/umux/root
- If not, convert each HDF5 file into a ROOT TTree.
- Optionally keep watching the directory tree at a fixed interval.

Important update
----------------
- The ROOT branch schema is no longer determined from the first channel group.
- Instead, the script inspects all valid channel groups and computes the
  intersection of usable dataset keys.
- This avoids schema corruption when the first channel (for example chan0)
  is missing datasets such as 'dt_us' or 'beam'.

Typical usage
-------------
    python heates_scan_hdf5_to_root.py
    python heates_scan_hdf5_to_root.py --watch
    python heates_scan_hdf5_to_root.py --watch --interval 60
    python heates_scan_hdf5_to_root.py --attrs timestamp energy rise_time postpeak_deriv
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
        print(
            "[ERROR] root_env Python was not found: {0}".format(ROOT_ENV_PYTHON),
            file=sys.stderr
        )
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
# Helper functions for HDF5 -> ROOT conversion
# ----------------------------------------------------------------------
def get_root_tag_from_dtype(dtype):
    """
    Convert a NumPy dtype into a ROOT leaf tag.

    Supported mappings
    ------------------
    int16   -> /S
    uint16  -> /s
    int32   -> /I
    int64   -> /L
    float32 -> /F
    float64 -> /D
    bool    -> /O
    """
    dtype = np.dtype(dtype)

    if dtype == np.dtype(np.int16):
        return "/S"
    if dtype == np.dtype(np.uint16):
        return "/s"
    if dtype == np.dtype(np.int32):
        return "/I"
    if dtype == np.dtype(np.int64):
        return "/L"
    if dtype == np.dtype(np.float32):
        return "/F"
    if dtype == np.dtype(np.float64):
        return "/D"
    if dtype == np.dtype(np.bool_):
        return "/O"

    return None


def get_channel_number_from_group_name(group_name):
    """
    Extract channel number from a group name like 'chan37'.

    Returns
    -------
    int
        Extracted channel number, or -1 if not found.
    """
    if re.search(r"\d", group_name):
        return int(re.sub(r"\D", "", group_name))
    return -1


def collect_common_dataset_schema(hf, attrs=None):
    """
    Inspect all channel groups and determine the common dataset schema.

    Policy
    ------
    - Use only groups that contain a non-empty 'timestamp' dataset.
    - Ignore metadata-like keys such as 'cuts', 'calibration', 'filters'.
    - Compute the intersection of usable dataset names across valid groups.
    - Keep only 1D datasets whose size matches timestamp size in each group.
    - Keep only datasets whose dtype is supported by ROOT leaf tags.
    - If attrs is specified, restrict the final intersection to attrs.

    Returns
    -------
    common_keys : list[str]
        Sorted list of dataset names common to all valid groups.
    dtype_map : dict[str, np.dtype]
        Representative dtype for each common key.
    valid_groups : list[str]
        Group names used in the schema decision.
    """
    remove_names = set(["cuts", "calibration", "filters"])
    common_keys = None
    dtype_map = {}
    incompatible_keys = set()
    valid_groups = []

    for group_name, ds in hf.items():
        if "timestamp" not in ds:
            print(
                "[WARN] group '{0}' does not contain 'timestamp'; excluded from schema."
                .format(group_name)
            )
            continue

        try:
            ref_array = ds["timestamp"][:]
        except Exception as exc:
            print(
                "[WARN] failed to read timestamp in group '{0}': {1}"
                .format(group_name, exc)
            )
            continue

        if ref_array.size == 0:
            print(
                "[WARN] group '{0}' has empty timestamp; excluded from schema."
                .format(group_name)
            )
            continue

        refsize = ref_array.size
        usable_keys = set()

        for leaf_name, leaf in ds.items():
            if leaf_name in remove_names:
                continue

            if attrs is not None and leaf_name not in attrs:
                continue

            try:
                arr = leaf[:]
            except Exception as exc:
                print(
                    "[WARN] failed to read dataset {0}/{1}: {2}"
                    .format(group_name, leaf_name, exc)
                )
                continue

            if not hasattr(arr, "shape"):
                continue

            if len(arr.shape) != 1:
                continue

            if arr.size != refsize:
                continue

            if arr.size == 0:
                continue

            tag = get_root_tag_from_dtype(arr.dtype)
            if tag is None:
                print(
                    "[WARN] unsupported dtype in schema scan: {0}/{1}, dtype={2}"
                    .format(group_name, leaf_name, arr.dtype)
                )
                continue

            if leaf_name in incompatible_keys:
                continue

            usable_keys.add(leaf_name)

            if leaf_name not in dtype_map:
                dtype_map[leaf_name] = np.dtype(arr.dtype)
            else:
                if np.dtype(dtype_map[leaf_name]) != np.dtype(arr.dtype):
                    print(
                        "[WARN] dtype mismatch for key '{0}': existing={1}, "
                        "group '{2}' has {3}. Excluding this key from schema."
                        .format(leaf_name, dtype_map[leaf_name], group_name, arr.dtype)
                    )
                    incompatible_keys.add(leaf_name)
                    if leaf_name in usable_keys:
                        usable_keys.discard(leaf_name)
                    if leaf_name in dtype_map:
                        del dtype_map[leaf_name]

        if common_keys is None:
            common_keys = usable_keys
        else:
            common_keys &= usable_keys

        valid_groups.append(group_name)

    if common_keys is None:
        common_keys = set()

    common_keys -= incompatible_keys
    common_keys = sorted(common_keys)

    # Keep dtype_map only for the final common schema
    dtype_map = dict(
        (key, np.dtype(dtype_map[key]))
        for key in common_keys
        if key in dtype_map
    )

    print("[INFO] valid groups used for schema = {0}".format(valid_groups))
    print("[INFO] common dataset keys         = {0}".format(common_keys))

    return common_keys, dtype_map, valid_groups


# ----------------------------------------------------------------------
# HDF5 -> ROOT conversion
# ----------------------------------------------------------------------
def mktree(hf, attrs=None):
    """
    Build a ROOT TTree from an opened HDF5 file object.

    New policy
    ----------
    - The ROOT schema is NOT determined by the first channel.
    - Instead, it is determined from the intersection of usable dataset keys
      across all valid channel groups.
    - This avoids the problem that an unstable chan0 can remove branches
      such as 'dt_us' or 'beam' from the whole TTree.
    """
    print("[INFO] start making tree for attrs = {0}".format(attrs))
    tree = ROOT.TTree("tree", "tree")

    common_keys, dtype_map, valid_groups = collect_common_dataset_schema(
        hf,
        attrs=attrs
    )

    if len(valid_groups) == 0:
        raise RuntimeError("No valid groups were found in the HDF5 file.")

    vals = {}

    # Always create channum
    vals["channum"] = np.zeros(1, dtype=np.int32)
    tree.Branch("channum", vals["channum"], "channum/I")

    # Create branches only from the common schema
    for leaf_name in common_keys:
        dtype = dtype_map[leaf_name]
        tag = get_root_tag_from_dtype(dtype)
        if tag is None:
            print(
                "[WARN] skip unsupported common key '{0}', dtype={1}"
                .format(leaf_name, dtype)
            )
            continue

        vals[leaf_name] = np.zeros(1, dtype=dtype)
        tree.Branch(leaf_name, vals[leaf_name], leaf_name + tag)

    # Fill events group by group
    for group_name in valid_groups:
        ds = hf[group_name]
        print("[INFO] processing group: {0}".format(group_name), flush=True)

        ref_array = ds["timestamp"][:]
        refsize = ref_array.size

        if refsize == 0:
            print(
                "[WARN] group '{0}' became empty during fill; skipping."
                .format(group_name)
            )
            continue

        data = {}
        data["channum"] = np.full(
            refsize,
            get_channel_number_from_group_name(group_name),
            dtype=np.int32
        )

        fillable = True
        for leaf_name in common_keys:
            if leaf_name not in ds:
                print(
                    "[WARN] common key '{0}' missing in group '{1}' during fill; "
                    "skipping group."
                    .format(leaf_name, group_name)
                )
                fillable = False
                break

            try:
                arr = ds[leaf_name][:]
            except Exception as exc:
                print(
                    "[WARN] failed to read dataset {0}/{1} during fill: {2}"
                    .format(group_name, leaf_name, exc)
                )
                fillable = False
                break

            if not hasattr(arr, "shape") or len(arr.shape) != 1:
                print(
                    "[WARN] dataset {0}/{1} is not 1D during fill; skipping group."
                    .format(group_name, leaf_name)
                )
                fillable = False
                break

            if arr.size != refsize:
                print(
                    "[WARN] dataset {0}/{1} size mismatch during fill: {2} != {3}; "
                    "skipping group."
                    .format(group_name, leaf_name, arr.size, refsize)
                )
                fillable = False
                break

            if np.dtype(arr.dtype) != np.dtype(dtype_map[leaf_name]):
                print(
                    "[WARN] dataset {0}/{1} dtype mismatch during fill: {2} != {3}; "
                    "skipping group."
                    .format(group_name, leaf_name, arr.dtype, dtype_map[leaf_name])
                )
                fillable = False
                break

            data[leaf_name] = arr

        if not fillable:
            continue

        for j in range(refsize):
            vals["channum"][0] = data["channum"][j]
            for leaf_name in common_keys:
                vals[leaf_name][0] = data[leaf_name][j]
            tree.Fill()

    print("[INFO] mktree done")
    print("[INFO] tree entries = {0}".format(tree.GetEntries()))
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
        print(
            "[INFO] done: found={0}, converted={1}, skipped={2}"
            .format(found, converted, skipped)
        )
        return

    while True:
        print("")
        print(
            "[INFO] watch scan started at {0}"
            .format(time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        found, converted, skipped = scan_and_convert(
            search_glob=args.glob,
            attrs=args.attrs,
            force=args.force,
            stable_check=(not args.no_stable_check),
        )
        print(
            "[INFO] watch cycle done: found={0}, converted={1}, skipped={2}"
            .format(found, converted, skipped)
        )
        print("[INFO] sleep {0} sec".format(args.interval))
        time.sleep(args.interval)


if __name__ == "__main__":
    main()