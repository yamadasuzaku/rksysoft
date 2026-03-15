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

Updated schema policy
---------------------
- The ROOT branch schema is no longer determined by the first channel group.
- Instead, the script inspects all valid channel groups and finds the group(s)
  with the largest number of usable event-wise dataset keys.
- That "max-key schema" is treated as the correct schema.
- If a bad pixel group is missing some datasets, those branches are filled with
  dtype-dependent default values instead of removing the branches globally.

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
    """
    if re.search(r"\d", group_name):
        return int(re.sub(r"\D", "", group_name))
    return -1


def get_default_value_for_dtype(dtype):
    """
    Return a dtype-compatible default scalar value.
    """
    dtype = np.dtype(dtype)

    if dtype == np.dtype(np.bool_):
        return False
    if np.issubdtype(dtype, np.integer):
        return 0
    if np.issubdtype(dtype, np.floating):
        return 0.0

    return 0


def is_usable_event_dataset(ds, leaf_name, leaf_obj, attrs=None):
    """
    Determine whether a dataset is a usable event-wise 1D dataset.

    Conditions
    ----------
    - Exclude metadata-like names
    - If attrs is specified, keep only names in attrs
    - Must be readable
    - Must be 1D
    - Must have the same length as timestamp
    - Must have a dtype supported by ROOT leaf tags
    """
    remove_names = set(["cuts", "calibration", "filters"])

    if leaf_name in remove_names:
        return False, None, None

    if attrs is not None and leaf_name not in attrs:
        return False, None, None

    if "timestamp" not in ds:
        return False, None, None

    try:
        ref_array = ds["timestamp"][:]
    except Exception:
        return False, None, None

    if ref_array.size == 0:
        return False, None, None

    try:
        arr = leaf_obj[:]
    except Exception:
        return False, None, None

    if not hasattr(arr, "shape"):
        return False, None, None

    if len(arr.shape) != 1:
        return False, None, None

    if arr.size != ref_array.size:
        return False, None, None

    if arr.size == 0:
        return False, None, None

    tag = get_root_tag_from_dtype(arr.dtype)
    if tag is None:
        return False, None, None

    return True, np.dtype(arr.dtype), arr.size


def collect_group_usable_schema(ds, attrs=None):
    """
    Collect usable dataset keys for one HDF5 group.

    Returns
    -------
    usable_keys : set[str]
    dtype_map : dict[str, np.dtype]
    refsize : int or None
    """
    usable_keys = set()
    dtype_map = {}

    if "timestamp" not in ds:
        return usable_keys, dtype_map, None

    try:
        ref_array = ds["timestamp"][:]
    except Exception:
        return usable_keys, dtype_map, None

    if ref_array.size == 0:
        return usable_keys, dtype_map, None

    refsize = ref_array.size

    for leaf_name, leaf_obj in ds.items():
        ok, dtype, _ = is_usable_event_dataset(ds, leaf_name, leaf_obj, attrs=attrs)
        if not ok:
            continue
        usable_keys.add(leaf_name)
        dtype_map[leaf_name] = dtype

    return usable_keys, dtype_map, refsize


def collect_max_schema(hf, attrs=None):
    """
    Determine the schema from the group(s) with the maximum number of usable keys.

    Policy
    ------
    - Inspect all channel groups
    - Count usable event-wise keys for each group
    - Find the maximum count
    - Use the union of keys among the max-count groups as the schema
    - Require dtype consistency across the selected max-count groups
    - Later, groups missing some schema keys are still filled, using defaults

    Returns
    -------
    schema_keys : list[str]
        Sorted list of schema keys
    dtype_map : dict[str, np.dtype]
        Dtype for each schema key
    valid_groups : list[str]
        All groups with non-empty timestamp
    template_groups : list[str]
        Groups that had the maximum number of usable keys
    """
    valid_groups = []
    group_key_map = {}
    group_dtype_map = {}
    key_count_map = {}

    for group_name, ds in hf.items():
        usable_keys, dtype_map, refsize = collect_group_usable_schema(ds, attrs=attrs)

        if refsize is None:
            print(
                "[WARN] group '{0}' does not have a valid non-empty timestamp; excluded."
                .format(group_name)
            )
            continue

        valid_groups.append(group_name)
        group_key_map[group_name] = usable_keys
        group_dtype_map[group_name] = dtype_map
        key_count_map[group_name] = len(usable_keys)

        print(
            "[INFO] group '{0}' usable key count = {1}"
            .format(group_name, len(usable_keys))
        )

    if len(valid_groups) == 0:
        raise RuntimeError("No valid groups were found in the HDF5 file.")

    max_count = max(key_count_map.values())
    template_groups = [
        group_name for group_name in valid_groups
        if key_count_map[group_name] == max_count
    ]

    print("[INFO] maximum usable key count = {0}".format(max_count))
    print("[INFO] template groups         = {0}".format(template_groups))

    # Use union across max-count groups
    schema_key_set = set()
    for group_name in template_groups:
        schema_key_set |= group_key_map[group_name]

    # Resolve dtype consistency across template groups
    dtype_map = {}
    final_schema_keys = []

    for key_name in sorted(schema_key_set):
        seen_dtypes = []
        for group_name in template_groups:
            if key_name in group_dtype_map[group_name]:
                seen_dtypes.append(np.dtype(group_dtype_map[group_name][key_name]))

        if len(seen_dtypes) == 0:
            continue

        first_dtype = seen_dtypes[0]
        if all(np.dtype(dt) == first_dtype for dt in seen_dtypes):
            dtype_map[key_name] = first_dtype
            final_schema_keys.append(key_name)
        else:
            print(
                "[WARN] dtype mismatch for key '{0}' across template groups; "
                "excluding from schema. dtypes={1}"
                .format(key_name, seen_dtypes)
            )

    print("[INFO] final schema keys       = {0}".format(final_schema_keys))

    return final_schema_keys, dtype_map, valid_groups, template_groups


# ----------------------------------------------------------------------
# HDF5 -> ROOT conversion
# ----------------------------------------------------------------------
def mktree(hf, attrs=None):
    """
    Build a ROOT TTree from an opened HDF5 file object.

    New policy
    ----------
    - The schema is determined from the group(s) with the largest number of
      usable event-wise dataset keys.
    - Missing datasets in incomplete groups are filled with dtype-dependent
      default values instead of removing the branch from the whole TTree.
    """
    print("[INFO] start making tree for attrs = {0}".format(attrs))
    tree = ROOT.TTree("tree", "tree")

    schema_keys, dtype_map, valid_groups, template_groups = collect_max_schema(
        hf,
        attrs=attrs
    )

    vals = {}

    # Always create channum
    vals["channum"] = np.zeros(1, dtype=np.int32)
    tree.Branch("channum", vals["channum"], "channum/I")

    # Create branches from the max-key schema
    for leaf_name in schema_keys:
        dtype = dtype_map[leaf_name]
        tag = get_root_tag_from_dtype(dtype)
        if tag is None:
            print(
                "[WARN] skip unsupported schema key '{0}', dtype={1}"
                .format(leaf_name, dtype)
            )
            continue

        vals[leaf_name] = np.zeros(1, dtype=dtype)
        tree.Branch(leaf_name, vals[leaf_name], leaf_name + tag)

    # Fill event data for every valid group
    for group_name in valid_groups:
        ds = hf[group_name]
        print("[INFO] processing group: {0}".format(group_name), flush=True)

        try:
            timestamp = ds["timestamp"][:]
        except Exception as exc:
            print(
                "[WARN] failed to read timestamp for group '{0}': {1}; skipping."
                .format(group_name, exc)
            )
            continue

        refsize = timestamp.size
        if refsize == 0:
            print(
                "[WARN] group '{0}' has empty timestamp during fill; skipping."
                .format(group_name)
            )
            continue

        data = {}
        data["channum"] = np.full(
            refsize,
            get_channel_number_from_group_name(group_name),
            dtype=np.int32
        )

        # Read schema keys when available; otherwise fill with defaults
        for leaf_name in schema_keys:
            expected_dtype = np.dtype(dtype_map[leaf_name])
            default_value = get_default_value_for_dtype(expected_dtype)

            if leaf_name not in ds:
                print(
                    "[WARN] group '{0}' missing '{1}'; filling default value {2}."
                    .format(group_name, leaf_name, default_value)
                )
                data[leaf_name] = np.full(refsize, default_value, dtype=expected_dtype)
                continue

            try:
                arr = ds[leaf_name][:]
            except Exception as exc:
                print(
                    "[WARN] failed to read {0}/{1}: {2}; filling defaults."
                    .format(group_name, leaf_name, exc)
                )
                data[leaf_name] = np.full(refsize, default_value, dtype=expected_dtype)
                continue

            if not hasattr(arr, "shape") or len(arr.shape) != 1:
                print(
                    "[WARN] dataset {0}/{1} is not 1D; filling defaults."
                    .format(group_name, leaf_name)
                )
                data[leaf_name] = np.full(refsize, default_value, dtype=expected_dtype)
                continue

            if arr.size != refsize:
                print(
                    "[WARN] dataset {0}/{1} size mismatch ({2} != {3}); filling defaults."
                    .format(group_name, leaf_name, arr.size, refsize)
                )
                data[leaf_name] = np.full(refsize, default_value, dtype=expected_dtype)
                continue

            if np.dtype(arr.dtype) != expected_dtype:
                print(
                    "[WARN] dataset {0}/{1} dtype mismatch ({2} != {3}); casting if possible."
                    .format(group_name, leaf_name, arr.dtype, expected_dtype)
                )
                try:
                    arr = arr.astype(expected_dtype)
                except Exception:
                    print(
                        "[WARN] cast failed for {0}/{1}; filling defaults."
                        .format(group_name, leaf_name)
                    )
                    arr = np.full(refsize, default_value, dtype=expected_dtype)

            data[leaf_name] = arr

        # Fill TTree
        for j in range(refsize):
            vals["channum"][0] = data["channum"][j]
            for leaf_name in schema_keys:
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
        help="Branch names to keep (default: keep all supported event-wise datasets).",
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