# HDF5 → ROOT Automatic Converter for uMUX Data

This script automatically converts **uMUX analysis output files (`*_mass.hdf5`)** into **ROOT TTree files (`.root`)**.

It is designed for use in experimental pipelines where HDF5 files are produced continuously and ROOT files are required for downstream analysis.

The script scans the uMUX output directory, checks whether a corresponding ROOT file already exists, and performs the conversion only if needed.

---

# Overview

The typical data flow in the uMUX analysis pipeline is:

```
DAQ
 ↓
MASS analysis
 ↓
*_mass.hdf5
 ↓
(scan_hdf5_to_root.py)
 ↓
ROOT TTree (*.root)
 ↓
physics analysis / visualization
```

This script automates the **HDF5 → ROOT conversion stage**.

---

# Features

* Automatically scans HDF5 files produced by MASS
* Converts missing files into ROOT TTrees
* Avoids reconversion if the ROOT file already exists
* Ensures execution inside a ROOT-compatible conda environment
* Optional directory monitoring mode
* Avoids processing files that are still being written
* Compatible with Python **3.9+**

---

# Directory Structure

Input HDF5 files are expected to be located in:

```
${HOME}/output/umux/
```

Example:

```
${HOME}/output/umux/20260304_run0004/20260304_run0004_mass.hdf5
```

Converted ROOT files are written to:

```
${HOME}/output/umux/root/
```

Example output:

```
${HOME}/output/umux/root/20260304_run0004_mass.root
```

This separation avoids cluttering the data directories.

---

# Requirements

## Python

Python **3.9 or newer** is recommended.

The script avoids version-specific features to maintain compatibility across common environments.

## Python packages

Required packages:

```
numpy
h5py
ROOT (PyROOT)
```

These should be installed inside a conda environment.

Example:

```
conda create -n root_env python=3.9
conda activate root_env
conda install root numpy h5py
```

---

# Environment Handling

The script automatically ensures that it runs inside a ROOT-enabled environment.

If the script is executed outside the environment, it automatically relaunches itself using:

```
${HOME}/miniforge3/envs/root_env/bin/python
```

This avoids issues caused by incompatible Python environments.

# Usage

## Run once (convert missing ROOT files)

```
python heates_scan_hdf5_to_root.py
```

The script will:

1. Scan all `*_mass.hdf5` files
2. Check whether the corresponding `.root` file exists
3. Convert only missing files

---

## Continuous monitoring mode

```
python heates_scan_hdf5_to_root.py --watch
```

The script will:

1. Scan directories
2. Convert missing files
3. Wait
4. Repeat

Default interval:

```
60 seconds
```

---

## Custom interval

```
python heates_scan_hdf5_to_root.py --watch --interval 120
```

---

## Force reconversion

Overwrite existing ROOT files.

```
python heates_scan_hdf5_to_root.py --force
```

---

## Restrict branches

Only selected datasets will be written into the ROOT TTree.

Example:

```
python heates_scan_hdf5_to_root.py \
  --attrs timestamp energy rise_time postpeak_deriv
```

---

# ROOT Tree Structure

The script reads datasets from the HDF5 structure and creates a ROOT `TTree`.

Example branches:

```
timestamp
energy
rise_time
postpeak_deriv
channum
```

Channels are extracted automatically from group names.

Example:

```
channel03 → channum = 3
```

---

# File Safety

The script avoids converting files that are still being written.

A simple stability check is used:

1. File size is measured
2. Wait a few seconds
3. File size is measured again

If the size changes, the file is skipped.

---

# Typical Deployment

A common experimental setup is to run the converter continuously on a server:

```
screen -S root_convert
python heates_scan_hdf5_to_root.py --watch
```

or

```
tmux new -s root_convert
python heates_scan_hdf5_to_root.py --watch
```

This enables near-real-time conversion.

---

# Logging (recommended)

Example logging setup:

```
python heates_scan_hdf5_to_root.py --watch \
  > root_converter.log 2>&1
```

---

# Design Philosophy

The script was written with the following goals:

* robustness for experimental pipelines
* minimal dependencies
* compatibility with multiple Python versions
* safe handling of partially written files
* simple integration with DAQ workflows

---

# Possible Future Extensions

Potential improvements include:

* inotify-based real-time detection
* multiprocessing conversion
* automatic quicklook plots
* web dashboard integration
* run-based directory organization

---

# License

This software is intended for research use.

Please contact the authors for collaboration or redistribution.

---

# Author

Developed for HEATES data analysis pipelines.