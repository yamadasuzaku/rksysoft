# Script for XRISM/Resolve: Add PREV/NEXT interval columns to Event File Processing 

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_prevnext.sh

This Bash script automates a series of preprocessing steps on XRISM/Resolve event files, including event filtering, column augmentation, GTI generation, and final GTI filtering. 
It is designed to ensure reproducibility and robustness in an international collaborative data analysis environment.

## Overview

The script performs the following operations:

1. **Check prerequisites**: Verifies the presence of required commands and input files.
2. **Remove BL (BaseLine) events** from a given `_uf.evt` file using `ITYPE<5`.
3. **Add previous/next interval columns** to the filtered event file.
4. **Generate a GTI (Good Time Interval)** from the corresponding `_cl.evt` file.
5. **Apply GTI filtering** to the processed event file using the generated GTI.

## Requirements

The following command-line tools must be available in your `$PATH`:

- [`resolve_util_ftselect.sh`](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_util_ftselect.sh)
- [`resolve_tool_addcol_prev_next_interval.py`](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_tool_addcol_prev_next_interval.py)
- [`resolve_util_ftmgtime.py`](resolve_util_ftmgtime.py)

Ensure these tools are installed or callable from the shell.

## Usage

```bash
./resolve_ftools_add_prevnext.sh <input_file_uf.evt>
````

For example:

```bash
resolve_ftools_add_prevnext.sh xa000114000rsl_p0px1000_uf.evt
```

## Input Files

* `xa<obsid>_uf.evt` — unfiltered event file (required)
* `xa<obsid>_cl.evt` — cleaned event file for GTI generation (required)

These must exist **before** running the script.

## Output Files

* `<obsid>_noBL.evt` — BL events removed (`ITYPE<5`)
* `<obsid>_noBL_prevnext.evt` — with additional `PREV_INTERVAL` and `NEXT_INTERVAL` columns
* `<obsid>_cl.gti` — GTI file generated from the CL file
* `cutclgti.evt` — final filtered event file with GTI applied

## Notes

* The script will exit immediately on any error, including missing commands or files.
* Make sure you are in the working directory where the input files are located, or specify the full path to the input file.