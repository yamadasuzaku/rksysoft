# Pixel-Based Cluster Analysis Script for Event Files

This Bash script is designed to automate pixel-level processing and clustering analysis of event files for XRISM/Resolve. 
It supports both individual pixel selection and full-array processing for 36 pixels (PIXEL = 0 to 35).

## Features

- Filter event data by pixel number or process all pixels
- Automatically construct output filenames with proper zero-padding
- Run clustering analysis using `resolve_ana_pixel_Ls_define_cluster.py`
- Post-process to extract clusters based on the `IMEMBER` column
- Supports robust command-line argument handling

## Requirements

Make sure the following command-line tools and scripts are available and in your `$PATH`:

- `ftselect` (from FTOOLS)
- `resolve_ana_pixel_Ls_define_cluster.py`

## Usage

```bash
./resolve_ftools_calc_clusters.sh -f <evtfile> [-p <pixel_number>] [-a]
````

### Options

| Option              | Description                           |
| ------------------- | ------------------------------------- |
| `-f <evtfile>`      | Input event file (**required**)       |
| `-p <pixel_number>` | Pixel number to process (default: 19) |
| `-a`                | Process all pixels (overrides `-p`)   |

## Examples

### 1. Process a single pixel (e.g., pixel 12):

```bash
./resolve_ftools_calc_clusters.sh -f xa000114000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt -p 12
```

This will:

* Extract events with `PIXEL == 12`
* Run clustering analysis
* Output several processed files

### 2. Process all pixels:

```bash
./resolve_ftools_calc_clusters.sh -f xa000114000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt -a
```

This will:

* Analyze all pixels in the input file
* Run clustering over the entire detector array

## Output Files

For single-pixel mode (`-p`):

* `<base>_pXX.evt` — Extracted pixel events
* `addcluster_<base>_pXX.evt` — Output after clustering
* `addcluster_<base>_pXX_imposi.evt` — `IMEMBER > 0`
* `addcluster_<base>_pXX_im1.evt` — `IMEMBER == 1`

For all-pixel mode (`-a`):

* `addcluster_<base>.evt` — Output after clustering
* `addcluster_<base>_imposi.evt` — `IMEMBER > 0`
* `addcluster_<base>_im1.evt` — `IMEMBER == 1`

> `<base>` is derived from the input filename by removing the `.evt` extension.

## Notes

* Pixel numbers must be integers between 0 and 35.
* The script exits with an error if the input file does not exist.
* Output filenames are automatically generated using zero-padded pixel indices.

## License

This script is distributed under the MIT License. See `LICENSE` for details.

## Acknowledgments

This tool was developed to support data processing tasks for pixel-based detectors, such as the XRISM/Resolve instrument. It facilitates consistent preprocessing and clustering analysis across research teams.