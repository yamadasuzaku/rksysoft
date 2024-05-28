# Script to add `PREV_INTERVAL` and `NEXT_INTERVAL` into uf.evt 

This program adds `PREV_INTERVAL` and `NEXT_INTERVAL` columns to a uf.evt file. 
It calculates the trigger time based on the input parameters and updates the specified columns accordingly.

## Features

- Calculates the trigger time (`SAMPLECNTTRIG_WO_VERNIER`) for each PIXEL based on WFRB_WRITE_LP, WFRB_SAMPLE_CNT, and TRIG_LP. 
- Computes the differences between consecutive elements in the counter list, considering 24-bit overflow.
- Updates the `PREV_INTERVAL` and `NEXT_INTERVAL` columns in the FITS file.
- Option to overwrite the original FITS file or create a new file with the updated data.

## Usage

``` bash:
#Example 1) Overwrite the original FITS file:
resolve_tool_addcol_prev_next_interval.py xa000114000rsl_p0px1000_uf.evt 

#Example 2) Create a new file:
resolve_tool_addcol_prev_next_interval.py xa000114000rsl_p0px1000_uf.evt -o xa000114000rsl_p0px1000_uf_prevnext.evt
``` 

## Requirements

- Python 3.x
- `numpy`
- `astropy`

You can install the required packages using pip:

```sh
pip install numpy astropy
