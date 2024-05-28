# Script to add `PREV_INTERVAL` and `NEXT_INTERVAL` 

The python script, [resolve_tool_addcol_prev_next_interval.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_tool_addcol_prev_next_interval.py) adds `PREV_INTERVAL` and `NEXT_INTERVAL` columns to a uf.evt file. 
It calculates the trigger time based on the input parameters and updates the specified columns accordingly.

[resolve_tool_map_prevnextinterval.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_tool_map_prevnextinterval.py) will map `PREV_INTERVAL` and `NEXT_INTERVAL` columns in a uf.evt file onto pr.evt. 

## History

- 2024.5.28, v1, S.Y.

## Features

- Calculates the trigger time (`SAMPLECNTTRIG_WO_VERNIER`) for each PIXEL based on WFRB_WRITE_LP, WFRB_SAMPLE_CNT, and TRIG_LP. 
- Computes the differences between consecutive elements in the counter list, considering 24-bit overflow.
- Updates the `PREV_INTERVAL` and `NEXT_INTERVAL` columns in the FITS file.
- Option to overwrite the original FITS file or create a new file with the updated data.
- Map `PREV_INTERVAL` and `NEXT_INTERVAL` columns onto pr.evt. 

## Usage

### Add `PREV_INTERVAL` and `NEXT_INTERVAL` into uf.evt

[resolve_tool_addcol_prev_next_interval.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_tool_addcol_prev_next_interval.py) is used as follows. 

``` bash:
#Example 1) Overwrite the original FITS file:
resolve_tool_addcol_prev_next_interval.py xa000114000rsl_p0px1000_uf.evt 

#Example 2) Create a new file:
resolve_tool_addcol_prev_next_interval.py xa000114000rsl_p0px1000_uf.evt -o xa000114000rsl_p0px1000_uf_prevnext.evt
``` 

### Map `PREV_INTERVAL` and `NEXT_INTERVAL` in uf.evt onto pr.evt

[resolve_tool_map_prevnextinterval.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/util/resolve_tool_map_prevnextinterval.py) is used as follows. 


``` bash:
# Example 1) Overwrite the original target FITS file:
resolve_tool_map_interval_quick.py xa000114000rsl_p0px1000_uf_prevnext.evt xa000114000rsl_a0pxpr_uf.evt 

# Example 2) Create a new target FITS file:
resolve_tool_map_interval_quick.py xa000114000rsl_p0px1000_uf_prevnext.evt xa000114000rsl_a0pxpr_uf.evt -o xa000114000rsl_a0pxpr_uf_fillprenext.evt
``` 

## Requirements

- Python 3.x
- `numpy`
- `astropy`

You can install the required packages using pip:

```sh
pip install numpy astropy
