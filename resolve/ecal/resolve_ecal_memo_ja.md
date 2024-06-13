# Method for the Energy Calibration using the residuals in PHA vs. PI plane

For each pixel and itype in the event file, temporarily save the data as a CSV file along with the TIME column. Use the entire observation time to fit two polynomials, and save the residuals in a CSV file.

[resolve_ecal_run_fitpoly.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ecal/resolve_ecal_run_fitpoly.sh)

Usage:

``` bash:
resolve_ecal_run_fitpoly.sh xa000126000rsl_p0px1000_cl.evt
```

Specify the event file to run the following two programs for all pixels and Hp in one go.


## Save TIME, PHA, and PI as a CSV file for each pixel and Hp only


[resolve_ecal_pha_pi.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ecal/resolve_ecal_pha_pi.py)

``` bash:
./resolve_ecal_pha_pi.py xa000126000rsl_p0px1000_cl.evt TIME 1,1 PHA,PI 1,1 --filters PIXEL==0,ITYPE==0 -o 00
```

- `TIME 1,1`: Obtain the TIME column from FITS EXTENSION 1. 1,1 is set to allow for different extensions in the following arguments.
- `PHA,PI 1,1`: Obtain PHA and PI from FITS EXTENSION 1.
- `--filters PIXEL==0,ITYPE==0`: Obtain data only from PIXEL 0 and Hp.
- `-o 00`: Append 00 to the filename. Since it is pixel 0, 00 is appended.


## Fit with polynomials and examine the time variation of residuals

[resolve_ecal_fitpoly_csv.py](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ecal/resolve_ecal_fitpoly_csv.py)

``` bash:
python resolve_ecal_fitpoly_csv.py fplot_xa000126000rsl_p0px1000_cl_p00.csv PHA PI 10000 4 4 fitpoly_p00.png
```

- `PHA PI 10000 4 4 fitpoly_p00.png`: Extract PHA and PI from the CSV file, divide them into two polynomials around PHA=10000, and approximate each with a 4th-degree polynomial.