#!/bin/bash

ls *.evt > f.list 

# 0-20 keV

resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 0 --filters "PIXEL==0"  -o pixel00_Hp
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 0 --filters "PIXEL==17" -o pixel17_Hp
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 0 --filters "PIXEL==18" -o pixel18_Hp
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 0 --filters "PIXEL==35" -o pixel35_Hp
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 0 -o pixelALL_Hp

resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 1 --filters "PIXEL==0"  -o pixel00_Mp
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 1 --filters "PIXEL==17" -o pixel17_Mp
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 1 --filters "PIXEL==18" -o pixel18_Mp
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 1 --filters "PIXEL==35" -o pixel35_Mp
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 1 -o pixelALL_Mp

resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 2 --filters "PIXEL==0"  -o pixel00_Ms
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 2 --filters "PIXEL==17" -o pixel17_Ms
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 2 --filters "PIXEL==18" -o pixel18_Ms
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 2 --filters "PIXEL==35" -o pixel35_Ms
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 2 -o pixelALL_Ms

resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 3 --filters "PIXEL==0"  -o pixel00_Lp
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 3 --filters "PIXEL==17" -o pixel17_Lp
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 3 --filters "PIXEL==18" -o pixel18_Lp
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 3 --filters "PIXEL==35" -o pixel35_Lp
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 3 -o pixelALL_Lp

resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 4 --filters "PIXEL==0"  -o pixel00_Ls
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 4 --filters "PIXEL==17" -o pixel17_Ls
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 4 --filters "PIXEL==18" -o pixel18_Ls
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 4 --filters "PIXEL==35" -o pixel35_Ls
resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 4 -o pixelALL_Ls

resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 0 --filters "PIXEL==0"  -o pixel00_Hp_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 0 --filters "PIXEL==17" -o pixel17_Hp_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 0 --filters "PIXEL==18" -o pixel18_Hp_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 0 --filters "PIXEL==35" -o pixel35_Hp_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 0 -o pixelALL_Hp_sd34

resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 1 --filters "PIXEL==0"  -o pixel00_Mp_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 1 --filters "PIXEL==17" -o pixel17_Mp_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 1 --filters "PIXEL==18" -o pixel18_Mp_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 1 --filters "PIXEL==35" -o pixel35_Mp_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 1 -o pixelALL_Mp_sd34

resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 2 --filters "PIXEL==0"  -o pixel00_Ms_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 2 --filters "PIXEL==17" -o pixel17_Ms_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 2 --filters "PIXEL==18" -o pixel18_Ms_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 2 --filters "PIXEL==35" -o pixel35_Ms_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 2 -o pixelALL_Ms_sd34

resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 3 --filters "PIXEL==0"  -o pixel00_Lp_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 3 --filters "PIXEL==17" -o pixel17_Lp_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 3 --filters "PIXEL==18" -o pixel18_Lp_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 3 --filters "PIXEL==35" -o pixel35_Lp_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 3 -o pixelALL_Lp_sd34

resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 4 --filters "PIXEL==0"  -o pixel00_Ls_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 4 --filters "PIXEL==17" -o pixel17_Ls_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 4 --filters "PIXEL==18" -o pixel18_Ls_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 4 --filters "PIXEL==35" -o pixel35_Ls_sd34
resolve_ana_pixel_hist1d_many_eventfiles.py flist_st34.list --x_col PI --xmin 0 --xmax 20000 --rebin 200 -i 4 -o pixelALL_Ls_sd34


# # 6-7.2 keV
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 0 --filters "PIXEL==0"  -o pixel00_Hp
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 0 --filters "PIXEL==17" -o pixel17_Hp
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 0 --filters "PIXEL==18" -o pixel18_Hp
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 0 --filters "PIXEL==35" -o pixel35_Hp
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 0 -o pixelALL_Hp
											      
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 1 --filters "PIXEL==0"  -o pixel00_Mp
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 1 --filters "PIXEL==17" -o pixel17_Mp
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 1 --filters "PIXEL==18" -o pixel18_Mp
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 1 --filters "PIXEL==35" -o pixel35_Mp
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 1 -o pixelALL_Mp
											      
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 2 --filters "PIXEL==0"  -o pixel00_Ms
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 2 --filters "PIXEL==17" -o pixel17_Ms
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 2 --filters "PIXEL==18" -o pixel18_Ms
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 2 --filters "PIXEL==35" -o pixel35_Ms
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 2 -o pixelALL_Ms
											      
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 3 --filters "PIXEL==0"  -o pixel00_Lp
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 3 --filters "PIXEL==17" -o pixel17_Lp
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 3 --filters "PIXEL==18" -o pixel18_Lp
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 3 --filters "PIXEL==35" -o pixel35_Lp
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 3 -o pixelALL_Lp
											      
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 4 --filters "PIXEL==0"  -o pixel00_Ls
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 4 --filters "PIXEL==17" -o pixel17_Ls
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 4 --filters "PIXEL==18" -o pixel18_Ls
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 4 --filters "PIXEL==35" -o pixel35_Ls
# resolve_ana_pixel_hist1d_many_eventfiles.py f.list --x_col PI --xmin 6300 --xmax 7300 --rebin 5 -i 4 -o pixelALL_Ls

