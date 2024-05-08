#!/usr/bin/env python

# Define constants
# pixel WFRB の大きさ 1 MB / 4 byte = (2**18) = hex(2**18)= 0x40000
WFRB_SIZE_PTR=0x40000

# Log data from psp syslog
syslogs = [
    ("2023-10-17 08:03:30 PSP_ID= 0", 0xad480040, 0xad4c0040, 0x4c0001, 0x4c30d3),
    ("2024-01-06 07:01:49 PSP_ID= 1", 0xb6a00040, 0xb6a40040, 0xa40001, 0xa430d5),
    ("2024-02-18 11:52:04 PSP_ID= 2", 0x93a00040, 0x93a40040, 0xa40001, 0xa430d4),
    ("2024-03-14 14:38:58 PSP_ID= 3", 0xe4680040, 0xe46c0040, 0x6c0001, 0x6c30d5),
    ("2024-05-07 19:56:35 PSP_ID= 0", 0x86bc0040, 0x86c00040, 0xc00001, 0xc030d6)
]

# Function to perform bit shift
def psp_get_wfrb_lap(lp):
    return lp >> 18

# Processing and displaying each log entry
for log, prev_wfrb_adm_sample_cnt, wfrb_adm_sample_cnt, prev_wfrb_adm_write_lp, wfrb_adm_write_lp in syslogs:
    print(log)
    print(f"  prev_wfrb_adm_sample_cnt: {prev_wfrb_adm_sample_cnt:x}, wfrb_adm_sample_cnt: {wfrb_adm_sample_cnt:x}, (prev_wfrb_adm_sample_cnt + {WFRB_SIZE_PTR:x}: {prev_wfrb_adm_sample_cnt + WFRB_SIZE_PTR:x})")
    print(f"  prev_wfrb_adm_write_lp (hex): {prev_wfrb_adm_write_lp:x}, wfrb_adm_write_lp: {wfrb_adm_write_lp:x}")
    print(f"  prev_wfrb_adm_write_lp (bin): {format(prev_wfrb_adm_write_lp, '032b')}, wfrb_adm_write_lp: {format(wfrb_adm_write_lp, '032b')}")
    print(f"  psp_get_wfrb_lap(prev_wfrb_adm_write_lp) (bin): {format(psp_get_wfrb_lap(prev_wfrb_adm_write_lp), '014b')}, psp_get_wfrb_lap(wfrb_adm_write_lp): {format(psp_get_wfrb_lap(wfrb_adm_write_lp), '014b')}")
