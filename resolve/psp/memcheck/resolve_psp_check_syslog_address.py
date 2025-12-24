#!/usr/bin/env python

# Define constants
# pixel WFRB の大きさ 1 MB / 4 byte = (2**18) = hex(2**18)= 0x40000
WFRB_SIZE_PTR=0x40000

# Log data from psp syslog
syslogs = [
    ("2023-10-17 08:03:30 PSP_ID= 0", 0xad480040, 0xad4c0040, 0x4c0001, 0x4c30d3),
    ("2024-01-06 07:01:49 PSP_ID= 1", 0xb6a00040, 0xb6a40040, 0xa40001, 0xa430d5),
    ("2024-02-18 11:52:04 PSP_ID= 2", 0x93a00040, 0x93a40040, 0xa40001, 0xa430d4),
    ("2024-02-28 22:11:34 PSP_ID= 0", 0x330c0040, 0x33100040, 0x100001, 0x1030d6),
    ("2024-03-14 14:38:58 PSP_ID= 3", 0xe4680040, 0xe46c0040, 0x6c0001, 0x6c30d5),
    ("2024-05-07 19:56:35 PSP_ID= 0", 0x86bc0040, 0x86c00040, 0xc00001, 0xc030d6),
    ("2024-07-22 16:24:06 PSP_ID= 0", 0x99900040, 0x99940040, 0x940001, 0x9430d6),    
    ("2024-08-17 06:06:22 PSP_ID= 2", 0x07a40040, 0x07a80040, 0xa80001, 0xa830d7),
    ("2024-09-21 10:49:31 PSP_ID= 2", 0xe1580040, 0xe15c0040, 0x5c0001, 0x5c30d3),
    ("2024-09-25 22:10:15 PSP_ID= 2", 0x01440040, 0x01480040, 0x480001, 0x4830d3),
    ("2024-10-03 00:49:18 PSP_ID= 1", 0xcafc0040, 0xcb000040, 0x000001, 0x0030d7),
    ("2024-10-27 12:58:15 PSP_ID= 0", 0xf4840040, 0xf4880040, 0x880001, 0x8830d4),
    ("2025-01-25 08:47:24 PSP_ID= 3", 0x8ad80040, 0x8adc0040, 0xdc0001, 0xdc30d4),
    ("2025-02-09 11:49:15 PSP_ID= 0", 0x58900040, 0x58940040, 0x940001, 0x9430d5),
    ("2025-03-07 02:44:38 PSP_ID= 1", 0xc9e80040, 0xc9ec0040, 0xec0001, 0xec30d6),
    ("2025-03-18 16:40:35 PSP_ID= 2", 0xb3600040, 0xb3640040, 0x640001, 0x6430d4),
    ("2025-05-21 22:36:31 PSP_ID= 0", 0xdb240040, 0xdb280040, 0x280001, 0x2830d2),
    ("2025-05-31 12:15:34 PSP_ID= 0", 0x431c0040, 0x43200040, 0x200001, 0x2030d6),
    ("2025-05-31 12:15:34 PSP_ID= 0", 0x5d940040, 0x5d980040, 0x980001, 0x9830d1),         
    ("2025-09-02 02:12:31 PSP_ID= 0", 0xcb300040, 0xcb340040, 0x340001, 0x3430d3),
    ("2025-09-18 20:33:38 PSP_ID= 3", 0x02600040, 0x02640040, 0x640001, 0x6430d5),                      
    ("2025-11-01 12:09:06 PSP_ID= 1", 0xfc380040, 0xfc3c0040, 0x3c0001, 0x3c30d5)
]

syslogs_samplerec = [
    ("2024-08-02 17:56:55 PSP_ID= 3", 0, 0xfc0000, 0x3029fffd),
    ("2024-08-02 17:56:55 PSP_ID= 2", 0, 0xfc0000, 0x3025fffe),
    ("2024-08-02 17:56:55 PSP_ID= 1", 0, 0xfc0000, 0x30230003),
    ("2024-08-02 17:56:55 PSP_ID= 0", 0, 0xfc0000, 0x30210002),
    ("2024-12-13 17:56:55 PSP_ID= 1", 0, 0xfc0a04, 0x262700ef),
    ("2024-12-13 17:56:55 PSP_ID= 3", 0, 0xfc0a0f, 0x262500fa),
    ("2025-09-04 14:17:32 PSP_ID= 3", 0, 0xfc0000, 0x3027ffff),
    ("2025-09-04 14:17:32 PSP_ID= 2", 0, 0xfc0000, 0x30250000),
    ("2025-09-04 14:17:32 PSP_ID= 1", 0, 0xfc0000, 0x30260002),
    ("2025-09-04 14:17:32 PSP_ID= 0", 0, 0xfc0000, 0x3022fffa),
    ("2025-10-17 05:05:02 PSP_ID= 3", 0, 0xac00d4, 0x262900bf), 
    ("2025-10-17 05:05:02 PSP_ID= 1", 0, 0xac00da, 0x262800c5), 
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

# Processing and displaying each log entry
for log, lap_lsb, lp, data in syslogs_samplerec:
    print(log, lap_lsb, lp, data)
