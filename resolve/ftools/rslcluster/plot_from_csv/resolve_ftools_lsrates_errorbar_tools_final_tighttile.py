#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resolve_ftools_lsrates_errorbar_tools_with_paircheck.py

Resolve の lsrates CSV を読み込み、

1) 実行前の no_stdcut / stdcut ペア整合性チェック
2) matplotlib による論文向け 6x6 tile の誤差棒付き図
3) Plotly によるデバッグ用 interactive 誤差棒付き散布図

を生成するツールである。

重要な設計
----------
実データ処理では、stdcut に必要な情報が欠損しているファイルがある確率で
混入し得る。その場合、no_stdcut と stdcut のファイルペア、または
(obs_id, pixel) の行ペアが壊れる。

このスクリプトでは、図を作る前に必ずペアチェックを行い、

    lsrates_pair_check_file_pairs.csv
    lsrates_pair_check_obsid_summary.csv
    lsrates_pair_check_missing_rows.csv

を出力する。問題のある OBSID/pixel は、デフォルトでは plot 対象から除外する。

最終版では、Poisson 誤差の計算に time_span ではなく、GTI 合計露光時間である
`t_exp_s` を使う。また、time_start から求めた ObsDate を出力 CSV と
Plotly hover 情報に追加する。

さらに、`t_exp_s` が短すぎる観測を早い段階で screening する。
デフォルトでは `t_exp_s <= 20 ks` を落とし、落ちた観測は短い CSV log に記録する。

Plotly debug 図では、`--debug-xlim` / `--debug-ylim` を個別に指定できる。
未指定の場合は、`--paper-xlim` / `--paper-ylim` を初期表示範囲として流用する。

必要に応じて、`object` 列に対する selection も行える。
`--object-include` はセミコロン区切りで複数候補を指定できる。
デフォルトは部分一致で、`--object-include-perfectmatch` を付けると完全一致になる。
デフォルトでは object name による selection は行わない。

必要に応じて、`--object-exclude OBJECT` により object name で除外することもできる。
こちらもセミコロン区切りで複数候補を指定でき、デフォルトは部分一致、
`--object-exclude-perfectmatch` を付けると完全一致になる。

必要に応じて、`obs_id_str` に対する include selection も行える。
`--obsid-include 001008010` のように指定し、セミコロン区切りで複数 OBSID も選択できる。
デフォルトでは OBSID による selection は行わない。

特殊運用などで除外したい OBSID がある場合は、`--obsid-exclude 000104100` のように指定できる。
こちらもセミコロン区切りで複数 OBSID を指定できる。

matplotlib の論文向け図では、デフォルトで no stdcut を黒、stdcut を青で表示する。
従来の pixel ごとの色分けも `--paper-color-by-pixel` で利用できる。
また、6x6 tile plot に加えて、全画素を 1 枚に重ねた overlay plot も生成する。
さらに、論文向けに panel 間の隙間をゼロにし、各 panel 左上に pixel 番号だけを置く
tight-packed tile plot も option で生成できる。
"""

from __future__ import annotations

import argparse
import datetime as dt
import colorsys
import glob
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
# Set the global font family to serif and prioritize Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import numpy as np
import pandas as pd
import plotly.graph_objects as go


CUT_NO_STDCUT = "no_stdcut"
CUT_STDCUT = "stdcut"
DEFAULT_TIME_COL = "t_exp_s"
DEFAULT_X_RATE_COL = "rate_all"
DEFAULT_X_COUNT_COL = "n_all"
DEFAULT_Y_COL = "rate_small_only"
DEFAULT_OUTPUT_DIR = "resolve_ftools_errorbar_outputs"
DEFAULT_MJD_REFERENCE_DAY = 58484.0
MJD_UNIX_EPOCH = 40587.0
ALL_PIXELS = list(range(36))

PAIR_LOG_FILE_PAIRS = "lsrates_pair_check_file_pairs.csv"
PAIR_LOG_SUMMARY = "lsrates_pair_check_obsid_summary.csv"
PAIR_LOG_PROBLEMS = "lsrates_pair_check_missing_rows.csv"
PAIR_LOG_CLEAN_KEYS = "lsrates_pair_check_clean_keys.csv"
EXPOSURE_SCREEN_LOG = "lsrates_exposure_screening_dropped.csv"
OBJECT_FILTER_LOG = "lsrates_object_filter_kept.csv"
OBJECT_EXCLUDE_LOG = "lsrates_object_exclude_dropped.csv"
OBSID_FILTER_LOG = "lsrates_obsid_filter_kept.csv"
OBSID_EXCLUDE_LOG = "lsrates_obsid_exclude_dropped.csv"
DEFAULT_MIN_EXPOSURE_KS = 6.5

# Resolve 6x6 physical layout
PIXEL_MAP_DETY_DETX_PIXEL = np.array([
    [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
     1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
    [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
     4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],
    [12, 11, 9, 19, 21, 23, 14, 13, 10, 20, 22, 24, 16, 15, 17, 18,
     25, 26, 8, 7, 0, 35, 33, 34, 6, 4, 2, 28, 31, 32, 5, 3, 1, 27,
     29, 30],
])

PIXEL_TO_AXIS: dict[int, tuple[int, int]] = {}
for dety, detx, pixel in PIXEL_MAP_DETY_DETX_PIXEL.T:
    PIXEL_TO_AXIS[int(pixel)] = (int(6 - dety), int(detx - 1))

Y_LABELS = {
    "rate_all": "all",
    "rate_both_pass": "both pass",
    "rate_large_only": "large only",
    "rate_small_only": "small only",
    "rate_both_flag": "both flag",
}

PIXEL_MARKER_SYMBOLS = [
    "circle", "square", "diamond", "cross", "x", "triangle-up",
    "triangle-down", "triangle-left", "triangle-right", "pentagon",
    "hexagon", "hexagon2", "octagon", "star", "hexagram",
    "star-triangle-up", "star-triangle-down", "star-square",
    "star-diamond", "diamond-tall", "diamond-wide", "hourglass",
    "bowtie", "circle-cross", "circle-x", "square-cross", "square-x",
    "diamond-cross", "diamond-x", "cross-thin", "x-thin",
    "asterisk", "hash", "y-up", "y-down", "line-ew",
]

COMPONENT_SYMBOLS = [
    "circle", "square", "diamond", "triangle-up", "triangle-down",
    "cross", "x", "star",
]

MATPLOTLIB_COMPONENT_MARKERS = {
    CUT_NO_STDCUT: "o",
    CUT_STDCUT: "s",
}

MATPLOTLIB_COMPONENT_LABELS = {
    CUT_NO_STDCUT: "no stdcut",
    CUT_STDCUT: "stdcut",
}

# Default paper-plot colors:
# In the 6x6 tile plot, pixel identity is already encoded by panel position.
# Therefore the default emphasizes cut type rather than pixel identity.
color1 = "#0077BB"  # deep cyan-blue
color2 = "#EE7733"  # warm orange
MATPLOTLIB_CUT_COLORS = {
    CUT_NO_STDCUT: color1,
    CUT_STDCUT: color2,
#    CUT_NO_STDCUT: "orange",
#    CUT_STDCUT: "tab:pink",
}


def generate_distinct_pixel_colors(n: int = 36) -> list[str]:
    """Generate 36 distinct-ish colors for pixel identification."""
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.46 if i % 2 == 0 else 0.60
        saturation = 0.72
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colors


PIXEL_COLORS = generate_distinct_pixel_colors(36)


# =============================================================================
# Basic utilities
# =============================================================================

def ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def pixel_to_axis_indices(pixel: int) -> tuple[int, int]:
    return PIXEL_TO_AXIS.get(int(pixel), (int(pixel) // 6, int(pixel) % 6))


def extract_obsid_from_filename(filename: str | Path) -> str:
    basename = os.path.basename(str(filename))
    match = re.search(r"xa(\d+)", basename)
    if match:
        return match.group(1)
    return "UNKNOWN"


def extract_cut_type_from_filename(filename: str | Path) -> str:
    basename = os.path.basename(str(filename))
    # Important: "no_stdcut" contains "stdcut"; check no_stdcut first.
    if CUT_NO_STDCUT in basename:
        return CUT_NO_STDCUT
    if CUT_STDCUT in basename:
        return CUT_STDCUT
    return CUT_NO_STDCUT


def parse_pixel_list(values: Sequence[str] | None) -> list[int]:
    if values is None:
        return []

    pixels: list[int] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token == "":
                continue
            pixel = int(token)
            if pixel < 0 or pixel > 35:
                raise ValueError(f"Pixel must be in 0..35: {pixel}")
            pixels.append(pixel)
    return sorted(set(pixels))


def available_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def normalize_obs_id_series(series: pd.Series) -> pd.Series:
    """Normalize obs_id to a 9-digit string where possible."""
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(9)



def seconds_since_mjd_reference_to_datetime(
    seconds: float,
    mjd_reference_day: float = DEFAULT_MJD_REFERENCE_DAY,
) -> dt.datetime | None:
    """Convert seconds from an MJD reference day to a UTC naive datetime.

    The CSV time_start column is interpreted as seconds from MJD 58484 by default.
    This implementation avoids an astropy dependency by using MJD 40587 as the
    Unix epoch.
    """
    try:
        sec = float(seconds)
    except Exception:
        return None

    if not np.isfinite(sec):
        return None

    unix_seconds = (float(mjd_reference_day) - MJD_UNIX_EPOCH) * 86400.0 + sec
    return dt.datetime.utcfromtimestamp(unix_seconds)


def format_obs_date_from_time_start(
    time_start: float,
    mjd_reference_day: float = DEFAULT_MJD_REFERENCE_DAY,
) -> str:
    """Return ISO-like UTC datetime string from time_start seconds."""
    obs_datetime = seconds_since_mjd_reference_to_datetime(
        seconds=time_start,
        mjd_reference_day=mjd_reference_day,
    )
    if obs_datetime is None:
        return ""
    return obs_datetime.isoformat(timespec="seconds")


def add_obs_date_column(
    df: pd.DataFrame,
    time_start_col: str = "time_start",
    output_col: str = "ObsDate",
    mjd_reference_day: float = DEFAULT_MJD_REFERENCE_DAY,
) -> pd.DataFrame:
    """Add ObsDate column derived from time_start."""
    out = df.copy()
    if time_start_col not in out.columns:
        out[output_col] = ""
        return out

    out[output_col] = [
        format_obs_date_from_time_start(t, mjd_reference_day=mjd_reference_day)
        for t in out[time_start_col]
    ]
    return out


# =============================================================================
# Pair check utilities
# =============================================================================

def is_stdcut_file(path: str | Path) -> bool:
    basename = os.path.basename(str(path))
    return "_stdcut_lsrates.csv" in basename or "stdcut_lsrates.csv" in basename


def is_no_stdcut_file(path: str | Path) -> bool:
    basename = os.path.basename(str(path))
    if is_stdcut_file(basename):
        return False
    return basename.endswith("_lsrates.csv")


def canonical_pair_key(path: str | Path) -> str:
    basename = os.path.basename(str(path))
    basename = basename.replace("_stdcut_lsrates.csv", "_lsrates.csv")
    return basename


def collect_file_pairs(pattern: str) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No CSV files matched: {pattern}")

    by_key: dict[str, dict[str, Path]] = {}
    ignored: list[Path] = []

    for path in paths:
        if is_stdcut_file(path):
            cut_type = CUT_STDCUT
        elif is_no_stdcut_file(path):
            cut_type = CUT_NO_STDCUT
        else:
            ignored.append(path)
            continue

        key = canonical_pair_key(path)
        by_key.setdefault(key, {})
        if cut_type in by_key[key]:
            print(f"[WARN] Duplicate {cut_type} file for key={key}:")
            print(f"       existing: {by_key[key][cut_type]}")
            print(f"       new:      {path}")
        by_key[key][cut_type] = path

    rows = []
    for key, item in sorted(by_key.items()):
        no_path = item.get(CUT_NO_STDCUT)
        st_path = item.get(CUT_STDCUT)
        obsid = extract_obsid_from_filename(no_path or st_path or key)
        rows.append(
            {
                "pair_key": key,
                "obsid_from_filename": obsid,
                "has_no_stdcut_file": no_path is not None,
                "has_stdcut_file": st_path is not None,
                "no_stdcut_file": str(no_path) if no_path is not None else "",
                "stdcut_file": str(st_path) if st_path is not None else "",
            }
        )

    return pd.DataFrame(rows), ignored


def read_one_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df["source_csv"] = path.name
    df["source_path"] = str(path)
    df["obsid_from_filename"] = extract_obsid_from_filename(path)
    df["cut_type"] = extract_cut_type_from_filename(path)

    if "obs_id" in df.columns:
        df["obs_id_str"] = normalize_obs_id_series(df["obs_id"])
    else:
        df["obs_id_str"] = df["obsid_from_filename"]

    if "pixel" not in df.columns:
        raise ValueError(f"pixel column is missing in {path}")
    df["pixel"] = df["pixel"].astype(int)

    if "object" not in df.columns:
        df["object"] = ""

    df = add_obs_date_column(df)

    return df


def read_csv_for_paircheck(path: str | Path, cut_type: str) -> pd.DataFrame:
    print(f"[READ] {path}")
    df = read_one_csv(path)
    df["cut_type"] = cut_type
    return df


def lookup_info(df: pd.DataFrame, obs_id_str: str, pixel: int) -> dict:
    sub = df[(df["obs_id_str"] == obs_id_str) & (df["pixel"] == pixel)]
    if len(sub) == 0:
        return {
            "object": "",
            "source_csv": "",
            "event_file": "",
            "time_span": "",
            "t_exp_s": "",
            "ObsDate": "",
        }

    first = sub.iloc[0]
    return {
        "object": first.get("object", ""),
        "source_csv": first.get("source_csv", ""),
        "event_file": first.get("event_file", ""),
        "time_span": first.get("time_span", ""),
        "t_exp_s": first.get("t_exp_s", ""),
        "ObsDate": first.get("ObsDate", ""),
    }


def summarize_one_pair(row: pd.Series) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Check one file pair and return problems, summary, clean keys."""
    pair_key = row["pair_key"]
    no_path = row["no_stdcut_file"]
    st_path = row["stdcut_file"]

    problem_rows = []
    clean_key_rows = []

    summary = {
        "pair_key": pair_key,
        "obsid_from_filename": row["obsid_from_filename"],
        "has_no_stdcut_file": bool(row["has_no_stdcut_file"]),
        "has_stdcut_file": bool(row["has_stdcut_file"]),
        "n_no_stdcut_rows": 0,
        "n_stdcut_rows": 0,
        "n_no_stdcut_pixels": 0,
        "n_stdcut_pixels": 0,
        "n_clean_keys": 0,
        "n_missing_in_stdcut": 0,
        "n_missing_in_no_stdcut": 0,
        "n_duplicate_keys": 0,
        "objects": "",
        "status": "OK",
    }

    if not row["has_no_stdcut_file"] or not row["has_stdcut_file"]:
        missing_side = CUT_NO_STDCUT if not row["has_no_stdcut_file"] else CUT_STDCUT
        existing_side = CUT_STDCUT if missing_side == CUT_NO_STDCUT else CUT_NO_STDCUT
        summary["status"] = "MISSING_FILE"

        problem_rows.append(
            {
                "problem_type": "missing_file",
                "pair_key": pair_key,
                "obs_id_str": row["obsid_from_filename"],
                "object": "",
                "pixel": "",
                "missing_side": missing_side,
                "existing_side": existing_side,
                "source_csv_existing": "",
                "event_file_existing": "",
                "time_span_existing": "",
                "t_exp_s_existing": "",
                "ObsDate_existing": "",
                "no_stdcut_file": no_path,
                "stdcut_file": st_path,
                "note": f"{missing_side} file is missing",
            }
        )
        return pd.DataFrame(problem_rows), summary, pd.DataFrame(clean_key_rows)

    no_df = read_csv_for_paircheck(no_path, CUT_NO_STDCUT)
    st_df = read_csv_for_paircheck(st_path, CUT_STDCUT)

    summary["n_no_stdcut_rows"] = len(no_df)
    summary["n_stdcut_rows"] = len(st_df)
    summary["n_no_stdcut_pixels"] = no_df["pixel"].nunique()
    summary["n_stdcut_pixels"] = st_df["pixel"].nunique()

    objects = sorted(
        set(no_df["object"].dropna().astype(str).tolist())
        | set(st_df["object"].dropna().astype(str).tolist())
    )
    summary["objects"] = ";".join(objects)

    no_keys_df = no_df[["obs_id_str", "pixel"]].drop_duplicates()
    st_keys_df = st_df[["obs_id_str", "pixel"]].drop_duplicates()

    no_key_set = set(map(tuple, no_keys_df[["obs_id_str", "pixel"]].to_numpy()))
    st_key_set = set(map(tuple, st_keys_df[["obs_id_str", "pixel"]].to_numpy()))

    duplicate_keys: set[tuple[str, int]] = set()
    duplicate_count = 0
    for cut_type, df in [(CUT_NO_STDCUT, no_df), (CUT_STDCUT, st_df)]:
        dup_mask = df.duplicated(["obs_id_str", "pixel"], keep=False)
        if dup_mask.any():
            dup = df[dup_mask].copy()
            duplicate_count += len(dup)
            for _, drow in dup.iterrows():
                key = (str(drow["obs_id_str"]), int(drow["pixel"]))
                duplicate_keys.add(key)
                problem_rows.append(
                    {
                        "problem_type": "duplicate_key",
                        "pair_key": pair_key,
                        "obs_id_str": drow["obs_id_str"],
                        "object": drow.get("object", ""),
                        "pixel": int(drow["pixel"]),
                        "missing_side": "",
                        "existing_side": cut_type,
                        "source_csv_existing": drow.get("source_csv", ""),
                        "event_file_existing": drow.get("event_file", ""),
                        "time_span_existing": drow.get("time_span", ""),
                        "t_exp_s_existing": drow.get("t_exp_s", ""),
                        "ObsDate_existing": drow.get("ObsDate", ""),
                        "no_stdcut_file": no_path,
                        "stdcut_file": st_path,
                        "note": f"duplicated obs_id_str/pixel key in {cut_type}",
                    }
                )

    missing_in_stdcut = sorted(no_key_set - st_key_set)
    missing_in_no_stdcut = sorted(st_key_set - no_key_set)

    summary["n_missing_in_stdcut"] = len(missing_in_stdcut)
    summary["n_missing_in_no_stdcut"] = len(missing_in_no_stdcut)
    summary["n_duplicate_keys"] = duplicate_count

    for obs_id_str, pixel in missing_in_stdcut:
        info = lookup_info(no_df, obs_id_str, pixel)
        problem_rows.append(
            {
                "problem_type": "missing_row",
                "pair_key": pair_key,
                "obs_id_str": obs_id_str,
                "object": info["object"],
                "pixel": pixel,
                "missing_side": CUT_STDCUT,
                "existing_side": CUT_NO_STDCUT,
                "source_csv_existing": info["source_csv"],
                "event_file_existing": info["event_file"],
                "time_span_existing": info["time_span"],
                "t_exp_s_existing": info["t_exp_s"],
                "ObsDate_existing": info["ObsDate"],
                "no_stdcut_file": no_path,
                "stdcut_file": st_path,
                "note": "row exists in no_stdcut but is missing in stdcut",
            }
        )

    for obs_id_str, pixel in missing_in_no_stdcut:
        info = lookup_info(st_df, obs_id_str, pixel)
        problem_rows.append(
            {
                "problem_type": "missing_row",
                "pair_key": pair_key,
                "obs_id_str": obs_id_str,
                "object": info["object"],
                "pixel": pixel,
                "missing_side": CUT_NO_STDCUT,
                "existing_side": CUT_STDCUT,
                "source_csv_existing": info["source_csv"],
                "event_file_existing": info["event_file"],
                "time_span_existing": info["time_span"],
                "t_exp_s_existing": info["t_exp_s"],
                "ObsDate_existing": info["ObsDate"],
                "no_stdcut_file": no_path,
                "stdcut_file": st_path,
                "note": "row exists in stdcut but is missing in no_stdcut",
            }
        )

    clean_key_set = (no_key_set & st_key_set) - duplicate_keys
    summary["n_clean_keys"] = len(clean_key_set)

    if missing_in_stdcut or missing_in_no_stdcut:
        summary["status"] = "MISSING_ROWS"
    if duplicate_count > 0:
        summary["status"] = (
            "DUPLICATE_KEYS"
            if summary["status"] == "OK"
            else summary["status"] + "+DUPLICATE_KEYS"
        )

    for obs_id_str, pixel in sorted(clean_key_set):
        info = lookup_info(no_df, obs_id_str, pixel)
        clean_key_rows.append(
            {
                "pair_key": pair_key,
                "obs_id_str": obs_id_str,
                "pixel": pixel,
                "object": info["object"],
                "no_stdcut_file": no_path,
                "stdcut_file": st_path,
            }
        )

    return pd.DataFrame(problem_rows), summary, pd.DataFrame(clean_key_rows)


def run_pair_check(
    pattern: str,
    outdir: Path,
    require_clean_pairs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run pair check and always write CSV logs."""
    print("[INFO] Running no_stdcut/stdcut pair check before plotting...")
    pair_df, ignored = collect_file_pairs(pattern)

    if ignored:
        print(f"[INFO] Ignored non-lsrates-like files: {len(ignored)}")
        for path in ignored[:10]:
            print(f"       {path}")
        if len(ignored) > 10:
            print("       ...")

    problem_tables = []
    summary_rows = []
    clean_key_tables = []

    for _, row in pair_df.iterrows():
        problem_df, summary, clean_keys = summarize_one_pair(row)
        if len(problem_df) > 0:
            problem_tables.append(problem_df)
        if len(clean_keys) > 0:
            clean_key_tables.append(clean_keys)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    problem_df = (
        pd.concat(problem_tables, ignore_index=True)
        if problem_tables
        else pd.DataFrame(
            columns=[
                "problem_type",
                "pair_key",
                "obs_id_str",
                "object",
                "pixel",
                "missing_side",
                "existing_side",
                "source_csv_existing",
                "event_file_existing",
                "time_span_existing",
                "t_exp_s_existing",
                "ObsDate_existing",
                "no_stdcut_file",
                "stdcut_file",
                "note",
            ]
        )
    )
    clean_keys_df = (
        pd.concat(clean_key_tables, ignore_index=True)
        if clean_key_tables
        else pd.DataFrame(
            columns=[
                "pair_key",
                "obs_id_str",
                "pixel",
                "object",
                "no_stdcut_file",
                "stdcut_file",
            ]
        )
    )

    outdir.mkdir(parents=True, exist_ok=True)
    pair_csv = outdir / PAIR_LOG_FILE_PAIRS
    summary_csv = outdir / PAIR_LOG_SUMMARY
    problem_csv = outdir / PAIR_LOG_PROBLEMS
    clean_csv = outdir / PAIR_LOG_CLEAN_KEYS

    pair_df.to_csv(pair_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    problem_df.to_csv(problem_csv, index=False)
    clean_keys_df.to_csv(clean_csv, index=False)

    n_problem_pairs = int((summary_df["status"] != "OK").sum()) if len(summary_df) else 0
    print("")
    print("===== Pair check summary =====")
    print(f"Total file pairs: {len(pair_df)}")
    print(f"Clean keys:       {len(clean_keys_df)}")
    print(f"Problem pairs:    {n_problem_pairs}")
    print(f"Problem rows:     {len(problem_df)}")
    print(f"[WRITE] {pair_csv}")
    print(f"[WRITE] {summary_csv}")
    print(f"[WRITE] {problem_csv}")
    print(f"[WRITE] {clean_csv}")

    if len(problem_df) > 0:
        print("")
        print("===== Pair-check problems detected =====")
        display_cols = ["problem_type", "obs_id_str", "object", "pixel", "missing_side", "note"]
        print(problem_df[display_cols].to_string(index=False))

        if require_clean_pairs:
            raise RuntimeError(
                "Pair check found missing files/rows or duplicated keys. "
                f"See {problem_csv}. Use --no-require-clean-pairs to continue "
                "with clean OBSID/pixel pairs only."
            )

    return pair_df, summary_df, problem_df, clean_keys_df


# =============================================================================
# CSV loading for plotting
# =============================================================================

def load_all_csv(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No CSV files matched: {pattern}")

    frames = []
    for path in paths:
        # Only read lsrates CSV files that are handled by the pair logic.
        if not (is_no_stdcut_file(path) or is_stdcut_file(path)):
            continue
        print(f"[LOAD] {path}")
        frames.append(read_one_csv(path))

    if len(frames) == 0:
        raise FileNotFoundError(f"No lsrates CSV files found from pattern: {pattern}")

    df = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Loaded files for plotting: {len(frames)}")
    print(f"[INFO] Total rows for plotting:   {len(df)}")
    return df


def filter_to_clean_keys(df: pd.DataFrame, clean_keys_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only OBSID/pixel keys that passed the no_stdcut/stdcut pair check."""
    if len(clean_keys_df) == 0:
        raise ValueError("No clean OBSID/pixel keys are available after pair check.")

    key_df = clean_keys_df[["obs_id_str", "pixel"]].drop_duplicates().copy()
    key_df["obs_id_str"] = key_df["obs_id_str"].astype(str)
    key_df["pixel"] = key_df["pixel"].astype(int)

    before = len(df)
    out = df.merge(key_df, on=["obs_id_str", "pixel"], how="inner")
    after = len(out)

    print(f"[INFO] Filtered plotting rows by clean pair keys: {before} -> {after}")
    return out


def screen_by_min_exposure(
    df: pd.DataFrame,
    outdir: Path,
    min_exposure_ks: float = DEFAULT_MIN_EXPOSURE_KS,
    time_col: str = DEFAULT_TIME_COL,
    log_filename: str = EXPOSURE_SCREEN_LOG,
) -> pd.DataFrame:
    """Drop rows with effective exposure <= min_exposure_ks and write a short log.

    The screening is applied to individual rows, i.e. to each
    (obs_id_str, pixel, cut_type) row.  This matches the lsrates CSV structure,
    where each row corresponds to one pixel and one cut condition.

    By default, rows with t_exp_s <= 20 ks are dropped.
    """
    if time_col not in df.columns:
        raise ValueError(f"Exposure screening requested, but {time_col} column is missing.")

    threshold_s = float(min_exposure_ks) * 1000.0
    work = df.copy()
    exposure = pd.to_numeric(work[time_col], errors="coerce")

    drop_mask = exposure.isna() | (exposure <= threshold_s)
    dropped = work[drop_mask].copy()
    kept = work[~drop_mask].copy()

    log_cols = available_columns(
        dropped,
        [
            "ObsDate",
            "obs_id_str",
            "object",
            "pixel",
            "cut_type",
            time_col,
            "time_start",
            "time_end",
            "time_span",
            "source_csv",
            "event_file",
        ],
    )

    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / log_filename

    if len(dropped) > 0:
        log_df = dropped[log_cols].copy()
        log_df.insert(0, "ExposureScreenThreshold_ks", float(min_exposure_ks))
        log_df.insert(1, "ExposureScreenThreshold_s", threshold_s)
        log_df.to_csv(log_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "ExposureScreenThreshold_ks",
                "ExposureScreenThreshold_s",
                "ObsDate",
                "obs_id_str",
                "object",
                "pixel",
                "cut_type",
                time_col,
                "time_start",
                "time_end",
                "time_span",
                "source_csv",
                "event_file",
            ]
        ).to_csv(log_path, index=False)

    n_obs_dropped = dropped["obs_id_str"].nunique() if "obs_id_str" in dropped.columns and len(dropped) else 0
    print("")
    print("===== Exposure screening =====")
    print(f"Exposure column:       {time_col}")
    print(f"Threshold:             {min_exposure_ks:.3g} ks ({threshold_s:.1f} s)")
    print(f"Rows before:           {len(work)}")
    print(f"Rows dropped:          {len(dropped)}")
    print(f"Rows kept:             {len(kept)}")
    print(f"Unique OBSIDs dropped: {n_obs_dropped}")
    print(f"[WRITE] {log_path}")

    if len(kept) == 0:
        raise ValueError(
            "No rows remain after exposure screening. "
            f"Try lowering --min-exposure-ks; current value is {min_exposure_ks}."
        )

    return kept



def parse_object_include_patterns(include_pattern: str | None) -> list[str]:
    """Parse semicolon-separated object include patterns."""
    if include_pattern is None:
        return []

    patterns = []
    for token in str(include_pattern).split(";"):
        token = token.strip()
        if token:
            patterns.append(token)
    return patterns


def filter_by_object_name(
    df: pd.DataFrame,
    outdir: Path,
    include_pattern: str | None = None,
    case_sensitive: bool = False,
    perfect_match: bool = False,
    log_filename: str = OBJECT_FILTER_LOG,
) -> pd.DataFrame:
    """Filter rows by matching against the object column.

    By default, no filtering is applied.

    If include_pattern is given, it may contain multiple candidates separated by
    semicolons, for example

        Abell2319;Abell2319_BS;Abell2319_BS2;Abell2319_Cor1

    The default matching mode is substring matching.  If perfect_match=True,
    exact equality is used instead.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / log_filename

    patterns = parse_object_include_patterns(include_pattern)

    if not patterns:
        pd.DataFrame(
            columns=[
                "ObjectFilterApplied",
                "ObjectIncludePatterns",
                "ObjectFilterMode",
                "ObjectFilterCaseSensitive",
                "ObsDate",
                "obs_id_str",
                "object",
                "pixel",
                "cut_type",
                "t_exp_s",
                "source_csv",
                "event_file",
            ]
        ).to_csv(log_path, index=False)
        print("[INFO] Object-name filter: disabled")
        print(f"[WRITE] {log_path}")
        return df

    if "object" not in df.columns:
        raise ValueError("Object-name filtering requested, but object column is missing.")

    object_series = df["object"].fillna("").astype(str)

    if case_sensitive:
        object_for_match = object_series
        patterns_for_match = patterns
    else:
        object_for_match = object_series.str.lower()
        patterns_for_match = [p.lower() for p in patterns]

    keep_mask = pd.Series(False, index=df.index)
    if perfect_match:
        allowed = set(patterns_for_match)
        keep_mask = object_for_match.isin(allowed)
    else:
        for pattern in patterns_for_match:
            keep_mask |= object_for_match.str.contains(pattern, regex=False, na=False)

    kept = df[keep_mask].copy()
    dropped = df[~keep_mask].copy()

    log_cols = available_columns(
        kept,
        [
            "ObsDate",
            "obs_id_str",
            "object",
            "pixel",
            "cut_type",
            "t_exp_s",
            "source_csv",
            "event_file",
        ],
    )

    if len(kept) > 0:
        log_df = kept[log_cols].copy()
        log_df.insert(0, "ObjectFilterApplied", True)
        log_df.insert(1, "ObjectIncludePatterns", ";".join(patterns))
        log_df.insert(2, "ObjectFilterMode", "perfect_match" if perfect_match else "substring")
        log_df.insert(3, "ObjectFilterCaseSensitive", bool(case_sensitive))
        log_df.to_csv(log_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "ObjectFilterApplied",
                "ObjectIncludePatterns",
                "ObjectFilterMode",
                "ObjectFilterCaseSensitive",
                "ObsDate",
                "obs_id_str",
                "object",
                "pixel",
                "cut_type",
                "t_exp_s",
                "source_csv",
                "event_file",
            ]
        ).to_csv(log_path, index=False)

    print("")
    print("===== Object-name filter =====")
    print(f"Patterns:             {';'.join(patterns)}")
    print(f"Mode:                 {'perfect_match' if perfect_match else 'substring'}")
    print(f"Case sensitive:       {case_sensitive}")
    print(f"Rows before:          {len(df)}")
    print(f"Rows kept:            {len(kept)}")
    print(f"Rows dropped:         {len(dropped)}")
    if len(kept) > 0 and "object" in kept.columns:
        kept_objects = sorted(kept["object"].dropna().astype(str).unique().tolist())
        print(f"Objects kept:         {';'.join(kept_objects)}")
    print(f"[WRITE] {log_path}")

    if len(kept) == 0:
        raise ValueError(
            "No rows remain after object-name filtering. "
            f"Check --object-include value: {';'.join(patterns)}"
        )

    return kept



def filter_out_object_name(
    df: pd.DataFrame,
    outdir: Path,
    exclude_pattern: str | None = None,
    case_sensitive: bool = False,
    perfect_match: bool = False,
    log_filename: str = OBJECT_EXCLUDE_LOG,
) -> pd.DataFrame:
    """Exclude rows by matching against the object column.

    By default, no filtering is applied. If exclude_pattern is given, it may
    contain multiple candidates separated by semicolons. The default matching
    mode is substring matching. If perfect_match=True, exact equality is used.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / log_filename

    patterns = parse_object_include_patterns(exclude_pattern)

    if not patterns:
        pd.DataFrame(
            columns=[
                "ObjectExcludeApplied",
                "ObjectExcludePatterns",
                "ObjectExcludeMode",
                "ObjectExcludeCaseSensitive",
                "ObsDate",
                "obs_id_str",
                "object",
                "pixel",
                "cut_type",
                "t_exp_s",
                "source_csv",
                "event_file",
            ]
        ).to_csv(log_path, index=False)
        print("[INFO] Object-name exclude filter: disabled")
        print(f"[WRITE] {log_path}")
        return df

    if "object" not in df.columns:
        raise ValueError("Object-name exclude filtering requested, but object column is missing.")

    object_series = df["object"].fillna("").astype(str)

    if case_sensitive:
        object_for_match = object_series
        patterns_for_match = patterns
    else:
        object_for_match = object_series.str.lower()
        patterns_for_match = [p.lower() for p in patterns]

    drop_mask = pd.Series(False, index=df.index)
    if perfect_match:
        excluded = set(patterns_for_match)
        drop_mask = object_for_match.isin(excluded)
    else:
        for pattern in patterns_for_match:
            drop_mask |= object_for_match.str.contains(pattern, regex=False, na=False)

    dropped = df[drop_mask].copy()
    kept = df[~drop_mask].copy()

    log_cols = available_columns(
        dropped,
        [
            "ObsDate",
            "obs_id_str",
            "object",
            "pixel",
            "cut_type",
            "t_exp_s",
            "source_csv",
            "event_file",
        ],
    )

    if len(dropped) > 0:
        log_df = dropped[log_cols].copy()
        log_df.insert(0, "ObjectExcludeApplied", True)
        log_df.insert(1, "ObjectExcludePatterns", ";".join(patterns))
        log_df.insert(2, "ObjectExcludeMode", "perfect_match" if perfect_match else "substring")
        log_df.insert(3, "ObjectExcludeCaseSensitive", bool(case_sensitive))
        log_df.to_csv(log_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "ObjectExcludeApplied",
                "ObjectExcludePatterns",
                "ObjectExcludeMode",
                "ObjectExcludeCaseSensitive",
                "ObsDate",
                "obs_id_str",
                "object",
                "pixel",
                "cut_type",
                "t_exp_s",
                "source_csv",
                "event_file",
            ]
        ).to_csv(log_path, index=False)

    print("")
    print("===== Object-name exclude filter =====")
    print(f"Patterns excluded:    {';'.join(patterns)}")
    print(f"Mode:                 {'perfect_match' if perfect_match else 'substring'}")
    print(f"Case sensitive:       {case_sensitive}")
    print(f"Rows before:          {len(df)}")
    print(f"Rows dropped:         {len(dropped)}")
    print(f"Rows kept:            {len(kept)}")
    if len(dropped) > 0 and "object" in dropped.columns:
        dropped_objects = sorted(dropped["object"].dropna().astype(str).unique().tolist())
        print(f"Objects dropped:      {';'.join(dropped_objects)}")
    print(f"[WRITE] {log_path}")

    if len(kept) == 0:
        raise ValueError(
            "No rows remain after object-name exclude filtering. "
            f"Check --object-exclude value: {';'.join(patterns)}"
        )

    return kept



def parse_obsid_include_patterns(include_pattern: str | None) -> list[str]:
    """Parse semicolon-separated OBSID include patterns and normalize to 9 digits."""
    if include_pattern is None:
        return []

    obsids = []
    for token in str(include_pattern).split(";"):
        token = token.strip()
        if not token:
            continue
        token = re.sub(r"\.0$", "", token)
        if token.isdigit():
            token = token.zfill(9)
        obsids.append(token)

    return obsids


def filter_by_obsid(
    df: pd.DataFrame,
    outdir: Path,
    include_pattern: str | None = None,
    log_filename: str = OBSID_FILTER_LOG,
) -> pd.DataFrame:
    """Filter rows by exact matching against obs_id_str.

    By default, no filtering is applied.  If include_pattern is given, it may
    contain multiple OBSIDs separated by semicolons, for example

        001008010;001008020

    OBSID tokens consisting only of digits are normalized with zfill(9), so
    "1008010" and "001008010" both become "001008010".
    """
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / log_filename

    obsids = parse_obsid_include_patterns(include_pattern)

    if not obsids:
        pd.DataFrame(
            columns=[
                "ObsidFilterApplied",
                "ObsidIncludePatterns",
                "ObsDate",
                "obs_id_str",
                "object",
                "pixel",
                "cut_type",
                "t_exp_s",
                "source_csv",
                "event_file",
            ]
        ).to_csv(log_path, index=False)
        print("[INFO] OBSID filter: disabled")
        print(f"[WRITE] {log_path}")
        return df

    if "obs_id_str" not in df.columns:
        raise ValueError("OBSID filtering requested, but obs_id_str column is missing.")

    obsid_series = (
        df["obs_id_str"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(9)
    )
    allowed = set(obsids)
    keep_mask = obsid_series.isin(allowed)

    kept = df[keep_mask].copy()
    dropped = df[~keep_mask].copy()

    log_cols = available_columns(
        kept,
        [
            "ObsDate",
            "obs_id_str",
            "object",
            "pixel",
            "cut_type",
            "t_exp_s",
            "source_csv",
            "event_file",
        ],
    )

    if len(kept) > 0:
        log_df = kept[log_cols].copy()
        log_df.insert(0, "ObsidFilterApplied", True)
        log_df.insert(1, "ObsidIncludePatterns", ";".join(obsids))
        log_df.to_csv(log_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "ObsidFilterApplied",
                "ObsidIncludePatterns",
                "ObsDate",
                "obs_id_str",
                "object",
                "pixel",
                "cut_type",
                "t_exp_s",
                "source_csv",
                "event_file",
            ]
        ).to_csv(log_path, index=False)

    print("")
    print("===== OBSID filter =====")
    print(f"OBSIDs:               {';'.join(obsids)}")
    print(f"Rows before:          {len(df)}")
    print(f"Rows kept:            {len(kept)}")
    print(f"Rows dropped:         {len(dropped)}")
    if len(kept) > 0:
        kept_obsids = sorted(kept["obs_id_str"].dropna().astype(str).unique().tolist())
        kept_objects = sorted(kept["object"].dropna().astype(str).unique().tolist()) if "object" in kept.columns else []
        print(f"OBSIDs kept:          {';'.join(kept_obsids)}")
        print(f"Objects kept:         {';'.join(kept_objects)}")
    print(f"[WRITE] {log_path}")

    if len(kept) == 0:
        raise ValueError(
            "No rows remain after OBSID filtering. "
            f"Check --obsid-include value: {';'.join(obsids)}"
        )

    return kept



def filter_out_obsid(
    df: pd.DataFrame,
    outdir: Path,
    exclude_pattern: str | None = None,
    log_filename: str = OBSID_EXCLUDE_LOG,
) -> pd.DataFrame:
    """Exclude rows by exact matching against obs_id_str.

    By default, no filtering is applied.  If exclude_pattern is given, it may
    contain multiple OBSIDs separated by semicolons, for example

        000104100;001008010

    OBSID tokens consisting only of digits are normalized with zfill(9).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / log_filename

    obsids = parse_obsid_include_patterns(exclude_pattern)

    if not obsids:
        pd.DataFrame(
            columns=[
                "ObsidExcludeApplied",
                "ObsidExcludePatterns",
                "ObsDate",
                "obs_id_str",
                "object",
                "pixel",
                "cut_type",
                "t_exp_s",
                "source_csv",
                "event_file",
            ]
        ).to_csv(log_path, index=False)
        print("[INFO] OBSID exclude filter: disabled")
        print(f"[WRITE] {log_path}")
        return df

    if "obs_id_str" not in df.columns:
        raise ValueError("OBSID exclude filtering requested, but obs_id_str column is missing.")

    obsid_series = (
        df["obs_id_str"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(9)
    )
    excluded = set(obsids)
    drop_mask = obsid_series.isin(excluded)

    dropped = df[drop_mask].copy()
    kept = df[~drop_mask].copy()

    log_cols = available_columns(
        dropped,
        [
            "ObsDate",
            "obs_id_str",
            "object",
            "pixel",
            "cut_type",
            "t_exp_s",
            "source_csv",
            "event_file",
        ],
    )

    if len(dropped) > 0:
        log_df = dropped[log_cols].copy()
        log_df.insert(0, "ObsidExcludeApplied", True)
        log_df.insert(1, "ObsidExcludePatterns", ";".join(obsids))
        log_df.to_csv(log_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "ObsidExcludeApplied",
                "ObsidExcludePatterns",
                "ObsDate",
                "obs_id_str",
                "object",
                "pixel",
                "cut_type",
                "t_exp_s",
                "source_csv",
                "event_file",
            ]
        ).to_csv(log_path, index=False)

    print("")
    print("===== OBSID exclude filter =====")
    print(f"OBSIDs excluded:      {';'.join(obsids)}")
    print(f"Rows before:          {len(df)}")
    print(f"Rows dropped:         {len(dropped)}")
    print(f"Rows kept:            {len(kept)}")
    if len(dropped) > 0:
        dropped_obsids = sorted(dropped["obs_id_str"].dropna().astype(str).unique().tolist())
        dropped_objects = sorted(dropped["object"].dropna().astype(str).unique().tolist()) if "object" in dropped.columns else []
        print(f"OBSIDs dropped:       {';'.join(dropped_obsids)}")
        print(f"Objects dropped:      {';'.join(dropped_objects)}")
    print(f"[WRITE] {log_path}")

    if len(kept) == 0:
        raise ValueError(
            "No rows remain after OBSID exclude filtering. "
            f"Check --obsid-exclude value: {';'.join(obsids)}"
        )

    return kept


# =============================================================================
# Error handling
# =============================================================================

def rate_col_to_count_col(rate_col: str) -> str:
    if not rate_col.startswith("rate_"):
        raise ValueError(f"rate column must start with 'rate_': {rate_col}")
    return "n_" + rate_col[len("rate_"):]


def compute_poisson_rate_error(counts: pd.Series | np.ndarray, exposure: pd.Series | np.ndarray) -> np.ndarray:
    counts_arr = np.asarray(counts, dtype=float)
    exposure_arr = np.asarray(exposure, dtype=float)

    err = np.full(len(counts_arr), np.nan, dtype=float)
    mask = np.isfinite(counts_arr) & np.isfinite(exposure_arr) & (counts_arr >= 0) & (exposure_arr > 0)
    err[mask] = np.sqrt(counts_arr[mask]) / exposure_arr[mask]
    return err


def add_rate_error_column(
    df: pd.DataFrame,
    rate_col: str,
    time_col: str = DEFAULT_TIME_COL,
) -> tuple[pd.DataFrame, str]:
    count_col = rate_col_to_count_col(rate_col)
    if count_col not in df.columns:
        raise ValueError(f"Required count column is missing: {count_col}")
    if time_col not in df.columns:
        raise ValueError(f"Required time column is missing: {time_col}")

    err_col = f"{rate_col}_err"
    out = df.copy()
    out[err_col] = compute_poisson_rate_error(out[count_col], out[time_col])
    return out, err_col


def add_no_stdcut_x_reference(
    df: pd.DataFrame,
    x_rate_col: str = DEFAULT_X_RATE_COL,
    x_count_col: str = DEFAULT_X_COUNT_COL,
    time_col: str = DEFAULT_TIME_COL,
) -> pd.DataFrame:
    required = {"obs_id_str", "pixel", "cut_type", x_rate_col, x_count_col, time_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns are missing for x reference: {sorted(missing)}")

    no_stdcut = df[df["cut_type"] == CUT_NO_STDCUT].copy()
    if len(no_stdcut) == 0:
        raise ValueError("No no_stdcut rows were found. x-axis requires no_stdcut reference rows.")

    x_table = no_stdcut[["obs_id_str", "pixel", x_rate_col, x_count_col, time_col]].rename(
        columns={
            x_rate_col: f"{x_rate_col}_no_stdcut",
            x_count_col: f"{x_count_col}_no_stdcut",
            time_col: f"{time_col}_no_stdcut",
        }
    )

    n_dup = x_table.duplicated(["obs_id_str", "pixel"]).sum()
    if n_dup > 0:
        print(f"[WARN] Found {n_dup} duplicated no_stdcut OBSID/pixel rows. Keeping first.")
        x_table = x_table.drop_duplicates(["obs_id_str", "pixel"], keep="first")

    out = df.merge(x_table, on=["obs_id_str", "pixel"], how="left")
    out[f"{x_rate_col}_no_stdcut_err"] = compute_poisson_rate_error(
        out[f"{x_count_col}_no_stdcut"],
        out[f"{time_col}_no_stdcut"],
    )

    n_missing = out[f"{x_rate_col}_no_stdcut"].isna().sum()
    if n_missing > 0:
        print(f"[WARN] {n_missing} rows have no matched no_stdcut x reference.")

    return out


# =============================================================================
# Data selection
# =============================================================================

def build_basic_plot_table(
    df: pd.DataFrame,
    y_col: str,
    cut_type: str,
    x_rate_col: str = DEFAULT_X_RATE_COL,
    time_col: str = DEFAULT_TIME_COL,
) -> pd.DataFrame:
    work = df.copy()
    work, y_err_col = add_rate_error_column(work, y_col, time_col=time_col)

    x_col = f"{x_rate_col}_no_stdcut"
    x_err_col = f"{x_rate_col}_no_stdcut_err"
    required = {x_col, x_err_col, y_col, y_err_col, "pixel", "cut_type"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    if cut_type != "all":
        work = work[work["cut_type"] == cut_type].copy()

    work = work[np.isfinite(pd.to_numeric(work[x_col], errors="coerce"))]
    work = work[np.isfinite(pd.to_numeric(work[y_col], errors="coerce"))]
    return work


def build_compare_plot_table(
    df: pd.DataFrame,
    y_col: str,
    time_col: str = DEFAULT_TIME_COL,
    x_rate_col: str = DEFAULT_X_RATE_COL,
) -> pd.DataFrame:
    work, y_err_col = add_rate_error_column(df.copy(), y_col, time_col=time_col)

    x_col = f"{x_rate_col}_no_stdcut"
    x_err_col = f"{x_rate_col}_no_stdcut_err"
    required = {x_col, x_err_col, y_col, y_err_col, "pixel", "cut_type"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    work = work[work["cut_type"].isin([CUT_NO_STDCUT, CUT_STDCUT])].copy()
    return work


# =============================================================================
# matplotlib paper plot
# =============================================================================

def plot_paper_tile_errorbar(
    df: pd.DataFrame,
    outdir: Path,
    y_col: str,
    compare_stdcut: bool,
    cut_type: str,
    x_rate_col: str,
    xscale: str,
    yscale: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    figsize: tuple[float, float],
    dpi: int,
    output_prefix: str,
    color_by_pixel: bool = False,
    tight_tile: bool = False,
) -> Path:
    fig, axes = plt.subplots(6, 6, figsize=figsize, sharex=True, sharey=True)
    x_col = f"{x_rate_col}_no_stdcut"
    x_err_col = f"{x_rate_col}_no_stdcut_err"
    y_err_col = f"{y_col}_err"
    component_handles = []
    component_labels = []

    for pixel in range(36):
        row, col = pixel_to_axis_indices(pixel)
        ax = axes[row, col]
        sub_pix = df[df["pixel"] == pixel].copy()

        plot_cut_types = [CUT_NO_STDCUT, CUT_STDCUT] if compare_stdcut else [cut_type]

        # Common axis configuration
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # Pixel label placement
        if tight_tile:
            ax.text(
                0.04, 0.96, f"{pixel}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=12, fontweight="normal",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.4),
                zorder=10,
            )
        else:
            ax.set_title(f"pix {pixel}", fontsize=12)

        if len(sub_pix) == 0:
            if not tight_tile:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            continue

        for plot_cut in plot_cut_types:
            sub = sub_pix[sub_pix["cut_type"] == plot_cut].copy()
            if len(sub) == 0:
                continue

            x = sub[x_col].astype(float).to_numpy()
            y = sub[y_col].astype(float).to_numpy()
            xerr = sub[x_err_col].astype(float).to_numpy()
            yerr = sub[y_err_col].astype(float).to_numpy()

            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(xerr) & np.isfinite(yerr)
            if xscale == "log":
                mask &= x > 0
            if yscale == "log":
                mask &= y > 0
            if np.sum(mask) == 0:
                continue

            color = PIXEL_COLORS[pixel] if color_by_pixel else MATPLOTLIB_CUT_COLORS.get(plot_cut, "black")
            marker = MATPLOTLIB_COMPONENT_MARKERS.get(plot_cut, "o")
            label = MATPLOTLIB_COMPONENT_LABELS.get(plot_cut, plot_cut)

            container = ax.errorbar(
                x[mask],
                y[mask],
                xerr=xerr[mask],
                yerr=yerr[mask],
                fmt=marker,
                markersize=3.5,
                elinewidth=0.8,
                capsize=1.8,
                linestyle="none",
                color=color,
                ecolor=color,
                alpha=0.85,
                markeredgecolor="black",
                markeredgewidth=0.3,
                label=label,
            )
            if label not in component_labels:
                component_handles.append(container)
                component_labels.append(label)

    if tight_tile:
        # Only outer tick labels in a compact 6x6 panel layout
        for i in range(6):
            for j in range(6):
                ax = axes[i, j]
                ax.tick_params(
                    axis="both",
                    which="both",
                    direction="in",
                    top=True,
                    right=True,
                    labelsize=10,
                    pad=1.5,
                )
                if i < 5:
                    ax.tick_params(labelbottom=False)
                if j > 0:
                    ax.tick_params(labelleft=False)

        fig.supxlabel("total rate before standard cut [count/s]", y=0.045, fontsize=14)
        fig.supylabel(f"{Y_LABELS.get(y_col, y_col)} rate [count/s]", x=0.045, fontsize=14)

        if component_handles:
            fig.legend(
                component_handles,
                component_labels,
                loc="upper right",
                bbox_to_anchor=(0.985, 0.985),
                fontsize=11,
                frameon=False,
            )

        if compare_stdcut:
            outfile = outdir / f"{output_prefix}_paper_tile_errorbar_compare_stdcut_{y_col}_tightpacked.png"
        else:
            outfile = outdir / f"{output_prefix}_paper_tile_errorbar_{y_col}_{cut_type}_tightpacked.png"

        fig.subplots_adjust(
            left=0.10,
            right=0.985,
            bottom=0.10,
            top=0.985,
            wspace=0.0,
            hspace=0.0,
        )
    else:
        for ax in axes[-1, :]:
            ax.set_xlabel("total rate before standard cut [count/s]")
        for ax in axes[:, 0]:
            ax.set_ylabel(f"{Y_LABELS.get(y_col, y_col)} rate [count/s]")

        if component_handles:
            fig.legend(component_handles, component_labels, loc="upper right", fontsize=11)

        if compare_stdcut:
            title = f"Resolve tile plot with Poisson errors: {Y_LABELS.get(y_col, y_col)} (stdcut comparison)"
            outfile = outdir / f"{output_prefix}_paper_tile_errorbar_compare_stdcut_{y_col}.png"
        else:
            title = f"Resolve tile plot with Poisson errors: {Y_LABELS.get(y_col, y_col)} ({cut_type})"
            outfile = outdir / f"{output_prefix}_paper_tile_errorbar_{y_col}_{cut_type}.png"

        fig.suptitle(title, fontsize=16, y=0.995)
        fig.tight_layout(rect=[0, 0, 0.97, 0.98])

    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    print(f"[WRITE] {outfile}")
    return outfile


def plot_paper_overlay_errorbar(
    df: pd.DataFrame,
    outdir: Path,
    y_col: str,
    compare_stdcut: bool,
    cut_type: str,
    x_rate_col: str,
    xscale: str,
    yscale: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    figsize: tuple[float, float],
    dpi: int,
    output_prefix: str,
    color_by_pixel: bool = False,
) -> Path:
    """Make a paper-style single-panel overlay plot using all pixels.

    By default, color encodes cut type:
      - no_stdcut: black
      - stdcut: tab:blue

    If color_by_pixel=True, the legacy pixel-color mode is used.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x_col = f"{x_rate_col}_no_stdcut"
    x_err_col = f"{x_rate_col}_no_stdcut_err"
    y_err_col = f"{y_col}_err"

    legend_keys: set[str] = set()
    handles = []
    labels = []

    plot_cut_types = [CUT_NO_STDCUT, CUT_STDCUT] if compare_stdcut else [cut_type]

    for pixel in range(36):
        sub_pix = df[df["pixel"] == pixel].copy()
        if len(sub_pix) == 0:
            continue

        for plot_cut in plot_cut_types:
            sub = sub_pix[sub_pix["cut_type"] == plot_cut].copy()
            if len(sub) == 0:
                continue

            x = sub[x_col].astype(float).to_numpy()
            y = sub[y_col].astype(float).to_numpy()
            xerr = sub[x_err_col].astype(float).to_numpy()
            yerr = sub[y_err_col].astype(float).to_numpy()

            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(xerr) & np.isfinite(yerr)
            if xscale == "log":
                mask &= x > 0
            if yscale == "log":
                mask &= y > 0
            if np.sum(mask) == 0:
                continue

            marker = MATPLOTLIB_COMPONENT_MARKERS.get(plot_cut, "o")
            cut_label = MATPLOTLIB_COMPONENT_LABELS.get(plot_cut, plot_cut)

            if color_by_pixel:
                color = PIXEL_COLORS[pixel]
                legend_key = f"pix {pixel}"
                label = f"pix {pixel}"
            else:
                color = MATPLOTLIB_CUT_COLORS.get(plot_cut, "black")
                legend_key = plot_cut
                label = cut_label

            container = ax.errorbar(
                x[mask],
                y[mask],
                xerr=xerr[mask],
                yerr=yerr[mask],
                fmt=marker,
                markersize=4.0,
                elinewidth=0.8,
                capsize=1.8,
                linestyle="none",
                color=color,
                ecolor=color,
                alpha=0.42 if color_by_pixel else 0.50,
                markeredgecolor="black" if not color_by_pixel else color,
                markeredgewidth=0.25,
                label=label,
                rasterized=True,
            )

            if legend_key not in legend_keys:
                legend_keys.add(legend_key)
                handles.append(container)
                labels.append(label)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel("total rate before standard cut [count/s]")
    ax.set_ylabel(f"{Y_LABELS.get(y_col, y_col)} rate [count/s]")

    if compare_stdcut:
        title = f"Resolve all-pixel overlay: {Y_LABELS.get(y_col, y_col)} (stdcut comparison)"
        outfile = outdir / f"{output_prefix}_paper_overlay_errorbar_compare_stdcut_{y_col}.png"
    else:
        title = f"Resolve all-pixel overlay: {Y_LABELS.get(y_col, y_col)} ({cut_type})"
        outfile = outdir / f"{output_prefix}_paper_overlay_errorbar_{y_col}_{cut_type}.png"

    ax.set_title(title)

    if handles:
        if color_by_pixel:
            ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=7,
                ncol=1,
                frameon=False,
            )
            fig.tight_layout(rect=[0, 0, 0.82, 1])
        else:
            ax.legend(handles, labels, loc="best", fontsize=11, frameon=False)
            fig.tight_layout()
    else:
        fig.tight_layout()

    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    print(f"[WRITE] {outfile}")
    return outfile


# =============================================================================
# Plotly debug plot
# =============================================================================

def get_pixel_color(pixel: int) -> str:
    return PIXEL_COLORS[int(pixel) % len(PIXEL_COLORS)]


def get_pixel_symbol(pixel: int) -> str:
    return PIXEL_MARKER_SYMBOLS[int(pixel) % len(PIXEL_MARKER_SYMBOLS)]


def get_component_symbol(index: int) -> str:
    return COMPONENT_SYMBOLS[index % len(COMPONENT_SYMBOLS)]


def configure_plotly_layout(
    fig: go.Figure,
    title: str,
    x_title: str,
    y_title: str,
    log_x: bool,
    log_y: bool,
    width: int,
    height: int,
    add_buttons: bool,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=width,
        height=height,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="closest",
        legend={
            "title": "Click legend to show/hide",
            "itemsizing": "constant",
            "groupclick": "togglegroup",
        },
        margin={"l": 80, "r": 40, "t": 90, "b": 70},
    )
    xaxis_kwargs = {"type": "log" if log_x else "linear"}
    yaxis_kwargs = {"type": "log" if log_y else "linear"}

    if xlim is not None:
        if log_x:
            if xlim[0] <= 0 or xlim[1] <= 0:
                raise ValueError(f"Plotly log x-axis requires positive limits: {xlim}")
            xaxis_kwargs["range"] = [np.log10(xlim[0]), np.log10(xlim[1])]
        else:
            xaxis_kwargs["range"] = [xlim[0], xlim[1]]

    if ylim is not None:
        if log_y:
            if ylim[0] <= 0 or ylim[1] <= 0:
                raise ValueError(f"Plotly log y-axis requires positive limits: {ylim}")
            yaxis_kwargs["range"] = [np.log10(ylim[0]), np.log10(ylim[1])]
        else:
            yaxis_kwargs["range"] = [ylim[0], ylim[1]]

    fig.update_xaxes(**xaxis_kwargs)
    fig.update_yaxes(**yaxis_kwargs)

    if add_buttons:
        n_traces = len(fig.data)
        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "direction": "right",
                    "x": 0.0,
                    "y": 1.14,
                    "xanchor": "left",
                    "yanchor": "top",
                    "buttons": [
                        {"label": "Show all", "method": "restyle", "args": [{"visible": [True] * n_traces}]},
                        {"label": "Hide all", "method": "restyle", "args": [{"visible": ["legendonly"] * n_traces}]},
                    ],
                }
            ]
        )


def make_customdata_and_hovertemplate(
    df: pd.DataFrame,
    hover_cols: list[str],
    x_label: str,
    y_label: str,
) -> tuple[np.ndarray | None, str]:
    existing = available_columns(df, hover_cols)
    if not existing:
        return None, f"{x_label}: %{{x:.6g}}<br>{y_label}: %{{y:.6g}}<extra></extra>"

    customdata = df[existing].to_numpy()
    hover_lines = [f"{x_label}: %{{x:.6g}}", f"{y_label}: %{{y:.6g}}"]
    for i, col in enumerate(existing):
        hover_lines.append(f"{col}: %{{customdata[{i}]}}")
    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"
    return customdata, hovertemplate


def plot_debug_plotly_errorbar(
    df: pd.DataFrame,
    outdir: Path,
    y_col: str,
    compare_stdcut: bool,
    cut_type: str,
    x_rate_col: str,
    pixels_to_include: list[int],
    selected_pixels: list[int],
    initial_visible: str,
    width: int,
    height: int,
    log_x: bool,
    log_y: bool,
    output_prefix: str,
    add_buttons: bool,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Path:
    fig = go.Figure()
    x_col = f"{x_rate_col}_no_stdcut"
    x_err_col = f"{x_rate_col}_no_stdcut_err"
    y_err_col = f"{y_col}_err"

    hover_cols = [
        "ObsDate", "obs_id_str", "pixel", "cut_type", "object", "source_csv",
        DEFAULT_TIME_COL, x_col, x_err_col, y_col, y_err_col,
    ]

    def decide_visibility(pixel: int) -> bool | str:
        if initial_visible == "all":
            return True
        if initial_visible == "none":
            return "legendonly"
        if initial_visible == "selected":
            if not selected_pixels:
                return True
            return True if pixel in selected_pixels else "legendonly"
        raise ValueError(f"Unknown initial_visible: {initial_visible}")

    for pixel in pixels_to_include:
        sub_pix = df[df["pixel"] == pixel].copy()
        if len(sub_pix) == 0:
            continue

        entries = [(CUT_NO_STDCUT, 0), (CUT_STDCUT, 1)] if compare_stdcut else [(cut_type, 0)]

        for entry_cut, i_component in entries:
            sub = sub_pix[sub_pix["cut_type"] == entry_cut].copy()
            if len(sub) == 0:
                continue

            sub = sub[np.isfinite(pd.to_numeric(sub[x_col], errors="coerce"))]
            sub = sub[np.isfinite(pd.to_numeric(sub[y_col], errors="coerce"))]
            sub = sub[np.isfinite(pd.to_numeric(sub[x_err_col], errors="coerce"))]
            sub = sub[np.isfinite(pd.to_numeric(sub[y_err_col], errors="coerce"))]
            if log_x:
                sub = sub[sub[x_col].astype(float) > 0]
            if log_y:
                sub = sub[sub[y_col].astype(float) > 0]
            if len(sub) == 0:
                continue

            if compare_stdcut:
                marker_symbol = get_component_symbol(i_component)
                trace_name = f"pix {pixel} / {entry_cut}"
                legendgroup = f"pixel:{pixel}"
                showlegend = True if entry_cut == CUT_NO_STDCUT else False
            else:
                marker_symbol = get_pixel_symbol(pixel)
                trace_name = f"pix {pixel}"
                legendgroup = f"pixel:{pixel}"
                showlegend = True

            customdata, hovertemplate = make_customdata_and_hovertemplate(
                sub,
                hover_cols=hover_cols,
                x_label=x_col,
                y_label=y_col,
            )

            fig.add_trace(
                go.Scatter(
                    x=sub[x_col].astype(float),
                    y=sub[y_col].astype(float),
                    mode="markers",
                    name=trace_name,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    visible=decide_visibility(pixel),
                    marker={
                        "size": 8,
                        "opacity": 0.85,
                        "color": get_pixel_color(pixel),
                        "symbol": marker_symbol,
                        "line": {"width": 0.6, "color": "rgba(30,30,30,0.65)"},
                    },
                    error_x={
                        "type": "data",
                        "array": sub[x_err_col].astype(float),
                        "visible": True,
                        "thickness": 0.8,
                        "width": 0,
                    },
                    error_y={
                        "type": "data",
                        "array": sub[y_err_col].astype(float),
                        "visible": True,
                        "thickness": 0.8,
                        "width": 0,
                    },
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                )
            )

    if len(fig.data) == 0:
        raise ValueError("No valid data points were available for Plotly plotting.")

    title = f"Resolve debug scatter with Poisson errors: {Y_LABELS.get(y_col, y_col)}"
    if compare_stdcut:
        title += " (stdcut comparison)"
        outfile = outdir / f"{output_prefix}_debug_plotly_errorbar_compare_stdcut_{y_col}.html"
    else:
        title += f" ({cut_type})"
        outfile = outdir / f"{output_prefix}_debug_plotly_errorbar_{y_col}_{cut_type}.html"

    configure_plotly_layout(
        fig,
        title=title,
        x_title="total rate before standard cut [count/s]",
        y_title=f"{Y_LABELS.get(y_col, y_col)} rate [count/s]",
        log_x=log_x,
        log_y=log_y,
        width=width,
        height=height,
        add_buttons=add_buttons,
        xlim=xlim,
        ylim=ylim,
    )

    fig.write_html(str(outfile), include_plotlyjs="cdn", full_html=True)
    print(f"[WRITE] {outfile}")
    return outfile


# =============================================================================
# CSV export
# =============================================================================

def export_plot_table(
    df: pd.DataFrame,
    outdir: Path,
    filename: str,
    x_rate_col: str,
    y_col: str,
) -> Path:
    cols = available_columns(
        df,
        [
            "ObsDate", "obs_id_str", "pixel", "cut_type", "object", "source_csv",
            "time_start", "time_end", "time_span", "t_exp_s",
            DEFAULT_TIME_COL,
            f"{x_rate_col}_no_stdcut", f"{x_rate_col}_no_stdcut_err",
            y_col, f"{y_col}_err",
            DEFAULT_X_COUNT_COL, rate_col_to_count_col(y_col),
        ],
    )
    outfile = outdir / filename
    df[cols].to_csv(outfile, index=False)
    print(f"[WRITE] {outfile}")
    return outfile


# =============================================================================
# CLI and workflow
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate pair-checked Resolve lsrates errorbar plots: "
            "(1) matplotlib paper tile and (2) Plotly debug scatter."
        )
    )
    parser.add_argument("pattern", help='CSV glob pattern, e.g. "p0px1000/*lsrates.csv".')
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}.")
    parser.add_argument("--time-col", default=DEFAULT_TIME_COL, help=f"Exposure/time column for Poisson errors. Default: {DEFAULT_TIME_COL}.")
    parser.add_argument(
        "--min-exposure-ks",
        type=float,
        default=DEFAULT_MIN_EXPOSURE_KS,
        help=(
            "Minimum effective exposure in ks. Rows with time-col <= this value "
            "are dropped before plotting. Default: 6.5 (ks)"
        ),
    )
    parser.add_argument(
        "--no-exposure-screen",
        action="store_true",
        help="Disable the short-exposure screening. Not recommended for final plots.",
    )
    parser.add_argument("--mjd-reference-day", type=float, default=DEFAULT_MJD_REFERENCE_DAY, help=f"MJD reference day for time_start -> ObsDate conversion. Default: {DEFAULT_MJD_REFERENCE_DAY}.")
    parser.add_argument("--x-rate-col", default=DEFAULT_X_RATE_COL, help=f"X-axis base rate column. Default: {DEFAULT_X_RATE_COL}.")
    parser.add_argument("--x-count-col", default=DEFAULT_X_COUNT_COL, help=f"Count column corresponding to x-rate. Default: {DEFAULT_X_COUNT_COL}.")
    parser.add_argument("--y-col", default=DEFAULT_Y_COL, help=f"Y-axis rate column. Default: {DEFAULT_Y_COL}.")
    parser.add_argument(
        "--object-include",
        default=None,
        help=(
            "Keep only rows whose object column matches this selection. "
            "Multiple candidates can be separated by semicolons. "
            "Default mode is substring matching. "
            "Example: --object-include 'Abell2319;Abell2319_BS;Abell2319_BS2'"
        ),
    )
    parser.add_argument(
        "--object-include-perfectmatch",
        action="store_true",
        help=(
            "Use exact object-name matching for --object-include. "
            "Default is substring matching."
        ),
    )
    parser.add_argument(
        "--object-case-sensitive",
        action="store_true",
        help="Make --object-include and --object-exclude matching case-sensitive. Default is case-insensitive.",
    )
    parser.add_argument(
        "--object-exclude",
        default=None,
        help=(
            "Exclude rows whose object column matches this selection. "
            "Multiple candidates can be separated by semicolons. "
            "Default mode is substring matching. "
            "Example: --object-exclude 'Abell2319;Abell2319_BS'"
        ),
    )
    parser.add_argument(
        "--object-exclude-perfectmatch",
        action="store_true",
        help=(
            "Use exact object-name matching for --object-exclude. "
            "Default is substring matching."
        ),
    )
    parser.add_argument(
        "--obsid-include",
        default=None,
        help=(
            "Keep only rows whose obs_id_str exactly matches the given OBSID(s). "
            "Multiple OBSIDs can be separated by semicolons. "
            "Example: --obsid-include '001008010;001008020'"
        ),
    )
    parser.add_argument(
        "--obsid-exclude",
        default=None,
        help=(
            "Exclude rows whose obs_id_str exactly matches the given OBSID(s). "
            "Multiple OBSIDs can be separated by semicolons. "
            "Example: --obsid-exclude '000104100;001008010'"
        ),
    )
    parser.add_argument("--cut-type", choices=[CUT_NO_STDCUT, CUT_STDCUT], default=CUT_NO_STDCUT, help="Cut type used in non-comparison mode.")
    parser.add_argument("--compare-stdcut", action="store_true", help="Overlay no_stdcut and stdcut for the same y column.")
    parser.add_argument("--pixels", nargs="+", default=None, help="Pixel list. Both '0 1 2' and '0,1,2' are accepted.")
    parser.add_argument("--all-pixels", action="store_true", help="Include all 36 pixels in the Plotly debug HTML.")
    parser.add_argument("--initial-visible", choices=["all", "selected", "none"], default="all", help="Initial visibility for Plotly traces.")

    parser.add_argument("--paper-xscale", choices=["log", "linear"], default="log", help="X-axis scale for paper plot.")
    parser.add_argument("--paper-yscale", choices=["log", "linear"], default="log", help="Y-axis scale for paper plot.")
    parser.add_argument("--paper-xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"), help="Common x-axis range for paper tile plot.")
    parser.add_argument("--paper-ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"), help="Common y-axis range for paper tile plot.")
    parser.add_argument("--paper-figsize", nargs=2, type=float, default=(12.0, 8.0), metavar=("WIDTH", "HEIGHT"), help="Figure size for paper plot in inches.")
    parser.add_argument("--paper-dpi", type=int, default=200, help="DPI for paper plot PNG.")
    parser.add_argument(
        "--paper-color-by-pixel",
        action="store_true",
        help=(
            "Use the legacy pixel-based color scheme in matplotlib paper plots. "
            "Default is cut-type colors: no_stdcut=black, stdcut=blue."
        ),
    )
    parser.add_argument(
        "--paper-tight-tile",
        action="store_true",
        help=(
            "Generate a tight-packed 6x6 tile plot for paper use: "
            "panel gaps are set to zero and each panel shows only the pixel number "
            "at the upper left."
        ),
    )
    parser.add_argument(
        "--paper-overlay-figsize",
        nargs=2,
        type=float,
        default=(6.4, 5.2),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size for the all-pixel matplotlib overlay plot in inches.",
    )
    parser.add_argument(
        "--no-paper-overlay",
        action="store_true",
        help="Do not generate the all-pixel matplotlib overlay plot.",
    )

    parser.add_argument("--debug-linear-x", action="store_true", help="Use a linear x-axis in Plotly debug plot.")
    parser.add_argument("--debug-linear-y", action="store_true", help="Use a linear y-axis in Plotly debug plot.")
    parser.add_argument("--debug-xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"), help="Initial x-axis range for Plotly debug plot. If omitted, --paper-xlim is used.")
    parser.add_argument("--debug-ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"), help="Initial y-axis range for Plotly debug plot. If omitted, --paper-ylim is used.")
    parser.add_argument("--debug-width", type=int, default=1280, help="Width of Plotly figure in pixels.")
    parser.add_argument("--debug-height", type=int, default=900, help="Height of Plotly figure in pixels.")
    parser.add_argument("--no-buttons", action="store_true", help="Do not add Show all / Hide all buttons to Plotly output.")

    parser.add_argument("--write-csv", action="store_true", help="Write the table used for plotting to CSV.")
    parser.add_argument("--output-prefix", default="resolve_ftools_lsrates", help="Prefix for output files.")

    parser.add_argument(
        "--no-pair-filter",
        action="store_true",
        help=(
            "Do not filter plotting rows by clean no_stdcut/stdcut pairs. "
            "Pair-check CSV logs are still written. Not recommended for final plots."
        ),
    )
    parser.add_argument(
        "--require-clean-pairs",
        action="store_true",
        help=(
            "Stop immediately if any pair-check problem is found. "
            "Default behavior is to continue using only clean OBSID/pixel pairs."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> list[Path]:
    outdir = ensure_outdir(args.outdir)
    selected_pixels = parse_pixel_list(args.pixels)
    pixels_to_include = ALL_PIXELS.copy() if args.all_pixels or not selected_pixels else selected_pixels

    # -------------------------------------------------------------------------
    # 0. Mandatory pair check before plotting
    # -------------------------------------------------------------------------
    _, _, problem_df, clean_keys_df = run_pair_check(
        pattern=args.pattern,
        outdir=outdir,
        require_clean_pairs=args.require_clean_pairs,
    )

    # -------------------------------------------------------------------------
    # 1. Load data and optionally filter by clean pairs
    # -------------------------------------------------------------------------
    df = load_all_csv(args.pattern)
    df = add_obs_date_column(df, mjd_reference_day=args.mjd_reference_day)

    if not args.no_pair_filter:
        df = filter_to_clean_keys(df, clean_keys_df)
    else:
        print("[WARN] --no-pair-filter was specified. Plotting may include broken stdcut/no_stdcut pairs.")

    if not args.no_exposure_screen:
        df = screen_by_min_exposure(
            df,
            outdir=outdir,
            min_exposure_ks=args.min_exposure_ks,
            time_col=args.time_col,
        )
    else:
        print("[WARN] --no-exposure-screen was specified. Short-exposure rows are kept.")
        empty_log = outdir / EXPOSURE_SCREEN_LOG
        pd.DataFrame(
            columns=[
                "ExposureScreenThreshold_ks",
                "ExposureScreenThreshold_s",
                "ObsDate",
                "obs_id_str",
                "object",
                "pixel",
                "cut_type",
                args.time_col,
                "time_start",
                "time_end",
                "time_span",
                "source_csv",
                "event_file",
            ]
        ).to_csv(empty_log, index=False)
        print(f"[WRITE] {empty_log}")

    df = filter_by_object_name(
        df,
        outdir=outdir,
        include_pattern=args.object_include,
        case_sensitive=args.object_case_sensitive,
        perfect_match=args.object_include_perfectmatch,
    )

    df = filter_out_object_name(
        df,
        outdir=outdir,
        exclude_pattern=args.object_exclude,
        case_sensitive=args.object_case_sensitive,
        perfect_match=args.object_exclude_perfectmatch,
    )

    df = filter_by_obsid(
        df,
        outdir=outdir,
        include_pattern=args.obsid_include,
    )

    df = filter_out_obsid(
        df,
        outdir=outdir,
        exclude_pattern=args.obsid_exclude,
    )

    df = add_no_stdcut_x_reference(
        df,
        x_rate_col=args.x_rate_col,
        x_count_col=args.x_count_col,
        time_col=args.time_col,
    )

    if args.compare_stdcut:
        plot_df = build_compare_plot_table(
            df,
            y_col=args.y_col,
            time_col=args.time_col,
            x_rate_col=args.x_rate_col,
        )
        csv_name = f"{args.output_prefix}_plot_table_compare_stdcut_{args.y_col}.csv"
    else:
        plot_df = build_basic_plot_table(
            df,
            y_col=args.y_col,
            cut_type=args.cut_type,
            x_rate_col=args.x_rate_col,
            time_col=args.time_col,
        )
        csv_name = f"{args.output_prefix}_plot_table_{args.y_col}_{args.cut_type}.csv"

    output_files: list[Path] = []

    if args.write_csv:
        output_files.append(
            export_plot_table(
                plot_df,
                outdir=outdir,
                filename=csv_name,
                x_rate_col=args.x_rate_col,
                y_col=args.y_col,
            )
        )

    output_files.append(
        plot_paper_tile_errorbar(
            df=plot_df,
            outdir=outdir,
            y_col=args.y_col,
            compare_stdcut=args.compare_stdcut,
            cut_type=args.cut_type,
            x_rate_col=args.x_rate_col,
            xscale=args.paper_xscale,
            yscale=args.paper_yscale,
            xlim=tuple(args.paper_xlim) if args.paper_xlim is not None else None,
            ylim=tuple(args.paper_ylim) if args.paper_ylim is not None else None,
            figsize=tuple(args.paper_figsize),
            dpi=args.paper_dpi,
            output_prefix=args.output_prefix,
            color_by_pixel=args.paper_color_by_pixel,
            tight_tile=args.paper_tight_tile,
        )
    )

    if not args.no_paper_overlay:
        output_files.append(
            plot_paper_overlay_errorbar(
                df=plot_df,
                outdir=outdir,
                y_col=args.y_col,
                compare_stdcut=args.compare_stdcut,
                cut_type=args.cut_type,
                x_rate_col=args.x_rate_col,
                xscale=args.paper_xscale,
                yscale=args.paper_yscale,
                xlim=tuple(args.paper_xlim) if args.paper_xlim is not None else None,
                ylim=tuple(args.paper_ylim) if args.paper_ylim is not None else None,
                figsize=tuple(args.paper_overlay_figsize),
                dpi=args.paper_dpi,
                output_prefix=args.output_prefix,
                color_by_pixel=args.paper_color_by_pixel,
            )
        )

    debug_xlim = tuple(args.debug_xlim) if args.debug_xlim is not None else (
        tuple(args.paper_xlim) if args.paper_xlim is not None else None
    )
    debug_ylim = tuple(args.debug_ylim) if args.debug_ylim is not None else (
        tuple(args.paper_ylim) if args.paper_ylim is not None else None
    )

    output_files.append(
        plot_debug_plotly_errorbar(
            df=plot_df,
            outdir=outdir,
            y_col=args.y_col,
            compare_stdcut=args.compare_stdcut,
            cut_type=args.cut_type,
            x_rate_col=args.x_rate_col,
            pixels_to_include=pixels_to_include,
            selected_pixels=selected_pixels,
            initial_visible=args.initial_visible,
            width=args.debug_width,
            height=args.debug_height,
            log_x=not args.debug_linear_x,
            log_y=not args.debug_linear_y,
            output_prefix=args.output_prefix,
            add_buttons=not args.no_buttons,
            xlim=debug_xlim,
            ylim=debug_ylim,
        )
    )

    if len(problem_df) > 0:
        print("")
        print("[WARN] Pair-check problems were detected.")
        print(f"[WARN] Problem rows were excluded from plots unless --no-pair-filter was used.")
        print(f"[WARN] See: {outdir / PAIR_LOG_PROBLEMS}")

    return output_files


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_files = run(args)
    print("[DONE]")
    for path in output_files:
        print(f"Open: {path}")


if __name__ == "__main__":
    main()
