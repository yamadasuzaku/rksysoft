#!/usr/bin/env python3
"""Download and plot RXTE/ASM definitive 1-dwell products.

Features
--------
- Time-series plots for total ASM rate, A/B/C band rates, and hardness C/B.
- Hardness-intensity diagram (HID) and color-color diagram (CCD).
- Independent inverse-variance time binning for the time series and diagrams.
- Default zero cut for non-positive rate/error values, with --no-zero-cut override.
- Concise statistical summaries by default.
- Verbose FITS/debug output only when --debug is specified.

Example
-------
python plot_rxte_asm_hid_ccd_stats.py --source gx5-1 --bin-days 1
python plot_rxte_asm_hid_ccd_stats.py --source gx5-1 --diagram-bin-days 3 --debug

Required packages
-----------------
numpy, astropy, matplotlib
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time

BASE_URL = (
    "https://heasarc.gsfc.nasa.gov/FTP/xte/data/archive/ASMProducts/"
    "definitive_1dwell"
)

BAND_LABELS = (
    "A: 1.3-3.0 keV",
    "B: 3.0-5.0 keV",
    "C: 5.0-12.1 keV",
)


@dataclass
class Logger:
    debug_enabled: bool = False

    def info(self, message: str = "") -> None:
        print(message)

    def debug(self, message: str) -> None:
        if self.debug_enabled:
            print(message)


@dataclass
class Series:
    mjd: np.ndarray
    rate: np.ndarray
    error: np.ndarray


@dataclass
class DiagramSeries:
    """Synchronized quantities used in the HID and color-color diagram."""

    mjd: np.ndarray
    intensity: np.ndarray
    intensity_error: np.ndarray
    soft_color: np.ndarray
    soft_color_error: np.ndarray
    hard_color: np.ndarray
    hard_color_error: np.ndarray


@dataclass
class FilterReport:
    label: str
    total_rows: int
    after_time: int
    after_error: int
    after_zero: int | None
    after_ssc: int | None
    after_chi2: int | None
    after_snr: int | None
    final_rows: int


@dataclass
class SummaryReport:
    name: str
    n_before_date: int
    n_after_date: int
    n_after_bin: int
    start_utc: str
    stop_utc: str
    duration_days: float
    y_min: float
    y_med: float
    y_max: float
    err_med: float
    err_max: float
    large_error_count: int
    large_error_fraction: float
    large_error_threshold: float
    extra_lines: list[str]


@dataclass
class DiagramReport:
    n_input: int
    n_basic_valid: int
    n_after_snr: int | None
    n_after_relerr: int | None
    n_final: int
    intensity_min: float
    intensity_max: float
    soft_min: float
    soft_max: float
    hard_min: float
    hard_max: float
    large_relerr_threshold: float
    large_relerr_intensity: int
    large_relerr_soft: int
    large_relerr_hard: int


def normalize_source_name(name: str) -> str:
    token = name.strip().lower().replace(" ", "")
    token = token.replace("_", "")
    if not re.fullmatch(r"[a-z0-9.+-]+", token):
        raise ValueError(
            f"Unsupported source token: {token!r}. "
            "For GX 5-1 use --source gx5-1."
        )
    return token


def download_file(
    url: str,
    destination: Path,
    logger: Logger,
    overwrite: bool = False,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        logger.debug(f"[cache] {destination}")
        return destination

    logger.info(f"[download] {url}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "rxte-asm-python/1.0"},
    )

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            content_length = response.headers.get("Content-Length")
            expected_size = (
                int(content_length)
                if content_length is not None
                else None
            )

            with temporary.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

        actual_size = temporary.stat().st_size

        if expected_size is not None and actual_size != expected_size:
            raise RuntimeError(
                "Downloaded file is incomplete: "
                f"expected={expected_size} bytes, "
                f"actual={actual_size} bytes, "
                f"url={url}"
            )

        # FITS を実際に最後まで読み、切断ファイルを検出する
        with fits.open(temporary, memmap=False) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    _ = np.asarray(hdu.data)

        temporary.replace(destination)

    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
        TypeError,
    ) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    return destination
    

def first_binary_table(hdul: fits.HDUList) -> fits.BinTableHDU:
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None:
            return hdu
    raise ValueError("No binary-table extension was found in the FITS file.")


def mjd_reference(*headers: fits.Header) -> float:
    for header in headers:
        if "MJDREF" in header:
            return float(header["MJDREF"])
        if "MJDREFI" in header or "MJDREFF" in header:
            return float(header.get("MJDREFI", 0.0)) + float(header.get("MJDREFF", 0.0))
    return 0.0


def time_to_mjd(time_values: np.ndarray, table_header: fits.Header, primary_header: fits.Header) -> np.ndarray:
    values = np.asarray(time_values, dtype=float)
    unit = str(table_header.get("TIMEUNIT", primary_header.get("TIMEUNIT", "d"))).strip().lower()
    timezero = float(table_header.get("TIMEZERO", primary_header.get("TIMEZERO", 0.0)))

    if unit.startswith("s"):
        values_days = (values + timezero) / 86400.0
    else:
        values_days = values + timezero

    ref = mjd_reference(table_header, primary_header)
    if ref != 0.0:
        return ref + values_days
    if np.nanmedian(values_days) > 40000:
        return values_days
    raise ValueError("Could not determine MJDREF from the FITS headers.")


def column_lookup(names: list[str]) -> dict[str, str]:
    return {name.upper(): name for name in names}


def find_column(names: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = column_lookup(names)
    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]
    return None


def describe_columns(table: fits.BinTableHDU, logger: Logger) -> None:
    logger.debug("[FITS columns]")
    for column in table.columns:
        logger.debug(f"  {column.name:20s} format={column.format!s:8s} unit={column.unit!s}")


def apply_common_filters(
    data: fits.FITS_rec,
    mjd: np.ndarray,
    error: np.ndarray,
    ssc: int | None,
    max_chi2: float | None,
    min_snr: float | None,
    rate_for_snr: np.ndarray,
    label: str,
    logger: Logger,
    zero_cut: bool = True,
) -> tuple[np.ndarray, FilterReport]:
    names = list(data.names)
    total_rows = len(data)

    mask = np.isfinite(mjd)
    after_time = int(mask.sum())

    if error.ndim == 1:
        good_error = np.isfinite(error)
    else:
        good_error = np.any(np.isfinite(error), axis=1)
    mask &= good_error
    after_error = int(mask.sum())

    after_zero: int | None = None
    if zero_cut:
        if rate_for_snr.ndim == 1:
            good_zero = (
                np.isfinite(rate_for_snr)
                & np.isfinite(error)
                & (rate_for_snr > 0)
                & (error > 0)
            )
        else:
            good_zero = np.any(
                np.isfinite(rate_for_snr)
                & np.isfinite(error)
                & (rate_for_snr > 0)
                & (error > 0),
                axis=1,
            )
        mask &= good_zero
        after_zero = int(mask.sum())

    after_ssc: int | None = None
    ssc_col = find_column(names, ("SSC_NUMBER", "SSC", "CAMERA"))
    if ssc is not None:
        if ssc_col is None:
            logger.info("[warning] No SSC_NUMBER column; --ssc was ignored.")
        else:
            keep = np.asarray(data[ssc_col], dtype=int) == ssc
            mask &= keep
            after_ssc = int(mask.sum())

    after_chi2: int | None = None
    if max_chi2 is not None:
        chi_col = find_column(names, ("RDCHI_SQ", "RED_CHI2", "REDCHISQ", "CHI2", "CHISQ"))
        if chi_col is None:
            logger.info("[warning] No chi-square column; --max-chi2 was ignored.")
        else:
            chi = np.asarray(data[chi_col], dtype=float)
            keep = np.isfinite(chi) & (chi <= max_chi2)
            mask &= keep
            after_chi2 = int(mask.sum())

    after_snr: int | None = None
    if min_snr is not None:
        if rate_for_snr.ndim == 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = rate_for_snr / error
            keep = np.isfinite(snr) & (snr >= min_snr)
        else:
            valid = np.isfinite(rate_for_snr) & np.isfinite(error) & (error > 0)
            snr = np.full_like(rate_for_snr, np.nan, dtype=float)
            snr[valid] = rate_for_snr[valid] / error[valid]
            keep = np.any(snr >= min_snr, axis=1)
        mask &= keep
        after_snr = int(mask.sum())

    report = FilterReport(
        label=label,
        total_rows=total_rows,
        after_time=after_time,
        after_error=after_error,
        after_zero=after_zero,
        after_ssc=after_ssc,
        after_chi2=after_chi2,
        after_snr=after_snr,
        final_rows=int(mask.sum()),
    )
    return mask, report


def zeroed_values_to_nan(rate: np.ndarray, error: np.ndarray, zero_cut: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return copies where non-positive rate/error samples are masked as NaN.

    For scalar series this is mostly redundant because row-level zero cut already
    removes non-positive samples.  For A/B/C band series it is important: one band
    can be zero while the other bands are still usable for ordinary light curves.
    Masking only the bad band keeps the row synchronized, while later color-ratio
    calculations naturally drop ratios that need that band.
    """
    clean_rate = np.asarray(rate, dtype=float).copy()
    clean_error = np.asarray(error, dtype=float).copy()
    if not zero_cut:
        return clean_rate, clean_error

    good = (
        np.isfinite(clean_rate)
        & np.isfinite(clean_error)
        & (clean_rate > 0)
        & (clean_error > 0)
    )
    clean_rate[~good] = np.nan
    clean_error[~good] = np.nan
    return clean_rate, clean_error


def read_total(
    path: Path,
    ssc: int | None,
    max_chi2: float | None,
    min_snr: float | None,
    logger: Logger,
    zero_cut: bool = True,
) -> tuple[Series, FilterReport]:
    with fits.open(path, memmap=True) as hdul:
        table = first_binary_table(hdul)
        describe_columns(table, logger)
        data = table.data
        names = list(data.names)

        time_col = find_column(names, ("TIME",))
        rate_col = find_column(names, ("RATE", "FLUX", "INTENSITY"))
        err_col = find_column(names, ("ERROR", "ERR", "RATE_ERR", "FLUX_ERR"))
        if not (time_col and rate_col and err_col):
            raise ValueError(f"Required total-rate columns were not found. Columns: {names}")

        rate = np.asarray(data[rate_col], dtype=float)
        error = np.asarray(data[err_col], dtype=float)
        if rate.ndim != 1 or error.ndim != 1:
            raise ValueError(
                f"The light-curve RATE/ERROR columns are not scalar: RATE{rate.shape}, ERROR{error.shape}"
            )

        mjd = time_to_mjd(data[time_col], table.header, hdul[0].header)
        mask, report = apply_common_filters(
            data, mjd, error, ssc, max_chi2, min_snr, rate, label="total", logger=logger, zero_cut=zero_cut
        )
        mask &= np.isfinite(rate)
        if not zero_cut:
            mask &= np.isfinite(error)
        rate_out, error_out = zeroed_values_to_nan(rate[mask], error[mask], zero_cut)
        report.final_rows = int(mask.sum())
        return Series(mjd[mask], rate_out, error_out), report


def scalar_band_columns(names: list[str]) -> tuple[list[str], list[str]] | None:
    rate_groups = (
        ("RATE1", "RATE2", "RATE3"),
        ("RATE_A", "RATE_B", "RATE_C"),
        ("A_RATE", "B_RATE", "C_RATE"),
        ("BAND1", "BAND2", "BAND3"),
        ("CHANNEL1", "CHANNEL2", "CHANNEL3"),
    )
    error_groups = (
        ("ERROR1", "ERROR2", "ERROR3"),
        ("ERROR_A", "ERROR_B", "ERROR_C"),
        ("A_ERROR", "B_ERROR", "C_ERROR"),
        ("BAND1_ERR", "BAND2_ERR", "BAND3_ERR"),
        ("CHANNEL1_ERR", "CHANNEL2_ERR", "CHANNEL3_ERR"),
    )
    lookup = column_lookup(names)
    for rates in rate_groups:
        if all(item in lookup for item in rates):
            for errors in error_groups:
                if all(item in lookup for item in errors):
                    return [lookup[item] for item in rates], [lookup[item] for item in errors]
    return None


def _decode_text_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "S":
        array = np.char.decode(array, "ascii", errors="replace")
    return np.char.strip(array.astype(str))


def identify_band_rows(data: fits.FITS_rec) -> np.ndarray:
    names = list(data.names)
    band_index = np.full(len(data), -1, dtype=np.int8)

    min_col = find_column(names, ("MINCHAN", "MIN_CHAN"))
    max_col = find_column(names, ("MAXCHAN", "MAX_CHAN"))
    if min_col is not None and max_col is not None:
        lo = np.asarray(data[min_col], dtype=float)
        hi = np.asarray(data[max_col], dtype=float)
        band_index[hi <= 1188] = 0
        band_index[(lo >= 1189) & (hi <= 1860)] = 1
        band_index[lo >= 1861] = 2

    band_col = find_column(names, ("BAND", "CHANNEL", "ENERGY_BAND"))
    if band_col is not None:
        labels = np.char.upper(_decode_text_array(data[band_col]))
        for i, text in enumerate(labels):
            if band_index[i] >= 0:
                continue
            compact = re.sub(r"[^A-Z0-9.+-]", "", text)
            if compact in {"A", "1", "CH1", "CHAN1", "BAND1"}:
                band_index[i] = 0
            elif compact in {"B", "2", "CH2", "CHAN2", "BAND2"}:
                band_index[i] = 1
            elif compact in {"C", "3", "CH3", "CHAN3", "BAND3"}:
                band_index[i] = 2
            elif "1.3" in text and "3.0" in text:
                band_index[i] = 0
            elif "3.0" in text and "5.0" in text:
                band_index[i] = 1
            elif "5.0" in text and ("12.1" in text or "12.2" in text):
                band_index[i] = 2
    return band_index


def pivot_long_format_bands(
    data: fits.FITS_rec,
    mjd: np.ndarray,
    rate: np.ndarray,
    error: np.ndarray,
    band_index: np.ndarray,
    row_mask: np.ndarray,
) -> Series:
    selected = row_mask & (band_index >= 0)
    if not np.any(selected):
        raise ValueError("No recognizable A/B/C rows remained. Inspect BAND, MINCHAN, and MAXCHAN.")

    names = list(data.names)
    idx = np.flatnonzero(selected)
    key_columns: list[np.ndarray] = [np.asarray(mjd[idx], dtype=np.float64)]
    key_names = ["mjd"]

    for field, candidates, dtype in (
        ("ssc", ("SSC_NUMBER", "SSC", "CAMERA"), np.int32),
        ("ndwell", ("N_DWELL",), np.int32),
        ("tstart", ("T_START_OBS",), np.float64),
        ("tstop", ("T_STOP_OBS",), np.float64),
    ):
        col = find_column(names, candidates)
        if col is not None:
            key_columns.append(np.asarray(data[col][idx], dtype=dtype))
            key_names.append(field)

    dtype_fields = [(name, values.dtype) for name, values in zip(key_names, key_columns)]
    keys = np.empty(idx.size, dtype=dtype_fields)
    for name, values in zip(key_names, key_columns):
        keys[name] = values

    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    group_mjd = mjd[idx[first]]
    wide_rate = np.full((first.size, 3), np.nan, dtype=float)
    wide_error = np.full((first.size, 3), np.nan, dtype=float)

    selected_band = band_index[idx]
    selected_rate = rate[idx]
    selected_error = error[idx]

    for channel in range(3):
        channel_rows = np.flatnonzero(selected_band == channel)
        if channel_rows.size == 0:
            continue
        groups = inverse[channel_rows]
        for group in np.unique(groups):
            use = channel_rows[groups == group]
            r = selected_rate[use]
            e = selected_error[use]
            valid = np.isfinite(r) & np.isfinite(e) & (e > 0)
            if not np.any(valid):
                continue
            weight = 1.0 / np.square(e[valid])
            wide_rate[group, channel] = np.sum(weight * r[valid]) / np.sum(weight)
            wide_error[group, channel] = np.sqrt(1.0 / np.sum(weight))

    keep = np.any(np.isfinite(wide_rate) & np.isfinite(wide_error), axis=1)
    return Series(group_mjd[keep], wide_rate[keep], wide_error[keep])


def read_bands(
    path: Path,
    ssc: int | None,
    max_chi2: float | None,
    min_snr: float | None,
    logger: Logger,
    zero_cut: bool = True,
) -> tuple[Series, FilterReport]:
    with fits.open(path, memmap=True) as hdul:
        table = first_binary_table(hdul)
        describe_columns(table, logger)
        data = table.data
        names = list(data.names)

        time_col = find_column(names, ("TIME",))
        if time_col is None:
            raise ValueError(f"TIME column was not found. Columns: {names}")

        rate_col = find_column(names, ("RATE", "RATES", "FLUX", "INTENSITY"))
        err_col = find_column(names, ("ERROR", "ERRORS", "ERR", "RATE_ERR", "FLUX_ERR"))
        if rate_col and err_col:
            candidate_rate = np.asarray(data[rate_col], dtype=float)
            candidate_error = np.asarray(data[err_col], dtype=float)

            if candidate_rate.ndim == 2 and candidate_rate.shape[1] >= 3:
                rate = candidate_rate[:, :3]
                error = candidate_error[:, :3]
                mjd = time_to_mjd(data[time_col], table.header, hdul[0].header)
                mask, report = apply_common_filters(
                    data, mjd, error, ssc, max_chi2, min_snr, rate, label="bands", logger=logger, zero_cut=zero_cut
                )
                mask &= np.any(np.isfinite(rate), axis=1)
                if not zero_cut:
                    mask &= np.any(np.isfinite(error), axis=1)
                rate_out, error_out = zeroed_values_to_nan(rate[mask], error[mask], zero_cut)
                report.final_rows = int(mask.sum())
                return Series(mjd[mask], rate_out, error_out), report

            if candidate_rate.ndim == 1 and candidate_error.ndim == 1:
                mjd = time_to_mjd(data[time_col], table.header, hdul[0].header)
                band_index = identify_band_rows(data)
                logger.debug(f"[color BAND values] {np.unique(_decode_text_array(data[find_column(names, ('BAND',))])).tolist() if find_column(names, ('BAND',)) is not None else 'N/A'}")
                min_col = find_column(names, ("MINCHAN",))
                max_col = find_column(names, ("MAXCHAN",))
                if min_col is not None and max_col is not None:
                    pairs = np.unique(
                        np.column_stack((
                            np.asarray(data[min_col], dtype=int),
                            np.asarray(data[max_col], dtype=int),
                        )),
                        axis=0,
                    )
                    logger.debug(f"[color channel ranges] {pairs.tolist()}")

                row_mask, report = apply_common_filters(
                    data,
                    mjd,
                    candidate_error,
                    ssc,
                    max_chi2,
                    min_snr,
                    candidate_rate,
                    label="bands-long",
                    logger=logger,
                    zero_cut=zero_cut,
                )
                row_mask &= np.isfinite(candidate_rate)
                if not zero_cut:
                    row_mask &= np.isfinite(candidate_error)
                pivoted = pivot_long_format_bands(
                    data, mjd, candidate_rate, candidate_error, band_index, row_mask
                )
                report.final_rows = int(pivoted.mjd.size)
                return pivoted, report

            raise ValueError(
                "Unrecognized RATE/ERROR dimensions in color file: "
                f"RATE{candidate_rate.shape}, ERROR{candidate_error.shape}"
            )

        scalar_columns = scalar_band_columns(names)
        if scalar_columns is None:
            raise ValueError(f"Three-band rate/error columns were not recognized. Columns: {names}")
        rate_names, error_names = scalar_columns
        rate = np.column_stack([np.asarray(data[name], dtype=float) for name in rate_names])
        error = np.column_stack([np.asarray(data[name], dtype=float) for name in error_names])
        mjd = time_to_mjd(data[time_col], table.header, hdul[0].header)
        mask, report = apply_common_filters(
            data, mjd, error, ssc, max_chi2, min_snr, rate, label="bands", logger=logger, zero_cut=zero_cut
        )
        mask &= np.any(np.isfinite(rate), axis=1)
        if not zero_cut:
            mask &= np.any(np.isfinite(error), axis=1)
        rate_out, error_out = zeroed_values_to_nan(rate[mask], error[mask], zero_cut)
        report.final_rows = int(mask.sum())
        return Series(mjd[mask], rate_out, error_out), report


def weighted_bin(series: Series, bin_days: float) -> Series:
    if bin_days <= 0:
        order = np.argsort(series.mjd)
        return Series(series.mjd[order], series.rate[order], series.error[order])

    mjd = np.asarray(series.mjd, dtype=float)
    rate = np.asarray(series.rate, dtype=float)
    error = np.asarray(series.error, dtype=float)
    if mjd.size == 0:
        return Series(mjd, rate, error)

    origin = np.floor(np.nanmin(mjd) / bin_days) * bin_days
    index = np.floor((mjd - origin) / bin_days).astype(np.int64)
    unique = np.unique(index)
    centers = origin + (unique + 0.5) * bin_days

    scalar = rate.ndim == 1
    if scalar:
        rate = rate[:, None]
        error = error[:, None]

    binned_rate = np.full((unique.size, rate.shape[1]), np.nan)
    binned_error = np.full_like(binned_rate, np.nan)

    for row, group in enumerate(unique):
        selected = index == group
        for channel in range(rate.shape[1]):
            r = rate[selected, channel]
            e = error[selected, channel]
            valid = np.isfinite(r) & np.isfinite(e) & (e > 0)
            if not np.any(valid):
                continue
            weight = 1.0 / np.square(e[valid])
            binned_rate[row, channel] = np.sum(weight * r[valid]) / np.sum(weight)
            binned_error[row, channel] = np.sqrt(1.0 / np.sum(weight))

    if scalar:
        keep = np.isfinite(binned_rate[:, 0]) & np.isfinite(binned_error[:, 0])
        return Series(centers[keep], binned_rate[keep, 0], binned_error[keep, 0])

    keep = np.any(np.isfinite(binned_rate) & np.isfinite(binned_error), axis=1)
    return Series(centers[keep], binned_rate[keep], binned_error[keep])


def mjd_to_datetime(mjd: np.ndarray) -> np.ndarray:
    return Time(mjd, format="mjd", scale="utc").to_datetime()


def select_date_range(series: Series, start: str | None, stop: str | None) -> Series:
    mask = np.ones(series.mjd.size, dtype=bool)
    if start:
        mask &= series.mjd >= Time(start, scale="utc").mjd
    if stop:
        mask &= series.mjd < Time(stop, scale="utc").mjd
    return Series(series.mjd[mask], series.rate[mask], series.error[mask])


def _safe_stats(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.nanmin(finite)), float(np.nanmedian(finite)), float(np.nanmax(finite))


def _utc_range_string(mjd: np.ndarray) -> tuple[str, str, float]:
    if mjd.size == 0:
        return "N/A", "N/A", np.nan
    start = Time(np.nanmin(mjd), format="mjd", scale="utc").isot
    stop = Time(np.nanmax(mjd), format="mjd", scale="utc").isot
    duration = float(np.nanmax(mjd) - np.nanmin(mjd))
    return start, stop, duration


def summarize_scalar_series(
    name: str,
    before_date: Series,
    after_date: Series,
    after_bin: Series,
    large_error_threshold: float,
) -> SummaryReport:
    if after_bin.mjd.size == 0:
        raise ValueError(f"No data remained for summary: {name}")
    relerr = np.full(after_bin.rate.shape, np.nan)
    valid = np.isfinite(after_bin.rate) & np.isfinite(after_bin.error) & (after_bin.rate != 0)
    relerr[valid] = np.abs(after_bin.error[valid] / after_bin.rate[valid])
    large = np.isfinite(relerr) & (relerr > large_error_threshold)
    y_min, y_med, y_max = _safe_stats(after_bin.rate)
    err_med = float(np.nanmedian(after_bin.error))
    err_max = float(np.nanmax(after_bin.error))
    start_utc, stop_utc, duration_days = _utc_range_string(after_bin.mjd)
    return SummaryReport(
        name=name,
        n_before_date=int(before_date.mjd.size),
        n_after_date=int(after_date.mjd.size),
        n_after_bin=int(after_bin.mjd.size),
        start_utc=start_utc,
        stop_utc=stop_utc,
        duration_days=duration_days,
        y_min=y_min,
        y_med=y_med,
        y_max=y_max,
        err_med=err_med,
        err_max=err_max,
        large_error_count=int(np.sum(large)),
        large_error_fraction=float(np.mean(large)) if large.size > 0 else np.nan,
        large_error_threshold=large_error_threshold,
        extra_lines=[],
    )


def summarize_band_series(
    name: str,
    before_date: Series,
    after_date: Series,
    after_bin: Series,
    large_error_threshold: float,
) -> SummaryReport:
    if after_bin.mjd.size == 0:
        raise ValueError(f"No data remained for summary: {name}")
    if after_bin.rate.ndim != 2 or after_bin.rate.shape[1] < 3:
        raise ValueError("Band summary requires 3 synchronized bands.")

    band_names = ["A", "B", "C"]
    extra_lines: list[str] = []
    all_large = np.zeros(after_bin.mjd.size, dtype=bool)
    all_rate = []
    all_err = []
    for i, band_name in enumerate(band_names):
        rate = after_bin.rate[:, i]
        err = after_bin.error[:, i]
        relerr = np.full(rate.shape, np.nan)
        valid = np.isfinite(rate) & np.isfinite(err) & (rate != 0)
        relerr[valid] = np.abs(err[valid] / rate[valid])
        large = np.isfinite(relerr) & (relerr > large_error_threshold)
        all_large |= large
        rmin, rmed, rmax = _safe_stats(rate)
        emed = float(np.nanmedian(err))
        emax = float(np.nanmax(err))
        extra_lines.append(
            f"  {band_name} band rate min/med/max     : {rmin:.6g} / {rmed:.6g} / {rmax:.6g} count/s"
        )
        extra_lines.append(
            f"  {band_name} band err  med/max         : {emed:.6g} / {emax:.6g} count/s"
        )
        extra_lines.append(
            f"  {band_name} band frac.err > {large_error_threshold:g}: {int(np.sum(large))} / {rate.size}"
        )
        all_rate.append(rate)
        all_err.append(err)

    stacked_rate = np.concatenate(all_rate)
    stacked_err = np.concatenate(all_err)
    y_min, y_med, y_max = _safe_stats(stacked_rate)
    err_med = float(np.nanmedian(stacked_err))
    err_max = float(np.nanmax(stacked_err))
    start_utc, stop_utc, duration_days = _utc_range_string(after_bin.mjd)
    return SummaryReport(
        name=name,
        n_before_date=int(before_date.mjd.size),
        n_after_date=int(after_date.mjd.size),
        n_after_bin=int(after_bin.mjd.size),
        start_utc=start_utc,
        stop_utc=stop_utc,
        duration_days=duration_days,
        y_min=y_min,
        y_med=y_med,
        y_max=y_max,
        err_med=err_med,
        err_max=err_max,
        large_error_count=int(np.sum(all_large)),
        large_error_fraction=float(np.mean(all_large)) if all_large.size > 0 else np.nan,
        large_error_threshold=large_error_threshold,
        extra_lines=extra_lines,
    )


def print_filter_report(report: FilterReport, logger: Logger) -> None:
    logger.info(f"\n[filter summary] {report.label}")
    logger.info(f"  raw rows in FITS                : {report.total_rows}")
    logger.info(f"  after finite TIME              : {report.after_time}")
    logger.info(f"  after finite ERROR             : {report.after_error}")
    if report.after_zero is not None:
        logger.info(f"  after --zero-cut RATE/ERROR   : {report.after_zero}")
    if report.after_ssc is not None:
        logger.info(f"  after --ssc cut                : {report.after_ssc}")
    if report.after_chi2 is not None:
        logger.info(f"  after --max-chi2 cut           : {report.after_chi2}")
    if report.after_snr is not None:
        logger.info(f"  after --min-snr cut            : {report.after_snr}")
    logger.info(f"  final rows kept                : {report.final_rows}")


def print_summary_report(summary: SummaryReport, logger: Logger) -> None:
    logger.info(f"\n[summary] {summary.name}")
    logger.info(f"  points before date selection   : {summary.n_before_date}")
    logger.info(f"  points after date selection    : {summary.n_after_date}")
    logger.info(f"  points after binning           : {summary.n_after_bin}")
    logger.info(f"  UTC start                      : {summary.start_utc}")
    logger.info(f"  UTC stop                       : {summary.stop_utc}")
    logger.info(f"  duration                       : {summary.duration_days:.6g} days")
    logger.info(f"  value min/med/max              : {summary.y_min:.6g} / {summary.y_med:.6g} / {summary.y_max:.6g}")
    logger.info(f"  error median/max               : {summary.err_med:.6g} / {summary.err_max:.6g}")
    logger.info(
        f"  frac.err > {summary.large_error_threshold:g} : "
        f"{summary.large_error_count} / {summary.n_after_bin} "
        f"({100.0 * summary.large_error_fraction:.2f}%)"
    )
    for line in summary.extra_lines:
        logger.info(line)


def plot_products(
    source_label: str,
    total: Series,
    bands: Series,
    bin_days: float,
    output: Path,
    show: bool,
    logger: Logger,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    total_dates = mjd_to_datetime(total.mjd)
    axes[0].errorbar(total_dates, total.rate, yerr=total.error, fmt=".", ms=2, lw=0.5)
    axes[0].set_ylabel("Total rate\n[count s$^{-1}$]")
    axes[0].grid(alpha=0.25)

    band_dates = mjd_to_datetime(bands.mjd)
    for channel, label in enumerate(BAND_LABELS):
        axes[1].errorbar(
            band_dates,
            bands.rate[:, channel],
            yerr=bands.error[:, channel],
            fmt=".",
            ms=2,
            lw=0.5,
            label=label,
        )
    axes[1].set_ylabel("Band rate\n[count s$^{-1}$]")
    axes[1].legend(ncol=3, fontsize="small")
    axes[1].grid(alpha=0.25)

    b = bands.rate[:, 1]
    c = bands.rate[:, 2]
    eb = bands.error[:, 1]
    ec = bands.error[:, 2]
    valid = (
        np.isfinite(b)
        & np.isfinite(c)
        & np.isfinite(eb)
        & np.isfinite(ec)
        & (b > 0)
        & (c > 0)
        & (eb > 0)
        & (ec > 0)
    )
    hardness = c[valid] / b[valid]
    hardness_error = hardness * np.sqrt(np.square(ec[valid] / c[valid]) + np.square(eb[valid] / b[valid]))
    axes[2].errorbar(band_dates[valid], hardness, yerr=hardness_error, fmt=".", ms=2, lw=0.5)
    axes[2].set_ylabel("Hardness C/B")
    axes[2].set_xlabel("Date (UTC)")
    axes[2].grid(alpha=0.25)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    cadence = "dwell" if bin_days <= 0 else f"{bin_days:g}-day inverse-variance bins"
    axes[0].set_title(f"RXTE/ASM definitive 1-dwell products: {source_label} ({cadence})")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    logger.info(f"[saved] {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def ratio_with_error(
    numerator: np.ndarray,
    numerator_error: np.ndarray,
    denominator: np.ndarray,
    denominator_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.asarray(numerator, dtype=float)
    numerator_error = np.asarray(numerator_error, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    denominator_error = np.asarray(denominator_error, dtype=float)

    ratio = np.full_like(numerator, np.nan, dtype=float)
    error = np.full_like(numerator, np.nan, dtype=float)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(numerator_error)
        & np.isfinite(denominator)
        & np.isfinite(denominator_error)
        & (numerator > 0)
        & (denominator > 0)
        & (numerator_error > 0)
        & (denominator_error > 0)
    )
    ratio[valid] = numerator[valid] / denominator[valid]
    error[valid] = ratio[valid] * np.sqrt(
        np.square(numerator_error[valid] / numerator[valid])
        + np.square(denominator_error[valid] / denominator[valid])
    )
    return ratio, error


def make_diagram_series(
    bands: Series,
    minimum_band_snr: float | None = None,
    maximum_relative_error: float | None = None,
    summary_large_relative_error: float = 0.5,
) -> tuple[DiagramSeries, DiagramReport]:
    rate = np.asarray(bands.rate, dtype=float)
    error = np.asarray(bands.error, dtype=float)
    if rate.ndim != 2 or error.ndim != 2 or rate.shape[1] < 3 or error.shape[1] < 3:
        raise ValueError("HID/CCD generation requires three synchronized A/B/C band columns.")

    a, b, c = rate[:, 0], rate[:, 1], rate[:, 2]
    ea, eb, ec = error[:, 0], error[:, 1], error[:, 2]

    intensity = a + b + c
    intensity_error = np.sqrt(np.square(ea) + np.square(eb) + np.square(ec))
    soft_color, soft_color_error = ratio_with_error(b, eb, a, ea)
    hard_color, hard_color_error = ratio_with_error(c, ec, b, eb)

    band_positive = (a > 0) & (b > 0) & (c > 0) & (ea > 0) & (eb > 0) & (ec > 0)
    basic = (
        np.isfinite(bands.mjd)
        & np.isfinite(intensity)
        & np.isfinite(intensity_error)
        & np.isfinite(soft_color)
        & np.isfinite(soft_color_error)
        & np.isfinite(hard_color)
        & np.isfinite(hard_color_error)
        & band_positive
        & (intensity > 0)
        & (intensity_error > 0)
        & (soft_color > 0)
        & (hard_color > 0)
    )

    after_snr = None
    valid = basic.copy()
    if minimum_band_snr is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_keep = (
                (a / ea >= minimum_band_snr)
                & (b / eb >= minimum_band_snr)
                & (c / ec >= minimum_band_snr)
            )
        valid &= snr_keep
        after_snr = int(valid.sum())

    after_relerr = None
    if maximum_relative_error is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_keep = (
                (intensity_error / intensity <= maximum_relative_error)
                & (soft_color_error / soft_color <= maximum_relative_error)
                & (hard_color_error / hard_color <= maximum_relative_error)
            )
        valid &= rel_keep
        after_relerr = int(valid.sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        intensity_rel = intensity_error / intensity
        soft_rel = soft_color_error / soft_color
        hard_rel = hard_color_error / hard_color
    basic_large_int = int(np.sum(basic & np.isfinite(intensity_rel) & (intensity_rel > summary_large_relative_error)))
    basic_large_soft = int(np.sum(basic & np.isfinite(soft_rel) & (soft_rel > summary_large_relative_error)))
    basic_large_hard = int(np.sum(basic & np.isfinite(hard_rel) & (hard_rel > summary_large_relative_error)))

    if int(np.sum(valid)) == 0:
        raise ValueError("No valid HID/CCD points remained after the diagram cuts.")

    diagram = DiagramSeries(
        mjd=np.asarray(bands.mjd[valid], dtype=float),
        intensity=np.asarray(intensity[valid], dtype=float),
        intensity_error=np.asarray(intensity_error[valid], dtype=float),
        soft_color=np.asarray(soft_color[valid], dtype=float),
        soft_color_error=np.asarray(soft_color_error[valid], dtype=float),
        hard_color=np.asarray(hard_color[valid], dtype=float),
        hard_color_error=np.asarray(hard_color_error[valid], dtype=float),
    )
    report = DiagramReport(
        n_input=int(rate.shape[0]),
        n_basic_valid=int(np.sum(basic)),
        n_after_snr=after_snr,
        n_after_relerr=after_relerr,
        n_final=int(np.sum(valid)),
        intensity_min=float(np.nanmin(diagram.intensity)),
        intensity_max=float(np.nanmax(diagram.intensity)),
        soft_min=float(np.nanmin(diagram.soft_color)),
        soft_max=float(np.nanmax(diagram.soft_color)),
        hard_min=float(np.nanmin(diagram.hard_color)),
        hard_max=float(np.nanmax(diagram.hard_color)),
        large_relerr_threshold=summary_large_relative_error,
        large_relerr_intensity=basic_large_int,
        large_relerr_soft=basic_large_soft,
        large_relerr_hard=basic_large_hard,
    )
    return diagram, report


def print_diagram_report(report: DiagramReport, logger: Logger) -> None:
    logger.info("\n[summary] HID / CCD")
    logger.info(f"  input synchronized points       : {report.n_input}")
    logger.info(f"  basic valid points              : {report.n_basic_valid}")
    if report.n_after_snr is not None:
        logger.info(f"  after --diagram-min-band-snr    : {report.n_after_snr}")
    if report.n_after_relerr is not None:
        logger.info(f"  after --diagram-max-relative-error : {report.n_after_relerr}")
    logger.info(f"  final plotted points            : {report.n_final}")
    logger.info(f"  intensity min/max               : {report.intensity_min:.6g} / {report.intensity_max:.6g}")
    logger.info(f"  soft color B/A min/max          : {report.soft_min:.6g} / {report.soft_max:.6g}")
    logger.info(f"  hard color C/B min/max          : {report.hard_min:.6g} / {report.hard_max:.6g}")
    logger.info(
        f"  basic valid points with frac.err > {report.large_relerr_threshold:g}: "
        f"intensity={report.large_relerr_intensity}, "
        f"soft={report.large_relerr_soft}, hard={report.large_relerr_hard}"
    )


def plot_hid_ccd(
    source_label: str,
    bands: Series,
    bin_days: float,
    output: Path,
    show: bool,
    logger: Logger,
    errorbar_mode: str = "auto",
    max_errorbar_points: int = 5000,
    minimum_band_snr: float | None = None,
    maximum_relative_error: float | None = None,
    summary_large_relative_error: float = 0.5,
) -> None:
    diagram, report = make_diagram_series(
        bands,
        minimum_band_snr=minimum_band_snr,
        maximum_relative_error=maximum_relative_error,
        summary_large_relative_error=summary_large_relative_error,
    )
    print_diagram_report(report, logger)

    npoint = diagram.mjd.size
    if errorbar_mode == "on":
        use_errorbars = True
    elif errorbar_mode == "off":
        use_errorbars = False
    else:
        use_errorbars = npoint <= max_errorbar_points

    if errorbar_mode == "auto" and not use_errorbars:
        logger.info(f"  error bars                      : omitted automatically (limit={max_errorbar_points})")
    else:
        logger.info(f"  error bars                      : {'on' if use_errorbars else 'off'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    if use_errorbars:
        axes[0].errorbar(
            diagram.intensity,
            diagram.hard_color,
            xerr=diagram.intensity_error,
            yerr=diagram.hard_color_error,
            fmt=".",
            ms=2.5,
            lw=0.35,
            elinewidth=0.35,
            capsize=0,
            alpha=0.65,
        )
        axes[1].errorbar(
            diagram.soft_color,
            diagram.hard_color,
            xerr=diagram.soft_color_error,
            yerr=diagram.hard_color_error,
            fmt=".",
            ms=2.5,
            lw=0.35,
            elinewidth=0.35,
            capsize=0,
            alpha=0.65,
        )
    else:
        axes[0].plot(diagram.intensity, diagram.hard_color, ".", ms=2.5, alpha=0.65)
        axes[1].plot(diagram.soft_color, diagram.hard_color, ".", ms=2.5, alpha=0.65)

    axes[0].set_xlabel("A + B + C intensity [count s$^{-1}$]")
    axes[0].set_ylabel("Hard color C/B")
    axes[0].set_title("Hardness-intensity diagram")
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("Soft color B/A")
    axes[1].set_ylabel("Hard color C/B")
    axes[1].set_title("Color-color diagram")
    axes[1].grid(alpha=0.25)

    cadence = "dwell" if bin_days <= 0 else f"{bin_days:g}-day inverse-variance bins"
    fig.suptitle(
        f"RXTE/ASM {source_label}: HID and CCD ({cadence})\n"
        "A=1.3-3.0 keV, B=3.0-5.0 keV, C=5.0-12.1 keV"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    logger.info(f"[saved] {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def diagram_output_from_timeseries(output: Path) -> Path:
    suffix = output.suffix if output.suffix else ".png"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}_hid_ccd{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and plot RXTE/ASM definitive light-curve and color products."
    )
    parser.add_argument("--source", default="gx5-1", help="HEASARC source token, e.g. gx5-1")
    parser.add_argument("--data-dir", type=Path, default=Path("rxte_asm_data"))
    parser.add_argument(
        "--bin-days",
        type=float,
        default=1.0,
        help="Time-series bin width in days; use 0 for dwell data.",
    )
    parser.add_argument(
        "--diagram-bin-days",
        type=float,
        default=None,
        help="Optional HID/CCD bin width in days. Default: use --bin-days.",
    )
    parser.add_argument(
        "--ssc",
        type=int,
        default=None,
        help="Optional SSC_NUMBER selection, usually 1, 2, 3, or 12",
    )
    parser.add_argument("--max-chi2", type=float, default=None, help="Optional fit-quality cut")
    parser.add_argument(
        "--min-snr",
        type=float,
        default=None,
        help="Optional S/N cut before binning. Avoid this for unbiased flux averages.",
    )
    parser.add_argument(
        "--zero-cut",
        dest="zero_cut",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop or mask non-positive RATE/ERROR samples before higher-level processing. "
            "Default: on. Use --no-zero-cut to preserve zero-valued samples where possible."
        ),
    )
    parser.add_argument("--start", help="Start date, e.g. 2009-01-01")
    parser.add_argument("--stop", help="Exclusive stop date, e.g. 2012-01-01")
    parser.add_argument("--overwrite", action="store_true", help="Redownload cached files")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Time-series output image. Default: SOURCE_rxte_asm.png",
    )
    parser.add_argument(
        "--diagram-output",
        type=Path,
        default=None,
        help="HID/CCD output image. Default: TIME_SERIES_STEM_hid_ccd.png",
    )
    parser.add_argument(
        "--no-diagrams",
        action="store_true",
        help="Do not generate the automatic hardness-intensity/color-color figure.",
    )
    parser.add_argument(
        "--diagram-errorbars",
        choices=("auto", "on", "off"),
        default="auto",
        help="Draw HID/CCD error bars. auto omits them for very large data sets.",
    )
    parser.add_argument(
        "--diagram-max-errorbar-points",
        type=int,
        default=5000,
        help="Maximum number of points for error bars in auto mode.",
    )
    parser.add_argument(
        "--diagram-min-band-snr",
        type=float,
        default=None,
        help="Optional minimum S/N required independently in A, B, and C.",
    )
    parser.add_argument(
        "--diagram-max-relative-error",
        type=float,
        default=None,
        help="Optional maximum fractional error for intensity and both colors.",
    )
    parser.add_argument(
        "--summary-large-relative-error",
        type=float,
        default=0.5,
        help="Threshold used in summary statistics for 'large fractional error'. Default: 0.5",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose FITS/debug output")
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = Logger(debug_enabled=args.debug)

    source = normalize_source_name(args.source)
    lc_name = f"xa_{source}_d1.lc"
    col_name = f"xa_{source}_d1.col"
    lc_path = args.data_dir / lc_name
    col_path = args.data_dir / col_name
    diagram_bin_days = args.bin_days if args.diagram_bin_days is None else args.diagram_bin_days

    try:
        download_file(
            f"{BASE_URL}/lightcurves/{urllib.parse.quote(lc_name, safe='+-._')}",
            lc_path,
            logger,
            args.overwrite,
        )
        download_file(
            f"{BASE_URL}/colors/{urllib.parse.quote(col_name, safe='+-._')}",
            col_path,
            logger,
            args.overwrite,
        )

        logger.debug(f"\n[read total] {lc_path}")
        total_raw, total_filter_report = read_total(
            lc_path, args.ssc, args.max_chi2, args.min_snr, logger, zero_cut=args.zero_cut
        )
        logger.debug(f"\n[read bands] {col_path}")
        bands_raw, bands_filter_report = read_bands(
            col_path, args.ssc, args.max_chi2, args.min_snr, logger, zero_cut=args.zero_cut
        )

        print_filter_report(total_filter_report, logger)
        print_filter_report(bands_filter_report, logger)

        total_date = select_date_range(total_raw, args.start, args.stop)
        bands_date = select_date_range(bands_raw, args.start, args.stop)
        total_bin = weighted_bin(total_date, args.bin_days)
        bands_bin = weighted_bin(bands_date, args.bin_days)
        bands_diagram = weighted_bin(bands_date, diagram_bin_days)

        if total_bin.mjd.size == 0 or bands_bin.mjd.size == 0:
            raise ValueError("No data remained after filtering/date selection/binning.")
        if not args.no_diagrams and bands_diagram.mjd.size == 0:
            raise ValueError("No band data remained for HID/CCD after diagram binning.")

        total_summary = summarize_scalar_series(
            "total light curve",
            total_raw,
            total_date,
            total_bin,
            args.summary_large_relative_error,
        )
        bands_summary = summarize_band_series(
            "A/B/C band light curves",
            bands_raw,
            bands_date,
            bands_bin,
            args.summary_large_relative_error,
        )
        print_summary_report(total_summary, logger)
        print_summary_report(bands_summary, logger)
        logger.info(
            f"\n[binning] time-series bin width = {args.bin_days:g} day(s), "
            f"HID/CCD bin width = {diagram_bin_days:g} day(s)"
        )
        logger.info(f"[zero cut] {'on' if args.zero_cut else 'off'}")

        output = args.output or Path(f"{source}_rxte_asm.png")
        plot_products(args.source, total_bin, bands_bin, args.bin_days, output, args.show, logger)

        if not args.no_diagrams:
            diagram_output = args.diagram_output or diagram_output_from_timeseries(output)
            plot_hid_ccd(
                source_label=args.source,
                bands=bands_diagram,
                bin_days=diagram_bin_days,
                output=diagram_output,
                show=args.show,
                logger=logger,
                errorbar_mode=args.diagram_errorbars,
                max_errorbar_points=args.diagram_max_errorbar_points,
                minimum_band_snr=args.diagram_min_band_snr,
                maximum_relative_error=args.diagram_max_relative_error,
                summary_large_relative_error=args.summary_large_relative_error,
            )

    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
