#!/usr/bin/env python3
"""
Autorun utility for XRISM Xtend pileup / quick-look calibration products.

This script searches cleaned Xtend event files for a given OBSID and runs a
standard chain:

  1. pileup check
  2. region generation
  3. PHA/RMF/ARF generation
  4. ARF and spectrum quick checks
  5. optional HTML report generation

It is intended to be executed from the directory that contains the OBSID
subdirectory, e.g.:

  xtend_autorun_docal.py 000125000
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Iterable, Sequence

TOPDIR = Path.cwd()
DEFAULT_OUTPUT_SUBDIR = "checkpileup_std"
DEFAULT_HTML_KEYWORD = "checkpileup_"
DEFAULT_HTML_VER = "v0"

STEP_TOOLS: dict[int, tuple[str, ...]] = {
    1: ("xtend_pileup_check_quick.sh",),
    2: ("xtend_util_genregion.py",),
    3: ("xtend_auto_gen_phaarfrmf.py",),
    4: ("xrism_util_plot_arf.py", "xrism_spec_qlfit_many.py"),
}
HTML_TOOL = "xrism_autorun_png2html.py"


class ConsoleColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Tee:
    """Write output to multiple streams, e.g. terminal and log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_stdout_stderr(log_path: Path):
    """Mirror stdout/stderr into a timestamped log file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = Tee(original_stderr, log_file)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def compact_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def color_print(message: str, color: str) -> None:
    print(f"{color}{message}{ConsoleColors.ENDC}")


def write_to_file(filename: Path | str, content: Iterable[str] | str) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fobj:
        if isinstance(content, str):
            fobj.write(content.rstrip("\n") + "\n")
        else:
            for one in content:
                fobj.write(str(one).rstrip("\n") + "\n")


def strip_gz_extension(filename: str | Path) -> str:
    """Return the filename string without a trailing .gz, if present."""

    text = str(filename)
    if text.endswith(".gz"):
        return text[:-3]
    return text


def resolve_optional_gzip(filepath: Path, *, require_uncompressed: bool = False) -> Path:
    """Resolve an input file that may exist as uncompressed or .gz.

    Policy:
      - If both exist, prefer the uncompressed file.
      - If only .gz exists, use it by default and print a warning.
      - If require_uncompressed=True, fail when only .gz exists.
    """

    gz_path = Path(str(filepath) + ".gz")
    if filepath.exists():
        return filepath
    if gz_path.exists():
        if require_uncompressed:
            raise FileNotFoundError(
                f"Only gzip-compressed file exists, but --require-uncompressed was set: {gz_path}"
            )
        color_print(
            f"[GZIP WARNING] Using compressed input: {gz_path}. "
            "Downstream tools must be able to read gzip-compressed FITS files.",
            ConsoleColors.WARNING,
        )
        return gz_path
    raise FileNotFoundError(f"File not found: {filepath} or {gz_path}")


def check_program_in_path(program_name: str) -> Path:
    program_path = shutil.which(program_name)
    if program_path is None:
        raise FileNotFoundError(f"Required program not found in $PATH: {program_name}")
    print(f"[PATH OK] {program_name}: {program_path}")
    return Path(program_path)


def required_programs_for(steps: Sequence[int], generate_html: bool) -> list[str]:
    programs: list[str] = []
    seen: set[str] = set()
    for step in sorted(set(steps)):
        for program in STEP_TOOLS[step]:
            if program not in seen:
                programs.append(program)
                seen.add(program)
    if generate_html and HTML_TOOL not in seen:
        programs.append(HTML_TOOL)
    return programs


def preflight_check_programs(steps: Sequence[int], generate_html: bool) -> None:
    print("\n[Preflight] Checking required helper programs in $PATH")
    missing: list[str] = []
    for program in required_programs_for(steps, generate_html):
        try:
            check_program_in_path(program)
        except FileNotFoundError as exc:
            color_print(str(exc), ConsoleColors.FAIL)
            missing.append(program)

    if missing:
        raise RuntimeError(
            "Missing required helper programs in $PATH: " + ", ".join(missing)
        )
    print("[Preflight] All required helper programs were found.\n")


def run_command(command: Sequence[str], *, on_error: str) -> bool:
    """Run a subprocess command and either stop or continue on failure."""

    program = command[0]
    print(f"[START:{now_string()}] >>> {program} <<<")
    print("Executing command: " + " ".join(command))
    try:
        subprocess.run(list(command), check=True)
    except subprocess.CalledProcessError as exc:
        message = f"Command failed with exit status {exc.returncode}: {' '.join(command)}"
        if on_error == "stop":
            raise RuntimeError(message) from exc
        color_print("[COMMAND WARNING] " + message, ConsoleColors.WARNING)
        print(f"[END:{now_string()}] >>> {program} <<<\n")
        return False

    print(f"[END:{now_string()}] >>> {program} <<<\n")
    return True


def make_symlink(source: Path, link_dir: Path) -> None:
    """Create or refresh a symlink in link_dir pointing to source."""

    link_path = link_dir / source.name

    if link_path.is_symlink():
        link_path.unlink()
        print(f"Removed existing symbolic link: {link_path.name}")
    elif link_path.exists():
        color_print(
            f"[LINK WARNING] {link_path.name} already exists and is not a symlink; keeping it.",
            ConsoleColors.WARNING,
        )
        return

    os.symlink(source, link_path)
    print(f"Created symbolic link: {link_path.name} -> {source}")


def dojob(
    runprog: str,
    arguments: Sequence[str] | None = None,
    *,
    gdir: Path | str,
    subdir: str | None = None,
    linkfiles: Sequence[Path] | None = None,
    on_error: str = "stop",
) -> bool:
    """Change into a work directory, create symlinks, and run a helper command."""

    gotodir = Path(gdir)
    if not gotodir.is_absolute():
        gotodir = TOPDIR / gotodir

    if not gotodir.exists():
        raise FileNotFoundError(f"The directory does not exist: {gotodir}")

    workdir = gotodir if subdir is None else gotodir / subdir
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {workdir}")

    previous_dir = Path.cwd()
    os.chdir(workdir)
    try:
        if linkfiles:
            for source in linkfiles:
                make_symlink(Path(source).resolve(), workdir)

        command = [runprog]
        if arguments:
            command.extend([str(arg) for arg in arguments])
        return run_command(command, on_error=on_error)
    finally:
        os.chdir(previous_dir)


def extract_data_class_and_ccd(filename: str | Path) -> tuple[str | None, str | None]:
    """Extract data class and infer included CCDs from an Xtend event filename.

    Example filename:
      xa300046010xtd_p0300000a0_cl.evt

    The script uses the 8 characters after `_p.` as data class and checks the
    second character to infer the CCD combination.
    """

    basename = Path(strip_gz_extension(filename)).name
    match = re.search(r"_p.([A-Za-z0-9]{8})", basename)
    if not match:
        color_print(f"Invalid filename format: {basename}", ConsoleColors.FAIL)
        return None, None

    data_class = match.group(1)
    second_digit = data_class[1]

    ccd_mapping = {
        "0": "CCD1,2,3,4",
        "1": "CCD1,2",
        "2": "CCD3,4",
    }
    included_ccds = ccd_mapping.get(second_digit, "Unknown")
    return data_class, included_ccds


def get_sorted_pha_files(gotodir: Path | str, nametag: str = "gopt") -> list[str]:
    files = glob.glob(str(Path(gotodir) / f"*_{nametag}.pha"))
    if not files:
        raise FileNotFoundError(f"No matching *_{nametag}.pha files found in: {gotodir}")

    sorted_files = sorted(files, key=os.path.getctime)
    return [Path(file).name for file in sorted_files]


def find_cleaned_event_files(
    obsid: str,
    event_cl_dir: Path,
    *,
    require_uncompressed: bool = False,
) -> list[Path]:
    """Find cleaned event files, avoiding duplicate .evt/.evt.gz processing.

    If both .evt and .evt.gz exist for the same logical file, the uncompressed
    .evt file is used. If only .evt.gz exists, it is used unless
    --require-uncompressed was requested.
    """

    pattern = str(event_cl_dir / f"xa{obsid}*_cl.evt*")
    raw_files = sorted(Path(path) for path in glob.glob(pattern))

    selected: dict[str, Path] = {}
    for path in raw_files:
        base_key = strip_gz_extension(path)
        if str(path).endswith(".gz"):
            uncompressed = Path(base_key)
            if uncompressed.exists():
                selected[base_key] = uncompressed
            elif require_uncompressed:
                raise FileNotFoundError(
                    f"Only gzip-compressed cleaned event exists, but --require-uncompressed was set: {path}"
                )
            else:
                selected[base_key] = path
        else:
            selected[base_key] = path

    files = sorted(selected.values())
    for path in files:
        if str(path).endswith(".gz"):
            color_print(
                f"[GZIP WARNING] Using compressed cleaned event: {path}. "
                "All downstream tools must accept .evt.gz input.",
                ConsoleColors.WARNING,
            )
    return files


def default_log_path(obsid: str, output_subdir: str, log_dir: str | None) -> Path:
    filename = f"xtend_autorun_docal_{obsid}_{compact_timestamp()}.log"
    if log_dir:
        return Path(log_dir).expanduser().resolve() / filename

    event_cl_dir = TOPDIR / obsid / "xtend" / "event_cl"
    if event_cl_dir.exists():
        return event_cl_dir / output_subdir / "logs" / filename
    return TOPDIR / "logs" / filename


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Xtend pileup check, region generation, response generation, and quick-look checks."
    )
    parser.add_argument("obsid", help="OBSID, e.g. 000125000")
    parser.add_argument(
        "--steps",
        "-s",
        nargs="*",
        type=int,
        choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        help=(
            "Specify steps to execute: "
            "1=pileup check, 2=create region, 3=create PHA/RMF/ARF, "
            "4=check PHA/RMF/ARF. Default: all steps."
        ),
    )
    parser.add_argument(
        "-n",
        "--numphoton",
        type=int,
        default=1_000_000,
        help="Number of photons for ARF generation. Default: 1000000.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation. By default, HTML generation is enabled.",
    )
    parser.add_argument(
        "--genhtml",
        "-html",
        dest="legacy_no_html",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--clobber",
        "-c",
        choices=["yes", "no"],
        default="no",
        help="Pass clobber flag to xtend_auto_gen_phaarfrmf.py. Default: no.",
    )
    parser.add_argument(
        "--on-error",
        choices=["stop", "continue"],
        default="stop",
        help=(
            "Behavior when a helper command fails. "
            "'stop' fails fast; 'continue' logs a warning and proceeds. Default: stop."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help=f"Working/output subdirectory under <OBSID>/xtend/event_cl/. Default: {DEFAULT_OUTPUT_SUBDIR}.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help=(
            "Directory for timestamped log files. "
            "Default: <OBSID>/xtend/event_cl/<output-subdir>/logs/."
        ),
    )
    parser.add_argument(
        "--require-uncompressed",
        action="store_true",
        help="Fail early if any required input is available only as .gz.",
    )
    parser.add_argument(
        "--html-keyword",
        default=DEFAULT_HTML_KEYWORD,
        help=f"Keyword passed to xrism_autorun_png2html.py. Default: {DEFAULT_HTML_KEYWORD}.",
    )
    parser.add_argument(
        "--html-ver",
        default=DEFAULT_HTML_VER,
        help=f"Version string passed to xrism_autorun_png2html.py. Default: {DEFAULT_HTML_VER}.",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> None:
    obsid = args.obsid
    steps = sorted(set(args.steps))
    generate_html = not args.no_html and not args.legacy_no_html

    if args.legacy_no_html:
        color_print(
            "[DEPRECATED OPTION] --genhtml/-html has been replaced by --no-html. "
            "For this run it is interpreted as --no-html.",
            ConsoleColors.WARNING,
        )

    print(f"TOPDIR: {TOPDIR}")
    print(f"OBSID: {obsid}")
    print(f"Steps: {steps}")
    print(f"Output subdirectory: {args.output_subdir}")
    print(f"HTML generation: {'enabled' if generate_html else 'disabled'}")
    print(f"Error behavior: {args.on_error}")
    print(f"Require uncompressed inputs: {args.require_uncompressed}")

    preflight_check_programs(steps, generate_html)

    event_cl_dir = TOPDIR / obsid / "xtend" / "event_cl"
    event_uf_dir = TOPDIR / obsid / "xtend" / "event_uf"
    auxil_dir = TOPDIR / obsid / "auxil"

    if not event_cl_dir.exists():
        raise FileNotFoundError(f"Xtend cleaned event directory does not exist: {event_cl_dir}")

    filenames = find_cleaned_event_files(
        obsid,
        event_cl_dir,
        require_uncompressed=args.require_uncompressed,
    )
    if not filenames:
        color_print(f"No cleaned .evt files found in {event_cl_dir}", ConsoleColors.WARNING)
        return

    ehk_path = resolve_optional_gzip(
        auxil_dir / f"xa{obsid}.ehk",
        require_uncompressed=args.require_uncompressed,
    )

    for event_path in filenames:
        clevt = event_path.name
        bimg_unzip_name = Path(strip_gz_extension(clevt)).name.replace("_cl.evt", ".bimg")
        bimg_path = resolve_optional_gzip(
            event_uf_dir / bimg_unzip_name,
            require_uncompressed=args.require_uncompressed,
        )
        bimg = bimg_path.name
        ehk = ehk_path.name

        print(f"\n{'=' * 80}")
        color_print(f"Processing file: {event_path}", ConsoleColors.OKCYAN)
        print(f"EHK file: {ehk_path}")
        print(f"BIMG file: {bimg_path}")

        data_class, ccd_info = extract_data_class_and_ccd(clevt)
        if data_class is None or ccd_info is None:
            continue

        color_print(f"  Data Class: {data_class}", ConsoleColors.OKGREEN)
        color_print(f"  Included CCDs: {ccd_info}", ConsoleColors.OKGREEN)

        if ccd_info not in {"CCD1,2,3,4", "CCD1,2"}:
            color_print("  Skipping analysis for this CCD combination.", ConsoleColors.WARNING)
            continue

        color_print("  Performing analysis...", ConsoleColors.OKBLUE)

        if 1 in steps:
            color_print("    Step 1: Check pileup", ConsoleColors.OKCYAN)
            dojob(
                "xtend_pileup_check_quick.sh",
                [clevt],
                gdir=event_cl_dir,
                subdir=args.output_subdir,
                linkfiles=[event_path],
                on_error=args.on_error,
            )

        if 2 in steps:
            color_print("    Step 2: Create region", ConsoleColors.OKCYAN)
            dojob(
                "xtend_util_genregion.py",
                [clevt],
                gdir=event_cl_dir,
                subdir=args.output_subdir,
                linkfiles=[event_path],
                on_error=args.on_error,
            )

        if 3 in steps:
            color_print("    Step 3: Create PHA, RMF, ARF", ConsoleColors.OKCYAN)
            dojob(
                "xtend_auto_gen_phaarfrmf.py",
                [clevt, "-e", ehk, "-b", bimg, "-n", str(args.numphoton), "-c", args.clobber],
                gdir=event_cl_dir,
                subdir=args.output_subdir,
                linkfiles=[event_path, ehk_path, bimg_path],
                on_error=args.on_error,
            )

        if 4 in steps:
            color_print("    Step 4: Check PHA, RMF, ARF", ConsoleColors.OKCYAN)
            dojob(
                "xrism_util_plot_arf.py",
                [],
                gdir=event_cl_dir,
                subdir=args.output_subdir,
                on_error=args.on_error,
            )

            workdir = event_cl_dir / args.output_subdir
            write_to_file(workdir / "f_gopt.list", get_sorted_pha_files(workdir))

            dojob(
                "xrism_spec_qlfit_many.py",
                ["f_gopt.list", "--fname", f"{obsid}_xtend_comp_all_in_out"],
                gdir=workdir,
                on_error=args.on_error,
            )
            dojob(
                "xrism_spec_qlfit_many.py",
                [
                    "f_gopt.list",
                    "--fname",
                    f"{obsid}_xtend_comp_all_in_out_narrow",
                    "--emin",
                    "6.0",
                    "--emax",
                    "7.5",
                    "--xscale",
                    "off",
                    "--progflags",
                    "1,1,1",
                ],
                gdir=workdir,
                on_error=args.on_error,
            )

    if generate_html:
        color_print("\nCreate HTML report", ConsoleColors.OKCYAN)
        run_command(
            [
                HTML_TOOL,
                obsid,
                "--keyword",
                args.html_keyword,
                "--ver",
                args.html_ver,
            ],
            on_error=args.on_error,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log_path = default_log_path(args.obsid, args.output_subdir, args.log_dir)

    with tee_stdout_stderr(log_path):
        print(f"Log file: {log_path}")
        print(f"Run started at: {now_string()}\n")
        try:
            run_pipeline(args)
        except Exception as exc:  # noqa: BLE001 - intentional top-level error handling
            color_print(f"\n[FATAL] {exc}", ConsoleColors.FAIL)
            traceback.print_exc()
            print(f"\nRun failed at: {now_string()}")
            print(f"Log file: {log_path}")
            return 1

        print(f"\nRun finished successfully at: {now_string()}")
        print(f"Log file: {log_path}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
