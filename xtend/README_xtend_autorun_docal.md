# XRISM Xtend pileup / quick-look autorun tool

`xtend_autorun_docal.py` is an autorun script for XRISM Xtend pileup / quick-look calibration products.  It searches cleaned Xtend event files for a given OBSID and runs a standard chain of pileup checking, region generation, PHA/RMF/ARF generation, quick-look response/spectrum checks, and optional HTML report generation.

This script is mainly intended for bright point-source Xtend checks, where one wants to inspect pileup effects and compare spectra extracted from standard regions in a reproducible way.

Related links:

- GitHub source: <https://github.com/yamadasuzaku/rksysoft/blob/main/xtend/xtend_autorun_docal.py>
- Qiita note: <https://qiita.com/yamadasuzaku/items/1d82c1a1c7d77127bc3d>

---

## 1. What this script does

For each matching Xtend cleaned event file under

```text
<OBSID>/xtend/event_cl/
```

this script can run the following steps.

| Step | Purpose | Helper program |
|---:|---|---|
| 1 | Check pileup | `xtend_pileup_check_quick.sh` |
| 2 | Generate extraction regions | `xtend_util_genregion.py` |
| 3 | Generate PHA/RMF/ARF | `xtend_auto_gen_phaarfrmf.py` |
| 4 | Check ARF and spectra | `xrism_util_plot_arf.py`, `xrism_spec_qlfit_many.py` |
| HTML | Generate an HTML summary page | `xrism_autorun_png2html.py` |

By default, all steps are executed and the HTML report is generated.

The default working/output directory is

```text
<OBSID>/xtend/event_cl/checkpileup_std/
```

This can be changed with `--output-subdir`.

---

## 2. Expected directory structure

Run this script from the directory that contains the OBSID directory.

Example:

```text
./
└── 000125000/
    ├── auxil/
    │   └── xa000125000.ehk or xa000125000.ehk.gz
    └── xtend/
        ├── event_cl/
        │   └── xa000125000*_cl.evt or xa000125000*_cl.evt.gz
        └── event_uf/
            └── *.bimg or *.bimg.gz
```

A typical run is therefore

```bash
cd /path/to/working_directory
xtend_autorun_docal.py 000125000
```

where `/path/to/working_directory/000125000/` exists.

---

## 3. Required environment

### 3.1 Python

The script is written for Python 3 and uses only standard-library modules.

### 3.2 HEASoft / XRISM software

The helper scripts eventually call HEASoft / XRISM tools such as event selection, response generation, ARF/RMF utilities, plotting scripts, and XSPEC-based quick-look fitting utilities.  Before running the autorun script, initialize the relevant analysis environment, for example:

```bash
source /path/to/headas-init.sh
source /path/to/caldbinit.sh
```

Use the correct paths for your local analysis environment.

### 3.3 rksysoft helper scripts in `$PATH`

The autorun script calls helper scripts by name.  Therefore, the relevant `rksysoft` directories must be included in `$PATH`.

Example:

```bash
export RESOLVETOOLS=$HOME/work/software/rksysoft

for subdir in resolve xtend xrism; do
    for dir in $(find "$RESOLVETOOLS/$subdir" -type d); do
        PATH="$dir:$PATH"
    done
done

export PATH
```

You can check the minimum dependency set for a full run with

```bash
which xtend_pileup_check_quick.sh
which xtend_util_genregion.py
which xtend_auto_gen_phaarfrmf.py
which xrism_util_plot_arf.py
which xrism_spec_qlfit_many.py
which xrism_autorun_png2html.py
```

The script also performs a preflight `$PATH` check before processing.  Only the programs needed for the requested `--steps` and HTML setting are checked.

---

## 4. Basic usage

### 4.1 Full default run

```bash
xtend_autorun_docal.py 000125000
```

This runs steps 1--4 and then generates the HTML report.

### 4.2 Run only selected steps

Run only pileup checking and region generation:

```bash
xtend_autorun_docal.py 000125000 -s 1 2
```

Regenerate spectra/responses and then check them:

```bash
xtend_autorun_docal.py 000125000 -s 3 4 -c yes
```

### 4.3 Change the ARF photon number

```bash
xtend_autorun_docal.py 000125000 -s 3 4 -n 1000000 -c yes
```

For quick testing, a smaller value may be useful:

```bash
xtend_autorun_docal.py 000125000 -s 3 4 -n 10000 -c yes
```

However, if the ARF generation fails or the ARF is too noisy, increase `--numphoton`.

### 4.4 Skip HTML generation

The old `--genhtml` option was confusing because it actually stopped HTML generation.  The official option is now `--no-html`.

```bash
xtend_autorun_docal.py 000125000 --no-html
```

For backward compatibility, the old `--genhtml` / `-html` option is still accepted as a deprecated alias for `--no-html`, but new scripts should not use it.

### 4.5 Use a different output directory

```bash
xtend_autorun_docal.py 000125000 --output-subdir checkpileup_test
```

The output will be written under

```text
<OBSID>/xtend/event_cl/checkpileup_test/
```

### 4.6 Continue even if a helper command fails

By default, the script stops immediately when a helper command fails.

```bash
xtend_autorun_docal.py 000125000 --on-error stop
```

To mimic the older “log the error and continue” behavior, use

```bash
xtend_autorun_docal.py 000125000 --on-error continue
```

For scientific calibration products, `--on-error stop` is recommended, because a failure in an earlier step can make later diagnostic plots misleading.

### 4.7 Timestamped log files

A timestamped log file is generated automatically.  By default it is written to

```text
<OBSID>/xtend/event_cl/<output-subdir>/logs/
```

Example filename:

```text
xtend_autorun_docal_000125000_20260618_153000.log
```

To write logs elsewhere:

```bash
xtend_autorun_docal.py 000125000 --log-dir ./logs_xtend
```

The log mirrors both standard output and standard error.

---

## 5. Command-line options

| Option | Default | Description |
|---|---:|---|
| `obsid` | required | OBSID to process, e.g. `000125000` |
| `-s`, `--steps` | `1 2 3 4` | Steps to execute |
| `-n`, `--numphoton` | `1000000` | Number of photons passed to `xtend_auto_gen_phaarfrmf.py` |
| `-c`, `--clobber` | `no` | Clobber flag passed to `xtend_auto_gen_phaarfrmf.py`; choose `yes` or `no` |
| `--no-html` | off | Skip HTML report generation |
| `--on-error` | `stop` | Choose `stop` or `continue` when a helper command fails |
| `--output-subdir` | `checkpileup_std` | Working/output subdirectory under `<OBSID>/xtend/event_cl/` |
| `--log-dir` | auto | Directory for timestamped log files |
| `--require-uncompressed` | off | Fail if a required input exists only as `.gz` |
| `--html-keyword` | `checkpileup_` | Keyword passed to `xrism_autorun_png2html.py` |
| `--html-ver` | `v0` | Version string passed to `xrism_autorun_png2html.py` |

Show help:

```bash
xtend_autorun_docal.py -h
```

---

## 6. Gzip input policy

The script explicitly handles `.gz` input files for cleaned event files, EHK files, and BIMG files.

The policy is:

1. If both uncompressed and compressed files exist, the uncompressed file is used.
2. If only the `.gz` file exists, the `.gz` file is used and a warning is printed.
3. If `--require-uncompressed` is specified, the script fails early when a required input is available only as `.gz`.

Examples:

```bash
# Allow .gz inputs when uncompressed files are absent
xtend_autorun_docal.py 000125000

# Require uncompressed inputs
xtend_autorun_docal.py 000125000 --require-uncompressed
```

Important: when a `.evt.gz`, `.ehk.gz`, or `.bimg.gz` file is used, the corresponding helper script receives that compressed filename through a symbolic link in the working directory.  Therefore, all downstream tools used by the helper scripts must be able to read gzip-compressed FITS files.  If a helper script or a non-CFITSIO-based tool expects uncompressed files, unzip the inputs first or use `--require-uncompressed`.

---

## 7. CCD selection logic

The script extracts the data class from the event filename using the pattern

```text
_p.XXXXXXXX
```

where `XXXXXXXX` is the 8-character data class.  The second character is used to infer which CCDs are included.

| Second character | Interpreted CCDs |
|---|---|
| `0` | `CCD1,2,3,4` |
| `1` | `CCD1,2` |
| `2` | `CCD3,4` |
| other | `Unknown` |

At present, the analysis is run only for

- `CCD1,2,3,4`
- `CCD1,2`

Files interpreted as `CCD3,4` or `Unknown` are skipped.

---

## 8. Output products

The default output directory is

```text
<OBSID>/xtend/event_cl/checkpileup_std/
```

Typical products include:

- pileup-check plots
- region files
- PHA files
- RMF files
- ARF files
- optimal-binning PHA files
- `f_gopt.list`
- ARF comparison plots
- quick-look XSPEC fitting logs
- spectrum comparison plots

The automatic log file is written under

```text
<OBSID>/xtend/event_cl/checkpileup_std/logs/
```

unless `--output-subdir` or `--log-dir` is changed.

---

## 9. Recommended workflows

### 9.1 First pass: pileup and region check only

```bash
xtend_autorun_docal.py 000125000 -s 1 2 --no-html
```

Inspect the generated region files and pileup diagnostic images.

### 9.2 Full response generation after confirming regions

```bash
xtend_autorun_docal.py 000125000 -s 3 4 -n 1000000 -c yes
```

### 9.3 Full fresh run into a test directory

```bash
xtend_autorun_docal.py 000125000 \
    --output-subdir checkpileup_test \
    --on-error stop \
    -n 1000000 \
    -c yes
```

### 9.4 Debug run with smaller ARF photon number

```bash
xtend_autorun_docal.py 000125000 \
    --output-subdir checkpileup_debug \
    -s 3 4 \
    -n 10000 \
    -c yes \
    --no-html
```

---

## 10. Troubleshooting

### `Required program not found in $PATH`

A helper script required by the requested steps was not found.

Check your `$PATH`, for example:

```bash
which xtend_pileup_check_quick.sh
which xtend_util_genregion.py
which xtend_auto_gen_phaarfrmf.py
which xrism_util_plot_arf.py
which xrism_spec_qlfit_many.py
which xrism_autorun_png2html.py
```

Then add the relevant `rksysoft` directories to `$PATH`.

### `Xtend cleaned event directory does not exist`

The script must be run from the parent directory of the OBSID directory.

Correct:

```bash
cd /path/to/working_directory
ls 000125000
xtend_autorun_docal.py 000125000
```

Incorrect:

```bash
cd /path/to/working_directory/000125000
xtend_autorun_docal.py 000125000
```

### `No cleaned .evt files found`

Check whether cleaned event files exist under

```text
<OBSID>/xtend/event_cl/
```

with names matching

```text
xa<OBSID>*_cl.evt
xa<OBSID>*_cl.evt.gz
```

### EHK or BIMG file not found

The script expects the EHK file under

```text
<OBSID>/auxil/xa<OBSID>.ehk
```

or

```text
<OBSID>/auxil/xa<OBSID>.ehk.gz
```

The BIMG file is inferred from the cleaned event filename by replacing `_cl.evt` with `.bimg`, and is expected under

```text
<OBSID>/xtend/event_uf/
```

### ARF generation does not update

If old raytrace or response files already exist, use

```bash
xtend_autorun_docal.py 000125000 -s 3 4 -c yes
```

### Later steps run after an earlier failure

Use the default `--on-error stop` mode to fail fast.

```bash
xtend_autorun_docal.py 000125000 --on-error stop
```

Use `--on-error continue` only when intentionally collecting partial outputs.

---

## 11. Notes for developers

Important behavior changes from the older script:

- `--genhtml` has been replaced by `--no-html`.
- The old `--genhtml` / `-html` option remains as a deprecated hidden alias for `--no-html`.
- Required helper programs are checked in a preflight step.
- The required helper set is minimized according to the selected `--steps` and HTML setting.
- Uncompressed inputs are preferred over `.gz` inputs when both exist.
- `.gz` inputs are allowed by default but produce explicit warnings.
- `--require-uncompressed` can be used to reject `.gz`-only inputs.
- Helper-command failure behavior is controlled by `--on-error stop|continue`.
- The output directory can be changed with `--output-subdir`.
- A timestamped log file is generated automatically.
- Symbolic links now point to absolute source paths, so nested output subdirectories are safer than before.

Potential future improvements:

- Add an option to process `CCD3,4` explicitly.
- Add a dry-run mode that only prints planned commands.
- Add a YAML/JSON configuration file mode for repeated calibration campaigns.
- Add a stronger product-existence check before Step 4.
- Add unit tests for filename parsing and gzip selection logic.
