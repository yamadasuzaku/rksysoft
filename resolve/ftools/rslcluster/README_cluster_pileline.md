# README: Clustering Pipeline Script (Updated 2026.1.15) 

This script automates a three-step clustering pipeline to identify and diagnose pseudo-event clusters in XRISM Resolve event files.

## Script Name

```bash
resolve_ftools_cluster_pipeline.sh
```

## Usage

```bash
./resolve_ftools_cluster_pipeline.sh <input_file_uf.evt>
```

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_cluster_pileline.sh

### Arguments

* `<input_file_uf.evt>`
  Resolve unfiltered event file with `_uf.evt` suffix.

\CID{220} **Important**
In addition to `<input_file_uf.evt>`, the corresponding

```text
<input_file>_cl.evt
```

must exist in the same directory.
This file is required internally by `resolve_ftools_add_prevnext.sh` to create `cl.gti`.

---

## Overview of Steps

### Step 1: Add Prev/Next Interval Information

```bash
resolve_ftools_add_prevnext.sh <input_file_uf.evt>
```

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_prevnext.sh

* Adds previous/next interval\UTF{2013}related columns to the event file.
* Internally generates a GTI (`cl.gti`) based on `<input_file>_cl.evt`.
* Produces an output file with suffix:

```text
_noBL_prevnext_cutclgti.evt
```

---

### Step 2: Detect Pseudo-Event Clusters (Large → Small)

This step is executed **sequentially in two modes**, using
`resolve_ftools_detect_pseudo_event_clusters.py`.

#### 2-1. Large Cluster Detection

```bash
resolve_ftools_detect_pseudo_event_clusters.py <base_file> \
    --mode large \
    --col_cluster ICLUSTERL \
    --col_member IMEMBERL \
    --outname large_ \
    -d
```

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_detect_pseudo_event_clusters.py

* Detects large-amplitude clusters (e.g., cosmic-ray-like events).
* Adds:

  * `ICLUSTERL`: cluster ID
  * `IMEMBERL`: cluster membership flag

#### 2-2. Small Cluster Detection

```bash
resolve_ftools_detect_pseudo_event_clusters.py large_<base_file> \
    --mode small \
    --col_cluster ICLUSTERS \
    --col_member IMEMBERS \
    --outname small_ \
    -d
```

* Detects smaller / slower clusters after removing large-cluster events.
* Adds:

  * `ICLUSTERS`
  * `IMEMBERS`

\CID{1840} For algorithmic details, see:

* **README_add_cluster.md**
  [XRISM Resolve Pseudo-Event Clustering Tool](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/README_add_cluster.md)

---

### Step 3: Cluster Diagnostic Check (QL)

```bash
resolve_ftools_qlcheck_cluster.py <final_evt_file>
```

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_qlcheck_cluster.py

* Performs a quick-look diagnostic check of clustering results.
* Generates summary plots and logs for validation.

---

## Output Files

Intermediate and final files are generated automatically by adding prefixes/suffixes to the input filename.

| Stage          | Output File Example                                  |
| -------------- | ---------------------------------------------------- |
| Input          | `xa000000000_uf.evt`                                 |
| Step 1         | `xa000000000_noBL_prevnext_cutclgti.evt`             |
| Step 2 (large) | `large_xa000000000_noBL_prevnext_cutclgti.evt`       |
| Step 2 (small) | `small_large_xa000000000_noBL_prevnext_cutclgti.evt` |
| Step 3         | QL diagnostic plots & logs                           |

---

## File Existence Checks

The script terminates immediately if any required file is missing:

* `<input_file_uf.evt>`
* `<input_file>_cl.evt`
* Intermediate `.evt` files generated at each step

This behavior prevents silent failures in downstream processing.

---

## Requirements

The following tools must be available in your `PATH`:

* `resolve_ftools_add_prevnext.sh`
* `resolve_ftools_detect_pseudo_event_clusters.py`
* `resolve_ftools_qlcheck_cluster.py`

These tools are assumed to be part of the `rslcluster` / Resolve analysis utilities.

---

## Notes

* Intermediate files are **overwritten without confirmation**.
* It is strongly recommended to run this script in a clean working directory.
* Debug mode (`-d`) is enabled by default in clustering steps.
* The pipeline is designed to be **fully sequential and fail-fast**.
