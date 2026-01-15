# README_how_to_test_20260115.md

## Purpose of This README

This document describes the **current manual testing procedure** for the XRISM Resolve
pseudo-event clustering and its impact on **Ls-related branching ratios**.

**Important note**

- This workflow is **not fully automated yet**.
- The purpose of this procedure is **step-by-step verification during the debug phase**.
- Each step is intentionally executed and checked manually to ensure physical and algorithmic correctness.

---

## 1. Create a Working Directory

Create a clean working directory for this test:

```bash
mkdir lscheck_20260115
cd lscheck_20260115
````

---

## 2. Prepare Input Event Files

Bring the following files into the working directory **via symbolic links**:

* `*_uf.evt` : unfiltered Resolve event file
* `*_cl.evt` : cleaned Resolve event file (used for GTI reference)

Example:

```bash
ln -s ../../../../Perseus_CAL1/000154000/resolve/event_uf/xa000154000rsl_p0px1000_uf.evt .
ln -s ../../../../Perseus_CAL1/000154000/resolve/event_cl/xa000154000rsl_p0px1000_cl.evt .
```

Confirm the directory contents:

```bash
ls
```

Expected:

```
xa000154000rsl_p0px1000_cl.evt
xa000154000rsl_p0px1000_uf.evt
```

---

## 3. Run the Pseudo-Event Clustering Pipeline

Run the clustering pipeline using the **uf.evt** file as input:

```bash
resolve_ftools_cluster_pileline.sh xa000154000rsl_p0px1000_uf.evt
```

This pipeline performs:

1. PREV/NEXT interval annotation
2. Large-cluster detection
3. Small-cluster detection
4. Basic diagnostics

---

## 4. Confirm Pipeline Outputs

After completion, the directory should contain files similar to:

```text
diagnostic_plots/
fig_cluster/

xa000154000rsl_p0px1000_uf_noBL_prevnext.evt
xa000154000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt
large_xa000154000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt
small_large_xa000154000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt

xa000154000rsl_p0px1000_cl.evt
xa000154000rsl_p0px1000_cl.gti
```

The key file for downstream assessment is:

```
small_large_xa000154000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt
```

---

## 5. Prepare Directory for Branching-Ratio Assessment

Create a directory for evaluating the impact of cluster cuts on Ls-related branching ratios:

```bash
mkdir assess_bratios
cd assess_bratios
```

Create symbolic links to:

* the clustered event file
* the reference `cl.evt`

```bash
ln -s ../small_large_xa000154000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt .
ln -s ../xa000154000rsl_p0px1000_cl.evt .
```

---

## 6. Apply GTI and Cluster-Based Screening

Run the screening utility to apply:

* GTI from `cl.evt`
* standard cuts
* cluster-based cuts (and their complements)

```bash
resolve_util_screen_for_lscluster.sh \
  small_large_xa000154000rsl_p0px1000_uf_noBL_prevnext_cutclgti.evt \
  xa000154000rsl_p0px1000_cl.evt
```

---

## 7. Confirm Screening Outputs

Expected output files include:

```text
small_large_*_clgti.evt
small_large_*_clgti_stdcut.evt
small_large_*_clgti_clustercut.evt
small_large_*_clgti_NOT_clustercut.evt
small_large_*_clgti_clustercutstdcut.evt

xsel_timefile.asc
xselect.log
```

The most important files for comparison are:

* **with cluster cut**

  ```
  *_clgti_clustercut.evt
  ```

* **without cluster cut**

  ```
  *_clgti.evt
  ```

---

## 8. Prepare Branching-Ratio Analysis

Create a directory for MKLC branching-ratio checks:

```bash
mkdir mklc_branch
cd mklc_branch
```

Link the two event files to be compared:

```bash
ln -s ../small_large_*_clgti.evt .
ln -s ../small_large_*_clgti_clustercut.evt .
```

Create file lists:

```bash
ls small_large_*_clgti.evt > eve.list
ls small_large_*_clgti_clustercut.evt > clustercut.list
```

---

## 9. Run Branching-Ratio Comparison

Prepare the execution script:

```bash
chmod +x run_mklc_branch.sh
```

Example `run_mklc_branch.sh`:

```sh
#!/bin/sh

resolve_ana_pixel_mklc_branch.py eve.list        -t 2048 -odir output_eve        -g -rmax 0.5 -yscaleing log -u --exclude-pixels 12
resolve_ana_pixel_mklc_branch.py clustercut.list -t 2048 -odir output_clustercut -g -rmax 0.5 -yscaleing log -u --exclude-pixels 1
```

Execute:

```bash
./run_mklc_branch.sh
```

---

## 10. What Is Being Verified at This Stage

This manual workflow is used to verify:

* Whether pseudo-event cluster removal changes **Ls-related branching ratios**
* Pixel-by-pixel consistency before and after cluster cuts
* Whether cluster cuts introduce unintended biases
* Physical plausibility of pseudo-Ls removal

---

## Status

* Logic validated step by step
* Full automation **not yet enabled**
* Thresholds and cut strategies **still under tuning**

This README will be updated once the workflow is stabilized and automated.

---

## Contact / Notes

This procedure is intended for **developers and reviewers** familiar with XRISM Resolve event analysis.

Please report:

* unexpected branching-ratio changes
* pixel-specific anomalies