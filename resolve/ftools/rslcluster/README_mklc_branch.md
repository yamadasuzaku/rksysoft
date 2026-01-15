# README_mklc_branch.md

## Overview

This document describes the **branching\UTF{2013}ratio\UTF{2013}based inference of Lp/Ls events** implemented in

```
resolve_ana_pixel_mklc_branch.py
```

and used via:

```sh
resolve_ana_pixel_mklc_branch.py eve.list        -t 2048 -odir output_eve        -g -rmax 0.5 -yscaleing log -u --exclude-pixels 12
resolve_ana_pixel_mklc_branch.py clustercut.list -t 2048 -odir output_clustercut -g -rmax 0.5 -yscaleing log -u --exclude-pixels 1
```

The purpose of this analysis is to:

* **Estimate the true underlying event rate λ (per pixel)** using *high- and mid-grade events*
  (**Hp, Mp, Ms**), which are least affected by pseudo-event contamination.
* **Predict the expected number of low-grade events (Lp, Ls)** from the inferred λ.
* **Quantify bias and confidence intervals** for Lp/Ls using **profile likelihood**, not ad-hoc error propagation.

This step is used to assess whether **pseudo Ls suppression or clustering cuts** introduce systematic bias in scientific products.

---

## Event Grades and Notation

We follow the Resolve internal grade definition:

| itype | Name | Description      |
| ----: | ---- | ---------------- |
|     0 | Hp   | High primary     |
|     1 | Mp   | Medium primary   |
|     2 | Ms   | Medium secondary |
|     3 | Lp   | Low primary      |
|     4 | Ls   | Low secondary    |

Throughout this document:

* **Hp/Mp/Ms** are referred to as **fit grades**
* **Lp/Ls** are referred to as **target grades**

---

## Processing Flow (Relevant Section Only)

This README focuses on the following code path:

```python
if args.plot_rate_vs_grade:
    plot_rate_vs_grade(...)

    results, outcsv, outpng = estimate_lp_ls_from_hpmpms_profilelik(
        event_list=event_list,
        plotpixels=plotpixels,
        itypenames=itypenames,
        timebinsize=args.timebinsize,
        output=args.output,
        ref_time=ref_time,
        gtiuse=args.gtiuse,
        debug=args.debug,
        rate_max=args.rate_max_ingratio,
        fit_grades=(0, 1, 2),     # Hp, Mp, Ms
        target_grades=(3, 4),     # Lp, Ls
        show=False,
        output_dir=args.output_dir
    )
```

---

## Step 1: Observables

For each **pixel** and **event file**, the following are measured directly from event-level data:

### Total exposure time

* If `--gtiuse` is enabled:

  ```
  T = Σ (GTI overlap durations)
  ```
* Otherwise:

  ```
  T = time_last − time_first
  ```

### Observed counts per grade

For each grade `g`:

```
K_g = number of events with ITYPE == g
```

In particular, we define:

```
K_fit = {K_Hp, K_Mp, K_Ms}
K_target = {K_Lp, K_Ls}
```

---

## Step 2: Physical Model (Branching Ratios)

We assume a **single underlying Poisson rate λ** (counts/s/pixel) per pixel.

For a given λ, the expected fraction of each grade is:

```
p_g(λ) = branching ratio for grade g
```

These are computed by:

```python
calc_branchingratios(rate)
```

which encodes the Resolve trigger dead-time structure and is normalized such that:

```
Σ_g p_g(λ) = 1
```

The expected mean count for grade `g` is:

```
μ_g(λ) = T × λ × p_g(λ)
```

---

## Step 3: Likelihood Function

Only **Hp/Mp/Ms** are used to estimate λ.

For a given pixel:

```
log L(λ)
  = Σ_{g ∈ {Hp,Mp,Ms}}
      [ K_g log( μ_g(λ) ) − μ_g(λ) ]
```

This is a **Poisson likelihood**, with factorial terms omitted (constant).

---

## Step 4: Maximum Likelihood Estimate of λ

The best-fit rate `\CID{1124}` is obtained by minimizing:

```
−log L(λ)
```

using bounded scalar minimization:

```python
minimize_scalar(method="bounded")
```

Bounds:

```
λ ∈ (0, rate_max)
```

---

## Step 5: Confidence Interval via Profile Likelihood

The confidence interval for λ is computed using **Wilks’ theorem**:

```
2ΔlogL = 2 [ logL(\CID{1124}) − logL(λ) ] \UTF{2264} 1
```

This corresponds to a **68% confidence interval for one parameter**.

Implementation details:

* λ is scanned on a dense logarithmic grid
* Interval boundaries are interpolated where `2ΔlogL = 1`

Result:

```
\CID{1124} ,  [ λ_lo , λ_hi ]
```

---

## Step 6: Prediction of Lp / Ls

Using the inferred λ:

```
M_g = T × \CID{1124} × p_g(\CID{1124})
```

for `g ∈ {Lp, Ls}`.

Confidence bounds are propagated conservatively:

```
M_g_lo = T × λ_hi × p_g(λ_hi)
M_g_hi = T × λ_lo × p_g(λ_lo)
```

---

## Step 7: Bias Estimation

For each target grade:

```
bias_g (%) = (K_g − M_g) / M_g × 100
```

with confidence bounds:

```
bias_lo, bias_hi
```

This directly answers:

> *How much Lp/Ls deviates from expectation based on Hp/Mp/Ms?*

---

## Output Products

### 1. CSV Summary

Saved as:

```
*_lp_ls_infer_profilelik.csv
```

Contains, per pixel:

* \CID{1124} and its confidence interval
* Observed Lp/Ls counts
* Predicted Lp/Ls counts
* Bias (%) with confidence bounds

### 2. Diagnostic Plots

Generated per event file:

* \CID{1124} vs pixel
* Exposure time vs pixel
* Observed vs predicted counts for **all grades**
* Bias vs pixel for Lp and Ls

These plots are intended for **flight review and pipeline validation**, not publication.

---

## Why This Matters

Low-grade events (especially **Ls**) are the most numerous and the most vulnerable to:

* cosmic-ray\UTF{2013}induced pseudo events
* waveform subtraction residuals
* clustering-based rejection

If Lp/Ls are used naively to estimate exposure or rate:

> **Ignoring pseudo-event contamination leads to systematically biased effective exposure and flux estimates.**

By anchoring λ to **Hp/Mp/Ms**, this method provides a **physics-based reference** against which Lp/Ls behavior can be validated.

---

## Status

* This tool is currently in **debug / validation phase**
* Thresholds and branching model are subject to revision
* Results should be interpreted comparatively (before/after cuts), not absolutely

---
