# README_mklc_branch.md

## Overview

This document describes the **branching\UTF{2013}ratio\UTF{2013}based inference of Lp/Ls events** implemented in

```

resolve_ana_pixel_mklc_branch.py

````

and used via:

```sh
resolve_ana_pixel_mklc_branch.py eve.list        -t 2048 -odir output_eve        -g -rmax 0.5 -yscaleing log -u --exclude-pixels 12
resolve_ana_pixel_mklc_branch.py clustercut.list -t 2048 -odir output_clustercut -g -rmax 0.5 -yscaleing log -u --exclude-pixels 1
````

The purpose of this analysis is to:

* **Estimate the true underlying event rate `lambda` (per pixel)** using *high- and mid-grade events*
  (**Hp, Mp, Ms**), which are least affected by pseudo-event contamination.
* **Predict the expected number of low-grade events (Lp, Ls)** from the inferred rate.
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
  T = sum of GTI overlap durations
  ```

* Otherwise:

  ```
  T = time_last minus time_first
  ```

### Observed counts per grade

For each grade `g`:

```
K_g = number of events with ITYPE equal to g
```

In particular, we define:

```
K_fit    = {K_Hp, K_Mp, K_Ms}
K_target = {K_Lp, K_Ls}
```

---

## Step 2: Physical Model (Branching Ratios)

We assume a **single underlying Poisson rate `lambda`** (counts per second per pixel).

For a given `lambda`, the expected fraction of each grade is:

```
p_g(lambda) = branching ratio for grade g
```

These are computed by:

```python
calc_branchingratios(rate)
```

which encodes the Resolve trigger dead-time structure and is normalized such that:

```
sum over g of p_g(lambda) = 1
```

The expected mean count for grade `g` is:

```
mu_g(lambda) = T * lambda * p_g(lambda)
```

---

## Step 3: Likelihood Function

Only **Hp/Mp/Ms** are used to estimate `lambda`.

For a given pixel:

```
log L(lambda)
  = sum over g in {Hp, Mp, Ms}
      [ K_g * log(mu_g(lambda)) - mu_g(lambda) ]
```

This is a **Poisson likelihood**, with constant terms omitted.

---

## Step 4: Maximum Likelihood Estimate of lambda

The best-fit rate, denoted as **`lambda_hat`**, is obtained by minimizing:

```
negative log L(lambda)
```

using bounded scalar minimization:

```python
minimize_scalar(method="bounded")
```

Bounds:

```
lambda in (0, rate_max)
```

---

## Step 5: Confidence Interval via Profile Likelihood

The confidence interval for `lambda` is computed using **Wilks’ theorem**:

```
2 * Delta(log L)
  = 2 * [ log L(lambda_hat) - log L(lambda) ]
  <= 1
```

This corresponds to a **68 percent confidence interval for one parameter**.

Implementation details:

* `lambda` is scanned on a dense logarithmic grid
* Interval boundaries are interpolated where `2 * Delta(log L) = 1`

Result:

```
lambda_hat, [ lambda_lo , lambda_hi ]
```

---

## Step 6: Prediction of Lp / Ls

Using the inferred `lambda_hat`:

```
M_g = T * lambda_hat * p_g(lambda_hat)
```

for `g` in `{Lp, Ls}`.

Confidence bounds are propagated conservatively:

```
M_g_lo = T * lambda_hi * p_g(lambda_hi)
M_g_hi = T * lambda_lo * p_g(lambda_lo)
```

---

## Step 7: Bias Estimation

For each target grade:

```
bias_g (percent)
  = (K_g - M_g) / M_g * 100
```

with confidence bounds:

```
bias_lo, bias_hi
```

This directly answers:

> How much do Lp and Ls deviate from expectation based on Hp, Mp, and Ms?

---

## Output Products

### 1. CSV Summary

Saved as:

```
*_lp_ls_infer_profilelik.csv
```

Contains, per pixel:

* `lambda_hat` and its confidence interval
* Observed Lp/Ls counts
* Predicted Lp/Ls counts
* Bias (percent) with confidence bounds

### 2. Diagnostic Plots

Generated per event file:

* `lambda_hat` versus pixel
* Exposure time versus pixel
* Observed versus predicted counts for **all grades**
* Bias versus pixel for Lp and Ls

These plots are intended for validation, not publication.


## Status

* This tool is currently in **debug and validation phase**
* Thresholds and branching model are subject to revision
* Results should be interpreted comparatively (before versus after cuts), not absolutely

---
