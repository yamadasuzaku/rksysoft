# XRISM Resolve Pseudo-Event Clustering Tool

## Overview

[`resolve_ftools_add_cluster.py`](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_cluster.py) is a Python-based tool for identifying *pseudo events* in X-ray event data from XRISM’s **Resolve** instrument. **pseudo events** are spurious detector triggers not caused by real astrophysical X-ray photons, but by cosmic rays or instrumental effects. 

This is called from [run_cluster_pipeline.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_cluster_pileline.sh), 
which is described in [README_cluster_pileline.md](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/README_cluster_pileline.md). 

https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/README_cluster_pileline.md

Beyond known cross-talk events, the tool can also catch *clustered events* such as cosmic ray hits that trigger multiple pixels at once. By clustering events in time, `resolve_ftools_add_cluster.py` provides a systematic way to flag all events that are part of a nearly coincident group. The output is an augmented event file where each event is annotated with cluster information, making it clear which events are isolated genuine triggers versus which occur in coincident groups (the latter being candidates for pseudo events or particle-induced events). This tool helps users of XRISM Resolve data to detect and mark pseudo events, ensuring that only true X-ray events are used in scientific analysis.

## Dependencies

To run [`resolve_ftools_add_cluster.py`](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_cluster.py), you will need the following Python environment and libraries:

* **Python 3.x** – The script is written in Python and requires a Python 3 interpreter.
* **Astropy** – Used for reading and writing FITS files (e.g. `astropy.io.fits`).
* **NumPy** – Used for efficient array and numerical operations (e.g. handling event time arrays).
* **Matplotlib** – (Optional) Used for generating diagnostic plots of event clusters.
* *(Additional standard libraries like `argparse` for argument parsing are used, but these come with Python by default.)*

Ensure these packages are installed. If you are using an Anaconda environment or pip, you can install any missing packages with the usual commands (e.g. `pip install astropy numpy matplotlib`). No proprietary or XRISM-specific library is required; the tool works with standard FITS files and Python scientific libraries.

## Usage

You can run the tool from the command line to process a Resolve event FITS file. 
The basic usage is to use pileline scripts: 

```bash
./run_cluster_pipeline.sh <input_file_uf.evt>
```

Please read befere you try.   
* [README_cluster_pileline.md](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/README_cluster_pileline.md)

In [run_cluster_pipeline.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_cluster_pileline.sh), 
[`resolve_ftools_add_cluster.py`](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/resolve_ftools_add_cluster.py) is used as follows. 

```bash:run_cluster_pipeline.sh
# --- Step 1: Add prev/next interval columns ---
echo ">>> Running resolve_ftools_add_prevnext.sh on $input_file"
resolve_ftools_add_prevnext.sh "$input_file"

# --- Step 2: Run clustering for large and small pseudo events ---
base_name="${input_file%.evt}_noBL_prevnext_cutclgti.evt"
check_file_exists "$base_name"

echo ">>> Running large cluster detection on $base_name"
resolve_ftools_add_cluster.py "$base_name" \
    --mode large \
    --col_cluster ICLUSTERL \
    --col_member IMEMBERL \
    --outname large_ \
    -d

echo ">>> Running small cluster detection on large_$base_name"
resolve_ftools_add_cluster.py "large_$base_name" \
    --mode small \
    --col_cluster ICLUSTERS \
    --col_member IMEMBERS \
    --outname small_ \
    -d

# --- Step 3: Run QL diagnostic tool ---
final_file="small_large_${input_file%.evt}_noBL_prevnext_cutclgti.evt"
check_file_exists "$final_file"
```

**Example:** Suppose you have an event file named `clean.evt`. To run the pseudo-event detection in "small cluster" mode and produce a diagnostic plot, you might use:

```bash
resolve_ftools_add_cluster.py clean.evt --mode small  --col_cluster ICLUSTERS --col_member IMEMBERS --outname small_ -d
```

## Arguments

The script accepts several command-line arguments and options. Here is a breakdown of each:

* **`input_events.fits` (positional)** – **Input event file** in FITS format. This should be the output of non-standard XRISM pipeline processing, containing a FITS table of events with PREV_INTERVAL and NEXT_INTERVAL by running [resolve_ftools_add_prevnext.sh](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/resolve_ftools_add_prevnext.sh) by reading [READNE_add_prevnext.md](https://github.com/yamadasuzaku/rksysoft/blob/main/resolve/ftools/rslcluster/README_add_prevnext.md). 

* **`-m, --mode {small|large}`** – **Clustering mode selection.** This option determines the algorithm or criteria used to cluster events:

  * **`small`** – Small-cluster mode (default). Optimized for finding **cross-talk pairs** and small event groups. In this mode, the algorithm looks for *near-coincident events typically involving one primary small pseudo event and one or more low-secondary events*. 

  * **`large`** – Large-cluster mode. This mode is used to detect **larger event clusters** where multiple pixels register significant events at nearly the same time. Such clusters may indicate a cosmic ray or particle hit affecting several pixels simultaneously, rather than mere cross-talk. In large mode, the time window for grouping may be slightly more permissive (to catch bursts of events), and if a cluster contains more than one high-energy event, the entire cluster can be flagged for further inspection. Essentially, *all events in a large cluster are treated as suspect*, since genuine astrophysical X-ray photons are highly unlikely to trigger multiple independent pixels at the same instant.

* **`-p, --usepixels`** – **Select pixels to be analyzed.** 

*(If additional options exist in the script, such as a custom time threshold or a verbose mode, document them similarly. In the current version, the above are the primary options.)*

## Clustering Modes: "large" vs "small"

The tool operates in two clustering modes, **small** and **large**, which differ in the scale of event grouping and their interpretation:

* **Small Cluster Mode:** This mode is geared towards identifying **small pseudo events**. 

* **Large Cluster Mode:** In large mode, the algorithm can group any events that occur within a short interval, much shorter than typical astrophysical event coincidence probability. If two or more events are found nearly simultaneous, they are clustered as one “large” event group. The output will flag **all** events in such a cluster as potential pseudo events (since they presumably originate from a single particle/cosmic event rather than separate X-rays). 

For normal or moderately bright X-ray sources, use both **small** and **large** mode.
For very high background conditions or calibration data, or if you want to be very strict, use **large** to catch everything coincident

## Output

After running the script, you will get:

* **Augmented FITS Event File:** The output FITS file (specified by you as `<output_clusters.fits>`) will contain the original event data plus additional columns that describe the clustering results. Key columns added include:

  * **`ICLUSTER`** – an integer label identifying the cluster to which the event belongs in each pixel. The ID is typically sequential (e.g., 0, 0, 2, 3, 0, 5, 6, 7, 0 ...) in the order clusters are found in time. If an event is not classified as a cluster, the ID is 0. That is, ICLUSTER>0 means clustered events, while ICLUSTER==0 means non-clustered events. 
  * **`IMEMBER`** – the number of events in that cluster. If an event is not classified as a cluster, the ID is 0. That is, IMEMBER>0 means clustered events, and IMEMBER==0 means non-clustered events. IMBMER==1 means the 1st event in each cluster. 

All original columns (such as TIME, PI/energy, DETector ID, etc.) from the input file are preserved in the output. The new columns appear in the EVENTS extension of the FITS file. 

By examining the output FITS file and plot, you can proceed to filter your data. For example, using FITS tools or Python, you could exclude all events with `ICLUSTER==0` from your spectrum extraction, thereby removing the possible pseudo events. 

## Notes

* **Usage in Analysis:** This tool does *not* automatically remove or discard any events; it only annotates them. It is up to the user to decide how to use this information. If you are working with very bright sources, applying this clustering (especially small mode) requires careful check. 

* **Time Window and Parameters:** The clustering algorithm uses a fixed time coincidence window to decide if events belong to the same cluster. This window is chosen based on the detector’s timing characteristics and known behavior. In most cases you will not need to adjust it. However, be aware that this is still preliminary version. 

* **Limitations:** There are a few important limitations and caveats to note:

  * *Missed Pseudo Events:* If the Resolve instrument was saturated or the onboard CPU dropped an event during a high count rate burst, a cluster's child could appear without its parent in the data (because the parent was lost). In such a case, the tool **may not work** as expected. 
  * *False Positives:* For typical astrophysical observations, the chance of two independent X-rays hitting at the nearly same time is negligible. However, if you are observing an extremely bright X-ray source, there is a slight possibility that two real events could occur very close in time and be mistakenly clustered. This is usually not a concern unless your count rate is exceptionally high. If needed, you could loosen the time window or inspect such clusters manually.
