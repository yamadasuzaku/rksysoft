#!/usr/bin/env python3
"""
Generate a mock ROOT file with a TTree for spectrum analysis practice.

This script creates:
- energy
- good
- channum
- several additional quality-related branches

The generated file is intended for uproot-based analysis tutorials.
"""

import argparse
import numpy as np
import uproot

MIN_ENE_BGD=5500
MAX_ENE_BGD=7000

def str2bool(value):
    """
    Robust boolean parser for argparse.
    """
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("true", "t", "yes", "y", "1"):
        return True
    if value in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def generate_channel_event_counts(
    n_channels,
    mean_events,
    n_bad_channels,
    rng
):
    """
    Generate per-channel event counts.

    Some channels are intentionally assigned zero events
    to mimic bad pixels or disconnected channels.
    """
    counts = rng.poisson(lam=mean_events, size=n_channels)

    # Force some channels to be bad (zero-event channels)
    bad_channels = rng.choice(n_channels, size=n_bad_channels, replace=False)
    counts[bad_channels] = 0

    return counts, np.sort(bad_channels)


def generate_mock_events_for_one_channel(
    ch,
    n_events,
    main_peak_energy,
    sub_peak_energy,
    main_sigma,
    sub_sigma,
    background_fraction,
    channel_gain_sigma,
    rng
):
    """
    Generate mock events for a single channel.

    The energy distribution consists of:
    - a strong main Gaussian line
    - a weaker sub Gaussian line
    - a uniform background

    Each channel has a small gain shift to mimic detector variation.
    """
    if n_events == 0:
        return None

    # Small channel-to-channel shift
    gain_shift = rng.normal(loc=0.0, scale=channel_gain_sigma)

    # Event composition
    n_main = int(n_events * (1.0 - background_fraction) * 0.78)
    n_sub = int(n_events * (1.0 - background_fraction) * 0.17)
    n_bkg = n_events - n_main - n_sub

    # Generate energies
    energy_main = rng.normal(
        loc=main_peak_energy + gain_shift,
        scale=main_sigma,
        size=n_main
    )
    energy_sub = rng.normal(
        loc=sub_peak_energy + gain_shift,
        scale=sub_sigma,
        size=n_sub
    )
    energy_bkg = rng.uniform(low=MIN_ENE_BGD, high=MAX_ENE_BGD, size=n_bkg)

    energy = np.concatenate([energy_main, energy_sub, energy_bkg])

    # Shuffle so the events are not ordered by component
    rng.shuffle(energy)

    # Good flag: mostly True, but some False
    good = rng.random(n_events) > 0.08

    # Additional branches with rough correlations
    pretrig_rms = rng.normal(loc=8.0, scale=2.0, size=n_events)
    pretrig_rms[~good] += rng.normal(loc=10.0, scale=3.0, size=np.sum(~good))

    rise_time = rng.normal(loc=25.0, scale=6.0, size=n_events)
    promptness = rng.normal(loc=0.5, scale=0.08, size=n_events)
    pulse_rms = rng.normal(loc=12.0, scale=2.5, size=n_events)
    filt_phase = rng.normal(loc=0.0, scale=0.03, size=n_events)

    # Channel number
    channum = np.full(n_events, ch, dtype=np.int32)

    return {
        "energy": energy.astype(np.float32),
        "good": good.astype(np.bool_),
        "channum": channum,
        "pretrig_rms": pretrig_rms.astype(np.float32),
        "rise_time": rise_time.astype(np.float32),
        "promptness": promptness.astype(np.float32),
        "pulse_rms": pulse_rms.astype(np.float32),
        "filt_phase": filt_phase.astype(np.float32),
    }


def concatenate_branch_dicts(dict_list):
    """
    Concatenate a list of event dictionaries into one branch dictionary.
    """
    if not dict_list:
        raise ValueError("No event dictionaries to concatenate.")

    keys = dict_list[0].keys()
    out = {}
    for key in keys:
        out[key] = np.concatenate([d[key] for d in dict_list], axis=0)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate a mock ROOT file for spectrum analysis practice."
    )
    parser.add_argument(
        "--output",
        default="mock_spectrum.root",
        help="Output ROOT file path"
    )
    parser.add_argument(
        "--tree-name",
        default="tree",
        help="TTree name"
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=101,
        help="Number of total channels"
    )
    parser.add_argument(
        "--mean-events",
        type=int,
        default=8000,
        help="Average number of events per active channel"
    )
    parser.add_argument(
        "--n-bad-channels",
        type=int,
        default=12,
        help="Number of zero-event channels"
    )
    parser.add_argument(
        "--main-peak-energy",
        type=float,
        default=5898.75,
        help="Main line energy in eV"
    )
    parser.add_argument(
        "--sub-peak-energy",
        type=float,
        default=6490.0,
        help="Secondary line energy in eV"
    )
    parser.add_argument(
        "--main-sigma",
        type=float,
        default=3.0,
        help="Sigma of the main Gaussian peak in eV"
    )
    parser.add_argument(
        "--sub-sigma",
        type=float,
        default=4.5,
        help="Sigma of the secondary Gaussian peak in eV"
    )
    parser.add_argument(
        "--background-fraction",
        type=float,
        default=0.10,
        help="Fraction of uniform background events"
    )
    parser.add_argument(
        "--channel-gain-sigma",
        type=float,
        default=1.2,
        help="Channel-by-channel energy shift sigma in eV"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed"
    )

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    counts, bad_channels = generate_channel_event_counts(
        n_channels=args.n_channels,
        mean_events=args.mean_events,
        n_bad_channels=args.n_bad_channels,
        rng=rng
    )

    event_dicts = []
    active_channels = []

    for ch in range(args.n_channels):
        n_events = int(counts[ch])
        result = generate_mock_events_for_one_channel(
            ch=ch,
            n_events=n_events,
            main_peak_energy=args.main_peak_energy,
            sub_peak_energy=args.sub_peak_energy,
            main_sigma=args.main_sigma,
            sub_sigma=args.sub_sigma,
            background_fraction=args.background_fraction,
            channel_gain_sigma=args.channel_gain_sigma,
            rng=rng
        )
        if result is None:
            continue
        event_dicts.append(result)
        active_channels.append(ch)

    all_data = concatenate_branch_dicts(event_dicts)

    with uproot.recreate(args.output) as fout:
        fout[args.tree_name] = all_data

    print("Created ROOT file:", args.output)
    print("Tree name         :", args.tree_name)
    print("Total events      :", len(all_data["energy"]))
    print("Active channels   :", len(active_channels))
    print("Bad channels      :", len(bad_channels))
    print("Bad channel list  :", bad_channels.tolist())


if __name__ == "__main__":
    main()
