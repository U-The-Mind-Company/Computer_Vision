#!/usr/bin/env python3
"""
Handtremor_Graphs.py.py

Usage:
    python Handtremor_Graphs.py <root_folder>

This script:
- Finds all per-frame CSVs inside <root_folder>.
- Groups by subject ID (from filenames like sub001_day1_before.csv).
- Computes metrics (Mean Tremor Amplitude, RMS, PSD 4–6 Hz).
- Interpolates up to 30 days.
- Plots Before vs After (Left vs Right) with log10 polynomial fits.
- Creates a folder per subject inside <root_folder>/output/<subject_id>/.

Expected columns: frame, hand_id, x_0..x_20, y_0..y_20
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize': (8,5), 'font.size': 11})

# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------
def compute_psd(signal_data, fs=30):
    if len(signal_data) < 10:
        return np.nan
    f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)))
    band_mask = (f >= 4) & (f <= 6)
    return np.trapz(Pxx[band_mask], f[band_mask])

def compute_rms(data):
    return np.sqrt(np.mean(np.square(data))) if len(data) else np.nan

def extract_info(filename):
    """Extract subject id, day, and phase from filename."""
    subject = re.search(r"(sub\d+)", filename, re.IGNORECASE)
    day_match = re.search(r"day(\d+)", filename, re.IGNORECASE)
    phase = "before" if "before" in filename.lower() else "after"
    subj_id = subject.group(1).lower() if subject else "unknown"
    day = int(day_match.group(1)) if day_match else np.nan
    return subj_id, day, phase

def process_subject(subject_id, files, root_folder, output_root):
    all_records = []
    for filepath in files:
        df = pd.read_csv(filepath)
        if "hand_id" not in df.columns:
            print(f"⚠️ Skipping {filepath} — missing 'hand_id'.")
            continue

        _, day, phase = extract_info(os.path.basename(filepath))

        for hand_id, group in df.groupby("hand_id"):
            side = "Left" if hand_id == 0 else "Right"
            coord_cols = [c for c in df.columns if c.startswith("x_") or c.startswith("y_")]
            hand_signal = group[coord_cols].to_numpy().flatten()

            mean_amp = np.nanmean(np.abs(np.diff(hand_signal)))
            rms_amp = compute_rms(hand_signal)
            psd_band = compute_psd(hand_signal)

            all_records.append([
                subject_id, day, phase, side,
                mean_amp, rms_amp, psd_band
            ])

    cols = ["Subject", "Day", "Phase", "Side",
            "Mean_Tremor_Amplitude", "RMS_Amplitude", "PSD_4_6Hz"]
    df_metrics = pd.DataFrame(all_records, columns=cols)

    if df_metrics.empty:
        print(f"❌ No valid data for {subject_id}")
        return

    # Interpolation up to Day 30
    days_full = np.arange(1, 31)
    final_df = pd.DataFrame()

    for (side, phase), group in df_metrics.groupby(["Side", "Phase"]):
        group = group.sort_values("Day").dropna(subset=["Day"])
        if len(group) < 2:
            continue

        f_amp = interp1d(group["Day"], group["Mean_Tremor_Amplitude"],
                         kind='linear', bounds_error=False, fill_value='extrapolate')
        f_rms = interp1d(group["Day"], group["RMS_Amplitude"],
                         kind='linear', bounds_error=False, fill_value='extrapolate')
        f_psd = interp1d(group["Day"], group["PSD_4_6Hz"],
                         kind='linear', bounds_error=False, fill_value='extrapolate')

        df_interp = pd.DataFrame({
            "Subject": subject_id,
            "Day": days_full,
            "Phase": phase,
            "Side": side,
            "Mean_Tremor_Amplitude": f_amp(days_full),
            "RMS_Amplitude": f_rms(days_full),
            "PSD_4_6Hz": f_psd(days_full)
        })
        final_df = pd.concat([final_df, df_interp], ignore_index=True)

    # Compute percentage change (After vs Before)
    before = final_df[final_df["Phase"] == "before"].set_index(["Day", "Side"])
    after = final_df[final_df["Phase"] == "after"].set_index(["Day", "Side"])
    pct_change = ((after - before) / before) * 100
    pct_change = pct_change.reset_index().fillna(0)

    for col in ["Mean_Tremor_Amplitude", "RMS_Amplitude", "PSD_4_6Hz"]:
        pct_change[col] = pct_change[col].apply(lambda x: "<0.1" if abs(x) < 0.1 else round(x, 2))

    # Save outputs
    subj_out_dir = os.path.join(output_root, subject_id)
    os.makedirs(subj_out_dir, exist_ok=True)

    summary_path = os.path.join(subj_out_dir, f"{subject_id}_summary.csv")
    final_df.to_csv(summary_path, index=False)
    print(f"✅ Saved {subject_id} metrics to {summary_path}")

    # ----------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------
    def plot_metric(metric, ylabel):
        plt.figure()
        for side in ["Left", "Right"]:
            for phase, style in zip(["before", "after"], ["--", "-"]):
                sub_df = final_df[(final_df["Side"] == side) & (final_df["Phase"] == phase)]
                if sub_df.empty:
                    continue
                plt.plot(sub_df["Day"], sub_df[metric],
                         label=f"{side}-{phase.capitalize()}", linestyle=style, linewidth=2)
                try:
                    log_y = np.log10(sub_df[metric])
                    coeffs = Polynomial.fit(sub_df["Day"], log_y, 3)
                    plt.plot(sub_df["Day"], 10 ** coeffs(sub_df["Day"]),
                             alpha=0.4, label=f"{side}-{phase} log10 polyfit")
                except Exception:
                    pass
        plt.xlabel("Day")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Day ({subject_id})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(subj_out_dir, f"{subject_id}_{metric}.png"))
        plt.close()

    plot_metric("Mean_Tremor_Amplitude", "Mean Tremor Amplitude")
    plot_metric("RMS_Amplitude", "RMS Amplitude")
    plot_metric("PSD_4_6Hz", "PSD (4–6 Hz)")

    print(f"📊 Saved plots for {subject_id} in {subj_out_dir}")

# ----------------------------------------------------------
# Entry Point
# ----------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_analyze_tremor_metrics.py <root_folder>")
        sys.exit(1)

    root_folder = sys.argv[1]
    output_root = os.path.join(root_folder, "output")
    os.makedirs(output_root, exist_ok=True)

    # Collect all CSVs
    all_csvs = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.endswith(".csv"):
                all_csvs.append(os.path.join(dirpath, f))

    if not all_csvs:
        print("❌ No CSV files found.")
        sys.exit(1)

    # Group CSVs by subject id
    subjects = {}
    for fpath in all_csvs:
        subj_id, _, _ = extract_info(os.path.basename(fpath))
        subjects.setdefault(subj_id, []).append(fpath)

    # Process each subject
    for subj, files in subjects.items():
        print(f"\n🔹 Processing {subj} ({len(files)} CSV files)...")
        process_subject(subj, files, root_folder, output_root)

    print("\n✅ All subjects processed. Results saved in:", output_root)
