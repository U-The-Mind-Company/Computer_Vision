import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import defaultdict
import re

def process_file(file_path, output_dir, sigma=2, frame_rate=30, window_size=15):
    df = pd.read_csv(file_path)

    # Apply Gaussian smoothing
    left_heel_x = gaussian_filter1d(df['left_heel_x'].values, sigma)
    left_heel_y = gaussian_filter1d(df['left_heel_y'].values, sigma)
    right_heel_x = gaussian_filter1d(df['right_heel_x'].values, sigma)
    right_heel_y = gaussian_filter1d(df['right_heel_y'].values, sigma)

    # Calculate dynamic thresholds
    def calculate_dynamic_threshold(heel_x, heel_y, window_size):
        movement = np.sqrt(np.diff(heel_x, prepend=heel_x[0])**2 + np.diff(heel_y, prepend=heel_y[0])**2)
        rolling_threshold = pd.Series(movement).rolling(window=window_size, min_periods=1).std() * 1.5
        return rolling_threshold.values

    left_threshold = calculate_dynamic_threshold(left_heel_x, left_heel_y, window_size)
    right_threshold = calculate_dynamic_threshold(right_heel_x, right_heel_y, window_size)

    # Detect heel strikes
    def detect_heel_strikes_with_dynamic_threshold(heel_x, heel_y, thresholds):
        heel_strikes = []
        swing_start = None
        for frame in range(1, len(heel_x)):
            x_diff = abs(heel_x[frame] - heel_x[frame - 1])
            y_diff = abs(heel_y[frame] - heel_y[frame - 1])

            if x_diff <= thresholds[frame] and y_diff <= thresholds[frame]:
                if swing_start is not None:  
                    heel_strikes.append(frame)  
                    swing_start = None
            else: 
                if swing_start is None:
                    swing_start = frame  
        return heel_strikes

    left_heel_strikes = detect_heel_strikes_with_dynamic_threshold(left_heel_x, left_heel_y, left_threshold)
    right_heel_strikes = detect_heel_strikes_with_dynamic_threshold(right_heel_x, right_heel_y, right_threshold)

    # Calculate step times
    def calculate_step_times(left_strikes, right_strikes):
        step_times = []
        i, j = 0, 0
        while i < len(left_strikes) and j < len(right_strikes):
            left_strike = left_strikes[i]
            right_strike = right_strikes[j]
            if left_strike < right_strike:
                step_duration = (right_strike - left_strike) / frame_rate
                if step_duration > 0.5:  
                    step_times.append(step_duration)
                i += 1
            else:
                step_duration = (left_strike - right_strike) / frame_rate
                if step_duration > 0.5:
                    step_times.append(step_duration)
                j += 1
        return step_times

    step_times = calculate_step_times(left_heel_strikes, right_heel_strikes)
    average_step_time = np.mean(step_times) if step_times else 0
    total_step_time = np.sum(step_times)

    # Print step results
    print(f"File: {os.path.basename(file_path)}")
    print("Step Times (s):", step_times)
    print("Total Steps:", len(step_times))
    print("Average Step Time (s):", average_step_time)
    print("Total Step Time (s):", total_step_time)

    # Return all results
    return {
        "file_name": os.path.basename(file_path),
        "average_step_time": average_step_time,
        "total_step_time": total_step_time,
        "total_steps": len(step_times),
    }

def plot_swing_times(file_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for result in file_results:
        file_name = result["file_name"]
        left_swing_times = result.get("left_swing_times", [])
        right_swing_times = result.get("right_swing_times", [])

        # Plot both sides together
        min_length = min(len(left_swing_times), len(right_swing_times))
        plt.figure(figsize=(10, 6))
        indices = range(min_length)
        plt.bar(indices, left_swing_times[:min_length], width=0.4, label="Left Swing Times", align='center')
        plt.bar(indices, right_swing_times[:min_length], width=0.4, label="Right Swing Times", align='edge')
        plt.xlabel("Swing Index")
        plt.ylabel("Swing Time (s)")
        plt.title(f"Comparison of Swing Times for {file_name}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{file_name}_combined_swing_times.png"))
        plt.close()

def process_multiple_files(file_paths, output_dir):
    results = []
    os.makedirs(output_dir, exist_ok=True)

    for file_path in file_paths:
        result = process_file(file_path, output_dir)
        results.append(result)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "step_analysis_summary.csv"), index=False)
    print("Results saved to:", os.path.join(output_dir, "step_analysis_summary.csv"))

def group_csvs_by_subject(output_dir):
    """
    Groups CSV files in the output_dir by subject ID (prefix before '_Day').
    Returns a dictionary {subject_id: [list_of_file_paths]} and a combined flat file_list.
    """

    csv_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {output_dir}")
        return []

    subject_groups = defaultdict(list)

    # Match subject ID before '_Day'
    pattern = re.compile(r'^(?P<subject>.+?)_Day', re.IGNORECASE)

    for f in csv_files:
        match = pattern.match(f)
        subject = match.group('subject') if match else "Unknown"
        full_path = os.path.abspath(os.path.join(output_dir, f))
        subject_groups[subject].append(full_path)

    # Sort files by numeric day if available
    day_pattern = re.compile(r'_Day(\d+)', re.IGNORECASE)
    for subject, files in subject_groups.items():
        def extract_day_num(fname):
            m = day_pattern.search(fname)
            return int(m.group(1)) if m else 9999
        subject_groups[subject] = sorted(files, key=extract_day_num)

    # Combine all subjects into one flat list
    file_list = []
    for subject, files in subject_groups.items():
        print(f"\nSubject: {subject}")
        for f in files:
            print(f"  {f}")
        file_list.extend(files)

    return file_list
# Input files and output directory
file_list = group_csvs_by_subject("folder_path")

output_directory = "C:/Users/gowri_xrbkm8c/VSenv/U/Step_Analysis_Output"
process_multiple_files(file_list, output_directory)
