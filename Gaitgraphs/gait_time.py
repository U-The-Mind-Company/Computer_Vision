import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys
import defaultdict
import re


def process_file(file_path, sigma=2, frame_rate=30, window_size=20):
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

    # Detect step times
    def detect_step_times(heel_x, heel_y, thresholds):
        step_times = []
        swing_start = None
        stance_start = None

        for frame in range(1, len(heel_x)):
            x_diff = abs(heel_x[frame] - heel_x[frame - 1])
            y_diff = abs(heel_y[frame] - heel_y[frame - 1])

            if x_diff > thresholds[frame] or y_diff > thresholds[frame]:
                if swing_start is None:
                    swing_start = frame
                if stance_start is not None:
                    stance_duration = (frame - stance_start) / frame_rate
                    step_times.append(('stance', stance_duration))
                    stance_start = None
            else:
                if swing_start is not None:
                    swing_duration = (frame - swing_start) / frame_rate
                    step_times.append(('swing', swing_duration))
                    swing_start = None
                if stance_start is None:
                    stance_start = frame

        if swing_start is not None:
            swing_duration = (len(heel_x) - swing_start) / frame_rate
            step_times.append(('swing', swing_duration))
        if stance_start is not None:
            stance_duration = (len(heel_x) - stance_start) / frame_rate
            step_times.append(('stance', stance_duration))

        return step_times

    left_step_times = detect_step_times(left_heel_x, left_heel_y, left_threshold)
    right_step_times = detect_step_times(right_heel_x, right_heel_y, right_threshold)

    # Extract swing times for visualization
    left_swing_times = [time for type_, time in left_step_times if type_ == 'swing' and time >= 0.7]
    right_swing_times = [time for type_, time in right_step_times if type_ == 'swing' and time >= 0.7]

    # Print results to console
    print(f"File: {os.path.basename(file_path)}")
    print(f"Total Swing Time (Left Heel): {sum(left_swing_times):.2f} seconds")
    print(f"Total Swing Time (Right Heel): {sum(right_swing_times):.2f} seconds")
    if left_swing_times:
        print(f"Average Swing Time (Left Heel): {np.mean(left_swing_times):.2f} seconds")
    else:
        print("Average Swing Time (Left Heel): No swings detected.")
    if right_swing_times:
        print(f"Average Swing Time (Right Heel): {np.mean(right_swing_times):.2f} seconds")
    else:
        print("Average Swing Time (Right Heel): No swings detected.")
    print("-" * 50)

    return {
        "file_name": os.path.basename(file_path),
        "left_swing_times": left_swing_times,
        "right_swing_times": right_swing_times,
    }


def plot_swing_times(file_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for result in file_results:
        file_name = result["file_name"]
        left_swing_times = result["left_swing_times"]
        right_swing_times = result["right_swing_times"]

        # Plot both sides together
        min_length = min(len(left_swing_times), len(right_swing_times))
        plt.figure(figsize=(10, 6))
        indices = range(min_length)
        plt.bar(indices, left_swing_times[:min_length], width=0.4, label="Left Swing Times", align='center')
        plt.bar(indices, right_swing_times[:min_length], width=0.4, label="Right Swing Times", align='edge')
        plt.xlabel("Swing Index")
        plt.ylabel("Swing Time (Factor)")
        plt.title(f"Comparison of Swing Times for {file_name}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{file_name}_combined_swing_times.png"))
        plt.close()

        # Plot the remaining swings of the larger dataset
        if len(left_swing_times) > min_length:
            plt.figure(figsize=(10, 6))
            plt.bar(range(min_length, len(left_swing_times)), left_swing_times[min_length:], color='blue', label="Remaining Left Swings")
            plt.xlabel("Swing Index")
            plt.ylabel("Swing Time (Factor)")
            plt.title(f"Remaining Left Swing Times for {file_name}")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{file_name}_remaining_left_swing_times.png"))
            plt.close()

        if len(right_swing_times) > min_length:
            plt.figure(figsize=(10, 6))
            plt.bar(range(min_length, len(right_swing_times)), right_swing_times[min_length:], color='orange', label="Remaining Right Swings")
            plt.xlabel("Swing Index")
            plt.ylabel("Swing Time (s)")
            plt.title(f"Remaining Right Swing Times for {file_name}")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{file_name}_remaining_right_swing_times.png"))
            plt.close()
        


def process_multiple_files(file_paths, output_dir):
    file_results = []

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Redirect stdout to a file in the output directory
    output_file_path = os.path.join(output_dir, 'output.txt')  # Save in the output folder
    original_stdout = sys.stdout
    with open(output_file_path, 'w') as f:
        sys.stdout = f  # Redirect stdout to the file

        # Process files and print results
        for file_path in file_paths:
            result = process_file(file_path)
            file_results.append(result)

        # Restore stdout after processing
        sys.stdout = original_stdout

    # Generate plots after restoring stdout
    plot_swing_times(file_results, output_dir)



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
output_directory = "C:/Users/gowri_xrbkm8c/VSenv/U/Gait_Time_Analysis"
process_multiple_files(file_list, output_directory)
