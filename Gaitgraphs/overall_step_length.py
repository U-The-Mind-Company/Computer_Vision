import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import defaultdict
import re

def calculate_average_step_lengths(file_paths, sigma=2, window_size=20, min_step_threshold=0.4, max_step_threshold=1.2):
    results = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)

        # Apply Gaussian smoothing
        df['smoothed_left_heel_y'] = gaussian_filter1d(df['left_heel_y'], sigma=sigma)
        df['smoothed_right_heel_y'] = gaussian_filter1d(df['right_heel_y'], sigma=sigma)

        # Dynamic thresholding for left and right heels
        df['left_dynamic_threshold'] = (
            df['smoothed_left_heel_y'].rolling(window=window_size).mean() -
            df['smoothed_left_heel_y'].rolling(window=window_size).std()
        )
        df['right_dynamic_threshold'] = (
            df['smoothed_right_heel_y'].rolling(window=window_size).mean() -
            df['smoothed_right_heel_y'].rolling(window=window_size).std()
        )

        # Identify heel strikes
        l_heel_strikes = df.index[
            (df['smoothed_left_heel_y'] < df['left_dynamic_threshold']) &
            (df['smoothed_left_heel_y'].shift(1) >= df['left_dynamic_threshold'].shift(1))
        ].tolist()
        r_heel_strikes = df.index[
            (df['smoothed_right_heel_y'] < df['right_dynamic_threshold']) &
            (df['smoothed_right_heel_y'].shift(1) >= df['right_dynamic_threshold'].shift(1))
        ].tolist()

        # Calculate scale factors
        df['scale_factor'] = df.apply(
            lambda row: 1.7 / math.sqrt(
                (row['face-60_y'] - min(row['smoothed_left_heel_y'], row['smoothed_right_heel_y']))**2 +
                (row['face-60_x'] - min(row['left_heel_x'], row['right_heel_x']))**2
            ) if math.sqrt(
                (row['face-60_y'] - min(row['smoothed_left_heel_y'], row['smoothed_right_heel_y']))**2 +
                (row['face-60_x'] - min(row['left_heel_x'], row['right_heel_x']))**2
            ) > 0 else 0,
            axis=1
        )

        # Calculate step lengths
        step_lengths = []
        for i in range(min(len(l_heel_strikes), len(r_heel_strikes)) - 1):
            if l_heel_strikes[i] < r_heel_strikes[i]:
                start_index = l_heel_strikes[i]
                end_index = r_heel_strikes[i]
            else:
                start_index = r_heel_strikes[i]
                end_index = l_heel_strikes[i]

            x1, y1 = df.loc[start_index, ['left_heel_x', 'smoothed_left_heel_y']] if start_index in l_heel_strikes else df.loc[start_index, ['right_heel_x', 'smoothed_right_heel_y']]
            x2, y2 = df.loc[end_index, ['right_heel_x', 'smoothed_right_heel_y']] if end_index in r_heel_strikes else df.loc[end_index, ['left_heel_x', 'smoothed_left_heel_y']]

            step_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            scale_factor = df.loc[start_index, 'scale_factor']
            real_step_length = step_distance * scale_factor

            if min_step_threshold <= real_step_length <= max_step_threshold:
                step_lengths.append(real_step_length)

        avg_step_length = np.mean(step_lengths) if step_lengths else 0

        # Append results
        results.append({
            "Day": os.path.basename(file_path).split('_')[0],
            "Avg_Step_Length": avg_step_length
        })

    return pd.DataFrame(results)

def plot_average_step_lengths(results_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Create line plot for average step lengths
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Day'], results_df['Avg_Step_Length'], marker='o', linestyle='-', color='green', label='Average Step Length')

    plt.xlabel('Days')
    plt.ylabel('Average Step Length (m)')
    plt.title('Average Step Lengths Over Days')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'average_step_lengths_line_plot.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Step length plot saved to {output_path}")
    print(results_df['Avg_Step_Length'])

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
output_directory = "C:/Users/gowri_xrbkm8c/VSenv/U/step_length_analysis"

# Process files and generate plot
results = calculate_average_step_lengths(file_list)
plot_average_step_lengths(results, output_directory)
