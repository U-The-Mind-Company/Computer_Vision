import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import defaultdict
import re

def calculate_average_stride_lengths(file_paths, output_dir, sigma=2, window_size=20):
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

        def calculate_stride_lengths(heel_strikes, heel_x_col, heel_y_col):
            stride_lengths = []
            for i in range(len(heel_strikes) - 1):
                start_idx, end_idx = heel_strikes[i], heel_strikes[i + 1]
                x1, y1 = df.loc[start_idx, [heel_x_col, heel_y_col]]
                x2, y2 = df.loc[end_idx, [heel_x_col, heel_y_col]]
                stride_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * df.loc[start_idx, 'scale_factor']
                if 0.4 <= stride_distance <= 2.0:
                    stride_lengths.append(stride_distance)
            return stride_lengths

        left_stride_lengths = calculate_stride_lengths(l_heel_strikes, 'left_heel_x', 'smoothed_left_heel_y')
        right_stride_lengths = calculate_stride_lengths(r_heel_strikes, 'right_heel_x', 'smoothed_right_heel_y')

        avg_left_stride = np.mean(left_stride_lengths) if left_stride_lengths else 0
        avg_right_stride = np.mean(right_stride_lengths) if right_stride_lengths else 0

        # Append results
        results.append({
            "Day": os.path.basename(file_path).split('_')[0],
            "Avg_Left_Stride_Length": avg_left_stride,
            "Avg_Right_Stride_Length": avg_right_stride
        })

    return pd.DataFrame(results)

def plot_average_stride_lengths(results_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Create bar chart for average stride lengths
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.35

    plt.bar(x - width / 2, results_df['Avg_Left_Stride_Length'], width, label='Left Stride Length')
    plt.bar(x + width / 2, results_df['Avg_Right_Stride_Length'], width, label='Right Stride Length')

    plt.xticks(x, results_df['Day'], rotation=45)
    plt.xlabel('Days')
    plt.ylabel('Average Stride Length (m)')
    plt.title('Average Stride Lengths Over Days')
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'average_stride_lengths_comparison.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Stride length comparison plot saved to {output_path}")

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
output_directory = "C:/Users/gowri_xrbkm8c/VSenv/U/stride_length_analysis"

# Process files and generate plot
results = calculate_average_stride_lengths(file_list, output_directory)
plot_average_stride_lengths(results, output_directory)
