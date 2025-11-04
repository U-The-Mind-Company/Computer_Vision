import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import defaultdict
import re

def process_and_extract_trends(file_list, sigma=2, frame_rate=30, window_size=20):
    """
    Processes files to calculate swing time trends directly from headers.

    :param file_list: List of file paths containing gait data.
    :param sigma: Smoothing factor for Gaussian filter.
    :param frame_rate: Frame rate of the data (frames per second).
    :param window_size: Window size for calculating dynamic thresholds.
    :return: Pandas DataFrame containing trends for plotting.
    """
    trends = []

    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)

            # Apply Gaussian smoothing
            left_heel_x = gaussian_filter1d(df['left_heel_x'].values, sigma)
            left_heel_y = gaussian_filter1d(df['left_heel_y'].values, sigma)
            right_heel_x = gaussian_filter1d(df['right_heel_x'].values, sigma)
            right_heel_y = gaussian_filter1d(df['right_heel_y'].values, sigma)

            # Calculate dynamic thresholds
            def calculate_dynamic_threshold(heel_x, heel_y):
                movement = np.sqrt(np.diff(heel_x, prepend=heel_x[0]) ** 2 + np.diff(heel_y, prepend=heel_y[0]) ** 2)
                rolling_threshold = pd.Series(movement).rolling(window=window_size, min_periods=1).std() * 1.5
                return rolling_threshold.values

            left_threshold = calculate_dynamic_threshold(left_heel_x, left_heel_y)
            right_threshold = calculate_dynamic_threshold(right_heel_x, right_heel_y)

            # Detect swing times
            def detect_swing_times(heel_x, heel_y, thresholds):
                swing_times = []
                swing_start = None
                for frame in range(1, len(heel_x)):
                    x_diff = abs(heel_x[frame] - heel_x[frame - 1])
                    y_diff = abs(heel_y[frame] - heel_y[frame - 1])

                    if x_diff > thresholds[frame] or y_diff > thresholds[frame]:
                        if swing_start is None:
                            swing_start = frame
                    else:
                        if swing_start is not None:
                            swing_duration = (frame - swing_start) / frame_rate
                            swing_times.append(swing_duration)
                            swing_start = None

                return swing_times

            left_swing_times = detect_swing_times(left_heel_x, left_heel_y, left_threshold)
            right_swing_times = detect_swing_times(right_heel_x, right_heel_y, right_threshold)

            avg_left_swing = np.mean(left_swing_times) if left_swing_times else 0
            avg_right_swing = np.mean(right_swing_times) if right_swing_times else 0
            avg_total_swing = (avg_left_swing + avg_right_swing) / 2

            trends.append({
                'file_name': os.path.basename(file_path),
                'avg_left_swing': avg_left_swing,
                'avg_right_swing': avg_right_swing,
                'avg_total_swing': avg_total_swing
            })
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return pd.DataFrame(trends)

def plot_trends_from_headers(trends_df, output_dir):
    """
    Plots line graphs showing trends directly from headers.

    :param trends_df: DataFrame containing trends data.
    :param output_dir: Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure file names are strings and handle missing or invalid values
    trends_df['file_name'] = trends_df['file_name'].fillna('Unknown').astype(str)

    # Filter to include only valid categories (sessions)
    valid_categories = [
        "Day1_Before.csv",
        "Day7_Before.csv",        
        "Day8_Before.csv",
        "Day14_Before.csv",
        "Day20_After.csv"
    ]
    trends_df['file_name'] = pd.Categorical(
        trends_df['file_name'], categories=valid_categories, ordered=True
    )
    trends_df = trends_df.dropna(subset=['file_name']).reset_index(drop=True)

    # Plot line graphs
    plt.figure(figsize=(12, 8))
    plt.plot(trends_df['file_name'], trends_df['avg_left_swing'], marker='o', label='Average Left Swing Time', linestyle='-')
    plt.plot(trends_df['file_name'], trends_df['avg_right_swing'], marker='o', label='Average Right Swing Time', linestyle='-')
    plt.plot(trends_df['file_name'], trends_df['avg_total_swing'], marker='o', label='Overall Average Swing Time', linestyle='--')

    plt.xlabel("Files (Sessions)")
    plt.ylabel("Swing Time (Factor)")
    plt.title("Swing Time Trends Across Sessions")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_path = os.path.join(output_dir, "swing_time_trends_from_headers.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Line graph saved to: {output_path}")
    print(trends_df['avg_left_swing'])
    print(trends_df['avg_right_swing'])
    print(trends_df['avg_total_swing'])
    
    return output_path


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
# Example usage
if __name__ == "__main__":
    file_list = group_csvs_by_subject("folder_path")

    output_directory = "C:/Users/gowri_xrbkm8c/VSenv/U/gait_time_trend"

    trends_df = process_and_extract_trends(file_list)
    plot_trends_from_headers(trends_df, output_directory)
