import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.ndimage import gaussian_filter1d

def extract_day_number(filename):
    """Extract day number from filename like 'sub001_day1_before.csv'"""
    match = re.search(r'sub\d{3}_day(\d+)_(before|after)', filename.lower())
    if match:
        return int(match.group(1)), match.group(2)
    return None, None

def calculate_tremor_amplitude(filepath):
    """Calculate tremor amplitude from hand keypoint data"""
    try:
        df = pd.read_csv(filepath)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Get x and y columns based on your CSV structure
        # Looking for columns like x_0, x_1, ... and y_0, y_1, ...
        x_columns = [col for col in df.columns if col.startswith('x_') and col[2:].isdigit()]
        y_columns = [col for col in df.columns if col.startswith('y_') and col[2:].isdigit()]
        
        if not x_columns or not y_columns:
            print(f"Warning: No keypoint columns found in {filepath}")
            return None
        
        # Sort columns to ensure matching pairs (x_0 with y_0, etc.)
        x_columns = sorted(x_columns, key=lambda x: int(x.split('_')[1]))
        y_columns = sorted(y_columns, key=lambda y: int(y.split('_')[1]))
        
        # Ensure we have matching number of x and y columns
        num_keypoints = min(len(x_columns), len(y_columns))
        x_columns = x_columns[:num_keypoints]
        y_columns = y_columns[:num_keypoints]
        
        # Calculate frame-to-frame distances for all keypoints
        all_distances = []
        
        for i in range(1, len(df)):
            x_coords_prev = df.loc[i - 1, x_columns].values
            y_coords_prev = df.loc[i - 1, y_columns].values
            x_coords_curr = df.loc[i, x_columns].values
            y_coords_curr = df.loc[i, y_columns].values
            
            # Calculate Euclidean distance for each keypoint
            frame_distances = []
            for x1, y1, x2, y2 in zip(x_coords_prev, y_coords_prev, x_coords_curr, y_coords_curr):
                # Skip if any value is NaN
                if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    frame_distances.append(distance)
            
            # Average distance across all valid keypoints for this frame
            if frame_distances:
                all_distances.append(np.mean(frame_distances))
        
        if not all_distances:
            print(f"Warning: Could not calculate distances for {filepath}")
            return None
            
        # Preprocessing: Interpolate any NaN values
        distances_series = pd.Series(all_distances)
        distances_series = distances_series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        # Apply Common Average Reference (CAR) - optional
        # This subtracts the mean to remove common noise
        distances_series = distances_series - distances_series.mean()
        
        # Apply Gaussian smoothing to reduce noise
        smoothed_distances = gaussian_filter1d(distances_series.values, sigma=2)
        
        # Return the average tremor amplitude (mean of absolute smoothed distances)
        avg_tremor = np.mean(np.abs(smoothed_distances))
        
        return avg_tremor
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def plot_tremor_data(directory_path):
    """
    Process all CSV files in directory and create before/after treatment plot
    
    Parameters:
    directory_path: Path to directory containing CSV files
    """
    # Dictionary to store data: {day_number: {'before': avg, 'after': avg}}
    tremor_data = {}
    
    # Get all CSV files in directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
    
    print(f"\nFound {len(csv_files)} CSV files to process...")
    
    # Process each file
    for filename in csv_files:
        day_num, timing = extract_day_number(filename)
        
        if day_num is None:
            print(f"Skipping {filename} - doesn't match expected format")
            continue
        
        print(f"Processing {filename}...")
        filepath = os.path.join(directory_path, filename)
        avg_tremor = calculate_tremor_amplitude(filepath)
        
        if avg_tremor is not None:
            if day_num not in tremor_data:
                tremor_data[day_num] = {}
            tremor_data[day_num][timing] = avg_tremor
            print(f"  Day {day_num} {timing}: Tremor amplitude = {avg_tremor:.2e}")
    
    if not tremor_data:
        print("No valid tremor data found")
        return
    
    # Prepare data for plotting - single continuous line
    days = sorted(tremor_data.keys())
    x_values = []
    y_values = []
    
    # Combine before and after measurements in chronological order
    for day in days:
        if 'before' in tremor_data[day]:
            x_values.append(day)
            y_values.append(tremor_data[day]['before'])
        
        if 'after' in tremor_data[day]:
            x_values.append(day + 0.5)
            y_values.append(tremor_data[day]['after'])
    
    # Create the plot with single continuous line (ORIGINAL STYLE)
    plt.figure(figsize=(12, 6))
    
    plt.plot(x_values, y_values, 'o-', color='#3b82f6',
             markersize=6, linewidth=2)
    
    plt.xlabel('Day', fontsize=12, fontweight='bold')
    plt.ylabel('Average Tremor Amplitude', fontsize=12, fontweight='bold')
    plt.title('Tremor Amplitude Over Treatment Period', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show all days
    if days:
        plt.xticks(range(min(days), max(days) + 1))
    
    plt.tight_layout()
    plt.savefig('tremor_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== Tremor Analysis Summary ===")
    print(f"Total days analyzed: {len(days)}")
    print(f"\nDay-by-day breakdown:")
    for day in days:
        before = tremor_data[day].get('before', 'N/A')
        after = tremor_data[day].get('after', 'N/A')
        print(f"Day {day}:")
        print(f"  Before: {before:.2e}" if isinstance(before, float) else f"  Before: {before}")
        print(f"  After:  {after:.2e}" if isinstance(after, float) else f"  After: {after}")
        if isinstance(before, float) and isinstance(after, float):
            change = ((after - before) / before) * 100
            print(f"  Change: {change:+.1f}%")

# Example usage
if __name__ == "__main__":
    # Dir Path
    data_directory = input("Enter the path to your tremor data directory: ").strip()
    
    # Remove quotes if user included them
    data_directory = data_directory.strip('"').strip("'")
    
    # Check if directory exists
    if not os.path.exists(data_directory):
        print(f"Directory '{data_directory}' not found.")
        print("Please check the path and try again.")
    else:
        plot_tremor_data(data_directory)
