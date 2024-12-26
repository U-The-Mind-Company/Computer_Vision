import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import fft, fftfreq

# Step 1: Load and preprocess the data
def load_data(file_path):
    """Load the CSV file containing keypoints data."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Step 2: Calculate Euclidean distances between consecutive frames
def calculate_euclidean_distances(df):
    """Calculate Euclidean distances between consecutive frames for all keypoints."""
    frame_column = 'Frame'
    keypoints_x_columns = [col for col in df.columns if col.endswith('_x')]
    keypoints_y_columns = [col for col in df.columns if col.endswith('_y')]

    distances = pd.DataFrame(columns=[frame_column] + [col[:-2] for col in keypoints_x_columns])

    for i in range(1, len(df)):
        x_coords_prev = df.loc[i - 1, keypoints_x_columns].values
        y_coords_prev = df.loc[i - 1, keypoints_y_columns].values
        x_coords_curr = df.loc[i, keypoints_x_columns].values
        y_coords_curr = df.loc[i, keypoints_y_columns].values

        frame_distances = [
            np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            for x1, y1, x2, y2 in zip(x_coords_prev, y_coords_prev, x_coords_curr, y_coords_curr)
        ]

        distances.loc[i] = [df.loc[i, frame_column]] + frame_distances

    return distances

# Step 3: Interpolate missing values
def check_and_interpolate(df):
    """Check for missing values and apply interpolation."""
    missing_values = df.isna().sum().sum()
    if missing_values > 0:
        print(f"Missing values detected: {missing_values}. Applying interpolation...")
        df_interpolated = df.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')
        print(f"Missing values after interpolation: {df_interpolated.isna().sum().sum()}")
        return df_interpolated
    else:
        print("No missing values found.")
        return df

# Step 4: Apply Common Average Reference (CAR)
def common_average_reference(df):
    """Apply Common Average Reference (CAR) row-wise."""
    row_means = df.mean(axis=1)
    return df.sub(row_means, axis=0)

# Step 5: Apply Gaussian smoothing
def gaussian_smooth(df, sigma=2):
    """Apply Gaussian filter to smooth the data."""
    return pd.DataFrame({col: gaussian_filter1d(df[col].values, sigma) for col in df.columns})

# Step 6: Process hand keypoints data for a single file
def process_hand_keypoints(input_file, output_prefix):
    df = load_data(input_file)
    if df is None:
        return None, None, None, None

    # Calculate Euclidean distances
    distances = calculate_euclidean_distances(df)
    distances.to_csv(f"{output_prefix}_euclidean_distances.csv", index=False)
    print(f"Euclidean distances calculated for {input_file}. Shape: {distances.shape}")

    # Interpolate missing values
    df_interpolated = check_and_interpolate(distances)
    print(f"Interpolation complete for {input_file}. Shape: {df_interpolated.shape}")

    # Apply Common Average Reference (CAR)
    df_car = common_average_reference(df_interpolated.iloc[:, 1:])  # Exclude 'Frame' for CAR
    df_car.insert(0, 'Frame', df_interpolated['Frame'])
    df_car.to_csv(f"{output_prefix}_car.csv", index=False)
    print(f"Common Average Reference complete for {input_file}. Shape: {df_car.shape}")

    # Apply Gaussian smoothing
    df_gaussian = gaussian_smooth(df_car.iloc[:, 1:])  # Exclude 'Frame' for smoothing
    df_gaussian.insert(0, 'Frame', df_car['Frame'])
    df_gaussian.to_csv(f"{output_prefix}_gaussian_after_car.csv", index=False)
    print(f"Gaussian smoothing complete for {input_file}. Shape: {df_gaussian.shape}")

    return df, df_interpolated, df_car, df_gaussian

# Step 7: Plot sample results
def plot_sample_results(original_df, interpolated_df, car_df, gaussian_df, sample_column):
    """Plot results for comparison."""
    plt.figure(figsize=(15, 10))
    plt.plot(original_df['Frame'], original_df[sample_column], label='Original', alpha=0.5, color='red')
    plt.plot(interpolated_df['Frame'], interpolated_df[sample_column], label='Interpolated', alpha=0.5, color='blue')
    plt.plot(car_df['Frame'], car_df[sample_column], label='CAR', alpha=0.7)
    plt.plot(gaussian_df['Frame'], gaussian_df[sample_column], label='Gaussian after CAR', alpha=0.7)
    plt.title(f'Comparison of Processing Methods for {sample_column}')
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# Step 8: Plot average amplitudes for two datasets
def calculate_average_amplitude(amplitudes):
    """Calculate the average amplitude for a given hand."""
    return amplitudes.mean(axis=1)

def plot_average_amplitude(frames, avg1, avg2, labels):
    """Plot the average amplitudes for two CSV files."""
    plt.figure(figsize=(14, 8))

    # Plot average amplitude for each dataset
    plt.plot(frames, avg1, label=labels[0], color='orange')
    plt.plot(frames, avg2, label=labels[1], color='crimson')

    plt.title('Overall Movement of the Hand')
    plt.xlabel('Frames')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.ylim(-8e-14, 8e-14)
    plt.xlim(0, 1000)  # Adjust Y limits as per your data range
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Step 9: Power Spectral Density (PSD) Analysis
def psd_analysis(pre_data, post_data):
    # Step 1: Calculate the average magnitude across all keypoints for each frame
    keypoint_cols = pre_data.columns[1:]  # Exclude 'Frame' column
    pre_data['Average_Magnitude'] = pre_data[keypoint_cols].mean(axis=1)
    post_data['Average_Magnitude'] = post_data[keypoint_cols].mean(axis=1)

    # Step 2: Scale the average magnitude to enhance visibility (optional scaling factor)
    scaling_factor = 1e14
    pre_magnitude_values = pre_data['Average_Magnitude'].dropna() * scaling_factor
    post_magnitude_values = post_data['Average_Magnitude'].dropna() * scaling_factor

    # Convert 'Average_Magnitude' to NumPy arrays for processing
    pre_magnitude_values_np = pre_magnitude_values.to_numpy()
    post_magnitude_values_np = post_magnitude_values.to_numpy()

    # PSD Analysis with Welch's method
    fs = 30  # Sample rate in Hz (adjust if needed)
    frequencies_pre, psd_values_pre = welch(pre_magnitude_values_np, fs, nperseg=256)
    frequencies_post, psd_values_post = welch(post_magnitude_values_np, fs, nperseg=256)

    # Define the tremor frequency band (3-12 Hz for essential tremors)
    band_low = 3
    band_high = 12

    # Calculate Band Power for the 3-12 Hz frequency range
    def calculate_band_power(frequencies, psd_values, band_low, band_high):
        band_indices = (frequencies >= band_low) & (frequencies <= band_high)
        band_power = np.trapz(psd_values[band_indices], frequencies[band_indices])
        return band_power

    # Calculate band power for both pre- and post-treatment
    band_power_pre = calculate_band_power(frequencies_pre, psd_values_pre, band_low, band_high)
    band_power_post = calculate_band_power(frequencies_post, psd_values_post, band_low, band_high)

    # Calculate the percentage change in band power
    change_in_band_power = ((band_power_post - band_power_pre) / band_power_pre) * 100

    # Plot PSD for pre- and post-treatment with band power comparison
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies_pre, psd_values_pre, label=" Day 3 Pre-treatment PSD")
    plt.plot(frequencies_post, psd_values_post, label="Day 3 Post-treatment PSD", linestyle='--')
    plt.fill_between(frequencies_pre, psd_values_pre, where=(frequencies_pre >= band_low) & (frequencies_pre <= band_high), color='green', alpha=0.3, label="Tremor Band (3-12 Hz)")
    plt.fill_between(frequencies_post, psd_values_post, where=(frequencies_post >= band_low) & (frequencies_post <= band_high), color='green', alpha=0.3)
    plt.title(f"Power Spectral Density (PSD) Comparison: Pre vs Post Treatment (Scaled by 1 x 10^14)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.ylim(0, 1.5)
    plt.legend()

    # Annotate Band Power Change on the Plot
    plt.annotate(f'Band Power Change: {change_in_band_power:.2f}%', 
                 xy=(frequencies_post[100], np.max(psd_values_post) * 0.8), 
                 xytext=(frequencies_post[100] + 1, np.max(psd_values_post) * 0.85),
                 arrowprops=dict(facecolor='black', arrowstyle="->"))

    # Display Band Power Values on the Plot
    plt.text(0.95, 0.95, 
             f'Pre-treatment Band Power (3-12 Hz): {band_power_pre:.4f}\n'
             f'Post-treatment Band Power (3-12 Hz): {band_power_post:.4f}\n'
             f'Change in Band Power: {change_in_band_power:.2f}%', 
             fontsize=12, color='black', 
             ha='right', va='top', transform=plt.gca().transAxes)

    plt.show()

    # Print out the calculated changes
    print(f"Pre-treatment Band Power (3-12 Hz): {band_power_pre:.4f}")
    print(f"Post-treatment Band Power (3-12 Hz): {band_power_post:.4f}")
    print(f"Change in Band Power: {change_in_band_power:.2f}%")

# Main script
# Main script
if __name__ == "__main__":
    print("Hand Tremor Analysis Program")
    print("-" * 30)
    
    # Get input file paths from user
    print("\nPlease enter the paths to your CSV files:")
    print("Example format: C:\\Users\\YourName\\Documents\\data.csv")
    
    # Get first file path
    while True:
        file1 = input("\nEnter path for the first CSV file (pre-treatment): ").strip('"').strip("'")
        if os.path.exists(file1):
            break
        print("Error: File not found. Please enter a valid file path.")
    
    # Get second file path
    while True:
        file2 = input("Enter path for the second CSV file (post-treatment): ").strip('"').strip("'")
        if os.path.exists(file2):
            break
        print("Error: File not found. Please enter a valid file path.")
    
    # Get output prefixes
    prefix1 = input("\nEnter prefix for first file outputs (e.g., 'day3'): ").strip()
    prefix2 = input("Enter prefix for second file outputs (e.g., 'day4after'): ").strip()
    
    input_files = [file1, file2]
    output_prefixes = [prefix1, prefix2]
    
    print("\nProcessing files...")
    print(f"File 1: {input_files[0]}")
    print(f"File 2: {input_files[1]}")
    
    # Process the hand keypoints data for both files
    results = []
    for input_file, output_prefix in zip(input_files, output_prefixes):
        print(f"\nProcessing {input_file}...")
        results.append(process_hand_keypoints(input_file, output_prefix))
    
    # Plot a sample result if data exists
    for result, input_file in zip(results, input_files):
        original_df, interpolated_df, car_df, gaussian_df = result
        if original_df is not None and not original_df.empty:
            sample_column = original_df.columns[1]  # Select first data column as sample
            plot_sample_results(original_df, interpolated_df, car_df, gaussian_df, sample_column)
        else:
            print(f"Processing failed or resulted in empty DataFrames for {input_file}. Unable to plot results.")
    
    # Perform PSD Analysis
    if len(results) == 2:
        _, _, _, df_gaussian_pre = results[0]
        _, _, _, df_gaussian_post = results[1]
        print("\nPerforming PSD Analysis...")
        psd_analysis(df_gaussian_pre, df_gaussian_post)
    
    print("\nAnalysis complete!")
