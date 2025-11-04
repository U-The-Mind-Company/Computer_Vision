import os
# Import the re module, which helps us work with text and patterns (regular expressions)
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
# Import savgol_filter for smoothing data (removing noise)
from scipy.signal import savgol_filter
# Import pearsonr for calculating correlation between two sets of numbers
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

# This function extracts the day number and before/after label from the filename
def extract_day_and_label(filename):
    # Example filename: "Smith, John: Day12_Before.csv"
    # We use a regular expression to find the day number and the label (Before or After)
    match = re.search(r'Day(\d+)_?(Before|After)?', filename, re.IGNORECASE)
    if match:
        day = int(match.group(1))  # Get the number after 'Day'
        label = match.group(2).capitalize() if match.group(2) else ""  # Get 'Before' or 'After'
        return day, label
    else:
        # If the filename doesn't match the pattern, show an error
        raise ValueError(f"Filename '{filename}' does not match expected pattern (should contain 'DayX_Before' or 'DayX_After').")

# This function reads all CSV files in a folder, adds the extracted day and label, and combines them into one DataFrame
def read_all_metrics_from_folder(folder_path):
    all_dfs = []  # This will hold all the data tables we read
    for fname in os.listdir(folder_path):  # Go through every file in the folder
        # Only look at CSV files that have 'day' in their name
        if fname.lower().endswith('.csv') and 'day' in fname.lower():
            day, label = extract_day_and_label(fname)  # Get the day and label from the filename
            df = pd.read_csv(os.path.join(folder_path, fname))  # Read the CSV file into a DataFrame
            df['Days'] = day  # Add the day as a new column
            df['Label'] = label  # Add the label as a new column
            df['SourceFile'] = fname  # Save the filename for reference
            all_dfs.append(df)  # Add this DataFrame to our list
    if not all_dfs:
        # If we didn't find any files, show an error
        raise ValueError("No valid CSV files found in the folder.")
    # Combine all the DataFrames into one big DataFrame
    df_all = pd.concat(all_dfs, ignore_index=True)
    # Sort the data by day so it's in order
    df_all = df_all.sort_values('Days')
    return df_all

# This function makes sure filenames are safe for saving (removes special characters)
def sanitize_filename(filename):
    # Replace any characters that are not allowed in filenames with an underscore
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# This function gives us friendly names for each metric, so our graphs are easy to read
def get_patient_friendly_labels():
    # This dictionary maps the technical column names to easy-to-read labels
    return {
        "DS_Asymmetry (%)": "Double Support Asymmetry",
        "Left_DS_Mean (s)": "Left Double Support Time",
        "Right_DS_Mean (s)": "Right Double Support Time",
        "Left_StepTime_CV (%)": "Left Step Time Variability",
        "Left_StepTime_Mean (s)": "Left Step Time",
        "Right_StepTime_CV (%)": "Right Step Time Variability",
        "Right_StepTime_Mean (s)": "Right Step Time",
        "StepTime_Asymmetry (%)": "Step Time Asymmetry",
        "Left_StepVelocity_CV (%)": "Left Step Velocity Variability",
        "Left_StepVelocity_Mean (m/s)": "Left Step Velocity",
        "Right_StepVelocity_CV (%)": "Right Step Velocity Variability",
        "Right_StepVelocity_Mean (m/s)": "Right Step Velocity",
        "StepVelocity_Asymmetry (%)": "Step Velocity Asymmetry",
        "Left_Step_Length_CV (%)": "Left Step Length Variability",
        "Left_Step_Length_Mean (m)": "Left Step Length",
        "Right_Step_Length_CV (%)": "Right Step Length Variability",
        "Right_Step_Length_Mean (m)": "Right Step Length",
        "Step_Length_Asymmetry (%)": "Step Length Asymmetry",
        "Left_Leg_Swing_CV (%)": "Left Leg Swing Variability",
        "Left_Leg_Swing_Mean (s)": "Left Leg Swing Time",
        "Right_Leg_Swing_CV (%)": "Right Leg Swing Variability",
        "Right_Leg_Swing_Mean (s)": "Right Leg Swing Time",
        "SwingTime_Asymmetry (%)": "Swing Time Asymmetry",
        "Speed (m/s)": "Walking Speed",
        # Upper body metrics
        'Head_Vertical_Amplitude (m)': "Head Vertical Amplitude",
        'Head_Lateral_Amplitude (m)': "Head Lateral Amplitude",
        'Head_Smoothness (RMS accel)': "Head Smoothness",
        'Shoulder_Height_Asymmetry (m)': "Shoulder Height Asymmetry",
        'Shoulder_Sway_Amplitude (m)': "Shoulder Sway Amplitude",
        'Left_Arm_Angle_Mean (deg)': "Left Arm Angle Mean",
        'Left_Arm_Swing_Amplitude (m)': "Left Arm Swing Amplitude",
        'Right_Arm_Angle_Mean (deg)': "Right Arm Angle Mean",
        'Right_Arm_Swing_Amplitude (m)': "Right Arm Swing Amplitude",
        'Arm_Swing_Asymmetry (%)': "Arm Swing Asymmetry",
        'Torso_Lean_Mean (m)': "Torso Lean Mean",
        'Torso_Arm_Coordination (r)': "Torso-Arm Coordination",
        'Upper_Body_Stability_Index': "Upper Body Stability Index"
    }

# This function calculates the trend of a metric over time
def calculate_trend_analysis(values, days):
    # If we don't have at least 2 days, we can't calculate a trend
    if len(days) < 2:
        return None
    # Use linear regression to find the best-fit line through the data
    slope, intercept, r_value, p_value, std_err = stats.linregress(days, values)
    return {
        'slope': slope,  # How much the metric changes per day
        'p_value': p_value,  # How likely it is that the trend is real (not random)
        'r_squared': r_value**2,  # How well the line fits the data (closer to 1 is better)
        'trend_strength': 'strong' if abs(slope) > np.std(values)/2 else 'moderate' if abs(slope) > np.std(values)/4 else 'weak',
        'direction': 'improving' if slope < 0 else 'worsening' if slope > 0 else 'stable',
        'percent_change': ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
    }

# This function sorts all the metrics into categories (time, distance, speed, etc.)
def classify_metrics(df):
    # We make a dictionary to hold lists of columns for each type of metric
    metric_types = {
        'time_based': [],
        'distance_based': [],
        'velocity_speed': [],
        'symmetry_asymmetry': [],
        'variability': [],
        'composite': []
    }
    for col in df.columns:  # Go through each column in the data
        if ('Time' in col or 'Swing' in col) and 'Asymmetry' not in col and 'CV' not in col:
            metric_types['time_based'].append(col)
        elif 'Length' in col and 'Asymmetry' not in col and 'CV' not in col:
            metric_types['distance_based'].append(col)
        elif ('Velocity' in col or 'Speed' in col) and 'Asymmetry' not in col and 'CV' not in col:
            metric_types['velocity_speed'].append(col)
        elif 'Asymmetry' in col:
            metric_types['symmetry_asymmetry'].append(col)
        elif 'CV' in col:
            metric_types['variability'].append(col)
        else:
            metric_types['composite'].append(col)
    return metric_types

# This function performs full body gait analysis (lower + upper body) for a single file
def analyze_fullbody_gait(df, height_m=None, sg_window=21, sg_poly=3, threshold_window=20, frame_rate=30):

    # --- Estimate height from body points if not provided ---
    # We'll use the vertical distance from the top of the head (nose or face-60) to the lowest heel point
    if height_m is None:
        # Use the mean of the vertical distance from face-60 (or nose if not present) to the lowest heel
        if 'face-60_y' in df.columns:
            head_y = df['face-60_y']
        elif 'nose_y' in df.columns:
            head_y = df['nose_y']
        else:
            raise ValueError("No head marker (face-60_y or nose_y) found for height estimation.")
        # Use the lowest point of either heel
        heel_y = np.minimum(df['left_heel_y'], df['right_heel_y'])
        # Height in pixels (vertical distance)
        height_pixels = np.abs(head_y - heel_y)
        # Use the median height across all frames to avoid outliers
        height_m = np.median(height_pixels)
        # Optionally, print a warning that this is in pixels, not meters
        print("Estimated height from body points (in pixels):", height_m)
        # If you want to convert to meters, you need a known reference or calibration
    
    
    
    # Smooth all relevant markers to reduce noise in the data
    markers_to_smooth = [
        'left_heel', 'right_heel', 'nose',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip'
    ]
    for marker in markers_to_smooth:
        for dim in ['x', 'y']:
            col = f'{marker}_{dim}'
            if col in df.columns:
                # Apply a smoothing filter to each marker's x and y coordinates
                df[f'smoothed_{col}'] = savgol_filter(df[col], sg_window, sg_poly)

     # Calculate scale factor (pixels to meters) using the distance from face to heel
    heel_y = np.minimum(df['left_heel_y'], df['right_heel_y'])
    heel_x = np.minimum(df['left_heel_x'], df['right_heel_x'])
    pixel_dist = np.sqrt(
        (df['face-60_x'] - heel_x)**2 +
        (df['face-60_y'] - heel_y)**2
    )
    scale_factors = height_m / pixel_dist
    df['scale_factor'] = scale_factors

    # Calculate how much the left and right heels move between frames
    left_movement = np.sqrt(
        np.diff(df['smoothed_left_heel_x'], prepend=df['smoothed_left_heel_x'][0])**2 +
        np.diff(df['smoothed_left_heel_y'], prepend=df['smoothed_left_heel_y'][0])**2
    )
    right_movement = np.sqrt(
        np.diff(df['smoothed_right_heel_x'], prepend=df['smoothed_right_heel_x'][0])**2 +
        np.diff(df['smoothed_right_heel_y'], prepend=df['smoothed_right_heel_y'][0])**2
    )
    # Calculate a threshold for detecting events (like steps)
    left_thresh = pd.Series(left_movement).rolling(threshold_window, min_periods=1).std().to_numpy() * 1.5
    right_thresh = pd.Series(right_movement).rolling(threshold_window, min_periods=1).std().to_numpy() * 1.5

    # This function detects gait events (like heel strikes and toe offs)
    def detect_events(x, y, thresh, event_type):
        x_diff = np.abs(np.diff(x, prepend=x[0]))
        y_diff = np.abs(np.diff(y, prepend=y[0]))
        stance = (x_diff <= thresh) & (y_diff <= thresh)
        if event_type == 'strike':
            swing_shift = np.roll(~stance, 1)
            swing_shift[0] = False
            return np.where(swing_shift & stance)[0]
        else:
            stance_shift = np.roll(stance, 1)
            stance_shift[0] = True
            return np.where(stance_shift & ~stance)[0]

    # Find the frames where each event happens for left and right feet
    left_strikes = detect_events(df['smoothed_left_heel_x'], df['smoothed_left_heel_y'], left_thresh, 'strike')
    right_strikes = detect_events(df['smoothed_right_heel_x'], df['smoothed_right_heel_y'], right_thresh, 'strike')
    left_toe_offs = detect_events(df['smoothed_left_heel_x'], df['smoothed_left_heel_y'], left_thresh, 'toe_off')
    right_toe_offs = detect_events(df['smoothed_right_heel_x'], df['smoothed_right_heel_y'], right_thresh, 'toe_off')

    # Calculate swing times (how long the foot is in the air)
    def calc_swing_times(toe_offs, strikes):
        if len(toe_offs) == 0 or len(strikes) == 0:
            return np.array([])
        idx_toe = np.array(toe_offs)
        idx_strike = np.array(strikes)
        next_strike_idx = np.searchsorted(idx_strike, idx_toe, side='right')
        valid = next_strike_idx < len(idx_strike)
        return (idx_strike[next_strike_idx[valid]] - idx_toe[valid]) / frame_rate

    # Calculate double support times (when both feet are on the ground)
    def calc_ds_times(strikes, opp_toe_offs):
        if len(strikes) == 0 or len(opp_toe_offs) == 0:
            return np.array([])
        idx_strike = np.array(strikes)
        idx_toe = np.array(opp_toe_offs)
        next_toe_idx = np.searchsorted(idx_toe, idx_strike, side='right')
        valid = next_toe_idx < len(idx_toe)
        return (idx_toe[next_toe_idx[valid]] - idx_strike[valid]) / frame_rate

    # Calculate distances between steps
    def calc_spatial_metrics(strikes, x_col, y_col):
        if len(strikes) < 2:
            return np.array([])
        x = df[x_col].values[strikes]
        y = df[y_col].values[strikes]
        scale = df['scale_factor'].values[strikes[:-1]]
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        return np.sqrt(dx**2 + dy**2) * scale

    # Calculate velocities for each step
    def calc_velocity_metrics(strikes, x_col, y_col):
        if len(strikes) < 2:
            return np.array([])
        idx1 = strikes[:-1]
        idx2 = strikes[1:]
        dx = df[x_col].values[idx2] - df[x_col].values[idx1]
        dy = df[y_col].values[idx2] - df[y_col].values[idx1]
        dist = np.sqrt(dx**2 + dy**2) * df['scale_factor'].values[idx1]
        time_step = (idx2 - idx1) / frame_rate
        valid = time_step > 0
        return dist[valid] / time_step[valid]

    # Calculate all the lower body metrics
    left_swing = calc_swing_times(left_toe_offs, left_strikes)
    right_swing = calc_swing_times(right_toe_offs, right_strikes)
    left_ds = calc_ds_times(left_strikes, right_toe_offs)
    right_ds = calc_ds_times(right_strikes, left_toe_offs)
    left_stride = calc_spatial_metrics(left_strikes, 'left_heel_x', 'smoothed_left_heel_y')
    right_stride = calc_spatial_metrics(right_strikes, 'right_heel_x', 'smoothed_right_heel_y')
    left_step_vel = calc_velocity_metrics(left_strikes, 'left_heel_x', 'smoothed_left_heel_y')
    right_step_vel = calc_velocity_metrics(right_strikes, 'right_heel_x', 'smoothed_right_heel_y')

    # Calculate step times (time between steps)
    left_step_times = np.diff(left_strikes) / frame_rate if len(left_strikes) > 1 else np.array([])
    right_step_times = np.diff(right_strikes) / frame_rate if len(right_strikes) > 1 else np.array([])

    # Calculate step lengths (distance between left and right strikes)
    def calc_step_lengths(strikes_a, strikes_b, xa, ya, xb, yb):
        if len(strikes_a) == 0 or len(strikes_b) == 0:
            return np.array([])
        idx_a = np.array(strikes_a)
        idx_b = np.array(strikes_b)
        next_b_idx = np.searchsorted(idx_b, idx_a, side='right')
        valid = next_b_idx < len(idx_b)
        idx_a_valid = idx_a[valid]
        idx_b_valid = idx_b[next_b_idx[valid]]
        dx = xb[idx_b_valid] - xa[idx_a_valid]
        dy = yb[idx_b_valid] - ya[idx_a_valid]
        return np.sqrt(dx**2 + dy**2)

    left_steps = calc_step_lengths(
        left_strikes, right_strikes,
        df['smoothed_left_heel_x'].values, df['smoothed_left_heel_y'].values,
        df['smoothed_right_heel_x'].values, df['smoothed_right_heel_y'].values
    )
    right_steps = calc_step_lengths(
        right_strikes, left_strikes,
        df['smoothed_right_heel_x'].values, df['smoothed_right_heel_y'].values,
        df['smoothed_left_heel_x'].values, df['smoothed_left_heel_y'].values
    )

    # Calculate velocities for all steps
    all_strikes = np.sort(np.concatenate([left_strikes, right_strikes]))
    def calc_all_step_velocities(strikes, x, y, scale):
        if len(strikes) < 2:
            return np.array([])
        idx1 = strikes[:-1]
        idx2 = strikes[1:]
        dx = x[idx2] - x[idx1]
        dy = y[idx2] - y[idx1]
        dist = np.sqrt(dx**2 + dy**2) * scale[idx1]
        time_step = (idx2 - idx1) / frame_rate
        valid = time_step > 0
        return dist[valid] / time_step[valid]
    step_velocities = calc_all_step_velocities(all_strikes, df['left_heel_x'].values, df['smoothed_left_heel_y'].values, df['scale_factor'].values)

    # Calculate overall walking speed
    if len(all_strikes) < 2:
        speed = 0.0
    else:
        start_idx = all_strikes[0]
        end_idx = all_strikes[-1]
        x = df['left_heel_x'].values
        y = df['smoothed_left_heel_y'].values
        scale = df['scale_factor'].values
        dx = np.diff(x[start_idx:end_idx+1])
        dy = np.diff(y[start_idx:end_idx+1])
        distances = np.sqrt(dx**2 + dy**2) * scale[start_idx:end_idx]
        total_distance = np.sum(distances)
        time_elapsed = (end_idx - start_idx) / frame_rate
        speed = total_distance / time_elapsed if time_elapsed > 0 else 0.0

    # Calculate upper body metrics
    head_vert = df['smoothed_nose_y'] * scale_factors
    head_vert_amp = np.ptp(head_vert)
    head_lateral_amp = np.ptp(df['smoothed_nose_x'] * scale_factors)
    head_smoothness = np.sqrt(np.mean(np.gradient(np.gradient(head_vert))**2))
    shoulder_height_asym = np.mean(
        (df['smoothed_left_shoulder_y'] - df['smoothed_right_shoulder_y']) * scale_factors
    )
    shoulder_sway = np.std(
        (df['smoothed_left_shoulder_x'] + df['smoothed_right_shoulder_x']) / 2 * scale_factors
    )

    # Analyze arm movement using vectors
    def vectorized_arm_analysis(s_xy, e_xy, w_xy):
        v1 = e_xy - s_xy
        v2 = w_xy - e_xy
        angles = np.abs(np.degrees(
            np.arctan2(v2[:,1], v2[:,0]) - np.arctan2(v1[:,1], v1[:,0])
        ))
        return {
            'angle_mean': np.mean(angles),
            'swing_amp': np.ptp(e_xy[:,0] * scale_factors),
            'angle_var': np.std(angles)
        }

    left_arm = vectorized_arm_analysis(
        df[['smoothed_left_shoulder_x', 'smoothed_left_shoulder_y']].values,
        df[['smoothed_left_elbow_x', 'smoothed_left_elbow_y']].values,
        df[['smoothed_left_wrist_x', 'smoothed_left_wrist_y']].values
    )
    right_arm = vectorized_arm_analysis(
        df[['smoothed_right_shoulder_x', 'smoothed_right_shoulder_y']].values,
        df[['smoothed_right_elbow_x', 'smoothed_right_elbow_y']].values,
        df[['smoothed_right_wrist_x', 'smoothed_right_wrist_y']].values
    )

    # Calculate how different the left and right arm swings are
    arm_swing_asymmetry = (
        np.abs(left_arm['swing_amp'] - right_arm['swing_amp']) /
        ((left_arm['swing_amp'] + right_arm['swing_amp']) / 2)
    ) * 100

    # Calculate how much the torso leans
    torso_lean = np.mean(
        ((df['smoothed_left_shoulder_y'] + df['smoothed_right_shoulder_y']) / 2 -
         (df['smoothed_left_hip_y'] + df['smoothed_right_hip_y']) / 2) * scale_factors
    )

    # Calculate how well the torso and arms move together
    torso_arm_coord = pearsonr(
        df['smoothed_left_elbow_x'].values,
        df['smoothed_right_heel_x'].values
    )[0]

    # Helper functions for safe mean, coefficient of variation, and asymmetry
    safe_mean = lambda x: np.nan if len(x) == 0 else np.mean(x)
    cv = lambda x: np.std(x)/safe_mean(x)*100 if len(x) > 0 and safe_mean(x) else np.nan
    asym = lambda l, r: abs(np.log(safe_mean(l)/safe_mean(r)))*100 if len(l)*len(r) > 0 else np.nan

    # Return all the calculated metrics as a dictionary
    return {
        'DS_Asymmetry (%)': asym(left_ds, right_ds),
        'Left_DS_Mean (s)': safe_mean(left_ds),
        'Right_DS_Mean (s)': safe_mean(right_ds),
        'Left_StepTime_CV (%)': cv(left_step_times),
        'Left_StepTime_Mean (s)': safe_mean(left_step_times),
        'Right_StepTime_CV (%)': cv(right_step_times),
        'Right_StepTime_Mean (s)': safe_mean(right_step_times),
        'StepTime_Asymmetry (%)': asym(left_step_times, right_step_times),
        'Left_StepVelocity_CV (%)': cv(left_step_vel),
        'Left_StepVelocity_Mean (m/s)': safe_mean(left_step_vel),
        'Right_StepVelocity_CV (%)': cv(right_step_vel),
        'Right_StepVelocity_Mean (m/s)': safe_mean(right_step_vel),
        'StepVelocity_Asymmetry (%)': asym(left_step_vel, right_step_vel),
        'Left_Step_Length_CV (%)': cv(left_steps),
        'Left_Step_Length_Mean (m)': safe_mean(left_steps),
        'Right_Step_Length_CV (%)': cv(right_steps),
        'Right_Step_Length_Mean (m)': safe_mean(right_steps),
        'Step_Length_Asymmetry (%)': asym(left_steps, right_steps),
        'Left_Leg_Swing_CV (%)': cv(left_swing),
        'Left_Leg_Swing_Mean (s)': safe_mean(left_swing),
        'Right_Leg_Swing_CV (%)': cv(right_swing),
        'Right_Leg_Swing_Mean (s)': safe_mean(right_swing),
        'SwingTime_Asymmetry (%)': asym(left_swing, right_swing),
        'Speed (m/s)': speed,
        'Head_Vertical_Amplitude (m)': head_vert_amp,
        'Head_Lateral_Amplitude (m)': head_lateral_amp,
        'Head_Smoothness (RMS accel)': head_smoothness,
        'Shoulder_Height_Asymmetry (m)': shoulder_height_asym,
        'Shoulder_Sway_Amplitude (m)': shoulder_sway,
        'Left_Arm_Angle_Mean (deg)': left_arm['angle_mean'],
        'Left_Arm_Swing_Amplitude (m)': left_arm['swing_amp'],
        'Right_Arm_Angle_Mean (deg)': right_arm['angle_mean'],
        'Right_Arm_Swing_Amplitude (m)': right_arm['swing_amp'],
        'Arm_Swing_Asymmetry (%)': arm_swing_asymmetry,
        'Torso_Lean_Mean (m)': torso_lean,
        'Torso_Arm_Coordination (r)': torso_arm_coord,
        'Upper_Body_Stability_Index': np.mean([
            1/max(head_smoothness, 1e-6),
            1/max(shoulder_sway, 1e-6),
            max(0, torso_arm_coord)
        ])
    }

# This function runs full body analysis for all files in a folder and returns a summary DataFrame
def analyze_folder_fullbody(folder_path):
    all_metrics = []  # This will hold the results for each file
    for fname in os.listdir(folder_path):
        # Only look at CSV files that have 'day' in their name
        if fname.lower().endswith('.csv') and 'day' in fname.lower():
            day, label = extract_day_and_label(fname)  # Get the day and label from the filename
            df = pd.read_csv(os.path.join(folder_path, fname))  # Read the CSV file
            df['Days'] = day  # Add the day as a new column
            df['Label'] = label  # Add the label as a new column
            df['SourceFile'] = fname  # Save the filename for reference
            metrics = analyze_fullbody_gait(df)  # Calculate all gait metrics for this file
            metrics['Days'] = day  # Add the day to the metrics
            metrics['Label'] = label  # Add the label to the metrics
            metrics['SourceFile'] = fname  # Add the filename to the metrics
            all_metrics.append(metrics)  # Add the metrics to our list
    # Combine all the metrics into one big table and sort by day
    return pd.DataFrame(all_metrics).sort_values('Days')


# This function provides easy-to-understand explanations for each metric type
def get_metric_explanations():
    return {
        'Asymmetry': (
            "Asymmetry measures the difference between the left and right sides. "
            "Values closer to 0% indicate more balanced and symmetric movement. "
            "High values may suggest instability or compensatory movement patterns."
        ),
        'Mean': (
            "Mean is the average value of the measurement across steps or time. "
            "It reflects typical performance and is useful for comparing before/after changes."
        ),
        'CV': (
            "Coefficient of Variation (CV) measures variability relative to the mean. "
            "High CV means more irregularity or inconsistency in movement, "
            "while low CV indicates more stable and consistent gait."
        ),
        'Speed': (
            "Walking speed indicates how quickly the person moves. "
            "Faster speeds are typically associated with better mobility and confidence, "
            "while slower speeds may indicate caution, weakness, or pain."
        ),
        'Step Time': (
            "Step Time refers to the duration of each step. "
            "Balanced and consistent step times between both feet are indicators of healthy gait."
        ),
        'Double Support Time': (
            "Double Support Time is when both feet are on the ground simultaneously. "
            "Increased double support time may reflect unsteadiness or reduced confidence."
        ),
        'Swing Time': (
            "Swing Time is when one foot is off the ground moving forward. "
            "Balanced swing times indicate good motor control and balance."
        ),
        'Stride Length': (
            "Stride Length measures how far a person travels in a single stride. "
            "Longer strides are often seen in healthy, confident gait."
        ),
        'Step Velocity': (
            "Step Velocity is the speed of each step. "
            "Higher values reflect more dynamic walking, while low values may indicate slow or hesitant gait."
        ),
        'Upper Body Stability Index': (
            "This composite measure reflects overall coordination and steadiness of the upper body. "
            "Higher values suggest better control and reduced swaying."
        )
    }

def describe_metric_trend(trend_info, col, unit):
    """
    Returns a patient-friendly description of the trend for a metric,
    with correct logic for direction and clinical meaning.
    
    """
    if not trend_info:
        return "Trend information is unavailable."

    # These are metrics where an increase is generally positive
    positive_if_increase = [
        'Speed', 'Velocity', 'Stride Length', 'Step Length', 'Swing Amplitude', 'Upper Body Stability Index'
    ]
    # These are metrics where a decrease is generally positive
    positive_if_decrease = [
        'Asymmetry', 'CV', 'Variability', 'Double Support', 'Step Time', 'Swing Time'
    ]

    col_lower = col.lower()
    if any(key.lower() in col_lower for key in positive_if_increase):
        better_direction = 'increase'
        worse_direction = 'decrease'
    elif any(key.lower() in col_lower for key in positive_if_decrease):
        better_direction = 'decrease'
        worse_direction = 'increase'
    else:
        better_direction = None
        worse_direction = None

    start_val = trend_info.get('start', None)
    end_val = trend_info.get('end', None)
    abs_change = end_val - start_val if (start_val is not None and end_val is not None) else None

    # Format the change string
    if abs_change is not None:
        abs_change_str = f"{'increased' if abs_change > 0 else 'decreased'} by {abs(abs_change):.2f}{unit}"
    else:
        abs_change_str = ""

    # Format the percent change string (always positive, with direction)
    if trend_info['percent_change'] != 0:
        percent_change_str = f"({'increased' if trend_info['percent_change'] > 0 else 'decreased'} by {abs(trend_info['percent_change']):.2f}%)"
    else:
        percent_change_str = "(no change)"

    # Decide if this is a positive or negative trend
    if better_direction and abs_change is not None:
        if (better_direction == 'increase' and abs_change > 0) or (better_direction == 'decrease' and abs_change < 0):
            trend_quality = "This change is generally considered an improvement."
        elif (worse_direction == 'increase' and abs_change > 0) or (worse_direction == 'decrease' and abs_change < 0):
            trend_quality = "This change may indicate a worsening of gait."
        else:
            trend_quality = ""
    else:
        trend_quality = ""

    # Educational statement about the trend line
    trend_line_note = (
        "Trend lines show the overall change over time, not just the final value."
    )
    # If the last value is higher but the trend is negative, add a clarifying note
    if abs_change > 0 and trend_info['slope'] < 0:
        trend_line_note += (
            " Even though the last value is higher, the overall pattern slightly declined, likely due to lower values earlier."
        )
    elif abs_change < 0 and trend_info['slope'] > 0:
        trend_line_note += (
            " Even though the last value is lower, the overall pattern slightly increased, likely due to higher values earlier."
        )

    return (
        f"Over the measurement period, this metric {abs_change_str} {percent_change_str}. "
        f"{trend_quality} "
        f"R-squared: {trend_info['r_squared']:.2f}, P-value: {trend_info['p_value']:.3f}. "
        f"\n{trend_line_note}"
    )

def plot_metrics_dashboard(df, save_dir=None):
    metric_types = classify_metrics(df)
    explanations = get_metric_explanations()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for metric_type in metric_types.keys():
            os.makedirs(os.path.join(save_dir, metric_type), exist_ok=True)
    if "Label" in df.columns:
        label_change_indices = df.index[df["Label"].str.lower() == "after"].tolist()
        if label_change_indices:
            after_start_idx = label_change_indices[0]
            after_start_day = df.loc[after_start_idx, "Days"]
        else:
            after_start_day = None
    else:
        after_start_day = None
    for metric_type, metric_cols in metric_types.items():
        for col in metric_cols:
            if col not in df.columns:
                continue
            # Only plot if the column is numeric and has at least 2 valid values
            try:
                y = pd.to_numeric(df[col], errors='coerce')
                x = pd.to_numeric(df['Days'], errors='coerce')
                mask = (~y.isna()) & (~x.isna())
                if mask.sum() < 2:
                    continue  # Not enough data to plot or fit a trend
                y = y[mask]
                x = x[mask]
            except Exception:
                continue  # Skip columns that can't be converted to numeric

            fig, ax = plt.subplots(figsize=(12, 6))
            patient_labels = get_patient_friendly_labels()
            ax.plot(x, y, marker='o', label=patient_labels.get(col, col), linewidth=2)
            # Calculate trend and also pass start/end for absolute change
            trend = calculate_trend_analysis(y.values, x.values)
            if trend:
                trend['start'] = y.values[0]
                trend['end'] = y.values[-1]
                trend_line = trend['slope'] * x.values + trend['slope'] * -x.values[0] + y.values[0]
                ax.plot(x, trend_line, '--', alpha=0.5, label=f"{patient_labels.get(col, col)} Trend")
                # Draw a vertical arrow from first to last value (net change)
                ax.annotate(
                    '', xy=(x.values[-1], y.values[-1]), xytext=(x.values[0], y.values[0]),
                    arrowprops=dict(facecolor='green' if trend['end'] > trend['start'] else 'red', 
                                    edgecolor='black', 
                                    arrowstyle='->', 
                                    lw=2)
                )
                # Label the net change
                net_change = trend['end'] - trend['start']
                net_label = f"Net change: {'↑' if net_change > 0 else '↓'} {abs(net_change):.2f}{col.split('(')[-1].replace(')','') if '(' in col else ''}"
                ax.text(
                    x.values[-1], y.values[-1], net_label,
                    color='green' if net_change > 0 else 'red', fontsize=10, va='bottom', ha='left'
                )
            unit = col.split('(')[-1].replace(')', '') if '(' in col else ''
            ax.set_xlabel('Days', fontsize=12)
            ax.set_ylabel(f'{unit}', fontsize=12)
            title = f"{patient_labels.get(col, col)} Over Time"
            if trend:
                title += f"\nTrend: {trend['direction']} ({trend['trend_strength']})"
            ax.set_title(title, fontsize=14)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            if after_start_day is not None:
                ax.axvline(after_start_day, color='orange', linestyle='--', linewidth=2, label='After phase starts')
                ax.legend()

            # --- Add metric explanation and trend summary to each plot ---
            explanation_key = None
            col_lower = col.lower()
            if 'asymmetry' in col_lower:
                explanation_key = 'Asymmetry'
            elif 'cv' in col_lower or 'variability' in col_lower:
                explanation_key = 'CV'
            elif 'speed' in col_lower or 'velocity' in col_lower:
                explanation_key = 'Speed'
            elif 'step time' in col_lower:
                explanation_key = 'Step Time'
            elif 'double support' in col_lower or 'ds' in col_lower:
                explanation_key = 'Double Support Time'
            elif 'swing' in col_lower and 'time' in col_lower:
                explanation_key = 'Swing Time'
            elif 'stride' in col_lower or 'length' in col_lower:
                explanation_key = 'Stride Length'
            elif 'upper_body_stability_index' in col_lower:
                explanation_key = 'Upper Body Stability Index'
            elif 'mean' in col_lower:
                explanation_key = 'Mean'
            else:
                explanation_key = 'Mean'  # Default to mean if nothing else matches

            explanation = explanations.get(explanation_key, "")
            # Place the explanation and trend summary below the plot area
            ax.text(
                0.01, -0.22, explanation,
                transform=ax.transAxes, fontsize=10, verticalalignment='top', wrap=True
            )
            ax.text(
                0.01, -0.36, describe_metric_trend(trend, col, unit),
                transform=ax.transAxes, fontsize=9, verticalalignment='top', color='blue', wrap=True
            )

            plt.tight_layout(rect=[0, 0.18, 1, 1])  # Make room for the text below the plot
            if save_dir:
                fname = os.path.join(save_dir, metric_type, sanitize_filename(f"{col}.png"))
                plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close()


# ---- MAIN ----
if __name__ == "__main__":
    # Set the path to your folder with gait metric CSV files
    folder_path = r"C:\Before and After Data\Body Points"
    # Set the folder where you want to save the dashboard plots
    save_dir = r"C:\Before and After Data\gait_analysis_dashboard"
    # Run full body analysis for all files in the folder
    summary_df = analyze_folder_fullbody(folder_path)
    # Save the summary as a CSV file
    summary_df.to_csv(os.path.join(save_dir, "gait_analysis_summary.csv"), index=False)
    # Create the dashboard plots and save them, indicating when "Before" turns to "After"
    plot_metrics_dashboard(summary_df, save_dir=save_dir)
    print(f"Dashboard plots and summary saved to {save_dir}")