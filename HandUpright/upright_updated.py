# Changes made: 1. Changed the code to incorporate the structure of the manual_landmarks.csv 2. Changed Output_root to Output_folder (Naming consistency) 3. Extended the Windows max_path limit so that the code can read the patient folder paths 

import os
import sys
import cv2
import pandas as pd
import numpy as np

# === Constants ===
MANUAL_LANDMARKS_FILE = r"C:\Users\aaish\Tremor\HandUpright\manual_landmarks.csv"
OUTPUT_FOLDER = "Processed_Upright_CSVs"
SUPPORTED_EXTS = (".mp4", ".mov")

# === Windows long-path helper (for CSV output paths) ===
def _win_long_path(p: str) -> str:
    """
    Return a Windows extended-length path (\\?\\...) if on Windows.
    Only used for writing CSVs to avoid MAX_PATH (260) issues.
    """
    if os.name != "nt":
        return p
    if not p:
        return p

    # Normalize and absolute
    p_abs = os.path.abspath(os.path.normpath(p))

    # Already extended?
    if p_abs.startswith("\\\\?\\"):
        return p_abs

    # UNC path: \\server\share\...  -> \\?\UNC\server\share\...
    if p_abs.startswith("\\\\"):
        return "\\\\?\\UNC" + p_abs[1:]

    # Local path: C:\... -> \\?\C:\...
    return "\\\\?\\" + p_abs

# === Load reference landmarks ===
if not os.path.exists(MANUAL_LANDMARKS_FILE):
    print(f"Error: '{MANUAL_LANDMARKS_FILE}' not found.")
    sys.exit(1)

ref_landmarks = pd.read_csv(MANUAL_LANDMARKS_FILE)
print(f"[INFO] Loaded manual reference landmarks from {MANUAL_LANDMARKS_FILE}")

# Detect landmark columns (x_0..x_n, y_0..y_n) and prepare a per-hand cache
x_cols = sorted(
    [c for c in ref_landmarks.columns if str(c).startswith("x_")],
    key=lambda s: int(str(s).split("_")[1])
)
y_cols = sorted(
    [c for c in ref_landmarks.columns if str(c).startswith("y_")],
    key=lambda s: int(str(s).split("_")[1])
)

if len(x_cols) == 0 or len(x_cols) != len(y_cols):
    raise ValueError(f"Bad landmark columns. x_cols={x_cols}, y_cols={y_cols}")

landmark_ids = [int(str(c).split("_")[1]) for c in x_cols]

# Build a lookup: hand_id -> list[(landmark_id, x, y)]
if "hand_id" not in ref_landmarks.columns:
    raise ValueError("Expected a 'hand_id' column in the manual landmarks CSV.")

landmarks_by_hand = {}
for _, row in ref_landmarks.iterrows():
    hid = int(row["hand_id"])
    coords = []
    for j in landmark_ids:
        coords.append((j, float(row[f"x_{j}"]), float(row[f"y_{j}"])))
    # If multiple rows per hand_id exist, the last row wins
    landmarks_by_hand[hid] = coords

# === Helper to process a video ===
def process_video(video_path, output_csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    video_name = os.path.basename(video_path)
    results = []
    frame_idx = 0

    # Choose which hand’s landmarks to use
    hand_id = 0  # change if you want hand 1, etc.

    if hand_id not in landmarks_by_hand:
        # Fallback to first available hand in the CSV
        first_hand = next(iter(landmarks_by_hand.keys()))
        print(f"[WARN] hand_id={hand_id} not in manual landmarks. Using hand_id={first_hand} instead.")
        hand_id = first_hand

    coords = landmarks_by_hand[hand_id]  # list of (landmark_id, x, y)

    print(f"[INFO] Processing {video_name} ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reuse the same manual landmarks for every frame
        for lm_id, x, y in coords:
            results.append([frame_idx, hand_id, lm_id, x, y])

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(results, columns=['frame', 'hand_id', 'landmark_id', 'x', 'y'])

    # Ensure directory exists (with long-path support)
    out_dir = os.path.dirname(output_csv_path)
    out_dir_lp = _win_long_path(out_dir)
    os.makedirs(out_dir_lp, exist_ok=True)

    # Write CSV using long-path
    output_csv_path_lp = _win_long_path(output_csv_path)
    df.to_csv(output_csv_path_lp, index=False)
    print(f"[SAVED] {output_csv_path}")

    return video_path, True

# === Main ===
if len(sys.argv) < 2:
    print("Usage: python process_upright.py <root_folder>")
    sys.exit(1)

root_folder = sys.argv[1]
output_base = os.path.join(root_folder, OUTPUT_FOLDER)
os.makedirs(_win_long_path(output_base), exist_ok=True)

for subdir, dirs, files in os.walk(root_folder):
    if os.path.basename(subdir).upper() == "HAND_UPRIGHT":
        subname = os.path.basename(os.path.dirname(subdir))
        relative_path = os.path.relpath(subdir, root_folder)

        # Extract subject ID (assumed to be first folder name under root)
        parts = relative_path.split(os.sep)
        subject_id = parts[0] if len(parts) > 0 else "unknown_subject"

        # Clean the relative path for filename
        clean_name = relative_path.replace(os.sep, "_")
        output_name = f"{subject_id}_{clean_name}.csv"
        output_csv = os.path.join(OUTPUT_FOLDER, output_name)

        for file in files:
            if file.lower().endswith(SUPPORTED_EXTS):
                video_path = os.path.join(subdir, file)
                process_video(video_path, output_csv)
