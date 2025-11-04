import os
import sys
import cv2
import pandas as pd
import numpy as np

# === Constants ===
MANUAL_LANDMARKS_FILE = "manual_landmarks.csv"
OUTPUT_FOLDER = "Processed_Upright_CSVs"
SUPPORTED_EXTS = (".mp4", ".mov")

# === Load reference landmarks ===
if not os.path.exists(MANUAL_LANDMARKS_FILE):
    print(f"Error: '{MANUAL_LANDMARKS_FILE}' not found.")
    sys.exit(1)

ref_landmarks = pd.read_csv(MANUAL_LANDMARKS_FILE)
print(f"[INFO] Loaded manual reference landmarks from {MANUAL_LANDMARKS_FILE}")

# === Helper to process a video ===
def process_video(video_path, output_csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    video_name = os.path.basename(video_path)
    results = []
    frame_idx = 0
    hand_id = 0  # constant or can be modified later

    print(f"[INFO] Processing {video_name} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # === (Optional) Preprocess ===
        # For upright adjustment, you could rotate or crop here if needed.
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Just reuse manual landmarks for now (constant across frames)
        for i in range(len(ref_landmarks)):
            x = ref_landmarks.iloc[i]['x']
            y = ref_landmarks.iloc[i]['y']
            results.append([frame_idx, hand_id, i, x, y])

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(results, columns=['frame', 'hand_id', 'landmark_id', 'x', 'y'])
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"[SAVED] {output_csv_path}")

# === Main ===
if len(sys.argv) < 2:
    print("Usage: python process_upright.py <root_folder>")
    sys.exit(1)

root_folder = sys.argv[1]
output_base = os.path.join(root_folder, OUTPUT_FOLDER)
os.makedirs(output_base, exist_ok=True)

for subdir, dirs, files in os.walk(root_folder):
    if os.path.basename(subdir).upper() == "HAND_UPRIGHT":
        subname = os.path.basename(os.path.dirname(subdir))
        #output_csv = os.path.join(output_base, f"{subname}_day_before_upright.csv")
        relative_path = os.path.relpath(subdir, root_folder)
        clean_name = relative_path.replace(os.sep, "_")
        output_csv = f"{clean_name}.csv"
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTS):
                video_path = os.path.join(subdir, file)
                process_video(video_path, output_csv)
