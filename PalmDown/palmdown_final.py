import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import sys

# === CONFIGURATION ===
if len(sys.argv) < 2:
    print("Usage: python process_with_ref_landmarks_v3.py <root_folder>")
    sys.exit(1)

root_folder = sys.argv[1]
manual_ref_csv = "manual_landmarks_palm.csv"  # your manual reference file
output_root = "ProcessedCSVs_palm"
os.makedirs(output_root, exist_ok=True)

# === LOAD MANUAL REFERENCE LANDMARKS ===
def load_reference_landmarks(csv_path):
    ref = {0: [], 1: []}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hand_id = int(row["hand_id"])
            xs = [float(row[f"x_{i}"]) for i in range(21)]
            ys = [float(row[f"y_{i}"]) for i in range(21)]
            ref[hand_id] = list(zip(xs, ys))
    return ref

ref_landmarks = load_reference_landmarks(manual_ref_csv)

# === MEDIAPIPE INITIALIZATION ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Could not open {video_path}")
        return

    frame_idx = 0
    all_rows = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Rotate to upright
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Default (reference) landmarks
            frame_landmarks = {0: ref_landmarks[0], 1: ref_landmarks[1]}

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    label = result.multi_handedness[hand_idx].classification[0].label
                    hand_id = 0 if label.lower() == "left" else 1

                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.append((lm.x * w, lm.y * h))
                    frame_landmarks[hand_id] = coords

            # Save one row per hand per frame
            for hand_id in [0, 1]:
                xs = [f"{p[0]:.2f}" for p in frame_landmarks[hand_id]]
                ys = [f"{p[1]:.2f}" for p in frame_landmarks[hand_id]]
                row = [frame_idx, hand_id] + xs + ys
                all_rows.append(row)

            frame_idx += 1

    cap.release()

    # === Save as CSV ===
    header = ["frame", "hand_id"] + [f"x_{i}" for i in range(21)] + [f"y_{i}" for i in range(21)]
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    print(f"✅ Saved: {output_csv}")


# === MAIN: Traverse all subfolders ===
for subdir, dirs, files in os.walk(root_folder):
    if os.path.basename(subdir) == "HAND_PALMHAND" or os.path.basename(subdir) == "HAND_PALMDOWN":
        # Build the hierarchical name based on parent folders
        relative_path = os.path.relpath(subdir, root_folder)
        clean_name = relative_path.replace(os.sep, "_")
        output_name = f"{clean_name}.csv"
        output_csv = os.path.join(output_root, output_name)

        # Process all videos inside HAND_UPRIGHT
        for file in files:
            if file.lower().endswith((".mp4", ".mov")):
                video_path = os.path.join(subdir, file)
                print(f"🎥 Processing {video_path} ...")
                process_video(video_path, output_csv)
