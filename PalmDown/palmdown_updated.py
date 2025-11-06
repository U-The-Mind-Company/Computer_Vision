import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

manual_ref_csv = "manual_landmarks_palm.csv"
output_root = "ProcessedCSVs_palm"

def load_reference_landmarks(csv_path):
    ref = {0: np.zeros((21, 2), dtype=np.float32), 1: np.zeros((21, 2), dtype=np.float32)}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hand_id = int(row["hand_id"])
            xs = np.array([float(row[f"x_{i}"]) for i in range(21)], dtype=np.float32)
            ys = np.array([float(row[f"y_{i}"]) for i in range(21)], dtype=np.float32)
            ref[hand_id] = np.column_stack((xs, ys))
    return ref


def process_video(video_path, output_csv, ref_landmarks):
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Could not open {video_path}")
        return video_path, False

    rows = []
    frame_idx = 0

    gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"GPU available: {gpu_available}")

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

            if gpu_available:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_rotated = cv2.cuda.rotate(gpu_frame, cv2.ROTATE_90_CLOCKWISE)
                gpu_rgb = cv2.cuda.cvtColor(gpu_rotated, cv2.COLOR_BGR2RGB)
                rgb = gpu_rgb.download()
            else:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w = rgb.shape[:2]
            result = hands.process(rgb)

            frame_landmarks = {0: ref_landmarks[0], 1: ref_landmarks[1]}

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    label = result.multi_handedness[hand_idx].classification[0].label.lower()
                    hand_id = 0 if label == "left" else 1
                    coords = np.array(
                        [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark],
                        dtype=np.float32
                    )
                    frame_landmarks[hand_id] = coords

            for hand_id in (0, 1):
                coords = frame_landmarks[hand_id]
                xs = np.round(coords[:, 0], 2).astype(str)
                ys = np.round(coords[:, 1], 2).astype(str)
                rows.append([frame_idx, hand_id, *xs, *ys])

            frame_idx += 1

    cap.release()

    header = ["frame", "hand_id"] + [f"x_{i}" for i in range(21)] + [f"y_{i}" for i in range(21)]
    with open(output_csv, "w", newline="") as f:
        csv.writer(f).writerows([header, *rows])

    return video_path, True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_gpu_opencv.py <root_folder>")
        sys.exit(1)

    root_folder = sys.argv[1]
    os.makedirs(output_root, exist_ok=True)
    ref_landmarks = load_reference_landmarks(manual_ref_csv)

    tasks = []
    for subdir, _, files in os.walk(root_folder):
        base = os.path.basename(subdir)
        if base in ("HAND_PALMHAND", "HAND_PALMDOWN"):
            relative_path = os.path.relpath(subdir, root_folder)
            subject_id = relative_path.split(os.sep)[0] if relative_path else "unknown_subject"
            clean_name = relative_path.replace(os.sep, "_")
            output_name = f"{subject_id}_{clean_name}.csv"
            output_csv = os.path.join(output_root, output_name)
            for file in files:
                if file.lower().endswith((".mp4", ".mov")):
                    tasks.append((os.path.join(subdir, file), output_csv))

    num_workers = min(max(1, os.cpu_count() - 1), 8)
    print(f"🧠 Found {len(tasks)} videos. Using {num_workers} workers.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_video, v, c, ref_landmarks) for v, c in tasks]
        for future in as_completed(futures):
            video_path, success = future.result()
            print(f"{'✅' if success else '❌'} {video_path}")

    print("🏁 All processing complete!")
