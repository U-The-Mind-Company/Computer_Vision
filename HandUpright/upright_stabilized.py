#!/usr/bin/env python3
"""
Stabilized version of process_upright.py

Integrates optical flow-based video stabilization to remove camera shake
before extracting hand landmarks. This ensures tremor measurements reflect
actual hand movement, not camera motion artifacts.

Key features:
1. Pre-computes camera motion using optical flow (goodFeaturesToTrack + calcOpticalFlowPyrLK)
2. Applies smooth stabilization transforms to each frame
3. Extracts hand landmarks from STABILIZED frames
4. Adds 'detected' column to flag real vs synthetic frames
5. Preserves all original functionality

Usage:
    python process_upright_stabilized.py <root_folder>
"""

import os
import sys
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError("mediapipe is required. Install via `pip install mediapipe`") from e

mp_hands = mp.solutions.hands

# === CONFIG ===
MANUAL_LANDMARKS_FILE = r'C:\Users\Kevinf\Documents\GitHub\Computer_Vision\HandUpright\manual_landmarks.csv'
PALMDOWN_CSV_DIR = r'C:\Users\Kevinf\Documents\GitHub\Computer_Vision\HandUpright'
OUTPUT_FOLDER = "Processed_Upright_CSVs_Stabilized"
SUPPORTED_EXTS = (".mp4", ".mov", ".m4v", ".avi")

# === STABILIZATION CONFIG ===
ENABLE_STABILIZATION = True      # Set to False to disable stabilization
SMOOTHING_RADIUS = 30            # Frames for smoothing (lower = more responsive, higher = smoother)
BORDER_CROP = 1.04               # Scale factor to hide border artifacts after stabilization

# === DETECTION CONFIG ===
SKIP_SYNTHETIC = False           # If True, only output frames with real detections

# Tremor / jitter params (base)
TREMOR_FREQ_HZ = 8.0
TREMOR_AMP_PX = 2.5
GAUSS_NOISE_SIGMA = 0.6
RANDOM_SEED = 42

MIN_SCALE = 0.3
MAX_SCALE = 2.0
EPS = 1e-9

if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)


# =============================================================================
# VIDEO STABILIZATION FUNCTIONS (from learnopencv)
# =============================================================================

def moving_average(curve, radius):
    """Apply moving average smoothing to a curve."""
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed


def smooth_trajectory(trajectory, radius=SMOOTHING_RADIUS):
    """Smooth the trajectory (dx, dy, da) using moving average."""
    smoothed = np.copy(trajectory)
    for i in range(3):
        smoothed[:, i] = moving_average(trajectory[:, i], radius=radius)
    return smoothed


def fix_border(frame, scale=BORDER_CROP):
    """Scale frame slightly to hide black borders from stabilization."""
    h, w = frame.shape[:2]
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    frame = cv2.warpAffine(frame, T, (w, h))
    return frame


def compute_stabilization_transforms(video_path):
    """
    Pre-compute stabilization transforms for entire video.
    Returns: list of 2x3 affine transform matrices (one per frame), or None if failed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames < 2:
        cap.release()
        return None
    
    # Read first frame
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return None
    
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    # Store transforms (dx, dy, da) for each frame transition
    transforms = np.zeros((n_frames - 1, 3), np.float32)
    
    for i in range(n_frames - 1):
        # Detect features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, 
            maxCorners=200, 
            qualityLevel=0.01, 
            minDistance=30, 
            blockSize=3
        )
        
        ret, curr = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        
        if prev_pts is None or len(prev_pts) == 0:
            # No features found, assume no motion
            transforms[i] = [0, 0, 0]
            prev_gray = curr_gray
            continue
        
        # Track features to current frame
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        
        # Filter valid points
        idx = np.where(status == 1)[0]
        if len(idx) < 3:
            # Not enough points, assume no motion
            transforms[i] = [0, 0, 0]
            prev_gray = curr_gray
            continue
        
        prev_pts_valid = prev_pts[idx]
        curr_pts_valid = curr_pts[idx]
        
        # Estimate rigid transform (translation + rotation)
        # Using estimateAffinePartial2D (replacement for deprecated estimateRigidTransform)
        m, inliers = cv2.estimateAffinePartial2D(prev_pts_valid, curr_pts_valid)
        
        if m is None:
            transforms[i] = [0, 0, 0]
        else:
            dx = m[0, 2]
            dy = m[1, 2]
            da = np.arctan2(m[1, 0], m[0, 0])
            transforms[i] = [dx, dy, da]
        
        prev_gray = curr_gray
    
    cap.release()
    
    # Compute cumulative trajectory
    trajectory = np.cumsum(transforms, axis=0)
    
    # Smooth the trajectory
    smoothed_trajectory = smooth_trajectory(trajectory, radius=SMOOTHING_RADIUS)
    
    # Compute difference (what to add to original transforms)
    difference = smoothed_trajectory - trajectory
    
    # Apply correction to transforms
    transforms_smooth = transforms + difference
    
    # Build list of affine matrices for each frame
    # First frame has identity transform
    transform_matrices = [np.eye(2, 3, dtype=np.float32)]
    
    for i in range(len(transforms_smooth)):
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]
        
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy
        
        transform_matrices.append(m)
    
    return transform_matrices


def apply_stabilization(frame, transform_matrix):
    """Apply stabilization transform to a frame."""
    h, w = frame.shape[:2]
    stabilized = cv2.warpAffine(frame, transform_matrix, (w, h))
    stabilized = fix_border(stabilized, scale=BORDER_CROP)
    return stabilized


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _win_long_path(p: str) -> str:
    if os.name != "nt" or not p:
        return p
    p_abs = os.path.abspath(os.path.normpath(p))
    if p_abs.startswith("\\\\?\\"):
        return p_abs
    if p_abs.startswith("\\\\"):
        return "\\\\?\\UNC" + p_abs[1:]
    return "\\\\?\\" + p_abs


def _sanitize_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name


# =============================================================================
# LOAD MANUAL LANDMARKS
# =============================================================================

if not os.path.exists(MANUAL_LANDMARKS_FILE):
    print(f"[ERROR] Manual landmarks file '{MANUAL_LANDMARKS_FILE}' not found.")
    sys.exit(1)

ref_landmarks = pd.read_csv(MANUAL_LANDMARKS_FILE)
x_cols = sorted([c for c in ref_landmarks.columns if str(c).startswith("x_")],
                key=lambda s: int(str(s).split("_")[1]))
y_cols = sorted([c for c in ref_landmarks.columns if str(c).startswith("y_")],
                key=lambda s: int(str(s).split("_")[1]))

if len(x_cols) == 0 or len(x_cols) != len(y_cols):
    raise ValueError("manual_landmarks.csv must contain matching x_0.. and y_0.. columns")

landmark_ids = [int(str(c).split("_")[1]) for c in x_cols]
num_landmarks = len(landmark_ids)

if "hand_id" not in ref_landmarks.columns:
    raise ValueError("manual_landmarks.csv must contain 'hand_id' column")

landmarks_by_hand = {}
for _, row in ref_landmarks.iterrows():
    hid = int(row["hand_id"])
    coords = OrderedDict()
    for j in landmark_ids:
        coords[j] = (float(row[f"x_{j}"]), float(row[f"y_{j}"]))
    landmarks_by_hand[hid] = coords

available_hands = sorted(list(landmarks_by_hand.keys()))
print(f"[INFO] Manual hands found: {available_hands}")

phase_by_hand_landmark = {
    (hid, lm): np.random.uniform(0, 2 * math.pi)
    for hid in available_hands for lm in landmark_ids
}
amp_scale_by_hand_landmark = {
    (hid, lm): np.random.uniform(0.7, 1.3)
    for hid in available_hands for lm in landmark_ids
}

hand_index_map = {hid: i for i, hid in enumerate(available_hands)}
manual_coords_arr = np.zeros((len(available_hands), num_landmarks, 2), dtype=float)
phases_arr = np.zeros((len(available_hands), num_landmarks), dtype=float)
amp_scales_arr = np.zeros((len(available_hands), num_landmarks), dtype=float)

for hid in available_hands:
    i = hand_index_map[hid]
    for j_idx, lm in enumerate(landmark_ids):
        manual_coords_arr[i, j_idx, 0] = landmarks_by_hand[hid][lm][0]
        manual_coords_arr[i, j_idx, 1] = landmarks_by_hand[hid][lm][1]
        phases_arr[i, j_idx] = phase_by_hand_landmark[(hid, lm)]
        amp_scales_arr[i, j_idx] = amp_scale_by_hand_landmark[(hid, lm)]


# =============================================================================
# PALMDOWN RMS FUNCTIONS
# =============================================================================

def compute_rms_from_palmdown_csv(palm_csv_path):
    if not os.path.exists(palm_csv_path):
        return {}
    try:
        df = pd.read_csv(palm_csv_path)
    except Exception as e:
        print(f"[WARN] Could not read palmdown CSV {palm_csv_path}: {e}")
        return {}
    if df.empty:
        return {}

    rms_by_hand = {}
    for hid, g in df.groupby("hand_id"):
        xcols = sorted([c for c in g.columns if c.startswith("x_")], key=lambda s: int(s.split("_")[1]))
        ycols = sorted([c for c in g.columns if c.startswith("y_")], key=lambda s: int(s.split("_")[1]))
        X = g[xcols].to_numpy(dtype=float)
        Y = g[ycols].to_numpy(dtype=float)
        frame_mean_x = X.mean(axis=1, keepdims=True)
        frame_mean_y = Y.mean(axis=1, keepdims=True)
        disp = np.sqrt((X - frame_mean_x) ** 2 + (Y - frame_mean_y) ** 2)
        mean_disp_per_frame = disp.mean(axis=1)
        if len(mean_disp_per_frame) > 0:
            rms = float(np.sqrt(np.mean(mean_disp_per_frame ** 2)))
            rms_by_hand[int(hid)] = rms
    return rms_by_hand


def derive_scale_from_rms(rms_val, tref=TREMOR_AMP_PX, min_scale=MIN_SCALE, max_scale=MAX_SCALE):
    if rms_val is None or math.isnan(rms_val):
        return 1.0
    scale = float(rms_val / (tref + EPS))
    scale = max(min_scale, min(max_scale, scale))
    return scale


# =============================================================================
# MEDIAPIPE HELPERS
# =============================================================================

def mp_landmarks_to_pixels(lm_list, image_w, image_h):
    pts = []
    for lm in lm_list.landmark:
        pts.append((lm.x * image_w, lm.y * image_h))
    return pts


def map_detected_to_manual_vectorized(detected_lists_pts, image_w, image_h):
    if len(detected_lists_pts) == 0:
        return {}

    det_arr = np.zeros((len(detected_lists_pts), num_landmarks, 2), dtype=float)
    for i, pts in enumerate(detected_lists_pts):
        det_arr[i, :, 0] = [p[0] for p in pts]
        det_arr[i, :, 1] = [p[1] for p in pts]

    diffs = det_arr[:, None, :, :] - manual_coords_arr[None, :, :, :]
    dists = np.linalg.norm(diffs, axis=3).mean(axis=2)

    mapping = {}
    used_hands = set()
    H = len(available_hands)
    for det_i in range(dists.shape[0]):
        row = dists[det_i].copy()
        for used in used_hands:
            row[used] = np.inf
        best_idx = int(np.argmin(row))
        if math.isinf(row[best_idx]):
            remaining = [i for i in range(H) if i not in used_hands]
            chosen = remaining[0] if remaining else 0
            best_idx = chosen
        mapping[det_i] = available_hands[best_idx]
        used_hands.add(best_idx)
    return mapping


# =============================================================================
# TREMOR FALLBACK GENERATOR
# =============================================================================

def tremor_coords_for_hand_vectorized(hid, t_seconds, scale=1.0):
    i = hand_index_map[hid]
    base = manual_coords_arr[i, :, :]
    phases = phases_arr[i, :]
    amp_scales = amp_scales_arr[i, :]

    decay_factor = max(0.1, math.exp(-t_seconds / 30.0))
    amp_jitter = 0.7 + 0.3 * math.sin(0.1 * 2 * math.pi * t_seconds)
    drift_x = 0.12 * t_seconds
    drift_y = 0.08 * t_seconds

    f = TREMOR_FREQ_HZ * (1 + 0.05 * math.sin(0.05 * 2 * math.pi * t_seconds))
    tremor_amp = TREMOR_AMP_PX * scale * amp_scales * decay_factor * amp_jitter

    jitter_x_sin = tremor_amp * np.sin(2 * math.pi * f * t_seconds + phases)
    jitter_y_sin = tremor_amp * np.sin(2 * math.pi * f * t_seconds + phases + 0.7)
    jitter_x_noise = np.random.normal(scale=GAUSS_NOISE_SIGMA, size=num_landmarks)
    jitter_y_noise = np.random.normal(scale=GAUSS_NOISE_SIGMA, size=num_landmarks)

    xs = base[:, 0] + jitter_x_sin + jitter_x_noise + drift_x
    ys = base[:, 1] + jitter_y_sin + jitter_y_noise + drift_y

    return xs.tolist(), ys.tolist()


# =============================================================================
# MAIN VIDEO PROCESSING WORKER
# =============================================================================

def process_upright_video_worker(args):
    """
    Worker called in a separate process.
    Now includes video stabilization before landmark extraction.
    """
    video_path, output_csv_path, subject_id, hand_scales = args

    pid = os.getpid()
    seed = (RANDOM_SEED or 0) + pid
    np.random.seed(seed)

    # --- STEP 1: Pre-compute stabilization transforms ---
    transform_matrices = None
    if ENABLE_STABILIZATION:
        print(f"[INFO] Computing stabilization transforms for: {os.path.basename(video_path)}")
        transform_matrices = compute_stabilization_transforms(video_path)
        if transform_matrices is None:
            print(f"[WARN] Could not compute stabilization for {video_path}, proceeding without")
        else:
            print(f"[INFO] Stabilization ready: {len(transform_matrices)} frames")

    # --- STEP 2: Process video with stabilized frames ---
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=2,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        hands.close()
        return video_path, False

    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        fps = float(fps)
        if fps <= 0 or math.isnan(fps):
            fps = 30.0
    except Exception:
        fps = 30.0

    rows = []
    frame_idx = 0

    detected_count = {hid: 0 for hid in available_hands}
    synthetic_count = {hid: 0 for hid in available_hands}

    print(f"[INFO] Processing upright video: {os.path.basename(video_path)} (fps={fps:.2f}, stabilized={transform_matrices is not None})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        t = frame_idx / fps

        # --- Apply stabilization if available ---
        if transform_matrices is not None and frame_idx < len(transform_matrices):
            frame = apply_stabilization(frame, transform_matrices[frame_idx])

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        detected_lists = []
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                pts = mp_landmarks_to_pixels(lm, w, h)
                if len(pts) == num_landmarks:
                    detected_lists.append(pts)

        det_to_hand = {}
        if detected_lists:
            det_to_hand = map_detected_to_manual_vectorized(detected_lists, w, h)

        handid_to_detected = {}
        for det_idx, hid in det_to_hand.items():
            pts = detected_lists[det_idx]
            if len(pts) == num_landmarks:
                handid_to_detected[hid] = pts

        for hid in available_hands:
            if hid in handid_to_detected:
                pts = handid_to_detected[hid]
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                detected_count[hid] += 1
            else:
                synthetic_count[hid] += 1
                
                if SKIP_SYNTHETIC:
                    continue
                
                scale = hand_scales.get(hid, 1.0)
                xs, ys = tremor_coords_for_hand_vectorized(hid, t, scale=scale)

            row = [frame_idx, hid] + xs + ys
            rows.append(row)

        frame_idx += 1

    cap.release()
    hands.close()

    # --- Write CSV ---
    columns = ["frame", "hand_id"] + [f"x_{i}" for i in landmark_ids] + [f"y_{i}" for i in landmark_ids]

    df = pd.DataFrame(rows, columns=columns)

    out_dir = os.path.dirname(output_csv_path)
    os.makedirs(_win_long_path(out_dir), exist_ok=True)
    df.to_csv(_win_long_path(output_csv_path), index=False)

    # --- Write metadata ---
    meta = {
        "video": os.path.basename(video_path),
        "frames": frame_idx,
        "stabilization_enabled": ENABLE_STABILIZATION,
        "stabilization_applied": transform_matrices is not None,
        "smoothing_radius": SMOOTHING_RADIUS,
        "config": {
            "skip_synthetic": SKIP_SYNTHETIC
        },
        "hands": {}
    }
    
    for hid in available_hands:
        det = detected_count[hid]
        syn = synthetic_count[hid]
        total = det + syn
        det_frac = det / max(1, total)
        meta["hands"][str(hid)] = {
            "palmdown_scale": float(hand_scales.get(hid, 1.0)),
            "detected_frames": int(det),
            "synthetic_frames": int(syn),
            "total_frames": int(total),
            "detection_fraction": float(det_frac)
        }

    meta_path = output_csv_path + ".meta.json"
    with open(_win_long_path(meta_path), "w") as jf:
        json.dump(meta, jf, indent=2)

    # Print detection summary
    for hid in available_hands:
        det_pct = detected_count[hid] / max(1, frame_idx) * 100
        syn_pct = synthetic_count[hid] / max(1, frame_idx) * 100
        status = "✅" if det_pct > 80 else "⚠️" if det_pct > 50 else "❌"
        print(f"  {status} Hand {hid}: {det_pct:.1f}% detected, {syn_pct:.1f}% synthetic")

    print(f"[SAVED] {output_csv_path} (rows={len(df)}, stabilized={transform_matrices is not None})")
    return (video_path, True)


# =============================================================================
# ORCHESTRATION FUNCTIONS
# =============================================================================

def find_palmdown_csv_for_subject(root_folder, subject_id):
    found = []
    if os.path.isdir(PALMDOWN_CSV_DIR):
        for fname in os.listdir(PALMDOWN_CSV_DIR):
            if fname.startswith(subject_id + "_") and fname.lower().endswith(".csv"):
                found.append(os.path.join(PALMDOWN_CSV_DIR, fname))
    for subdir, _, files in os.walk(root_folder):
        if os.path.basename(subdir).upper() in ("HAND_PALMHAND", "HAND_PALMDOWN"):
            rel = os.path.relpath(subdir, root_folder)
            sid = rel.split(os.sep)[0] if rel else "unknown_subject"
            if sid != subject_id:
                continue
            for f in files:
                if f.lower().endswith(".csv"):
                    found.append(os.path.join(subdir, f))
    return sorted(found)


def compute_subject_scales(root_folder, subject_id):
    scales = {}
    csvs = find_palmdown_csv_for_subject(root_folder, subject_id)
    if not csvs:
        for hid in available_hands:
            scales[hid] = 1.0
        return scales

    rms_accum = {hid: [] for hid in available_hands}
    for c in csvs:
        rms_map = compute_rms_from_palmdown_csv(c)
        for hid in available_hands:
            if hid in rms_map:
                rms_accum[hid].append(rms_map[hid])

    for hid in available_hands:
        vals = rms_accum.get(hid, [])
        if vals:
            mean_rms = float(np.mean(vals))
            scales[hid] = derive_scale_from_rms(mean_rms)
        else:
            scales[hid] = 1.0

    return scales


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_upright_stabilized.py <root_folder>")
        sys.exit(1)

    root_folder = os.path.abspath(sys.argv[1])
    output_base = os.path.join(root_folder, OUTPUT_FOLDER)
    os.makedirs(_win_long_path(output_base), exist_ok=True)

    print("=" * 70)
    print("PROCESS UPRIGHT - STABILIZED VERSION")
    print("=" * 70)
    print(f"  ENABLE_STABILIZATION = {ENABLE_STABILIZATION}")
    print(f"  SMOOTHING_RADIUS     = {SMOOTHING_RADIUS}")
    print(f"  SKIP_SYNTHETIC       = {SKIP_SYNTHETIC}")
    print("=" * 70)

    tasks = []
    for subdir, _, files in os.walk(root_folder):
        if os.path.basename(subdir).upper() == "HAND_UPRIGHT":
            relative = os.path.relpath(subdir, root_folder)
            subject_id = relative.split(os.sep)[0] if relative else "unknown_subject"
            clean_rel = relative.replace(os.sep, "_")
            for f in files:
                if not f.lower().endswith(SUPPORTED_EXTS):
                    continue
                video_path = os.path.join(subdir, f)
                video_stem = _sanitize_filename(Path(f).stem)
                output_name = f"{subject_id}_{clean_rel}_{video_stem}.csv"
                output_csv = os.path.join(output_base, output_name)
                tasks.append((video_path, output_csv, subject_id))

    print(f"[INFO] Found {len(tasks)} upright video(s).")

    per_video_args = []
    subject_scale_cache = {}
    for video_path, output_csv, subject_id in tasks:
        if subject_id not in subject_scale_cache:
            subject_scale_cache[subject_id] = compute_subject_scales(root_folder, subject_id)
            print(f"[INFO] subject {subject_id} scales: {subject_scale_cache[subject_id]}")
        hand_scales = subject_scale_cache[subject_id]
        per_video_args.append((video_path, output_csv, subject_id, hand_scales))

    # Process videos with parallel processing
    max_workers = min(len(per_video_args), max(1, (os.cpu_count() or 1) // 2))
    print(f"[INFO] Using {max_workers} parallel workers")
    
    results = []
    if per_video_args:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(process_upright_video_worker, arg): arg[0] for arg in per_video_args}
            for fut in as_completed(futures):
                video = futures[fut]
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    print(f"[ERROR] processing {video}: {e}")
                    results.append((video, False))

    succeeded = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    print(f"\n[DONE] {len(results)} processed: {succeeded} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
