#!/usr/bin/env python3
"""
process_upright.py

- Uses manual_landmarks CSV (manual_landmarks.csv) with columns:
    hand_id, x_0..x_20, y_0..y_20
- Looks for palmdown session data (CSV output from your palmdown_updated.py) under ProcessedCSVs_palm/
  and computes per-hand RMS amplitude to derive a scale for upright fallback tremor.
- Processes videos in HAND_UPRIGHT folders:
  - Uses MediaPipe per-frame if available (mapped to manual hand_id by minimal distance)
  - Falls back to simulated tremor using manual landmarks but scaled by palmdown-derived factor
- Writes per-video CSV in exact wide format:
    frame,hand_id,x_0..x_N,y_0..y_N
- Writes companion JSON metadata: same filename + ".meta.json" with detection fractions, scales used.
"""

import os
import sys
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --- MediaPipe import (required) ---
try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError("mediapipe is required. Install via `pip install mediapipe`") from e

mp_hands = mp.solutions.hands

# === CONFIG ===
MANUAL_LANDMARKS_FILE = "manual_landmarks.csv"   # your manual landmarks for upright fallback
PALMDOWN_CSV_DIR = "ProcessedCSVs_palm"         # where palmdown_updated.py wrote CSVs
OUTPUT_FOLDER = "Processed_Upright_CSVs"
SUPPORTED_EXTS = (".mp4", ".mov", ".m4v", ".avi")

# Tremor / jitter params (base)
TREMOR_FREQ_HZ = 8.0         # base freq (Hz)
TREMOR_AMP_PX = 2.5          # base amplitude (pixels) — used as reference for deriving scale
GAUSS_NOISE_SIGMA = 0.6
RANDOM_SEED = 42

# Scale clipping
MIN_SCALE = 0.3
MAX_SCALE = 2.0
EPS = 1e-9

if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

# --- Helpers ---
def _win_long_path(p: str) -> str:
    """Return Windows extended path for writing if needed; harmless on other OS."""
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

# --- Load manual landmarks (used for fallback) ---
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

# Build manual lookup: hand_id -> OrderedDict(lm -> (x_base, y_base))
landmarks_by_hand = {}
for _, row in ref_landmarks.iterrows():
    hid = int(row["hand_id"])
    coords = OrderedDict()
    for j in landmark_ids:
        coords[j] = (float(row[f"x_{j}"]), float(row[f"y_{j}"]))
    landmarks_by_hand[hid] = coords

available_hands = sorted(list(landmarks_by_hand.keys()))
print(f"[INFO] Manual hands found: {available_hands}")

# Pre-generate phase and amplitude scales per (hand, landmark) for organic motion
phase_by_hand_landmark = {
    (hid, lm): np.random.uniform(0, 2 * math.pi)
    for hid in available_hands for lm in landmark_ids
}
amp_scale_by_hand_landmark = {
    (hid, lm): np.random.uniform(0.7, 1.3)
    for hid in available_hands for lm in landmark_ids
}

# --- Palmdown RMS -> derive per-hand scale ---
def compute_rms_from_palmdown_csv(palm_csv_path):
    """
    Read a palmdown CSV (same wide format as your palmdown script produces)
    and compute RMS amplitude per hand_id based on per-frame average landmark displacement.
    Returns: dict hand_id -> rms (or {} if file missing or empty)
    """
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
        # get x,y columns
        xcols = sorted([c for c in g.columns if c.startswith("x_")], key=lambda s: int(s.split("_")[1]))
        ycols = sorted([c for c in g.columns if c.startswith("y_")], key=lambda s: int(s.split("_")[1]))
        X = g[xcols].to_numpy(dtype=float)    # frames x landmarks
        Y = g[ycols].to_numpy(dtype=float)
        # per-frame, compute landmark displacements relative to that frame mean (so measuring spread)
        mean_frame = np.stack([X.mean(axis=1), Y.mean(axis=1)], axis=1)  # frames x 2 (unused shape)
        # displacement per frame per landmark
        disp = np.sqrt((X - X.mean(axis=1, keepdims=True))**2 + (Y - Y.mean(axis=1, keepdims=True))**2)
        mean_disp_per_frame = disp.mean(axis=1)  # frames
        # RMS over time
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

# --- MediaPipe helpers ---
def mp_landmarks_to_pixels(lm_list, image_w, image_h):
    pts = []
    for lm in lm_list.landmark:
        pts.append((lm.x * image_w, lm.y * image_h))
    return pts

def mean_landmark_distance(list_a, list_b):
    a = np.array(list_a, dtype=float)
    b = np.array(list_b, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.mean(np.linalg.norm(a - b, axis=1)))

def map_detected_to_manual(detected_lists, image_w, image_h):
    """
    Map each detected landmark list (mediapipe LandmarkList) to a manual hand_id by minimal average distance.
    Returns mapping detected_index -> hand_id
    """
    detected_pixel_lists = []
    for d in detected_lists:
        pts = mp_landmarks_to_pixels(d, image_w, image_h) if hasattr(d, "landmark") else d
        detected_pixel_lists.append(pts)

    mapping = {}
    used_hand_ids = set()
    for det_idx, det_pts in enumerate(detected_pixel_lists):
        best_h, best_dist = None, float("inf")
        for hid in available_hands:
            manual_pts = [landmarks_by_hand[hid][lm] for lm in landmark_ids]
            d = mean_landmark_distance(det_pts, manual_pts)
            if d < best_dist and hid not in used_hand_ids:
                best_dist = d
                best_h = hid
        if best_h is None:
            remaining = [h for h in available_hands if h not in used_hand_ids]
            best_h = remaining[0] if remaining else available_hands[0]
        mapping[det_idx] = best_h
        used_hand_ids.add(best_h)
    return mapping

# --- Tremor fallback generator with scale applied ---
def tremor_coords_for_hand(hid, t_seconds, scale=1.0):
    """Return lists xs, ys for given hand id at time t_seconds applying palmdown-derived scale."""
    base_coords = landmarks_by_hand[hid]
    xs, ys = [], []

    # amplitude decay + slow modulation (realistic)
    decay_factor = max(0.1, math.exp(-t_seconds / 30.0))
    amp_jitter = 0.7 + 0.3 * math.sin(0.1 * 2 * math.pi * t_seconds)
    drift_x = 0.12 * t_seconds
    drift_y = 0.08 * t_seconds

    for lm in landmark_ids:
        x_base, y_base = base_coords[lm]
        phase = phase_by_hand_landmark[(hid, lm)]
        amp_scale = amp_scale_by_hand_landmark[(hid, lm)]

        f = TREMOR_FREQ_HZ * (1 + 0.05 * math.sin(0.05 * 2 * math.pi * t_seconds))
        tremor_amp = TREMOR_AMP_PX * scale * amp_scale * decay_factor * amp_jitter

        jitter_x_sin = tremor_amp * math.sin(2 * math.pi * f * t_seconds + phase)
        jitter_y_sin = tremor_amp * math.sin(2 * math.pi * f * t_seconds + phase + 0.7)
        jitter_x_noise = np.random.normal(scale=GAUSS_NOISE_SIGMA)
        jitter_y_noise = np.random.normal(scale=GAUSS_NOISE_SIGMA)

        x_final = float(x_base + jitter_x_sin + jitter_x_noise + drift_x)
        y_final = float(y_base + jitter_y_sin + jitter_y_noise + drift_y)

        xs.append(x_final)
        ys.append(y_final)

    return xs, ys

# --- Main per-video processing (Upright) ---
def process_upright_video(video_path, output_csv_path, hand_scales):
    """
    hand_scales: dict hand_id -> scale (float). If missing, default scale=1.0
    Writes CSV with columns: frame,hand_id,x_0..x_N,y_0..y_N
    Also writes a companion metadata JSON with detection fractions and applied scales.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
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

    # metadata counters
    detected_count = {hid: 0 for hid in available_hands}
    total_frames = 0

    print(f"[INFO] Processing upright video: {os.path.basename(video_path)} (fps={fps:.2f})")

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            t = frame_idx / fps

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            detected_lists = []
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    detected_lists.append(lm)

            det_to_hand = {}
            if detected_lists:
                det_to_hand = map_detected_to_manual(detected_lists, w, h)

            # build reverse mapping: hand_id -> detected pts (if present)
            handid_to_detected = {}
            for det_idx, hid in det_to_hand.items():
                # convert to pixels
                pts = mp_landmarks_to_pixels(detected_lists[det_idx], w, h)
                if len(pts) == num_landmarks:
                    handid_to_detected[hid] = pts

            # for each expected manual hand_id, emit a row (detected or fallback)
            for hid in available_hands:
                total_frames += 1
                if hid in handid_to_detected:
                    pts = handid_to_detected[hid]
                    xs = [float(p[0]) for p in pts]
                    ys = [float(p[1]) for p in pts]
                    detected_count[hid] += 1
                else:
                    scale = hand_scales.get(hid, 1.0)
                    xs, ys = tremor_coords_for_hand(hid, t, scale=scale)

                row = [frame_idx, hid] + xs + ys
                rows.append(row)

            frame_idx += 1

    cap.release()

    # write CSV (exact requested format)
    columns = ["frame", "hand_id"] + [f"x_{i}" for i in landmark_ids] + [f"y_{i}" for i in landmark_ids]
    df = pd.DataFrame(rows, columns=columns)

    out_dir = os.path.dirname(output_csv_path)
    os.makedirs(_win_long_path(out_dir), exist_ok=True)
    df.to_csv(_win_long_path(output_csv_path), index=False)

    # write companion metadata JSON
    meta = {
        "video": os.path.basename(video_path),
        "frames": frame_idx,
        "hands": {}
    }
    for hid in available_hands:
        det_frac = detected_count[hid] / max(1, frame_idx)
        meta["hands"][str(hid)] = {
            "palmdown_scale": float(hand_scales.get(hid, 1.0)),
            "detected_frames": int(detected_count[hid]),
            "frame_count": int(frame_idx),
            "detection_fraction": float(det_frac)
        }

    meta_path = output_csv_path + ".meta.json"
    with open(_win_long_path(meta_path), "w") as jf:
        json.dump(meta, jf, indent=2)

    print(f"[SAVED] {output_csv_path}  (rows={len(df)}), meta -> {meta_path}")
    return (video_path, True)

# --- Top-level orchestration ---
def find_palmdown_csv_for_subject(root_folder, subject_id):
    """
    Try to locate palmdown CSV(s) for this subject in PALMDOWN_CSV_DIR or under the root folder.
    Returns a list of CSV paths.
    """
    found = []
    # First: check PALMDOWN_CSV_DIR for files starting with subject_id_
    if os.path.isdir(PALMDOWN_CSV_DIR):
        for fname in os.listdir(PALMDOWN_CSV_DIR):
            if fname.startswith(subject_id + "_") and fname.lower().endswith(".csv"):
                found.append(os.path.join(PALMDOWN_CSV_DIR, fname))
    # Second: search under root for HAND_PALMDOWN folders and csvs inside
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
    """
    For a subject, find palmdown CSV(s), compute RMS per hand, derive scale factors.
    Returns dict hand_id -> scale.
    """
    scales = {}
    csvs = find_palmdown_csv_for_subject(root_folder, subject_id)
    if not csvs:
        # No palmdown CSVs — return default scale=1.0 for all hands
        for hid in available_hands:
            scales[hid] = 1.0
        return scales

    # aggregate RMS values across all palmdown CSVs (if multiple) by mean
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
        print("Usage: python process_upright.py <root_folder>")
        sys.exit(1)

    root_folder = os.path.abspath(sys.argv[1])
    output_base = os.path.join(root_folder, OUTPUT_FOLDER)
    os.makedirs(_win_long_path(output_base), exist_ok=True)

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

    for video_path, output_csv, subject_id in tasks:
        # compute subject-level palmdown scales (mean across palmdown CSVs)
        hand_scales = compute_subject_scales(root_folder, subject_id)
        print(f"[INFO] subject {subject_id} scales: {hand_scales}")
        try:
            process_upright_video(video_path, output_csv, hand_scales)
        except Exception as e:
            print(f"[ERROR] processing {video_path}: {e}")

if __name__ == "__main__":
    main()
