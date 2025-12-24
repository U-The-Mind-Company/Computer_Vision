#!/usr/bin/env python3
"""
Optimized + Refactored Stabilized Upright Processor

Behavior Preserved:
- Same CLI
- Same outputs
- Same stabilization algorithm
- Same tremor behavior
- Same multiprocessing behavior

Improvements:
- Faster stabilization (less numpy overhead)
- Reduced allocations in hot loops
- Eliminated OpenCV multiprocessing thread contention
- Cleaner structure & comments
- Safer I/O handling
"""

import os
import sys
import json
import math
from collections import OrderedDict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError("mediapipe is required. Install via `pip install mediapipe`") from e

# Prevent OpenCV from spinning worker threads per process.
cv2.setNumThreads(1)

mp_hands = mp.solutions.hands

# ---------------- CONFIG ----------------
MANUAL_LANDMARKS_FILE = r'C:\\Users\\Kevinf\\Documents\\GitHub\\Computer_Vision\\HandUpright\\manual_landmarks.csv'
PALMDOWN_CSV_DIR = r'C:\\Users\\Kevinf\\Documents\\GitHub\\Computer_Vision\\HandUpright'
OUTPUT_FOLDER = "Processed_Upright_CSVs_Stabilized"
SUPPORTED_EXTS = (".mp4", ".mov", ".m4v", ".avi")

ENABLE_STABILIZATION = True
SMOOTHING_RADIUS = 30
BORDER_CROP = 1.04

SKIP_SYNTHETIC = False
TREMOR_FREQ_HZ = 8.0
TREMOR_AMP_PX = 2.5
GAUSS_NOISE_SIGMA = 0.6
RANDOM_SEED = 42

MIN_SCALE = 0.3
MAX_SCALE = 2.0
EPS = 1e-9

if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)


# -------------------------------------------------------
# Stabilization Helpers  (performance tuned, behavior same)
# -------------------------------------------------------

def moving_average(curve, radius):
    window = 2 * radius + 1
    filt = np.ones(window, dtype=np.float32) / window
    pad = np.pad(curve, (radius, radius), mode="edge")
    smooth = np.convolve(pad, filt, mode="same")
    return smooth[radius:-radius]


def smooth_trajectory(traj, radius=SMOOTHING_RADIUS):
    out = traj.copy()
    for i in range(3):
        out[:, i] = moving_average(traj[:, i], radius)
    return out


def fix_border(frame, scale=BORDER_CROP):
    h, w = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(frame, matrix, (w, h))


def compute_stabilization_transforms(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 2:
        cap.release()
        return None

    ok, prev = cap.read()
    if not ok:
        cap.release()
        return None

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n - 1, 3), np.float32)

    good = cv2.goodFeaturesToTrack
    flow = cv2.calcOpticalFlowPyrLK
    est = cv2.estimateAffinePartial2D
    atan2 = np.arctan2

    for i in range(n - 1):
        pts_prev = good(prev_gray, 200, 0.01, 30)
        ok, curr = cap.read()
        if not ok:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        if pts_prev is None or len(pts_prev) < 3:
            prev_gray = curr_gray
            continue

        pts_curr, status, _ = flow(prev_gray, curr_gray, pts_prev, None)
        valid = status.reshape(-1) == 1

        if valid.sum() < 3:
            prev_gray = curr_gray
            continue

        m, _ = est(pts_prev[valid], pts_curr[valid])
        if m is not None:
            transforms[i, 0] = m[0, 2]
            transforms[i, 1] = m[1, 2]
            transforms[i, 2] = atan2(m[1, 0], m[0, 0])

        prev_gray = curr_gray

    cap.release()

    traj = np.cumsum(transforms, axis=0)
    smoothed = smooth_trajectory(traj)
    transforms_smooth = transforms + (smoothed - traj)

    mats = [np.eye(2, 3, dtype=np.float32)]
    cos = np.cos
    sin = np.sin
    for dx, dy, da in transforms_smooth:
        c, s = cos(da), sin(da)
        mats.append(np.array([[c, -s, dx], [s, c, dy]], dtype=np.float32))

    return mats


def apply_stabilization(frame, matrix):
    h, w = frame.shape[:2]
    stabilized = cv2.warpAffine(frame, matrix, (w, h))
    return fix_border(stabilized)


# ---------------------------------------------
# Windows path helpers
# ---------------------------------------------

def _win_long_path(p):
    if os.name != "nt" or not p:
        return p
    p = os.path.abspath(os.path.normpath(p))
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):
        return "\\\\?\\UNC" + p[1:]
    return "\\\\?\\" + p


def _sanitize(name):
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name


# ---------------------------------------------
# Manual Landmark Load (unchanged behavior)
# ---------------------------------------------

if not os.path.exists(MANUAL_LANDMARKS_FILE):
    print(f"[ERROR] Manual landmarks not found: {MANUAL_LANDMARKS_FILE}")
    sys.exit(1)

ref_landmarks = pd.read_csv(MANUAL_LANDMARKS_FILE)

x_cols = sorted([c for c in ref_landmarks.columns if c.startswith("x_")], key=lambda s: int(s.split("_")[1]))
y_cols = sorted([c for c in ref_landmarks.columns if c.startswith("y_")], key=lambda s: int(s.split("_")[1]))

if not x_cols or len(x_cols) != len(y_cols):
    raise ValueError("manual_landmarks.csv requires matching x_# and y_# columns")

landmark_ids = [int(c.split("_")[1]) for c in x_cols]
num_landmarks = len(landmark_ids)

if "hand_id" not in ref_landmarks.columns:
    raise ValueError("manual_landmarks.csv requires 'hand_id' column")

available_hands = sorted(ref_landmarks["hand_id"].unique())
print(f"[INFO] Manual hands found: {available_hands}")

hand_index_map = {hid: i for i, hid in enumerate(available_hands)}
manual = np.zeros((len(available_hands), num_landmarks, 2), float)

for _, row in ref_landmarks.iterrows():
    i = hand_index_map[int(row["hand_id"])]
    for j in landmark_ids:
        manual[i, j, 0] = row[f"x_{j}"]
        manual[i, j, 1] = row[f"y_{j}"]

phases = np.random.uniform(0, 2 * np.pi, manual.shape[:2])
amp_scales = np.random.uniform(0.7, 1.3, manual.shape[:2])


# ---------------------------------------------
# Tremor Fallback
# ---------------------------------------------

def tremor_coords(hid, t, scale=1.0):
    i = hand_index_map[hid]
    base = manual[i]
    ph = phases[i]
    amps = amp_scales[i]

    decay = max(0.1, math.exp(-t / 30.0))
    amp_j = 0.7 + 0.3 * math.sin(0.2 * math.pi * t)
    drift_x, drift_y = 0.12 * t, 0.08 * t

    f = TREMOR_FREQ_HZ * (1 + 0.05 * math.sin(0.1 * math.pi * t))
    trem_amp = TREMOR_AMP_PX * scale * amps * decay * amp_j

    xs = base[:, 0] + trem_amp * np.sin(2 * np.pi * f * t + ph) + np.random.normal(0, GAUSS_NOISE_SIGMA, num_landmarks) + drift_x
    ys = base[:, 1] + trem_amp * np.sin(2 * np.pi * f * t + ph + 0.7) + np.random.normal(0, GAUSS_NOISE_SIGMA, num_landmarks) + drift_y
    return xs.tolist(), ys.tolist()


# ---------------------------------------------
# MediaPipe Helpers
# ---------------------------------------------

def mp_to_px(lm_list, w, h):
    return [(lm.x * w, lm.y * h) for lm in lm_list.landmark]


def compute_rms(csv):
    if not os.path.exists(csv):
        return {}
    try:
        df = pd.read_csv(csv)
    except:
        return {}
    if df.empty:
        return {}

    result = {}
    for hid, g in df.groupby("hand_id"):
        X = g[[c for c in g.columns if c.startswith("x_")]].to_numpy()
        Y = g[[c for c in g.columns if c.startswith("y_")]].to_numpy()
        mx = X.mean(axis=1, keepdims=True)
        my = Y.mean(axis=1, keepdims=True)
        disp = np.sqrt((X - mx) ** 2 + (Y - my) ** 2).mean(axis=1)
        if len(disp):
            result[int(hid)] = float(np.sqrt(np.mean(disp**2)))
    return result


def derive_scale(rms):
    if rms is None or math.isnan(rms):
        return 1.0
    s = rms / (TREMOR_AMP_PX + EPS)
    return max(MIN_SCALE, min(MAX_SCALE, float(s)))


# ---------------------------------------------
# Worker
# ---------------------------------------------

def process_worker(args):
    video_path, out_csv, subject_id, hand_scales = args

    seed = (RANDOM_SEED or 0) + os.getpid()
    np.random.seed(seed)

    transforms = None
    if ENABLE_STABILIZATION:
        print(f"[INFO] Stabilizing: {os.path.basename(video_path)}")
        transforms = compute_stabilization_transforms(video_path)

    hands = mp_hands.Hands(False, 2, 0.5, 0.5)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return video_path, False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    frame = 0

    det_cnt = {h: 0 for h in available_hands}
    syn_cnt = {h: 0 for h in available_hands}

    while True:
        ok, img = cap.read()
        if not ok:
            break

        h, w = img.shape[:2]
        t = frame / fps

        if transforms and frame < len(transforms):
            img = apply_stabilization(img, transforms[frame])

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        detected = {}
        if result.multi_hand_landmarks:
            pts = [mp_to_px(lm, w, h) for lm in result.multi_hand_landmarks if len(lm.landmark) == num_landmarks]
            for idx, hid in enumerate(available_hands[:len(pts)]):
                detected[hid] = pts[idx]

        for hid in available_hands:
            if hid in detected:
                xs = [float(p[0]) for p in detected[hid]]
                ys = [float(p[1]) for p in detected[hid]]
                det_cnt[hid] += 1
            else:
                syn_cnt[hid] += 1
                if SKIP_SYNTHETIC:
                    continue
                xs, ys = tremor_coords(hid, t, scale=hand_scales.get(hid, 1.0))

            rows.append([frame, hid] + xs + ys)

        frame += 1

    cap.release()
    hands.close()

    cols = ["frame", "hand_id"] + [f"x_{i}" for i in range(num_landmarks)] + [f"y_{i}" for i in range(num_landmarks)]
    df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(_win_long_path(out_csv), index=False)

    meta = {
        "video": os.path.basename(video_path),
        "frames": frame,
        "stabilization": ENABLE_STABILIZATION,
        "hands": {
            str(h): {
                "scale": float(hand_scales.get(h, 1.0)),
                "detected": int(det_cnt[h]),
                "synthetic": int(syn_cnt[h])
            } for h in available_hands
        }
    }

    with open(out_csv + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] {os.path.basename(video_path)} -> {out_csv}")
    return video_path, True


# ---------------------------------------------
# Subject scale loader
# ---------------------------------------------

def find_palmdown(root, subject):
    found = []
    if os.path.isdir(PALMDOWN_CSV_DIR):
        for f in os.listdir(PALMDOWN_CSV_DIR):
            if f.startswith(subject + "_") and f.endswith(".csv"):
                found.append(os.path.join(PALMDOWN_CSV_DIR, f))
    return found


def compute_scales(root, subject):
    rms = []
    for f in find_palmdown(root, subject):
        m = compute_rms(f)
        if m:
            rms.extend(m.values())
    if not rms:
        return {h: 1.0 for h in available_hands}
    mean = float(np.mean(rms))
    return {h: derive_scale(mean) for h in available_hands}


# ---------------------------------------------
# MAIN
# ---------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_upright_stabilized_optimized.py <root_folder>")
        sys.exit(1)

    root = os.path.abspath(sys.argv[1])
    out_base = os.path.join(root, OUTPUT_FOLDER)
    os.makedirs(out_base, exist_ok=True)

    tasks = []
    for sub, _, files in os.walk(root):
        if os.path.basename(sub).upper() == "HAND_UPRIGHT":
            rel = os.path.relpath(sub, root)
            subject = rel.split(os.sep)[0]
            clean = rel.replace(os.sep, "_")
            for f in files:
                if not f.lower().endswith(SUPPORTED_EXTS):
                    continue
                vid = os.path.join(sub, f)
                stem = _sanitize(Path(f).stem)
                name = f"{subject}_{clean}_{stem}.csv"
                tasks.append((vid, os.path.join(out_base, name), subject))

    print(f"[INFO] Found {len(tasks)} videos.")

    args = []
    cache = {}

    for v, o, s in tasks:
        if s not in cache:
            cache[s] = compute_scales(root, s)
        args.append((v, o, s, cache[s]))

    workers = min(len(args), max(1, (os.cpu_count() or 2) // 2))
    print(f"[INFO] Using {workers} workers")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_worker, a): a[0] for a in args}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[ERROR] {futs[fut]}: {e}")
                results.append((futs[fut], False))

    ok = sum(1 for _, r in results if r)
    fail = len(results) - ok
    print(f"[SUMMARY] {ok} succeeded, {fail} failed.")


if __name__ == "__main__":
    main()
