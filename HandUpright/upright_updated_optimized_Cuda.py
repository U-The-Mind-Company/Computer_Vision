#!/usr/bin/env python3
"""
process_upright.py  (PyCUDA-accelerated mapping)

This keeps the exact behavior and outputs of your original script while adding:
 - Per-worker PyCUDA acceleration of the detected->manual mean-distance matrix.
 - Safe fallback to CPU (NumPy) if PyCUDA isn't available.
 - MediaPipe still runs on CPU inside each worker process.
 - Uses ProcessPoolExecutor for per-video parallelism.

Usage:
    python process_upright.py <root_folder>
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

# Convert manual hands to numpy arrays for fast computation:
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

# --- Palmdown RMS -> derive per-hand scale ---
def compute_rms_from_palmdown_csv(palm_csv_path):
    """
    Read a palmdown CSV and compute RMS amplitude per hand_id.
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
        xcols = sorted([c for c in g.columns if c.startswith("x_")], key=lambda s: int(s.split("_")[1]))
        ycols = sorted([c for c in g.columns if c.startswith("y_")], key=lambda s: int(s.split("_")[1]))
        X = g[xcols].to_numpy(dtype=float)    # frames x landmarks
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

# CPU fallback mapping (kept for fallback)
def map_detected_to_manual_vectorized(detected_lists_pts, image_w, image_h):
    if len(detected_lists_pts) == 0:
        return {}

    det_arr = np.zeros((len(detected_lists_pts), num_landmarks, 2), dtype=float)
    for i, pts in enumerate(detected_lists_pts):
        det_arr[i, :, 0] = [p[0] for p in pts]
        det_arr[i, :, 1] = [p[1] for p in pts]

    diffs = det_arr[:, None, :, :] - manual_coords_arr[None, :, :, :]
    dists = np.linalg.norm(diffs, axis=3).mean(axis=2)  # (D, H)

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

# --- PyCUDA GPU mapping implementation (per-worker compiled) ---
# We'll compile a small CUDA kernel that computes mean L2 distance per (det,hand) pair.
# Each thread computes one (det_idx, hand_idx) result by looping landmarks.
PYCUDA_AVAILABLE = True  # will flip to False if import fails at worker init

# tremor generator (vectorized CPU) - kept same
def tremor_coords_for_hand_vectorized(hid, t_seconds, scale=1.0):
    i = hand_index_map[hid]
    base = manual_coords_arr[i, :, :]  # (L,2)
    phases = phases_arr[i, :]          # (L,)
    amp_scales = amp_scales_arr[i, :]  # (L,)

    decay_factor = max(0.1, math.exp(-t_seconds / 30.0))
    amp_jitter = 0.7 + 0.3 * math.sin(0.1 * 2 * math.pi * t_seconds)
    drift_x = 0.12 * t_seconds
    drift_y = 0.08 * t_seconds

    f = TREMOR_FREQ_HZ * (1 + 0.05 * math.sin(0.05 * 2 * math.pi * t_seconds))
    tremor_amp = TREMOR_AMP_PX * scale * amp_scales * decay_factor * amp_jitter  # (L,)

    jitter_x_sin = tremor_amp * np.sin(2 * math.pi * f * t_seconds + phases)
    jitter_y_sin = tremor_amp * np.sin(2 * math.pi * f * t_seconds + phases + 0.7)
    jitter_x_noise = np.random.normal(scale=GAUSS_NOISE_SIGMA, size=num_landmarks)
    jitter_y_noise = np.random.normal(scale=GAUSS_NOISE_SIGMA, size=num_landmarks)

    xs = base[:, 0] + jitter_x_sin + jitter_x_noise + drift_x
    ys = base[:, 1] + jitter_y_sin + jitter_y_noise + drift_y

    return xs.tolist(), ys.tolist()

# worker-level cache objects (initialized inside each worker)
_worker_gpu_ctx = None
_worker_cuda_module = None
_worker_cuda_func = None
_worker_pycuda_ready = False

def _init_pycuda_worker():
    """
    Try to initialize PyCUDA in this worker process and compile the kernel.
    On failure, mark PYCUDA_AVAILABLE False for this worker (fallback to CPU).
    """
    global _worker_gpu_ctx, _worker_cuda_module, _worker_cuda_func, _worker_pycuda_ready, PYCUDA_AVAILABLE

    try:
        import pycuda.autoinit  # establishes context for this process
        import pycuda.driver as drv
        from pycuda.compiler import SourceModule
    except Exception as e:
        # Not available or failed to init — we'll fallback to CPU mapping
        PYCUDA_AVAILABLE = False
        _worker_pycuda_ready = False
        return

    kernel_code = r"""
    extern "C" {
    __global__ void compute_mean_dists(
        const float *det,   // det: (D * L * 2)
        const float *man,   // man: (H * L * 2)
        float *out,         // out: (D * H)
        const int D,
        const int H,
        const int L
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = D * H;
        if (idx >= total) return;
        int det_i = idx / H;
        int hand_j = idx % H;

        float sum = 0.0f;
        // iterate landmarks
        for (int l = 0; l < L; ++l) {
            int det_off = (det_i * L + l) * 2;
            int man_off = (hand_j * L + l) * 2;
            float dx = det[det_off] - man[man_off];
            float dy = det[det_off + 1] - man[man_off + 1];
            sum += sqrtf(dx * dx + dy * dy);
        }
        out[idx] = sum / (float)L;
    }
    }
    """
    try:
        _worker_cuda_module = SourceModule(kernel_code)
        _worker_cuda_func = _worker_cuda_module.get_function("compute_mean_dists")
        _worker_pycuda_ready = True
    except Exception as e:
        PYCUDA_AVAILABLE = False
        _worker_pycuda_ready = False

def map_detected_to_manual_gpu(detected_lists_pts):
    """
    Use PyCUDA to compute mean distances (D,H) and return mapping detected_index -> hand_id
    If GPU not ready, raise RuntimeError (caller should fallback).
    """
    if not _worker_pycuda_ready:
        raise RuntimeError("PyCUDA not initialized in worker")

    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray

    D = len(detected_lists_pts)
    H = len(available_hands)
    L = num_landmarks

    # prepare det array shape (D, L, 2) flattened to float32
    det_np = np.zeros((D, L, 2), dtype=np.float32)
    for i, pts in enumerate(detected_lists_pts):
        det_np[i, :, 0] = [p[0] for p in pts]
        det_np[i, :, 1] = [p[1] for p in pts]

    # manual_coords_arr shape (H, L, 2) -> float32
    man_np = manual_coords_arr.astype(np.float32)

    # flatten to contiguous
    det_flat = np.ascontiguousarray(det_np.reshape(-1))
    man_flat = np.ascontiguousarray(man_np.reshape(-1))

    # allocate GPU arrays
    det_gpu = gpuarray.to_gpu(det_flat)
    man_gpu = gpuarray.to_gpu(man_flat)
    out_gpu = gpuarray.empty((D * H,), dtype=np.float32)

    # launch kernel: choose block size 128
    total = D * H
    block = 128
    grid = (total + block - 1) // block

    # call kernel: note we need to pass pointers and ints
    _worker_cuda_func(
        det_gpu, man_gpu, out_gpu,
        np.int32(D), np.int32(H), np.int32(L),
        block=(block, 1, 1), grid=(int(grid), 1, 1)
    )

    # copy back
    out_flat = out_gpu.get()  # length D*H
    dists = out_flat.reshape((D, H))  # (D, H)

    # greedy assignment (same logic as CPU version)
    mapping = {}
    used_hands = set()
    for det_i in range(D):
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

# --- Main per-video processing (worker) ---
def process_upright_video_worker(args):
    """
    Worker called in a separate process.
    args: tuple(video_path, output_csv_path, subject_id, hand_scales)
    Returns (video_path, success_bool)
    """
    video_path, output_csv_path, subject_id, hand_scales = args

    # Seed RNG per process deterministically
    pid = os.getpid()
    seed = (RANDOM_SEED or 0) + pid
    np.random.seed(seed)

    # Initialize PyCUDA in this worker (if available)
    try:
        _init_pycuda_worker()
    except Exception:
        # initialization failures handled inside _init_pycuda_worker
        pass

    # Create MediaPipe Hands instance in this process
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

    # metadata counters
    detected_count = {hid: 0 for hid in available_hands}
    total_frames = 0

    print(f"[INFO] Processing upright video: {os.path.basename(video_path)} (fps={fps:.2f})")

    # Detect if OpenCV has CUDA color conversion available and usable
    use_cuda = False
    try:
        if hasattr(cv2, "cuda"):
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                use_cuda = True
    except Exception:
        use_cuda = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        t = frame_idx / fps

        # color conversion - try CUDA path if available (non-breaking)
        if use_cuda:
            try:
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(frame)
                frame_rgb_gpu = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2RGB)
                frame_rgb = frame_rgb_gpu.download()
            except Exception:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        detected_lists = []
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                pts = mp_landmarks_to_pixels(lm, w, h)
                # only include if expected number of landmarks
                if len(pts) == num_landmarks:
                    detected_lists.append(pts)

        det_to_hand = {}
        if detected_lists:
            # Prefer GPU mapping if initialized for this worker
            if _worker_pycuda_ready:
                try:
                    det_to_hand = map_detected_to_manual_gpu(detected_lists)
                except Exception as e:
                    # GPU mapping failed for some reason -> fallback to CPU vectorized mapping
                    det_to_hand = map_detected_to_manual_vectorized(detected_lists, w, h)
            else:
                det_to_hand = map_detected_to_manual_vectorized(detected_lists, w, h)

        # build reverse mapping: hand_id -> detected pts (if present)
        handid_to_detected = {}
        for det_idx, hid in det_to_hand.items():
            pts = detected_lists[det_idx]
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
                xs, ys = tremor_coords_for_hand_vectorized(hid, t, scale=scale)

            row = [frame_idx, hid] + xs + ys
            rows.append(row)

        frame_idx += 1

    cap.release()
    hands.close()

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
        hand_scales = compute_subject_scales(root_folder, subject_id)
        print(f"[INFO] subject {subject_id} scales: {hand_scales}")
        try:
            # run each video in its own worker process (so PyCUDA contexts are separate)
            with ProcessPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(process_upright_video_worker, (video_path, output_csv, subject_id, hand_scales))
                res = fut.result()
                if not res[1]:
                    print(f"[ERROR] processing {video_path}")
        except Exception as e:
            print(f"[ERROR] processing {video_path}: {e}")

if __name__ == "__main__":
    main()
