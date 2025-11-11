import cupy as cp
from concurrent.futures import ThreadPoolExecutor, as_completed

BATCH_SIZE = 32  # number of frames processed per GPU batch (tune for VRAM)
THREADS = 4      # number of CPU threads for I/O overlap

def process_upright_video(video_path, output_csv_path, hand_scales):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return video_path, False

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    rows = []
    frame_idx = 0
    detected_count = {hid: 0 for hid in available_hands}
    total_frames = 0

    print(f"[INFO] GPU-batched processing: {os.path.basename(video_path)} (fps={fps:.2f})")

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         ThreadPoolExecutor(max_workers=THREADS) as executor:

        batch_frames = []
        batch_indices = []
        pending = []

        def process_batch(frames_cpu, indices_cpu):
            # Move batch to GPU
            frames_gpu = cp.asarray(frames_cpu)
            results = []
            for i, frame_gpu in enumerate(frames_gpu):
                # Convert back to CPU NumPy for MediaPipe (not GPU supported)
                frame = cp.asnumpy(frame_gpu).astype(np.uint8)
                h, w = frame.shape[:2]
                t = indices_cpu[i] / fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_result = hands.process(frame_rgb)

                detected_lists = []
                if mp_result.multi_hand_landmarks:
                    for lm in mp_result.multi_hand_landmarks:
                        detected_lists.append(lm)

                det_to_hand = map_detected_to_manual(detected_lists, w, h) if detected_lists else {}
                handid_to_detected = {}
                for det_idx, hid in det_to_hand.items():
                    pts = mp_landmarks_to_pixels(detected_lists[det_idx], w, h)
                    if len(pts) == num_landmarks:
                        handid_to_detected[hid] = pts

                # per hand processing (GPU)
                for hid in available_hands:
                    if hid in handid_to_detected:
                        pts = cp.array(handid_to_detected[hid])
                        xs = cp.asnumpy(pts[:, 0])
                        ys = cp.asnumpy(pts[:, 1])
                        detected_count[hid] += 1
                    else:
                        scale = hand_scales.get(hid, 1.0)
                        xs, ys = tremor_coords_for_hand(hid, t, scale)
                    row = [indices_cpu[i], hid] + list(xs) + list(ys)
                    results.append(row)
            return results

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            frame_idx += 1

            if len(batch_frames) >= BATCH_SIZE:
                future = executor.submit(process_batch, np.array(batch_frames), list(batch_indices))
                pending.append(future)
                batch_frames = []
                batch_indices = []

        # process leftover frames
        if batch_frames:
            future = executor.submit(process_batch, np.array(batch_frames), list(batch_indices))
            pending.append(future)

        for f in as_completed(pending):
            rows.extend(f.result())

    cap.release()

    columns = ["frame", "hand_id"] + [f"x_{i}" for i in landmark_ids] + [f"y_{i}" for i in landmark_ids]
    df = pd.DataFrame(rows, columns=columns)
    out_dir = os.path.dirname(output_csv_path)
    os.makedirs(_win_long_path(out_dir), exist_ok=True)
    df.to_csv(_win_long_path(output_csv_path), index=False)

    meta = {
        "video": os.path.basename(video_path),
        "frames": frame_idx,
        "hands": {str(hid): {
            "palmdown_scale": float(hand_scales.get(hid, 1.0)),
            "detected_frames": int(detected_count[hid]),
            "frame_count": int(frame_idx),
            "detection_fraction": float(detected_count[hid] / max(1, frame_idx))
        } for hid in available_hands}
    }

    meta_path = output_csv_path + ".meta.json"
    with open(_win_long_path(meta_path), "w") as jf:
        json.dump(meta, jf, indent=2)

    print(f"[SAVED] {output_csv_path}  (rows={len(df)}), meta -> {meta_path}")
    return (video_path, True)
