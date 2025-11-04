#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from src.mmpose_analyzer import PosturalAnalyzer

VALID_VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV'}

def is_video_file(fname):
    return os.path.splitext(fname)[1] in VALID_VIDEO_EXTS

def sanitize_relpath(relpath):
    """
    Turn a relative path into a safe single token for filenames:
    replace path separators with '_' and strip dots/slashes.
    """
    if relpath is None:
        return ''
    s = relpath.strip().strip(os.sep).replace(os.sep, '_').replace('/', '_').replace('\\', '_')
    if s == '' or s == '.':
        return ''
    return s

def find_and_move_first_csv(src_dir, dst_path):
    """
    Look for any .csv file under src_dir (non-recursive first, then recursive).
    If found, move the first one to dst_path (overwrite if exists).
    Return True on success, False if not found.
    """
    if not os.path.isdir(src_dir):
        return False

    # Non-recursive check
    for f in os.listdir(src_dir):
        if f.lower().endswith('.csv'):
            src = os.path.join(src_dir, f)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if os.path.exists(dst_path):
                os.remove(dst_path)
            shutil.move(src, dst_path)
            return True

    # Recursive search
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                src = os.path.join(root, f)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                shutil.move(src, dst_path)
                return True

    return False

def process_one_video(analyzer, video_path, output_dir, root_folder, return_vis):
    print(f"\n--> Processing: {video_path}")

    video_dir = os.path.dirname(video_path)
    try:
        rel = os.path.relpath(video_dir, start=root_folder)
    except Exception:
        rel = os.path.basename(video_dir)

    rel_clean = sanitize_relpath(rel)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    # Final CSV name: <rel_clean>_<video_basename>.csv (if rel_clean empty, use video_basename.csv)
    if rel_clean:
        desired_csv_name = f"{rel_clean}_{video_basename}.csv"
    else:
        desired_csv_name = f"{video_basename}.csv"
    desired_csv_path = os.path.join(output_dir, desired_csv_name)

    # Temporary output directory where PosturalAnalyzer will write outputs
    tmp_out = os.path.join(output_dir, "__tmp_processing__", rel_clean or video_basename)
    os.makedirs(tmp_out, exist_ok=True)

    # Call analyzer
    try:
        analyzer.process_video(video_path=video_path, output_dir=tmp_out, return_vis=return_vis)
    except TypeError:
        # Fallback if signature differs
        analyzer.process_video(video_path, tmp_out, return_vis)

    # Find and move first CSV from tmp_out to desired_csv_path
    moved = find_and_move_first_csv(tmp_out, desired_csv_path)
    if moved:
        print(f"  ✅ Saved CSV as: {desired_csv_path}")
    else:
        print(f"  ⚠️ No CSV found in analyzer output (looked in {tmp_out}).")

    # Cleanup tmp folder if empty
    try:
        if os.path.isdir(tmp_out) and not os.listdir(tmp_out):
            os.rmdir(tmp_out)
        parent_tmp = os.path.join(output_dir, "__tmp_processing__")
        if os.path.isdir(parent_tmp) and not os.listdir(parent_tmp):
            os.rmdir(parent_tmp)
    except Exception:
        pass

    print(f"--> Done: {video_path}")

def process_root_folder(analyzer, root_folder, output_dir, return_vis):
    found_any = False
    for current_root, dirs, files in os.walk(root_folder):
        folder_name = os.path.basename(current_root).lower()
        if folder_name in ('whole_body', 'wholebody', 'whole_body'):
            found_any = True
            print(f"\nFound WHOLE_BODY folder: {current_root}")
            video_files = [os.path.join(current_root, f) for f in files if is_video_file(f)]
            if not video_files:
                print("  No supported video files found in this WHOLE_BODY folder.")
                continue
            for v in sorted(video_files):
                process_one_video(analyzer, v, output_dir, root_folder, return_vis)

    if not found_any:
        print("No WHOLE_BODY subfolders were found under the provided directory.")
    else:
        print("\nAll done.")

def main(args):
    path = args.path
    use_gpu = args.use_gpu
    return_vis = args.return_vis
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    analyzer = PosturalAnalyzer(inferencer='wholebody', use_gpu=use_gpu)

    if os.path.isfile(path):
        if not is_video_file(path):
            print("Given file is not a supported video type.")
            sys.exit(1)
        root_folder = os.path.dirname(path)
        process_one_video(analyzer, path, output_dir, root_folder, return_vis)
        return

    if os.path.isdir(path):
        process_root_folder(analyzer, path, output_dir, return_vis)
        return

    print("Provided path is neither a file nor a directory. Exiting.")
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a video file or a root folder (searches for WHOLE_BODY subfolders)."
    )
    parser.add_argument('path', type=str, help="Path to a video file or a root folder to search.")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Directory to save the output CSVs.")
    parser.add_argument('--use_gpu', action='store_true', help="Use GPU for inference.")
    parser.add_argument('--return_vis', action='store_true', help="Save visualization of keypoints in video.")
    args = parser.parse_args()
    main(args)
