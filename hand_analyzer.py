import argparse
from src.mmpose_analyzer import PosturalAnalyzer

def main(args):
    # Initialize the PosturalAnalyzer with the specified inferencer (hand/wholebody)
    analyzer = PosturalAnalyzer(inferencer='hand', use_gpu=args.use_gpu)

    # Process the video
    analyzer.process_video(
        video_path=args.video_path,
        # output_dir=args.output_dir,
        return_vis=args.return_vis,
    )

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process video to extract keypoints and tremor amplitudes.")

    # Required arguments
    parser.add_argument('video_path', type=str, help="Path to the input video file.")
    
    # Optional arguments
    # parser.add_argument('--output_dir', type=str, default='outputs', help="Directory to save the output files.")
    # parser.add_argument('--inferencer', type=str, default='wholebody', choices=['hand', 'wholebody'], help="Type of inferencer to use ('hand' or 'wholebody').")
    parser.add_argument('--use_gpu', action='store_true', help="Flag to use GPU for inference.")
    parser.add_argument('--return_vis', action='store_true', help="Flag to save visualization of keypoints in video.")

    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)
