import os
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
import torch
import csv
import os
import time
import matplotlib.pyplot as plt
import numpy as np

class PosturalAnalyzer():
    def __init__(self, inferencer='hand', use_gpu=False):
        if use_gpu and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.inferencer_name = inferencer
        self.inferencer = MMPoseInferencer(inferencer, device=device)
        self.landmark_list = []  # Initialize the landmark list
        if inferencer == 'hand':
            self.body_parts ={
                                0: 'Wrist',
                                1: 'Thumb_CMC',
                                2: 'Thumb_MCP',
                                3: 'Thumb_IP',
                                4: 'Thumb_Tip',
                                5: 'Index_MCP',
                                6: 'Index_PIP',
                                7: 'Index_DIP',
                                8: 'Index_Tip',
                                9: 'Middle_MCP',
                                10: 'Middle_PIP',
                                11: 'Middle_DIP',
                                12: 'Middle_Tip',
                                13: 'Ring_MCP',
                                14: 'Ring_PIP',
                                15: 'Ring_DIP',
                                16: 'Ring_Tip',
                                17: 'Pinky_MCP',
                                18: 'Pinky_PIP',
                                19: 'Pinky_DIP',
                                20: 'Pinky_Tip'
                            }


        elif inferencer == 'wholebody':
            self.body_parts = {
                '0': 'nose', '1': 'left_eye', '2': 'right_eye', '3': 'left_ear', '4': 'right_ear', '5': 'left_shoulder',
                '6': 'right_shoulder', '7': 'left_elbow', '8': 'right_elbow', '9': 'left_wrist', '10': 'right_wrist', 
                '11': 'left_hip', '12': 'right_hip', '13': 'left_knee', '14': 'right_knee', '15': 'left_ankle', '16': 'right_ankle', 
                '17': 'left_big_toe', '18': 'left_small_toe', '19': 'left_heel', '20': 'right_big_toe', '21': 'right_small_toe', 
                '22': 'right_heel', '23': 'face-0', '24': 'face-1', '25': 'face-2', '26': 'face-3', '27': 'face-4', '28': 'face-5', 
                '29': 'face-6', '30': 'face-7', '31': 'face-8', '32': 'face-9', '33': 'face-10', '34': 'face-11', '35': 'face-12', 
                '36': 'face-13', '37': 'face-14', '38': 'face-15', '39': 'face-16', '40': 'face-17', '41': 'face-18', '42': 'face-19',
                '43': 'face-20', '44': 'face-21', '45': 'face-22', '46': 'face-23', '47': 'face-24', '48': 'face-25', '49': 'face-26',
                '50': 'face-27', '51': 'face-28', '52': 'face-29', '53': 'face-30', '54': 'face-31', '55': 'face-32', 
                '56': 'face-33', '57': 'face-34', '58': 'face-35', '59': 'face-36', '60': 'face-37', '61': 'face-38', 
                '62': 'face-39','63': 'face-40', '64': 'face-41', '65': 'face-42', '66': 'face-43', '67': 'face-44', '68': 'face-45', 
                '69': 'face-46', '70': 'face-47', '71': 'face-48', '72': 'face-49', '73': 'face-50', '74': 'face-51', 
                '75': 'face-52', '76': 'face-53', '77': 'face-54', '78': 'face-55', '79': 'face-56', '80': 'face-57', 
                '81': 'face-58', '82': 'face-59', '83': 'face-60', '84': 'face-61', '85': 'face-62', '86': 'face-63', 
                '87': 'face-64', '88': 'face-65', '89': 'face-66', '90': 'face-67', '91': 'left_hand_root', '92': 'left_thumb1', 
                '93': 'left_thumb2', '94': 'left_thumb3', '95': 'left_thumb4', '96': 'left_forefinger1', '97': 'left_forefinger2',
                '98': 'left_forefinger3', '99': 'left_forefinger4', '100': 'left_middle_finger1', '101': 'left_middle_finger2', 
                '102': 'left_middle_finger3', '103': 'left_middle_finger4', '104': 'left_ring_finger1', '105': 'left_ring_finger2',
                '106': 'left_ring_finger3', '107': 'left_ring_finger4', '108': 'left_pinky_finger1', '109': 'left_pinky_finger2',
                '110': 'left_pinky_finger3', '111': 'left_pinky_finger4', '112': 'right_hand_root', '113': 'right_thumb1', 
                '114': 'right_thumb2', '115': 'right_thumb3', '116': 'right_thumb4', '117': 'right_forefinger1', 
                '118': 'right_forefinger2', '119': 'right_forefinger3', '120': 'right_forefinger4', 
                '121': 'right_middle_finger1', '122': 'right_middle_finger2', '123': 'right_middle_finger3', 
                '124': 'right_middle_finger4', '125': 'right_ring_finger1', '126': 'right_ring_finger2', 
                '127': 'right_ring_finger3', '128': 'right_ring_finger4', '129': 'right_pinky_finger1', 
                '130': 'right_pinky_finger2', '131': 'right_pinky_finger3', '132': 'right_pinky_finger4'}

    def process_video(self, video_path, output_dir="outputs", return_vis=True):
        # Create output directory and subdirectory for video file if it doesn't exist
        video_name = os.path.basename(video_path).replace('.mp4', '')
        video_output_dir = os.path.join(output_dir,f'{self.inferencer_name}',video_name,)
        os.makedirs(video_output_dir, exist_ok=True)  # Ensure all directories are created

        video_capture = cv2.VideoCapture(video_path)
        original_fps = video_capture.get(cv2.CAP_PROP_FPS)  # Get the original FPS of the input video
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

        # Define the codec and create a VideoWriter object for saving the output
        output_file_path = os.path.join(video_output_dir, f'{video_name}_processed.mp4')
        out = cv2.VideoWriter(output_file_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),  # Codec for mp4
                            original_fps,  # Use original FPS
                            (frame_width, frame_height))

        visualisations = []
        preds = []
        start_time = time.time()

        # Output CSV for keypoints data
        csv_output_path = os.path.join(video_output_dir, f'{video_name}_keypoints.csv')
        tremor_csv_output_path = os.path.join(video_output_dir, f'{video_name}_tremor_amplitudes.csv')

        with open(csv_output_path, mode='w', newline='') as csv_file, \
             open(tremor_csv_output_path, mode='w', newline='') as tremor_csv_file:
            
            writer = csv.writer(csv_file)
            tremor_writer = csv.writer(tremor_csv_file)
            
            # Writing headers for keypoints and tremor amplitudes CSVs
            keypoints_header = ['Frame'] + [self.body_parts[kp] for kp in self.body_parts.keys()]
            writer.writerow(keypoints_header)
            tremor_writer.writerow(['Frame', 'Tremor_Amplitude'])
            
            prev_landmarks = None
            tremor_amplitudes = []

            # Process video frame-by-frame
            frame_count = 0
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break  # No more frames

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for the inferencer

                # Get hand landmarks
                result_generator = self.inferencer(frame_rgb, show=False, out_dir=None, return_vis=return_vis)
                result = next(result_generator)

                # Ensure predictions exist
                if 'predictions' in result and result['predictions']:
                    keypoints = result['predictions'][0][0]['keypoints']  # Store landmarks
                    self.landmark_list.append(keypoints)  # Save the landmarks

                    # Prepare row with frame number and keypoints as (x, y) tuples
                    row = [frame_count]
                    for i in range(len(self.body_parts.keys())):
                        # Check if the keypoint exists, otherwise append (None, None)
                        if i < len(keypoints):
                            row.append(keypoints[i])  # Append (x, y) tuple
                        else:
                            row.append((None, None))  # Append (None, None) if keypoint is missing

                    writer.writerow(row)
                    
                    if return_vis:
                        vis = result['visualization'][0]
                        visualisations.append(vis)

                        # Convert the visualization back to BGR for saving
                        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                        out.write(vis_bgr)  # Write the frame to the output video

                    current_tremor_amplitudes = []
                    prev_landmarks, current_tremor_amplitudes = self.update_tremor_analysis(prev_landmarks, current_tremor_amplitudes)
                    tremor_amplitudes.extend(current_tremor_amplitudes)

                    # Save tremor amplitude for this frame
                    if current_tremor_amplitudes:
                        tremor_writer.writerow([frame_count, current_tremor_amplitudes[-1]])

                frame_count += 1

            # Cleanup after finishing the video processing
            video_capture.release()
            out.release()

        # plt.clf()  # Clear the plot
        time_taken = time.time() - start_time

        print(f"Processed {frame_count}/{total_frames} frames")
        print(f"Time taken: {time_taken:.2f} seconds")
        print(f"Average time per frame: {time_taken / frame_count:.4f} seconds")
        print(f"Processed video saved to {output_file_path}")

        print(f"Keypoints time series saved to {csv_output_path}")
        print(f"Tremor amplitudes saved to {tremor_csv_output_path}")
        plot_path = os.path.join(output_dir,f'{self.inferencer_name}',os.path.basename(video_path).replace('.mp4', ''), os.path.basename(video_path).replace('.mp4', '_tremor_plot.png'))
        self.plot_tremor_signal(tremor_amplitudes,plot_path )

        # return tremor_amplitudes
    
    def update_tremor_analysis(self, prev_landmarks, current_tremor_amplitudes):
        """
        Calculate the tremor amplitude based on hand landmarks' displacement.
        :param prev_landmarks: Hand landmarks from the previous frame.
        :param current_tremor_amplitudes: List of tremor amplitudes to update.
        :return: Updated landmarks and tremor amplitudes for the current frame.
        """
        if len(self.landmark_list) > 1:
            current_landmarks = np.array(self.landmark_list[-1])

            if prev_landmarks is None:
                prev_landmarks = current_landmarks
                return prev_landmarks, current_tremor_amplitudes

            # Calculate displacement (tremor amplitude) between consecutive frames
            displacement = np.linalg.norm(current_landmarks - prev_landmarks)
            tremor_amplitude = displacement / len(current_landmarks)
            current_tremor_amplitudes.append(tremor_amplitude)

            print(f"Tremor Amplitude: {tremor_amplitude:.2f}")
            return current_landmarks, current_tremor_amplitudes

        else:
            return prev_landmarks, current_tremor_amplitudes

    def plot_tremor_signal(self, tremor_amplitudes, output=None):
        avg_amplitude = np.mean(tremor_amplitudes) if len(tremor_amplitudes) > 0 else 0

        # Create time values based on the number of frames
        time = np.arange(len(tremor_amplitudes))

        # Plot the tremor signal
        plt.figure(figsize=(12, 6))
        plt.plot(time, tremor_amplitudes, label='Tremor Amplitude', color='blue')
        plt.axhline(y=avg_amplitude, color='red', linestyle='--', label=f'Avg. Amplitude: {avg_amplitude:.2f}')
        plt.xlabel('Time (frames)')
        plt.ylabel('Tremor Amplitude')
        plt.title('Tremor Amplitude Over Time')
        plt.legend()
        if output:
            plt.savefig(output)
        plt.tight_layout()
        plt.show()
