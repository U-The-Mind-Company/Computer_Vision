# PosturalAnalyzer Video Processing Tool

This tool processes videos to extract keypoints and tremor amplitudes using the [MMPose](https://github.com/open-mmlab/mmpose) model. The tool supports whole-body and hand keypoint detection and can optionally generate visualizations.


---
## Installation

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name tremor-analysis python=3.8 -y
conda activate tremor-analysis
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

```{warning}
This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

**Step 3.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv/tree/2.x) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
```

Note that some of the demo scripts in MMPose require [MMDetection](https://github.com/open-mmlab/mmdetection) (mmdet)  for human detection. If you want to run these demo scripts with mmdet, you can easily install mmdet as a dependency by running:

```shell
mim install "mmdet>=3.1.0"
```

```{note}
Here are the version correspondences between mmdet, mmpose and mmcv:

- mmdet 2.x <=> mmpose 0.x <=> mmcv 1.x
- mmdet 3.x <=> mmpose 1.x <=> mmcv 2.x

If you encounter version incompatibility issues, please check the correspondence using `pip list | grep mm` and upgrade or downgrade the dependencies accordingly. Please note that `mmcv-full` is only for `mmcv 1.x`, so please uninstall it first, and then use `mim install mmcv` to install `mmcv 2.x`.
```

To use mmpose as a dependency or third-party package, install it with pip:

```shell
mim install "mmpose>=1.1.0"
```

---

## Usage

### Command-Line Arguments

The script provides the following options to process video files:

- **`video_path`** (required): Path to the input video file that you want to process.
- **`--use_gpu`** (optional, flag): Use this flag if you want to run the inference on the GPU.
- **`--return_vis`** (optional, flag): Use this flag if you want to generate and save visualizations of the keypoints in the video.

### Example Usage

#### Run without GPU (CPU-based inference):

```bash
python wholebody_analyzer.py /path/to/your/video.mp4
```

#### Run with GPU:

```bash
python wholebody_analyzer.py /path/to/your/video.mp4 --use_gpu
```

This command will use the GPU for inference (if available).

#### Run with Keypoint Visualizations:

```bash
python wholebody_analyzer.py /path/to/your/video.mp4 --return_vis
```

This will save a processed video with keypoints visualized.

#### Run with Both GPU and Visualizations:

```bash
python wholebody_analyzer.py /path/to/your/video.mp4 --use_gpu --return_vis
```

This command will use the GPU and also save visualized keypoints in the processed video.

## Output

- **Processed Video**: If `--return_vis` is used, a video with keypoint visualizations will be saved.
- **Keypoints CSV**: A CSV file containing keypoints for each frame will be generated in the default output directory.
- **Tremor Amplitudes CSV**: A CSV file containing tremor amplitude calculations per frame will be saved.

## Default Output Directory

By default, all output files (processed video, CSVs) are saved in an `outputs/` folder.

---

Feel free to customize the output directory and inferencer type by modifying the script. You can enable the commented-out options for `output_dir` and `inferencer` as required.

---
