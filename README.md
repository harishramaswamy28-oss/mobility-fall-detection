# Mobility and Fall Detection

Research prototype for camera-based mobility monitoring and fall detection. The current work uses RGB video, MediaPipe pose landmarks, frame/sequence features, and recording-level machine-learning evaluation. A separate experiment compares RGB-only monocular depth models with URFD Kinect depth as an offline reference.

## Project status

- Pose landmarks can be extracted from URFD RGB sequences.
- Fall and ADL classification experiments are implemented in `src/`.
- Depth Anything V2 Metric Indoor Small and Metric3D v2 ViT-Small have been compared on six URFD cam0 sequences.
- The monocular-depth comparison is experimental. Metric3D currently uses a provisional 525 px focal length. Neither model has yet been validated for deployment-camera 3D coordinates, body angles, gait velocity, or fall detection.

## Environment

The project was developed with Python 3.11 on Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The MediaPipe pose model file is not included. Place the downloaded `pose_landmarker_full.task` in `models/`.

## URFD data

Download the dataset from the [official URFD page](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) and extract the files locally. The RGB frames, Kinect depth frames, and synchronization CSVs are intentionally excluded from Git because they take several gigabytes and are distributed under the dataset's license. The small annotation CSVs in `data/urfd/annotations/` are retained for the classifier experiments.

Expected local paths include:

```text
data/urfd/raw/UR_fall_detection_dataset_cam0_rgb/<sequence>-cam0-rgb/
data/urfd/depth_sample/<sequence>-cam0-d/...
data/urfd/metadata_sample/<sequence>-data.csv
```

## Main commands

Extract pose landmarks for one sequence (for example, `fall-02`):

```powershell
python .\src\pose_sequence.py fall-02
```

Run the tabular classifier benchmark and comparison:

```powershell
python .\src\benchmark_fall_classifiers.py
python .\src\compare_fall_models.py
```

The RGB-only depth comparison is an optional experiment. It requires the two model checkpoints in `models/depth_models/`, the official model source trees in `third_party/Depth-Anything-V2/` and `third_party/Metric3D/`, and a PyTorch runtime available in `.depth_runtime/`. Outputs are written under `outputs/depth_model_comparison/` and are ignored by Git.

## Repository contents

- `src/` — pose extraction, feature extraction, classifier experiments, and depth-model comparison.
- `data/urfd/annotations/` — compact annotation CSVs used by the classifier pipeline.
- `requirements.txt` — core Python dependencies.

Model weights, downloaded video/depth data, virtual environments, vendored repositories, and generated outputs are kept out of version control. See `.gitignore` for the full list.
