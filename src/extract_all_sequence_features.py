"""Extract scale-normalized temporal features for every URFD RGB sequence."""

from pathlib import Path
import csv

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RGB_ROOT = ROOT / "data" / "urfd" / "raw" / "UR_fall_detection_dataset_cam0_rgb"
MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"
OUTPUT_CSV = ROOT / "outputs" / "features" / "all-sequence-ml-features.csv"

FPS = 30.0
BASELINE_SECONDS = 2.0
CHANGE_WINDOW_SECONDS = 1.0

FEATURE_COLUMNS = [
    "pose_detection_rate",
    "baseline_body_height_px",
    "max_torso_tilt_deg",
    "p90_torso_tilt_deg",
    "mean_torso_tilt_deg",
    "min_height_ratio",
    "p10_height_ratio",
    "max_tilt_rise_1s_deg",
    "max_height_drop_1s_ratio",
    "fraction_tilt_over_45_deg",
    "fraction_height_below_70pct",
]


def midpoint(landmarks, first, second, width, height):
    a, b = landmarks[first], landmarks[second]
    return ((a.x + b.x) * width / 2.0, (a.y + b.y) * height / 2.0)


def extract_one_sequence(detector, folder):
    image_paths = sorted(folder.glob("*.png"))
    tilts = np.full(len(image_paths), np.nan, dtype=float)
    heights = np.full(len(image_paths), np.nan, dtype=float)

    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if not result.pose_landmarks:
            continue

        landmarks = result.pose_landmarks[0]
        shoulder = midpoint(landmarks, 11, 12, width, height)
        hip = midpoint(landmarks, 23, 24, width, height)
        ankle = midpoint(landmarks, 27, 28, width, height)
        tilts[index] = np.degrees(
            np.arctan2(abs(hip[0] - shoulder[0]), abs(hip[1] - shoulder[1]))
        )
        heights[index] = abs(ankle[1] - shoulder[1])

    valid = np.isfinite(tilts) & np.isfinite(heights) & (heights > 0)
    baseline_valid = valid & (np.arange(len(image_paths)) / FPS <= BASELINE_SECONDS)
    if not valid.any() or not baseline_valid.any():
        return None

    baseline_height = float(np.median(heights[baseline_valid]))
    height_ratio = heights / baseline_height
    lag = max(1, round(FPS * CHANGE_WINDOW_SECONDS))
    tilt_rises = []
    height_drops = []
    for index in range(lag, len(image_paths)):
        if valid[index] and valid[index - lag]:
            tilt_rises.append(tilts[index] - tilts[index - lag])
            height_drops.append(height_ratio[index - lag] - height_ratio[index])

    valid_tilts = tilts[valid]
    valid_ratios = height_ratio[valid]
    return {
        "sequence": folder.name,
        "label": int(folder.name.startswith("fall-")),
        "frame_count": len(image_paths),
        "pose_detection_rate": float(valid.mean()) if len(valid) else 0.0,
        "baseline_body_height_px": baseline_height,
        "max_torso_tilt_deg": float(np.max(valid_tilts)),
        "p90_torso_tilt_deg": float(np.percentile(valid_tilts, 90)),
        "mean_torso_tilt_deg": float(np.mean(valid_tilts)),
        "min_height_ratio": float(np.min(valid_ratios)),
        "p10_height_ratio": float(np.percentile(valid_ratios, 10)),
        "max_tilt_rise_1s_deg": float(max(tilt_rises)) if tilt_rises else 0.0,
        "max_height_drop_1s_ratio": float(max(height_drops)) if height_drops else 0.0,
        "fraction_tilt_over_45_deg": float(np.mean(valid_tilts >= 45.0)),
        "fraction_height_below_70pct": float(np.mean(valid_ratios <= 0.70)),
    }


def main():
    if not RGB_ROOT.exists():
        raise FileNotFoundError(f"RGB dataset folder not found:\n{RGB_ROOT}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Pose model not found:\n{MODEL_PATH}")

    folders = sorted(
        path for path in RGB_ROOT.iterdir()
        if path.is_dir() and (path.name.startswith("fall-") or path.name.startswith("adl-"))
    )
    if not folders:
        raise RuntimeError(f"No fall-/adl- sequence folders found in {RGB_ROOT}")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    try:
        for index, folder in enumerate(folders, start=1):
            print(f"[{index}/{len(folders)}] Extracting {folder.name}", flush=True)
            row = extract_one_sequence(detector, folder)
            if row is None:
                print("  Skipped: no usable pose baseline", flush=True)
            else:
                rows.append(row)
    finally:
        detector.close()

    columns = ["sequence", "label", "frame_count", *FEATURE_COLUMNS]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sequences represented: {len(rows)}")
    print(f"Fall sequences: {sum(row['label'] == 1 for row in rows)}")
    print(f"ADL sequences: {sum(row['label'] == 0 for row in rows)}")
    print(f"Saved machine-learning features to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
