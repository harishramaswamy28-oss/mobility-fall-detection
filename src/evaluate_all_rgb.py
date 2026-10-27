from pathlib import Path
import csv

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RGB_ROOT = ROOT / "data" / "urfd" / "raw" / "UR_fall_detection_dataset_cam0_rgb"
MODEL = ROOT / "models" / "pose_landmarker_full.task"
OUTPUT = ROOT / "outputs" / "detections" / "all-sequences-rgb-summary.csv"

FPS = 30.0
BASELINE_SECONDS = 2.0
LOOKBACK_SECONDS = 1.0
MIN_TILT = 60.0
MAX_HEIGHT_RATIO = 0.75
MIN_TILT_RISE = 45.0
MIN_CONSECUTIVE = 3


def midpoint(landmarks, first, second, width, height):
    a = landmarks[first]
    b = landmarks[second]
    return ((a.x + b.x) * width / 2, (a.y + b.y) * height / 2)


def pose_features(landmarks, width, height):
    shoulder = midpoint(landmarks, 11, 12, width, height)
    hip = midpoint(landmarks, 23, 24, width, height)
    ankle = midpoint(landmarks, 27, 28, width, height)

    tilt = np.degrees(
        np.arctan2(abs(hip[0] - shoulder[0]), abs(hip[1] - shoulder[1]))
    )
    body_height = abs(ankle[1] - shoulder[1])
    return float(tilt), float(body_height)


def longest_confirmed_run(candidates):
    confirmed = []
    run_start = None

    for i, candidate in enumerate(candidates):
        if candidate and run_start is None:
            run_start = i

        if run_start is not None and (not candidate or i == len(candidates) - 1):
            run_end = i if candidate else i - 1
            if run_end - run_start + 1 >= MIN_CONSECUTIVE:
                confirmed.extend(range(run_start, run_end + 1))
            run_start = None

    return confirmed


def process_sequence(detector, folder):
    images = sorted(folder.glob("*.png"))
    frame_features = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            frame_features.append(None)
            continue

        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if not result.pose_landmarks:
            frame_features.append(None)
        else:
            frame_features.append(
                pose_features(result.pose_landmarks[0], width, height)
            )

    baseline_heights = [
        item[1]
        for i, item in enumerate(frame_features)
        if item is not None and i / FPS <= BASELINE_SECONDS
    ]

    if not baseline_heights:
        return len(images), 0, None

    baseline = float(np.median(baseline_heights))
    candidates = [False] * len(images)
    lookback = round(LOOKBACK_SECONDS * FPS)

    for i, current in enumerate(frame_features):
        previous_index = i - lookback
        if current is None or previous_index < 0:
            continue

        previous = frame_features[previous_index]
        if previous is None:
            continue

        tilt, body_height = current
        previous_tilt, _ = previous
        height_ratio = body_height / baseline

        candidates[i] = (
            tilt >= MIN_TILT
            and height_ratio <= MAX_HEIGHT_RATIO
            and tilt - previous_tilt >= MIN_TILT_RISE
        )

    confirmed = longest_confirmed_run(candidates)
    first_time = confirmed[0] / FPS if confirmed else None
    detected_frames = sum(item is not None for item in frame_features)

    return len(images), detected_frames, first_time


def main():
    if not RGB_ROOT.exists():
        raise FileNotFoundError(f"RGB dataset folder not found:\n{RGB_ROOT}")
    if not MODEL.exists():
        raise FileNotFoundError(f"Pose model not found:\n{MODEL}")

    folders = sorted(
        path for path in RGB_ROOT.iterdir()
        if path.is_dir() and (path.name.startswith("fall-") or path.name.startswith("adl-"))
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    try:
        for number, folder in enumerate(folders, start=1):
            print(f"[{number}/{len(folders)}] Processing {folder.name} ...")
            total, detected, first_time = process_sequence(detector, folder)
            rows.append({
                "sequence": folder.name,
                "class": "fall" if folder.name.startswith("fall-") else "adl",
                "frames": total,
                "pose_detected_frames": detected,
                "fall_candidate": first_time is not None,
                "first_candidate_time_s": first_time,
            })
    finally:
        detector.close()

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fall_rows = [row for row in rows if row["class"] == "fall"]
    adl_rows = [row for row in rows if row["class"] == "adl"]

    print(f"Sequences processed: {len(rows)}")
    print(f"Fall sequences flagged: {sum(r['fall_candidate'] for r in fall_rows)} / {len(fall_rows)}")
    print(f"ADL sequences flagged: {sum(r['fall_candidate'] for r in adl_rows)} / {len(adl_rows)}")
    print(f"Summary saved to:\n{OUTPUT}")


if __name__ == "__main__":
    main()