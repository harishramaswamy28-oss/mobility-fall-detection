from pathlib import Path
import csv
import sys

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"

# MediaPipe's standard 33 landmark names.
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def load_timestamps(sync_csv):
    """Read the verified URFD frame-to-time mapping."""
    timestamps = {}

    with sync_csv.open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            frame_number = int(row[0])
            time_ms = float(row[1])
            timestamps[frame_number] = time_ms

    return timestamps


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python .\\src\\pose_sequence.py <sequence-id>  (e.g. fall-02)")

    sequence_id = sys.argv[1]
    rgb_folder = (
        PROJECT_ROOT
        / "data"
        / "urfd"
        / "raw"
        / "UR_fall_detection_dataset_cam0_rgb"
        / f"{sequence_id}-cam0-rgb"
    )
    sync_csv = PROJECT_ROOT / "data" / "urfd" / "metadata_sample" / f"{sequence_id}-data.csv"
    output_csv = PROJECT_ROOT / "outputs" / "poses" / f"{sequence_id}-pose-landmarks.csv"

    if not rgb_folder.exists():
        raise FileNotFoundError(f"RGB folder not found:\n{rgb_folder}")

    if not sync_csv.exists():
        raise FileNotFoundError(f"Synchronization CSV not found:\n{sync_csv}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Pose model not found:\n{MODEL_PATH}")

    image_paths = sorted(rgb_folder.glob("*.png"))
    if not image_paths:
        raise RuntimeError("No PNG frames were found.")

    timestamps = load_timestamps(sync_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    detected_frames = 0
    missing_frames = 0

    fieldnames = [
        "frame_number",
        "time_ms",
        "frame_file",
        "landmark_index",
        "landmark_name",
        "x_normalized",
        "y_normalized",
        "z_mediapipe_relative",
        "visibility",
        "presence",
    ]

    try:
        with output_csv.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            for frame_number, image_path in enumerate(image_paths, start=1):
                bgr_image = cv2.imread(str(image_path))
                if bgr_image is None:
                    print(f"Skipped unreadable image: {image_path.name}")
                    missing_frames += 1
                    continue

                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_image,
                )

                result = detector.detect(mp_image)

                if not result.pose_landmarks:
                    print(f"No pose detected: {image_path.name}")
                    missing_frames += 1
                    continue

                detected_frames += 1
                landmarks = result.pose_landmarks[0]

                for landmark_index, landmark in enumerate(landmarks):
                    writer.writerow(
                        {
                            "frame_number": frame_number,
                            "time_ms": timestamps.get(frame_number),
                            "frame_file": image_path.name,
                            "landmark_index": landmark_index,
                            "landmark_name": LANDMARK_NAMES[landmark_index],
                            "x_normalized": landmark.x,
                            "y_normalized": landmark.y,
                            # This is MediaPipe's relative z value, not metric depth.
                            "z_mediapipe_relative": landmark.z,
                            "visibility": landmark.visibility,
                            "presence": landmark.presence,
                        }
                    )

    finally:
        detector.close()

    print(f"Frames examined: {len(image_paths)}")
    print(f"Frames with a detected pose: {detected_frames}")
    print(f"Frames without a pose: {missing_frames}")
    print(f"Saved landmark table to:\n{output_csv}")


if __name__ == "__main__":
    main()
