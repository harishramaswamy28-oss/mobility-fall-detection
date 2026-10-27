from pathlib import Path

import cv2
import mediapipe as mp


# Project paths: this script is inside src/, so its parent folder is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "urfd"
    / "raw"
    / "UR_fall_detection_dataset_cam0_rgb"
    / "fall-01-cam0-rgb"
    / "fall-01-cam0-rgb-080.png"
)

MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "fall-01-frame-080-pose.png"

# MediaPipe landmark-index pairs that form the visible body skeleton.
CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso
    (23, 24),  # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
    (27, 29), (29, 31),  # left foot
    (28, 30), (30, 32),  # right foot
]


def pixel_position(landmark, width, height):
    """Convert MediaPipe's normalized landmark coordinates into image pixels."""
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"RGB frame not found:\n{IMAGE_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Pose model not found:\n{MODEL_PATH}")

    bgr_image = cv2.imread(str(IMAGE_PATH))
    if bgr_image is None:
        raise RuntimeError("OpenCV could not read the RGB frame.")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    try:
        result = detector.detect(mp_image)
    finally:
        detector.close()

    if not result.pose_landmarks:
        raise RuntimeError("No person was detected in this frame.")

    landmarks = result.pose_landmarks[0]
    height, width = bgr_image.shape[:2]
    annotated = bgr_image.copy()

    # Draw skeleton lines first.
    for start_index, end_index in CONNECTIONS:
        start = pixel_position(landmarks[start_index], width, height)
        end = pixel_position(landmarks[end_index], width, height)
        cv2.line(annotated, start, end, (0, 255, 0), 2)

    # Draw all 33 detected body landmarks.
    for landmark in landmarks:
        point = pixel_position(landmark, width, height)
        cv2.circle(annotated, point, 4, (0, 0, 255), -1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), annotated)

    print(f"Detected {len(landmarks)} pose landmarks.")
    print(f"Saved annotated image to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()