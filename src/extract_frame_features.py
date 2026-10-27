from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "outputs" / "poses" / "adl-01-pose-with-depth.csv"
FEATURE_CSV = PROJECT_ROOT / "outputs" / "features" / "adl-01-frame-features.csv"
PLOT_PATH = PROJECT_ROOT / "outputs" / "plots" / "adl-01-feature-timeline.png"


def mean_point(frame_data, names):
    """Return the average pixel position for named landmarks."""
    points = frame_data[frame_data["landmark_name"].isin(names)]

    if len(points) != len(names):
        return None

    return (
        points["x_pixel"].mean(),
        points["y_pixel"].mean(),
    )


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input data not found:\n{INPUT_CSV}")

    poses = pd.read_csv(INPUT_CSV)
    all_frames = pd.DataFrame(
        {
            "frame_number": range(1, 151),
            "time_ms": [(frame - 1) * 1000 / 29.97 for frame in range(1, 151)],
        }
    )

    feature_rows = []

    for _, frame in all_frames.iterrows():
        frame_number = int(frame["frame_number"])
        frame_data = poses[poses["frame_number"] == frame_number]

        row = {
            "frame_number": frame_number,
            "time_ms": frame["time_ms"],
            "pose_detected": not frame_data.empty,
            "landmark_count": len(frame_data),
            "valid_depth_landmarks": int(frame_data["depth_mm"].notna().sum()),
            "median_depth_mm": frame_data["depth_mm"].median(),
            "body_width_px": np.nan,
            "body_height_px": np.nan,
            "torso_tilt_from_vertical_deg": np.nan,
        }

        if not frame_data.empty:
            shoulder = mean_point(frame_data, ["left_shoulder", "right_shoulder"])
            hip = mean_point(frame_data, ["left_hip", "right_hip"])

            x_span = frame_data["x_pixel"].max() - frame_data["x_pixel"].min()
            row["body_width_px"] = x_span

            if shoulder is not None and hip is not None:
                shoulder_x, shoulder_y = shoulder
                hip_x, hip_y = hip

                # 0° means upright; 90° means torso is horizontal in the image.
                row["torso_tilt_from_vertical_deg"] = np.degrees(
                    np.arctan2(abs(hip_x - shoulder_x), abs(hip_y - shoulder_y))
                )

            ankle = mean_point(frame_data, ["left_ankle", "right_ankle"])
            if shoulder is not None and ankle is not None:
                row["body_height_px"] = abs(ankle[1] - shoulder[1])

        feature_rows.append(row)

    features = pd.DataFrame(feature_rows)

    FEATURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_CSV, index=False)

    time_seconds = features["time_ms"] / 1000

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(time_seconds, features["torso_tilt_from_vertical_deg"], marker="o")
    axes[0].set_ylabel("Torso tilt (degrees)")
    axes[0].set_title("adl-01: pose and depth feature timeline")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_seconds, features["body_height_px"], marker="o", color="tab:green")
    axes[1].set_ylabel("Body height (pixels)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_seconds, features["median_depth_mm"], marker="o", color="tab:orange")
    axes[2].set_ylabel("Median depth (mm)")
    axes[2].set_xlabel("Time from sequence start (seconds)")
    axes[2].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=150)
    plt.close(figure)

    print(f"Frames represented: {len(features)}")
    print(f"Frames with a detected pose: {features['pose_detected'].sum()}")
    print(f"Saved frame features to:\n{FEATURE_CSV}")
    print(f"Saved timeline plot to:\n{PLOT_PATH}")


if __name__ == "__main__":
    main()