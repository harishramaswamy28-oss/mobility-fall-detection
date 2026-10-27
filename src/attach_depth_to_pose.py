from pathlib import Path
import csv

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POSE_CSV = PROJECT_ROOT / "outputs" / "poses" / "adl-01-pose-landmarks.csv"
DEPTH_FOLDER = PROJECT_ROOT / "data" / "urfd" / "depth_sample" / "adl-01-cam0-d"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "poses" / "adl-01-pose-with-depth.csv"

# Official URFD scale for fall sequences from camera 0.
DEPTH_SCALE_MM = 7000.0
SAMPLE_RADIUS = 2


def sample_depth(depth_image, x_normalized, y_normalized):
    """Return a robust local depth estimate around one 2D landmark."""
    height, width = depth_image.shape

    x_pixel = max(0, min(width - 1, round(x_normalized * (width - 1))))
    y_pixel = max(0, min(height - 1, round(y_normalized * (height - 1))))

    x_start = max(0, x_pixel - SAMPLE_RADIUS)
    x_end = min(width, x_pixel + SAMPLE_RADIUS + 1)
    y_start = max(0, y_pixel - SAMPLE_RADIUS)
    y_end = min(height, y_pixel + SAMPLE_RADIUS + 1)

    patch = depth_image[y_start:y_end, x_start:x_end]
    valid_values = patch[patch > 0]

    if valid_values.size == 0:
        return x_pixel, y_pixel, None, None

    raw_depth = float(np.median(valid_values))
    depth_mm = DEPTH_SCALE_MM * raw_depth / 65535.0
    return x_pixel, y_pixel, raw_depth, depth_mm


def main():
    if not POSE_CSV.exists():
        raise FileNotFoundError(f"Pose CSV not found:\n{POSE_CSV}")

    if not DEPTH_FOLDER.exists():
        raise FileNotFoundError(f"Depth folder not found:\n{DEPTH_FOLDER}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    valid_depth_rows = 0
    current_frame = None
    current_depth_image = None

    with POSE_CSV.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames + [
            "x_pixel",
            "y_pixel",
            "depth_raw_uint16",
            "depth_mm",
        ]

        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                frame_number = int(row["frame_number"])

                if frame_number != current_frame:
                    depth_path = DEPTH_FOLDER / f"adl-01-cam0-d-{frame_number:03d}.png"
                    current_depth_image = cv2.imread(
                        str(depth_path),
                        cv2.IMREAD_UNCHANGED,
                    )

                    if current_depth_image is None:
                        raise RuntimeError(f"Could not read depth frame:\n{depth_path}")

                    current_frame = frame_number

                x_pixel, y_pixel, raw_depth, depth_mm = sample_depth(
                    current_depth_image,
                    float(row["x_normalized"]),
                    float(row["y_normalized"]),
                )

                row["x_pixel"] = x_pixel
                row["y_pixel"] = y_pixel
                row["depth_raw_uint16"] = raw_depth
                row["depth_mm"] = depth_mm

                writer.writerow(row)
                rows_written += 1

                if depth_mm is not None:
                    valid_depth_rows += 1

    print(f"Landmark rows written: {rows_written}")
    print(f"Rows with valid Kinect depth: {valid_depth_rows}")
    print(f"Saved depth-enriched pose data to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()