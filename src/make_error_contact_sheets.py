from pathlib import Path
import csv

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RGB_ROOT = ROOT / "data" / "urfd" / "raw" / "UR_fall_detection_dataset_cam0_rgb"
SUMMARY = ROOT / "outputs" / "detections" / "all-sequences-rgb-summary.csv"
OUTPUT = ROOT / "outputs" / "diagnostics" / "misclassified-sequences.jpg"

TILE_W = 320
TILE_H = 180
LABEL_H = 30
FRAMES_PER_SEQUENCE = 8


def main():
    with SUMMARY.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    errors = [
        row for row in rows
        if (row["class"] == "fall" and row["fall_candidate"] == "False")
        or (row["class"] == "adl" and row["fall_candidate"] == "True")
    ]

    if not errors:
        print("No misclassified sequences found.")
        return

    sheet = np.full(
        (len(errors) * (TILE_H + LABEL_H), FRAMES_PER_SEQUENCE * TILE_W, 3),
        245,
        dtype=np.uint8,
    )

    for row_index, row in enumerate(errors):
        folder = RGB_ROOT / row["sequence"]
        image_paths = sorted(folder.glob("*.png"))
        y_base = row_index * (TILE_H + LABEL_H)

        cv2.putText(
            sheet,
            f"{row['sequence']}  |  actual: {row['class']}  |  flagged: {row['fall_candidate']}",
            (8, y_base + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        if not image_paths:
            continue

        indexes = np.linspace(
            0, len(image_paths) - 1, FRAMES_PER_SEQUENCE, dtype=int
        )

        for col, index in enumerate(indexes):
            image = cv2.imread(str(image_paths[index]))
            if image is None:
                continue

            tile = cv2.resize(image, (TILE_W, TILE_H))
            frame_number = index + 1
            cv2.putText(
                tile,
                f"frame {frame_number}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                f"frame {frame_number}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            x_base = col * TILE_W
            y_image = y_base + LABEL_H
            sheet[y_image:y_image + TILE_H, x_base:x_base + TILE_W] = tile

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), sheet)
    print(f"Misclassified sequences: {len(errors)}")
    print(f"Saved contact sheet to:\n{OUTPUT}")


if __name__ == "__main__":
    main()