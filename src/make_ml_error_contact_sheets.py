"""Render RGB examples for the largest recording-held-out model errors."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RGB_ROOT = ROOT / "data" / "urfd" / "raw" / "UR_fall_detection_dataset_cam0_rgb"
PREDICTIONS = ROOT / "outputs" / "detections" / "fall-model-cross-validation.csv"
OUTPUT_DIR = ROOT / "outputs" / "diagnostics"

TILE_WIDTH = 320
TILE_HEIGHT = 240
FRAMES_PER_SEQUENCE = 4
SEQUENCES_PER_SHEET = 5
MIN_FRAME_GAP = 5


def choose_examples(rows, error_column, probability_ascending=False):
    counts = rows.groupby("sequence")[error_column].sum().sort_values(ascending=False)
    selected_sequences = counts[counts > 0].head(SEQUENCES_PER_SHEET).index
    examples = []

    for sequence in selected_sequences:
        group = rows[(rows["sequence"] == sequence) & rows[error_column]].copy()
        group = group.sort_values("fall_probability", ascending=probability_ascending)
        chosen = []
        for _, row in group.iterrows():
            frame = int(row["frame_number"])
            if all(abs(frame - int(item["frame_number"])) >= MIN_FRAME_GAP for item in chosen):
                chosen.append(row)
            if len(chosen) == FRAMES_PER_SEQUENCE:
                break
        examples.append((sequence, int(counts[sequence]), chosen))

    return examples


def render_sheet(examples, title, output_path):
    label_height = 32
    header_height = 34
    row_height = label_height + TILE_HEIGHT
    sheet = np.full(
        (header_height + max(1, len(examples)) * row_height, FRAMES_PER_SEQUENCE * TILE_WIDTH, 3),
        245,
        dtype=np.uint8,
    )

    for row_index, (sequence, error_count, selected_rows) in enumerate(examples):
        y = header_height + row_index * row_height
        cv2.putText(
            sheet,
            f"{sequence} | error frames: {error_count}",
            (8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        folder = RGB_ROOT / f"{sequence}-cam0-rgb"

        for column, row in enumerate(selected_rows):
            frame = int(row["frame_number"])
            image_path = folder / f"{sequence}-cam0-rgb-{frame:03d}.png"
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            tile = cv2.resize(image, (TILE_WIDTH, TILE_HEIGHT))
            score = float(row["fall_probability"])
            caption = f"frame {frame} | P(fall)={score:.2f}"
            cv2.rectangle(tile, (0, 0), (TILE_WIDTH, 30), (0, 0, 0), -1)
            cv2.putText(
                tile,
                caption,
                (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            x = column * TILE_WIDTH
            y_image = y + label_height
            sheet[y_image:y_image + TILE_HEIGHT, x:x + TILE_WIDTH] = tile

    cv2.putText(
        sheet,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    print(f"Saved: {output_path}")


def main():
    if not PREDICTIONS.exists():
        raise FileNotFoundError(f"Cross-validation predictions not found:\n{PREDICTIONS}")
    rows = pd.read_csv(PREDICTIONS)
    rows["is_adl"] = rows["sequence"].str.startswith("adl-")
    rows["false_positive"] = (rows["target"] == 0) & (rows["prediction"] == 1) & rows["is_adl"]
    rows["missed_fall"] = (rows["target"] == 1) & (rows["prediction"] == 0)

    fp_examples = choose_examples(rows, "false_positive")
    fn_examples = choose_examples(rows, "missed_fall", probability_ascending=True)
    render_sheet(
        fp_examples,
        "Largest ADL false positives (out-of-fold predictions)",
        OUTPUT_DIR / "ml-false-positive-contact-sheet.jpg",
    )
    render_sheet(
        fn_examples,
        "Largest missed-fall frame groups (out-of-fold predictions)",
        OUTPUT_DIR / "ml-missed-fall-contact-sheet.jpg",
    )


if __name__ == "__main__":
    main()
