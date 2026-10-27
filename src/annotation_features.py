"""Read URFD's official frame annotations and make scale-normalized inputs."""

from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "sequence",
    "frame_number",
    "posture_label",
    "height_width_ratio",
    "major_minor_ratio",
    "bbox_occupancy",
    "max_std_xz_mm",
    "height_height_max_ratio",
    "person_height_mm",
    "floor_distance_mm",
    "p40",
]

BASE_FEATURE_COLUMNS = [
    "height_width_ratio",
    "major_minor_ratio",
    "bbox_occupancy",
    "max_std_xz_over_height",
    "height_height_max_ratio",
    "floor_distance_over_height",
    "p40",
]
CHANGE_WINDOWS_FRAMES = (10, 30)
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [
    f"delta_{window}f_{feature}"
    for window in CHANGE_WINDOWS_FRAMES
    for feature in BASE_FEATURE_COLUMNS
]


def load_annotation_file(path):
    """Load a headerless official URFD feature/label file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"URFD annotation file not found:\n{path}")
    raw = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    raw["sequence"] = raw["sequence"].astype(str)
    raw["frame_number"] = pd.to_numeric(raw["frame_number"], errors="coerce")
    raw["posture_label"] = pd.to_numeric(raw["posture_label"], errors="coerce")
    raw = raw.sort_values(["sequence", "frame_number"]).reset_index(drop=True)
    height = pd.to_numeric(raw["person_height_mm"], errors="coerce").replace(0, float("nan"))

    features = pd.DataFrame(index=raw.index)
    features["height_width_ratio"] = pd.to_numeric(raw["height_width_ratio"], errors="coerce")
    features["major_minor_ratio"] = pd.to_numeric(raw["major_minor_ratio"], errors="coerce")
    features["bbox_occupancy"] = pd.to_numeric(raw["bbox_occupancy"], errors="coerce")
    features["max_std_xz_over_height"] = pd.to_numeric(
        raw["max_std_xz_mm"], errors="coerce"
    ) / height
    features["height_height_max_ratio"] = pd.to_numeric(
        raw["height_height_max_ratio"], errors="coerce"
    )
    features["floor_distance_over_height"] = pd.to_numeric(
        raw["floor_distance_mm"], errors="coerce"
    ) / height
    features["p40"] = pd.to_numeric(raw["p40"], errors="coerce")

    # The frame label means posture (-1 not lying, 0 transitioning/falling,
    # 1 lying). Train the event model on transition frames in fall sequences;
    # lying frames alone resemble intentional ADL lying and are not event starts.
    sequence_is_fall = raw["sequence"].str.startswith("fall-")
    target = (sequence_is_fall & raw["posture_label"].eq(0)).astype(int)

    frame_numbers = raw["frame_number"].to_numpy(dtype=float)
    base_values = features[BASE_FEATURE_COLUMNS].to_numpy(dtype=float)
    temporal = pd.DataFrame(index=raw.index)
    for window in CHANGE_WINDOWS_FRAMES:
        deltas = [[float("nan")] * len(BASE_FEATURE_COLUMNS) for _ in range(len(raw))]
        for positions in raw.groupby("sequence", sort=False).indices.values():
            group_frames = frame_numbers[positions]
            for local_index, global_index in enumerate(positions):
                desired_past_frame = group_frames[local_index] - window
                past_index = int(
                    np.searchsorted(group_frames, desired_past_frame, side="right")
                ) - 1
                if past_index < 0:
                    continue
                frame_gap = group_frames[local_index] - group_frames[past_index]
                if frame_gap <= 0 or frame_gap > window * 2:
                    continue
                past_global_index = positions[past_index]
                deltas[global_index] = (
                    (base_values[global_index] - base_values[past_global_index])
                    * (window / frame_gap)
                ).tolist()
        delta_frame = pd.DataFrame(
            deltas,
            columns=[f"delta_{window}f_{feature}" for feature in BASE_FEATURE_COLUMNS],
        )
        temporal = pd.concat([temporal, delta_frame], axis=1)

    features = pd.concat([features, temporal], axis=1)
    table = pd.concat([raw[["sequence", "frame_number", "posture_label"]], features], axis=1)
    table["target"] = target
    table = table.dropna(subset=["frame_number", "posture_label"]).reset_index(drop=True)
    table["frame_number"] = table["frame_number"].astype(int)
    table["target"] = table["target"].astype(int)
    return table


def load_all_annotations(annotations_dir):
    """Combine the fall and ADL tables as one frame-level training dataset."""
    annotations_dir = Path(annotations_dir)
    fall = load_annotation_file(annotations_dir / "urfall-cam0-falls.csv")
    adl = load_annotation_file(annotations_dir / "urfall-cam0-adls.csv")
    return pd.concat([fall, adl], ignore_index=True)
