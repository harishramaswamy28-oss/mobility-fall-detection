from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from annotation_features import load_annotation_file
from logistic_model import predict_fall_probability


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_SECONDS = 2.0
LOOKBACK_SECONDS = 1.0

# Preliminary thresholds, based only on our two inspected sequences.
TILT_DEGREES = 60.0
HEIGHT_RATIO = 0.75
TILT_CHANGE_DEGREES = 45.0
DEPTH_CHANGE_MM = 500.0
MIN_CONSECUTIVE_FRAMES = 3


def add_one_second_changes(features):
    """Compare every valid pose frame with its closest earlier frame ~1 s ago."""
    times = features["time_ms"].to_numpy() / 1000
    tilt = features["torso_tilt_from_vertical_deg"].to_numpy()
    depth = features["median_depth_mm"].to_numpy()

    tilt_change = np.full(len(features), np.nan)
    depth_change = np.full(len(features), np.nan)

    for index, current_time in enumerate(times):
        earlier_indexes = np.where(times <= current_time - LOOKBACK_SECONDS)[0]

        if len(earlier_indexes) == 0:
            continue

        earlier_index = earlier_indexes[-1]

        if not np.isnan(tilt[index]) and not np.isnan(tilt[earlier_index]):
            tilt_change[index] = tilt[index] - tilt[earlier_index]

        if not np.isnan(depth[index]) and not np.isnan(depth[earlier_index]):
            depth_change[index] = depth[index] - depth[earlier_index]

    features["tilt_change_1s_deg"] = tilt_change
    features["depth_change_1s_mm"] = depth_change
    return features


def keep_consecutive_candidates(features):
    """Keep candidates that remain true for several consecutive frames."""
    confirmed = np.zeros(len(features), dtype=bool)
    run_start = None

    for index, is_candidate in enumerate(features["raw_candidate"]):
        if is_candidate and run_start is None:
            run_start = index

        if (not is_candidate or index == len(features) - 1) and run_start is not None:
            run_end = index if is_candidate else index - 1
            run_length = run_end - run_start + 1

            if run_length >= MIN_CONSECUTIVE_FRAMES:
                confirmed[run_start:run_end + 1] = True

            run_start = None

    features["fall_candidate"] = confirmed
    return features


def _legacy_threshold_main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python .\\src\\detect_fall_event.py fall-01"
        )

    sequence_name = sys.argv[1]
    input_csv = (
        PROJECT_ROOT / "outputs" / "features"
        / f"{sequence_name}-frame-features.csv"
    )
    output_csv = (
        PROJECT_ROOT / "outputs" / "detections"
        / f"{sequence_name}-heuristic-analysis.csv"
    )
    plot_path = (
        PROJECT_ROOT / "outputs" / "plots"
        / f"{sequence_name}-heuristic-analysis.png"
    )

    if not input_csv.exists():
        raise FileNotFoundError(f"Feature file not found:\n{input_csv}")

    features = pd.read_csv(input_csv)
    valid_baseline = features[
        (features["time_ms"] / 1000 <= BASELINE_SECONDS)
        & features["body_height_px"].notna()
    ]

    if valid_baseline.empty:
        raise RuntimeError("No valid pose frames were available for the baseline.")

    baseline_height = valid_baseline["body_height_px"].median()
    features["height_ratio_to_baseline"] = (
        features["body_height_px"] / baseline_height
    )

    features = add_one_second_changes(features)

    features["raw_candidate"] = (
        (features["torso_tilt_from_vertical_deg"] >= TILT_DEGREES)
        & (features["height_ratio_to_baseline"] <= HEIGHT_RATIO)
        & (features["tilt_change_1s_deg"] >= TILT_CHANGE_DEGREES)
        & (features["depth_change_1s_mm"].abs() >= DEPTH_CHANGE_MM)
    ).fillna(False)

    features = keep_consecutive_candidates(features)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)

    time_seconds = features["time_ms"] / 1000
    candidates = features[features["fall_candidate"]]

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(
        time_seconds,
        features["torso_tilt_from_vertical_deg"],
        label="Torso tilt",
    )
    axes[0].axhline(TILT_DEGREES, color="red", linestyle="--", label="Tilt threshold")
    axes[0].set_ylabel("Tilt (degrees)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time_seconds,
        features["height_ratio_to_baseline"],
        label="Height ratio",
        color="tab:green",
    )
    axes[1].axhline(
        HEIGHT_RATIO,
        color="red",
        linestyle="--",
        label="Height threshold",
    )

    if not candidates.empty:
        axes[1].scatter(
            candidates["time_ms"] / 1000,
            candidates["height_ratio_to_baseline"],
            color="red",
            label="Confirmed fall candidate",
            zorder=3,
        )

    axes[1].set_ylabel("Height / baseline")
    axes[1].set_xlabel("Time from sequence start (seconds)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.suptitle(f"{sequence_name}: preliminary multi-cue fall heuristic")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print(f"Baseline body height: {baseline_height:.1f} pixels")
    print(f"Confirmed candidate frames: {len(candidates)}")

    if candidates.empty:
        print("No fall event candidate was found.")
    else:
        first = candidates.iloc[0]
        print(
            "First fall event candidate: "
            f"frame {int(first['frame_number'])}, "
            f"time {first['time_ms'] / 1000:.2f} s"
        )

    print(f"Saved analysis to:\n{output_csv}")
    print(f"Saved plot to:\n{plot_path}")


def _sequence_level_main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python .\\src\\detect_fall_event.py <sequence-folder-name>\n"
            "Example: python .\\src\\detect_fall_event.py fall-01-cam0-rgb"
        )

    sequence = sys.argv[1]
    features_path = PROJECT_ROOT / "outputs" / "features" / "all-sequence-ml-features.csv"
    model_path = PROJECT_ROOT / "models" / "fall_frame_classifier.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{model_path}\n"
            "Run src/train_fall_model.py first."
        )
    if not features_path.exists():
        raise FileNotFoundError(
            f"Sequence features not found:\n{features_path}\n"
            "Run src/extract_all_sequence_features.py first."
        )

    rows = pd.read_csv(features_path)
    selected = rows[rows["sequence"] == sequence]
    if selected.empty:
        raise KeyError(f"Sequence {sequence!r} is not in {features_path}")

    with model_path.open("rb") as model_file:
        bundle = pickle.load(model_file)
    x = selected[bundle["feature_columns"]].replace([np.inf, -np.inf], np.nan)
    model = bundle["model"]
    fall_probability = float(
        predict_fall_probability(model, x.to_numpy(dtype=float))[0]
    )
    prediction = int(fall_probability >= 0.5)

    print(f"Sequence: {sequence}")
    print(f"Model classification: {'FALL sequence' if prediction else 'ADL sequence'}")
    print(f"Estimated fall probability: {fall_probability:.1%}")
    print("This is a whole-sequence classification, not a real-time event alarm.")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python .\\src\\detect_fall_event.py <sequence-id>\n"
            "Examples: fall-01 or adl-01"
        )

    sequence = sys.argv[1]
    if not (sequence.startswith("fall-") or sequence.startswith("adl-")):
        raise ValueError("Sequence ID must begin with 'fall-' or 'adl-'.")

    model_path = PROJECT_ROOT / "models" / "fall_frame_classifier.pkl"
    annotation_name = (
        "urfall-cam0-falls.csv" if sequence.startswith("fall-")
        else "urfall-cam0-adls.csv"
    )
    annotation_path = PROJECT_ROOT / "data" / "urfd" / "annotations" / annotation_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{model_path}\nRun src/train_fall_model.py first."
        )

    rows = load_annotation_file(annotation_path)
    selected = rows[rows["sequence"] == sequence].copy()
    if selected.empty:
        raise KeyError(f"Sequence {sequence!r} was not found in {annotation_path}")

    with model_path.open("rb") as model_file:
        bundle = pickle.load(model_file)
    x = selected[bundle["feature_columns"]].to_numpy(dtype=float)
    selected["fall_probability"] = predict_fall_probability(bundle["model"], x)
    selected["predicted_fall_frame"] = (
        selected["fall_probability"] >= bundle["decision_cutoff"]
    )

    output_path = (
        PROJECT_ROOT / "outputs" / "detections"
        / f"{sequence}-ml-frame-predictions.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected[
        ["sequence", "frame_number", "posture_label", "target", "fall_probability", "predicted_fall_frame"]
    ].to_csv(output_path, index=False)

    positive = selected[selected["predicted_fall_frame"]]
    print(f"Sequence: {sequence}")
    print(f"Frames scored: {len(selected)}")
    print(f"Frames classified as fall-related: {len(positive)}")
    if positive.empty:
        print("No fall-related frames predicted.")
    else:
        print(
            f"Predicted fall-related frame range: {int(positive['frame_number'].min())}"
            f"–{int(positive['frame_number'].max())}"
        )
        print(f"Peak fall probability: {selected['fall_probability'].max():.1%}")
    print(f"Saved per-frame predictions to:\n{output_path}")
    print("Use cross-validation results for performance; this sequence output uses a model trained on all clips.")


if __name__ == "__main__":
    main()
