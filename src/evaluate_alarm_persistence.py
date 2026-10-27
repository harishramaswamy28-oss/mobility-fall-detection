"""Check how short temporal confirmation affects sequence-level alarms."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "outputs" / "detections" / "classifier-benchmark-oof.csv"
OUTPUT = ROOT / "outputs" / "detections" / "alarm-persistence-summary.csv"
PERSISTENCE_ROWS = (1, 2, 3, 5, 8, 10, 12, 15, 20, 25)
SCORE_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
MAX_FRAME_GAP = 3

SCORE_SUFFIX = "_score"


def positive_runs(group, score_column, threshold):
    ordered = group.sort_values("frame_number")
    is_positive = ordered[score_column].to_numpy() >= threshold
    frame_numbers = ordered["frame_number"].to_numpy(dtype=int)
    targets = ordered["target"].to_numpy(dtype=int)
    runs = []
    current = []
    previous_frame = None

    for index, (frame, positive) in enumerate(zip(frame_numbers, is_positive)):
        if positive and (previous_frame is None or frame - previous_frame <= MAX_FRAME_GAP):
            current.append(index)
        else:
            if current:
                runs.append(current)
            current = [index] if positive else []
        previous_frame = frame if positive else None
    if current:
        runs.append(current)

    return [(len(run), bool(targets[run].any())) for run in runs]


def main():
    if not PREDICTIONS.exists():
        raise FileNotFoundError(f"Benchmark predictions not found:\n{PREDICTIONS}")
    data = pd.read_csv(PREDICTIONS)
    records = []

    score_columns = [column for column in data if column.endswith(SCORE_SUFFIX)]
    for score_column in score_columns:
        for threshold in SCORE_THRESHOLDS:
            for persistence in PERSISTENCE_ROWS:
                fall_hits = 0
                adl_alarms = 0
                for sequence, group in data.groupby("sequence"):
                    runs = positive_runs(group, score_column, threshold)
                    qualified = [
                        event_hit
                        for length, event_hit in runs
                        if length >= persistence
                    ]
                    if sequence.startswith("fall-") and any(qualified):
                        fall_hits += 1
                    if sequence.startswith("adl-") and qualified:
                        adl_alarms += 1
                records.append(
                    {
                        "model": score_column.removesuffix("_score"),
                        "score_threshold": threshold,
                        "minimum_positive_rows": persistence,
                        "fall_recordings_detected": fall_hits,
                        "fall_recordings_total": 30,
                        "adl_recordings_with_alarm": adl_alarms,
                        "adl_recordings_total": 40,
                    }
                )

    result = pd.DataFrame(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    best = result.sort_values(
        ["fall_recordings_detected", "adl_recordings_with_alarm"],
        ascending=[False, True],
    )
    print("Best fall-detection / ADL-alarm tradeoffs (screening only):")
    print(best.head(20).to_string(index=False))
    print(f"Saved event-level persistence summary to:\n{OUTPUT}")


if __name__ == "__main__":
    main()
