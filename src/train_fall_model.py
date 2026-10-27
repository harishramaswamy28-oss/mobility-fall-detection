"""Train and cross-validate a frame-level fall-transition classifier."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from annotation_features import FEATURE_COLUMNS, load_all_annotations
from logistic_model import fit_logistic, predict_fall_probability


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = ROOT / "data" / "urfd" / "annotations"
MODEL_PATH = ROOT / "models" / "fall_frame_classifier.pkl"
CV_OUTPUT = ROOT / "outputs" / "detections" / "fall-model-cross-validation.csv"
DECISION_CUTOFF = 0.5


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training features not found:\n{DATA_PATH}\n"
            "Run src/extract_all_sequence_features.py first."
        )
    data = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in data.columns if c not in IDENTIFIER_COLUMNS]
    if data["label"].nunique() != 2:
        raise ValueError("Training requires both fall and ADL sequence labels.")
    class_counts = data["label"].value_counts()
    folds = min(5, int(class_counts.min()))
    if folds < 2:
        raise ValueError("At least two examples from each class are required.")

    x = data[feature_columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y = data["label"].astype(int)
    y_array = y.to_numpy()

    # Stratify by recording class so every fold contains falls and ADLs.
    rng = np.random.default_rng(42)
    fold_indices = [[] for _ in range(folds)]
    for label in (0, 1):
        class_indices = np.flatnonzero(y_array == label)
        rng.shuffle(class_indices)
        for index, sample_index in enumerate(class_indices):
            fold_indices[index % folds].append(int(sample_index))

    predictions = np.zeros(len(y_array), dtype=int)
    for validation_indices in fold_indices:
        validation_indices = np.asarray(sorted(validation_indices), dtype=int)
        training_indices = np.setdiff1d(np.arange(len(y_array)), validation_indices)
        fold_model = fit_logistic(x[training_indices], y_array[training_indices])
        fold_probabilities = predict_fall_probability(fold_model, x[validation_indices])
        predictions[validation_indices] = (fold_probabilities >= 0.5).astype(int)

    true_positive = int(np.sum((y_array == 1) & (predictions == 1)))
    true_negative = int(np.sum((y_array == 0) & (predictions == 0)))
    false_positive = int(np.sum((y_array == 0) & (predictions == 1)))
    false_negative = int(np.sum((y_array == 1) & (predictions == 0)))
    accuracy = (true_positive + true_negative) / len(y_array)
    recall = true_positive / max(1, true_positive + false_negative)
    precision = true_positive / max(1, true_positive + false_positive)
    specificity = true_negative / max(1, true_negative + false_positive)
    balanced_accuracy = (recall + specificity) / 2
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    print(f"Sequences used: {len(data)} ({int(y.sum())} falls, {int((y == 0).sum())} ADLs)")
    print(f"Stratified recording-level CV folds: {folds}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")
    print(f"Fall precision: {precision:.3f}")
    print(f"Fall recall: {recall:.3f}")
    print(f"Fall F1: {f1:.3f}")
    print(f"Confusion matrix (true ADL/fall rows, predicted ADL/fall cols): [[{true_negative}, {false_positive}], [{false_negative}, {true_positive}]]")

    CV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sequence": data["sequence"],
        "true_label": y,
        "cross_validated_prediction": predictions,
        "correct": predictions == y,
    }).to_csv(CV_OUTPUT, index=False)

    final_model = fit_logistic(x, y_array)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(
            {
                "model": final_model,
                "feature_columns": feature_columns,
                "training_sequences": data["sequence"].tolist(),
                "target": "sequence-level fall vs ADL classification",
                "identity_note": (
                    "URFD sequence metadata has no participant IDs; these metrics do not "
                    "establish performance on unseen individuals."
                ),
            },
            model_file,
        )
    print(f"Cross-validation predictions saved to:\n{CV_OUTPUT}")
    print(f"Final model trained on all available sequences and saved to:\n{MODEL_PATH}")
    print("Validation is recording-level; URFD participant identities are unavailable.")


def train_from_frame_annotations():
    """Train on labeled frames and validate with entire recordings held out."""
    if not ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(f"URFD annotation folder not found:\n{ANNOTATIONS_DIR}")
    data = load_all_annotations(ANNOTATIONS_DIR)
    if data["target"].nunique() != 2:
        raise ValueError("The annotation tables must contain both target classes.")

    groups = data[["sequence"]].drop_duplicates().copy()
    groups["sequence_class"] = groups["sequence"].str.startswith("fall-").astype(int)
    group_counts = groups["sequence_class"].value_counts()
    folds = min(5, int(group_counts.min()))
    if folds < 2:
        raise ValueError("At least two recordings from each class are required.")

    rng = np.random.default_rng(42)
    fold_sequences = [[] for _ in range(folds)]
    for label in (0, 1):
        sequence_names = groups.loc[
            groups["sequence_class"] == label, "sequence"
        ].to_numpy()
        rng.shuffle(sequence_names)
        for index, name in enumerate(sequence_names):
            fold_sequences[index % folds].append(name)

    x = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = data["target"].to_numpy(dtype=int)
    probabilities = np.zeros(len(data), dtype=float)
    for held_out_sequences in fold_sequences:
        validation_mask = data["sequence"].isin(held_out_sequences).to_numpy()
        training_mask = ~validation_mask
        fold_model = fit_logistic(x[training_mask], y[training_mask])
        probabilities[validation_mask] = predict_fall_probability(
            fold_model, x[validation_mask]
        )

    predictions = (probabilities >= DECISION_CUTOFF).astype(int)
    tp = int(np.sum((y == 1) & (predictions == 1)))
    tn = int(np.sum((y == 0) & (predictions == 0)))
    fp = int(np.sum((y == 0) & (predictions == 1)))
    fn = int(np.sum((y == 1) & (predictions == 0)))
    accuracy = (tp + tn) / len(y)
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = (recall + specificity) / 2
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    fall_rows = data["sequence"].str.startswith("fall-")
    adl_rows = ~fall_rows
    fall_sequence_hits = sum(
        bool(((group["target"] == 1) & (predictions[group.index] == 1)).any())
        for _, group in data[fall_rows].groupby("sequence")
    )
    adl_sequence_false_alarms = sum(
        bool((predictions[group.index] == 1).any())
        for _, group in data[adl_rows].groupby("sequence")
    )

    print(f"Labeled frames: {len(data)}")
    print(f"Fall-positive frames: {int(y.sum())}; normal frames: {int((y == 0).sum())}")
    print(f"Recordings: {groups['sequence_class'].value_counts().to_dict()} (0=ADL, 1=fall)")
    print(f"Recording-held-out folds: {folds}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")
    print(f"Fall precision: {precision:.3f}")
    print(f"Fall recall: {recall:.3f}")
    print(f"Fall F1: {f1:.3f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: [[{tn}, {fp}], [{fn}, {tp}]]")
    print(f"Fall recordings with at least one transition frame detected: {fall_sequence_hits}/{groups['sequence_class'].sum()}")
    print(f"ADL recordings with any predicted fall frame: {adl_sequence_false_alarms}/{int((groups['sequence_class'] == 0).sum())}")

    CV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sequence": data["sequence"],
        "frame_number": data["frame_number"],
        "target": y,
        "fall_probability": probabilities,
        "prediction": predictions,
        "correct": predictions == y,
    }).to_csv(CV_OUTPUT, index=False)

    final_model = fit_logistic(x, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump({
            "model": final_model,
            "feature_columns": FEATURE_COLUMNS,
            "decision_cutoff": DECISION_CUTOFF,
                "target": "fall-transition frame versus non-fall frame",
            "training_sequences": groups["sequence"].tolist(),
            "identity_note": (
                "Validation holds out recordings. The files do not map recordings to actors, "
                "so unseen-person performance is not established."
            ),
        }, model_file)
    print(f"Cross-validation frame predictions saved to:\n{CV_OUTPUT}")
    print(f"Frame-level model saved to:\n{MODEL_PATH}")
    print("Frame features are depth-derived; live deployment still needs matching feature extraction.")


if __name__ == "__main__":
    train_from_frame_annotations()
