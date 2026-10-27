"""Compare linear logistic and nonlinear RTrees models on held-out recordings."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from annotation_features import FEATURE_COLUMNS, load_all_annotations
from logistic_model import fit_logistic, predict_fall_probability


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = ROOT / "data" / "urfd" / "annotations"
OUTPUT_CSV = ROOT / "outputs" / "detections" / "fall-model-comparison-cv.csv"
FOLDS = 5
RANDOM_SEED = 42
TREE_COUNT = 100
DECISION_CUTOFF = 0.5


def make_forest():
    model = cv2.ml.RTrees_create()
    model.setMaxDepth(12)
    model.setMinSampleCount(25)
    model.setRegressionAccuracy(0.0)
    model.setUseSurrogates(False)
    model.setMaxCategories(2)
    model.setCalculateVarImportance(True)
    model.setActiveVarCount(0)
    model.setPriors(np.asarray([0.5, 0.5], dtype=np.float32))
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, TREE_COUNT, 0.0))
    return model


def forest_probabilities(model, x):
    votes = model.getVotes(np.asarray(x, dtype=np.float32), 0)
    class_ids = votes[0].astype(int)
    fall_column = int(np.flatnonzero(class_ids == 1)[0])
    class_votes = votes[1:].astype(float)
    totals = np.maximum(class_votes.sum(axis=1), 1.0)
    return class_votes[:, fall_column] / totals


def metrics(y, probabilities):
    prediction = probabilities >= DECISION_CUTOFF
    tp = int(np.sum((y == 1) & prediction))
    tn = int(np.sum((y == 0) & ~prediction))
    fp = int(np.sum((y == 0) & prediction))
    fn = int(np.sum((y == 1) & ~prediction))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    data = load_all_annotations(ANNOTATIONS_DIR)
    x = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = data["target"].to_numpy(dtype=int)

    groups = data[["sequence"]].drop_duplicates().copy()
    groups["sequence_class"] = groups["sequence"].str.startswith("fall-").astype(int)
    folds = min(FOLDS, int(groups["sequence_class"].value_counts().min()))
    rng = np.random.default_rng(RANDOM_SEED)
    fold_sequences = [[] for _ in range(folds)]
    for label in (0, 1):
        sequence_names = groups.loc[
            groups["sequence_class"] == label, "sequence"
        ].to_numpy()
        rng.shuffle(sequence_names)
        for index, name in enumerate(sequence_names):
            fold_sequences[index % folds].append(name)

    logistic_oof = np.zeros(len(data), dtype=float)
    forest_oof = np.zeros(len(data), dtype=float)
    for fold_number, held_out_sequences in enumerate(fold_sequences, start=1):
        validation_mask = data["sequence"].isin(held_out_sequences).to_numpy()
        training_mask = ~validation_mask
        x_train, x_valid = x[training_mask], x[validation_mask]
        y_train = y[training_mask]

        logistic = fit_logistic(x_train, y_train)
        logistic_oof[validation_mask] = predict_fall_probability(logistic, x_valid)

        forest = make_forest()
        forest.train(
            np.asarray(x_train, dtype=np.float32),
            cv2.ml.ROW_SAMPLE,
            np.asarray(y_train, dtype=np.int32).reshape(-1, 1),
        )
        forest_oof[validation_mask] = forest_probabilities(forest, x_valid)
        print(f"Completed fold {fold_number}/{folds}", flush=True)

    for name, probabilities in [
        ("Logistic regression", logistic_oof),
        ("Balanced RTrees", forest_oof),
    ]:
        result = metrics(y, probabilities)
        print(
            f"{name}: accuracy={result['accuracy']:.3f}, "
            f"balanced_accuracy={result['balanced_accuracy']:.3f}, "
            f"precision={result['precision']:.3f}, recall={result['recall']:.3f}, "
            f"F1={result['f1']:.3f}, confusion=[[{result['tn']}, {result['fp']}], "
            f"[{result['fn']}, {result['tp']}]]"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sequence": data["sequence"],
            "frame_number": data["frame_number"],
            "target": y,
            "logistic_probability": logistic_oof,
            "logistic_prediction": logistic_oof >= DECISION_CUTOFF,
            "forest_probability": forest_oof,
            "forest_prediction": forest_oof >= DECISION_CUTOFF,
        }
    ).to_csv(OUTPUT_CSV, index=False)
    print(f"Comparison predictions saved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
