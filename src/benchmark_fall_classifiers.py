"""Screen several tabular classifiers with identical recording-held-out folds.

This is a model-screening experiment, not a final unbiased performance estimate.
All frames from one sequence stay in the same fold.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from annotation_features import FEATURE_COLUMNS, load_all_annotations
from logistic_model import fit_logistic, predict_fall_probability


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = ROOT / "data" / "urfd" / "annotations"
OUTPUT_CSV = ROOT / "outputs" / "detections" / "classifier-benchmark-oof.csv"
FOLDS = 5
SEED = 42
TREE_COUNT = 80
K_NEIGHBOR_SWEEP = (5, 7, 11, 15, 21, 31, 41, 51, 61, 71, 81, 101)


def make_folds(data):
    groups = data[["sequence"]].drop_duplicates().copy()
    groups["class"] = groups["sequence"].str.startswith("fall-").astype(int)
    fold_count = min(FOLDS, int(groups["class"].value_counts().min()))
    rng = np.random.default_rng(SEED)
    fold_sequences = [[] for _ in range(fold_count)]
    for label in (0, 1):
        names = groups.loc[groups["class"] == label, "sequence"].to_numpy()
        rng.shuffle(names)
        for index, name in enumerate(names):
            fold_sequences[index % fold_count].append(name)
    return fold_sequences


def balanced_rows(x, y, seed):
    rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(y == label) for label in (0, 1)]
    size = min(len(indices[0]), len(indices[1]))
    chosen = np.concatenate([rng.choice(group, size=size, replace=False) for group in indices])
    rng.shuffle(chosen)
    return x[chosen], y[chosen]


def standardize(x_train, x_valid):
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(x_train), x_train, medians)
    valid = np.where(np.isfinite(x_valid), x_valid, medians)
    means = train.mean(axis=0)
    scales = train.std(axis=0)
    scales[scales < 1e-8] = 1.0
    return ((train - means) / scales).astype(np.float32), ((valid - means) / scales).astype(np.float32)


def rtrees_model(x_train, y_train, x_valid):
    model = cv2.ml.RTrees_create()
    model.setMaxDepth(12)
    model.setMinSampleCount(25)
    model.setUseSurrogates(False)
    model.setMaxCategories(2)
    model.setPriors(np.asarray([0.5, 0.5], np.float32))
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, TREE_COUNT, 0.0))
    model.train(x_train, cv2.ml.ROW_SAMPLE, y_train.astype(np.int32).reshape(-1, 1))
    votes = model.getVotes(x_valid, 0)
    class_ids = votes[0].astype(int)
    fall_column = int(np.flatnonzero(class_ids == 1)[0])
    class_votes = votes[1:].astype(float)
    return class_votes[:, fall_column] / np.maximum(class_votes.sum(axis=1), 1.0)


def boost_model(x_train, y_train, x_valid):
    model = cv2.ml.Boost_create()
    model.setBoostType(cv2.ml.Boost_GENTLE)
    model.setWeakCount(100)
    model.setWeightTrimRate(0.95)
    model.setMaxDepth(2)
    model.setPriors(np.asarray([0.5, 0.5], np.float32))
    model.train(x_train, cv2.ml.ROW_SAMPLE, y_train.astype(np.int32).reshape(-1, 1))
    _, result = model.predict(x_valid)
    return result.reshape(-1).astype(float)


def svm_model(x_train, y_train, x_valid):
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setC(2.0)
    model.setGamma(1.0 / x_train.shape[1])
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 5000, 1e-5))
    model.train(x_train, cv2.ml.ROW_SAMPLE, y_train.astype(np.int32).reshape(-1, 1))
    _, result = model.predict(x_valid)
    return result.reshape(-1).astype(float)


def make_svm(kernel):
    """Return a decision classifier for a selected OpenCV SVM kernel."""
    def predict(x_train, y_train, x_valid):
        model = cv2.ml.SVM_create()
        model.setType(cv2.ml.SVM_C_SVC)
        model.setKernel(kernel)
        model.setC(2.0)
        model.setGamma(1.0 / x_train.shape[1])
        if kernel == cv2.ml.SVM_POLY:
            model.setDegree(3)
        model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 5000, 1e-5))
        model.train(x_train, cv2.ml.ROW_SAMPLE, y_train.astype(np.int32).reshape(-1, 1))
        _, result = model.predict(x_valid)
        return result.reshape(-1).astype(float)
    return predict


def decision_tree_model(x_train, y_train, x_valid):
    model = cv2.ml.DTrees_create()
    model.setMaxDepth(12)
    model.setMinSampleCount(25)
    model.setMaxCategories(2)
    model.setCVFolds(0)
    model.train(x_train, cv2.ml.ROW_SAMPLE, y_train.astype(np.int32).reshape(-1, 1))
    _, result = model.predict(x_valid)
    return result.reshape(-1).astype(float)


def make_boost(boost_type):
    def predict(x_train, y_train, x_valid):
        model = cv2.ml.Boost_create()
        model.setBoostType(boost_type)
        model.setWeakCount(100)
        model.setWeightTrimRate(0.95)
        model.setMaxDepth(2)
        model.setPriors(np.asarray([0.5, 0.5], np.float32))
        model.train(x_train, cv2.ml.ROW_SAMPLE, y_train.astype(np.int32).reshape(-1, 1))
        _, result = model.predict(x_valid)
        return result.reshape(-1).astype(float)
    return predict


def make_knn(k_neighbors):
    """Return a KNN classifier that outputs the fraction of fall neighbors."""
    def predict(x_train, y_train, x_valid):
        model = cv2.ml.KNearest_create()
        model.setIsClassifier(True)
        model.setDefaultK(k_neighbors)
        model.train(x_train, cv2.ml.ROW_SAMPLE, y_train.astype(np.float32).reshape(-1, 1))
        _, _, neighbor_responses, _ = model.findNearest(x_valid, k_neighbors)
        return np.mean(neighbor_responses == 1, axis=1)
    return predict


def bayes_model(x_train, y_train, x_valid):
    model = cv2.ml.NormalBayesClassifier_create()
    model.train(x_train, cv2.ml.ROW_SAMPLE, y_train.astype(np.int32).reshape(-1, 1))
    _, result = model.predict(x_valid)
    return result.reshape(-1).astype(float)


def ann_model(x_train, y_train, x_valid):
    model = cv2.ml.ANN_MLP_create()
    model.setLayerSizes(np.asarray([x_train.shape[1], 16, 2], np.int32))
    model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
    model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.01, 0.1)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 250, 1e-4))
    responses = np.zeros((len(y_train), 2), np.float32)
    responses[np.arange(len(y_train)), y_train.astype(int)] = 1.0
    model.train(x_train, cv2.ml.ROW_SAMPLE, responses)
    _, scores = model.predict(x_valid)
    scores = scores.astype(float)
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores[:, 1] / exp_scores.sum(axis=1)


def calc_metrics(data, scores):
    y = data["target"].to_numpy(dtype=int)
    pred = scores >= 0.5
    tp = int(np.sum((y == 1) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    fall_data = data[data["sequence"].str.startswith("fall-")]
    adl_data = data[data["sequence"].str.startswith("adl-")]
    fall_hits = sum(
        bool(((group["target"].to_numpy() == 1) & pred[group.index.to_numpy()]).any())
        for _, group in fall_data.groupby("sequence")
    )
    adl_alarms = sum(
        bool(pred[group.index.to_numpy()].any())
        for _, group in adl_data.groupby("sequence")
    )
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fall_recordings_hit": fall_hits,
        "adl_recordings_flagged": adl_alarms,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    data = load_all_annotations(ANNOTATIONS_DIR).reset_index(drop=True)
    x = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = data["target"].to_numpy(dtype=int)
    folds = make_folds(data)
    model_functions = {
        "Logistic": None,
        "RTrees": rtrees_model,
        "Boost": boost_model,
        "SVM_RBF": svm_model,
        "KNN": make_knn(11),
        "NormalBayes": bayes_model,
        "ANN_MLP": ann_model,
        "DecisionTree": decision_tree_model,
        "SVM_Linear": make_svm(cv2.ml.SVM_LINEAR),
        "SVM_Polynomial": make_svm(cv2.ml.SVM_POLY),
        "SVM_Sigmoid": make_svm(cv2.ml.SVM_SIGMOID),
        "Boost_Discrete": make_boost(cv2.ml.Boost_DISCRETE),
        "Boost_Real": make_boost(cv2.ml.Boost_REAL),
        "Boost_Logit": make_boost(cv2.ml.Boost_LOGIT),
    }
    model_functions.update(
        {f"KNN_k{k}": make_knn(k) for k in K_NEIGHBOR_SWEEP if k != 11}
    )
    oof_scores = {name: np.zeros(len(data), dtype=float) for name in model_functions}

    for fold_number, held_out in enumerate(folds, start=1):
        valid_mask = data["sequence"].isin(held_out).to_numpy()
        train_mask = ~valid_mask
        x_train_raw, x_valid_raw = x[train_mask], x[valid_mask]
        y_train = y[train_mask]
        x_train, x_valid = standardize(x_train_raw, x_valid_raw)
        x_balanced, y_balanced = balanced_rows(x_train, y_train, SEED + fold_number)

        logistic = fit_logistic(x_train_raw, y_train)
        oof_scores["Logistic"][valid_mask] = predict_fall_probability(
            logistic, x_valid_raw
        )
        for name, function in model_functions.items():
            if function is None:
                continue
            try:
                oof_scores[name][valid_mask] = function(
                    x_balanced, y_balanced, x_valid
                )
            except Exception as error:
                oof_scores[name][valid_mask] = np.nan
                print(f"Fold {fold_number}: {name} failed: {error}", flush=True)
        print(f"Completed held-out recording fold {fold_number}/{len(folds)}", flush=True)

    summary = []
    output = data[["sequence", "frame_number", "target"]].copy()
    for name, scores in oof_scores.items():
        if np.isnan(scores).any():
            print(f"{name}: incomplete; excluded from summary.")
            continue
        result = calc_metrics(data, scores)
        summary.append({"model": name, **result})
        output[f"{name}_score"] = scores
        output[f"{name}_prediction"] = scores >= 0.5
        print(
            f"{name}: acc={result['accuracy']:.3f}, bal_acc={result['balanced_accuracy']:.3f}, "
            f"precision={result['precision']:.3f}, recall={result['recall']:.3f}, "
            f"F1={result['f1']:.3f}, fall clips hit={result['fall_recordings_hit']}/30, "
            f"ADL clips flagged={result['adl_recordings_flagged']}/40, "
            f"confusion=[[{result['tn']},{result['fp']}],[{result['fn']},{result['tp']}]]"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)
    summary_frame = pd.DataFrame(summary).sort_values("balanced_accuracy", ascending=False)
    summary_path = OUTPUT_CSV.with_name("classifier-benchmark-summary.csv")
    summary_frame.to_csv(summary_path, index=False)
    print(f"Out-of-fold scores saved to:\n{OUTPUT_CSV}")
    print(f"Model comparison summary saved to:\n{summary_path}")
    print("These are model-screening results; selecting the winner on these folds can overestimate final performance.")


if __name__ == "__main__":
    main()
