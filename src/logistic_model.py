"""Small NumPy-only logistic classifier for sequence-level fall detection."""

import numpy as np


def fit_logistic(x, y, regularization=0.1, max_iter=5000, learning_rate=0.1):
    """Fit a binary logistic model with median imputation and standardization."""
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float)
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = ~np.isfinite(x)
    x[missing] = np.take(medians, np.where(missing)[1])

    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-8] = 1.0
    x = (x - means) / scales
    x = np.column_stack([np.ones(len(x)), x])

    class_counts = np.bincount(y.astype(int), minlength=2)
    sample_weights = np.where(y == 1, len(y) / (2 * class_counts[1]), len(y) / (2 * class_counts[0]))
    weights = np.zeros(x.shape[1], dtype=float)

    for _ in range(max_iter):
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(x @ weights, -35, 35)))
        gradient = (x.T @ (sample_weights * (probabilities - y))) / sample_weights.sum()
        gradient[1:] += regularization * weights[1:]
        updated = weights - learning_rate * gradient
        if np.linalg.norm(updated - weights) < 1e-8:
            weights = updated
            break
        weights = updated

    return {
        "medians": medians,
        "means": means,
        "scales": scales,
        "weights": weights,
        "feature_count": len(medians),
    }


def predict_fall_probability(model, x):
    """Return learned fall probabilities for one or more feature rows."""
    x = np.asarray(x, dtype=float).copy()
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != model["feature_count"]:
        raise ValueError("Input feature count does not match the trained model.")
    missing = ~np.isfinite(x)
    x[missing] = np.take(model["medians"], np.where(missing)[1])
    x = (x - model["means"]) / model["scales"]
    x = np.column_stack([np.ones(len(x)), x])
    scores = np.clip(x @ model["weights"], -35, 35)
    return 1.0 / (1.0 + np.exp(-scores))
