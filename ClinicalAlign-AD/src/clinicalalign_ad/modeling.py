"""Dependency-light, leakage-safe baseline modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""

    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute binary ROC AUC using average ranks for tied scores."""

    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    positive = y == 1
    negative = y == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    statistic = ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0
    return float(statistic / (n_positive * n_negative))


def binary_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return discrimination, classification, and calibration metrics."""

    y = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    predicted = probabilities >= threshold
    positive = y == 1
    negative = y == 0
    tp = int((predicted & positive).sum())
    tn = int((~predicted & negative).sum())
    fp = int((predicted & negative).sum())
    fn = int((~predicted & positive).sum())

    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "auc": roc_auc(y, probabilities),
        "accuracy": float((predicted == positive).mean()),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "brier": float(np.mean((probabilities - y) ** 2)),
    }


def choose_youden_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a classification threshold from training predictions only."""

    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], probabilities]))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        metrics = binary_metrics(y, probabilities, float(threshold))
        score = metrics["sensitivity"] + metrics["specificity"] - 1.0
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def stratified_holdout_indices(
    y_true: np.ndarray, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic holdout with proportional class allocation."""

    y = np.asarray(y_true, dtype=int)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target_total = int(np.ceil(len(y) * float(test_fraction)))
    raw = counts * float(test_fraction)
    allocation = np.floor(raw).astype(int)
    remainder = target_total - int(allocation.sum())
    order = np.argsort(-(raw - allocation), kind="stable")
    for position in order[:remainder]:
        allocation[position] += 1

    test_parts = []
    for class_value, class_test_n in zip(classes, allocation):
        indices = np.flatnonzero(y == class_value)
        rng.shuffle(indices)
        test_parts.append(indices[:class_test_n])
    test_indices = np.sort(np.concatenate(test_parts))
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[test_indices] = False
    return np.flatnonzero(train_mask), test_indices


def stratified_folds(y_true: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Return deterministic validation-index folds with class stratification."""

    y = np.asarray(y_true, dtype=int)
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for class_value in np.unique(y):
        indices = np.flatnonzero(y == class_value)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            folds[position % n_folds].append(int(index))
    return [np.asarray(sorted(fold), dtype=int) for fold in folds]


@dataclass
class MedianStandardizer:
    """Training-only median imputation, scaling, and missing indicators."""

    feature_names: list[str] | None = None
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    indicator_mask: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, features: Iterable[str]) -> "MedianStandardizer":
        self.feature_names = list(features)
        values = frame[self.feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        values[~np.isfinite(values)] = np.nan
        self.indicator_mask = np.isnan(values).any(axis=0)
        with np.errstate(all="ignore"):
            self.medians = np.nanmedian(values, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.0)
        imputed = np.where(np.isnan(values), self.medians, values)
        self.means = imputed.mean(axis=0)
        self.scales = imputed.std(axis=0)
        self.scales = np.where(self.scales > 1e-12, self.scales, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if any(value is None for value in [self.feature_names, self.medians, self.means, self.scales, self.indicator_mask]):
            raise RuntimeError("Preprocessor must be fit before transform")
        values = frame[self.feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        values[~np.isfinite(values)] = np.nan
        missing = np.isnan(values)
        imputed = np.where(missing, self.medians, values)
        standardized = (imputed - self.means) / self.scales
        if self.indicator_mask.any():
            standardized = np.column_stack(
                [standardized, missing[:, self.indicator_mask].astype(float)]
            )
        return standardized

    def fit_transform(self, frame: pd.DataFrame, features: Iterable[str]) -> np.ndarray:
        return self.fit(frame, features).transform(frame)


@dataclass
class LogisticL2:
    penalty: float = 0.1
    coefficients: np.ndarray | None = None

    def fit(self, x: np.ndarray, y_true: np.ndarray) -> "LogisticL2":
        y = np.asarray(y_true, dtype=float)
        design = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            linear = design @ parameters
            loss = np.mean(np.logaddexp(0.0, linear) - y * linear)
            penalty_loss = 0.5 * self.penalty * np.dot(parameters[1:], parameters[1:])
            residual = sigmoid(linear) - y
            gradient = design.T @ residual / len(y)
            gradient[1:] += self.penalty * parameters[1:]
            return float(loss + penalty_loss), gradient

        result = minimize(
            objective,
            np.zeros(design.shape[1]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success:
            raise RuntimeError(f"Logistic regression failed to converge: {result.message}")
        self.coefficients = result.x
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model must be fit before prediction")
        design = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
        return sigmoid(design @ self.coefficients)


@dataclass
class GaussianNB:
    variance_smoothing: float = 1e-9
    classes: np.ndarray | None = None
    priors: np.ndarray | None = None
    means: np.ndarray | None = None
    variances: np.ndarray | None = None

    def fit(self, x: np.ndarray, y_true: np.ndarray) -> "GaussianNB":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y_true, dtype=int)
        self.classes = np.array([0, 1])
        self.priors = np.array([(y == value).mean() for value in self.classes])
        self.means = np.vstack([x[y == value].mean(axis=0) for value in self.classes])
        variances = np.vstack([x[y == value].var(axis=0) for value in self.classes])
        epsilon = self.variance_smoothing * max(float(x.var(axis=0).max()), 1.0)
        self.variances = variances + epsilon
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if any(value is None for value in [self.classes, self.priors, self.means, self.variances]):
            raise RuntimeError("Model must be fit before prediction")
        x = np.asarray(x, dtype=float)
        log_probabilities = []
        for index in range(2):
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * self.variances[index])
                + (x - self.means[index]) ** 2 / self.variances[index],
                axis=1,
            )
            log_probabilities.append(np.log(self.priors[index]) + log_likelihood)
        log_probabilities = np.column_stack(log_probabilities)
        log_probabilities -= log_probabilities.max(axis=1, keepdims=True)
        probabilities = np.exp(log_probabilities)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities[:, 1]


@dataclass
class KNearestNeighbors:
    neighbors: int = 11
    x_train: np.ndarray | None = None
    y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y_true: np.ndarray) -> "KNearestNeighbors":
        self.x_train = np.asarray(x, dtype=float)
        self.y_train = np.asarray(y_true, dtype=float)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Model must be fit before prediction")
        x = np.asarray(x, dtype=float)
        distances = np.sum((x[:, None, :] - self.x_train[None, :, :]) ** 2, axis=2)
        neighbors = min(int(self.neighbors), len(self.x_train))
        nearest = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
        return self.y_train[nearest].mean(axis=1)


def make_model(model_name: str, parameter: float | int) -> Any:
    if model_name == "logistic_l2":
        return LogisticL2(penalty=float(parameter))
    if model_name == "gaussian_nb":
        return GaussianNB(variance_smoothing=float(parameter))
    if model_name == "knn":
        return KNearestNeighbors(neighbors=int(parameter))
    raise ValueError(f"Unknown model: {model_name}")


def tune_parameter(
    model_name: str,
    parameters: Iterable[float | int],
    frame: pd.DataFrame,
    y_true: np.ndarray,
    features: list[str],
    n_folds: int,
    seed: int,
) -> tuple[float | int, pd.DataFrame, np.ndarray]:
    """Tune one model hyperparameter using training-only stratified CV."""

    y = np.asarray(y_true, dtype=int)
    folds = stratified_folds(y, n_folds, seed)
    records = []
    best_parameter: float | int | None = None
    best_auc = -np.inf
    best_predictions: np.ndarray | None = None
    for parameter in parameters:
        predictions = np.full(len(frame), np.nan)
        for validation_indices in folds:
            training_mask = np.ones(len(frame), dtype=bool)
            training_mask[validation_indices] = False
            training_indices = np.flatnonzero(training_mask)
            preprocessor = MedianStandardizer()
            x_train = preprocessor.fit_transform(frame.iloc[training_indices], features)
            x_valid = preprocessor.transform(frame.iloc[validation_indices])
            model = make_model(model_name, parameter).fit(x_train, y[training_indices])
            predictions[validation_indices] = model.predict_proba(x_valid)
        auc = roc_auc(y, predictions)
        records.append({"parameter": parameter, "cv_auc": auc})
        if auc > best_auc + 1e-12:
            best_auc = auc
            best_parameter = parameter
            best_predictions = predictions.copy()
    if best_parameter is None:
        raise RuntimeError(f"Could not tune model {model_name}")
    if best_predictions is None:
        raise RuntimeError(f"Could not retain cross-validated predictions for {model_name}")
    return best_parameter, pd.DataFrame(records), best_predictions


def bootstrap_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
    metric_names: Iterable[str] = (
        "auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "brier",
    ),
) -> dict[str, tuple[float, float]]:
    """Compute percentile intervals using class-stratified bootstrap samples."""

    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == value) for value in [0, 1]]
    values = {name: [] for name in metric_names}
    for _ in range(int(replicates)):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        metrics = binary_metrics(y[sampled], probabilities[sampled], threshold)
        for name in metric_names:
            values[name].append(metrics[name])
    return {
        name: (
            float(np.nanpercentile(metric_values, 2.5)),
            float(np.nanpercentile(metric_values, 97.5)),
        )
        for name, metric_values in values.items()
    }


def paired_bootstrap_auc_difference(
    y_true: np.ndarray,
    candidate_probabilities: np.ndarray,
    reference_probabilities: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return observed and percentile CI for paired AUC difference."""

    y = np.asarray(y_true, dtype=int)
    candidate = np.asarray(candidate_probabilities, dtype=float)
    reference = np.asarray(reference_probabilities, dtype=float)
    observed = roc_auc(y, candidate) - roc_auc(y, reference)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == value) for value in [0, 1]]
    differences = []
    for _ in range(int(replicates)):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        differences.append(
            roc_auc(y[sampled], candidate[sampled])
            - roc_auc(y[sampled], reference[sampled])
        )
    return (
        float(observed),
        float(np.nanpercentile(differences, 2.5)),
        float(np.nanpercentile(differences, 97.5)),
    )