"""
optimization_bagging.py
-------------------------
Hyperparameter tuning (Bayesian / Hyperband / BOHB) for the Chapter-4
ParallelBaggingEnsemble (SVM + RF only).

This mirrors src/optimization.py (used for the Chapter-3 boosting
ensemble) exactly in structure, but tunes only the 2 bagging base models
instead of 4. Kept as a fully separate file so Chapter 3's optimization.py
is never touched or risked by Chapter 4 changes.
"""

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from .bagging_ensemble import ParallelBaggingEnsemble

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial):
    return {
        "svm": {
            "C": trial.suggest_float("svm_C", 1e-2, 100.0, log=True),
            "kernel": trial.suggest_categorical("svm_kernel", ["rbf", "linear"]),
            "gamma": trial.suggest_categorical("svm_gamma", ["scale", "auto"]),
        },
        "rf": {
            "n_estimators": trial.suggest_int("rf_n_estimators", 50, 300),
            "max_depth": trial.suggest_int("rf_max_depth", 3, 20),
        },
    }


def _objective(trial, X, y, cv_folds):
    params = _suggest_params(trial)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = ParallelBaggingEnsemble(params=params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(np.mean(preds == y[val_idx]))

        trial.report(np.mean(scores), step=len(scores))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def run_bayesian(X, y, n_trials=30, cv_folds=3):
    """Pure Bayesian optimization (TPE sampler, no pruning)."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(lambda t: _objective(t, X, y, cv_folds), n_trials=n_trials)
    return study


def run_hyperband(X, y, n_trials=30, cv_folds=3):
    """Hyperband: random sampling + successive-halving pruning."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=cv_folds, reduction_factor=2
        ),
    )
    study.optimize(lambda t: _objective(t, X, y, cv_folds), n_trials=n_trials)
    return study


def run_bohb(X, y, n_trials=30, cv_folds=3):
    """BOHB: Bayesian (TPE) sampling + Hyperband successive-halving pruning."""
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=cv_folds, reduction_factor=2
        ),
    )
    study.optimize(lambda t: _objective(t, X, y, cv_folds), n_trials=n_trials)
    return study


def params_from_study(study):
    """Convert a flat Optuna best_params dict back into the nested
    {"svm": {...}, "rf": {...}} format."""
    bp = study.best_params
    return {
        "svm": {"C": bp["svm_C"], "kernel": bp["svm_kernel"], "gamma": bp["svm_gamma"]},
        "rf": {"n_estimators": bp["rf_n_estimators"], "max_depth": bp["rf_max_depth"]},
    }