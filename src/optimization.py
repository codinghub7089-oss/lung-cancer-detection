"""
optimization.py
-----------------
Implements the three hyperparameter-tuning strategies compared in the
thesis (Sec 2.4 / Fig. 3.9):

    1. Bayesian Optimization  -> Optuna TPESampler
    2. Hyperband              -> Optuna HyperbandPruner + RandomSampler
    3. BOHB                   -> Optuna TPESampler + HyperbandPruner
                                  (this combination IS BOHB: Bayesian
                                  model-based sampling + Hyperband's
                                  successive-halving resource allocation)

Each search tunes RF, MLP, LR, and SVM hyperparameters jointly for the
SequentialBoostingEnsemble, using 5-fold CV accuracy as the objective.
"""

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .boosting_ensemble import SequentialBoostingEnsemble

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial):
    return {
        "rf": {
            "n_estimators": trial.suggest_int("rf_n_estimators", 50, 300),
            "max_depth": trial.suggest_int("rf_max_depth", 3, 20),
        },
        "mlp": {
            "hidden_layer_sizes": (trial.suggest_int("mlp_hidden", 32, 256),),
            "alpha": trial.suggest_float("mlp_alpha", 1e-5, 1e-1, log=True),
        },
        "lr": {
            "C": trial.suggest_float("lr_C", 1e-3, 10.0, log=True),
        },
        "svm": {
            "C": trial.suggest_float("svm_C", 1e-2, 100.0, log=True),
            "kernel": trial.suggest_categorical("svm_kernel", ["rbf", "linear"]),
            "gamma": trial.suggest_categorical("svm_gamma", ["scale", "auto"]),
        },
    }


def _objective(trial, X, y, cv_folds, epochs_budget=None):
    params = _suggest_params(trial)
    # `epochs_budget` (a.k.a. "resource" in Hyperband) is only meaningful
    # for the MLP; we map it to max_iter to emulate Hyperband's successive
    # halving on training budget.
    if epochs_budget is not None:
        params["mlp"]["max_iter"] = int(epochs_budget)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = SequentialBoostingEnsemble(params=params)
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
    {"rf": {...}, "mlp": {...}, "lr": {...}, "svm": {...}} format."""
    bp = study.best_params
    return {
        "rf": {"n_estimators": bp["rf_n_estimators"], "max_depth": bp["rf_max_depth"]},
        "mlp": {"hidden_layer_sizes": (bp["mlp_hidden"],), "alpha": bp["mlp_alpha"]},
        "lr": {"C": bp["lr_C"]},
        "svm": {"C": bp["svm_C"], "kernel": bp["svm_kernel"], "gamma": bp["svm_gamma"]},
    }
