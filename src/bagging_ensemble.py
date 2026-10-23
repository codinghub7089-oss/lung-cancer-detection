"""
bagging_ensemble.py
---------------------
Implements the "optimization-driven parallel ensemble-based (bagging)"
approach from Chapter 4 of the thesis: SVM and RF are trained in PARALLEL,
each on a bootstrap-resampled subset of the training data (classic
bagging / Bootstrap Aggregating), and their predictions are combined by
averaging predicted class probabilities (soft voting).

This is deliberately a separate, independent class from
SequentialBoostingEnsemble (src/boosting_ensemble.py) -- Chapter 3's
boosting pipeline is completely untouched by this file.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample


class ParallelBaggingEnsemble:
    """
    Parallel bagging ensemble of SVM and RF (thesis Chapter 4).

    Each base model is trained independently on its own bootstrap sample
    (sampling with replacement from the training set) -- this is the
    "Bootstrap Aggregating" referred to throughout the thesis's Sec 2.3.2
    and Sec 3.3.6. The two models' predicted probabilities are then
    averaged (soft voting) for the final prediction.
    """

    MODEL_NAMES = ["svm", "rf"]

    def __init__(self, params=None, random_state=42, n_bootstrap_sets=1):
        """
        params : dict, optional
            e.g. {"svm": {...}, "rf": {...}} -- hyperparameters for each
            base model (filled in by the optimization step). Falls back
            to reasonable defaults.
        n_bootstrap_sets : int
            Number of bootstrap resamples per model to average over
            (classic bagging uses many; we default to 1 bootstrap sample
            per model per fit() call, which is the minimal, fastest form
            of bagging -- increase for a smoother, more averaged ensemble
            at the cost of training time).
        """
        self.random_state = random_state
        self.params = params or {}
        self.n_bootstrap_sets = max(1, int(n_bootstrap_sets))
        self.models = {}
        self.classes_ = None

    def _build_model(self, name):
        p = self.params.get(name, {})
        if name == "svm":
            return SVC(probability=True, random_state=self.random_state, **p)
        if name == "rf":
            return RandomForestClassifier(random_state=self.random_state, **p)
        raise ValueError(f"Unknown model {name}")

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.random_state)

        for name in self.MODEL_NAMES:
            fitted_bootstraps = []
            for i in range(self.n_bootstrap_sets):
                seed = rng.randint(0, 1_000_000)
                X_bs, y_bs = resample(
                    X, y, replace=True, n_samples=len(y), random_state=seed,
                    stratify=y,
                )
                model = self._build_model(name)
                model.fit(X_bs, y_bs)
                fitted_bootstraps.append(model)
            self.models[name] = fitted_bootstraps

        return self

    def predict_proba(self, X):
        agg = np.zeros((X.shape[0], len(self.classes_)))
        total_models = 0
        for name, bootstraps in self.models.items():
            for model in bootstraps:
                agg += model.predict_proba(X)
                total_models += 1
        agg /= total_models
        return agg

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def individual_scores(self, X, y):
        """Per-model accuracy, averaged across bootstrap copies of each
        model (used for thesis Fig. 4.3 / 4.7 / 4.8-style comparisons)."""
        scores = {}
        for name, bootstraps in self.models.items():
            accs = [accuracy_score(y, m.predict(X)) for m in bootstraps]
            scores[name] = float(np.mean(accs))
        return scores