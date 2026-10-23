"""
boosting_ensemble.py
---------------------
Implements the "optimization-driven ensemble-based (boosting)" approach
from Chapter 3 of the thesis: SVM, RF, LR, MLP are combined sequentially
(boosting), where each subsequent model focuses more on the samples the
previous model got wrong (AdaBoost-style sample reweighting), and the
final prediction is a weighted vote of all four models
(weighted by each model's own accuracy, i.e. "model confidence").

Order used (as listed in the thesis / flowchart, Fig. 3.2): RF -> MLP -> LR -> SVM
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class SequentialBoostingEnsemble:
    """
    Sequential (boosting-style) ensemble of RF -> MLP -> LR -> SVM.

    Each model is trained with sample weights updated using the classic
    AdaBoost.M1 update rule based on the previous model's errors on the
    training set. Final prediction = weighted majority vote (weights =
    each model's log-odds "say", same as AdaBoost's alpha).
    """

    MODEL_ORDER = ["rf", "mlp", "lr", "svm"]

    def __init__(self, params=None, random_state=42):
        """
        params : dict, optional
            e.g. {"rf": {...}, "mlp": {...}, "lr": {...}, "svm": {...}}
            Hyperparameters for each base model (filled in by the
            optimization step). Falls back to reasonable defaults.
        """
        self.random_state = random_state
        self.params = params or {}
        self.models = {}
        self.alphas = {}  # boosting "say" weight per model

    def _build_model(self, name):
        p = self.params.get(name, {})
        if name == "rf":
            return RandomForestClassifier(random_state=self.random_state, **p)
        if name == "mlp":
            return MLPClassifier(max_iter=500, random_state=self.random_state, **p)
        if name == "lr":
            return LogisticRegression(max_iter=1000, random_state=self.random_state, **p)
        if name == "svm":
            return SVC(probability=True, random_state=self.random_state, **p)
        raise ValueError(f"Unknown model {name}")

    def fit(self, X, y):
        n = len(y)
        sample_weight = np.ones(n) / n

        for name in self.MODEL_ORDER:
            model = self._build_model(name)
            try:
                model.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                # some estimators (e.g. certain MLP configs) may not accept
                # sample_weight directly -> resample instead
                idx = np.random.choice(n, size=n, replace=True, p=sample_weight)
                model.fit(X[idx], y[idx])

            preds = model.predict(X)
            err = np.sum(sample_weight * (preds != y)) / np.sum(sample_weight)
            err = np.clip(err, 1e-6, 1 - 1e-6)  # avoid div-by-zero / log(0)

            alpha = 0.5 * np.log((1 - err) / err)
            self.models[name] = model
            self.alphas[name] = alpha

            # AdaBoost.M1 sample-weight update
            sample_weight *= np.exp(-alpha * np.where(preds == y, 1, -1))
            sample_weight /= sample_weight.sum()

        return self

    def predict_proba(self, X):
        classes = self.models["rf"].classes_
        agg = np.zeros((X.shape[0], len(classes)))
        total_alpha = sum(self.alphas.values())
        for name, model in self.models.items():
            w = self.alphas[name] / total_alpha
            agg += w * model.predict_proba(X)
        return agg

    def predict(self, X):
        proba = self.predict_proba(X)
        classes = self.models["rf"].classes_
        return classes[np.argmax(proba, axis=1)]

    def individual_scores(self, X, y):
        """Per-model accuracy (used for Fig. 3.3 / 3.4-style comparisons)."""
        return {name: accuracy_score(y, m.predict(X)) for name, m in self.models.items()}
