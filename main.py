"""
main.py
--------
Runs the full Chapter-3 "optimization-driven ensemble-based (boosting)"
pipeline from the thesis:

    LC25000 (lung subset) -> preprocessing -> feature extraction
    -> SequentialBoostingEnsemble(RF, MLP, LR, SVM)
    -> hyperparameter tuning (Bayesian / Hyperband / BOHB)
    -> evaluation + plots (saved to ./outputs)

Usage
-----
    python main.py --data_root "/path/to/Test Set" --quick

    --data_root2   OPTIONAL second split folder to MERGE with --data_root
                   (e.g. "Train and Validation Set"), so you load the FULL
                   lung dataset instead of just one pre-made split:

    python main.py --data_root "/path/to/Test Set" ^
                    --data_root2 "/path/to/Train and Validation Set"

    --quick        use classical (fast, no-download) features + a small
                    subset of images, for a smoke test of the pipeline.
    (omit --quick) use VGG16 CNN features on the full dataset (slower,
                    downloads ImageNet weights, needs internet + GPU
                    recommended) -- matches the thesis's reported setup.
"""

import argparse
import os
import json
import numpy as np
import joblib

from src.data_loader import load_lc25000, split_data, group_split
from src.preprocessing import preprocess_pipeline
from src.feature_extraction import cnn_features, classical_features_rich
from src.boosting_ensemble import SequentialBoostingEnsemble
from src.optimization import run_bayesian, run_hyperband, run_bohb, params_from_study
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_optimization_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                         help="Path to a folder containing lung_aca/, "
                              "lung_scc/, lung_n/ (e.g. the 'Test Set' split)")
    parser.add_argument("--data_root2", default=None,
                         help="OPTIONAL second split folder (e.g. 'Train and "
                              "Validation Set') to MERGE with --data_root, "
                              "so the full lung dataset (~15,000 images) is "
                              "used instead of just one split.")
    parser.add_argument("--quick", action="store_true",
                         help="Fast smoke-test mode: classical features, "
                              "50 images/class, fewer optuna trials")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--plain_split", action="store_true",
                         help="Use a plain random train/test split (matches "
                              "the thesis's literal Sec 3.3 description, but "
                              "can leak near-duplicate augmented images "
                              "between train/test on LC25000). Default "
                              "behavior instead uses a leakage-aware "
                              "group_split() for a more honest accuracy "
                              "estimate -- recommended.")
    parser.add_argument("--n_trials", type=int, default=None,
                         help="Optuna trials per optimization method "
                              "(Bayesian/Hyperband/BOHB each run this many). "
                              "Default: 30 (full run) / 5 (--quick). This is "
                              "the single biggest time knob -- SVM training "
                              "on thousands of samples, repeated across "
                              "trials x 3 methods x cv_folds, is what makes "
                              "a full run slow. Try --n_trials 8 for a "
                              "much faster (but less thoroughly searched) run.")
    parser.add_argument("--cv_folds", type=int, default=3,
                         help="Cross-validation folds used inside each "
                              "optimization trial. Default 3. Try 2 to cut "
                              "optimization time further.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    max_per_class = 50 if args.quick else None
    n_trials = args.n_trials if args.n_trials is not None else (5 if args.quick else 30)

    data_roots = [args.data_root] + ([args.data_root2] if args.data_root2 else [])

    print(">> Loading LC25000 (lung subset, binary Malignant/Benign)...")
    if len(data_roots) > 1:
        print(f"   Merging {len(data_roots)} split folders: {data_roots}")
    X_raw, y, class_names = load_lc25000(
        data_roots, binary=True, image_size=(224, 224), max_per_class=max_per_class
    )
    print(f"Loaded {len(X_raw)} images, classes: {class_names}")

    if args.plain_split:
        print(">> Using plain random split (may leak near-duplicates)...")
        X_train_raw, X_test_raw, y_train, y_test = split_data(X_raw, y, test_size=0.2)
    else:
        print(">> Using leakage-aware group_split (near-duplicates kept "
              "together on one side)...")
        X_train_raw, X_test_raw, y_train, y_test = group_split(X_raw, y, test_size=0.2)

    print(">> Preprocessing (grayscale + sigmoid stretching)...")
    train_gray = preprocess_pipeline(X_train_raw)
    test_gray = preprocess_pipeline(X_test_raw)

    print(">> Extracting features...")
    if args.quick:
        X_train = classical_features_rich(X_train_raw, train_gray)
        X_test = classical_features_rich(X_test_raw, test_gray)
    else:
        # CNN features use the original RGB images (VGG16 expects 3 channels)
        X_train = cnn_features(X_train_raw)
        X_test = cnn_features(X_test_raw)

    results = {}
    trained_models = {}  # keep every trained model so we can save the best one

    # ---- Baseline (no optimization) ----
    print(">> Training baseline SequentialBoostingEnsemble (default params)...")
    baseline = SequentialBoostingEnsemble()
    baseline.fit(X_train, y_train)
    baseline_preds = baseline.predict(X_test)
    results["baseline"] = compute_metrics(y_test, baseline_preds)
    trained_models["baseline"] = baseline
    plot_confusion_matrix(
        y_test, baseline_preds, class_names,
        "Confusion Matrix - Baseline Boosting Ensemble",
        os.path.join(args.out_dir, "cm_baseline.png"),
    )
    print("Baseline:", results["baseline"])

    # ---- Optimization strategies ----
    for name, runner in [
        ("bayesian", run_bayesian),
        ("hyperband", run_hyperband),
        ("bohb", run_bohb),
    ]:
        print(f">> Running {name} hyperparameter optimization...")
        study = runner(X_train, y_train, n_trials=n_trials, cv_folds=args.cv_folds)
        best_params = params_from_study(study)

        model = SequentialBoostingEnsemble(params=best_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = compute_metrics(y_test, preds)
        results[name] = metrics
        trained_models[name] = model
        print(f"{name} best CV score: {study.best_value:.4f} | Test metrics: {metrics}")

        plot_confusion_matrix(
            y_test, preds, class_names,
            f"Confusion Matrix - Boosting Ensemble ({name.upper()})",
            os.path.join(args.out_dir, f"cm_{name}.png"),
        )

    # ---- Summary comparison plot (Fig. 3.9-style) ----
    acc_comparison = {
        "Bayesian": results["bayesian"]["accuracy"] * 100,
        "Hyperband": results["hyperband"]["accuracy"] * 100,
        "BOHB": results["bohb"]["accuracy"] * 100,
    }
    plot_optimization_comparison(
        acc_comparison, os.path.join(args.out_dir, "optimization_comparison.png")
    )

    print("\n===== FINAL RESULTS =====")
    for k, v in results.items():
        print(f"{k:10s} -> accuracy: {v['accuracy']*100:.2f}%  "
              f"precision: {v['precision']*100:.2f}%  "
              f"recall: {v['recall']*100:.2f}%  f1: {v['f1']*100:.2f}%")

    best_method = max(
        ["bayesian", "hyperband", "bohb"], key=lambda k: results[k]["accuracy"]
    )
    print(f"\nBest optimization method: {best_method.upper()} "
          f"({results[best_method]['accuracy']*100:.2f}% accuracy)")
    print(f"All plots saved to: {args.out_dir}/")

    # ---- Save the best model + metadata, so predict.py can reuse it ----
    best_model = trained_models[best_method]
    model_path = os.path.join(args.out_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)

    meta = {
        "best_method": best_method,
        "class_names": class_names,          # index 0 = Benign, 1 = Malignant
        "feature_type": "classical_rich" if args.quick else "cnn_vgg16",
        "image_size": [224, 224],
        "sigmoid_y": 10.0,
        "sigmoid_z": 0.5,
        "accuracy": results[best_method]["accuracy"],
    }
    meta_path = os.path.join(args.out_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved best model ({best_method}) to: {model_path}")
    print(f"Saved model metadata to: {meta_path}")
    print(f"\n>> To test on a NEW image, run:\n"
          f"   python predict.py --image \"path/to/image.jpeg\" "
          f"--model_dir {args.out_dir}")


if __name__ == "__main__":
    main()