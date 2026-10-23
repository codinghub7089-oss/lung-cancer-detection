"""
predict_bagging.py
--------------------
Loads the best bagging model saved by main_bagging.py
(outputs_bagging/best_model_bagging.joblib + model_meta_bagging.json) and
runs it on ONE OR MORE new histopathology images, printing the predicted
class (Benign / Malignant) and confidence for each.

This is a separate script from predict.py (Chapter 3 boosting model) so
the two never conflict.

Usage
-----
    One image:
        python predict_bagging.py --image "path/to/some_image.jpeg"

    Multiple images (space-separated, each its own quoted path):
        python predict_bagging.py --image "path/to/img1.jpeg" "path/to/img2.jpeg"

    (optional) --model_dir outputs_bagging   (default, same folder
                main_bagging.py saved to)
"""

import argparse
import json
import os
import sys

import cv2
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import preprocess_pipeline_bagging
from src.feature_extraction import cnn_features, classical_features_rich


def predict_one(image_path, model, class_names, image_size, feature_type):
    """Runs the full pipeline on one image and returns (label, confidence,
    full_probabilities) or (None, None, None) if the image could not be read."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ERROR: could not read image at '{image_path}'. "
              f"Check the path and that it's a valid .jpeg/.jpg/.png file.")
        return None, None, None

    img = cv2.resize(img, image_size)
    img_batch = np.expand_dims(img, axis=0)  # shape (1, H, W, 3)

    rgb_enh_batch, gray_batch = preprocess_pipeline_bagging(img_batch)

    if feature_type == "cnn_vgg16":
        features = cnn_features(rgb_enh_batch)
    else:
        features = classical_features_rich(rgb_enh_batch, gray_batch)

    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    pred_label = class_names[pred]
    confidence = proba[pred] * 100
    return pred_label, confidence, proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, nargs="+",
                         help="Path(s) to one or more histopathology images "
                              "(.jpeg/.jpg/.png). Separate multiple paths "
                              "with a space, each in its own quotes.")
    parser.add_argument("--model_dir", default="outputs_bagging",
                         help="Folder containing best_model_bagging.joblib "
                              "and model_meta_bagging.json (saved by "
                              "main_bagging.py). Default: outputs_bagging")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "best_model_bagging.joblib")
    meta_path = os.path.join(args.model_dir, "model_meta_bagging.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print(f"ERROR: could not find '{model_path}' or '{meta_path}'.")
        print("Run main_bagging.py first (full training run) -- it saves "
              "these files automatically when it finishes.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    class_names = meta["class_names"]          # e.g. ["Benign", "Malignant"]
    image_size = tuple(meta["image_size"])      # (224, 224)
    feature_type = meta["feature_type"]          # "cnn_vgg16" or "classical_rich"

    print(f">> Loading bagging model (best method: {meta['best_method']}, "
          f"trained accuracy: {meta['accuracy']*100:.2f}%)...")
    model = joblib.load(model_path)  # loaded ONCE, reused for every image below

    print(f">> Running prediction on {len(args.image)} image(s)...\n")

    results = []
    for image_path in args.image:
        print(f">> Processing: {image_path}")
        label, confidence, proba = predict_one(
            image_path, model, class_names, image_size, feature_type
        )
        if label is None:
            continue  # error already printed inside predict_one
        results.append((image_path, label, confidence, proba))

        print(f"   Predicted   : {label}")
        print(f"   Confidence  : {confidence:.2f}%")
        for name, p in zip(class_names, proba):
            print(f"     {name:10s}: {p*100:.2f}%")
        print()

    if len(results) > 1:
        print("===== SUMMARY (Bagging: SVM + RF) =====")
        for image_path, label, confidence, _ in results:
            print(f"{os.path.basename(image_path):30s} -> {label:10s} ({confidence:.2f}%)")


if __name__ == "__main__":
    main()