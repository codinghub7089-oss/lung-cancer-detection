"""
predict.py
-----------
Loads the best model saved by main.py (outputs/best_model.joblib +
outputs/model_meta.json) and runs it on ONE new histopathology image,
printing the predicted class (Benign / Malignant) and confidence.

Usage
-----
    python predict.py --image "path/to/some_image.jpeg"

    (optional) --model_dir outputs     (default: outputs, same folder main.py saved to)

What it does, step by step (mirrors main.py's pipeline exactly so the
new image is treated the same way the training images were):
    1. Load the image (cv2)
    2. Resize to 224x224
    3. grayscale + sigmoid stretching (src/preprocessing.py)
    4. Feature extraction - VGG16 CNN features OR classical_features_rich,
       whichever was used for the saved model (recorded in model_meta.json)
    5. model.predict() + model.predict_proba() -> class + confidence
"""

import argparse
import json
import os
import sys

import cv2
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import preprocess_pipeline
from src.feature_extraction import cnn_features, classical_features_rich


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True,
                         help="Path to a single histopathology image "
                              "(.jpeg/.jpg/.png)")
    parser.add_argument("--model_dir", default="outputs",
                         help="Folder containing best_model.joblib and "
                              "model_meta.json (saved by main.py). Default: outputs")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "best_model.joblib")
    meta_path = os.path.join(args.model_dir, "model_meta.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print(f"ERROR: could not find '{model_path}' or '{meta_path}'.")
        print("Run main.py first (full training run) -- it saves these "
              "files automatically when it finishes.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    class_names = meta["class_names"]          # e.g. ["Benign", "Malignant"]
    image_size = tuple(meta["image_size"])      # (224, 224)
    sigmoid_y = meta["sigmoid_y"]
    sigmoid_z = meta["sigmoid_z"]
    feature_type = meta["feature_type"]          # "cnn_vgg16" or "classical_rich"

    print(f">> Loading model (best method: {meta['best_method']}, "
          f"trained accuracy: {meta['accuracy']*100:.2f}%)...")
    model = joblib.load(model_path)

    print(f">> Loading image: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: could not read image at '{args.image}'. "
              f"Check the path and that it's a valid .jpeg/.jpg/.png file.")
        sys.exit(1)
    img = cv2.resize(img, image_size)
    img_batch = np.expand_dims(img, axis=0)  # shape (1, H, W, 3)

    print(">> Preprocessing (grayscale + sigmoid stretching)...")
    gray_batch = preprocess_pipeline(img_batch, sigmoid_y=sigmoid_y, sigmoid_z=sigmoid_z)

    print(f">> Extracting features ({feature_type})...")
    if feature_type == "cnn_vgg16":
        features = cnn_features(img_batch)
    else:
        features = classical_features_rich(img_batch, gray_batch)

    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    pred_label = class_names[pred]
    confidence = proba[pred] * 100

    print("\n===== PREDICTION =====")
    print(f"Image       : {args.image}")
    print(f"Predicted   : {pred_label}")
    print(f"Confidence  : {confidence:.2f}%")
    print("\nFull probability breakdown:")
    for name, p in zip(class_names, proba):
        print(f"  {name:10s}: {p*100:.2f}%")


if __name__ == "__main__":
    main()