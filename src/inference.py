"""Shared image inference helpers for the local UI."""

import json
import os

import cv2
import joblib
import numpy as np

from .feature_extraction import cnn_features, classical_features_rich
from .preprocessing import preprocess_pipeline, preprocess_pipeline_bagging


MODEL_CONFIGS = {
    "Boosting": {
        "directory": "outputs_boosting_quick",
        "model_file": "best_model.joblib",
        "metadata_file": "model_meta.json",
        "preprocessing": "boosting",
    },
    "Bagging": {
        "directory": "outputs_bagging",
        "model_file": "best_model_bagging.joblib",
        "metadata_file": "model_meta_bagging.json",
        "preprocessing": "bagging",
    },
}


def model_is_available(project_root, model_name):
    """Return whether the model's artifact and metadata are present."""
    config = MODEL_CONFIGS[model_name]
    model_path = os.path.join(project_root, config["directory"], config["model_file"])
    metadata_path = os.path.join(project_root, config["directory"], config["metadata_file"])
    return os.path.isfile(model_path) and os.path.isfile(metadata_path)


def load_model_bundle(project_root, model_name):
    """Load one of the project's saved models and its metadata."""
    config = MODEL_CONFIGS[model_name]
    model_dir = os.path.join(project_root, config["directory"])
    model_path = os.path.join(model_dir, config["model_file"])
    metadata_path = os.path.join(model_dir, config["metadata_file"])

    if not os.path.isfile(model_path) or not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"{model_name} model files are missing from {model_dir}")

    with open(metadata_path, encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    model = joblib.load(model_path)
    return model, metadata, config["preprocessing"]


def predict_image_bytes(image_bytes, model, metadata, preprocessing):
    """Run the matching preprocessing and classifier pipeline on uploaded bytes."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode this file as an image. Upload a valid JPG or PNG.")

    image_size = tuple(metadata["image_size"])
    image = cv2.resize(image, image_size)
    image_batch = np.expand_dims(image, axis=0)

    if preprocessing == "bagging":
        color_batch, gray_batch = preprocess_pipeline_bagging(image_batch)
    else:
        gray_batch = preprocess_pipeline(
            image_batch,
            sigmoid_y=metadata.get("sigmoid_y", 10.0),
            sigmoid_z=metadata.get("sigmoid_z", 0.5),
        )
        color_batch = image_batch

    if metadata["feature_type"] == "cnn_vgg16":
        features = cnn_features(color_batch)
    elif metadata["feature_type"] == "classical_rich":
        features = classical_features_rich(color_batch, gray_batch)
    else:
        raise ValueError(f"Unsupported feature type: {metadata['feature_type']}")

    probabilities = model.predict_proba(features)[0]
    predicted_index = int(np.argmax(probabilities))
    class_names = metadata["class_names"]
    return {
        "label": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            name: float(probability)
            for name, probability in zip(class_names, probabilities)
        },
    }
