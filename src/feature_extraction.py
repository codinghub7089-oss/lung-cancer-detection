"""
feature_extraction.py
----------------------
The thesis states Keras/TensorFlow was used for "Model Implementation"
(Sec 3.5) but does not specify the exact feature extractor. The
standard, reproducible choice for classical ML classifiers (SVM/RF/MLP/LR)
on histopathology images is CNN bottleneck ("transfer learning") features
from a pretrained network.

Two extractors are provided:
    1. cnn_features()      -> VGG16 (ImageNet) global-average-pooled features
                               (recommended, matches "Keras" mention in thesis)
    2. classical_features() -> fast, no-download fallback for quick testing
                               (pixel statistics + histogram) — use only for
                               a smoke test, not for reporting final results.
"""

import numpy as np


def cnn_features(images_rgb, batch_size=32):
    """
    Extract 512-d bottleneck features using VGG16 (include_top=False,
    pooling='avg'), pretrained on ImageNet.

    Parameters
    ----------
    images_rgb : np.ndarray (N, 224, 224, 3), uint8, RGB or BGR order
                 (cv2 loads BGR — that's fine, VGG16 preprocess_input
                 expects BGR too, so no channel swap is needed if the
                 images came from cv2.imread).

    Returns
    -------
    np.ndarray (N, 512) feature matrix
    """
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

    base_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
    X = images_rgb.astype(np.float32)
    X = preprocess_input(X)
    features = base_model.predict(X, batch_size=batch_size, verbose=1)
    return features


def classical_features(gray_images):
    """
    Lightweight fallback feature extractor (no downloads / no GPU needed).
    Uses per-image histogram (32 bins) + basic intensity statistics.
    Intended for quick pipeline smoke-testing only (very low accuracy
    is expected/normal with this one -- use classical_features_rich()
    below, or cnn_features(), for real accuracy numbers).

    Parameters
    ----------
    gray_images : np.ndarray (N, H, W) grayscale images (0-255)

    Returns
    -------
    np.ndarray (N, 36) feature matrix
    """
    feats = []
    for img in gray_images:
        hist, _ = np.histogram(img, bins=32, range=(0, 255))
        hist = hist / (hist.sum() + 1e-8)
        stats = np.array([img.mean(), img.std(), img.min(), img.max()])
        feats.append(np.concatenate([hist, stats]))
    return np.array(feats)


def classical_features_rich(rgb_images, gray_images):
    """
    Stronger no-download, no-GPU feature extractor. Combines:
      - Per-channel color histograms (16 bins x 3 channels = 48)
      - Grayscale intensity histogram (32 bins)
      - Basic intensity statistics (mean, std, min, max)
      - Edge density (Canny)
      - Texture / sharpness (Laplacian variance)
      - HOG (Histogram of Oriented Gradients) shape/texture descriptor

    This is the recommended extractor for --quick mode: it needs only
    OpenCV + numpy (already installed) but captures color, texture and
    shape information, which is what actually drives classification
    accuracy on histopathology images -- unlike the bare `classical_features()`
    above, which only looks at overall brightness distribution.

    Parameters
    ----------
    rgb_images : np.ndarray (N, H, W, 3) original color images (0-255, uint8)
                 (use the images BEFORE grayscale preprocessing)
    gray_images : np.ndarray (N, H, W) grayscale images (0-255, uint8)
                  (the output of preprocess_pipeline())

    Returns
    -------
    np.ndarray (N, D) feature matrix
    """
    import cv2

    hog = cv2.HOGDescriptor(
        _winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8),
        _cellSize=(8, 8), _nbins=9,
    )

    feats = []
    for rgb, gray in zip(rgb_images, gray_images):
        # 1. Color histograms (captures staining differences between classes)
        color_hist = []
        for ch in range(3):
            h, _ = np.histogram(rgb[:, :, ch], bins=16, range=(0, 255))
            color_hist.append(h / (h.sum() + 1e-8))
        color_hist = np.concatenate(color_hist)

        # 2. Grayscale intensity histogram
        gh, _ = np.histogram(gray, bins=32, range=(0, 255))
        gh = gh / (gh.sum() + 1e-8)

        # 3. Basic statistics
        stats = np.array([gray.mean(), gray.std(), gray.min(), gray.max()])

        # 4. Edge density (cell boundary / nuclei density proxy)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.array([edges.mean() / 255.0])

        # 5. Texture sharpness
        lap_var = np.array([cv2.Laplacian(gray, cv2.CV_64F).var()])

        # 6. HOG shape/texture descriptor
        resized = cv2.resize(gray, (64, 64))
        hog_feat = hog.compute(resized).flatten()

        feats.append(np.concatenate(
            [color_hist, gh, stats, edge_density, lap_var, hog_feat]
        ))

    return np.array(feats)