"""
preprocessing.py
-----------------
Implements the two preprocessing steps described in the thesis
(Sections 3.2.1 and 3.2.2):

    1. Grayscale Transformation  -> O = T(i)
    2. Sigmoid Stretching        -> S(x) = 1 / (1 + e^-y(x - z))
"""

import numpy as np
import cv2


def grayscale_transform(images):
    """
    Converts an array of RGB/BGR images to single-channel grayscale.

    Parameters
    ----------
    images : np.ndarray (N, H, W, 3)

    Returns
    -------
    np.ndarray (N, H, W) grayscale images
    """
    gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images])
    return gray


def sigmoid_stretch(images, y=10.0, z=0.5):
    """
    Applies sigmoid contrast stretching, Eq. (3.2) in the thesis:

        S(x) = 1 / (1 + e^(-y * (x - z)))

    Parameters
    ----------
    images : np.ndarray, pixel values in [0, 255] or already normalized [0,1]
    y : float
        Controls steepness of the S-curve (contrast strength).
    z : float
        Midpoint of the curve (brightness level around which contrast
        is stretched). z is in normalized [0,1] pixel space.

    Returns
    -------
    np.ndarray, same shape, contrast-stretched, scaled back to [0,255] uint8
    """
    imgs = images.astype(np.float64) / 255.0
    stretched = 1.0 / (1.0 + np.exp(-y * (imgs - z)))
    # re-normalize to [0,255]
    stretched = (stretched - stretched.min()) / (stretched.max() - stretched.min() + 1e-8)
    return (stretched * 255).astype(np.uint8)


def preprocess_pipeline(images, sigmoid_y=10.0, sigmoid_z=0.5):
    """
    Full preprocessing pipeline used in Chapter 3 (boosting) experiment:
    grayscale -> sigmoid stretch.

    Returns grayscale, contrast-enhanced images ready for feature extraction.
    """
    gray = grayscale_transform(images)
    stretched = sigmoid_stretch(gray, y=sigmoid_y, z=sigmoid_z)
    return stretched


# ---------------------------------------------------------------------------
# Chapter 4 (bagging) preprocessing: Gaussian Filter + Adaptive Histogram
# Equalization (AHE), as described in thesis Sections 4.1.1 and 4.1.2.
# These are separate, additive functions -- they do NOT replace or modify
# anything used by the Chapter 3 boosting pipeline above.
# ---------------------------------------------------------------------------

def gaussian_filter(images, kernel_size=5, sigma=1.0):
    """
    Applies 2D Gaussian smoothing/denoising (thesis Eq. 4.2), as used in
    Chapter 4 before classification.

    Parameters
    ----------
    images : np.ndarray (N, H, W, 3) color images, uint8
    kernel_size : int (odd number) -- Gaussian kernel size
    sigma : float -- standard deviation of the Gaussian

    Returns
    -------
    np.ndarray (N, H, W, 3), same dtype/shape, denoised
    """
    import cv2
    out = np.array([
        cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma)
        for img in images
    ])
    return out


def adaptive_histogram_equalization(images, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Adaptive Histogram Equalization (thesis Sec 4.1.2: image split
    into 8x8 tiles, histogram equalized locally). Implemented with OpenCV's
    CLAHE (Contrast Limited AHE), applied on the L-channel in LAB color
    space so color information is preserved while local contrast is
    enhanced (the thesis's stated goal).

    Parameters
    ----------
    images : np.ndarray (N, H, W, 3) color images, uint8
    clip_limit : float -- contrast limiting threshold (prevents noise
                 amplification, as CLAHE is described as a safer AHE variant)
    tile_grid_size : tuple -- matches the thesis's 8x8 tiling

    Returns
    -------
    np.ndarray (N, H, W, 3), same dtype/shape, contrast-enhanced
    """
    import cv2
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    out = []
    for img in images:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        merged = cv2.merge((l_eq, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        out.append(enhanced)
    return np.array(out)


def preprocess_pipeline_bagging(images, gaussian_kernel=5, gaussian_sigma=1.0,
                                 clahe_clip=2.0, clahe_tile=(8, 8)):
    """
    Full preprocessing pipeline used in Chapter 4 (bagging) experiment:
    Gaussian filter -> Adaptive Histogram Equalization (thesis Fig. 4.1 /
    Fig. 4.2).

    Parameters
    ----------
    images : np.ndarray (N, H, W, 3) color images, uint8

    Returns
    -------
    rgb_enhanced : np.ndarray (N, H, W, 3) color images, uint8 -- denoised
                   + contrast-enhanced, ready for CNN feature extraction
    gray_enhanced : np.ndarray (N, H, W) grayscale version of the same,
                    for classical feature extraction (classical_features_rich)
    """
    import cv2
    denoised = gaussian_filter(images, kernel_size=gaussian_kernel, sigma=gaussian_sigma)
    enhanced = adaptive_histogram_equalization(denoised, clip_limit=clahe_clip,
                                                tile_grid_size=clahe_tile)
    gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in enhanced])
    return enhanced, gray