"""
data_loader.py
---------------
Loads the LC25000 histopathological image dataset.

Expected folder structure (as downloaded from Kaggle:
"Lung and Colon Cancer Histopathological Images"):

    lung_colon_image_set/
        lung_image_sets/
            lung_aca/      -> Lung Adenocarcinoma   (Malignant)
            lung_scc/      -> Lung Squamous Cell Ca. (Malignant)
            lung_n/        -> Lung Benign (Normal)

For the thesis's Chapter-3 boosting experiment, only the *lung* subset is
used and the 3 lung sub-classes are collapsed into a binary label:
    Malignant = lung_aca + lung_scc
    Benign    = lung_n
(If you want the full 5-class Colon+Lung setup used in Chapter 4, set
`binary=False` and pass the parent `lung_colon_image_set` folder — all
5 sub-folders will then be used as separate classes.)
"""

import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GroupShuffleSplit

LUNG_CLASS_MAP = {
    "lung_aca": "Malignant",
    "lung_scc": "Malignant",
    "lung_n": "Benign",
}

BINARY_LABEL_MAP = {"Benign": 0, "Malignant": 1}


def _list_images(folder, exts=(".jpeg", ".jpg", ".png")):
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)


def load_lc25000(
    dataset_root,
    binary=True,
    image_size=(224, 224),
    max_per_class=None,
):
    """
    Parameters
    ----------
    dataset_root : str OR list[str]
        Path(s) to folder(s) that directly contain the class sub-folders
        (lung_aca/, lung_n/, lung_scc/, ...).
        - Pass a single string for one split, e.g. ".../Test Set".
        - Pass a LIST of strings to merge multiple splits into one dataset,
          e.g. [".../Test Set", ".../Train and Validation Set"] -- this is
          how you load the FULL lung dataset (~15,000 images) instead of
          just one pre-made split.
    binary : bool
        If True -> Malignant vs Benign (matches thesis Chapter 3).
        If False -> full 5-class problem (matches thesis Chapter 4 conf. matrix).
    image_size : tuple
        Resize target (as done in thesis preprocessing).
    max_per_class : int or None
        Cap images per class (applied PER ROOT FOLDER, then combined) for
        quick local testing (use None for full run).

    Returns
    -------
    X : np.ndarray  (N, H, W, 3) uint8 images
    y : np.ndarray  (N,) integer labels
    class_names : list[str]
    """
    # normalize to a list so single-folder and multi-folder (merged) loading
    # share the same code path
    roots = [dataset_root] if isinstance(dataset_root, str) else list(dataset_root)

    X, y = [], []

    if binary:
        class_names = ["Benign", "Malignant"]
        for subfolder, label_name in LUNG_CLASS_MAP.items():
            for root in roots:
                folder = os.path.join(root, subfolder)
                files = _list_images(folder)
                if max_per_class:
                    files = files[:max_per_class]
                for f in files:
                    img = cv2.imread(f)
                    if img is None:
                        continue  # skip unreadable/corrupt files
                    img = cv2.resize(img, image_size)
                    X.append(img)
                    y.append(BINARY_LABEL_MAP[label_name])
    else:
        # 5-class: colon_aca, colon_n, lung_aca, lung_n, lung_scc
        # discover subfolders from the first root (all roots are expected
        # to share the same class sub-folder names)
        subfolders = sorted(
            [d for d in os.listdir(roots[0])
             if os.path.isdir(os.path.join(roots[0], d))]
        )
        class_names = subfolders
        label_map = {name: i for i, name in enumerate(subfolders)}
        for subfolder in subfolders:
            for root in roots:
                folder = os.path.join(root, subfolder)
                files = _list_images(folder)
                if max_per_class:
                    files = files[:max_per_class]
                for f in files:
                    img = cv2.imread(f)
                    if img is None:
                        continue
                    img = cv2.resize(img, image_size)
                    X.append(img)
                    y.append(label_map[subfolder])

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int64)
    return X, y, class_names


def split_data(X, y, test_size=0.2, seed=42):
    """
    Plain random 80/20 split (as specified in the thesis, Sec 3.3).

    NOTE: LC25000 is built by heavily augmenting (rotating/flipping) a
    small number of original source images. A plain random split can put
    near-duplicate (rotated/flipped) versions of the SAME source image in
    both train and test, which leaks information and inflates accuracy
    toward 100%. Use `group_split()` below for a more honest, leakage-free
    evaluation.
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def _average_hash(img_bgr, hash_size=8):
    """
    Simple 64-bit perceptual "average hash" (aHash) of an image, used to
    detect near-duplicate (e.g. rotated/flipped-but-similar-brightness)
    images. Not rotation-invariant in a strict sense, but in practice
    LC25000's augmented copies tend to collide or land very close together
    under this hash, which is enough to catch most train/test leakage.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = small.mean()
    bits = (small > avg).astype(np.uint8).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def group_split(X, y, test_size=0.2, seed=42, hash_size=8):
    """
    Leakage-aware 80/20 split: images are first fingerprinted with a
    perceptual hash, and all images sharing the same hash (near-duplicates,
    e.g. rotated/flipped copies of the same source image) are kept
    together on the SAME side of the split (either all in train, or all
    in test) — never split across both.

    This gives a more honest, defensible "unseen data" accuracy estimate
    than a plain random split on an augmented dataset like LC25000.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    groups = np.array([_average_hash(img, hash_size) for img in X])
    n_unique_groups = len(np.unique(groups))
    print(f"   [group_split] {len(X)} images -> {n_unique_groups} unique "
          f"near-duplicate groups (dataset has significant augmentation "
          f"redundancy if this number is much smaller than {len(X)})")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]