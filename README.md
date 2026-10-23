# Lung Cancer Detection — Boosting Ensemble (SVM + RF + MLP + LR)

Implementation of the **Chapter 3** approach from the thesis
*"Optimization Driven Ensemble-based Machine Learning Approach for
Lung Cancer Detection"*: a sequential boosting ensemble of SVM, Random
Forest, MLP, and Logistic Regression, tuned with Bayesian, Hyperband,
and BOHB hyperparameter optimization, on the LC25000 histopathology dataset.

## 1. Setup

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using the virtual environment's Python directly avoids PowerShell execution
policy errors when activating `Activate.ps1`.

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Get the dataset

Download **"Lung and Colon Cancer Histopathological Images (LC25000)"**
from Kaggle:
https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

Unzip it — you'll get a folder like:
```
lung_colon_image_set/
    lung_image_sets/
        lung_aca/
        lung_n/
        lung_scc/
    colon_image_sets/
        colon_aca/
        colon_n/
```

## 3. Run

**Quick smoke test** (no GPU, no internet after download, ~1 min,
uses a tiny classical feature extractor — just to confirm the
pipeline runs on your machine):
```bash
python main.py --data_root path/to/lung_colon_image_set/lung_image_sets --quick
```

**Full run** (matches the thesis setup — VGG16 CNN features, full
dataset, 30 Optuna trials per optimizer; needs internet on first run
to download ImageNet weights, GPU strongly recommended):
```bash
python main.py --data_root path/to/lung_colon_image_set/lung_image_sets
```

Results (confusion matrices + optimization comparison chart) are saved to `outputs/`.

## 4. Project structure

```
main.py                    # orchestrates the full pipeline
src/
  data_loader.py           # LC25000 loading + train/test split
  preprocessing.py         # grayscale transform + sigmoid stretching (Sec 3.2)
  feature_extraction.py    # VGG16 CNN features / classical fallback
  boosting_ensemble.py     # SequentialBoostingEnsemble (RF->MLP->LR->SVM)
  optimization.py          # Bayesian / Hyperband / BOHB tuning (Optuna)
  evaluate.py              # metrics + confusion matrix + comparison plots
```

## 5. Notes / decisions made where the thesis was ambiguous

- **Feature extractor**: the thesis mentions Keras/TensorFlow for
  "Model Implementation" but doesn't specify the exact feature
  extraction method for the classical ML models. This code uses
  **VGG16 (ImageNet) bottleneck features** — the standard, reproducible
  transfer-learning approach for this kind of task. Swap this out in
  `src/feature_extraction.py` if your actual thesis experiments used
  something else (e.g. raw flattened pixels, GLCM texture features).
- **"Boosting" mechanics**: implemented as classic AdaBoost.M1-style
  sequential reweighting across RF → MLP → LR → SVM, combined via a
  weighted vote. If your thesis defined boosting differently (e.g. a
  stacking meta-learner), adjust `src/boosting_ensemble.py` accordingly.
- **BOHB**: Optuna has no single built-in "BOHB" object; BOHB *is*
  formally the combination of a Bayesian (TPE) sampler with Hyperband's
  successive-halving pruner — that's exactly what `run_bohb()` does.
- Numbers you get from a real run **will differ** from the thesis's
  96.5% — those depend on the exact dataset split, feature extractor,
  and hyperparameter search space, none of which are fully specified in
  the document. Treat this code as a faithful, working reproduction of
  the *architecture*, not a byte-for-byte reproduction of the reported
  numbers.

## 6. Next steps

- Run `--quick` first to confirm everything works end-to-end on your machine.
- Then run the full version and compare your confusion matrices /
  accuracy numbers against thesis Figures 3.5–3.9.
- If your numbers differ meaningfully, the usual causes are: image
  preprocessing parameters (`sigmoid_y`, `sigmoid_z` in
  `preprocessing.py`), feature extractor choice, or hyperparameter
  search ranges (`optimization.py`).
