# Breast Cancer Diagnosis — End-to-End ML Pipeline

An end-to-end machine-learning pipeline that classifies breast tumors as **malignant** or **benign**,
built with TensorFlow/Keras. The project is structured as a learning journey through the full ML
lifecycle: data loading, exploratory analysis, preprocessing, model building, evaluation, error
analysis, and ensembling.

> **Status:** two working Jupyter notebooks plus a Streamlit web app that serves the trained model.

---

## Why there are two notebooks

This repository contains **two** pipeline notebooks that share almost identical code but use **different
datasets**. That is a deliberate choice, and the story behind it is itself one of the project's main lessons.

### 1. `breast_cancer_pipeline.ipynb` — the "Enhanced" dataset

The first notebook uses the Kaggle [Enhanced Breast Cancer Diagnostic Dataset](https://www.kaggle.com/datasets/shivasingh4945/enhanced-breast-cancer-diagnostic-dataset)
(5,500 samples, 15 features). The pipeline runs end-to-end and scores a **near-perfect ~100% accuracy**.

That sounds great, but a perfect score on real medical data is a **warning sign**, not a trophy. On
investigation, the perfection is not a bug in our code — the split is correctly stratified, the scaler is
fit on training data only, and there are no duplicate rows leaking between train and test. The cause is the
**dataset itself**: its engineered features (e.g. `malignancy_risk_score`, `tumor_aggressiveness`, and
several interaction terms) make the two classes almost perfectly separable. Even a plain Random Forest hits
~99.5% accuracy with an AUC of ~1.0. The model is already at the dataset's ceiling — which means there is
**no meaningful model-improvement work to practice on it**. Tuning would just be polishing a number that
reflects the data, not the model.

### 2. `breast_cancer_pipeline_wdbc.ipynb` — the original WDBC dataset

To have something real to learn from, the second notebook swaps in the **original Wisconsin Diagnostic
Breast Cancer (WDBC)** dataset — the classic teaching set the "enhanced" data was derived from. It ships
inside scikit-learn (`sklearn.datasets.load_breast_cancer`), so there's no download. It has **569 samples
and 30 genuine measurements** (no pre-engineered "risk score" features), and lands around **96–100%
accuracy with real, studyable errors**. That makes it the right place to practice model iteration.

**In short:** the enhanced dataset is *too easy* to teach model improvement; WDBC rewards it. Keeping both
preserves the finished work and provides a genuine learning surface.

> **Gotcha worth knowing:** scikit-learn encodes the WDBC target as `0 = malignant, 1 = benign` — the
> opposite of this project's `Malignant = 1` convention. The WDBC notebook flips it at load time so that
> Malignant = 1 everywhere, keeping the two notebooks consistent.

---

## Pipeline stages

Both notebooks follow the same structure; the WDBC notebook extends it with two extra analysis stages.

| Stage | What it covers |
|-------|----------------|
| 1. Setup & data loading | Imports, reproducibility seeds, load data |
| 2. Exploratory data analysis | Class balance, feature distributions, correlations |
| 3. Preprocessing & splitting | Label encoding, 70/15/15 stratified split, `StandardScaler` (fit on train only), class weights |
| 4. Model building | Feedforward neural net (2 hidden layers, dropout, sigmoid output) |
| 5. Evaluation & tuning | Classification report, confusion matrix, ROC-AUC, threshold tuning |
| 6. Save & interpret | Save model + scaler, permutation feature importance |
| 7. Error analysis *(WDBC only)* | Find the hard cases, compare against Random Forest / Gradient Boosting, profile misclassified tumors |
| 8. Ensemble *(WDBC only)* | Soft-voting ensemble + threshold tuning to recover a missed malignant case |

### Key modelling choices

- **Framework:** TensorFlow / Keras feedforward network with dropout + `EarlyStopping`
- **Split:** 70 / 15 / 15 stratified train / validation / test
- **No data leakage:** `StandardScaler` is fit only on the training set, then applied to validation and test
- **Class imbalance:** handled with class weights (~63% benign / ~37% malignant) rather than resampling
- **Evaluation focus:** in a medical setting a *false negative* (missed cancer) is the costly error, so we
  watch **recall** more closely than raw accuracy

### Highlights from the WDBC error analysis

- The neural net gets a clean sweep on the test split, but the tree models each miss the *same* genuinely
  hard tumor — a small, only-mildly-irregular malignancy that sits closer to the benign cluster.
- Different model families disagree precisely on the ambiguous cases, which is a caution against trusting a
  single model or a single split.
- A soft-voting **ensemble combined with threshold tuning** (threshold ≈ 0.35) catches that hard case at
  **recall 1.000 with precision 1.000** — a result neither ensembling nor threshold tuning achieves alone.

---

## The web app

A [Streamlit](https://streamlit.io) app ([`app.py`](app.py)) serves the WDBC model interactively, with
two pages:

- **Predict** — enter tumor measurements (or load an example) and get a malignancy probability.
- **About the model** — the ROC curve, confusion matrix, cross-validation breakdown and feature
  importance, so a visitor can judge how good the model actually is.

**Design decisions worth knowing:**

- **It serves the WDBC model, not the enhanced one.** The enhanced model's inputs include
  `malignancy_risk_score` — asking a user to supply that in order to predict malignancy would be circular.
- **30 features is an unusable form.** The Predict page shows the six highest-importance features as
  sliders, defaults the other 24 to the dataset median behind an expander, and offers one-click example
  cases loaded from real dataset rows — including *the hard case* that the tree models misclassify.
- **Sliders are bounded** by each feature's real min/max, so physically impossible tumors can't be entered.
- **The decision threshold is exposed, not hidden.** It defaults to the recall-favouring 0.35 from
  Stage 8, with the trade-off explained in the UI.
- **It reports cross-validated metrics** (recall 0.96), not the flattering single-split score.
- **A permanent "not a medical device" warning** is always visible.

### No TensorFlow at inference

The trained network is a small MLP (30 → 64 → 32 → 1, ~4,100 parameters), so a forward pass is three
matrix multiplies. [`export_bundle.py`](export_bundle.py) exports the weights, the scaler parameters,
the feature ranges and the example rows into a **~20 KB** `model_bundle.npz`, and
[`inference.py`](inference.py) runs the forward pass in pure NumPy. Separately,
[`compute_about_data.py`](compute_about_data.py) precomputes the About page's metrics (ROC points,
confusion matrices, cross-validation results, permutation importance) into a small `about_data.npz`, so
the charts render with no heavy libraries either.

The deployed app therefore needs only `streamlit` and `numpy` — no TensorFlow, scikit-learn or joblib —
so it cold-starts in under a second instead of tens of seconds. The exporter verifies the NumPy output
matches Keras across all 569 samples (max difference ~6e-08) and refuses to write a mismatched bundle.

---

## Repository structure

```
breast-cancer-diagnosis/
├── breast_cancer_pipeline.ipynb          # Notebook 1 — Enhanced dataset (near-perfect, "too easy")
├── breast_cancer_pipeline_wdbc.ipynb     # Notebook 2 — Real WDBC dataset (error analysis, ensemble, CV)
├── dataset/
│   └── breast_cancer_enhanced_dataset.csv  # Data for notebook 1 (WDBC loads from scikit-learn)
├── app.py                                # Streamlit web app — entry point (navigation + disclaimer)
├── views/
│   ├── predict.py                        # Predict page
│   └── about.py                          # About-the-model page (charts)
├── appdata.py                            # Shared cached loaders for the pages
├── inference.py                          # Dependency-free NumPy forward pass
├── export_bundle.py                      # Keras model + scaler -> model_bundle.npz (with parity check)
├── compute_about_data.py                 # Precomputes About-page metrics -> about_data.npz
├── deploy_hf.py                          # Builds (and uploads) the Hugging Face Space
├── model_bundle.npz                      # ~20 KB weights + scaler + feature ranges + examples
├── about_data.npz                        # Precomputed ROC / confusion / CV / importance data
├── space/                                # Deployable Hugging Face Space (generated by deploy_hf.py)
├── breast_cancer_model.keras             # Saved model — enhanced
├── best_model.keras                      # Best-checkpoint model — enhanced
├── scaler.pkl                            # Fitted StandardScaler — enhanced
├── breast_cancer_model_wdbc.keras        # Saved model — WDBC
├── best_model_wdbc.keras                 # Best-checkpoint model — WDBC
├── scaler_wdbc.pkl                       # Fitted StandardScaler — WDBC
├── requirements.txt
└── README.md
```

---

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter and open a notebook

```bash
jupyter notebook
```

- `breast_cancer_pipeline_wdbc.ipynb` needs no data download — it loads WDBC from scikit-learn.
- `breast_cancer_pipeline.ipynb` reads `dataset/breast_cancer_enhanced_dataset.csv` (included).

Run the cells top to bottom. Each notebook saves its trained model (`*.keras`) and fitted scaler
(`*.pkl`) so predictions can be reproduced later without retraining.

### 3. Run the web app

```bash
python -m streamlit run app.py
```

It opens at `http://localhost:8501`. If you retrain the model, regenerate both data artifacts first:

```bash
python export_bundle.py
python compute_about_data.py
```

### 4. Deploy to Hugging Face Spaces

Log in once (this opens a browser to create/paste an access token):

```bash
hf auth login
```

Then build the Space folder and upload it in one step:

```bash
python deploy_hf.py --repo-id YOUR_USERNAME/breast-cancer-diagnosis
```

Add `--private` to keep the Space unlisted. Running `deploy_hf.py` with no arguments just rebuilds
`space/` locally without uploading anything.

### Reproducibility

The WDBC notebook enables full TensorFlow determinism
(`keras.utils.set_random_seed` + `tf.config.experimental.enable_op_determinism()`), so every result —
including which specific tumor is misclassified — is **identical on every run**. Without this, neural-net
training is noisy and the error-analysis narrative would drift from run to run.

---

## Requirements

- Python 3.9+
- **Development** (notebooks, training, exporting): TensorFlow, pandas, NumPy, matplotlib, seaborn,
  scikit-learn, joblib, Streamlit — see [`requirements.txt`](requirements.txt)
- **Deployed app**: `streamlit` and `numpy` only — see [`space/requirements.txt`](space/requirements.txt)

---

## Roadmap / next steps

- **Batch CSV upload** — score many tumors at once and download the results
- **Model-agreement panel** — show the neural net, random forest and gradient boosting predictions
  side by side, which is far more honest about uncertainty on borderline cases
- **Hyperparameter tuning** (e.g. `keras_tuner`), evaluated with the Stage 9 CV loop rather than a single split
- **Explainability** with SHAP (`shap.DeepExplainer`) to explain individual predictions
- **Probability calibration** before trusting any threshold more seriously

Done so far: two-notebook pipeline with error analysis, ensembling and cross-validation; a NumPy-served
Streamlit app deployed to Hugging Face Spaces; and a two-page UI with an About-the-model page.

---

## Dataset credits

- **Enhanced Breast Cancer Diagnostic Dataset** by *shivasingh4945* on Kaggle
- **Wisconsin Diagnostic Breast Cancer (WDBC)** — bundled with scikit-learn; originally from the UCI
  Machine Learning Repository
