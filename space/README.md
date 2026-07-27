---
title: Breast Cancer Diagnosis
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# Breast cancer diagnosis

Interactive demo of a neural network that classifies breast tumors as malignant or
benign, trained on the Wisconsin Diagnostic Breast Cancer dataset (569 samples,
30 features).

> **Educational demo only — not a medical device and not for clinical use.**
> Trained on a small public research dataset to practise machine learning.

## About

Two pages: an interactive **Predict** page, and an **About the model** page with the
ROC curve, confusion matrix, cross-validation breakdown and feature importance.

- **Model:** feedforward network (30 - 64 - 32 - 1) with dropout, trained in Keras.
- **Cross-validated performance:** recall 0.96 (± 0.03), AUC 0.994 over 5 stratified
  folds — not the flattering single-split score.
- **Decision threshold:** defaults to 0.35 rather than 0.5, tuned to favour recall,
  because a missed cancer is costlier than a false alarm. The threshold is adjustable.
- **Inference:** runs in pure NumPy from a ~20 KB weight bundle, so the Space starts
  in under a second without TensorFlow.

Try the example cases — including "the hard case", a small but genuinely malignant
tumor that tree-based models misclassify.

Full training pipeline and analysis:
[github.com/okfaygo/breast-cancer-diagnosis](https://github.com/okfaygo/breast-cancer-diagnosis)
