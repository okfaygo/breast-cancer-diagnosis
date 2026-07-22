"""
Build (and optionally upload) the Hugging Face Space for this project.

The Space needs only three runtime files — app.py, inference.py and
model_bundle.npz — plus its own README.md (with HF metadata) and a minimal
requirements.txt. This script assembles them into ./space/ so there is a single
source of truth at the repo root and the copies can't drift.

Build only:
    python deploy_hf.py

Build and upload (requires `hf auth login` first):
    python deploy_hf.py --repo-id YOUR_USERNAME/breast-cancer-diagnosis
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SPACE_DIR = Path("space")
RUNTIME_FILES = ["app.py", "inference.py", "model_bundle.npz"]

# sdk_version is deliberately omitted so Hugging Face uses its current default
# Streamlit — pinning a version it doesn't offer is a common first-deploy failure.
README = """---
title: Breast Cancer Diagnosis
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
---

# Breast cancer diagnosis

Interactive demo of a neural network that classifies breast tumors as malignant or
benign, trained on the Wisconsin Diagnostic Breast Cancer dataset (569 samples,
30 features).

> **Educational demo only — not a medical device and not for clinical use.**
> Trained on a small public research dataset to practise machine learning.

## About

- **Model:** feedforward network (30 - 64 - 32 - 1) with dropout, trained in Keras.
- **Cross-validated performance:** recall 0.96 (± 0.04), AUC 0.994 over 5 stratified
  folds — not the flattering single-split score.
- **Decision threshold:** defaults to 0.35 rather than 0.5, tuned to favour recall,
  because a missed cancer is costlier than a false alarm. The threshold is adjustable.
- **Inference:** runs in pure NumPy from a 20 KB weight bundle, so the Space starts
  in under a second without TensorFlow.

Try the example cases — including "the hard case", a small but genuinely malignant
tumor that tree-based models misclassify.

Full training pipeline and analysis:
[github.com/okfaygo/breast-cancer-diagnosis](https://github.com/okfaygo/breast-cancer-diagnosis)
"""

REQUIREMENTS = "streamlit\nnumpy\n"


def build() -> None:
    SPACE_DIR.mkdir(exist_ok=True)
    for name in RUNTIME_FILES:
        source = Path(name)
        if not source.exists():
            raise SystemExit(f"missing {name} — run `python export_bundle.py` first")
        shutil.copy2(source, SPACE_DIR / name)
    (SPACE_DIR / "README.md").write_text(README, encoding="utf-8")
    (SPACE_DIR / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")

    total_kb = sum((SPACE_DIR / f).stat().st_size for f in RUNTIME_FILES) / 1024
    print(f"built {SPACE_DIR}/ — {len(RUNTIME_FILES) + 2} files, {total_kb:.1f} KB of runtime assets")
    for path in sorted(SPACE_DIR.iterdir()):
        print(f"  {path.name}")


def upload(repo_id: str, private: bool) -> None:
    from huggingface_hub import create_repo, get_token, upload_folder

    if not get_token():
        raise SystemExit("not logged in — run `hf auth login` first")

    create_repo(repo_id, repo_type="space", space_sdk="streamlit",
                private=private, exist_ok=True)
    upload_folder(folder_path=str(SPACE_DIR), repo_id=repo_id, repo_type="space",
                  commit_message="Deploy breast cancer diagnosis app")
    print(f"\ndeployed → https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", help="e.g. yourname/breast-cancer-diagnosis")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    args = parser.parse_args()

    build()
    if args.repo_id:
        upload(args.repo_id, args.private)
    else:
        print("\nbuild only — pass --repo-id USERNAME/SPACE to upload")


if __name__ == "__main__":
    main()
