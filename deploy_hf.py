"""
Build (and optionally upload) the Hugging Face Space for this project.

The Space needs the app's runtime files (app.py, the views/ pages, the shared
loaders and the two .npz data files) plus its own Dockerfile, README.md (with HF
metadata) and a minimal requirements.txt. This script assembles them into ./space/
so there is a single source of truth at the repo root and the copies can't drift.

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
RUNTIME_FILES = ["app.py", "appdata.py", "inference.py", "model_bundle.npz", "about_data.npz"]
RUNTIME_DIRS = ["views"]  # copied recursively, minus Python caches

# Hugging Face's API no longer accepts "streamlit" as a Space SDK (only gradio,
# docker or static), so the Streamlit app runs inside a Docker Space. app_port tells
# HF which port the container serves on.
README = """---
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
"""

REQUIREMENTS = "streamlit\nnumpy\n"

# Runs as a non-root user (HF Spaces convention) with a writable HOME so Streamlit
# can create its config/cache. Binds 0.0.0.0:8501 to match app_port in the README.
DOCKERFILE = """FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \\
    PATH=/home/user/.local/bin:$PATH \\
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""


def build() -> None:
    SPACE_DIR.mkdir(exist_ok=True)
    for name in RUNTIME_FILES:
        source = Path(name)
        if not source.exists():
            raise SystemExit(
                f"missing {name} — run `python export_bundle.py` and "
                f"`python compute_about_data.py` first"
            )
        shutil.copy2(source, SPACE_DIR / name)
    for name in RUNTIME_DIRS:
        source = Path(name)
        if not source.exists():
            raise SystemExit(f"missing {name}/ directory")
        shutil.copytree(source, SPACE_DIR / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                        dirs_exist_ok=True)
    (SPACE_DIR / "README.md").write_text(README, encoding="utf-8")
    (SPACE_DIR / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (SPACE_DIR / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")

    print(f"built {SPACE_DIR}/")
    for path in sorted(SPACE_DIR.rglob("*")):
        if "__pycache__" not in path.parts and path.is_file():
            print(f"  {path.relative_to(SPACE_DIR)}  ({path.stat().st_size / 1024:.1f} KB)")


def upload(repo_id: str, private: bool) -> None:
    from huggingface_hub import create_repo, get_token, upload_folder

    if not get_token():
        raise SystemExit("not logged in — run `hf auth login` first")

    create_repo(repo_id, repo_type="space", space_sdk="docker",
                private=private, exist_ok=True)
    upload_folder(folder_path=str(SPACE_DIR), repo_id=repo_id, repo_type="space",
                  commit_message="Deploy breast cancer diagnosis app",
                  ignore_patterns=["__pycache__/*", "*.pyc"])
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
