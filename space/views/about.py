"""About-the-model page — how the network was built and how well it actually does."""

import numpy as np
import streamlit as st

from appdata import DEFAULT_THRESHOLD, get_about_data

d = get_about_data()

st.subheader("About the model")
st.markdown(
    "A feedforward neural network (multilayer perceptron) that classifies a tumor as "
    "malignant or benign from 30 cell-nucleus measurements. Everything below is measured "
    "on held-out data — see the [training notebook]"
    "(https://github.com/okfaygo/breast-cancer-diagnosis) for the full pipeline."
)

st.markdown("#### Architecture")
arch_cols = st.columns(4)
arch_cols[0].metric("Inputs", "30", help="cell-nucleus measurements")
arch_cols[1].metric("Hidden layers", "2", help="64 and 32 units, ReLU, dropout 0.3")
arch_cols[2].metric("Parameters", "~4,100")
arch_cols[3].metric("Output", "1", help="sigmoid — probability of malignancy")
st.caption("30 → 64 → 32 → 1, trained in Keras with class weights and early stopping.")

st.divider()

# --- ROC curve (single held-out test split) ---
st.markdown("#### ROC curve")
st.caption(
    f"On the held-out test set ({int(d['test_pos'])} malignant, {int(d['test_neg'])} benign). "
    "The curve hugging the top-left corner means the model separates the classes well."
)
roc_points = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(d["roc_fpr"], d["roc_tpr"])]
diagonal = [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]
st.vega_lite_chart(
    {
        "height": 300,
        "layer": [
            {
                "data": {"values": diagonal},
                "mark": {"type": "line", "strokeDash": [4, 4], "color": "#9aa0a6"},
                "encoding": {
                    "x": {"field": "fpr", "type": "quantitative"},
                    "y": {"field": "tpr", "type": "quantitative"},
                },
            },
            {
                "data": {"values": roc_points},
                "mark": {"type": "line", "color": "#e24b4a", "strokeWidth": 2.5},
                "encoding": {
                    "x": {"field": "fpr", "type": "quantitative",
                          "title": "False positive rate", "scale": {"domain": [0, 1]}},
                    "y": {"field": "tpr", "type": "quantitative",
                          "title": "True positive rate (recall)", "scale": {"domain": [0, 1]}},
                },
            },
        ],
    },
    use_container_width=True,
)
st.caption(f"Test-set AUC: **{float(d['roc_auc']):.3f}** (1.0 is perfect separation).")

st.divider()

# --- Confusion matrix at the default threshold ---
st.markdown("#### Confusion matrix")
st.caption(
    f"At the default decision threshold of {DEFAULT_THRESHOLD}, on the held-out test set. "
    "The costly error is a false negative — a malignant tumor called benign."
)
cm = d["cm_035"]
tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])


def cell(count: int, label: str, good: bool) -> str:
    color = "var(--text-success)" if good else "var(--text-error)"
    return (
        f"<div style='padding:14px;border:1px solid rgba(128,128,128,0.3);border-radius:8px;"
        f"text-align:center'><div style='font-size:26px;font-weight:600;color:{color}'>{count}"
        f"</div><div style='font-size:12px;opacity:0.75'>{label}</div></div>"
    )


grid = st.columns(2)
grid[0].markdown(cell(tn, "True negative (benign ✓)", True), unsafe_allow_html=True)
grid[1].markdown(cell(fp, "False positive (false alarm)", False), unsafe_allow_html=True)
grid2 = st.columns(2)
grid2[0].markdown(cell(fn, "False negative (missed cancer)", False), unsafe_allow_html=True)
grid2[1].markdown(cell(tp, "True positive (malignant ✓)", True), unsafe_allow_html=True)

st.divider()

# --- Cross-validation (the honest, generalisation estimate) ---
st.markdown("#### Cross-validation")
st.caption(
    "5-fold stratified cross-validation — the trustworthy estimate, since it evaluates "
    "every sample once instead of relying on a single lucky split. Mean ± standard "
    "deviation across folds."
)
models = [str(m) for m in d["cv_model_names"]]
metric_keys = [("cv_accuracy", "Accuracy"), ("cv_recall", "Recall"),
               ("cv_precision", "Precision"), ("cv_auc", "AUC")]

header = "| Model | " + " | ".join(label for _, label in metric_keys) + " |"
divider = "|" + "---|" * (len(metric_keys) + 1)
rows = []
for col, model in enumerate(models):
    cells = []
    for key, _ in metric_keys:
        values = d[key][:, col]
        cells.append(f"{values.mean():.3f} ± {values.std():.3f}")
    rows.append(f"| {model} | " + " | ".join(cells) + " |")
st.markdown("\n".join([header, divider, *rows]))

st.caption("Per-fold recall (malignant) — the spread is why a single split can't be trusted:")
recall_rows = []
for fold in range(d["cv_recall"].shape[0]):
    for col, model in enumerate(models):
        recall_rows.append({
            "fold": f"Fold {fold + 1}",
            "model": model,
            "recall": round(float(d["cv_recall"][fold, col]), 3),
        })
st.vega_lite_chart(
    {
        "height": 260,
        "data": {"values": recall_rows},
        "mark": "bar",
        "encoding": {
            "x": {"field": "fold", "type": "nominal", "title": None},
            "xOffset": {"field": "model"},
            "y": {"field": "recall", "type": "quantitative",
                  "scale": {"domain": [0.7, 1.0]}, "title": "Malignant recall"},
            "color": {"field": "model", "type": "nominal",
                      "legend": {"orient": "bottom", "title": None}},
        },
    },
    use_container_width=True,
)

st.divider()

# --- Feature importance ---
st.markdown("#### Which measurements matter most")
st.caption(
    "Permutation importance for the neural net: how much test accuracy drops when each "
    "feature is shuffled. This is why the Predict page surfaces these as sliders."
)
feats = [str(f) for f in d["importance_features"]]
vals = d["importance_values"]
order = np.argsort(vals)[::-1][:12]
imp_rows = [{"feature": feats[i], "importance": round(float(vals[i]), 4)} for i in order]
st.vega_lite_chart(
    {
        "height": 340,
        "data": {"values": imp_rows},
        "mark": {"type": "bar", "color": "#378add"},
        "encoding": {
            "y": {"field": "feature", "type": "nominal", "sort": "-x", "title": None},
            "x": {"field": "importance", "type": "quantitative",
                  "title": "mean accuracy drop when shuffled"},
        },
    },
    use_container_width=True,
)
