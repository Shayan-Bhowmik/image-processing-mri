import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import re

from src.inference import (
    aggregate_patient_score,
    build_gradcam_for_slice,
    load_trained_model,
    predict_slices,
    preprocess_uploaded_nifti,
)


st.set_page_config(page_title="Brain MRI Decision Support", page_icon="🧠", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;600&display=swap');

:root {
    --bg-main: #0d1317;
    --bg-card: #101d42;
    --bg-card-2: #232ed1;
    --text-main: #89d2dc;
    --text-muted: #6564db;
    --accent: #232ed1;
    --accent-soft: #6564db;
    --danger: #101d42;
    --safe: #6564db;
}

.stApp {
    background: var(--bg-main);
    color: var(--text-main);
}

[data-testid="block-container"] {
    padding-top: 1.6rem;
    padding-bottom: 2.4rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.2px;
}

body, p, div, span, label {
    font-family: 'IBM Plex Sans', sans-serif;
}

.hero {
    background: #101d42;
    border-radius: 18px;
    padding: 22px 26px;
    color: #89d2dc;
    margin-bottom: 16px;
    border: 1px solid #232ed1;
    box-shadow: 0 8px 20px #0d1317;
    animation: rise-in 420ms ease-out;
}

.hero-title {
    font-size: 1.65rem;
    font-weight: 700;
    margin-bottom: 4px;
}

.hero-subtitle {
    opacity: 0.92;
    font-size: 0.95rem;
    color: #6564db;
}

.chip {
    display: inline-block;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 8px;
    margin-top: 10px;
}

.chip-safe {
    background: #6564db;
    color: #0d1317;
}

.chip-risk {
    background: #101d42;
    color: #89d2dc;
}

.section-wrap {
    background: #101d42;
    border: 1px solid #232ed1;
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 14px;
}

.info-card {
    background: var(--bg-card);
    border: 1px solid #232ed1;
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: 0 4px 10px #0d1317;
    margin-bottom: 0.65rem;
}

.card-label {
    color: var(--text-muted);
    font-size: 0.86rem;
    margin-bottom: 2px;
}

.card-value {
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--text-main);
}

.mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.86rem;
}

.subtle {
    color: var(--text-muted);
    font-size: 0.9rem;
}

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid #232ed1;
    border-radius: 14px;
    padding: 12px;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 8px;
}

.summary-card {
    background: #101d42;
    border: 1px solid #101d42;
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 8px 16px #0d1317;
    transition: transform 180ms ease, border-color 180ms ease;
}

.summary-card:nth-child(1) {
    background: #232ed1;
    border-color: #232ed1;
}

.summary-card:nth-child(2) {
    background: #6564db;
    border-color: #6564db;
}

.summary-card:nth-child(3) {
    background: #101d42;
    border-color: #101d42;
}

.summary-card:nth-child(4) {
    background: #0d1317;
    border-color: #0d1317;
}

.summary-card:hover {
    transform: translateY(-2px);
    border-color: #6564db;
}

.summary-label {
    color: #89d2dc;
    font-size: 0.84rem;
    margin-bottom: 4px;
    letter-spacing: 0.15px;
}

.summary-value {
    color: #89d2dc;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    line-height: 1.05;
    letter-spacing: 0.25px;
}

.summary-value-sm {
    color: #89d2dc;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.35rem;
    line-height: 1.15;
    white-space: normal;
    word-break: break-word;
}

.summary-helper {
    color: #89d2dc;
    font-size: 0.79rem;
    margin-top: 2px;
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 8px 0 4px 0;
}

.detail-card {
    background: #101d42;
    border: 1px solid #232ed1;
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: 0 5px 12px #0d1317;
}

.detail-label {
    color: #6564db;
    font-size: 0.82rem;
    margin-bottom: 3px;
}

.detail-value {
    color: #89d2dc;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.05;
}

.detail-helper {
    color: #6564db;
    font-size: 0.77rem;
    margin-top: 3px;
}

.section-divider {
    margin: 14px 0 10px 0;
    border-top: 1px solid #232ed1;
}

.chip-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
}

.status-bar {
    display: grid;
    grid-template-columns: 1.1fr 1fr 1fr;
    gap: 10px;
    margin: 6px 0 14px 0;
}

.status-pill {
    background: #101d42;
    border: 1px solid #232ed1;
    border-radius: 999px;
    padding: 9px 13px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 6px 14px #0d1317;
    min-width: 0;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex: 0 0 auto;
    margin-left: 2px;
}

.status-pill > div {
    padding-left: 3px;
    min-width: 0;
}

.status-dot-ok {
    background: #6564db;
    box-shadow: 0 0 0 4px #232ed1;
}

.status-dot-wait {
    background: #232ed1;
    box-shadow: 0 0 0 4px #101d42;
}

.status-label {
    color: var(--text-muted);
    font-size: 0.76rem;
    letter-spacing: 0.18px;
    line-height: 1.1;
    margin-bottom: 4px;
}

.status-value {
    color: #89d2dc;
    font-size: 0.88rem;
    font-weight: 600;
    line-height: 1.1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

[data-baseweb="tab-list"] {
    gap: 8px;
    padding: 4px;
    border-radius: 999px;
    background: #101d42;
    border: 1px solid #232ed1;
    width: fit-content;
    margin-top: 0.2rem;
    margin-bottom: 0.65rem;
}

[data-baseweb="tab"] {
    border-radius: 999px;
    border: 1px solid transparent;
    color: #6564db;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 7px 16px;
    transition: all 180ms ease;
}

[data-baseweb="tab"][aria-selected="true"] {
    color: #89d2dc;
    border-color: #6564db;
    background: #232ed1;
    box-shadow: 0 0 0 1px #6564db, 0 6px 14px #0d1317;
}

[data-baseweb="tab"]:hover {
    color: #89d2dc;
    border-color: #6564db;
}

.viz-strip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
    margin: 0.15rem 0 0.55rem 0;
}

.viz-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #89d2dc;
    letter-spacing: 0.18px;
}

.viz-note {
    font-size: 0.82rem;
    color: #6564db;
}

.viz-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 5px 11px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid #6564db;
    background: #101d42;
    color: #89d2dc;
}

.viz-panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #89d2dc;
    margin: 0 0 6px 4px;
}

[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid #232ed1;
    box-shadow: 0 8px 16px #0d1317;
    background: #101d42;
}

[data-testid="stImage"] {
    padding: 6px;
    border-radius: 16px;
    background: #101d42;
    border: 1px solid #232ed1;
}

[data-testid="stImage"] + div {
    color: #6564db;
    font-size: 0.82rem;
}

@media (max-width: 1000px) {
    .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .detail-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .status-bar {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 640px) {
    .summary-grid {
        grid-template-columns: 1fr;
    }

    .detail-grid {
        grid-template-columns: 1fr;
    }
}

[data-testid="stSidebar"] {
    background: #101d42;
}

[data-testid="stSidebar"] * {
    color: #89d2dc;
}

.stMarkdown, .stCaption, .stSlider, .stTextInput, .stToggle {
    color: var(--text-main);
}

[data-testid="stHeader"] {
    background: #0d1317;
    border-bottom: 1px solid #232ed1;
    backdrop-filter: blur(9px) saturate(120%);
}

[data-testid="stHeader"]::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: -1px;
    height: 1px;
    background: #232ed1;
}

[data-testid="stToolbar"] {
    right: 0.9rem;
    gap: 0.55rem;
    align-items: center;
}

[data-testid="collapsedControl"] {
    left: 0.9rem;
}

[data-testid="collapsedControl"] button {
    border: 1px solid transparent !important;
    background: transparent !important;
    padding: 0.24rem !important;
    min-width: 1.8rem !important;
}

[data-testid="collapsedControl"] button:hover {
    border-color: transparent !important;
    background: transparent !important;
}

[data-testid="stToolbar"] button,
[data-testid="stToolbar"] a,
[data-testid="collapsedControl"] button {
    border-radius: 999px !important;
    border: 1px solid #232ed1 !important;
    background: #101d42 !important;
    color: #89d2dc !important;
    padding: 0.28rem 0.72rem !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    line-height: 1 !important;
    transition: all 170ms ease;
}

[data-testid="stToolbar"] button:hover,
[data-testid="stToolbar"] a:hover,
[data-testid="collapsedControl"] button:hover {
    border-color: #6564db !important;
    background: #232ed1 !important;
}

[data-testid="stToolbar"] * {
    color: #89d2dc !important;
}

[data-testid="collapsedControl"] button svg {
    display: none !important;
}

[data-testid="collapsedControl"] button::before {
    content: "";
    display: block;
    width: 18px;
    height: 3px;
    background: #89d2dc;
    border-radius: 999px;
    box-shadow: 0 -6px 0 #89d2dc, 0 6px 0 #89d2dc;
}

/* Replace default red accents in native Streamlit controls */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #89d2dc !important;
    border-color: #89d2dc !important;
}

.stSlider [data-baseweb="slider"] * {
    color: #89d2dc !important;
}

.stToggle [data-baseweb="switch"] {
    background: #6564db !important;
}

.stToggle [data-baseweb="switch"] [data-testid="stMarkdownContainer"] {
    color: #89d2dc !important;
}

.stToggle button[role="switch"][aria-checked="true"] {
    background: #6564db !important;
}

.stToggle button[role="switch"] > div {
    background: #89d2dc !important;
}

.stProgress > div > div > div > div {
    background: #89d2dc !important;
}

[data-baseweb="tab-highlight"] {
    background: #89d2dc !important;
}

@keyframes rise-in {
    from {
        transform: translateY(8px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-title">Brain MRI Decision Support</div>
  <div class="hero-subtitle">Upload a NIfTI volume, inspect the model's prediction, and review Grad-CAM attention maps per slice.</div>
</div>
""",
    unsafe_allow_html=True,
)
st.caption("Research use only. Not for clinical diagnosis.")


@st.cache_resource
def get_model(checkpoint_path: str):
    return load_trained_model(checkpoint_path=checkpoint_path)


def render_status_bar(slot, model_ready: bool, file_name: str | None):
    model_state = "Loaded" if model_ready else "Waiting"
    model_dot_class = "status-dot-ok" if model_ready else "status-dot-wait"
    file_state = file_name if file_name else "No file uploaded"
    file_dot_class = "status-dot-ok" if file_name else "status-dot-wait"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    slot.markdown(
        f"""
<div class="status-bar">
    <div class="status-pill">
        <span class="status-dot {model_dot_class}"></span>
        <div>
            <div class="status-label">Model</div>
            <div class="status-value">{model_state}</div>
        </div>
    </div>
    <div class="status-pill">
        <span class="status-dot {file_dot_class}"></span>
        <div>
            <div class="status-label">File</div>
            <div class="status-value">{file_state}</div>
        </div>
    </div>
    <div class="status-pill">
        <span class="status-dot status-dot-ok"></span>
        <div>
            <div class="status-label">Run Timestamp</div>
            <div class="status-value">{ts}</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def summarize_decision(score: float, threshold_value: float) -> tuple[str, str]:
    margin = score - threshold_value
    if margin >= 0.15:
        return "Strong positive signal", f"Score exceeds threshold by {margin:.3f}"
    if margin >= 0:
        return "Borderline positive signal", f"Score exceeds threshold by {margin:.3f}"
    if margin >= -0.1:
        return "Borderline negative signal", f"Score below threshold by {abs(margin):.3f}"
    return "Strong negative signal", f"Score below threshold by {abs(margin):.3f}"


@st.cache_data
def load_reference_metrics(log_path: str = "PROJECT_LOG.md") -> dict[str, float]:
    metrics: dict[str, float] = {}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return metrics

    # Pull latest benchmark values from the project log if available.
    test_acc_matches = re.findall(r"Test Accuracy:\s*~?([0-9]+(?:\.[0-9]+)?)%", text, flags=re.IGNORECASE)
    auc_matches = re.findall(r"ROC-AUC:\s*~?([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    patient_acc_matches = re.findall(r"Patient-level accuracy:\s*~?([0-9]+(?:\.[0-9]+)?)%", text, flags=re.IGNORECASE)

    if test_acc_matches:
        metrics["test_accuracy"] = float(test_acc_matches[-1])
    if auc_matches:
        metrics["roc_auc"] = float(auc_matches[-1])
    if patient_acc_matches:
        metrics["patient_accuracy"] = float(patient_acc_matches[-1])
    return metrics


@st.cache_data
def load_calibrated_threshold(
    calibration_path: str = "outputs/calibration/recommended_threshold.json",
    fallback: float = 0.5,
) -> float:
    path = Path(calibration_path)
    if not path.exists():
        return fallback

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        threshold = float(payload.get("recommended_threshold", fallback))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback

    return float(np.clip(threshold, 0.0, 1.0))


st.sidebar.header("Controls")
checkpoint_path = st.sidebar.text_input("Checkpoint path", "checkpoints/best_model.pth")
default_threshold = load_calibrated_threshold()
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(default_threshold),
    step=0.01,
)
show_gradcam = st.sidebar.toggle("Show Grad-CAM", value=True)
st.sidebar.caption("Tune prediction and visualization settings before reviewing slices.")
st.sidebar.caption(f"Calibrated default threshold: {default_threshold:.2f}")
st.sidebar.divider()

with st.sidebar.expander("Grad-CAM Quality", expanded=False):
    gradcam_smooth_kernel = st.slider(
        "Smoothing kernel (odd)",
        min_value=1,
        max_value=15,
        value=5,
        step=2,
    )
    gradcam_clip_low, gradcam_clip_high = st.slider(
        "Heatmap percentile clip",
        min_value=0.0,
        max_value=100.0,
        value=(2.0, 99.5),
        step=0.5,
    )
    heatmap_display_threshold = st.slider(
        "Overlay saliency threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.01,
    )
    focus_percentile = st.slider(
        "Focus top percentile",
        min_value=70,
        max_value=99,
        value=90,
        step=1,
    )

uploaded_file = st.file_uploader("Upload MRI volume (.nii or .nii.gz)", type=["nii", "nii.gz"])
status_slot = st.empty()

render_status_bar(status_slot, model_ready=False, file_name=uploaded_file.name if uploaded_file else None)

if uploaded_file is None:
    st.info("Upload a file to run inference.")
    st.stop()

try:
    model, device = get_model(checkpoint_path)
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

try:
    prep = preprocess_uploaded_nifti(uploaded_file.getvalue(), uploaded_file.name)
except Exception as exc:
    st.error(f"Could not preprocess MRI file: {exc}")
    st.stop()

render_status_bar(status_slot, model_ready=True, file_name=uploaded_file.name)

input_batch = prep["input_batch"]
valid_slices = prep["valid_slices"]

slice_preds, slice_probs = predict_slices(model, input_batch, device)
patient_score = aggregate_patient_score(slice_probs, top_k=10)
pred_label = 1 if patient_score >= threshold else 0
pred_text = "Tumor-like pattern" if pred_label == 1 else "Normal-like pattern"
decision_title, decision_detail = summarize_decision(float(patient_score), float(threshold))
predicted_class_probability = float(patient_score if pred_label == 1 else (1.0 - patient_score))
decision_margin = float(abs(patient_score - threshold))

slice_binary = (slice_probs >= threshold).astype(np.int32)
slice_consistency = float(np.mean(slice_binary == pred_label)) if len(slice_binary) > 0 else 0.0

entropy = -(patient_score * np.log(patient_score + 1e-8) + (1.0 - patient_score) * np.log(1.0 - patient_score + 1e-8))
normalized_entropy = float(entropy / np.log(2.0))
uncertainty = float(np.clip(normalized_entropy, 0.0, 1.0))
confidence_score = float(np.clip((predicted_class_probability * 0.7 + decision_margin * 0.3), 0.0, 1.0))

reference_metrics = load_reference_metrics()

top_k_slices = min(5, len(slice_probs))
top_indices = np.argsort(slice_probs)[-top_k_slices:][::-1]

st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
st.subheader("Study Snapshot")

uploaded_name = uploaded_file.name
st.markdown(
    f"""
<div class="subtle">Loaded volume</div>
<div class="mono">{uploaded_name}</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.subheader("Prediction Summary")
st.markdown(
        f"""
<div class="summary-grid">
    <div class="summary-card">
        <div class="summary-label">Patient score</div>
        <div class="summary-value">{patient_score:.3f}</div>
        <div class="summary-helper">Aggregated top-k slice confidence</div>
    </div>
    <div class="summary-card">
        <div class="summary-label">Decision threshold</div>
        <div class="summary-value">{threshold:.2f}</div>
        <div class="summary-helper">Cutoff used for patient-level label</div>
    </div>
    <div class="summary-card">
        <div class="summary-label">Model decision</div>
        <div class="summary-value-sm">{pred_text}</div>
        <div class="summary-helper">Binary class outcome</div>
    </div>
    <div class="summary-card">
        <div class="summary-label">Valid slices</div>
        <div class="summary-value">{len(valid_slices)}</div>
        <div class="summary-helper">Slices included in inference</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
)

decision_chip = (
    '<span class="chip chip-risk">High-Risk Pattern</span>'
    if pred_label == 1
    else '<span class="chip chip-safe">Low-Risk Pattern</span>'
)
st.markdown(decision_chip, unsafe_allow_html=True)
st.progress(min(max(patient_score, 0.0), 1.0), text="Patient-level score")
st.caption(f"{decision_title}: {decision_detail}")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Result Reliability")

reference_accuracy = reference_metrics.get("test_accuracy")
reference_patient_accuracy = reference_metrics.get("patient_accuracy")
reference_auc = reference_metrics.get("roc_auc")

reference_text = (
    f"{reference_accuracy:.2f}%"
    if reference_accuracy is not None
    else "N/A"
)

st.markdown(
    f"""
<div class="detail-grid">
    <div class="detail-card">
        <div class="detail-label">Prediction confidence</div>
        <div class="detail-value">{predicted_class_probability * 100:.1f}%</div>
        <div class="detail-helper">Probability assigned to selected class</div>
    </div>
    <div class="detail-card">
        <div class="detail-label">Decision robustness</div>
        <div class="detail-value">{decision_margin * 100:.1f}%</div>
        <div class="detail-helper">Distance from threshold</div>
    </div>
    <div class="detail-card">
        <div class="detail-label">Slice consistency</div>
        <div class="detail-value">{slice_consistency * 100:.1f}%</div>
        <div class="detail-helper">Slices agreeing with patient decision</div>
    </div>
    <div class="detail-card">
        <div class="detail-label">Reference test accuracy</div>
        <div class="detail-value">{reference_text}</div>
        <div class="detail-helper">Most recent project benchmark</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("Advanced reliability details", expanded=False):
    st.markdown(
        f"""
- Composite confidence score: **{confidence_score * 100:.1f}%** (combines class probability + decision margin).
- Prediction uncertainty (entropy-based): **{uncertainty * 100:.1f}%**.
- Reference patient-level accuracy: **{reference_patient_accuracy:.2f}%**.
- Reference ROC-AUC: **{reference_auc:.4f}**.
"""
        if reference_patient_accuracy is not None and reference_auc is not None
        else f"""
- Composite confidence score: **{confidence_score * 100:.1f}%** (combines class probability + decision margin).
- Prediction uncertainty (entropy-based): **{uncertainty * 100:.1f}%**.
- Reference accuracy values are unavailable in `PROJECT_LOG.md`.
"""
    )

with st.expander("How to interpret this output", expanded=False):
    st.markdown(
        """
1. The patient score summarizes suspicious evidence across slices.
2. The decision threshold controls whether the case is flagged as class=1.
3. Grad-CAM highlights regions that most influenced the selected slice decision.
4. Always combine model output with expert review and source imaging context.
"""
    )

max_index = len(valid_slices) - 1
default_index = int(np.argmax(slice_probs))
slice_index = st.slider("Slice index", min_value=0, max_value=max_index, value=default_index, step=1)


slice_img = input_batch[slice_index][1].detach().cpu().numpy()


brain_pixels = slice_img[np.abs(slice_img) > 1e-6]
if brain_pixels.size > 0:
    p_low, p_high = np.percentile(brain_pixels, [1.0, 99.0])
else:
    p_low, p_high = float(slice_img.min()), float(slice_img.max())

slice_img = np.clip((slice_img - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
selected_prob = float(slice_probs[slice_index])

st.markdown(
    f"""
<div class="info-card">
  <div class="card-label">Selected slice probability (class=1)</div>
  <div class="card-value">{selected_prob:.3f}</div>
</div>
""",
    unsafe_allow_html=True,
)

viz_tab, chart_tab, ranking_tab = st.tabs(["Slice Viewer", "Probability Trend", "Top Slices"])

with viz_tab:
    st.markdown(
        f"""
<div class="viz-strip">
    <div>
        <div class="viz-title">Grad-CAM Diagnostic Gallery</div>
        <div class="viz-note">Visual comparison of source MRI, focused activation map, and blended overlay.</div>
    </div>
    <div class="viz-chip">Slice {slice_index} | p(class=1) {selected_prob:.3f}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if show_gradcam:
        try:
            selected_target_class = int(slice_preds[slice_index])
            heatmap = build_gradcam_for_slice(
                model,
                device,
                input_batch[slice_index],
                target_class=selected_target_class,
                smooth_kernel=gradcam_smooth_kernel,
                clip_percentiles=(gradcam_clip_low, gradcam_clip_high),
                apply_brain_mask=True,
                brain_mask_threshold=0.05,
            )
        except Exception as exc:
            st.error(f"Grad-CAM failed: {exc}")
            st.stop()

        heatmap_color = plt.cm.viridis(heatmap)[:, :, :3]
        base_rgb = np.repeat(slice_img[..., None], 3, axis=2)
        display_brain_mask = (slice_img > 0.08).astype(np.float32)


        nonzero_cam = heatmap[heatmap > 0]
        if nonzero_cam.size > 0:
            focus_cut = float(np.percentile(nonzero_cam, focus_percentile))
        else:
            focus_cut = heatmap_display_threshold

        focus_threshold = max(heatmap_display_threshold, focus_cut)
        cam_focus = np.clip((heatmap - focus_threshold) / (1.0 - focus_threshold + 1e-8), 0.0, 1.0)
        cam_focus = np.power(cam_focus, 0.65) * display_brain_mask


        heatmap_alpha = (cam_focus * 0.95)[..., None]
        heatmap_on_brain = base_rgb * (1.0 - heatmap_alpha) + heatmap_color * heatmap_alpha


        alpha = (cam_focus * 0.65)[..., None]
        overlay = base_rgb * (1.0 - alpha) + heatmap_color * alpha

        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="viz-panel-title">MRI Slice</div>', unsafe_allow_html=True)
        c1.image(slice_img, use_container_width=True, clamp=True)
        c2.markdown('<div class="viz-panel-title">Grad-CAM on Brain</div>', unsafe_allow_html=True)
        c2.image(heatmap_on_brain, use_container_width=True, clamp=True)
        c3.markdown('<div class="viz-panel-title">Overlay</div>', unsafe_allow_html=True)
        c3.image(overlay, use_container_width=True, clamp=True)
    else:
        st.markdown('<div class="viz-panel-title">MRI Slice</div>', unsafe_allow_html=True)
        st.image(slice_img, use_container_width=True, clamp=True)

with chart_tab:
    trend_x = np.arange(len(slice_probs))
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0d1317")
    ax.set_facecolor("#101d42")
    ax.plot(trend_x, slice_probs, color="#232ed1", linewidth=2.0, label="Tumor probability")
    ax.axhline(y=threshold, color="#6564db", linestyle="--", linewidth=1.4, label="Decision threshold")
    ax.axvline(x=slice_index, color="#89d2dc", linestyle=":", linewidth=1.2, label="Selected slice")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Slice index", color="#89d2dc")
    ax.set_ylabel("Probability", color="#89d2dc")
    ax.tick_params(colors="#89d2dc")
    for spine in ax.spines.values():
        spine.set_color("#232ed1")
    ax.grid(color="#232ed1")
    legend = ax.legend(loc="upper right")
    legend.get_frame().set_facecolor("#0d1317")
    legend.get_frame().set_edgecolor("#232ed1")
    for text in legend.get_texts():
        text.set_color("#89d2dc")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Peaks indicate slices the model considers most suspicious.")

with ranking_tab:
    ranking_df = pd.DataFrame(
        {
            "rank": np.arange(1, top_k_slices + 1),
            "slice_index": top_indices,
            "tumor_probability": [float(slice_probs[idx]) for idx in top_indices],
            "predicted_class": [int(slice_preds[idx]) for idx in top_indices],
        }
    )
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)
    st.caption("Use these slices as a fast review shortlist for expert verification.")
