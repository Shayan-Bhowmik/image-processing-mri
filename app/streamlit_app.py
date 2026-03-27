import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from src.inference import (
    aggregate_patient_score,
    build_gradcam_for_slice,
    load_trained_model,
    predict_slices,
    preprocess_uploaded_nifti,
)


st.set_page_config(page_title="Brain MRI Decision Support", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;600&display=swap');

:root {
    --bg-main: #0d141b;
    --bg-card: #13202b;
    --text-main: #e8f0f7;
    --text-muted: #9fb3c8;
    --accent: #35c6a4;
    --accent-soft: #1a3a3a;
    --danger: #ff7d66;
    --safe: #57d18f;
}

.stApp {
    background: radial-gradient(circle at 12% 10%, #1a2a3a 0%, #0d141b 58%);
    color: var(--text-main);
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.2px;
}

body, p, div, span, label {
    font-family: 'IBM Plex Sans', sans-serif;
}

.hero {
    background: linear-gradient(120deg, #1a8f8d 0%, #274f8a 100%);
    border-radius: 18px;
    padding: 22px 26px;
    color: #f4fffb;
    margin-bottom: 16px;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
}

.hero-title {
    font-size: 1.65rem;
    font-weight: 700;
    margin-bottom: 4px;
}

.hero-subtitle {
    opacity: 0.92;
    font-size: 0.95rem;
    color: #d9ecf5;
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
    background: #17382a;
    color: var(--safe);
}

.chip-risk {
    background: #41241f;
    color: var(--danger);
}

.info-card {
    background: var(--bg-card);
    border: 1px solid #2a3c4e;
    border-radius: 14px;
    padding: 12px 14px;
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

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid #2a3c4e;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101b25 0%, #162431 100%);
}

[data-testid="stSidebar"] * {
    color: #e2edf7;
}

.stMarkdown, .stCaption, .stSlider, .stTextInput, .stToggle {
    color: var(--text-main);
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


st.sidebar.header("Controls")
checkpoint_path = st.sidebar.text_input("Checkpoint path", "checkpoints/best_model.pth")
threshold = st.sidebar.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
show_gradcam = st.sidebar.toggle("Show Grad-CAM", value=True)

uploaded_file = st.file_uploader("Upload MRI volume (.nii or .nii.gz)", type=["nii", "nii.gz"])

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

input_batch = prep["input_batch"]
valid_slices = prep["valid_slices"]

_, slice_probs = predict_slices(model, input_batch, device)
patient_score = aggregate_patient_score(slice_probs, top_k=10)
pred_label = 1 if patient_score >= threshold else 0
pred_text = "Tumor-like pattern" if pred_label == 1 else "Normal-like pattern"

st.subheader("Prediction Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Patient score", f"{patient_score:.3f}")
col2.metric("Threshold", f"{threshold:.2f}")
col3.metric("Decision", pred_text)
col4.metric("Valid slices", len(valid_slices))

decision_chip = (
    '<span class="chip chip-risk">High-Risk Pattern</span>'
    if pred_label == 1
    else '<span class="chip chip-safe">Low-Risk Pattern</span>'
)
st.markdown(decision_chip, unsafe_allow_html=True)
st.progress(min(max(patient_score, 0.0), 1.0), text="Patient-level score")

max_index = len(valid_slices) - 1
default_index = int(np.argmax(slice_probs))
slice_index = st.slider("Slice index", min_value=0, max_value=max_index, value=default_index, step=1)

# Use resized middle channel from model input so overlay matches Grad-CAM shape.
slice_img = input_batch[slice_index][1].detach().cpu().numpy()
slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
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

viz_tab, chart_tab = st.tabs(["Slice Viewer", "Probability Trend"])

with viz_tab:
    if show_gradcam:
        try:
            heatmap = build_gradcam_for_slice(model, device, input_batch[slice_index])
        except Exception as exc:
            st.error(f"Grad-CAM failed: {exc}")
            st.stop()

        heatmap_color = plt.cm.jet(heatmap)[:, :, :3]
        overlay = slice_img[..., None] * 0.4 + heatmap_color * 0.6

        c1, c2, c3 = st.columns(3)
        c1.image(slice_img, caption="MRI Slice", use_container_width=True, clamp=True)
        c2.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True, clamp=True)
        c3.image(overlay, caption="Overlay", use_container_width=True, clamp=True)
    else:
        st.image(slice_img, caption="MRI Slice", use_container_width=True, clamp=True)

with chart_tab:
    chart_df = pd.DataFrame(
        {
            "slice_index": np.arange(len(slice_probs)),
            "tumor_probability": slice_probs,
        }
    )
    st.area_chart(chart_df.set_index("slice_index"))
    st.caption("Peak values indicate slices the model considers most suspicious.")
