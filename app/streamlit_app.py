import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import re
import sys
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (
    aggregate_patient_score,
    build_gradcam_for_slice,
    load_trained_model,
    predict_slices,
    preprocess_uploaded_nifti,
)


DARK_THEME = {
    "bg_main": "#0d1317",
    "bg_sidebar": "#101d42",
    "bg_header": "#0d1317",
    "bg_card": "#101d42",
    "bg_surface": "#0d1317",
    "text_main": "#ffffff",
    "text_muted": "#ffffff",
    "accent": "#232ed1",
    "accent_soft": "#6564db",
    "border": "#232ed1",
    "shadow": "rgba(13, 19, 23, 0.72)",
    "on_accent": "#ffffff",
    "summary_1": "#232ed1",
    "summary_2": "#6564db",
    "summary_3": "#101d42",
    "summary_4": "#0d1317",
    "chart_bg": "#0d1317",
    "chart_panel": "#101d42",
    "chart_grid": "#232ed1",
    "chart_legend": "#0d1317",
    "chart_text": "#ffffff",
    "chart_line": "#89d2dc",
    "chart_fill": "rgba(137, 210, 220, 0.14)",
    "chart_threshold": "#7fffd4",
    "chart_selected": "#ffffff",
    "control_icon": "#ffffff",
    "control_bg": "transparent",
}

LIGHT_THEME = {
    "bg_main": "#f4f7fb",
    "bg_sidebar": "#e8eefc",
    "bg_header": "#f4f7fb",
    "bg_card": "#ffffff",
    "bg_surface": "#eef3ff",
    "text_main": "#0f172a",
    "text_muted": "#334155",
    "accent": "#2742b7",
    "accent_soft": "#3555d1",
    "border": "#b8c9e6",
    "shadow": "rgba(22, 34, 68, 0.12)",
    "on_accent": "#ffffff",
    "summary_1": "#dfe7ff",
    "summary_2": "#cfd8ff",
    "summary_3": "#eef2ff",
    "summary_4": "#ffffff",
    "chart_bg": "#f4f7fb",
    "chart_panel": "#ffffff",
    "chart_grid": "#c7d4ef",
    "chart_legend": "#f8fafc",
    "chart_text": "#172033",
    "chart_line": "#2742b7",
    "chart_fill": "rgba(39, 66, 183, 0.12)",
    "chart_threshold": "#0f172a",
    "chart_selected": "#3555d1",
    "control_icon": "#000000",
    "control_bg": "#dbe4f3",
}


def get_theme(light_mode: bool) -> dict[str, str]:
    return LIGHT_THEME if light_mode else DARK_THEME


st.set_page_config(page_title="Synapse X", page_icon="𝕏", layout="wide")

light_mode = True
theme = get_theme(light_mode)

theme_vars = f"""
:root {{
    --bg-main: {theme['bg_main']};
    --bg-sidebar: {theme['bg_sidebar']};
    --bg-header: {theme['bg_header']};
    --bg-card: {theme['bg_card']};
    --bg-surface: {theme['bg_surface']};
    --text-main: {theme['text_main']};
    --text-muted: {theme['text_muted']};
    --accent: {theme['accent']};
    --accent-soft: {theme['accent_soft']};
    --border: {theme['border']};
    --shadow: {theme['shadow']};
    --on-accent: {theme['on_accent']};
    --summary-1: {theme['summary_1']};
    --summary-2: {theme['summary_2']};
    --summary-3: {theme['summary_3']};
    --summary-4: {theme['summary_4']};
    --chart-bg: {theme['chart_bg']};
    --chart-panel: {theme['chart_panel']};
    --chart-grid: {theme['chart_grid']};
    --chart-legend: {theme['chart_legend']};
    --chart-text: {theme['chart_text']};
    --control-icon: {theme['control_icon']};
    --control-bg: {theme['control_bg']};
}}
"""

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;600&display=swap');
"""
    + theme_vars
    + """

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
    background: var(--bg-card);
    border-radius: 18px;
    padding: 22px 26px;
    color: var(--text-main);
    margin-bottom: 16px;
    border: 1px solid var(--border);
    box-shadow: 0 8px 20px var(--shadow);
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
    color: var(--text-muted);
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
    background: var(--accent-soft);
    color: var(--on-accent);
}

.chip-risk {
    background: var(--bg-card);
    color: var(--text-main);
}

.section-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 14px;
}

.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: 0 4px 10px var(--shadow);
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
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stMetric"] * {
    color: var(--text-main) !important;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 8px;
}

.summary-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 8px 16px var(--shadow);
    transition: transform 180ms ease, border-color 180ms ease;
}

.summary-card:nth-child(1) {
    background: var(--summary-1);
    border-color: var(--accent);
}

.summary-card:nth-child(2) {
    background: var(--summary-2);
    border-color: var(--accent-soft);
}

.summary-card:nth-child(3) {
    background: var(--summary-3);
    border-color: var(--border);
}

.summary-card:nth-child(4) {
    background: var(--summary-4);
    border-color: var(--border);
}

.summary-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent-soft);
}

.summary-label {
    color: var(--text-main);
    font-size: 0.84rem;
    margin-bottom: 4px;
    letter-spacing: 0.15px;
}

.summary-value {
    color: var(--text-main);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    line-height: 1.05;
    letter-spacing: 0.25px;
}

.summary-value-sm {
    color: var(--text-main);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.35rem;
    line-height: 1.15;
    white-space: normal;
    word-break: break-word;
}

.summary-helper {
    color: var(--text-main);
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
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: 0 5px 12px var(--shadow);
}

.detail-label {
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-bottom: 3px;
}

.detail-value {
    color: var(--text-main);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.05;
}

.detail-helper {
    color: var(--text-muted);
    font-size: 0.77rem;
    margin-top: 3px;
}

.section-divider {
    margin: 14px 0 10px 0;
    border-top: 1px solid var(--border);
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
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 6px 14px var(--shadow);
    min-width: 0;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex: 0 0 auto;
    margin-left: 2px;
    transition: background-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
}

.status-pill > div {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 2px;
    min-width: 0;
}

.status-dot-ok {
    background: #7fffd4;
    box-shadow: 0 0 0 4px rgba(127, 255, 212, 0.18);
}

.status-dot-wait {
    background: #8b95a7;
    box-shadow: 0 0 0 4px rgba(139, 149, 167, 0.18);
}

.status-pill:hover .status-dot {
    transform: scale(1.08);
}

.status-label {
    color: var(--text-muted);
    font-size: 0.76rem;
    letter-spacing: 0.18px;
    line-height: 1;
    margin-bottom: 0;
}

.status-value {
    color: var(--text-main);
    font-size: 0.88rem;
    font-weight: 600;
    line-height: 1.05;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.status-value-wait {
    font-size: 0.95rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: 0.01em;
    text-shadow: 0 0 8px rgba(255, 255, 255, 0.18);
}

[data-baseweb="tab-list"] {
    gap: 8px;
    padding: 4px;
    border-radius: 999px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    width: fit-content;
    margin-top: 0.2rem;
    margin-bottom: 0.65rem;
}

[data-baseweb="tab"] {
    border-radius: 999px;
    border: 1px solid transparent;
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.9rem;
    padding: 7px 16px;
    transition: all 180ms ease;
}

[data-baseweb="tab"][aria-selected="true"] {
    color: var(--text-main);
    border-color: var(--accent-soft);
    background: #5aa7ff;
    box-shadow: 0 0 0 1px var(--accent-soft), 0 6px 14px var(--shadow);
}

[data-baseweb="tab"]:hover {
    color: var(--text-main);
    border-color: var(--accent-soft);
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
    color: var(--text-main);
    letter-spacing: 0.18px;
}

.viz-note {
    font-size: 0.82rem;
    color: var(--text-muted);
}

.viz-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 5px 11px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid var(--accent-soft);
    background: var(--bg-card);
    color: var(--text-main);
}

.viz-panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-main);
    margin: 0 0 6px 4px;
}

[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: 0 8px 16px var(--shadow);
    background: var(--bg-card);
}

[data-testid="stImage"] {
    padding: 6px;
    border-radius: 16px;
    background: var(--bg-card);
    border: 1px solid var(--border);
}

[data-testid="stImage"] + div {
    color: var(--text-muted);
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
    background: var(--bg-sidebar);
}

[data-testid="stSidebar"] * {
    color: var(--text-main);
}

.stMarkdown, .stCaption, .stSlider, .stTextInput, .stToggle {
    color: var(--text-main);
}

.stButton button,
.stDownloadButton button,
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"] {
    color: var(--on-accent) !important;
    -webkit-text-fill-color: var(--on-accent) !important;
    font-weight: 700;
}

[data-testid="stBaseButton-primary"] *,
[data-testid="stBaseButton-secondary"] *,
.stButton button *,
.stDownloadButton button * {
    color: var(--on-accent) !important;
    -webkit-text-fill-color: var(--on-accent) !important;
}

[data-testid="stAlert"],
[data-testid="stAlert"] * {
    color: var(--text-main) !important;
}

[data-testid="stHeader"] {
    background: var(--bg-header);
    border-bottom: 1px solid var(--border);
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
    background: var(--control-bg) !important;
    padding: 0.24rem !important;
    min-width: 1.8rem !important;
    position: relative !important;
    color: transparent !important;
}

[data-testid="collapsedControl"] button:hover {
    border-color: transparent !important;
    background: var(--control-bg) !important;
}

[data-testid="stToolbar"] button,
[data-testid="stToolbar"] a,
[data-testid="collapsedControl"] button {
    border-radius: 999px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    color: var(--text-main) !important;
    padding: 0.28rem 0.72rem !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    line-height: 1 !important;
    transition: all 170ms ease;
}

[data-testid="collapsedControl"] button *,
[data-testid="collapsedControl"] button svg {
    color: var(--control-icon) !important;
    fill: var(--control-icon) !important;
    stroke: var(--control-icon) !important;
}

[data-testid="collapsedControl"] button svg,
[data-testid="collapsedControl"] button svg * {
    display: none !important;
}

[data-testid="stToolbar"] button:hover,
[data-testid="stToolbar"] a:hover,
[data-testid="collapsedControl"] button:hover {
    border-color: var(--accent-soft) !important;
    background: var(--accent) !important;
}

[data-testid="stToolbar"] * {
    color: var(--text-main) !important;
}

[data-testid="collapsedControl"] button::before {
    content: "";
    display: block;
    width: 9px;
    height: 9px;
    border-top: 3px solid var(--control-icon);
    border-right: 3px solid var(--control-icon);
    transform: rotate(45deg) translateY(-1px);
    margin-left: 1px;
}

/* Replace default red accents in native Streamlit controls */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--text-main) !important;
    border-color: var(--text-main) !important;
}

.stSlider [data-baseweb="slider"] * {
    color: var(--text-main) !important;
}

.stToggle [data-baseweb="switch"] {
    background: #89d2dc !important;
}

.stToggle [data-baseweb="switch"] [data-testid="stMarkdownContainer"] {
    color: var(--text-main) !important;
}

.stToggle button[role="switch"][aria-checked="true"] {
    background: #89d2dc !important;
}

.stToggle button[role="switch"] > div {
    background: var(--text-main) !important;
}

.stProgress > div > div > div > div {
    background: var(--text-main) !important;
}

[data-baseweb="tab-highlight"] {
    background: var(--text-main) !important;
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
    [data-testid="stDataFrame"] {
        background: var(--bg-card);
    }

    [data-testid="stDataFrame"] tbody tr {
        background: var(--bg-surface) !important;
    }

    [data-testid="stDataFrame"] tbody tr:nth-child(odd) {
        background: #e3f2fd !important;
    }

    [data-testid="stDataFrame"] thead th {
        background: #5aa7ff !important;
        color: white !important;
        font-weight: 600;
    }

    [data-testid="stDataFrame"] td {
        color: var(--text-main) !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid #000000 !important;
        border-radius: 12px !important;
        background: var(--bg-card) !important;
        margin-bottom: 10px !important;
    }

    [data-testid="stExpander"] details {
        border: none !important;
    }

    .score-wrap {
        margin-top: 8px;
        margin-bottom: 6px;
    }

    .score-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 6px;
    }

    .score-title {
        color: var(--text-main);
        font-weight: 600;
        font-size: 0.95rem;
    }

    .score-value {
        color: var(--accent);
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
    }

    .score-track {
        position: relative;
        height: 13px;
        width: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #dff0ff 0%, #edf6ff 100%);
        border: 1px solid #b7d8ff;
        overflow: hidden;
        box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.08);
    }

    .score-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #70b8ff 0%, #4f9df6 100%);
        box-shadow: 0 0 0 1px rgba(79, 157, 246, 0.14), 0 4px 10px rgba(79, 157, 246, 0.24);
    }

    .score-threshold {
        position: absolute;
        top: -2px;
        width: 2px;
        height: 17px;
        background: #1e3a8a;
        border-radius: 2px;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.8);
    }

    .score-legend {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 6px;
        font-size: 0.78rem;
        color: var(--text-muted);
    }

    .score-th-label {
        color: #1e3a8a;
        font-weight: 600;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="hero">
    <div class="hero-title">Synapse X</div>
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
    model_value_class = "status-value" if model_ready else "status-value status-value-wait"
    file_state = file_name if file_name else "No file uploaded"
    file_dot_class = "status-dot-ok" if file_name else "status-dot-wait"
    file_value_class = "status-value" if file_name else "status-value status-value-wait"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    slot.markdown(
        f"""
<div class="status-bar">
    <div class="status-pill">
        <span class="status-dot {model_dot_class}"></span>
        <div>
            <div class="status-label">Model</div>
            <div class="{model_value_class}">{model_state}</div>
        </div>
    </div>
    <div class="status-pill">
        <span class="status-dot {file_dot_class}"></span>
        <div>
            <div class="status-label">File</div>
            <div class="{file_value_class}">{file_state}</div>
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


def render_patient_score_bar(score: float, threshold_value: float) -> None:
    score_clamped = float(np.clip(score, 0.0, 1.0))
    threshold_clamped = float(np.clip(threshold_value, 0.0, 1.0))
    score_pct = score_clamped * 100
    threshold_pct = threshold_clamped * 100
    st.markdown(
        f"""
<div class="score-wrap">
    <div class="score-header">
        <span class="score-title">Patient-level score</span>
        <span class="score-value">{score_clamped:.3f}</span>
    </div>
    <div class="score-track">
        <div class="score-fill" style="width: {score_pct:.1f}%;"></div>
        <div class="score-threshold" style="left: {threshold_pct:.1f}%;"></div>
    </div>
    <div class="score-legend">
        <span>0.00</span>
        <span class="score-th-label">Threshold {threshold_clamped:.2f}</span>
        <span>1.00</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def download_stem(filename: str) -> str:
    name = Path(filename).name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def build_study_report(
    uploaded_name: str,
    patient_score: float,
    threshold: float,
    pred_text: str,
    decision_title: str,
    decision_detail: str,
    confidence_score: float,
    uncertainty: float,
    slice_consistency: float,
    valid_slice_count: int,
    selected_slice_index: int,
    selected_slice_probability: float,
    top_indices: np.ndarray,
    slice_probs: np.ndarray,
) -> str:
    lines = [
        "Synapse X Report",
        "",
        f"File: {uploaded_name}",
        f"Patient score: {patient_score:.3f}",
        f"Decision threshold: {threshold:.2f}",
        f"Model decision: {pred_text}",
        f"Decision summary: {decision_title}",
        f"Decision detail: {decision_detail}",
        f"Confidence score: {confidence_score * 100:.1f}%",
        f"Uncertainty: {uncertainty * 100:.1f}%",
        f"Slice consistency: {slice_consistency * 100:.1f}%",
        f"Valid slices: {valid_slice_count}",
        f"Selected slice index: {selected_slice_index}",
        f"Selected slice probability: {selected_slice_probability:.3f}",
        "",
        "Top slices:",
    ]

    for rank, idx in enumerate(top_indices, start=1):
        lines.append(f"  {rank}. Slice {int(idx)} - probability {float(slice_probs[idx]):.3f}")

    lines.append("")
    lines.append("Generated by Synapse X.")
    return "\n".join(lines)


def generate_pdf_report(report_text: str, filename: str = "report") -> bytes:
    """
    Generate a PDF report from report text.
    Returns bytes that can be downloaded.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 0.5 * inch
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, height - margin, "Synapse X Report")
    
    # Content
    c.setFont("Helvetica", 10)
    y = height - margin - 0.3 * inch
    line_height = 0.2 * inch
    
    for line in report_text.split("\n"):
        if y < margin:  # New page if needed
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - margin
        
        if line.strip():
            c.drawString(margin, y, line)
        y -= line_height
    
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def create_gradcam_composite_image(slice_img: np.ndarray, heatmap_on_brain: np.ndarray, overlay: np.ndarray) -> bytes:
    """
    Create a composite image with three panels (MRI Slice, Grad-CAM on Brain, Overlay) with headings.
    Returns bytes that can be downloaded as PNG.
    """
    # Convert numpy arrays to PIL images
    slice_pil = Image.fromarray((slice_img * 255).astype(np.uint8))
    heatmap_pil = Image.fromarray((heatmap_on_brain * 255).astype(np.uint8))
    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
    
    # Resize all images to the same height for consistency
    target_height = 300
    aspect_ratio = slice_pil.width / slice_pil.height
    target_width = int(target_height * aspect_ratio)
    
    slice_pil = slice_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
    heatmap_pil = heatmap_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
    overlay_pil = overlay_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Create titles for each panel
    title_height = 40
    panel_width = target_width
    panel_height = target_height + title_height
    
    # Create composite image with space for titles
    composite_width = panel_width * 3 + 30  # 3 panels + gaps
    composite_height = panel_height + 20
    composite = Image.new("RGB", (composite_width, composite_height), color=(255, 255, 255))
    
    # Paste images and add titles
    draw = ImageDraw.Draw(composite)
    title_font_size = 16
    
    titles = ["MRI Slice", "Grad-CAM on Brain", "Overlay"]
    images = [slice_pil, heatmap_pil, overlay_pil]
    
    for idx, (title, img) in enumerate(zip(titles, images)):
        x_offset = 10 + idx * (panel_width + 10)
        # Draw title
        title_y = 5
        draw.text((x_offset + 5, title_y), title, fill=(0, 0, 0))
        # Paste image below title
        composite.paste(img, (x_offset, title_height))
    
    # Save to bytes
    buffer = io.BytesIO()
    composite.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


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
checkpoint_path = "checkpoints/best_model.pth"
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
download_base_name = download_stem(uploaded_name)
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
render_patient_score_bar(float(patient_score), float(threshold))
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

advanced_box = st.container(border=True)
with advanced_box:
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

interpretation_box = st.container(border=True)
with interpretation_box:
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

report_text = build_study_report(
    uploaded_name=uploaded_name,
    patient_score=float(patient_score),
    threshold=float(threshold),
    pred_text=pred_text,
    decision_title=decision_title,
    decision_detail=decision_detail,
    confidence_score=float(confidence_score),
    uncertainty=float(uncertainty),
    slice_consistency=float(slice_consistency),
    valid_slice_count=len(valid_slices),
    selected_slice_index=slice_index,
    selected_slice_probability=selected_prob,
    top_indices=top_indices,
    slice_probs=slice_probs,
)

st.subheader("Downloads")
download_report_col, download_hint_col = st.columns([1, 1])
with download_report_col:
    pdf_report = generate_pdf_report(report_text, filename=download_base_name)
    st.download_button(
        label="Download report",
        data=pdf_report,
        file_name=f"{download_base_name}_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
with download_hint_col:
    st.caption("Use the report button for the PDF summary. The Grad-CAM image download appears below the gallery when Grad-CAM is enabled.")

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

        composite_image = create_gradcam_composite_image(slice_img, heatmap_on_brain, overlay)
        st.download_button(
            label="Download Grad-CAM image",
            data=composite_image,
            file_name=f"{download_base_name}_gradcam_composite.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.markdown('<div class="viz-panel-title">MRI Slice</div>', unsafe_allow_html=True)
        st.image(slice_img, use_container_width=True, clamp=True)

with chart_tab:
    trend_x = np.arange(len(slice_probs))
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(theme["chart_bg"])
    ax.set_facecolor(theme["chart_panel"])
    ax.plot(
        trend_x,
        slice_probs,
        color=theme["chart_line"],
        linewidth=2.8,
        marker="o",
        markersize=4.5,
        markerfacecolor=theme["chart_line"],
        markeredgecolor=theme["chart_panel"],
        markeredgewidth=0.9,
        label="Tumor probability",
        zorder=3,
    )
    ax.fill_between(trend_x, slice_probs, 0.0, color=theme["chart_line"], alpha=0.12, zorder=1)
    ax.axhline(y=threshold, color=theme["chart_threshold"], linestyle="--", linewidth=1.8, label="Decision threshold", zorder=2)
    ax.axvline(x=slice_index, color=theme["chart_selected"], linestyle=":", linewidth=1.6, label="Selected slice", zorder=2)
    ax.scatter([slice_index], [selected_prob], color=theme["chart_selected"], s=54, zorder=4, edgecolors=theme["chart_panel"], linewidths=1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.5, max(len(slice_probs) - 0.5, 0.5))
    ax.set_xlabel("Slice index", color=theme["chart_text"], labelpad=8)
    ax.set_ylabel("Probability", color=theme["chart_text"], labelpad=8)
    ax.tick_params(colors=theme["chart_text"], labelsize=9, width=0.8, length=4)
    for spine in ax.spines.values():
        spine.set_color(theme["border"])
        spine.set_linewidth(1.0)
    ax.grid(color=theme["chart_grid"], alpha=0.32, linewidth=0.8)
    legend = ax.legend(loc="upper right")
    legend.get_frame().set_facecolor(theme["chart_legend"])
    legend.get_frame().set_edgecolor(theme["border"])
    for text in legend.get_texts():
        text.set_color(theme["chart_text"])
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
    ranking_styler = (
        ranking_df.style.format({"tumor_probability": "{:.3f}"})
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#8ec8ff"),
                        ("color", "#0f172a"),
                        ("font-weight", "600"),
                    ],
                }
            ]
        )
        .set_properties(**{"background-color": "#eaf4ff", "color": "#0f172a"})
    )
    st.dataframe(ranking_styler, use_container_width=True, hide_index=True)
    st.caption("Use these slices as a fast review shortlist for expert verification.")
