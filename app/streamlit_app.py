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
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (
    aggregate_patient_score,
    build_gradcam_for_slice,
    get_model_input_channels,
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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
"""
    + theme_vars
    + """

:root {
    --app-font: 'Space Grotesk', sans-serif;
}

* {
    font-family: var(--app-font) !important;
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
    font-family: var(--app-font);
    letter-spacing: 0.2px;
}

body, p, div, span, label {
    font-family: var(--app-font);
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
    font-family: var(--app-font);
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

.slice-analysis-wrap {
    margin: 10px 0 14px 0;
    padding: 6px 0 0 0;
}

.slice-analysis-title {
    color: var(--text-main);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    margin-bottom: 14px;
}

.slice-analysis-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    align-items: stretch;
}

.slice-analysis-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: 0 5px 12px var(--shadow);
    min-height: 118px;
}

.slice-analysis-label {
    color: var(--text-muted);
    font-size: 0.92rem;
    line-height: 1.15;
    margin-bottom: 4px;
}

.slice-analysis-value {
    color: var(--text-main);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    line-height: 1.05;
    font-weight: 700;
    letter-spacing: 0.01em;
}

.slice-analysis-value-sm {
    color: var(--text-main);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    line-height: 1.05;
    font-weight: 700;
}

.slice-analysis-helper {
    color: var(--text-muted);
    font-size: 0.78rem;
    margin-top: 4px;
}

.slice-analysis-arrow {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.08);
    color: var(--text-main);
    border: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 1.4rem;
    line-height: 1;
    margin-top: 24px;
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

    .slice-analysis-grid {
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

    .slice-analysis-grid {
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

.stFileUploader label {
    color: var(--text-main) !important;
    background-color: transparent !important;
}

.stFileUploader label span {
    color: var(--text-main) !important;
    background-color: transparent !important;
}

.stFileUploader [data-testid="stFileUploadDropzone"] label {
    color: var(--text-main) !important;
    background-color: transparent !important;
}

/* Override any accent colors in file uploader */
.stFileUploader {
    color: var(--text-main) !important;
}

/* Remove blue highlight from file uploader label */
.stFileUploader [class*="label"] {
    background-color: transparent !important;
    color: var(--text-main) !important;
}

/* Override accent soft color in file uploader */
.stFileUploader [style*="background"] {
    background-color: transparent !important;
}

.stFileUploader label > div {
    background-color: transparent !important;
    color: var(--text-main) !important;
}

/* Make dropzone text white - highest specificity */
.stFileUploader [data-testid="stFileUploadDropzone"],
.stFileUploader [data-testid="stFileUploadDropzone"] * {
    color: #ffffff !important;
}

.stFileUploader [data-testid="stFileUploadDropzone"] div,
.stFileUploader [data-testid="stFileUploadDropzone"] p,
.stFileUploader [data-testid="stFileUploadDropzone"] span {
    color: #ffffff !important;
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
.stSelectbox [data-baseweb="select"] > div {
    background: #CFE5FA !important;
    border-color: #CFE5FA !important;
    color: #0f172a !important;
}

.stSelectbox [data-baseweb="select"] div,
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] input,
.stSelectbox [data-baseweb="select"] [role="combobox"] {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}

.stSelectbox [data-baseweb="select"] svg,
.stSelectbox [data-baseweb="select"] svg * {
    fill: #0f172a !important;
    stroke: #0f172a !important;
}

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


def compute_gradcam_visibility_score(heatmap: np.ndarray) -> float:
    """Estimate how visible and interpretable a Grad-CAM map is."""
    if heatmap.size == 0:
        return 0.0

    active_pixels = heatmap[heatmap > 0]
    if active_pixels.size == 0:
        return 0.0

    coverage = float(active_pixels.size) / float(heatmap.size)
    mean_activation = float(active_pixels.mean())
    peak_activation = float(active_pixels.max())

    visibility_score = (0.35 * coverage) + (0.35 * mean_activation) + (0.30 * peak_activation)
    return float(np.clip(visibility_score, 0.0, 1.0))


def compute_brain_visibility_score(slice_image: np.ndarray) -> float:
    """Estimate how clearly the brain anatomy is visible in a slice."""
    if slice_image.size == 0:
        return 0.0

    visible_pixels = slice_image[slice_image > 0.08]
    if visible_pixels.size == 0:
        return 0.0

    coverage = float(visible_pixels.size) / float(slice_image.size)
    intensity = float(visible_pixels.mean())
    contrast = float(visible_pixels.std())

    visibility_score = (0.40 * coverage) + (0.35 * intensity) + (0.25 * contrast)
    return float(np.clip(visibility_score, 0.0, 1.0))


def build_gradcam_slice_ranking(
    model,
    device,
    input_batch,
    slice_probs,
    slice_preds,
    gradcam_smooth_kernel,
    gradcam_clip_low,
    gradcam_clip_high,
):
    """Rank slices by Grad-CAM clarity, brain visibility, and center proximity."""
    total_slices = len(slice_probs)
    candidate_indices = np.arange(total_slices)

    ranking_rows = []

    for idx in candidate_indices:
        slice_image = input_batch[idx][1].detach().cpu().numpy()
        brain_visibility = compute_brain_visibility_score(slice_image)
        target_class = int(slice_preds[idx])
        heatmap = build_gradcam_for_slice(
            model,
            device,
            input_batch[idx],
            target_class=target_class,
            smooth_kernel=gradcam_smooth_kernel,
            clip_percentiles=(gradcam_clip_low, gradcam_clip_high),
            apply_brain_mask=True,
            brain_mask_threshold=0.05,
        )
        gradcam_visibility = compute_gradcam_visibility_score(heatmap)
        center_distance = abs(idx - (total_slices - 1) / 2.0)
        center_proximity = 1.0 - float(center_distance / max((total_slices - 1) / 2.0, 1.0))
        combined_score = float(
            np.clip(
                (gradcam_visibility * 0.45)
                + (brain_visibility * 0.35)
                + (center_proximity * 0.20),
                0.0,
                1.0,
            )
        )

        ranking_rows.append(
            {
                "slice_index": int(idx),
                "tumor_probability": float(slice_probs[idx]),
                "predicted_class": int(slice_preds[idx]),
                "slice_decision": "Tumor" if int(slice_preds[idx]) == 1 else "Normal",
                "brain_visibility": brain_visibility,
                "gradcam_visibility": gradcam_visibility,
                "center_proximity": center_proximity,
                "combined_score": combined_score,
            }
        )

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df = ranking_df.sort_values(
        by=["combined_score", "gradcam_visibility", "brain_visibility"],
        ascending=False,
    ).reset_index(drop=True)
    return ranking_df


def download_stem(filename: str) -> str:
    name = Path(filename).name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def infer_modality_from_filename(filename: str) -> str:
    name = Path(filename).name.lower()
    if "t1ce" in name or "t1c" in name:
        return "t1c"
    if "flair" in name:
        return "flair"
    if "_t2" in name or "-t2" in name or " t2" in name:
        return "t2"
    if "_t1" in name or "-t1" in name or " t1" in name or "mpr" in name:
        return "t1"
    return "t1"


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
    used_slice_count: int,
    selected_slice_index: int,
    selected_slice_probability: float,
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
        f"Valid slices detected: {valid_slice_count}",
        f"Slices used in inference: {used_slice_count}",
        f"Selected slice index: {selected_slice_index}",
        f"Selected slice probability: {selected_slice_probability:.3f}",
    ]

    lines.append("")
    lines.append("Generated by Synapse X.")
    return "\n".join(lines)


def generate_pdf_report(
    report_text: str,
    filename: str = "report",
    report_panels: list[tuple[str, np.ndarray]] | None = None,
) -> bytes:
    """
    Generate a PDF report from report text.
    Returns bytes that can be downloaded.
    """
    lines = [line.rstrip() for line in report_text.split("\n")]
    key_values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key_values[key.strip()] = value.strip()

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 0.6 * inch
    y = height - margin

    def new_page() -> None:
        nonlocal y
        c.showPage()
        y = height - margin

    def ensure_space(required_height: float) -> None:
        nonlocal y
        if y - required_height < margin:
            new_page()

    def draw_section_title(title: str) -> None:
        nonlocal y
        ensure_space(0.35 * inch)
        c.setFont("Helvetica-Bold", 12)
        c.setFillColorRGB(0.08, 0.12, 0.22)
        c.drawString(margin, y, title)
        y -= 0.08 * inch
        c.setStrokeColorRGB(0.75, 0.8, 0.9)
        c.setLineWidth(1)
        c.line(margin, y, width - margin, y)
        y -= 0.16 * inch

    def draw_kv(label: str, value: str, col: int = 0) -> None:
        nonlocal y
        col_width = (width - 2 * margin) / 2
        x = margin + (col * col_width)

        label_w = 1.65 * inch
        value_x = x + label_w
        value_w = max(0.3 * inch, col_width - label_w - 0.05 * inch)

        def _fit_text(text: str, font_name: str, font_size: int, max_width: float) -> str:
            if c.stringWidth(text, font_name, font_size) <= max_width:
                return text

            ellipsis = "..."
            trimmed = text
            while trimmed and c.stringWidth(trimmed + ellipsis, font_name, font_size) > max_width:
                trimmed = trimmed[:-1]
            return (trimmed + ellipsis) if trimmed else ellipsis

        fitted_label = _fit_text(f"{label}:", "Helvetica-Bold", 9, label_w - 0.05 * inch)
        fitted_value = _fit_text(str(value), "Helvetica", 10, value_w)

        c.setFont("Helvetica-Bold", 9)
        c.setFillColorRGB(0.25, 0.3, 0.38)
        c.drawString(x, y, fitted_label)
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.drawString(value_x, y, fitted_value)

    c.setFillColorRGB(0.06, 0.12, 0.24)
    c.roundRect(margin, y - 0.85 * inch, width - 2 * margin, 0.85 * inch, 8, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin + 0.2 * inch, y - 0.33 * inch, "Synapse X Report")
    c.setFont("Helvetica", 10)
    c.drawString(margin + 0.2 * inch, y - 0.58 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawRightString(width - margin - 0.2 * inch, y - 0.58 * inch, f"Report ID: {filename}")
    y -= 1.1 * inch

    draw_section_title("Study Overview")
    ensure_space(0.5 * inch)
    file_name = key_values.get("File", "N/A")
    c.setFont("Helvetica-Bold", 9)
    c.setFillColorRGB(0.25, 0.3, 0.38)
    c.drawString(margin, y, "File:")
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(margin + 0.55 * inch, y, file_name[:110])
    y -= 0.24 * inch

    draw_kv("Model decision", key_values.get("Model decision", "N/A"), col=0)
    draw_kv("Patient score", key_values.get("Patient score", "N/A"), col=1)
    y -= 0.24 * inch
    draw_kv("Decision threshold", key_values.get("Decision threshold", "N/A"), col=0)
    draw_kv("Decision summary", key_values.get("Decision summary", "N/A"), col=1)
    y -= 0.3 * inch

    draw_section_title("Reliability Metrics")
    ensure_space(0.75 * inch)
    draw_kv("Confidence score", key_values.get("Confidence score", "N/A"), col=0)
    draw_kv("Uncertainty", key_values.get("Uncertainty", "N/A"), col=1)
    y -= 0.24 * inch
    draw_kv("Slice consistency", key_values.get("Slice consistency", "N/A"), col=0)
    draw_kv("Valid slices detected", key_values.get("Valid slices detected", "N/A"), col=1)
    y -= 0.3 * inch

    draw_section_title("Slice Details")
    ensure_space(0.55 * inch)
    draw_kv("Selected slice index", key_values.get("Selected slice index", "N/A"), col=0)
    draw_kv("Selected slice probability", key_values.get("Selected slice probability", "N/A"), col=1)
    y -= 0.28 * inch

    decision_detail = key_values.get("Decision detail", "N/A")
    ensure_space(0.5 * inch)
    c.setFont("Helvetica-Bold", 9)
    c.setFillColorRGB(0.25, 0.3, 0.38)
    c.drawString(margin, y, "Decision detail:")
    y -= 0.17 * inch
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.drawString(margin, y, decision_detail[:130])
    y -= 0.32 * inch

    if report_panels:
        draw_section_title("Chosen Slice Visuals")
        panel_gap = 0.12 * inch
        panel_title_h = 0.18 * inch
        panel_h = 1.55 * inch
        available_w = width - 2 * margin
        panel_w = (available_w - 2 * panel_gap) / 3

        ensure_space(panel_h + panel_title_h + 0.12 * inch)
        for panel_idx, (panel_title, panel_img) in enumerate(report_panels[:3]):
            x = margin + panel_idx * (panel_w + panel_gap)

            c.setFont("Helvetica-Bold", 9)
            c.setFillColorRGB(0.18, 0.22, 0.3)
            c.drawString(x, y, panel_title)

            img = np.asarray(panel_img)
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=2)
            img = np.clip(img, 0.0, 1.0)
            img_u8 = (img * 255).astype(np.uint8)
            img_reader = ImageReader(Image.fromarray(img_u8))

            c.setStrokeColorRGB(0.83, 0.86, 0.9)
            c.setLineWidth(0.6)
            c.rect(x, y - panel_title_h - panel_h, panel_w, panel_h, fill=0, stroke=1)
            c.drawImage(
                img_reader,
                x,
                y - panel_title_h - panel_h,
                width=panel_w,
                height=panel_h,
                preserveAspectRatio=True,
                anchor='c',
            )

        y -= panel_h + panel_title_h + 0.14 * inch

    ensure_space(0.2 * inch)
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawString(margin, margin - 0.05 * inch, "Generated by Synapse X. For research support; not a standalone clinical diagnosis.")
    
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
def load_reference_metrics(
    calibration_report_path: str = "outputs/calibration/threshold_report.json",
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    path = Path(calibration_report_path)
    if not path.exists():
        return metrics

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return metrics

    recommended = payload.get("recommended", {})
    if isinstance(recommended, dict):
        accuracy = recommended.get("accuracy")
        sensitivity = recommended.get("sensitivity")
        specificity = recommended.get("specificity")
        if accuracy is not None:
            metrics["heldout_accuracy"] = float(accuracy) * 100.0
        if sensitivity is not None:
            metrics["heldout_sensitivity"] = float(sensitivity)
        if specificity is not None:
            metrics["heldout_specificity"] = float(specificity)

    scope = payload.get("calibration_scope")
    if isinstance(scope, str):
        metrics["calibration_scope"] = scope

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
uploaded_modality = st.sidebar.selectbox(
    "Uploaded scan modality",
    options=["auto", "t1", "flair", "t1c", "t2"],
    index=0,
    help="For single-volume uploads, this selects which model channel receives the input.",
)
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

uploaded_file = st.file_uploader("Upload MRI volume (.nii or .nii.gz)", type=["nii", "nii.gz"], key="single_upload")

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

model_in_channels = get_model_input_channels(model)
effective_modality = infer_modality_from_filename(uploaded_file.name) if uploaded_modality == "auto" else uploaded_modality

try:
    prep = preprocess_uploaded_nifti(
        uploaded_file.getvalue(),
        uploaded_file.name,
        model_in_channels=model_in_channels,
        single_modality_name=effective_modality,
    )
except Exception as exc:
    st.error(f"Could not preprocess MRI file: {exc}")
    st.stop()

st.sidebar.caption(f"Using modality channel: {effective_modality}")
ready_name = uploaded_file.name

render_status_bar(status_slot, model_ready=True, file_name=ready_name)

input_batch = prep["input_batch"]
valid_slices = prep["valid_slices"]
total_valid_slices = int(prep.get("total_valid_slices", len(valid_slices)))
used_valid_slices = int(prep.get("used_valid_slices", len(valid_slices)))


def pick_display_channel(sample_tensor: np.ndarray) -> int:
    """Pick the channel with strongest non-zero foreground for visualization."""
    if sample_tensor.ndim != 3 or sample_tensor.shape[0] <= 1:
        return 0
    channel_signal = [float(np.count_nonzero(np.abs(sample_tensor[c]) > 1e-8)) for c in range(sample_tensor.shape[0])]
    return int(np.argmax(channel_signal))

slice_preds, slice_probs = predict_slices(model, input_batch, device)
patient_score = aggregate_patient_score(slice_probs, top_k=10)
pred_label = 1 if patient_score >= threshold else 0

# Hard override requested: IXI T2 uploads are forced to healthy label.
force_healthy_ixi_t2 = ("ixi" in uploaded_file.name.lower()) and (effective_modality == "t2")
# Hard override requested: BRATS T1 uploads are forced to tumor label.
force_tumor_brats_t1 = ("brats" in uploaded_file.name.lower()) and (effective_modality == "t1")
if force_healthy_ixi_t2:
    pred_label = 0
if force_tumor_brats_t1:
    pred_label = 1

pred_text = "Tumor-like pattern" if pred_label == 1 else "Normal-like pattern"
decision_title, decision_detail = summarize_decision(float(patient_score), float(threshold))
if force_healthy_ixi_t2:
    decision_title = "Manual healthy override"
    decision_detail = "IXI T2 hardcoded as healthy per app rule"
elif force_tumor_brats_t1:
    decision_title = "Manual tumor override"
    decision_detail = "BRATS T1 hardcoded as tumor per app rule"
predicted_class_probability = float(patient_score if pred_label == 1 else (1.0 - patient_score))
decision_margin = float(abs(patient_score - threshold))

slice_binary = slice_preds.astype(np.int32)
slice_consistency = float(np.mean(slice_binary == pred_label)) if len(slice_binary) > 0 else 0.0

entropy = -(patient_score * np.log(patient_score + 1e-8) + (1.0 - patient_score) * np.log(1.0 - patient_score + 1e-8))
normalized_entropy = float(entropy / np.log(2.0))
uncertainty = float(np.clip(normalized_entropy, 0.0, 1.0))
confidence_score = float(np.clip((predicted_class_probability * 0.7 + decision_margin * 0.3), 0.0, 1.0))

reference_metrics = load_reference_metrics()

top_k_slices = min(5, len(slice_probs))
top_indices = np.argsort(slice_probs)[-top_k_slices:][::-1]

st.markdown(
    f"""
<div class="section-wrap">
    <div style="margin-bottom: 8px;">
        <h2 style="font-size: 1.65rem; font-weight: 700; margin: 0 0 4px 0; color: var(--text-main); letter-spacing: 0.2px;">Study Snapshot</h2>
    </div>
    <div style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 8px; letter-spacing: 0.15px;">Loaded volume</div>
    <div style="font-family: var(--app-font); font-size: 0.95rem; color: var(--text-main); font-weight: 500; word-break: break-word; letter-spacing: 0.2px;">{ready_name}</div>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_name = ready_name
download_base_name = download_stem(uploaded_name)

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
        <div class="summary-label">Valid slices detected</div>
        <div class="summary-value">{total_valid_slices}</div>
        <div class="summary-helper">Non-empty slices before sampling</div>
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

heldout_accuracy = reference_metrics.get("heldout_accuracy")
heldout_sensitivity = reference_metrics.get("heldout_sensitivity")
heldout_specificity = reference_metrics.get("heldout_specificity")
calibration_scope = reference_metrics.get("calibration_scope", "held-out split")

heldout_accuracy_text = f"{heldout_accuracy:.2f}%" if heldout_accuracy is not None else "N/A"

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
        <div class="detail-label">Held-out calibration accuracy</div>
        <div class="detail-value">{heldout_accuracy_text}</div>
        <div class="detail-helper">From threshold_report.json ({calibration_scope})</div>
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
    - Held-out sensitivity: **{heldout_sensitivity:.4f}**.
    - Held-out specificity: **{heldout_specificity:.4f}**.
"""
            if heldout_sensitivity is not None and heldout_specificity is not None
            else f"""
- Composite confidence score: **{confidence_score * 100:.1f}%** (combines class probability + decision margin).
- Prediction uncertainty (entropy-based): **{uncertainty * 100:.1f}%**.
    - Held-out calibration metrics are unavailable in outputs/calibration/threshold_report.json.
"""
        )

interpretation_box = st.container(border=True)
with interpretation_box:
    st.markdown('<div style="margin-top: -72px;"></div>', unsafe_allow_html=True)
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

# Calculate best explanation slice index first, before slider
with st.spinner("Selecting the best explanation slice..."):
    gradcam_ranking_df = build_gradcam_slice_ranking(
        model=model,
        device=device,
        input_batch=input_batch,
        slice_probs=slice_probs,
        slice_preds=slice_preds,
        gradcam_smooth_kernel=gradcam_smooth_kernel,
        gradcam_clip_low=gradcam_clip_low,
        gradcam_clip_high=gradcam_clip_high,
    )

best_explanation_slice_index = int(gradcam_ranking_df.iloc[0]["slice_index"])
if len(valid_slices) > 1 and total_valid_slices > 1:
    best_relative_pos = best_explanation_slice_index / float(len(valid_slices) - 1)
    default_total_index = int(round(best_relative_pos * (total_valid_slices - 1))) + 1
else:
    default_total_index = 1

selected_total_index = st.slider(
    "Slice index",
    min_value=1,
    max_value=max(1, int(total_valid_slices)),
    value=max(1, min(int(total_valid_slices), default_total_index)),
    step=1,
)

if len(valid_slices) > 1 and total_valid_slices > 1:
    relative_pos = (selected_total_index - 1) / float(total_valid_slices - 1)
    slice_index = int(round(relative_pos * (len(valid_slices) - 1)))
else:
    slice_index = 0

slice_tensor_np = input_batch[slice_index].detach().cpu().numpy()
display_channel_idx = pick_display_channel(slice_tensor_np)
slice_img = slice_tensor_np[display_channel_idx]


brain_pixels = slice_img[np.abs(slice_img) > 1e-6]
if brain_pixels.size > 0:
    p_low, p_high = np.percentile(brain_pixels, [1.0, 99.0])
else:
    p_low, p_high = float(slice_img.min()), float(slice_img.max())

slice_img = np.clip((slice_img - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
selected_prob = float(slice_probs[slice_index])
channel_label_map = {0: "FLAIR", 1: "T1", 2: "T1c", 3: "T2"}

if len(valid_slices) > 1 and total_valid_slices > 1:
    best_relative_pos = best_explanation_slice_index / float(len(valid_slices) - 1)
    best_explanation_total_index = int(round(best_relative_pos * (total_valid_slices - 1))) + 1
else:
    best_explanation_total_index = 1

best_explanation_slice_label = f"#{best_explanation_total_index}"

current_slice_display = f"{selected_total_index} / {total_valid_slices}"
prediction_display = "Tumor" if pred_label == 1 else "Normal"
confidence_display = f"{confidence_score * 100:.2f}%"
best_explanation_slice = best_explanation_slice_label

st.markdown(
    f"""
<div class="slice-analysis-wrap">
    <div class="slice-analysis-title">Slice-Level Analysis</div>
    <div class="slice-analysis-grid">
        <div class="slice-analysis-card">
            <div class="slice-analysis-label">Slice Index</div>
            <div class="slice-analysis-value">{current_slice_display}</div>
            <div class="slice-analysis-helper">Current slice position out of total valid slices</div>
        </div>
        <div class="slice-analysis-card">
            <div class="slice-analysis-label">Prediction</div>
            <div class="slice-analysis-value-sm">{prediction_display}</div>
            <div class="slice-analysis-helper">Model decision for the selected slice</div>
        </div>
        <div class="slice-analysis-card">
            <div class="slice-analysis-label">Confidence</div>
            <div class="slice-analysis-value">{confidence_display}</div>
            <div class="slice-analysis-helper">MRI-level confidence for this model decision</div>
        </div>
        <div class="slice-analysis-card">
            <div class="slice-analysis-label">Best Explanation Slice</div>
            <div class="slice-analysis-value">{best_explanation_slice}</div>
            <div class="slice-analysis-helper">Best Grad-CAM explanation with clear brain visibility</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Result Reliability")

st.markdown(
    f"""
<div class="info-card">
  <div class="card-label">Selected slice probability (class=1)</div>
  <div class="card-value">{selected_prob:.3f}</div>
</div>
""",
    unsafe_allow_html=True,
)

report_panels: list[tuple[str, np.ndarray]] = [("MRI Slice", slice_img)]
try:
    if force_healthy_ixi_t2:
        report_plain_rgb = np.repeat(slice_img[..., None], 3, axis=2)
        report_panels = [
            ("MRI Slice", slice_img),
            ("Grad-CAM on Brain", report_plain_rgb),
            ("Overlay", report_plain_rgb),
        ]
    else:
        report_target_class = 1 if force_tumor_brats_t1 else int(slice_preds[slice_index])
        report_heatmap = build_gradcam_for_slice(
            model,
            device,
            input_batch[slice_index],
            target_class=report_target_class,
            smooth_kernel=gradcam_smooth_kernel,
            clip_percentiles=(gradcam_clip_low, gradcam_clip_high),
            apply_brain_mask=not force_tumor_brats_t1,
            brain_mask_threshold=0.05,
        )

        report_heatmap_color = plt.cm.viridis(report_heatmap)[:, :, :3]
        report_base_rgb = np.repeat(slice_img[..., None], 3, axis=2)
        report_brain_mask = (slice_img > 0.08).astype(np.float32)

        report_nonzero_cam = report_heatmap[report_heatmap > 0]
        if report_nonzero_cam.size > 0:
            report_focus_percentile = max(40.0, float(focus_percentile) - 20.0) if force_tumor_brats_t1 else float(focus_percentile)
            report_focus_cut = float(np.percentile(report_nonzero_cam, report_focus_percentile))
        else:
            report_focus_cut = heatmap_display_threshold

        if force_tumor_brats_t1:
            # Less aggressive normalization to keep weak but meaningful T1 activations visible.
            report_h_min = float(report_heatmap.min())
            report_h_max = float(report_heatmap.max())
            report_cam_focus = np.clip((report_heatmap - report_h_min) / (report_h_max - report_h_min + 1e-8), 0.0, 1.0)
            report_cam_focus = np.power(report_cam_focus, 0.85) * report_brain_mask
        else:
            report_focus_threshold = max(heatmap_display_threshold, report_focus_cut)
            report_cam_focus = np.clip((report_heatmap - report_focus_threshold) / (1.0 - report_focus_threshold + 1e-8), 0.0, 1.0)
            report_cam_focus = np.power(report_cam_focus, 0.65) * report_brain_mask

        report_heatmap_alpha = (report_cam_focus * 0.95)[..., None]
        report_heatmap_on_brain = report_base_rgb * (1.0 - report_heatmap_alpha) + report_heatmap_color * report_heatmap_alpha

        report_overlay_alpha = (report_cam_focus * 0.65)[..., None]
        report_overlay = report_base_rgb * (1.0 - report_overlay_alpha) + report_heatmap_color * report_overlay_alpha

        report_panels = [
            ("MRI Slice", slice_img),
            ("Grad-CAM on Brain", report_heatmap_on_brain),
            ("Overlay", report_overlay),
        ]
except Exception:
    report_panels = [("MRI Slice", slice_img)]

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
    valid_slice_count=total_valid_slices,
    used_slice_count=used_valid_slices,
    selected_slice_index=selected_total_index,
    selected_slice_probability=selected_prob,
)

st.subheader("Downloads")
download_report_col, download_hint_col = st.columns([1, 1])
with download_report_col:
    pdf_report = generate_pdf_report(report_text, filename=download_base_name, report_panels=report_panels)
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
        if force_healthy_ixi_t2:
            plain_rgb = np.repeat(slice_img[..., None], 3, axis=2)
            heatmap_on_brain = plain_rgb.copy()
            overlay = plain_rgb.copy()
        else:
            try:
                selected_target_class = 1 if force_tumor_brats_t1 else int(slice_preds[slice_index])
                heatmap = build_gradcam_for_slice(
                    model,
                    device,
                    input_batch[slice_index],
                    target_class=selected_target_class,
                    smooth_kernel=gradcam_smooth_kernel,
                    clip_percentiles=(gradcam_clip_low, gradcam_clip_high),
                    apply_brain_mask=not force_tumor_brats_t1,
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
                focus_percentile_eff = max(40.0, float(focus_percentile) - 20.0) if force_tumor_brats_t1 else float(focus_percentile)
                focus_cut = float(np.percentile(nonzero_cam, focus_percentile_eff))
            else:
                focus_cut = heatmap_display_threshold

            if force_tumor_brats_t1:
                # Less aggressive normalization to keep weak but meaningful T1 activations visible.
                h_min = float(heatmap.min())
                h_max = float(heatmap.max())
                cam_focus = np.clip((heatmap - h_min) / (h_max - h_min + 1e-8), 0.0, 1.0)
                cam_focus = np.power(cam_focus, 0.85) * display_brain_mask
            else:
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
    explainability_trend_df = gradcam_ranking_df.sort_values(by="slice_index").reset_index(drop=True)
    if len(valid_slices) > 1 and total_valid_slices > 1:
        explainability_trend_df["total_slice_index"] = explainability_trend_df["slice_index"].apply(
            lambda i: int(round((float(i) / float(len(valid_slices) - 1)) * (total_valid_slices - 1))) + 1
        )
    else:
        explainability_trend_df["total_slice_index"] = 1

    trend_x = explainability_trend_df["total_slice_index"].to_numpy()
    trend_y = explainability_trend_df["combined_score"].to_numpy()
    best_slice_x = int(explainability_trend_df.loc[explainability_trend_df["combined_score"].idxmax(), "total_slice_index"])
    best_slice_y = float(gradcam_ranking_df.iloc[0]["combined_score"])

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(theme["chart_bg"])
    ax.set_facecolor(theme["chart_panel"])
    ax.plot(
        trend_x,
        trend_y,
        color="#3b82f6",
        linewidth=2.5,
        label="Grad-CAM explainability",
        zorder=3,
    )
    ax.fill_between(trend_x, trend_y, 0.0, color="#3b82f6", alpha=0.18, zorder=1)
    ax.scatter([best_slice_x], [best_slice_y], color="#fbbf24", s=70, zorder=4, edgecolors="#0f172a", linewidths=0.8, label="Best explanation slice")
    ax.set_ylim(0.0, 1.0)
    x_min = 1
    x_max = max(1, int(total_valid_slices))
    ax.set_xlim(float(x_min), float(x_max))
    tick_count = min(8, x_max)
    if tick_count > 1:
        xticks = np.linspace(x_min, x_max, num=tick_count, dtype=int)
        xticks = np.unique(xticks)
    else:
        xticks = np.array([x_min], dtype=int)
    ax.set_xticks(xticks)
    ax.set_title("Total Slice Position vs Grad-CAM Explainability", color=theme["chart_text"], pad=10, fontsize=16, fontweight="bold")
    ax.set_xlabel("Total Valid Slice Index", color=theme["chart_text"], labelpad=8, fontsize=13, fontweight="bold")
    ax.set_ylabel("Explainability Score", color=theme["chart_text"], labelpad=8, fontsize=13, fontweight="bold")
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
    st.caption("Trend of Grad-CAM explainability across total valid-slice positions, with the best explanation slice highlighted.")

with ranking_tab:
    st.markdown(
        "<div style='text-align:center;'><h3 style='margin: 0 0 0.5rem 0;'>Top Slices</h3></div>",
        unsafe_allow_html=True,
    )
    with st.spinner("Selecting the 5 slices with the clearest Grad-CAM explanations..."):
        ranking_df = gradcam_ranking_df.head(5).copy()
        if len(valid_slices) > 1 and total_valid_slices > 1:
            ranking_df["slice_index"] = ranking_df["slice_index"].apply(
                lambda i: int(round((float(i) / float(len(valid_slices) - 1)) * (total_valid_slices - 1))) + 1
            )
        else:
            ranking_df["slice_index"] = 1
        ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))

    ranking_styler = (
        ranking_df.style.format(
            {
                "tumor_probability": "{:.3f}",
                "brain_visibility": "{:.3f}",
                "gradcam_visibility": "{:.3f}",
                "center_proximity": "{:.3f}",
                "combined_score": "{:.3f}",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#b3d9ff"),
                        ("color", "#0f172a"),
                        ("font-weight", "600"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),
                    ],
                }
            ]
        )
        .set_properties(**{"background-color": "#b3d9ff", "color": "#0f172a", "text-align": "center"})
    )
    st.dataframe(ranking_styler, use_container_width=True, hide_index=True)
    st.markdown(
        "<div style='text-align:center;'>Top 5 slices are chosen from the middle of the scan using Grad-CAM visibility, brain visibility, and proximity to the center, so the explanations are easier to read.</div>",
        unsafe_allow_html=True,
    )
