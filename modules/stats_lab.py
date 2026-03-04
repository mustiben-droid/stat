"""
StatLab Pro — SPSS/JASP-style Statistical Analyzer
====================================================
Requirements:
    pip install streamlit pandas numpy scipy statsmodels pingouin plotly

Run:
    streamlit run statlab_pro.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
import io

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatLab Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  — Maximum JASP/SPSS visual fidelity
# ─────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* ── Reset & Base ───────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; }

    html, body, .stApp {
        background-color: #13151b !important;
        color: #e2e4ed !important;
        font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif !important;
    }

    /* ── Hide Streamlit chrome ──────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }

    /* ── Top header bar ─────────────────────────────────── */
    .app-header {
        background: linear-gradient(90deg, #0d0f14 0%, #161920 100%);
        border-bottom: 1px solid #252832;
        padding: 0 24px;
        height: 50px;
        display: flex;
        align-items: center;
        gap: 16px;
        position: sticky;
        top: 0;
        z-index: 999;
        margin: -1rem -1rem 1.5rem -1rem;
    }
    .app-header-logo {
        font-size: 15px; font-weight: 800; color: #4f9cf9;
        letter-spacing: .4px; display: flex; align-items: center; gap: 8px;
    }
    .app-header-sub {
        font-size: 11px; color: #4a5060; letter-spacing: .3px;
        border-left: 1px solid #252832; padding-left: 14px;
    }
    .app-header-badge {
        margin-left: auto; background: #1e2a3a; color: #4f9cf9;
        font-size: 10px; font-weight: 700; padding: 3px 10px;
        border-radius: 20px; border: 1px solid #2a3f5f; letter-spacing: .5px;
    }

    /* ── Sidebar ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0f1117 !important;
        border-right: 1px solid #1e2230 !important;
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }
    .sidebar-logo {
        background: linear-gradient(135deg, #0d1520 0%, #111827 100%);
        border-bottom: 1px solid #1e2230;
        padding: 16px 16px 12px;
        margin-bottom: 4px;
    }
    .sidebar-logo-title {
        font-size: 16px; font-weight: 800; color: #4f9cf9;
        letter-spacing: .3px; display: flex; align-items: center; gap: 8px;
    }
    .sidebar-logo-sub {
        font-size: 10px; color: #3a4255; margin-top: 3px;
        text-transform: uppercase; letter-spacing: .8px;
    }
    .sidebar-section {
        font-size: 9px; color: #2e3545; font-weight: 700;
        letter-spacing: 1.2px; text-transform: uppercase;
        padding: 12px 16px 4px; margin-top: 4px;
    }
    .sidebar-divider {
        height: 1px; background: #1a1e28; margin: 6px 12px;
    }

    /* Sidebar widgets */
    section[data-testid="stSidebar"] label {
        color: #9aa0b4 !important; font-size: 12px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-size: 12px !important; color: #c0c5d4 !important;
        padding: 2px 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        color: #4f9cf9 !important;
    }
    section[data-testid="stSidebar"] .stFileUploader {
        background: #161920 !important;
        border: 1px dashed #2a2f3d !important;
        border-radius: 6px;
    }

    /* ── Main content ───────────────────────────────────── */
    .main .block-container {
        padding: 1rem 1.5rem 2rem !important;
        max-width: 1400px !important;
    }

    /* ── Page title ─────────────────────────────────────── */
    .page-title {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 4px;
    }
    .page-title-icon {
        font-size: 22px;
    }
    .page-title-text {
        font-size: 20px; font-weight: 800; color: #e2e4ed;
        letter-spacing: -.2px;
    }
    .page-breadcrumb {
        font-size: 11px; color: #3a4255; margin-bottom: 16px;
        display: flex; align-items: center; gap: 6px;
    }
    .page-breadcrumb span { color: #2a3040; }

    /* ── Output panels ──────────────────────────────────── */
    .output-panel {
        background: #1a1d26;
        border: 1px solid #232735;
        border-radius: 8px;
        margin-bottom: 18px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,.35);
    }
    .panel-title {
        background: linear-gradient(90deg, #0d1018 0%, #11141c 100%);
        padding: 9px 16px;
        font-size: 11.5px;
        font-weight: 700;
        color: #4f9cf9;
        letter-spacing: .5px;
        border-bottom: 1px solid #1e2230;
        display: flex;
        align-items: center;
        gap: 8px;
        text-transform: uppercase;
    }
    .panel-title::before {
        content: '';
        display: inline-block;
        width: 3px; height: 12px;
        background: #4f9cf9;
        border-radius: 2px;
    }
    .panel-body { padding: 0; }

    /* ── Stat tables ────────────────────────────────────── */
    .stat-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        font-family: 'IBM Plex Mono', 'Fira Code', monospace;
    }
    .stat-table thead tr {
        background: #0f1117;
    }
    .stat-table th {
        color: #5a6478;
        padding: 8px 16px;
        text-align: right;
        font-size: 10.5px;
        letter-spacing: .6px;
        border-bottom: 2px solid #1e2230;
        font-weight: 700;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .stat-table th:first-child { text-align: left; padding-left: 16px; }
    .stat-table td {
        padding: 7px 16px;
        text-align: right;
        border-bottom: 1px solid #1a1d2620;
        font-variant-numeric: tabular-nums;
        color: #d0d3de;
        transition: background .1s;
    }
    .stat-table td:first-child {
        text-align: left;
        color: #9aa0b4;
        font-weight: 500;
    }
    .stat-table tbody tr:nth-child(even) { background: #1a1d26; }
    .stat-table tbody tr:nth-child(odd)  { background: #161920; }
    .stat-table tbody tr:hover           { background: #1f2535; }
    .stat-table tbody tr:last-child td   { border-bottom: none; }

    /* Total row */
    .stat-table tr.total-row td {
        font-weight: 700; color: #c0c5d4;
        border-top: 2px solid #2a2f3d;
        background: #111318 !important;
    }

    /* ── Notes ──────────────────────────────────────────── */
    .stat-note {
        font-size: 11px; color: #4a5060;
        padding: 8px 16px 10px;
        border-top: 1px solid #1a1d2630;
        background: #0f1117;
        font-style: italic;
        line-height: 1.6;
    }
    .stat-note b, .stat-note strong { color: #6b7280; font-style: normal; }

    /* ── Significance colors ─────────────────────────────── */
    .sig     { color: #fbbf24 !important; font-weight: 700; }
    .sig001  { color: #f87171 !important; font-weight: 700; }
    .nonsig  { color: #4a5568; }

    /* ── Alpha / reliability badge ───────────────────────── */
    .reliability-box {
        display: flex; gap: 40px; align-items: flex-start;
        padding: 16px 20px; background: #0f1117;
        border-bottom: 1px solid #1e2230;
    }
    .alpha-block { text-align: center; }
    .alpha-label { font-size: 10px; color: #4a5568; font-weight: 700;
                   letter-spacing: .8px; text-transform: uppercase; margin-bottom: 4px; }
    .alpha-value { font-size: 36px; font-weight: 800; line-height: 1; }
    .alpha-good       { color: #34d399; }
    .alpha-acceptable { color: #fbbf24; }
    .alpha-poor       { color: #f87171; }
    .alpha-interpret {
        font-size: 12px; color: #6b7280; margin-top: 8px;
        padding: 6px 12px; background: #161920;
        border-radius: 4px; border-left: 3px solid #2a2f3d;
    }

    /* ── Metric cards ────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #161920 0%, #1a1d26 100%);
        border: 1px solid #232735; border-radius: 8px;
        padding: 14px 20px; text-align: center;
        min-width: 110px;
    }
    .metric-card-label {
        font-size: 10px; color: #4a5568; font-weight: 700;
        text-transform: uppercase; letter-spacing: .8px; margin-bottom: 6px;
    }
    .metric-card-value {
        font-size: 22px; font-weight: 800; color: #e2e4ed;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-card-sub {
        font-size: 10px; color: #3a4255; margin-top: 4px;
    }

    /* ── Streamlit widget overrides ──────────────────────── */
    /* Multiselect */
    [data-testid="stMultiSelect"] > div > div {
        background: #161920 !important;
        border-color: #2a2f3d !important;
        border-radius: 6px !important;
        color: #c0c5d4 !important;
    }
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background: #1e2a3a !important;
        color: #4f9cf9 !important;
        border: 1px solid #2a3f5f !important;
        border-radius: 4px !important;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: #161920 !important;
        border-color: #2a2f3d !important;
        border-radius: 6px !important;
        color: #c0c5d4 !important;
    }

    /* Checkbox */
    [data-testid="stCheckbox"] label {
        color: #9aa0b4 !important; font-size: 12.5px !important;
    }
    [data-testid="stCheckbox"] input:checked + div {
        background: #4f9cf9 !important;
        border-color: #4f9cf9 !important;
    }

    /* Number input */
    [data-testid="stNumberInput"] input {
        background: #161920 !important;
        border-color: #2a2f3d !important;
        color: #c0c5d4 !important;
        border-radius: 6px !important;
    }

    /* Slider */
    [data-testid="stSlider"] div[role="slider"] {
        background: #4f9cf9 !important;
    }
    [data-testid="stSlider"] div[data-testid="stTickBar"] {
        color: #4a5568 !important;
    }

    /* Radio */
    [data-testid="stRadio"] label {
        color: #9aa0b4 !important;
        font-size: 12.5px !important;
    }
    [data-testid="stRadio"] input:checked + div {
        background: #4f9cf9 !important;
        border-color: #4f9cf9 !important;
    }

    /* Run button */
    .stButton > button {
        background: linear-gradient(135deg, #3a7bd5 0%, #4f9cf9 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 24px !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        letter-spacing: .3px !important;
        box-shadow: 0 2px 8px rgba(79,156,249,.3) !important;
        transition: all .2s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563c7 0%, #3a8ae8 100%) !important;
        box-shadow: 0 4px 16px rgba(79,156,249,.45) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background: #161920 !important;
        border: 1px dashed #2a2f3d !important;
        border-radius: 8px !important;
        color: #6b7280 !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #4f9cf9 !important;
        background: #1a2030 !important;
    }

    /* Tabs */
    [data-testid="stTabs"] [data-testid="stTab"] {
        background: transparent !important;
        color: #6b7280 !important;
        border: none !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }
    [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
        color: #4f9cf9 !important;
        border-bottom: 2px solid #4f9cf9 !important;
    }

    /* Success / info / warning banners */
    [data-testid="stSuccess"] {
        background: #0d2010 !important; border-color: #1a5c2a !important;
        color: #34d399 !important; border-radius: 6px !important;
    }
    [data-testid="stInfo"] {
        background: #0d1a30 !important; border-color: #1e3a5f !important;
        color: #60a5fa !important; border-radius: 6px !important;
    }
    [data-testid="stError"] {
        background: #200d0d !important; border-color: #5c1a1a !important;
        color: #f87171 !important; border-radius: 6px !important;
    }

    /* Divider */
    hr { border-color: #1e2230 !important; margin: 12px 0 !important; }

    /* Expander */
    [data-testid="stExpander"] {
        background: #161920 !important;
        border: 1px solid #232735 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary {
        color: #9aa0b4 !important; font-weight: 600 !important;
    }

    /* Caption */
    .stCaption { color: #3a4255 !important; font-size: 11px !important; }

    /* Plotly chart container */
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }

    /* Sidebar group label */
    .sidebar-group {
        font-size: 9px; color: #2e3545; font-weight: 700;
        letter-spacing: 1.2px; text-transform: uppercase;
        padding: 10px 14px 3px;
    }

    /* Analysis section heading */
    .analysis-header {
        background: linear-gradient(90deg, #0d1018, #13151b);
        border: 1px solid #1e2230;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .analysis-header-icon { font-size: 24px; }
    .analysis-header-title { font-size: 17px; font-weight: 800; color: #e2e4ed; }
    .analysis-header-sub { font-size: 11px; color: #3a4255; margin-top: 2px; }

    /* Options panel */
    .options-panel {
        background: #161920;
        border: 1px solid #1e2230;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .options-panel-title {
        font-size: 10px; color: #4a5568; font-weight: 700;
        letter-spacing: .8px; text-transform: uppercase;
        margin-bottom: 10px;
        display: flex; align-items: center; gap: 6px;
    }
    .options-panel-title::before {
        content: ''; display: inline-block;
        width: 3px; height: 10px;
        background: #2a3f5f; border-radius: 2px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #2a2f3d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a4050; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────
def fmt(val, dec=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{dec}f}"

def fmt_p(p):
    if p is None or np.isnan(p): return "—"
    if p < .001: return '<span class="sig">&lt; .001 <span style="font-size:10px">***</span></span>'
    stars = "**" if p < .01 else "*" if p < .05 else ""
    cls = "sig" if p < .05 else "nonsig"
    return f'<span class="{cls}">{p:.3f} <span style="font-size:10px">{stars}</span></span>'

def sig_note():
    return (
        '<div class="stat-note">'
        '<strong>Note.</strong> '
        '<span style="color:#fbbf24">*</span> p &lt; .05 &nbsp;'
        '<span style="color:#fbbf24">**</span> p &lt; .01 &nbsp;'
        '<span style="color:#fbbf24">***</span> p &lt; .001'
        '</div>'
    )

def render_table(headers, rows, caption=None, note_text=None):
    html = '<div class="output-panel">'
    if caption:
        html += f'<div class="panel-title">{caption}</div>'
    html += '<table class="stat-table"><thead><tr>'
    for h in headers:
        html += f"<th>{h}</th>"
    html += "</tr></thead><tbody>"
    for row in rows:
        html += "<tr>"
        for cell in row:
            html += f"<td>{cell}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    html += sig_note()
    if note_text:
        html += f'<div class="stat-note" style="border-top:none;padding-top:2px"><strong>Note.</strong> {note_text}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def note(text):
    st.markdown(
        f'<div class="stat-note" style="background:#0f1117;padding:8px 16px;'
        f'border-radius:0 0 8px 8px;border-top:1px solid #1a1d2630">'
        f'<strong>Note.</strong> {text}</div>',
        unsafe_allow_html=True
    )

# ── Top header bar
st.markdown("""
<div class="app-header">
  <div class="app-header-logo">🔬 StatLab Pro</div>
  <div class="app-header-sub">SPSS / JASP-style Statistical Analyzer</div>
  <div class="app-header-badge">v2.0</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DEMO DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def make_demo():
    np.random.seed(42)
    n = 90
    majors = np.tile(["Psychology", "Education", "Social Work"], n // 3)
    genders = np.random.choice(["Female", "Male"], n, p=[.55, .45])
    base = np.where(majors == "Psychology", 75, np.where(majors == "Education", 70, 68))
    df = pd.DataFrame({
        "ID": range(1, n+1),
        "Major": majors,
        "Gender": genders,
        "Pre_Score":  np.round(base + np.random.normal(0, 8, n), 1),
        "Mid_Score":  np.round(base + 5 + np.random.normal(0, 8, n), 1),
        "Post_Score": np.round(base + 10 + np.random.normal(0, 8, n), 1),
        "Item1": np.random.randint(1, 6, n),
        "Item2": np.random.randint(1, 6, n),
        "Item3": np.random.randint(1, 6, n),
        "Item4": np.random.randint(1, 6, n),
        "Item5": np.random.randint(1, 6, n),
        "Anxiety":    np.round(np.random.uniform(20, 60, n), 1),
        "Motivation": np.round(np.random.uniform(30, 70, n), 1),
    })
    return df


# ─────────────────────────────────────────────────────────────
# SIDEBAR — DATA LOADING + ANALYSIS MENU
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-title">🔬 StatLab Pro</div>
      <div class="sidebar-logo-sub">SPSS / JASP Statistical Suite</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data loading ──
    st.markdown('<div class="sidebar-group">📂 Dataset</div>', unsafe_allow_html=True)
    use_demo = st.button("⚡ Load Demo Data", use_container_width=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if "df" not in st.session_state:
        st.session_state.df = None

    if use_demo:
        st.session_state.df = make_demo()
        st.success(f"Demo loaded: {len(st.session_state.df)} rows")
    if uploaded:
        st.session_state.df = pd.read_csv(uploaded)
        st.success(f"Loaded: {len(st.session_state.df)} rows")

    df = st.session_state.df

    if df is not None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols     = df.select_dtypes(exclude=np.number).columns.tolist()
        all_cols     = df.columns.tolist()
        st.markdown(f"""
        <div style="background:#0f1117;border:1px solid #1e2230;border-radius:6px;
                    padding:10px 14px;margin:8px 0 4px;font-size:11px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span style="color:#4a5568;font-weight:700;text-transform:uppercase;font-size:9px;letter-spacing:.8px">Rows</span>
            <span style="color:#e2e4ed;font-family:'IBM Plex Mono',monospace;font-weight:700">{len(df)}</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span style="color:#4a5568;font-weight:700;text-transform:uppercase;font-size:9px;letter-spacing:.8px">Numeric</span>
            <span style="color:#4f9cf9;font-family:'IBM Plex Mono',monospace;font-weight:700">{len(numeric_cols)}</span>
          </div>
          <div style="display:flex;justify-content:space-between">
            <span style="color:#4a5568;font-weight:700;text-transform:uppercase;font-size:9px;letter-spacing:.8px">Categorical</span>
            <span style="color:#a78bfa;font-family:'IBM Plex Mono',monospace;font-weight:700">{len(cat_cols)}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        numeric_cols, cat_cols, all_cols = [], [], []

    st.divider()

    # ── Analysis menu ──
    MENU = {
        "🔍 Explore": ["Descriptive Statistics", "Frequencies", "Crosstabs & Chi-Square"],
        "📐 T-Tests":  ["Independent Samples T-Test", "Paired Samples T-Test", "One-Sample T-Test"],
        "📊 ANOVA":    ["One-Way ANOVA", "Two-Way ANOVA", "Repeated Measures ANOVA", "ANCOVA"],
        "🔗 Regression": ["Bivariate Correlation", "Linear Regression", "Logistic Regression"],
        "🛡️ Scale":    ["Reliability (Cronbach's α)", "Exploratory Factor Analysis"],
        "🃏 Nonparametric": ["Mann-Whitney U", "Wilcoxon Signed-Rank", "Kruskal-Wallis"],
    }

    selected = None
    for group, items in MENU.items():
        st.markdown(f'<div class="sidebar-group">{group}</div>', unsafe_allow_html=True)
        choice = st.radio(group, items, label_visibility="collapsed", key=f"radio_{group}")
        if choice:
            selected = choice


# ─────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────
if df is None:
    st.markdown("""
    <div style="text-align:center; padding:80px 0; color:#3a4255;">
        <div style="font-size:64px;margin-bottom:20px;filter:grayscale(30%)">📂</div>
        <div style="font-size:22px; color:#6b7280; font-weight:800; margin:0 0 8px; letter-spacing:-.3px;">No Dataset Loaded</div>
        <div style="font-size:13px;color:#3a4255;margin-bottom:32px">Upload a CSV file or load the demo dataset from the sidebar to begin</div>
        <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap">
          <div style="background:#161920;border:1px solid #232735;border-radius:8px;padding:16px 24px;min-width:160px">
            <div style="font-size:24px;margin-bottom:8px">🎓</div>
            <div style="font-size:12px;font-weight:700;color:#9aa0b4">Demo Dataset</div>
            <div style="font-size:11px;color:#3a4255;margin-top:4px">90 students · 3 majors<br>Pre/Mid/Post scores</div>
          </div>
          <div style="background:#161920;border:1px solid #232735;border-radius:8px;padding:16px 24px;min-width:160px">
            <div style="font-size:24px;margin-bottom:8px">📊</div>
            <div style="font-size:12px;font-weight:700;color:#9aa0b4">Your Data</div>
            <div style="font-size:11px;color:#3a4255;margin-top:4px">Upload any CSV file<br>UTF-8 encoded</div>
          </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

ANALYSIS_META = {
    "Descriptive Statistics":      ("📊", "Explore"),
    "Frequencies":                 ("📋", "Explore"),
    "Crosstabs & Chi-Square":      ("🔲", "Explore"),
    "Independent Samples T-Test":  ("⚖️",  "T-Tests"),
    "Paired Samples T-Test":       ("🔁", "T-Tests"),
    "One-Sample T-Test":           ("🎯", "T-Tests"),
    "One-Way ANOVA":               ("📈", "ANOVA"),
    "Two-Way ANOVA":               ("📐", "ANOVA"),
    "Repeated Measures ANOVA":     ("🔄", "ANOVA"),
    "ANCOVA":                      ("🧮", "ANOVA"),
    "Bivariate Correlation":       ("🔗", "Regression"),
    "Linear Regression":           ("📉", "Regression"),
    "Logistic Regression":         ("🔮", "Regression"),
    "Reliability (Cronbach's α)":  ("🛡️",  "Scale"),
    "Exploratory Factor Analysis": ("🧬", "Scale"),
    "Mann-Whitney U":              ("🃏", "Nonparametric"),
    "Wilcoxon Signed-Rank":        ("📌", "Nonparametric"),
    "Kruskal-Wallis":              ("🏔️",  "Nonparametric"),
}

icon, group = ANALYSIS_META.get(selected, ("🔬", "Analysis"))
st.markdown(f"""
<div class="analysis-header">
  <div class="analysis-header-icon">{icon}</div>
  <div>
    <div class="analysis-header-title">{selected}</div>
    <div class="analysis-header-sub">{group} › {selected}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════
if selected == "Descriptive Statistics":
    col1, col2 = st.columns([1.2, 1])
    with col1:
        vars_d = st.multiselect("Dependent Variables", numeric_cols)
        group_d = st.selectbox("Split by (optional)", ["None"] + all_cols)
    with col2:
        st.markdown("**Statistics to include:**")
        inc_n      = st.checkbox("N", value=True)
        inc_mean   = st.checkbox("Mean", value=True)
        inc_std    = st.checkbox("Std. Deviation", value=True)
        inc_sem    = st.checkbox("Std. Error of Mean", value=True)
        inc_med    = st.checkbox("Median", value=True)
        inc_range  = st.checkbox("Min / Max", value=True)
        inc_skew   = st.checkbox("Skewness", value=True)
        inc_kurt   = st.checkbox("Kurtosis", value=True)
        inc_ci     = st.checkbox("95% CI for Mean", value=True)

    if st.button("▶ Run Descriptives") and vars_d:
        def run_desc(data_sub, label=""):
            rows = []
            for v in vars_d:
                s = data_sub[v].dropna()
                n = len(s)
                m = s.mean(); sd = s.std(); se = sd / np.sqrt(n)
                ci_lo = m - 1.96 * se; ci_hi = m + 1.96 * se
                row = [f"{label}{v}" if label else v]
                if inc_n:    row.append(n)
                if inc_mean: row.append(fmt(m))
                if inc_std:  row.append(fmt(sd))
                if inc_sem:  row.append(fmt(se))
                if inc_med:  row.append(fmt(s.median()))
                if inc_range:row += [fmt(s.min()), fmt(s.max())]
                if inc_skew: row.append(fmt(stats.skew(s)))
                if inc_kurt: row.append(fmt(stats.kurtosis(s)))
                if inc_ci:   row.append(f"[{fmt(ci_lo,2)}, {fmt(ci_hi,2)}]")
                rows.append(row)
            return rows

        headers = ["Variable"]
        if inc_n:     headers.append("N")
        if inc_mean:  headers.append("Mean")
        if inc_std:   headers.append("Std. Dev.")
        if inc_sem:   headers.append("SE Mean")
        if inc_med:   headers.append("Median")
        if inc_range: headers += ["Min", "Max"]
        if inc_skew:  headers.append("Skewness")
        if inc_kurt:  headers.append("Kurtosis")
        if inc_ci:    headers.append("95% CI [Lo, Hi]")

        if group_d == "None":
            rows = run_desc(df)
        else:
            rows = []
            for grp, sub in df.groupby(group_d):
                rows += run_desc(sub, label=f"{grp} — ")

        render_table(headers, rows, caption="Descriptive Statistics")

        # Boxplot
        if len(vars_d) <= 6:
            melted = df[vars_d].melt(var_name="Variable", value_name="Value")
            fig = px.box(melted, x="Variable", y="Value", color="Variable",
                         template="plotly_dark",
                         color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(showlegend=False, paper_bgcolor="#1a1d23",
                              plot_bgcolor="#22262f", margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 2. FREQUENCIES
# ══════════════════════════════════════════════════════════════
elif selected == "Frequencies":
    var_f = st.selectbox("Variable", all_cols)
    if st.button("▶ Run Frequencies"):
        vc = df[var_f].value_counts(dropna=False)
        total = vc.sum()
        rows = []
        cum = 0
        for val, freq in vc.items():
            pct = freq / total * 100
            cum += pct
            rows.append([str(val), freq, f"{pct:.1f}%", f"{pct:.1f}%", f"{cum:.1f}%"])
        rows.append(["<b>Total</b>", total, "100.0%", "100.0%", "—"])
        render_table(["Value", "Frequency", "Percent", "Valid Percent", "Cumulative %"],
                     rows, caption=f"Frequencies — {var_f}")
        fig = px.bar(x=vc.index.astype(str), y=vc.values,
                     labels={"x": var_f, "y": "Frequency"},
                     template="plotly_dark", color=vc.values,
                     color_continuous_scale="Blues")
        fig.update_layout(paper_bgcolor="#1a1d23", plot_bgcolor="#22262f",
                          showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 3. CROSSTABS & CHI-SQUARE
# ══════════════════════════════════════════════════════════════
elif selected == "Crosstabs & Chi-Square":
    col1, col2 = st.columns(2)
    row_v = col1.selectbox("Row Variable", all_cols)
    col_v = col2.selectbox("Column Variable", [c for c in all_cols if c != row_v])
    show_exp   = st.checkbox("Show Expected Counts", value=True)
    show_resid = st.checkbox("Show Standardized Residuals", value=False)

    if st.button("▶ Run Crosstabs"):
        ct = pd.crosstab(df[row_v], df[col_v], margins=True, margins_name="Total")
        st.markdown('<div class="output-panel"><div class="panel-title">Crosstabulation</div><div class="panel-body">', unsafe_allow_html=True)
        st.dataframe(ct, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

        chi2, p, dof, expected = stats.chi2_contingency(
            pd.crosstab(df[row_v], df[col_v])
        )
        n = len(df.dropna(subset=[row_v, col_v]))
        cramers_v = np.sqrt(chi2 / (n * (min(pd.crosstab(df[row_v], df[col_v]).shape) - 1)))
        render_table(
            ["", "Value", "df", "Asymp. Sig. (2-sided)"],
            [
                ["Pearson Chi-Square", fmt(chi2), dof, fmt_p(p)],
                ["Cramér's V", fmt(cramers_v), "—", "—"],
                ["N of Valid Cases", n, "—", "—"],
            ],
            caption="Chi-Square Tests"
        )
        if show_exp:
            exp_df = pd.DataFrame(expected,
                                  index=pd.crosstab(df[row_v], df[col_v]).index,
                                  columns=pd.crosstab(df[row_v], df[col_v]).columns)
            st.markdown('<div class="output-panel"><div class="panel-title">Expected Counts</div><div class="panel-body">', unsafe_allow_html=True)
            st.dataframe(exp_df.round(1), use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 4. INDEPENDENT SAMPLES T-TEST
# ══════════════════════════════════════════════════════════════
elif selected == "Independent Samples T-Test":
    col1, col2 = st.columns([1.2, 1])
    with col1:
        dv = col1.selectbox("Test Variable (DV)", numeric_cols)
        iv = col1.selectbox("Grouping Variable", all_cols)
    with col2:
        st.markdown("**Options:**")
        opt_levene = st.checkbox("Levene's Test for Equality of Variances", True)
        opt_desc   = st.checkbox("Group Descriptives", True)
        opt_ci     = st.checkbox("95% CI for Mean Difference", True)
        opt_es     = st.checkbox("Effect Size (Cohen's d)", True)
        opt_plot   = st.checkbox("Raincloud / Box Plot", True)
        confidence = st.slider("Confidence Level", 90, 99, 95)

    if st.button("▶ Run T-Test"):
        groups = df[iv].dropna().unique()
        if len(groups) < 2:
            st.error("Need at least 2 groups.")
            st.stop()
        g1_label, g2_label = groups[0], groups[1]
        g1 = df[df[iv] == g1_label][dv].dropna()
        g2 = df[df[iv] == g2_label][dv].dropna()

        if opt_desc:
            render_table(
                ["Group", "N", "Mean", "Std. Deviation", "Std. Error Mean"],
                [
                    [g1_label, len(g1), fmt(g1.mean()), fmt(g1.std()), fmt(g1.sem())],
                    [g2_label, len(g2), fmt(g2.mean()), fmt(g2.std()), fmt(g2.sem())],
                ],
                caption="Group Statistics"
            )

        lev_stat, lev_p = stats.levene(g1, g2)
        if opt_levene:
            render_table(
                ["", "F", "df1", "df2", "Sig."],
                [["Levene's Test", fmt(lev_stat), 1,
                  len(g1)+len(g2)-2, fmt_p(lev_p)]],
                caption="Test of Equality of Variances"
            )

        # Equal + Welch
        t_eq, p_eq = stats.ttest_ind(g1, g2, equal_var=True)
        t_wl, p_wl = stats.ttest_ind(g1, g2, equal_var=False)
        df_eq = len(g1) + len(g2) - 2
        sp = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / df_eq)
        cohen_d = (g1.mean() - g2.mean()) / sp
        alpha = 1 - confidence / 100
        t_crit = stats.t.ppf(1 - alpha/2, df_eq)
        se_diff = sp * np.sqrt(1/len(g1) + 1/len(g2))
        diff = g1.mean() - g2.mean()
        ci_lo = diff - t_crit * se_diff
        ci_hi = diff + t_crit * se_diff

        rows = [["Equal variances assumed", fmt(t_eq), fmt(df_eq,0),
                 fmt_p(p_eq), fmt(cohen_d) if opt_es else None,
                 f"[{fmt(ci_lo)}, {fmt(ci_hi)}]" if opt_ci else None],
                ["Equal variances not assumed (Welch)", fmt(t_wl), "—",
                 fmt_p(p_wl), "—" if opt_es else None, "—" if opt_ci else None]]
        headers = ["", "t", "df", "Sig. (2-tailed)"]
        if opt_es: headers.append("Cohen's d")
        if opt_ci: headers.append(f"{confidence}% CI [Lo, Hi]")
        rows = [[c for c in r if c is not None] for r in rows]
        render_table(headers, rows, caption="Independent Samples Test")

        if opt_plot:
            fig = px.box(df[df[iv].isin([g1_label, g2_label])],
                         x=iv, y=dv, color=iv, points="all",
                         template="plotly_dark",
                         color_discrete_sequence=["#4f9cf9", "#3ecf8e"])
            fig.update_layout(paper_bgcolor="#1a1d23", plot_bgcolor="#22262f",
                              showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 5. PAIRED SAMPLES T-TEST
# ══════════════════════════════════════════════════════════════
elif selected == "Paired Samples T-Test":
    col1, col2 = st.columns([1.2, 1])
    v1 = col1.selectbox("Variable 1 (Time 1)", numeric_cols)
    v2 = col1.selectbox("Variable 2 (Time 2)", [c for c in numeric_cols if c != v1])
    opt_desc = col2.checkbox("Paired Descriptives", True)
    opt_ci   = col2.checkbox("95% CI for Mean Difference", True)
    opt_es   = col2.checkbox("Cohen's dz", True)
    opt_plot = col2.checkbox("Difference Histogram", True)

    if st.button("▶ Run Paired T-Test"):
        pair_df = df[[v1, v2]].dropna()
        a, b = pair_df[v1], pair_df[v2]
        t_stat, p_val = stats.ttest_rel(a, b)
        diffs = a - b
        n = len(diffs)
        md = diffs.mean(); sd_d = diffs.std(); se_d = sd_d / np.sqrt(n)
        ci_lo = md - 1.96 * se_d; ci_hi = md + 1.96 * se_d
        dz = md / sd_d

        if opt_desc:
            render_table(
                ["", "Mean", "N", "Std. Deviation", "Std. Error Mean"],
                [[v1, fmt(a.mean()), n, fmt(a.std()), fmt(a.sem())],
                 [v2, fmt(b.mean()), n, fmt(b.std()), fmt(b.sem())]],
                caption="Paired Samples Statistics"
            )

        row = [f"{v1} − {v2}", fmt(md), fmt(sd_d), fmt(se_d)]
        headers = ["Pair", "Mean Diff.", "Std. Dev.", "SE Mean"]
        if opt_ci:
            row.append(f"[{fmt(ci_lo)}, {fmt(ci_hi)}]")
            headers.append("95% CI [Lo, Hi]")
        row += [fmt(t_stat), n - 1, fmt_p(p_val)]
        headers += ["t", "df", "Sig. (2-tailed)"]
        if opt_es:
            row.append(fmt(dz))
            headers.append("Cohen's dz")
        render_table(headers, [row], caption="Paired Samples Test")

        if opt_plot:
            fig = px.histogram(diffs, nbins=20, template="plotly_dark",
                               labels={"value": f"{v1} − {v2}", "count": "Frequency"},
                               color_discrete_sequence=["#4f9cf9"])
            fig.add_vline(x=0, line_dash="dash", line_color="#f45b5b")
            fig.add_vline(x=md, line_color="#ffdd57", annotation_text=f"Mean Diff = {fmt(md)}")
            fig.update_layout(paper_bgcolor="#1a1d23", plot_bgcolor="#22262f",
                              showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 6. ONE-SAMPLE T-TEST
# ══════════════════════════════════════════════════════════════
elif selected == "One-Sample T-Test":
    dv  = st.selectbox("Test Variable", numeric_cols)
    mu0 = st.number_input("Test Value (μ₀)", value=0.0)
    opt_ci = st.checkbox("95% CI for Mean Difference", True)
    opt_es = st.checkbox("Cohen's d", True)

    if st.button("▶ Run One-Sample T-Test"):
        s = df[dv].dropna()
        t, p = stats.ttest_1samp(s, mu0)
        n = len(s); m = s.mean(); sd = s.std(); se = s.sem()
        ci_lo = m - 1.96*se; ci_hi = m + 1.96*se
        d = (m - mu0) / sd
        render_table(
            ["Variable", "N", "Mean", "Std. Dev.", "Std. Error"],
            [[dv, n, fmt(m), fmt(sd), fmt(se)]],
            caption="One-Sample Statistics"
        )
        row = [dv, fmt(t), n-1, fmt_p(p), fmt(m - mu0)]
        headers = ["Variable", "t", "df", "Sig. (2-tailed)", "Mean Diff."]
        if opt_ci:
            row.append(f"[{fmt(ci_lo)}, {fmt(ci_hi)}]")
            headers.append("95% CI [Lo, Hi]")
        if opt_es:
            row.append(fmt(d))
            headers.append("Cohen's d")
        render_table(headers, [row], caption=f"One-Sample Test  (test value = {mu0})")


# ══════════════════════════════════════════════════════════════
# 7. ONE-WAY ANOVA
# ══════════════════════════════════════════════════════════════
elif selected == "One-Way ANOVA":
    col1, col2 = st.columns([1.2, 1])
    dv     = col1.selectbox("Dependent Variable", numeric_cols)
    factor = col1.selectbox("Factor (Between-Subjects)", all_cols)
    opt_desc  = col2.checkbox("Descriptive Statistics", True)
    opt_lev   = col2.checkbox("Levene's Test", True)
    opt_post  = col2.checkbox("Post-Hoc: Tukey HSD", True)
    opt_bonf  = col2.checkbox("Post-Hoc: Bonferroni", False)
    opt_eta   = col2.checkbox("Effect Size (η², ω²)", True)
    opt_plot  = col2.checkbox("Means Plot", True)

    if st.button("▶ Run One-Way ANOVA"):
        sub = df[[dv, factor]].dropna()
        groups = sub.groupby(factor)[dv].apply(list)
        group_names = groups.index.tolist()
        group_arrs  = [np.array(g) for g in groups]

        if opt_desc:
            rows = [[name, len(arr), fmt(arr.mean()), fmt(arr.std()),
                     fmt(stats.sem(arr)), fmt(arr.mean() - 1.96*stats.sem(arr)),
                     fmt(arr.mean() + 1.96*stats.sem(arr))]
                    for name, arr in zip(group_names, group_arrs)]
            render_table(
                ["Group", "N", "Mean", "Std. Dev.", "Std. Error", "CI Lower", "CI Upper"],
                rows, caption="Descriptive Statistics"
            )

        f_stat, p_val = stats.f_oneway(*group_arrs)
        N = len(sub); k = len(group_arrs)
        grand_mean = sub[dv].mean()
        ss_between = sum(len(arr)*(arr.mean()-grand_mean)**2 for arr in group_arrs)
        ss_within  = sum(((arr - arr.mean())**2).sum() for arr in group_arrs)
        ss_total   = ss_between + ss_within
        df_b = k - 1; df_w = N - k
        ms_b = ss_between / df_b; ms_w = ss_within / df_w
        eta2  = ss_between / ss_total
        omega2 = (ss_between - df_b * ms_w) / (ss_total + ms_w)

        rows_an = [
            ["Between Groups", fmt(ss_between), df_b, fmt(ms_b), fmt(f_stat), fmt_p(p_val),
             fmt(eta2) if opt_eta else None, fmt(omega2) if opt_eta else None],
            ["Within Groups",  fmt(ss_within),  df_w, fmt(ms_w), "—", "—",
             "—" if opt_eta else None, "—" if opt_eta else None],
            ["Total",          fmt(ss_total),   N-1,  "—", "—", "—",
             "—" if opt_eta else None, "—" if opt_eta else None],
        ]
        headers_an = ["Source", "SS", "df", "MS", "F", "Sig."]
        if opt_eta: headers_an += ["η²", "ω²"]
        rows_an = [[c for c in r if c is not None] for r in rows_an]
        render_table(headers_an, rows_an, caption="ANOVA Table")

        if opt_lev:
            lev, lev_p = stats.levene(*group_arrs)
            render_table(["Statistic", "df1", "df2", "Sig."],
                         [[fmt(lev), k-1, N-k, fmt_p(lev_p)]],
                         caption="Levene's Test for Homogeneity of Variances")

        if opt_post:
            tukey = pairwise_tukeyhsd(sub[dv], sub[factor])
            tdf = pd.DataFrame(data=tukey._results_table.data[1:],
                               columns=tukey._results_table.data[0])
            rows_t = [[row["group1"], row["group2"], fmt(float(row["meandiff"])),
                       fmt(float(row["lower"])), fmt(float(row["upper"])),
                       fmt_p(float(row["p-adj"]))]
                      for _, row in tdf.iterrows()]
            render_table(["Group (I)", "Group (J)", "Mean Diff. (I-J)",
                          "Lower Bound", "Upper Bound", "Sig."],
                         rows_t, caption="Post-Hoc: Tukey HSD")

        if opt_bonf:
            pairs = [(g1, g2) for i, g1 in enumerate(group_names)
                     for g2 in group_names[i+1:]]
            raw_ps = [stats.ttest_ind(np.array(groups[g1]),
                                      np.array(groups[g2]))[1] for g1, g2 in pairs]
            _, adj_ps, _, _ = multipletests(raw_ps, method="bonferroni")
            rows_b = [[g1, g2, fmt(np.array(groups[g1]).mean() - np.array(groups[g2]).mean()),
                       fmt_p(ap)]
                      for (g1, g2), ap in zip(pairs, adj_ps)]
            render_table(["Group (I)", "Group (J)", "Mean Diff. (I-J)", "Sig. (Bonferroni)"],
                         rows_b, caption="Post-Hoc: Bonferroni")

        if opt_plot:
            means_df = sub.groupby(factor)[dv].agg(["mean", "sem"]).reset_index()
            means_df.columns = [factor, "mean", "sem"]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=means_df[factor], y=means_df["mean"],
                error_y=dict(type="data", array=means_df["sem"]*1.96, visible=True),
                marker_color="#4f9cf9", marker_line_color="#2a3f5f",
                marker_line_width=1
            ))
            fig.update_layout(title=f"Means ± 95% CI: {dv} by {factor}",
                              xaxis_title=factor, yaxis_title=f"Mean {dv}",
                              template="plotly_dark", paper_bgcolor="#1a1d23",
                              plot_bgcolor="#22262f", margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 8. REPEATED MEASURES ANOVA
# ══════════════════════════════════════════════════════════════
elif selected == "Repeated Measures ANOVA":
    col1, col2 = st.columns([1.2, 1])
    levels    = col1.multiselect("Time Points / Levels", numeric_cols)
    between   = col1.selectbox("Between-Subjects Factor (optional)", ["None"] + all_cols)
    opt_spher = col2.checkbox("Mauchly's Sphericity Test", True)
    opt_desc  = col2.checkbox("Descriptive Statistics", True)
    opt_eta   = col2.checkbox("Effect Size (η²p)", True)
    opt_plot  = col2.checkbox("Profile Plot", True)

    if st.button("▶ Run RM ANOVA") and len(levels) > 1:
        sub = df[levels + ([between] if between != "None" else [])].dropna().copy()
        sub["ID"] = range(len(sub))
        long_df = pd.melt(sub, id_vars=["ID"] + ([between] if between != "None" else []),
                          value_vars=levels, var_name="Time", value_name="Score")
        long_df["Time"] = pd.Categorical(long_df["Time"], categories=levels, ordered=True)

        if opt_desc:
            desc = long_df.groupby("Time")["Score"].agg(
                N="count", Mean="mean", SD="std"
            ).reset_index()
            rows_d = [[row["Time"], int(row["N"]), fmt(row["Mean"]), fmt(row["SD"])]
                      for _, row in desc.iterrows()]
            render_table(["Time Point", "N", "Mean", "Std. Dev."], rows_d,
                         caption="Descriptive Statistics")

        # Pingouin RM ANOVA
        try:
            if between == "None":
                aov = pg.rm_anova(data=long_df, dv="Score", within="Time",
                                  subject="ID", detailed=True)
            else:
                aov = pg.mixed_anova(data=long_df, dv="Score", within="Time",
                                     between=between, subject="ID")
            aov_rows = []
            for _, row in aov.iterrows():
                r = [row.get("Source", "—"), fmt(row.get("SS", np.nan)),
                     fmt(row.get("ddof1", row.get("DF", np.nan)), 0),
                     fmt(row.get("MS", row.get("SS", np.nan) / max(row.get("ddof1", 1), 1))),
                     fmt(row.get("F", np.nan)),
                     fmt_p(row.get("p-unc", row.get("p-GG-corr", np.nan)))]
                if opt_eta:
                    r.append(fmt(row.get("np2", row.get("eta_sq", np.nan))))
                aov_rows.append(r)
            headers_a = ["Source", "SS", "df", "MS", "F", "p"]
            if opt_eta: headers_a.append("η²p")
            render_table(headers_a, aov_rows, caption="Within-Subjects Effects")

            if opt_spher and between == "None":
                spher = pg.sphericity(data=long_df, dv="Score", within="Time", subject="ID")
                render_table(
                    ["", "Mauchly's W", "Approx. χ²", "df", "Sig."],
                    [["Sphericity", fmt(spher.W), fmt(spher.chi2), fmt(spher.dof, 0),
                      fmt_p(spher.pval)]],
                    caption="Mauchly's Test of Sphericity"
                )
        except Exception as e:
            st.error(f"RM ANOVA error: {e}")

        if opt_plot:
            means_t = long_df.groupby(["Time"] + ([between] if between != "None" else []))["Score"].mean().reset_index()
            if between != "None":
                fig = px.line(means_t, x="Time", y="Score", color=between,
                              markers=True, template="plotly_dark",
                              color_discrete_sequence=px.colors.qualitative.Bold)
            else:
                fig = px.line(means_t, x="Time", y="Score", markers=True,
                              template="plotly_dark",
                              color_discrete_sequence=["#4f9cf9"])
            fig.update_layout(paper_bgcolor="#1a1d23", plot_bgcolor="#22262f",
                              yaxis_title="Mean Score", margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 9. BIVARIATE CORRELATION
# ══════════════════════════════════════════════════════════════
elif selected == "Bivariate Correlation":
    col1, col2 = st.columns([1.2, 1])
    vars_c  = col1.multiselect("Variables", numeric_cols)
    method  = col2.radio("Method", ["Pearson", "Spearman", "Kendall"])
    opt_n   = col2.checkbox("Show N", True)
    opt_sig = col2.checkbox("Show Significance", True)
    opt_ci  = col2.checkbox("95% CI (Pearson only)", False)
    opt_heatmap = col2.checkbox("Heatmap", True)

    if st.button("▶ Run Correlation") and len(vars_c) >= 2:
        sub = df[vars_c].dropna()
        n = len(sub)
        m = method.lower()

        # Build matrix
        html = '<div class="output-panel"><div class="panel-title">Correlation Matrix</div><div class="panel-body">'
        html += '<table class="stat-table"><thead><tr><th></th>'
        for v in vars_c: html += f"<th>{v}</th>"
        html += "</tr></thead><tbody>"

        for i, vi in enumerate(vars_c):
            html += f"<tr><td><b>{vi}</b></td>"
            for j, vj in enumerate(vars_c):
                if i == j:
                    html += "<td style='color:#5a6070'>—</td>"
                else:
                    a, b_arr = sub[vi], sub[vj]
                    if m == "pearson": r, p = stats.pearsonr(a, b_arr)
                    elif m == "spearman": r, p = stats.spearmanr(a, b_arr)
                    else: r, p = stats.kendalltau(a, b_arr)
                    stars = pStar = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                    r_col = "#4f9cf9" if abs(r) > .5 else "#e8eaf0"
                    cell = f'<span style="color:{r_col};font-weight:700">{r:.3f}</span><span style="color:#ffdd57">{stars}</span>'
                    if opt_sig:
                        p_str = "< .001" if p < .001 else f"{p:.3f}"
                        cell += f'<br><span style="font-size:10px;color:#8891a4">p = {p_str}</span>'
                    if opt_n:
                        cell += f'<br><span style="font-size:10px;color:#5a6070">n = {n}</span>'
                    html += f"<td>{cell}</td>"
            html += "</tr>"
        html += "</tbody></table>"
        html += sig_note()
        html += '</div></div>'
        st.markdown(html, unsafe_allow_html=True)

        if opt_heatmap:
            corr_mat = sub.corr(method=m)
            fig = px.imshow(corr_mat, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            template="plotly_dark")
            fig.update_layout(paper_bgcolor="#1a1d23", plot_bgcolor="#22262f",
                              margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 10. LINEAR REGRESSION
# ══════════════════════════════════════════════════════════════
elif selected == "Linear Regression":
    col1, col2 = st.columns([1.2, 1])
    dv  = col1.selectbox("Dependent Variable", numeric_cols)
    ivs = col1.multiselect("Independent Variable(s)", [c for c in numeric_cols if c != dv])
    opt_model = col2.checkbox("Model Summary", True)
    opt_anova = col2.checkbox("ANOVA Table", True)
    opt_coef  = col2.checkbox("Coefficients", True)
    opt_ci    = col2.checkbox("95% CI for B", True)
    opt_std   = col2.checkbox("Standardized β", True)
    opt_diag  = col2.checkbox("Residual Plot", True)

    if st.button("▶ Run Regression") and ivs:
        sub = df[[dv] + ivs].dropna()
        X = sm.add_constant(sub[ivs])
        y = sub[dv]
        model = sm.OLS(y, X).fit()

        if opt_model:
            render_table(
                ["R", "R²", "Adj. R²", "Std. Error of Estimate", "F", "df1", "df2", "Sig. F"],
                [[fmt(np.sqrt(model.rsquared)), fmt(model.rsquared),
                  fmt(model.rsquared_adj), fmt(np.sqrt(model.mse_resid)),
                  fmt(model.fvalue), len(ivs), int(model.df_resid),
                  fmt_p(model.f_pvalue)]],
                caption="Model Summary"
            )

        if opt_anova:
            ss_reg = model.ess; ss_res = model.ssr; ss_tot = ss_reg + ss_res
            render_table(
                ["", "Sum of Squares", "df", "Mean Square", "F", "Sig."],
                [["Regression", fmt(ss_reg), len(ivs), fmt(ss_reg/len(ivs)),
                  fmt(model.fvalue), fmt_p(model.f_pvalue)],
                 ["Residual", fmt(ss_res), int(model.df_resid),
                  fmt(model.mse_resid), "—", "—"],
                 ["Total", fmt(ss_tot), int(model.df_resid)+len(ivs), "—", "—", "—"]],
                caption="ANOVA"
            )

        if opt_coef:
            # Standardized betas via correlation
            std_betas = {}
            for iv_ in ivs:
                std_betas[iv_] = model.params[iv_] * sub[iv_].std() / sub[dv].std()
            rows_c = []
            for name in ["const"] + ivs:
                b = model.params[name]; se = model.bse[name]
                t = model.tvalues[name]; p = model.pvalues[name]
                row = [name if name != "const" else "(Constant)",
                       fmt(b), fmt(se),
                       fmt(std_betas.get(name, np.nan)) if opt_std else None,
                       fmt(t), fmt_p(p)]
                if opt_ci:
                    ci = model.conf_int()
                    row.append(f"[{fmt(ci.loc[name, 0])}, {fmt(ci.loc[name, 1])}]")
                rows_c.append([c for c in row if c is not None])
            headers_c = ["", "B", "Std. Error"]
            if opt_std: headers_c.append("β")
            headers_c += ["t", "Sig."]
            if opt_ci: headers_c.append("95% CI [Lo, Hi]")
            render_table(headers_c, rows_c, caption="Coefficients")

        if opt_diag:
            fitted = model.fittedvalues; resid = model.resid
            fig = px.scatter(x=fitted, y=resid, template="plotly_dark",
                             labels={"x": "Fitted Values", "y": "Residuals"},
                             color_discrete_sequence=["#4f9cf9"])
            fig.add_hline(y=0, line_dash="dash", line_color="#f45b5b")
            fig.update_layout(title="Residuals vs. Fitted",
                              paper_bgcolor="#1a1d23", plot_bgcolor="#22262f",
                              margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 11. RELIABILITY (CRONBACH'S ALPHA)
# ══════════════════════════════════════════════════════════════
elif selected == "Reliability (Cronbach's α)":
    col1, col2 = st.columns([1.2, 1])
    items = col1.multiselect("Scale Items", numeric_cols)
    opt_item  = col2.checkbox("Item Statistics", True)
    opt_iftd  = col2.checkbox("Item-Total Statistics & α if Deleted", True)
    opt_scale = col2.checkbox("Scale Statistics", True)
    opt_corr  = col2.checkbox("Inter-Item Correlation Matrix", False)

    if st.button("▶ Run Reliability") and len(items) >= 2:
        sub = df[items].dropna()
        n = len(sub); k = len(items)

        # Cronbach's alpha
        item_vars  = sub.var(axis=0, ddof=1)
        total_var  = sub.sum(axis=1).var(ddof=1)
        alpha      = (k / (k - 1)) * (1 - item_vars.sum() / total_var)

        # Badge
        cls = "alpha-good" if alpha > .8 else "alpha-acceptable" if alpha > .7 else "alpha-poor"
        label = ("Excellent (α > .90)" if alpha > .9 else
                 "Good (α > .80)" if alpha > .8 else
                 "Acceptable (α > .70)" if alpha > .7 else
                 "Questionable (α > .60)" if alpha > .6 else "Poor (α < .60)")
        bar_color = "#34d399" if alpha > .8 else "#fbbf24" if alpha > .7 else "#f87171"
        bar_width  = max(4, min(100, int(alpha * 100)))
        st.markdown(f"""
        <div class="output-panel">
          <div class="panel-title">Reliability Statistics</div>
          <div class="reliability-box">
            <div class="alpha-block">
              <div class="alpha-label">Cronbach's Alpha</div>
              <div class="alpha-value {cls}">{alpha:.3f}</div>
              <div style="margin-top:8px;background:#1e2230;border-radius:4px;height:4px;width:120px">
                <div style="background:{bar_color};height:4px;border-radius:4px;width:{bar_width}%"></div>
              </div>
            </div>
            <div class="alpha-block">
              <div class="alpha-label">N of Items</div>
              <div class="alpha-value" style="color:#e2e4ed">{k}</div>
            </div>
            <div class="alpha-block">
              <div class="alpha-label">N of Cases</div>
              <div class="alpha-value" style="color:#e2e4ed">{n}</div>
            </div>
            <div class="alpha-interpret">{label} internal consistency</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if opt_item:
            rows_i = [[item, fmt(sub[item].mean()), fmt(sub[item].std()), n]
                      for item in items]
            render_table(["Item", "Mean", "Std. Dev.", "N"], rows_i,
                         caption="Item Statistics")

        if opt_iftd:
            rows_t = []
            for item in items:
                rest = sub.drop(columns=[item]).sum(axis=1)
                r_total = sub[item].corr(rest)
                sub_items = [i for i in items if i != item]
                sub_mat = sub[sub_items]
                k2 = len(sub_items)
                if k2 >= 2:
                    iv2 = sub_mat.var(ddof=1)
                    tv2 = sub_mat.sum(axis=1).var(ddof=1)
                    alpha_del = (k2 / (k2-1)) * (1 - iv2.sum() / tv2)
                else:
                    alpha_del = np.nan
                rows_t.append([item, fmt(r_total), fmt(alpha_del)])
            render_table(
                ["Item", "Corrected Item-Total r", "Cronbach's α if Item Deleted"],
                rows_t, caption="Item-Total Statistics"
            )

        if opt_scale:
            total = sub.sum(axis=1)
            render_table(
                ["Mean", "Variance", "Std. Deviation", "N of Items"],
                [[fmt(total.mean()), fmt(total.var(ddof=1)),
                  fmt(total.std(ddof=1)), k]],
                caption="Scale Statistics"
            )

        if opt_corr:
            corr_mat = sub.corr()
            fig = px.imshow(corr_mat, text_auto=".2f", aspect="auto",
                            color_continuous_scale="Blues", zmin=0, zmax=1,
                            template="plotly_dark")
            fig.update_layout(paper_bgcolor="#1a1d23", plot_bgcolor="#22262f",
                              margin=dict(t=20, b=20),
                              title="Inter-Item Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 12. MANN-WHITNEY U
# ══════════════════════════════════════════════════════════════
elif selected == "Mann-Whitney U":
    dv = st.selectbox("Test Variable", numeric_cols)
    iv = st.selectbox("Grouping Variable", all_cols)
    opt_desc = st.checkbox("Descriptive Statistics", True)
    opt_es   = st.checkbox("Effect Size (r = Z/√N)", True)

    if st.button("▶ Run Mann-Whitney U"):
        groups = df[iv].dropna().unique()
        if len(groups) < 2:
            st.error("Need at least 2 groups.")
        else:
            g1_label, g2_label = groups[0], groups[1]
            g1 = df[df[iv] == g1_label][dv].dropna()
            g2 = df[df[iv] == g2_label][dv].dropna()

            if opt_desc:
                render_table(
                    ["Group", "N", "Median", "Mean Rank"],
                    [[g1_label, len(g1), fmt(g1.median()),
                      fmt(g1.rank().mean())],
                     [g2_label, len(g2), fmt(g2.median()),
                      fmt(g2.rank().mean())]],
                    caption="Descriptive Statistics"
                )

            u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            n = len(g1) + len(g2)
            z = stats.norm.ppf(p_val / 2)
            r_es = abs(z) / np.sqrt(n)
            row = [fmt(u_stat, 0),
                   fmt(u_stat + len(g1)*(len(g1)+1)/2, 0),
                   fmt(z), fmt_p(p_val)]
            headers = ["Mann-Whitney U", "Wilcoxon W", "Z", "Asymp. Sig. (2-tailed)"]
            if opt_es:
                row.append(fmt(r_es))
                headers.append("r (effect size)")
            render_table(headers, [row], caption="Test Statistics")


# ══════════════════════════════════════════════════════════════
# 13. WILCOXON SIGNED-RANK
# ══════════════════════════════════════════════════════════════
elif selected == "Wilcoxon Signed-Rank":
    v1 = st.selectbox("Variable 1 (Time 1)", numeric_cols)
    v2 = st.selectbox("Variable 2 (Time 2)", [c for c in numeric_cols if c != v1])

    if st.button("▶ Run Wilcoxon"):
        pair_df = df[[v1, v2]].dropna()
        w_stat, p_val = stats.wilcoxon(pair_df[v1], pair_df[v2])
        n = len(pair_df)
        diffs = pair_df[v1] - pair_df[v2]
        neg = (diffs < 0).sum(); pos = (diffs > 0).sum(); ties = (diffs == 0).sum()

        render_table(
            ["", "N"],
            [[f"Negative Ranks ({v1} < {v2})", neg],
             [f"Positive Ranks ({v1} > {v2})", pos],
             ["Ties", ties],
             ["Total", n]],
            caption="Ranks"
        )
        z = stats.norm.ppf(p_val / 2)
        render_table(
            ["Wilcoxon W", "Z", "Asymp. Sig. (2-tailed)"],
            [[fmt(w_stat, 0), fmt(z), fmt_p(p_val)]],
            caption="Test Statistics"
        )


# ══════════════════════════════════════════════════════════════
# 14. KRUSKAL-WALLIS
# ══════════════════════════════════════════════════════════════
elif selected == "Kruskal-Wallis":
    dv     = st.selectbox("Test Variable", numeric_cols)
    factor = st.selectbox("Grouping Variable", all_cols)
    opt_post = st.checkbox("Post-Hoc: Dunn's Test (Bonferroni)", True)
    opt_es   = st.checkbox("Effect Size (η²H)", True)

    if st.button("▶ Run Kruskal-Wallis"):
        sub = df[[dv, factor]].dropna()
        groups_map = sub.groupby(factor)[dv].apply(list)
        group_names = groups_map.index.tolist()
        group_arrs  = [np.array(g) for g in groups_map]

        # Ranks table
        all_vals = np.concatenate(group_arrs)
        all_ranks = stats.rankdata(all_vals)
        offset = 0
        rows_r = []
        for name, arr in zip(group_names, group_arrs):
            ranks = all_ranks[offset:offset+len(arr)]
            rows_r.append([name, len(arr), fmt(ranks.mean())])
            offset += len(arr)
        render_table(["Group", "N", "Mean Rank"], rows_r, caption="Ranks")

        h_stat, p_val = stats.kruskal(*group_arrs)
        N = len(all_vals); k = len(group_arrs)
        eta2_h = (h_stat - k + 1) / (N - k)
        row = [fmt(h_stat), k - 1, fmt_p(p_val)]
        headers = ["Chi-Square (H)", "df", "Asymp. Sig."]
        if opt_es:
            row.append(fmt(eta2_h))
            headers.append("η²H (effect size)")
        render_table(headers, [row], caption="Test Statistics")

        if opt_post and k > 2:
            try:
                dunn = pg.pairwise_tests(data=sub, dv=dv, between=factor,
                                         parametric=False, padjust="bonf")
                rows_d = [[row["A"], row["B"],
                           fmt(float(row.get("U-val", row.get("W-val", np.nan))), 0),
                           fmt_p(float(row["p-corr"]))]
                          for _, row in dunn.iterrows()]
                render_table(["Group (I)", "Group (J)", "Statistic", "Adj. Sig. (Bonferroni)"],
                             rows_d, caption="Post-Hoc: Dunn's Test")
            except Exception as e:
                st.warning(f"Post-hoc failed: {e}")


# ══════════════════════════════════════════════════════════════
# 15. LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════
elif selected == "Logistic Regression":
    col1, col2 = st.columns([1.2, 1])
    dv  = col1.selectbox("Dependent Variable (Binary)", all_cols)
    ivs = col1.multiselect("Independent Variable(s)", numeric_cols)
    opt_or = col2.checkbox("Odds Ratios", True)
    opt_ci = col2.checkbox("95% CI for OR", True)
    opt_class = col2.checkbox("Classification Table", True)

    if st.button("▶ Run Logistic Regression") and ivs:
        sub = df[[dv] + ivs].dropna().copy()
        sub[dv] = pd.Categorical(sub[dv]).codes  # encode to 0/1
        X = sm.add_constant(sub[ivs])
        y = sub[dv]
        try:
            model = sm.Logit(y, X).fit(disp=False)
            params = model.params; bse = model.bse
            tvals = model.tvalues; pvals = model.pvalues
            ci = model.conf_int()
            ors = np.exp(params)
            ci_or = np.exp(ci)

            rows_c = []
            for name in ["const"] + ivs:
                row = [name if name != "const" else "(Constant)",
                       fmt(params[name]), fmt(bse[name]),
                       fmt(tvals[name]), fmt_p(pvals[name])]
                if opt_or:
                    row.append(fmt(ors[name]))
                if opt_ci:
                    row.append(f"[{fmt(ci_or.loc[name, 0])}, {fmt(ci_or.loc[name, 1])}]")
                rows_c.append(row)
            headers_c = ["", "B", "Std. Error", "Wald z", "Sig."]
            if opt_or: headers_c.append("Odds Ratio")
            if opt_ci: headers_c.append("95% CI for OR")
            render_table(headers_c, rows_c, caption="Variables in the Equation")

            render_table(
                ["", "Value"],
                [["−2 Log Likelihood", fmt(model.llnull * -2)],
                 ["Cox & Snell R²", fmt(1 - np.exp((model.llnull - model.llf) * 2 / len(y)))],
                 ["Nagelkerke R²", fmt((1 - np.exp((model.llnull - model.llf) * 2 / len(y))) /
                                       (1 - np.exp(model.llnull * 2 / len(y))))],
                 ["AIC", fmt(model.aic)], ["BIC", fmt(model.bic)]],
                caption="Model Summary"
            )

            if opt_class:
                pred = (model.fittedvalues >= 0.5).astype(int)
                ct = pd.crosstab(y, pred, rownames=["Observed"], colnames=["Predicted"])
                correct = np.diag(ct.values).sum() / ct.values.sum() * 100
                st.markdown('<div class="output-panel"><div class="panel-title">Classification Table</div><div class="panel-body">', unsafe_allow_html=True)
                st.dataframe(ct, use_container_width=True)
                st.markdown(f'<div class="stat-note"><span>Note.</span> Overall correct classification: {correct:.1f}%</div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Logistic regression error: {e}")


# ══════════════════════════════════════════════════════════════
# 16. ANCOVA
# ══════════════════════════════════════════════════════════════
elif selected == "ANCOVA":
    col1, col2 = st.columns([1.2, 1])
    dv      = col1.selectbox("Dependent Variable", numeric_cols)
    factor  = col1.selectbox("Fixed Factor", all_cols)
    covars  = col1.multiselect("Covariate(s)", [c for c in numeric_cols if c != dv])
    opt_eta = col2.checkbox("Effect Size (η²p)", True)
    opt_desc = col2.checkbox("Descriptives", True)

    if st.button("▶ Run ANCOVA") and covars:
        sub = df[[dv, factor] + covars].dropna()
        cov_str = " + ".join([f'Q("{c}")' for c in covars])
        formula = f'Q("{dv}") ~ C(Q("{factor}")) + {cov_str}'
        try:
            model = ols(formula, data=sub).fit()
            aov_table = sm.stats.anova_lm(model, typ=3)
            ss_total = aov_table["sum_sq"].sum()
            rows_a = []
            for src, row in aov_table.iterrows():
                eta2p = row["sum_sq"] / (row["sum_sq"] + aov_table.loc["Residual", "sum_sq"])
                r = [src, fmt(row["sum_sq"]), fmt(row["df"], 0),
                     fmt(row["sum_sq"] / max(row["df"], 1)),
                     fmt(row["F"]) if not np.isnan(row["F"]) else "—",
                     fmt_p(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else "—"]
                if opt_eta: r.append(fmt(eta2p) if src != "Residual" else "—")
                rows_a.append(r)
            headers_a = ["Source", "SS", "df", "MS", "F", "Sig."]
            if opt_eta: headers_a.append("η²p")
            render_table(headers_a, rows_a, caption="Tests of Between-Subjects Effects (ANCOVA)")
        except Exception as e:
            st.error(f"ANCOVA error: {e}")


# ══════════════════════════════════════════════════════════════
# 17. TWO-WAY ANOVA
# ══════════════════════════════════════════════════════════════
elif selected == "Two-Way ANOVA":
    col1, col2 = st.columns([1.2, 1])
    dv      = col1.selectbox("Dependent Variable", numeric_cols)
    factor1 = col1.selectbox("Factor A", all_cols)
    factor2 = col1.selectbox("Factor B", [c for c in all_cols if c != factor1])
    opt_eta  = col2.checkbox("Effect Size (η²p)", True)
    opt_inter = col2.checkbox("Interaction Plot", True)

    if st.button("▶ Run Two-Way ANOVA"):
        sub = df[[dv, factor1, factor2]].dropna()
        formula = f'Q("{dv}") ~ C(Q("{factor1}")) * C(Q("{factor2}"))'
        try:
            model = ols(formula, data=sub).fit()
            aov_table = sm.stats.anova_lm(model, typ=3)
            rows_a = []
            for src, row in aov_table.iterrows():
                eta2p = row["sum_sq"] / (row["sum_sq"] + aov_table.loc["Residual", "sum_sq"])
                r = [src, fmt(row["sum_sq"]), fmt(row["df"], 0),
                     fmt(row["sum_sq"] / max(row["df"], 1)),
                     fmt(row["F"]) if not np.isnan(row["F"]) else "—",
                     fmt_p(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else "—"]
                if opt_eta: r.append(fmt(eta2p) if src != "Residual" else "—")
                rows_a.append(r)
            headers_a = ["Source", "SS", "df", "MS", "F", "Sig."]
            if opt_eta: headers_a.append("η²p")
            render_table(headers_a, rows_a, caption="Between-Subjects Effects")

            if opt_inter:
                means = sub.groupby([factor1, factor2])[dv].mean().reset_index()
                fig = px.line(means, x=factor2, y=dv, color=factor1, markers=True,
                              template="plotly_dark",
                              title=f"Interaction: {factor1} × {factor2}",
                              color_discrete_sequence=px.colors.qualitative.Bold)
                fig.update_layout(paper_bgcolor="#1a1d23", plot_bgcolor="#22262f",
                                  margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Two-Way ANOVA error: {e}")


# ══════════════════════════════════════════════════════════════
# 18. EFA
# ══════════════════════════════════════════════════════════════
elif selected == "Exploratory Factor Analysis":
    items = st.multiselect("Variables for EFA", numeric_cols)
    n_factors = st.slider("Number of Factors", 1, 10, 2)
    rotation  = st.radio("Rotation", ["varimax", "oblimin", "none"])
    opt_loadings = st.checkbox("Factor Loadings", True)
    opt_comm     = st.checkbox("Communalities", True)
    opt_scree    = st.checkbox("Scree Plot", True)

    if st.button("▶ Run EFA") and len(items) >= 3:
        try:
            sub = df[items].dropna()
            corr_mat = sub.corr()

            if opt_scree:
                eigenvalues = np.linalg.eigvalsh(corr_mat)[::-1]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(1, len(eigenvalues)+1)),
                                          y=eigenvalues, mode="lines+markers",
                                          marker_color="#4f9cf9"))
                fig.add_hline(y=1, line_dash="dash", line_color="#f45b5b",
                              annotation_text="Kaiser criterion (λ=1)")
                fig.update_layout(title="Scree Plot", xaxis_title="Factor",
                                  yaxis_title="Eigenvalue", template="plotly_dark",
                                  paper_bgcolor="#1a1d23", plot_bgcolor="#22262f",
                                  margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            from sklearn.decomposition import FactorAnalysis
            fa = FactorAnalysis(n_components=n_factors, rotation=rotation if rotation != "none" else None)
            fa.fit(sub)
            loadings = pd.DataFrame(fa.components_.T, index=items,
                                     columns=[f"F{i+1}" for i in range(n_factors)])
            comm = (loadings**2).sum(axis=1)

            if opt_loadings:
                rows_l = [[item] + [fmt(loadings.loc[item, f"F{i+1}"]) for i in range(n_factors)]
                          for item in items]
                rows_l.append(["<b>Eigenvalue</b>"] +
                               [fmt((loadings[f"F{i+1}"]**2).sum()) for i in range(n_factors)])
                rows_l.append(["<b>% Variance</b>"] +
                               [fmt((loadings[f"F{i+1}"]**2).sum()/len(items)*100, 1)+"%" for i in range(n_factors)])
                render_table(["Item"] + [f"F{i+1}" for i in range(n_factors)],
                             rows_l, caption=f"Factor Loadings ({rotation} rotation)")

            if opt_comm:
                render_table(["Item", "Communality"],
                             [[item, fmt(comm[item])] for item in items],
                             caption="Communalities")
        except Exception as e:
            st.error(f"EFA error: {e}. Make sure scikit-learn is installed.")


# ══════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    '<div style="text-align:center;font-size:10px;color:#1e2230;padding:20px 0 8px;'
    'letter-spacing:.5px;text-transform:uppercase;font-weight:700">'
    'StatLab Pro &nbsp;·&nbsp; scipy &nbsp;·&nbsp; statsmodels &nbsp;·&nbsp; pingouin &nbsp;·&nbsp; plotly'
    '</div>',
    unsafe_allow_html=True
)import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
import google.generativeai as genai

def render_stats_lab(df: pd.DataFrame):
    st.header("🔬 Statistic Analyzer & AI Research Partner")
    
    # ניקוי שמות עמודות וזיהוי סוגים
    df.columns = df.columns.str.strip()
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # משתנים קטגוריאליים (כמו Major, Class, Gender)
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # תפריט ניתוחים
    analysis_type = st.sidebar.radio("בחר סוג ניתוח", [
        "📊 Descriptives",
        "📈 ANOVA (Repeated Measures)",
        "🎯 Simple Main Effects",
        "🧪 T-Tests",
        "🛡️ Reliability (Cronbach's Alpha)",
        "🔗 Correlations Matrix",
        "🔲 Frequencies (Chi-Square)"
    ])

    st.divider()

    # אתחול זיכרון AI
    if 'global_context' not in st.session_state:
        st.session_state['global_context'] = "No analysis performed yet."

    # --- 1. DESCRIPTIVES (Major זמין כאן) ---
    if analysis_type == "📊 Descriptives":
        vars_d = st.multiselect("משתנים לניתוח (מספריים):", numeric_cols)
        group_d = st.selectbox("פלח לפי (כאן Major אמור להיות):", ["ללא"] + all_cols)
        if vars_d:
            if group_d == "ללא":
                res = df[vars_d].describe().T
            else:
                res = df.groupby(group_d)[vars_d].describe().stack(level=0)
            st.table(res.style.format("{:.2f}"))
            st.session_state['global_context'] = f"Descriptives for {vars_d} grouped by {group_d}."

    # --- 2. ANOVA (Major זמין כאן) ---
    elif analysis_type == "📈 ANOVA (Repeated Measures)":
        levels = st.multiselect("רמות זמן (מספרי):", numeric_cols)
        between = st.selectbox("גורם בין-נבדקי (Major):", all_cols)
        if st.button("הרץ ANOVA"):
            if len(levels) > 1:
                tdf = df[levels + [between]].dropna().copy()
                tdf['ID'] = range(len(tdf))
                long_df = pd.melt(tdf, id_vars=['ID', between], value_vars=levels, var_name='Time', value_name='Score')
                long_df['Time'] = pd.Categorical(long_df['Time'], categories=levels, ordered=True)
                model = ols(f'Score ~ C(Time) * C(Q("{between}"))', data=long_df).fit()
                res = sm.stats.anova_lm(model, typ=3)
                st.table(res.style.format("{:.3f}"))
                st.session_state['global_context'] = f"ANOVA on {levels} by {between}."

    # --- 3. SIMPLE MAIN EFFECTS (כאן Major קריטי) ---
    elif analysis_type == "🎯 Simple Main Effects":
        target_v = st.selectbox("בחר רמת זמן לבדיקה:", numeric_cols)
        group_v = st.selectbox("השוואה בין קבוצות (Major):", all_cols)
        if st.button("הרץ Simple Main Effect"):
            formula = f'Q("{target_v}") ~ C(Q("{group_v}"))'
            model = ols(formula, data=df).fit()
            res = sm.stats.anova_lm(model, typ=3)
            st.table(res.style.format("{:.3f}").highlight_between(subset=['PR(>F)'], left=0, right=0.05, color='#ffcccc'))
            st.session_state['global_context'] = f"Simple Main Effect of {group_v} on {target_v}."

    # --- 4. T-TESTS (Major זמין ב-Independent) ---
    elif analysis_type == "🧪 T-Tests":
        t_type = st.radio("סוג מבחן", ["Independent (השוואת Major)", "Paired (לפני/אחרי)"])
        if t_type == "Independent (השוואת Major)":
            dv = st.selectbox("משתנה תלוי:", numeric_cols)
            iv = st.selectbox("משתנה קבוצה (Major):", all_cols)
            if st.button("בצע T-Test"):
                groups = df[iv].unique()
                if len(groups) >= 2:
                    g1 = df[df[iv] == groups[0]][dv].dropna()
                    g2 = df[df[iv] == groups[1]][dv].dropna()
                    t_stat, p = stats.ttest_ind(g1, g2)
                    st.metric("p-value", f"{p:.4f}")
                    st.session_state['global_context'] = f"T-test on {dv} by {iv} ({groups[0]} vs {groups[1]})."
        else:
            v1 = st.selectbox("זמן 1:", numeric_cols)
            v2 = st.selectbox("זמן 2:", numeric_cols)
            if st.button("בצע Paired T-Test"):
                t_stat, p = stats.ttest_rel(df[v1].dropna(), df[v2].dropna())
                st.metric("p-value", f"{p:.4f}")

    # --- 5. RELIABILITY (מספרי בלבד) ---
    elif analysis_type == "🛡️ Reliability (Cronbach's Alpha)":
        items = st.multiselect("בחר פריטים לשאלון:", numeric_cols)
        if st.button("חשב אלפא"):
            idat = df[items].dropna()
            k = len(items)
            alpha = (k/(k-1)) * (1 - idat.var().sum() / idat.sum(axis=1).var())
            st.metric("Cronbach's Alpha (α)", f"{alpha:.3f}")

    # --- AI RECOMMENDATIONS (Gemini 2.0 Flash) ---
    st.divider()
    if st.button("💡 קבל הצעות להמשך הניתוח"):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = f"Context: {st.session_state['global_context']}\nAll columns: {all_cols}\nProvide research advice in Hebrew."
            response = model.generate_content(prompt)
            st.info(response.text)
        except Exception as e: st.error(e)
