"""
The Performance Ledger — Student Performance Dashboard
Run with: streamlit run dashboard_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="The Performance Ledger",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# First-boot safeguard — on a fresh deploy (e.g. Streamlit Community Cloud),
# model_assets.pkl / processed_students.csv won't exist yet because they're
# build artifacts, not committed to the repo. Train once, automatically,
# instead of crashing with a FileNotFoundError.
# ----------------------------------------------------------------------------
if not (os.path.exists("model_assets.pkl") and os.path.exists("processed_students.csv")):
    with st.spinner("First-time setup: training the model (this only happens once)…"):
        import train_model
        train_model.main()

# ----------------------------------------------------------------------------
# Theme system — Light / Dark, toggle lives in the sidebar (see below).
# Session state is read here (before any CSS is built) so the whole script
# renders consistently on every rerun, including the very first load.
# ----------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

THEMES = {
    "Light": {
        "bg": "#FFFFFF", "surface": "#FFFFFF", "surface_2": "#F4F6FA",
        "text_head": "#1F2A2A", "text_body": "#55606A", "text_muted": "#8A93A0",
        "sage": "#2FB985", "blue": "#5B8DEF", "lavender": "#9B7BF0", "coral": "#F2665C",
        "warning": "#FFA45C", "success": "#2FB985", "info": "#7FB3FF",
        "card_border": "rgba(31,42,42,0.06)", "grid": "rgba(31,42,42,0.08)",
        "chart_line": "rgba(31,42,42,0.15)", "hover_bg": "#FFFFFF",
        "shadow": "0 10px 28px rgba(31,42,42,0.08)",
        "shadow_hover": "0 18px 42px rgba(47,185,133,0.18)",
        "hero_gradient": "linear-gradient(135deg, rgba(155,123,240,0.14), rgba(47,185,133,0.12) 55%, rgba(91,141,239,0.10))",
        "scrollbar": "rgba(31,42,42,0.15)",
    },
    "Dark": {
        "bg": "#0F1420", "surface": "#171E2E", "surface_2": "#1B2233",
        "text_head": "#F3F6F8", "text_body": "#B7C1D1", "text_muted": "#8992A3",
        "sage": "#3EDB99", "blue": "#6FA0FF", "lavender": "#B48CFF", "coral": "#FF7A6E",
        "warning": "#FFB26B", "success": "#3EDB99", "info": "#7FB3FF",
        "card_border": "rgba(255,255,255,0.08)", "grid": "rgba(255,255,255,0.08)",
        "chart_line": "rgba(255,255,255,0.16)", "hover_bg": "#1B2233",
        "shadow": "0 10px 28px rgba(0,0,0,0.45)",
        "shadow_hover": "0 18px 46px rgba(62,219,153,0.28)",
        "hero_gradient": "linear-gradient(135deg, rgba(180,140,255,0.20), rgba(62,219,153,0.16) 55%, rgba(111,160,255,0.16))",
        "scrollbar": "rgba(255,255,255,0.15)",
    },
}
T = THEMES["Dark" if st.session_state.dark_mode else "Light"]

# ----------------------------------------------------------------------------
# Design tokens (variable names kept stable so every chart/component call
# below keeps working unmodified — only the resolved values change with theme)
# ----------------------------------------------------------------------------
INK       = T["bg"]          # background primary
SURFACE   = T["surface"]     # card / panel surface
SURFACE_2 = T["surface_2"]   # sidebar bg, table stripe, hover tint
PARCHMENT = T["text_head"]   # primary heading text
SLATE     = T["text_body"]   # secondary / body text
GOLD      = T["sage"]        # accent — Emerald Sage
GOLD_SOFT = f"rgba({int(T['sage'][1:3],16)},{int(T['sage'][3:5],16)},{int(T['sage'][5:7],16)},0.18)"
SAGE      = T["sage"]        # High performance
BLUE      = T["blue"]        # Average performance
RUST      = T["coral"]       # Low performance

# extra tokens used for cards, contrast text and chart accents
CARD_BORDER  = T["card_border"]
SHADOW       = T["shadow"]
SHADOW_HOVER = T["shadow_hover"]
TEXT_MUTED   = T["text_muted"]
LAVENDER     = T["lavender"]  # highlight
SUCCESS      = T["success"]
WARNING      = T["warning"]
INFO         = T["info"]
WHITE        = "#FFFFFF"
HERO_GRADIENT = T["hero_gradient"]
GRID         = T["grid"]         # plotly gridline color
CHART_LINE   = T["chart_line"]   # plotly axis line color
HOVER_BG     = T["hover_bg"]     # plotly tooltip background
SCROLLBAR    = T["scrollbar"]

PERF_COLORS = {"High": SAGE, "Average": BLUE, "Low": RUST}


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def darken(hex_color, amount=0.15):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r, g, b = [max(0, int(c * (1 - amount))) for c in (r, g, b)]
    return f"#{r:02x}{g:02x}{b:02x}"


GOLD_HOVER = darken(GOLD, 0.15)

# ----------------------------------------------------------------------------
# Global CSS — Poppins throughout, pastel palette via CSS custom properties
# ----------------------------------------------------------------------------
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* ---------- CSS variables (theme) ---------- */
    :root {{
        --bg: {INK};
        --surface: {SURFACE};
        --surface-2: {SURFACE_2};
        --text-heading: {PARCHMENT};
        --text-sub: #4B5D5A;
        --text-body: {SLATE};
        --text-muted: {TEXT_MUTED};
        --text-placeholder: #9AA3AA;
        --accent-sage: {SAGE};
        --accent-dusty-blue: {BLUE};
        --accent-lavender: {LAVENDER};
        --success: {SUCCESS};
        --warning: {WARNING};
        --danger: {RUST};
        --info: {INFO};
        --card-border: {CARD_BORDER};
        --shadow-card: {SHADOW};
        --shadow-card-hover: {SHADOW_HOVER};
        --radius-btn: 12px;
        --radius-input: 12px;
        --radius-card: 18px;
        --radius-table: 14px;
    }}

    /* ---------- Typography ---------- */
    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif !important;
        color: var(--text-body);
    }}
    .stApp {{
        background: var(--bg);
        color: var(--text-body);
        transition: background-color 0.4s ease, color 0.4s ease;
    }}
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        transition: background-color 0.4s ease;
        background: var(--bg) !important;
    }}
    /* Dark/Light toggle switch accenting (best-effort across Streamlit versions) */
    div[data-baseweb="switch"] div[aria-checked="true"] {{
        background: var(--accent-sage) !important;
    }}
    h1, h2, h3 {{
        font-family: 'Poppins', sans-serif !important;
        color: var(--text-heading) !important;
        font-weight: 700 !important;
        letter-spacing: 0.1px;
        line-height: 1.3;
    }}
    p, span, label, .stMarkdown {{
        line-height: 1.6;
    }}
    .eyebrow {{
        font-family: 'Poppins', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent-sage);
        margin-bottom: 0.3rem;
    }}
    .ledger-rule {{
        border: none;
        border-top: 1px solid var(--card-border);
        margin: 0.8rem 0 1.8rem 0;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: var(--surface-2);
        border-right: 1px solid var(--card-border);
    }}
    section[data-testid="stSidebar"] * {{
        color: var(--text-heading) !important;
    }}
    section[data-testid="stSidebar"] label[data-baseweb="radio"] {{
        padding: 0.5rem 0.7rem;
        border-radius: 10px;
        transition: background 0.2s ease;
        margin-bottom: 2px;
    }}
    section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {{
        background: {hex_to_rgba(GOLD, 0.15)};
    }}
    section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {{
        background: var(--accent-sage);
    }}
    section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) * {{
        color: #FFFFFF !important;
    }}

    /* ---------- Hero banner ---------- */
    .hero-wrap {{
        background: {HERO_GRADIENT};
        border-radius: 24px;
        padding: 2rem 2.2rem 1.6rem 2.2rem;
        margin-bottom: 1.4rem;
        border: 1px solid var(--card-border);
    }}

    /* ---------- Cards ---------- */
    .ledger-card {{
        background: var(--surface);
        border: 1px solid var(--card-border);
        border-radius: var(--radius-card);
        box-shadow: var(--shadow-card);
        padding: 1.3rem 1.5rem;
        height: 100%;
        position: relative;
        overflow: hidden;
        transition: box-shadow 0.3s ease, transform 0.3s ease;
        animation: fadeInUp 0.4s ease;
    }}
    .ledger-card::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-sage), var(--accent-dusty-blue), var(--accent-lavender));
    }}
    .ledger-card:hover {{
        box-shadow: var(--shadow-card-hover);
        transform: translateY(-3px);
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .ledger-metric-label {{
        font-family: 'Poppins', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
    }}
    .ledger-metric-value {{
        font-family: 'Poppins', sans-serif;
        font-size: 2.0rem;
        font-weight: 700;
        color: var(--text-heading);
        margin-top: 0.2rem;
    }}
    .ledger-metric-sub {{
        font-family: 'Poppins', sans-serif;
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--accent-sage);
        margin-top: 0.15rem;
    }}

    /* ---------- Pills / badges ---------- */
    .pill {{
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 999px;
        font-family: 'Poppins', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.03em;
        font-weight: 600;
    }}

    /* ---------- Forms / inputs ---------- */
    div[data-testid="stForm"] {{
        background: var(--surface);
        border: 1px solid var(--card-border);
        border-radius: var(--radius-card);
        box-shadow: var(--shadow-card);
        padding: 1.6rem;
    }}
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
        background: var(--surface) !important;
        border-radius: var(--radius-input) !important;
        border: 1px solid var(--card-border) !important;
        color: var(--text-body) !important;
    }}
    .stTextInput input:focus, .stNumberInput input:focus {{
        border: 1px solid var(--accent-sage) !important;
        box-shadow: 0 0 0 3px {hex_to_rgba(GOLD, 0.25)} !important;
    }}
    ::placeholder {{
        color: var(--text-placeholder) !important;
    }}

    /* Widget labels (slider/selectbox/radio/text-input captions) — Streamlit
       leaves these unstyled by default, so without an explicit override they
       fall back to the browser/OS color-scheme preference instead of our
       theme, which is invisible in Light mode when that preference is Dark. */
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
    .stSlider label p, .stRadio label p, .stSelectbox label p {{
        color: var(--text-body) !important;
    }}
    [data-testid="stCaptionContainer"] {{
        color: var(--text-muted) !important;
    }}

    /* Sliders: track, handle, current-value bubble, min/max tick labels */
    .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {{
        color: var(--text-muted) !important;
    }}
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: var(--accent-sage) !important;
    }}
    .stSlider [data-testid="stThumbValue"], .stSlider div[data-baseweb="slider"] + div {{
        color: var(--text-heading, var(--accent-sage)) !important;
    }}

    /* Radio buttons (main content, not just sidebar) */
    div[role="radiogroup"] label p, div[role="radiogroup"] label span {{
        color: var(--text-body) !important;
    }}

    /* Selectbox: closed control + dropdown icon */
    div[data-baseweb="select"] {{
        background: var(--surface) !important;
    }}
    div[data-baseweb="select"] * {{
        color: var(--text-body) !important;
        fill: var(--text-body) !important;
    }}
    /* Selectbox open dropdown menu (rendered in a portal) */
    ul[data-testid="stSelectboxVirtualDropdown"], div[data-baseweb="popover"] div[data-baseweb="menu"] {{
        background: var(--surface) !important;
        border: 1px solid var(--card-border) !important;
    }}
    ul[data-testid="stSelectboxVirtualDropdown"] li, div[data-baseweb="popover"] li {{
        color: var(--text-body) !important;
        background: var(--surface) !important;
    }}
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover, div[data-baseweb="popover"] li:hover {{
        background: {hex_to_rgba(GOLD, 0.12)} !important;
    }}

    /* ---------- Buttons ---------- */
    .stButton>button, .stFormSubmitButton>button {{
        background: var(--accent-sage);
        color: #FFFFFF;
        border: none;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        letter-spacing: 0.02em;
        border-radius: var(--radius-btn);
        padding: 0.55rem 1.4rem;
        transition: background 0.3s ease, box-shadow 0.3s ease;
    }}
    .stButton>button:hover, .stFormSubmitButton>button:hover {{
        background: {GOLD_HOVER};
        color: #FFFFFF;
        box-shadow: var(--shadow-card-hover);
    }}

    /* ---------- Tabs ---------- */
    div[data-baseweb="tab-list"] {{
        border-bottom: 1px solid var(--card-border);
        gap: 4px;
    }}
    button[data-baseweb="tab"] {{
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        letter-spacing: 0.02em;
        color: var(--text-muted);
        border-radius: 10px 10px 0 0;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--accent-sage) !important;
    }}

    /* ---------- Tables ---------- */
    .stDataFrame {{
        border: 1px solid var(--card-border);
        border-radius: var(--radius-table);
        box-shadow: var(--shadow-card);
        overflow: hidden;
    }}

    /* ---------- Misc ---------- */
    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    input[type="radio"], input[type="checkbox"] {{
        accent-color: var(--accent-sage);
    }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: var(--card-border); border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)


def plotly_theme(fig, height=420):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, sans-serif", color=PARCHMENT, size=13),
        title_font=dict(family="Poppins, sans-serif", color=PARCHMENT, size=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PARCHMENT, size=12)),
        hoverlabel=dict(font=dict(color=PARCHMENT, size=12), bgcolor=HOVER_BG),
        height=height,
        margin=dict(l=40, r=30, t=60, b=40),
    )
    fig.update_xaxes(
        gridcolor=GRID, zerolinecolor=CHART_LINE,
        tickfont=dict(color=PARCHMENT, size=12),
        title_font=dict(color=SLATE, size=13),
        linecolor=CHART_LINE,
    )
    fig.update_yaxes(
        gridcolor=GRID, zerolinecolor=CHART_LINE,
        tickfont=dict(color=PARCHMENT, size=12),
        title_font=dict(color=SLATE, size=13),
        linecolor=CHART_LINE,
    )
    return fig


def section_header(eyebrow, title):
    st.markdown(f"""
        <div class="eyebrow">{eyebrow}</div>
        <h2 style="margin-top:0;">{title}</h2>
        <hr class="ledger-rule">
    """, unsafe_allow_html=True)


def ledger_metric(label, value, sub=None):
    sub_html = f'<div class="ledger-metric-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
        <div class="ledger-card">
            <div class="ledger-metric-label">{label}</div>
            <div class="ledger-metric-value">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)


def perf_pill(label):
    color = PERF_COLORS.get(label, SLATE)
    return f'<span class="pill" style="background:{color}22; color:{color}; border:1px solid {color}66;">{label}</span>'


# ----------------------------------------------------------------------------
# Data & model loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("processed_students.csv")


@st.cache_resource
def load_model():
    return joblib.load("model_assets.pkl")


df = load_data()
assets = load_model()
FEATURE_COLS = assets["feature_cols"]
SUBJECTS = assets["subjects"]
CLF = assets["classifier"]
REG = assets["regressor"]
CLF_ACC = assets["clf_accuracy"]

LABEL_MAPS = {
    "Resources": {0: "Low", 1: "Medium", 2: "High"},
    "Motivation": {0: "Low", 1: "Medium", 2: "High"},
    "StressLevel": {0: "Low", 1: "Medium", 2: "High"},
    "Extracurricular": {0: "No", 1: "Yes"},
    "Internet": {0: "No", 1: "Yes"},
    "Discussions": {0: "No", 1: "Yes"},
    "EduTech": {0: "No", 1: "Yes"},
}

# ----------------------------------------------------------------------------
# Recommendation engine
# For each behavioral factor, define a "healthy" threshold. A student's gap
# against that threshold is weighted by how important the factor actually is
# in the trained Random Forest (feature_importances_), so recommendations
# surface the highest-*impact* fixes first, not just the biggest raw gaps.
# ----------------------------------------------------------------------------
FEATURE_IMPORTANCE = dict(zip(FEATURE_COLS, CLF.feature_importances_))

RECOMMENDATION_RULES = {
    "AssignmentCompletion": {"direction": "min", "threshold": 70, "unit": "%",
        "tip": "Complete more assignments — this is the single strongest predictor of exam performance in this model."},
    "Attendance": {"direction": "min", "threshold": 75, "unit": "%",
        "tip": "Improve class attendance — attendance below 75% strongly correlates with lower exam scores."},
    "StudyHours": {"direction": "min", "threshold": 15, "unit": "hrs/week",
        "tip": "Increase weekly study time — students studying 15+ hrs/week average noticeably higher scores."},
    "OnlineCourses": {"direction": "min", "threshold": 3, "unit": "courses",
        "tip": "Enroll in a few supplementary online courses to reinforce weaker topics."},
    "Motivation": {"direction": "min", "threshold": 1, "unit": "level (0=Low,1=Med,2=High)",
        "tip": "Build motivation through goal-setting, study groups, or small milestone rewards."},
    "StressLevel": {"direction": "max", "threshold": 1, "unit": "level (0=Low,1=Med,2=High)",
        "tip": "Manage stress levels — high stress is associated with lower exam scores; consider time-management support."},
    "Resources": {"direction": "min", "threshold": 1, "unit": "level (0=Low,1=Med,2=High)",
        "tip": "Seek better access to learning resources — textbooks, tutoring, or study materials."},
    "Discussions": {"direction": "eq", "target": 1, "unit": "",
        "tip": "Start participating in class discussions — participants score higher on average."},
    "EduTech": {"direction": "eq", "target": 1, "unit": "",
        "tip": "Start using educational technology tools/apps to supplement classroom learning."},
    "Internet": {"direction": "eq", "target": 1, "unit": "",
        "tip": "Reliable internet access makes online resources and courses easier to use consistently."},
}


def get_recommendations(row, top_n=3):
    """Return up to top_n prioritized recommendations for a single student row."""
    gaps = []
    for feat, rule in RECOMMENDATION_RULES.items():
        val = row[feat]
        importance = FEATURE_IMPORTANCE.get(feat, 0)
        triggered, severity = False, 0.0
        if rule["direction"] == "min" and val < rule["threshold"]:
            triggered = True
            severity = (rule["threshold"] - val) / max(rule["threshold"], 1)
        elif rule["direction"] == "max" and val > rule["threshold"]:
            triggered = True
            severity = (val - rule["threshold"]) / max(val, 1)
        elif rule["direction"] == "eq" and val != rule["target"]:
            triggered = True
            severity = 1.0
        if triggered:
            gaps.append({
                "feature": feat, "tip": rule["tip"], "value": val,
                "unit": rule.get("unit", ""),
                "score": importance * (0.4 + severity),
            })
    gaps.sort(key=lambda g: g["score"], reverse=True)
    return gaps[:top_n]


def get_strengths(row, top_n=2):
    """Return the student's top well-performing factors (inverse of gaps)."""
    strengths = []
    for feat, rule in RECOMMENDATION_RULES.items():
        val = row[feat]
        importance = FEATURE_IMPORTANCE.get(feat, 0)
        is_strong = False
        if rule["direction"] == "min" and val >= rule["threshold"]:
            is_strong = True
        elif rule["direction"] == "max" and val <= rule["threshold"]:
            is_strong = True
        elif rule["direction"] == "eq" and val == rule["target"]:
            is_strong = True
        if is_strong:
            strengths.append({"feature": feat, "value": val, "score": importance})
    strengths.sort(key=lambda s: s["score"], reverse=True)
    return strengths[:top_n]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    top_l, top_r = st.columns([3, 1.3])
    with top_l:
        st.markdown(f"""
            <div class="eyebrow">Est. Module 4</div>
            <h2 style="margin-top:0; margin-bottom:0;">The Performance<br>Ledger</h2>
        """, unsafe_allow_html=True)
    with top_r:
        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
        st.toggle("🌙", key="dark_mode", help="Toggle dark / light mode")
    st.markdown("<hr class='ledger-rule'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Overview", "Predicted Scores", "Student Comparison", "Subject-wise Analysis", "Performance Trends", "Recommendations"],
        label_visibility="collapsed",
    )

    st.markdown("<hr class='ledger-rule'>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="ledger-metric-label">Model</div>
        <div style="font-family:'Poppins',sans-serif; font-size:0.85rem; color:{PARCHMENT};">
            Random Forest Classifier<br>
            <span style="color:{GOLD}; font-weight:600;">{CLF_ACC:.1%}</span> test accuracy
        </div>
    """, unsafe_allow_html=True)
    st.caption("Subject-wise scores are simulated from ExamScore for demonstration — swap in real per-subject data when available.")

# ============================================================================
# OVERVIEW
# ============================================================================
if page == "Overview":
    st.markdown(f"""
        <div class="hero-wrap">
            <div class="eyebrow">Student Performance Prediction System</div>
            <h1 style="margin-top:0;">Performance Ledger</h1>
            <p style="color:{SLATE}; max-width:640px; margin-bottom:0;">
                A record of predicted outcomes, drawn from attendance, study habits,
                participation and engagement across {len(df):,} students.
            </p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ledger_metric("Total Students", f"{len(df):,}")
    with c2:
        ledger_metric("Avg. Predicted Score", f"{df['PredictedScore'].mean():.1f}", "out of 100")
    with c3:
        high_pct = (df['PredictedPerformance'] == 'High').mean()
        ledger_metric("High Performers", f"{high_pct:.1%}", "of student body")
    with c4:
        ledger_metric("Model Accuracy", f"{CLF_ACC:.1%}", "Random Forest, held-out test set")

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        section_header("Distribution", "Predicted Performance Across the Cohort")
        counts = df['PredictedPerformance'].value_counts().reindex(['Low', 'Average', 'High'])
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[PERF_COLORS[c] for c in counts.index],
            text=counts.values, textposition="outside",
        ))
        fig.update_layout(title="Number of Students by Predicted Category")
        st.plotly_chart(plotly_theme(fig, 360), use_container_width=True)

    with right:
        section_header("Drivers", "What Predicts Performance")
        imp = pd.Series(CLF.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(7)
        fig = go.Figure(go.Bar(
            x=imp.values, y=imp.index, orientation='h',
            marker_color=GOLD,
        ))
        fig.update_layout(title="Top Feature Importances (Random Forest)")
        st.plotly_chart(plotly_theme(fig, 360), use_container_width=True)

    section_header("3D View", "The Performance Landscape")
    st.caption("Rotate, zoom, and pan — each point is a student, positioned by Study Hours, Attendance, and Assignment Completion, colored by predicted performance.")

    sample = df.sample(min(1500, len(df)), random_state=42)
    fig3d = go.Figure()
    for cat in ['Low', 'Average', 'High']:
        sub = sample[sample['PredictedPerformance'] == cat]
        fig3d.add_trace(go.Scatter3d(
            x=sub['StudyHours'], y=sub['Attendance'], z=sub['AssignmentCompletion'],
            mode='markers',
            name=cat,
            marker=dict(size=4, color=PERF_COLORS[cat], opacity=0.75,
                        line=dict(width=0)),
            text=sub['StudentID'],
            hovertemplate="<b>%{text}</b><br>Study Hours: %{x}<br>Attendance: %{y}<br>Assignment: %{z}<extra></extra>",
        ))
    fig3d.update_layout(
        scene=dict(
            xaxis=dict(title="Study Hours", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID,
                       tickfont=dict(color=PARCHMENT, size=11), title_font=dict(color=SLATE, size=12)),
            yaxis=dict(title="Attendance %", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID,
                       tickfont=dict(color=PARCHMENT, size=11), title_font=dict(color=SLATE, size=12)),
            zaxis=dict(title="Assignment %", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID,
                       tickfont=dict(color=PARCHMENT, size=11), title_font=dict(color=SLATE, size=12)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(plotly_theme(fig3d, 560), use_container_width=True)

    section_header("Sample of Records", "Recent Entries in the Ledger")
    display_cols = ['StudentID', 'StudyHours', 'Attendance', 'AssignmentCompletion',
                     'ExamScore', 'PredictedScore', 'PredictedPerformance']
    st.dataframe(df[display_cols].head(25), use_container_width=True, hide_index=True)


# ============================================================================
# PREDICTED SCORES
# ============================================================================
elif page == "Predicted Scores":
    section_header("Predict", "Score a New Student")

    st.write(f"Enter a student's profile to predict their exam score and performance category.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            study_hours = st.slider("Study Hours / week", 0, 40, 18)
            attendance = st.slider("Attendance (%)", 0, 100, 80)
            assignment = st.slider("Assignment Completion (%)", 0, 100, 75)
            online_courses = st.slider("Online Courses Taken", 0, 20, 5)
            age = st.slider("Age", 15, 30, 20)
        with c2:
            resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1)
            motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
            stress = st.selectbox("Stress Level", ["Low", "Medium", "High"], index=1)
            learning_style = st.selectbox("Learning Style", [0, 1, 2, 3], index=0)
        with c3:
            extracurricular = st.radio("Extracurricular Activity", ["No", "Yes"], horizontal=True)
            internet = st.radio("Internet Access", ["No", "Yes"], horizontal=True)
            discussions = st.radio("Participates in Discussions", ["No", "Yes"], horizontal=True)
            edutech = st.radio("Uses EduTech Tools", ["No", "Yes"], horizontal=True)
            gender = st.radio("Gender Group", ["Group A", "Group B"], horizontal=True)

        submitted = st.form_submit_button("Predict Performance")

    if submitted:
        inv_map = {"Low": 0, "Medium": 1, "High": 2, "No": 0, "Yes": 1,
                   "Group A": 0, "Group B": 1}
        row = pd.DataFrame([{
            "StudyHours": study_hours,
            "Attendance": attendance,
            "Resources": inv_map[resources],
            "Extracurricular": inv_map[extracurricular],
            "Motivation": inv_map[motivation],
            "Internet": inv_map[internet],
            "Gender": inv_map[gender],
            "Age": age,
            "LearningStyle": learning_style,
            "OnlineCourses": online_courses,
            "Discussions": inv_map[discussions],
            "AssignmentCompletion": assignment,
            "EduTech": inv_map[edutech],
            "StressLevel": inv_map[stress],
        }])[FEATURE_COLS]

        pred_score = REG.predict(row)[0]
        pred_class = CLF.predict(row)[0]
        proba = dict(zip(CLF.classes_, CLF.predict_proba(row)[0]))

        st.markdown("<hr class='ledger-rule'>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1.4])
        with r1:
            st.markdown(f"""
                <div class="ledger-card">
                    <div class="ledger-metric-label">Predicted Exam Score</div>
                    <div class="ledger-metric-value">{pred_score:.1f}</div>
                    <div style="margin-top:0.6rem;">{perf_pill(pred_class)}</div>
                </div>
            """, unsafe_allow_html=True)
        with r2:
            fig = go.Figure(go.Bar(
                x=list(proba.values()), y=list(proba.keys()), orientation='h',
                marker_color=[PERF_COLORS[k] for k in proba.keys()],
                text=[f"{v:.1%}" for v in proba.values()], textposition="outside",
            ))
            fig.update_layout(title="Prediction Confidence by Category", xaxis_range=[0, 1])
            st.plotly_chart(plotly_theme(fig, 260), use_container_width=True)

    st.markdown("<hr class='ledger-rule'>", unsafe_allow_html=True)
    section_header("Browse", "Predicted Scores for Existing Students")
    search_id = st.text_input("Filter by Student ID (e.g. STU00001)", "")
    filtered = df if not search_id else df[df['StudentID'].str.contains(search_id, case=False)]
    st.dataframe(
        filtered[['StudentID', 'ExamScore', 'PredictedScore', 'PredictedPerformance']].head(200),
        use_container_width=True, hide_index=True
    )


# ============================================================================
# STUDENT COMPARISON
# ============================================================================
elif page == "Student Comparison":
    section_header("Compare", "Side-by-Side Student Comparison")

    ids = st.multiselect(
        "Select 2–4 Student IDs to compare",
        options=df['StudentID'].tolist(),
        default=df['StudentID'].tolist()[:2],
        max_selections=4,
    )

    if len(ids) < 2:
        st.info("Select at least two students to compare.")
    else:
        subset = df[df['StudentID'].isin(ids)].set_index('StudentID')

        cols = st.columns(len(ids))
        for i, sid in enumerate(ids):
            row = subset.loc[sid]
            with cols[i]:
                st.markdown(f"""
                    <div class="ledger-card">
                        <div class="ledger-metric-label">{sid}</div>
                        <div class="ledger-metric-value" style="font-size:1.5rem;">{row['PredictedScore']:.1f}</div>
                        <div style="margin-top:0.4rem;">{perf_pill(row['PredictedPerformance'])}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.write("")
        section_header("Profile", "Normalized Feature Comparison")

        radar_feats = ['StudyHours', 'Attendance', 'AssignmentCompletion', 'OnlineCourses', 'ExamScore']
        norm = df[radar_feats].copy()
        norm = (norm - norm.min()) / (norm.max() - norm.min())
        norm['StudentID'] = df['StudentID']

        fig = go.Figure()
        palette = [SAGE, BLUE, LAVENDER, RUST]
        for i, sid in enumerate(ids):
            vals = norm[norm['StudentID'] == sid][radar_feats].values.flatten().tolist()
            vals += vals[:1]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=radar_feats + [radar_feats[0]],
                fill='toself', name=sid,
                line_color=palette[i % len(palette)],
                opacity=0.75,
            ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID, showticklabels=False),
                angularaxis=dict(gridcolor=GRID),
            ),
            showlegend=True,
        )
        st.plotly_chart(plotly_theme(fig, 460), use_container_width=True)

        section_header("Detail", "Raw Feature Values")
        st.dataframe(subset[FEATURE_COLS + ['ExamScore', 'PredictedScore', 'PredictedPerformance']],
                     use_container_width=True)


# ============================================================================
# SUBJECT-WISE ANALYSIS
# ============================================================================
elif page == "Subject-wise Analysis":
    section_header("By Subject", "Subject-wise Performance Analysis")
    st.caption("⚠ The source dataset provides only an overall exam score — subject-level marks shown "
               "here are simulated (ExamScore ± seeded noise) purely to demonstrate this view. "
               "Replace with real per-subject columns for production use.")

    tab1, tab2 = st.tabs(["Cohort Averages", "Individual Student"])

    with tab1:
        subj_avg = df[SUBJECTS].mean().sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=subj_avg.values, y=subj_avg.index, orientation='h',
            marker_color=GOLD, text=[f"{v:.1f}" for v in subj_avg.values], textposition="outside",
        ))
        fig.update_layout(title="Average Score by Subject (All Students)", xaxis_range=[0, 100])
        st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)

        fig2 = go.Figure()
        for subj in SUBJECTS:
            fig2.add_trace(go.Violin(y=df[subj], name=subj, box_visible=True, meanline_visible=True,
                                      line_color=GOLD, fillcolor=hex_to_rgba(GOLD, 0.25), opacity=0.7))
        fig2.update_layout(title="Score Spread by Subject")
        st.plotly_chart(plotly_theme(fig2, 420), use_container_width=True)

    with tab2:
        sid = st.selectbox("Select a Student", df['StudentID'].tolist())
        srow = df[df['StudentID'] == sid].iloc[0]

        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.markdown(f"""
                <div class="ledger-card">
                    <div class="ledger-metric-label">{sid}</div>
                    <div class="ledger-metric-value" style="font-size:1.6rem;">{srow['PredictedScore']:.1f}</div>
                    <div style="margin-top:0.4rem;">{perf_pill(srow['PredictedPerformance'])}</div>
                    <div class="ledger-metric-sub" style="margin-top:0.8rem;">Overall Exam Score: {srow['ExamScore']:.0f}</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            subj_scores = srow[SUBJECTS]
            fig = go.Figure(go.Bar(
                x=SUBJECTS, y=subj_scores.values,
                marker_color=[SAGE if v >= 75 else (BLUE if v >= 55 else RUST) for v in subj_scores.values],
                text=[f"{v:.1f}" for v in subj_scores.values], textposition="outside",
            ))
            fig.add_hline(y=df[SUBJECTS].mean().mean(), line_dash="dot", line_color=SLATE,
                          annotation_text="cohort avg", annotation_font_color=SLATE)
            fig.update_layout(title=f"Subject Scores — {sid}", yaxis_range=[0, 100])
            st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)


# ============================================================================
# PERFORMANCE TRENDS
# ============================================================================
elif page == "Performance Trends":
    section_header("Trends", "How Behavior Relates to Performance")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            df, x="StudyHours", y="ExamScore", color="PredictedPerformance",
            color_discrete_map=PERF_COLORS, opacity=0.55,
            trendline="ols", trendline_scope="overall",
        )
        fig.update_layout(title="Study Hours vs. Exam Score")
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(plotly_theme(fig, 400), use_container_width=True)

    with c2:
        fig = px.scatter(
            df, x="Attendance", y="ExamScore", color="PredictedPerformance",
            color_discrete_map=PERF_COLORS, opacity=0.55,
            trendline="ols", trendline_scope="overall",
        )
        fig.update_layout(title="Attendance vs. Exam Score")
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(plotly_theme(fig, 400), use_container_width=True)

    section_header("Segments", "Average Score by Study Hours & Attendance Band")
    df_binned = df.copy()
    df_binned['StudyBand'] = pd.cut(df_binned['StudyHours'], bins=4)
    df_binned['AttendBand'] = pd.cut(df_binned['Attendance'], bins=4)
    pivot = df_binned.pivot_table(values='ExamScore', index='StudyBand', columns='AttendBand', aggfunc='mean')
    pivot.index = [f"{int(i.left)}–{int(i.right)}h" for i in pivot.index]
    pivot.columns = [f"{int(c.left)}–{int(c.right)}%" for c in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, RUST], [0.5, BLUE], [1, SAGE]],
        text=np.round(pivot.values, 1), texttemplate="%{text}",
        colorbar=dict(title="Avg Score"),
    ))
    fig.update_layout(title="Average Exam Score — Study Hours × Attendance", xaxis_title="Attendance Band",
                      yaxis_title="Study Hours Band")
    st.plotly_chart(plotly_theme(fig, 420), use_container_width=True)

    st.caption("Same data, viewed as a 3D surface — rotate and tilt to see the peak performance zone.")
    fig3d_surface = go.Figure(go.Surface(
        z=pivot.values, x=list(range(len(pivot.columns))), y=list(range(len(pivot.index))),
        colorscale=[[0, RUST], [0.5, BLUE], [1, SAGE]],
        colorbar=dict(title="Avg Score"),
        showscale=True,
    ))
    fig3d_surface.update_layout(
        title="Exam Score Surface — Study Hours × Attendance",
        scene=dict(
            xaxis=dict(title="Attendance Band", tickvals=list(range(len(pivot.columns))), ticktext=list(pivot.columns),
                       backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID,
                       tickfont=dict(color=PARCHMENT, size=11), title_font=dict(color=SLATE, size=12)),
            yaxis=dict(title="Study Hours Band", tickvals=list(range(len(pivot.index))), ticktext=list(pivot.index),
                       backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID,
                       tickfont=dict(color=PARCHMENT, size=11), title_font=dict(color=SLATE, size=12)),
            zaxis=dict(title="Avg Exam Score", backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID,
                       tickfont=dict(color=PARCHMENT, size=11), title_font=dict(color=SLATE, size=12)),
        ),
    )
    st.plotly_chart(plotly_theme(fig3d_surface, 520), use_container_width=True)

    section_header("Factors", "Motivation & Stress Effects")
    c3, c4 = st.columns(2)
    with c3:
        tmp = df.copy()
        tmp['MotivationLabel'] = tmp['Motivation'].map({0: 'Low', 1: 'Medium', 2: 'High'})
        avg = tmp.groupby('MotivationLabel')['ExamScore'].mean().reindex(['Low', 'Medium', 'High'])
        fig = go.Figure(go.Bar(x=avg.index, y=avg.values, marker_color=GOLD,
                               text=[f"{v:.1f}" for v in avg.values], textposition="outside"))
        fig.update_layout(title="Avg Exam Score by Motivation")
        st.plotly_chart(plotly_theme(fig, 340), use_container_width=True)
    with c4:
        tmp['StressLabel'] = tmp['StressLevel'].map({0: 'Low', 1: 'Medium', 2: 'High'})
        avg2 = tmp.groupby('StressLabel')['ExamScore'].mean().reindex(['Low', 'Medium', 'High'])
        fig = go.Figure(go.Bar(x=avg2.index, y=avg2.values, marker_color=RUST,
                               text=[f"{v:.1f}" for v in avg2.values], textposition="outside"))
        fig.update_layout(title="Avg Exam Score by Stress Level")
        st.plotly_chart(plotly_theme(fig, 340), use_container_width=True)


# ============================================================================
# RECOMMENDATIONS
# ============================================================================
elif page == "Recommendations":
    section_header("Guidance", "Personalized Recommendation Engine")
    st.caption("Rule-based recommendations, prioritized by each factor's actual importance in the Random Forest model — "
               "not just the biggest raw gap, but the biggest *impact* fix.")

    sid = st.selectbox("Select a Student", df['StudentID'].tolist(), key="reco_student")
    row = df[df['StudentID'] == sid].iloc[0]

    top_l, top_r = st.columns([1, 2])
    with top_l:
        st.markdown(f"""
            <div class="ledger-card">
                <div class="ledger-metric-label">{sid}</div>
                <div class="ledger-metric-value" style="font-size:1.6rem;">{row['PredictedScore']:.1f}</div>
                <div style="margin-top:0.5rem;">{perf_pill(row['PredictedPerformance'])}</div>
                <div class="ledger-metric-sub" style="margin-top:0.8rem;">
                    Attendance {row['Attendance']:.0f}% · Study {row['StudyHours']:.0f}h/wk · Assignments {row['AssignmentCompletion']:.0f}%
                </div>
            </div>
        """, unsafe_allow_html=True)

    recommendations = get_recommendations(row, top_n=3)
    strengths = get_strengths(row, top_n=2)

    with top_r:
        if not recommendations:
            st.markdown(f"""
                <div class="ledger-card">
                    <div class="ledger-metric-label" style="color:{SAGE};">On Track</div>
                    <div style="color:{PARCHMENT}; margin-top:0.4rem;">
                        This student is meeting the healthy threshold on every tracked factor. Focus on maintaining
                        current habits, and consider stretch goals like a higher AssignmentCompletion target.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            priority_labels = ["Top Priority", "Secondary", "Also Consider"]
            priority_colors = [RUST, WARNING, BLUE]
            for i, rec in enumerate(recommendations):
                label = priority_labels[i] if i < len(priority_labels) else "Also Consider"
                color = priority_colors[i] if i < len(priority_colors) else BLUE
                st.markdown(f"""
                    <div class="ledger-card" style="margin-bottom:0.9rem;">
                        <span class="pill" style="background:{hex_to_rgba(color,0.15)}; color:{color}; border:1px solid {hex_to_rgba(color,0.4)};">{label}</span>
                        <div class="ledger-metric-value" style="font-size:1.1rem; margin-top:0.5rem;">{rec['feature']}</div>
                        <div class="ledger-metric-sub" style="color:{TEXT_MUTED}; margin-bottom:0.4rem;">
                            Current: {rec['value']} {rec['unit']}
                        </div>
                        <div style="color:{SLATE};">{rec['tip']}</div>
                    </div>
                """, unsafe_allow_html=True)

    if strengths:
        section_header("Strengths", "What's Already Working")
        cols = st.columns(len(strengths))
        for i, s in enumerate(strengths):
            with cols[i]:
                st.markdown(f"""
                    <div class="ledger-card">
                        <div class="ledger-metric-label" style="color:{SAGE};">{s['feature']}</div>
                        <div class="ledger-metric-value" style="font-size:1.3rem;">{s['value']}</div>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("<hr class='ledger-rule'>", unsafe_allow_html=True)
    section_header("Cohort View", "Most Common Recommendations Across All Students")
    st.caption("Which factors trigger a recommendation most often across the full student body — useful for spotting systemic issues (e.g. a cohort-wide attendance problem).")

    sample_for_cohort = df.sample(min(3000, len(df)), random_state=7)
    tally = {}
    for _, r in sample_for_cohort.iterrows():
        for rec in get_recommendations(r, top_n=3):
            tally[rec['feature']] = tally.get(rec['feature'], 0) + 1
    tally_series = pd.Series(tally).sort_values(ascending=True)

    if len(tally_series) > 0:
        fig = go.Figure(go.Bar(
            x=tally_series.values, y=tally_series.index, orientation='h',
            marker_color=GOLD, text=tally_series.values, textposition="outside",
        ))
        fig.update_layout(title=f"Recommendation Frequency (sample of {len(sample_for_cohort):,} students)")
        st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)
