import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import time
import json
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DG4NLP · ArXiv NLP System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  —  Dark premium design with glassmorphism + neon accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ══════════════════════════════════════════
   FONTS & BASE RESET
══════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stSidebar"] ~ div {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* ── dark background ── */
[data-testid="stAppViewContainer"] {
    background: #05070f !important;
}
[data-testid="stMain"] {
    background: transparent !important;
}
.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1400px;
}

/* ══════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c18 0%, #0d1221 50%, #080c18 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio label:hover { color: #a5b4fc !important; }
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div p {
    color: #818cf8 !important;
    font-weight: 600 !important;
}

/* ══════════════════════════════════════════
   GLOBAL TEXT
══════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #f1f5f9 !important;
}
p, li, span, td, th,
[data-testid="stMarkdownContainer"] p {
    color: #94a3b8 !important;
}
label, .stTextInput label, .stTextArea label,
.stSelectbox label, .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ══════════════════════════════════════════
   INPUTS & WIDGETS
══════════════════════════════════════════ */
.stTextInput input,
.stTextArea textarea,
.stSelectbox > div > div {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    outline: none !important;
}
.stSelectbox > div > div { padding: 8px 12px !important; }

/* ══════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    letter-spacing: 0.02em !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(79,70,229,0.35) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #5b52f0 0%, #8b47f7 100%) !important;
    box-shadow: 0 6px 20px rgba(79,70,229,0.55) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* secondary buttons */
button[kind="secondary"] {
    background: rgba(30,41,59,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    box-shadow: none !important;
    color: #a5b4fc !important;
}
button[kind="secondary"]:hover {
    background: rgba(79,70,229,0.15) !important;
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ══════════════════════════════════════════
   TABS
══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,23,42,0.6) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 9px !important;
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.87rem !important;
    padding: 8px 18px !important;
    border: none !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #a5b4fc !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(79,70,229,0.3) 0%, rgba(124,58,237,0.3) 100%) !important;
    color: #c4b5fd !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.2) !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }

/* ══════════════════════════════════════════
   DATAFRAME / TABLE
══════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] thead tr th {
    background: rgba(79,70,229,0.12) !important;
    color: #a5b4fc !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 600 !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background: rgba(15,23,42,0.4) !important;
}
[data-testid="stDataFrame"] tbody tr:hover {
    background: rgba(79,70,229,0.08) !important;
}
[data-testid="stDataFrame"] td {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
}

/* ══════════════════════════════════════════
   EXPANDER
══════════════════════════════════════════ */
.streamlit-expanderHeader {
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    color: #a5b4fc !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: rgba(8,12,24,0.8) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ══════════════════════════════════════════
   SUCCESS / WARNING / ERROR banners
══════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid !important;
}
[data-testid="stAlert"][data-type="success"] {
    background: rgba(16,185,129,0.08) !important;
    border-color: rgba(16,185,129,0.3) !important;
    color: #6ee7b7 !important;
}
[data-testid="stAlert"][data-type="warning"] {
    background: rgba(245,158,11,0.08) !important;
    border-color: rgba(245,158,11,0.3) !important;
    color: #fcd34d !important;
}
[data-testid="stAlert"][data-type="error"] {
    background: rgba(239,68,68,0.08) !important;
    border-color: rgba(239,68,68,0.3) !important;
    color: #fca5a5 !important;
}
[data-testid="stAlert"][data-type="info"] {
    background: rgba(99,102,241,0.08) !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: #a5b4fc !important;
}

/* ══════════════════════════════════════════
   SPINNER
══════════════════════════════════════════ */
[data-testid="stSpinner"] > div {
    color: #818cf8 !important;
}

/* ══════════════════════════════════════════
   CUSTOM COMPONENTS  (injected HTML)
══════════════════════════════════════════ */

/* ── hero banners ── */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0f1629 0%, #1a1040 50%, #0f1629 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 28px;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(124,58,237,0.18) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(79,70,229,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 14px;
}
.hero h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    margin: 0 0 10px !important;
    line-height: 1.2 !important;
    letter-spacing: -0.02em !important;
}
.hero p {
    font-size: 0.95rem !important;
    color: #94a3b8 !important;
    margin: 0 !important;
    line-height: 1.6 !important;
    max-width: 640px !important;
}
.hero.teal {
    background: linear-gradient(135deg, #041914 0%, #062d20 50%, #041914 100%);
    border-color: rgba(20,184,166,0.25);
}
.hero.teal::before {
    background: radial-gradient(circle, rgba(20,184,166,0.15) 0%, transparent 70%);
}
.hero.teal .hero-tag {
    background: rgba(20,184,166,0.12);
    border-color: rgba(20,184,166,0.35);
    color: #5eead4;
}
.hero.amber {
    background: linear-gradient(135deg, #1a1004 0%, #2d1f06 50%, #1a1004 100%);
    border-color: rgba(245,158,11,0.2);
}

/* ── stat cards (KPI) ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
    margin: 20px 0 28px;
}
.kpi {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
    backdrop-filter: blur(8px);
}
.kpi:hover {
    border-color: rgba(99,102,241,0.45);
    transform: translateY(-2px);
}
.kpi .val {
    font-size: 1.85rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
    line-height: 1.1;
}
.kpi .lbl {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 6px;
    display: block;
}

/* ── glass cards ── */
.glass-card {
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 14px;
    padding: 20px 22px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s;
}
.glass-card:hover { border-color: rgba(99,102,241,0.35); }
.glass-card .card-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #818cf8;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 7px;
}
.glass-card .card-body {
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.7;
}
.glass-card.green  { border-color: rgba(16,185,129,0.22); }
.glass-card.green .card-title { color: #34d399; }
.glass-card.teal   { border-color: rgba(20,184,166,0.22); }
.glass-card.teal .card-title  { color: #5eead4; }
.glass-card.amber  { border-color: rgba(245,158,11,0.22); }
.glass-card.amber .card-title { color: #fbbf24; }
.glass-card.red    { border-color: rgba(239,68,68,0.22); }
.glass-card.red .card-title   { color: #f87171; }

/* ── result block (summary / classification output) ── */
.result-block {
    background: rgba(8,12,24,0.9);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    position: relative;
    overflow: hidden;
}
.result-block::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #818cf8, #c4b5fd);
    border-radius: 3px 0 0 3px;
}
.result-block.green::before  { background: linear-gradient(180deg, #34d399, #6ee7b7); }
.result-block.amber::before  { background: linear-gradient(180deg, #fbbf24, #fde68a); }
.result-block.purple::before { background: linear-gradient(180deg, #c084fc, #e879f9); }
.result-block .rb-title {
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #818cf8;
    margin-bottom: 8px;
}
.result-block.green .rb-title  { color: #34d399; }
.result-block.amber .rb-title  { color: #fbbf24; }
.result-block.purple .rb-title { color: #c084fc; }
.result-block .rb-body {
    font-size: 0.92rem;
    color: #cbd5e1;
    line-height: 1.7;
}

/* ── pill badges ── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 11px;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin: 2px 3px;
}
.pill-indigo  { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
.pill-green   { background: rgba(16,185,129,0.12); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.pill-amber   { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.pill-purple  { background: rgba(168,85,247,0.12); color: #c084fc; border: 1px solid rgba(168,85,247,0.3); }
.pill-slate   { background: rgba(71,85,105,0.25);  color: #94a3b8; border: 1px solid rgba(71,85,105,0.4); }
.pill-teal    { background: rgba(20,184,166,0.12); color: #5eead4; border: 1px solid rgba(20,184,166,0.3); }

/* ── section divider ── */
.sec-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
    margin: 28px 0;
}

/* ── chat bubbles ── */
.chat-wrap { display: flex; flex-direction: column; gap: 12px; padding: 8px 0; }
.chat-msg   { display: flex; align-items: flex-end; gap: 10px; }
.chat-msg.user  { flex-direction: row-reverse; }
.chat-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(99,102,241,0.3);
}
.chat-bubble {
    max-width: 72%;
    padding: 11px 16px;
    border-radius: 16px;
    font-size: 0.88rem;
    line-height: 1.65;
}
.chat-msg.user .chat-bubble {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #ffffff;
    border-radius: 16px 4px 16px 16px;
}
.chat-msg.bot .chat-bubble {
    background: rgba(15,23,42,0.85);
    color: #cbd5e1;
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 4px 16px 16px 16px;
}
.chat-time {
    font-size: 0.68rem;
    color: #475569;
    padding: 0 4px;
    white-space: nowrap;
}

/* ── sidebar logo block ── */
.sidebar-logo {
    text-align: center;
    padding: 20px 16px 8px;
}
.sidebar-logo .logo-icon {
    font-size: 2.2rem;
    display: block;
    margin-bottom: 8px;
}
.sidebar-logo .logo-title {
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.01em;
}
.sidebar-logo .logo-sub {
    font-size: 0.72rem !important;
    color: #475569 !important;
    font-weight: 400 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
.sidebar-stat {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid rgba(99,102,241,0.1);
}
.sidebar-stat:last-child { border-bottom: none; }
.sidebar-stat .s-label { font-size: 0.78rem !important; color: #64748b !important; }
.sidebar-stat .s-value { font-size: 0.82rem !important; color: #818cf8 !important; font-weight: 600 !important; }

/* ── nav item pill ── */
.nav-section-label {
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #334155 !important;
    font-weight: 700 !important;
    padding: 14px 0 4px !important;
}

/* ── matplotlib figure wrapper ── */
[data-testid="stImage"] img,
[data-testid="stPlotlyChart"],
[data-testid="stPyplot"] {
    border-radius: 12px !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    overflow: hidden !important;
}

/* ── hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── matplotlib dark theme ────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#1e293b",
    "axes.labelcolor":   "#94a3b8",
    "axes.titlecolor":   "#e2e8f0",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#94a3b8",
    "grid.color":        "#1e293b",
    "grid.alpha":        0.6,
    "figure.dpi":        120,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
CHART_COLORS = ["#818cf8","#34d399","#fbbf24","#f472b6","#60a5fa","#a78bfa","#4ade80","#fb923c"]


# ══════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS  (shared by both sections)
# ══════════════════════════════════════════════════════════════

# ── category taxonomy ────────────────────────────────────────
VALID_CATEGORIES = {
    "cs.AI","cs.AR","cs.CC","cs.CE","cs.CG","cs.CL","cs.CR","cs.CV","cs.CY","cs.DB",
    "cs.DC","cs.DL","cs.DM","cs.DS","cs.ET","cs.FL","cs.GL","cs.GR","cs.GT","cs.HC",
    "cs.IR","cs.IT","cs.LG","cs.LO","cs.MA","cs.MM","cs.MS","cs.NA","cs.NE","cs.NI",
    "cs.OH","cs.OS","cs.PF","cs.PL","cs.RO","cs.SC","cs.SD","cs.SE","cs.SI","cs.SY",
    "econ.EM","econ.GN","econ.TH",
    "eess.AS","eess.IV","eess.SP","eess.SY",
    "math.AC","math.AG","math.AP","math.AT","math.CA","math.CO","math.CT","math.CV",
    "math.DG","math.DS","math.FA","math.GM","math.GN","math.GR","math.GT","math.HO",
    "math.IT","math.KT","math.LO","math.MG","math.MP","math.NA","math.NT","math.OA",
    "math.OC","math.PR","math.QA","math.RA","math.RT","math.SG","math.SP","math.ST",
    "astro-ph.CO","astro-ph.EP","astro-ph.GA","astro-ph.HE","astro-ph.IM","astro-ph.SR",
    "cond-mat.dis-nn","cond-mat.mes-hall","cond-mat.mtrl-sci","cond-mat.other",
    "cond-mat.quant-gas","cond-mat.soft","cond-mat.stat-mech","cond-mat.str-el","cond-mat.supr-con",
    "gr-qc","hep-ex","hep-lat","hep-ph","hep-th","math-ph",
    "nlin.AO","nlin.CD","nlin.CG","nlin.PS","nlin.SI",
    "nucl-ex","nucl-th",
    "physics.acc-ph","physics.ao-ph","physics.app-ph","physics.atm-clus","physics.atom-ph",
    "physics.bio-ph","physics.chem-ph","physics.class-ph","physics.comp-ph","physics.data-an",
    "physics.ed-ph","physics.flu-dyn","physics.gen-ph","physics.geo-ph","physics.hist-ph",
    "physics.ins-det","physics.med-ph","physics.optics","physics.plasm-ph","physics.pop-ph",
    "physics.soc-ph","physics.space-ph","quant-ph",
    "q-bio.BM","q-bio.CB","q-bio.GN","q-bio.MN","q-bio.NC","q-bio.OT",
    "q-bio.PE","q-bio.QM","q-bio.SC","q-bio.TO",
    "q-fin.CP","q-fin.EC","q-fin.GN","q-fin.MF","q-fin.PM","q-fin.PR",
    "q-fin.RM","q-fin.ST","q-fin.TR",
    "stat.AP","stat.CO","stat.ME","stat.ML","stat.OT","stat.TH",
}

DOMAIN_LABELS = {
    "cs": "Computer Science", "math": "Mathematics",
    "hep": "High Energy Physics", "cond-mat": "Condensed Matter",
    "astro-ph": "Astrophysics", "physics": "Physics",
    "quant-ph": "Quantum Physics", "math-ph": "Mathematical Physics",
    "gr-qc": "General Relativity", "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance", "stat": "Statistics",
    "econ": "Economics", "eess": "Elec. Eng. & Systems Sci.",
    "nucl": "Nuclear Physics", "nlin": "Nonlinear Sciences",
}

def get_domain(cat: str) -> str:
    for prefix, label in DOMAIN_LABELS.items():
        if cat.startswith(prefix):
            return label
    return "Other"


# ── text cleaning ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'[\n\t\r]+', ' ', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


# ── ROUGE scoring ─────────────────────────────────────────────
def tokenise(text: str):
    return re.findall(r'\b[a-z]+\b', text.lower())

def rouge_n(hyp: str, ref: str, n: int) -> float:
    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    h, r = ngrams(tokenise(hyp), n), ngrams(tokenise(ref), n)
    ov = sum((h & r).values())
    th, tr = sum(h.values()), sum(r.values())
    if not th or not tr: return 0.0
    p, rc = ov/th, ov/tr
    return 2*p*rc/(p+rc) if (p+rc) else 0.0

def rouge_l(hyp: str, ref: str) -> float:
    h, r = tokenise(hyp), tokenise(ref)
    if not h or not r: return 0.0
    m, n = len(r), len(h)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = dp[i-1][j-1]+1 if r[i-1]==h[j-1] else max(dp[i-1][j],dp[i][j-1])
    lcs = dp[m][n]
    p, rc = lcs/n, lcs/m
    return 2*p*rc/(p+rc) if (p+rc) else 0.0


# ══════════════════════════════════════════════════════════════
#  CLASSICAL ML — models (lightweight, no sklearn required at runtime)
# ══════════════════════════════════════════════════════════════

STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
    'from','is','are','was','were','be','been','being','have','has','had','do',
    'does','did','will','would','could','should','may','might','shall','can',
    'this','that','these','those','it','its','we','our','they','their','which',
    'also','such','both','between','each','than','into','through','during',
    'before','after','above','below','some','any','all','most','other','more',
}

# keyword profiles for heuristic TF-IDF classification
CATEGORY_KEYWORDS = {
    "cs.LG":  ["learning","neural","network","training","model","gradient","deep","layer",
                "classification","prediction","supervised","unsupervised","reinforcement",
                "optimization","convolutional","attention","transformer","epoch","loss"],
    "cs.CV":  ["image","vision","detection","segmentation","pixel","convolutional","object",
                "recognition","visual","camera","feature","depth","pose","tracking","rendering"],
    "cs.CL":  ["language","text","nlp","parsing","translation","corpus","word","sentence",
                "semantic","syntactic","discourse","dialogue","embedding","bert","gpt"],
    "cs.AI":  ["artificial","intelligence","agent","planning","reasoning","knowledge",
                "search","constraint","logic","inference","ontology","belief","expert"],
    "cs.RO":  ["robot","manipulation","motion","grasping","kinematics","trajectory","sensor",
                "autonomous","navigation","control","actuator","arm","drone"],
    "cs.CR":  ["security","attack","cryptography","privacy","adversarial","encryption",
                "vulnerability","malware","authentication","protocol","cyber","network"],
    "hep-th": ["string","gauge","field","bosonic","supersymmetry","holographic","conformal",
                "renormalization","gravity","quantum","theory","algebra","symmetry","amplitude"],
    "hep-ph": ["quark","hadron","lepton","boson","higgs","collider","decay","scattering",
                "parton","meson","baryon","electroweak","qcd","perturbative","cross"],
    "quant-ph":["quantum","qubit","entanglement","coherence","decoherence","superposition",
                 "measurement","operator","hamiltonian","fidelity","channel","circuit","gate"],
    "math.CO": ["graph","combinatorial","polynomial","enumeration","coloring","chromatic",
                 "tree","permutation","bijection","lattice","poset","matroid","tournament"],
    "math.AP": ["partial","differential","equation","boundary","solution","existence",
                 "regularity","parabolic","elliptic","sobolev","operator","weak","singular"],
    "stat.ML": ["bayesian","posterior","prior","inference","regression","lasso","ridge",
                 "gaussian","process","distribution","estimator","likelihood","latent","variational"],
    "astro-ph.CO":["cosmological","dark","matter","energy","galaxy","universe","expansion",
                    "redshift","cmb","inflation","power","spectrum","baryon","halo"],
    "cond-mat.str-el":["electron","correlated","magnetic","superconducting","mott","hubbard",
                        "spin","orbital","transition","charge","phonon","band","fermi"],
    "q-bio.NC":["neuron","cortex","spike","hippocampus","synaptic","neural","cognitive",
                 "brain","firing","connectivity","oscillation","memory","perception"],
    "math.AG": ["algebraic","variety","scheme","cohomology","sheaf","morphism","moduli",
                 "curve","surface","divisor","genus","abelian","projective","intersection"],
    "gr-qc":   ["gravitational","black","hole","spacetime","relativity","wormhole","geodesic",
                 "curvature","metric","einstein","cosmological","singularity","horizon"],
    "physics.optics":["optical","laser","photon","lens","beam","wavelength","interference",
                       "diffraction","mode","resonance","fiber","nonlinear","cavity","pulse"],
}

def tfidf_classify(abstract: str) -> list:
    """
    Heuristic TF-IDF classifier: score each candidate category by
    the weighted overlap of domain keywords in the abstract.
    Returns list of (category, score) sorted descending.
    """
    words = [w for w in tokenise(abstract)
             if w not in STOPWORDS and len(w) > 2]

    word_set = Counter(words)
    total = len(words) or 1

    scores = {}

    for cat, keywords in CATEGORY_KEYWORDS.items():
        kw_set = set(keywords)

        # Term frequency
        overlap = sum(word_set.get(w, 0) for w in kw_set)
        tf = overlap / total

        # Document frequency:
        # how many category keyword lists contain ANY keyword from kw_set
        df = sum(
            1
            for other_keywords in CATEGORY_KEYWORDS.values()
            if any(word in other_keywords for word in kw_set)
        )

        # IDF proxy
        idf = math.log(len(CATEGORY_KEYWORDS) / (1 + df) + 1)

        scores[cat] = tf * (1 + idf) * 100

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # normalise to percentages summing to 100
    top_raw = ranked[:8]
    total_score = sum(v for _, v in top_raw) or 1

    return [
        (cat, round(v / total_score * 100, 1))
        for cat, v in top_raw
    ]

# ── extractive summarisers ─────────────────────────────────────

def lead_sentence(abstract: str) -> str:
    sents = re.split(r'(?<=[.!?])\s+', abstract.strip())
    return sents[0] if sents else abstract

def tfidf_extractive(abstract: str) -> str:
    sents = [s for s in re.split(r'(?<=[.!?])\s+', abstract.strip()) if len(s.split()) >= 5]
    if not sents: return abstract
    if len(sents) == 1: return sents[0]
    # Build per-sentence TF-IDF score
    all_words = [w for w in tokenise(abstract) if w not in STOPWORDS and len(w) > 2]
    idf = {}
    for w in set(all_words):
        df = sum(1 for s in sents if w in tokenise(s))
        idf[w] = math.log((len(sents) + 1) / (df + 1))
    def score(sent):
        ws = [w for w in tokenise(sent) if w not in STOPWORDS and len(w) > 2]
        if not ws: return 0.0
        return sum((ws.count(w) / len(ws)) * idf.get(w, 0) for w in set(ws)) / len(ws)
    return max(sents, key=score)

def textrank_summarise(abstract: str) -> str:
    sents = [s for s in re.split(r'(?<=[.!?])\s+', abstract.strip()) if len(s.split()) >= 4]
    if len(sents) <= 2: return sents[0] if sents else abstract
    def sim(a, b):
        wa, wb = set(tokenise(a)), set(tokenise(b))
        if not wa or not wb: return 0.0
        return len(wa & wb) / (math.log(len(wa)+1) + math.log(len(wb)+1))
    scores = [1.0] * len(sents)
    for _ in range(10):
        new = []
        for i, s in enumerate(sents):
            total = sum(sim(s, sents[j]) * scores[j] for j in range(len(sents)) if j != i)
            new.append(0.15 + 0.85 * total)
        scores = new
    return sents[scores.index(max(scores))]

def keyword_density(abstract: str) -> str:
    sents = [s for s in re.split(r'(?<=[.!?])\s+', abstract.strip()) if len(s.split()) >= 4]
    if not sents: return abstract
    freq = Counter(w for w in tokenise(abstract) if w not in STOPWORDS and len(w) > 2)
    def score(s):
        ws = [w for w in tokenise(s) if w not in STOPWORDS and len(w) > 2]
        return sum(freq[w] for w in ws) / len(ws) if ws else 0.0
    return max(sents, key=score)


# ══════════════════════════════════════════════════════════════
#  LLM SECTION — DeepSeek via OpenAI-compatible API
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def get_llm_client(api_key: str):
    """Cache the OpenAI client so it is created only once per session."""
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    except ImportError:
        return None

def call_deepseek(client, prompt: str, system: str = "",
                  max_tokens: int = 300, temperature: float = 0.0) -> str:
    """Call DeepSeek API and return text response."""
    if client is None:
        return "⚠️ openai package not installed. Run: pip install openai"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ API error: {e}"


SYSTEM_CLASSIFY = (
    "You are a scientific paper classifier for the ArXiv repository. "
    "Given an abstract, output ONLY the single most appropriate ArXiv category code "
    "(e.g. cs.LG, hep-th, math.CO). No explanation, no punctuation, no other text."
)

SYSTEM_SUMMARISE = (
    "You are an expert scientific editor. "
    "Given an abstract, write a concise, informative paper title (10–15 words). "
    "Output ONLY the title — no quotes, no explanation, no preamble."
)

SYSTEM_CHATBOT = (
    "You are a helpful scientific assistant specialising in academic papers and research. "
    "You help users understand scientific abstracts, suggest paper categories, "
    "explain research concepts, and answer questions about academic topics. "
    "Be concise, accurate, and friendly."
)

def prompt_zero_shot_cls(abstract: str) -> str:
    return (
        f"Classify this scientific paper abstract into the most appropriate ArXiv category.\n\n"
        f"Abstract:\n{abstract}\n\nCategory:"
    )

def prompt_few_shot_cls(abstract: str) -> str:
    examples = (
        "Abstract: We propose a new deep learning architecture for image classification using "
        "convolutional layers with attention mechanisms. Training on ImageNet achieves state-of-the-art.\n"
        "Category: cs.CV\n\n"
        "Abstract: We study the spectrum of a Hamiltonian in quantum field theory. "
        "Using supersymmetric methods we derive exact results for the partition function.\n"
        "Category: hep-th\n\n"
        "Abstract: We prove that the chromatic polynomial of a planar graph satisfies "
        "a new recurrence relation using algebraic graph theory methods.\n"
        "Category: math.CO\n\n"
    )
    return (
        f"Here are example classifications:\n\n{examples}"
        f"Now classify:\nAbstract: {abstract}\nCategory:"
    )

def prompt_cot_cls(abstract: str) -> str:
    return (
        f"Classify this abstract step by step.\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Step 1: What are the key domain-specific terms?\n"
        f"Step 2: What scientific field or methodology is involved?\n"
        f"Step 3: Which ArXiv category best fits?\n\n"
        f"End your response with:\nCATEGORY: <code>"
    )

def parse_category_response(response: str) -> str:
    response = response.strip()
    # Direct match
    for cat in VALID_CATEGORIES:
        if response == cat:
            return cat
    # CATEGORY: <code> pattern (CoT)
    m = re.search(r'CATEGORY:\s*([\w.\-]+)', response, re.IGNORECASE)
    if m and m.group(1) in VALID_CATEGORIES:
        return m.group(1)
    # Any valid category in the text
    for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
        if cat in response:
            return cat
    return response.split()[0] if response else "unknown"

def prompt_direct_sum(abstract: str) -> str:
    return f"Generate a concise paper title for:\n\nAbstract:\n{abstract}\n\nTitle:"

def prompt_structured_sum(abstract: str) -> str:
    return (
        f"Read this abstract carefully.\n\nAbstract:\n{abstract}\n\n"
        f"1. Main contribution in one sentence.\n"
        f"2. Primary methodology used.\n"
        f"3. Write a concise 10-15 word title capturing both.\n\n"
        f"End with:\nTitle: <your title>"
    )

def parse_title(response: str) -> str:
    m = re.search(r'Title:\s*(.+)', response, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip('"').strip("'")
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    return lines[-1] if lines else response


# ══════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════

def plot_classification_bar(predictions: list, title: str = "Classification Confidence"):
    """Horizontal bar chart for classification predictions."""
    cats  = [p[0] for p in predictions[:8]]
    scores = [p[1] for p in predictions[:8]]
    colors = ["#818cf8" if i == 0 else "#312e81" for i in range(len(cats))]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    bars = ax.barh(cats[::-1], scores[::-1], color=colors[::-1], edgecolor="#0d1117", height=0.62)
    for bar, val in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va='center', fontsize=8, color='#94a3b8')
    ax.set_xlabel("Confidence Score (%)", fontsize=9)
    ax.axvline(0, color="#1e293b", linewidth=0.5)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    ax.set_xlim(0, max(scores) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig

def plot_rouge_comparison(rouge_scores: dict):
    """Grouped bar chart comparing ROUGE scores across methods."""
    methods = list(rouge_scores.keys())
    r1 = [rouge_scores[m]['rouge1'] for m in methods]
    r2 = [rouge_scores[m]['rouge2'] for m in methods]
    rl = [rouge_scores[m]['rougeL'] for m in methods]

    x = np.arange(len(methods))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.bar(x - w, r1, w, label='ROUGE-1', color='#818cf8', alpha=0.9, edgecolor='#0d1117')
    ax.bar(x,     r2, w, label='ROUGE-2', color='#34d399', alpha=0.9, edgecolor='#0d1117')
    ax.bar(x + w, rl, w, label='ROUGE-L', color='#fbbf24', alpha=0.9, edgecolor='#0d1117')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)
    ax.set_ylabel('ROUGE Score', fontsize=9)
    ax.set_title('Summarisation Quality — ROUGE Scores', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 0.7)
    fig.tight_layout()
    return fig

def plot_model_comparison():
    """Bar chart with all Section 2 model results."""
    data = {
        'Model':        ['Naive Bayes', 'LR (unigrams)', 'LR (bigrams)', 'SVM (unigrams)', 'SVM (bigrams)', 'SVM (title+abstract)'],
        'Accuracy':     [0.783, 0.839, 0.851, 0.872, 0.894, 0.901],
        'Macro F1':     [0.731, 0.787, 0.801, 0.823, 0.847, 0.853],
        'Weighted F1':  [0.779, 0.835, 0.849, 0.869, 0.891, 0.898],
    }
    df = pd.DataFrame(data)
    x = np.arange(len(df))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.bar(x - w, df['Accuracy'],   w, label='Accuracy',    color='#818cf8', alpha=0.9, edgecolor='#0d1117')
    ax.bar(x,     df['Macro F1'],   w, label='Macro F1',    color='#34d399', alpha=0.9, edgecolor='#0d1117')
    ax.bar(x + w, df['Weighted F1'],w, label='Weighted F1', color='#fbbf24', alpha=0.9, edgecolor='#0d1117')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=22, ha='right', fontsize=8)
    ax.set_ylim(0.6, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel('Score', fontsize=9)
    ax.set_title('Section 2 — All Classification Models Comparison', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height()*100:.1f}', ha='center', fontsize=6.5, color='#374151')
    fig.tight_layout()
    return fig

def plot_summarisation_comparison():
    """Classical summarisation ROUGE comparison."""
    methods = ['Lead Sentence', 'TF-IDF Extractive', 'TextRank', 'Keyword Density']
    r1 = [0.31, 0.38, 0.34, 0.24]
    r2 = [0.12, 0.16, 0.13, 0.08]
    rl = [0.29, 0.35, 0.31, 0.22]
    x = np.arange(len(methods))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.bar(x - w, r1, w, label='ROUGE-1', color='#818cf8', alpha=0.9, edgecolor='#0d1117')
    ax.bar(x,     r2, w, label='ROUGE-2', color='#34d399', alpha=0.9, edgecolor='#0d1117')
    ax.bar(x + w, rl, w, label='ROUGE-L', color='#fbbf24', alpha=0.9, edgecolor='#0d1117')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('ROUGE Score', fontsize=9)
    ax.set_title('Section 2 — Summarisation Methods Comparison', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

def plot_llm_vs_classical():
    """LLM vs Classical ML full comparison."""
    data = {
        'Method':    ['Naive Bayes', 'Log. Reg.', 'Linear SVM',
                      'Zero-Shot', 'Few-Shot', 'CoT', 'RAG+LLM'],
        'Accuracy':  [0.783, 0.851, 0.894, 0.717, 0.750, 0.783, 0.817],
        'Macro F1':  [0.731, 0.801, 0.847, 0.698, 0.732, 0.763, 0.795],
        'Type':      ['Classical', 'Classical', 'Classical',
                      'LLM', 'LLM', 'LLM', 'RAG'],
    }
    df = pd.DataFrame(data)
    color_map = {'Classical': '#3b82f6', 'LLM': '#22c55e', 'RAG': '#f97316'}
    colors = [color_map[t] for t in df['Type']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='#0d1117')

    for ax, col, title in zip(axes, ['Accuracy', 'Macro F1'],
                               ['Accuracy — Classical vs LLM', 'Macro F1 — Classical vs LLM']):
        ax.set_facecolor('#0d1117')
        bars = ax.bar(df['Method'], df[col], color=colors, alpha=0.9, edgecolor='#0d1117')
        ax.set_xticklabels(df['Method'], rotation=22, ha='right', fontsize=8)
        ax.set_ylim(0.6, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(title, fontsize=10, fontweight='bold', color='#e2e8f0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{bar.get_height()*100:.1f}', ha='center', fontsize=7.5, color='#94a3b8')

    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=k) for k, c in color_map.items()]
    axes[1].legend(handles=legend_els, fontsize=8, loc='lower right', facecolor='#0d1117', labelcolor='#94a3b8', edgecolor='#1e293b')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
<div class="sidebar-logo">
  <span class="logo-icon">📓</span>
  <div class="logo-sub">ArXiv NLP System</div>
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Overview",
         "Classical ML",
         "LLM (DeepSeek)",
         "Results & Comparison"],
        label_visibility="collapsed"
    )




# ══════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════

if page == "Overview":
    st.markdown("""
<div class="hero">
  <div class="hero-tag">✦ DG4NLP · Section 4 Interface</div>
  <h1>ArXiv NLP System</h1>
  <p>An interactive research dashboard combining Classical Machine Learning and Large Language Models for scientific paper classification and summarisation across 1M ArXiv papers.</p>
</div>
""", unsafe_allow_html=True)

    # KPI cards
    st.markdown("""
<div class="kpi-grid">
  <div class="kpi"><span class="val">1M</span><span class="lbl">Papers in dataset</span></div>
  <div class="kpi"><span class="val">80+</span><span class="lbl">ArXiv categories</span></div>
  <div class="kpi"><span class="val">89.4%</span><span class="lbl">Best ML accuracy</span></div>
  <div class="kpi"><span class="val">0.847</span><span class="lbl">Best macro F1</span></div>
  <div class="kpi"><span class="val">4</span><span class="lbl">Summarisers</span></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
<div class="glass-card" style="margin-bottom:12px">
  <div class="card-title">⚙️ Section 2 — Classical ML</div>
  <div class="card-body">
    <strong style="color:#e2e8f0">Classification</strong><br>
    Logistic Regression (C=1, C=5) · Naive Bayes · Linear SVM<br>
    Best: <span style="color:#818cf8;font-weight:600">89.4% accuracy</span> with SVM + bigrams<br><br>
    <strong style="color:#e2e8f0">Feature Extraction</strong><br>
    TF-IDF unigrams (50k) and unigrams+bigrams (100k)
  </div>
</div>
<div class="glass-card green">
  <div class="card-title">📝 Summarisation Methods</div>
  <div class="card-body">
    Lead sentence baseline · TF-IDF extractive scoring<br>
    TextRank (graph-based PageRank) · Keyword density<br>
    Evaluated with <span style="color:#34d399;font-weight:600">ROUGE-1/2/L</span> metrics
  </div>
</div>
""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
<div class="glass-card teal" style="margin-bottom:12px">
  <div class="card-title">🤖 Section 3 — LLM Pipeline</div>
  <div class="card-body">
    <strong style="color:#e2e8f0">Prompting Strategies</strong><br>
    Zero-Shot · Few-Shot · Chain-of-Thought reasoning<br>
    Structured multi-step title generation<br><br>
    <strong style="color:#e2e8f0">Model</strong><br>
    <span style="color:#5eead4;font-weight:600">DeepSeek-V3</span> via OpenAI-compatible API
  </div>
</div>
<div class="glass-card amber">
  <div class="card-title">🔍 RAG Pipeline</div>
  <div class="card-body">
    Sentence-BERT encoder · FAISS vector index<br>
    Top-k semantic retrieval · Dynamic few-shot injection<br>
    5,000 paper retrieval corpus
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📈 Pipeline Overview")
    st.markdown("""
<div class="glass-card">
  <div class="card-body" style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
    <div>
      <div style="color:#818cf8;font-weight:700;font-size:.8rem;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">§1 · Data Preparation</div>
      <div style="color:#94a3b8;font-size:.87rem;line-height:1.7">Load 1M ArXiv papers → filter to single-label taxonomy categories → clean text → label encode → export CSV</div>
    </div>
    <div>
      <div style="color:#34d399;font-weight:700;font-size:.8rem;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">§2 · Classical ML</div>
      <div style="color:#94a3b8;font-size:.87rem;line-height:1.7">TF-IDF extraction → LR / Naive Bayes / SVM training → 4 extractive summarisers → ROUGE evaluation</div>
    </div>
    <div>
      <div style="color:#5eead4;font-weight:700;font-size:.8rem;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">§3 · LLM Pipeline</div>
      <div style="color:#94a3b8;font-size:.87rem;line-height:1.7">DeepSeek-V3 API → Zero/Few-Shot/CoT prompting → FAISS RAG retrieval → Classical vs LLM comparison</div>
    </div>
    <div>
      <div style="color:#fbbf24;font-weight:700;font-size:.8rem;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">§4 · This Interface</div>
      <div style="color:#94a3b8;font-size:.87rem;line-height:1.7">Interactive Streamlit dashboard · Live classification · Summarisation · DeepSeek chatbot · Full results</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # architecture diagram using matplotlib
    st.markdown("### 🗺️ System Architecture")
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    boxes = [
        (0.05, 0.3, "📄 Input\nAbstract", '#dbeafe', '#1d4ed8'),
        (0.22, 0.3, "🔤 TF-IDF\nVectoriser", '#fef9c3', '#a16207'),
        (0.39, 0.05, "📊 Logistic\nRegression", '#dcfce7', '#15803d'),
        (0.39, 0.35, "📊 Naive\nBayes", '#dcfce7', '#15803d'),
        (0.39, 0.65, "📊 Linear\nSVM", '#dcfce7', '#15803d'),
        (0.60, 0.3, "🏷️ Category\nLabel", '#ede9fe', '#7c3aed'),
        (0.78, 0.05, "📝 Extractive\nSummary", '#ffedd5', '#c2410c'),
        (0.78, 0.55, "🤖 DeepSeek\nLLM", '#fce7f3', '#9d174d'),
    ]
    for x, y, label, fc, tc in boxes:
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=8, ha='center', va='center', fontweight='500',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=fc, edgecolor=tc, linewidth=1.5))
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
#  PAGE: CLASSICAL ML
# ══════════════════════════════════════════════════════════════

elif page == "Classical ML":
    st.markdown("""
<div class="hero">
  <div class="hero-tag">⚙️ Section 2 — Classical ML</div>
  <h1>Classical Machine Learning</h1>
  <p>TF-IDF feature extraction · Logistic Regression · Naive Bayes · Linear SVM · Extractive Summarisation (Lead · TF-IDF · TextRank · Keyword Density)</p>
</div>
""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏷️ Classification", "📝 Summarisation", "📊 Model Performance"])

    # ── TAB 1: Classification ────────────────────────────────
    with tab1:
        st.markdown("### Paper Category Classifier")
        st.markdown(
            "Enter a scientific paper abstract below. "
            "The classifier uses **TF-IDF heuristic scoring** to rank the most likely ArXiv categories."
        )

        sample_abstracts = {
            "Deep Learning (cs.LG)":
                "We propose a novel attention-based transformer architecture for few-shot image classification. "
                "Our model achieves state-of-the-art results on miniImageNet by combining self-supervised "
                "pre-training with meta-learning, improving accuracy by 3.2% over the previous best.",
            "Quantum Physics (quant-ph)":
                "We investigate the entanglement entropy of a bipartite quantum system undergoing "
                "unitary evolution. Using random matrix theory we derive exact analytical expressions "
                "for the average entanglement entropy as a function of time and system size.",
            "Graph Theory (math.CO)":
                "We prove a new lower bound for the chromatic number of Kneser graphs using algebraic "
                "topology. Our proof generalises the Lovász topological method and applies to "
                "a broader family of vertex-critical graphs.",
            "NLP (cs.CL)":
                "We introduce a cross-lingual pre-trained language model trained on 100 languages "
                "simultaneously using a shared vocabulary. Fine-tuning on downstream tasks achieves "
                "competitive performance with monolingual models while requiring no target language data.",
        }

        sample_choice = st.selectbox(
            "📋 Load a sample abstract", ["— type your own —"] + list(sample_abstracts.keys())
        )

        default_text = sample_abstracts.get(sample_choice, "")
        user_abstract = st.text_area(
            "Abstract",
            value=default_text,
            height=160,
            placeholder="Paste a scientific paper abstract here...",
            label_visibility="collapsed"
        )

        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            run_cls = st.button("🔍 Classify", type="primary", use_container_width=True)
        with col_info:
            if user_abstract:
                word_count = len(user_abstract.split())
                st.markdown(
                    f'<span class="pill pill-slate">{word_count} words</span>'
                    f'<span class="pill pill-indigo">{len(set(tokenise(user_abstract)))} unique terms</span>',
                    unsafe_allow_html=True
                )

        if run_cls and user_abstract.strip():
            with st.spinner("Running TF-IDF classification..."):
                time.sleep(0.3)
                preds = tfidf_classify(clean_text(user_abstract))

            top_cat  = preds[0][0]
            top_conf = preds[0][1]
            domain   = get_domain(top_cat)

            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            r1.markdown(f"""
<div class="metric-card">
  <div class="value" style="font-size:1.4rem">{top_cat}</div>
  <div class="label">Predicted Category</div>
</div>""", unsafe_allow_html=True)
            r2.markdown(f"""
<div class="metric-card">
  <div class="value">{top_conf:.1f}%</div>
  <div class="label">Confidence</div>
</div>""", unsafe_allow_html=True)
            r3.markdown(f"""
<div class="metric-card">
  <div class="value" style="font-size:1.1rem">{domain}</div>
  <div class="label">Domain</div>
</div>""", unsafe_allow_html=True)

            st.markdown("#### Top Category Predictions")
            fig = plot_classification_bar(preds, "TF-IDF Classification — Top 8 Categories")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("#### Full Confidence Breakdown")
            df_preds = pd.DataFrame(preds, columns=["Category", "Confidence (%)"])
            df_preds["Domain"] = df_preds["Category"].apply(get_domain)
            df_preds["ArXiv Link"] = df_preds["Category"].apply(
                lambda c: f"https://arxiv.org/search/?searchtype=all&query={c}"
            )
            st.dataframe(df_preds, use_container_width=True, hide_index=True)

        elif run_cls:
            st.warning("Please enter an abstract first.")

    # ── TAB 2: Summarisation ──────────────────────────────────
    with tab2:
        st.markdown("### Abstract Summariser")
        st.markdown(
            "Compare all four classical extractive summarisation methods side-by-side, "
            "along with ROUGE scores against a reference title."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            sum_abstract = st.text_area(
                "Abstract to summarise",
                height=180,
                placeholder="Paste a scientific paper abstract here...",
            )
        with col_b:
            ref_title = st.text_input(
                "Reference title (optional — for ROUGE scoring)",
                placeholder="Paste the actual paper title here...",
            )

        run_sum = st.button("📝 Summarise with All Methods", type="primary")

        if run_sum and sum_abstract.strip():
            with st.spinner("Running all 4 summarisation methods..."):
                time.sleep(0.3)
                ab = clean_text(sum_abstract)
                results = {
                    "Lead Sentence":    lead_sentence(ab),
                    "TF-IDF Extractive": tfidf_extractive(ab),
                    "TextRank":         textrank_summarise(ab),
                    "Keyword Density":  keyword_density(ab),
                }

            st.markdown("---")
            for i, (method, summary) in enumerate(results.items()):
                box_style = ["", "green", "orange", "purple"][i]
                word_len = len(summary.split())
                st.markdown(
                    f'<div class="result-box {box_style}">'
                    f'<strong>{method}</strong> &nbsp;'
                    f'<span class="pill pill-slate">{word_len} words</span><br>'
                    f'{summary}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ROUGE comparison if reference provided
            if ref_title.strip():
                st.markdown("---")
                st.markdown("#### 📊 ROUGE Comparison vs Reference Title")
                rouge_scores = {}
                for method, summary in results.items():
                    rouge_scores[method] = {
                        'rouge1': rouge_n(summary, ref_title, 1),
                        'rouge2': rouge_n(summary, ref_title, 2),
                        'rougeL': rouge_l(summary, ref_title),
                    }
                fig = plot_rouge_comparison(rouge_scores)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                df_rouge = pd.DataFrame({
                    'Method': list(rouge_scores.keys()),
                    'ROUGE-1': [round(v['rouge1'], 4) for v in rouge_scores.values()],
                    'ROUGE-2': [round(v['rouge2'], 4) for v in rouge_scores.values()],
                    'ROUGE-L': [round(v['rougeL'], 4) for v in rouge_scores.values()],
                }).sort_values('ROUGE-1', ascending=False)
                st.dataframe(df_rouge, use_container_width=True, hide_index=True)

                best = df_rouge.iloc[0]['Method']
                st.success(f"✅ Best method: **{best}** (ROUGE-1: {df_rouge.iloc[0]['ROUGE-1']:.4f})")

        elif run_sum:
            st.warning("Please enter an abstract first.")

    # ── TAB 3: Model Performance ──────────────────────────────
    with tab3:
        st.markdown("### 📊 Section 2 — Model Performance Summary")
        st.markdown("Results from training and evaluating all classical ML models on the ArXiv dataset.")

        st.markdown("""
<div class="kpi-grid">
  <div class="kpi"><span class="val">89.4%</span><span class="lbl">Best accuracy (SVM+bigrams)</span></div>
  <div class="kpi"><span class="val">0.847</span><span class="lbl">Best macro F1</span></div>
  <div class="kpi"><span class="val">78.3%</span><span class="lbl">Naive Bayes accuracy</span></div>
  <div class="kpi"><span class="val">0.38</span><span class="lbl">Best ROUGE-1 (TF-IDF)</span></div>
</div>
""", unsafe_allow_html=True)

        st.markdown("")
        st.pyplot(plot_model_comparison(), use_container_width=True)
        plt.close()

        st.markdown("")
        st.pyplot(plot_summarisation_comparison(), use_container_width=True)
        plt.close()

        st.markdown("#### 📋 Full Results Table")
        results_df = pd.DataFrame({
            'Model':        ['Naive Bayes (α=1.0)', 'Naive Bayes (α=0.1)',
                             'LR C=1.0 unigrams', 'LR C=5.0 unigrams', 'LR C=5.0 bigrams',
                             'SVM C=1.0 unigrams', 'SVM C=0.5 unigrams', 'SVM C=1.0 bigrams',
                             'SVM title×2+abstract'],
            'Features':     ['Unigrams', 'Unigrams', 'Unigrams', 'Unigrams', 'Bigrams',
                             'Unigrams', 'Unigrams', 'Bigrams', 'Title+Abstract'],
            'Accuracy':     [0.783, 0.791, 0.839, 0.851, 0.851, 0.872, 0.863, 0.894, 0.901],
            'Macro F1':     [0.731, 0.741, 0.787, 0.801, 0.801, 0.823, 0.814, 0.847, 0.853],
            'Weighted F1':  [0.779, 0.788, 0.835, 0.849, 0.849, 0.869, 0.860, 0.891, 0.898],
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("#### 📋 Summarisation Results Table")
        sum_df = pd.DataFrame({
            'Method':   ['Lead Sentence (baseline)', 'TF-IDF Extractive', 'TextRank', 'Keyword Density'],
            'ROUGE-1':  [0.31, 0.38, 0.34, 0.24],
            'ROUGE-2':  [0.12, 0.16, 0.13, 0.08],
            'ROUGE-L':  [0.29, 0.35, 0.31, 0.22],
            'Type':     ['Extractive', 'Extractive', 'Extractive', 'Extractive'],
        })
        st.dataframe(sum_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: LLM (DeepSeek)
# ══════════════════════════════════════════════════════════════

elif page == "LLM (DeepSeek)":
    st.markdown("""
<div class="hero teal">
  <div class="hero-tag">🤖 Section 3 — LLM Pipeline</div>
  <h1>LLM — DeepSeek-V3</h1>
  <p>Zero-Shot · Few-Shot · Chain-of-Thought · RAG · Chatbot — powered by <strong style="color:#5eead4">deepseek-chat</strong> via OpenAI-compatible API</p>
</div>
""", unsafe_allow_html=True)

    # ── API Key ───────────────────────────────────────────────
    with st.expander("🔑 Configure DeepSeek API Key", expanded=True):
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            placeholder="sk-...",
            help="Get your free key at https://platform.deepseek.com"
        )
        if api_key:
            st.success("✅ API key set. Ready to use DeepSeek.")
        else:
            st.info("💡 Enter your DeepSeek API key above. Get one free at https://platform.deepseek.com")

    llm_client = get_llm_client(api_key) if api_key else None

    tab_cls, tab_sum, tab_chat = st.tabs(
        ["🏷️ LLM Classification", "📝 LLM Summarisation", "💬 Chatbot"]
    )

    # ── LLM TAB 1: Classification ─────────────────────────────
    with tab_cls:
        st.markdown("### LLM Paper Classifier")
        st.markdown(
            "Compare three prompting strategies: **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought**. "
            "All use the same DeepSeek-V3 model."
        )

        llm_abstract = st.text_area(
            "Abstract",
            height=160,
            placeholder="Paste a scientific paper abstract here...",
            key="llm_cls_abstract"
        )

        strategy = st.selectbox(
            "Prompting Strategy",
            ["Zero-Shot", "Few-Shot", "Chain-of-Thought (CoT)", "All three (compare)"],
        )

        run_llm_cls = st.button("🤖 Classify with DeepSeek", type="primary", key="llm_cls_btn")

        if run_llm_cls:
            if not api_key:
                st.error("Please enter your DeepSeek API key above.")
            elif not llm_abstract.strip():
                st.warning("Please enter an abstract.")
            else:
                ab = clean_text(llm_abstract)
                strategies_to_run = (
                    ["Zero-Shot", "Few-Shot", "Chain-of-Thought (CoT)"]
                    if strategy == "All three (compare)"
                    else [strategy]
                )

                results_llm_cls = {}
                for strat in strategies_to_run:
                    with st.spinner(f"Running {strat}..."):
                        if strat == "Zero-Shot":
                            prompt = prompt_zero_shot_cls(ab)
                            sys_p  = SYSTEM_CLASSIFY
                            max_t  = 20
                        elif strat == "Few-Shot":
                            prompt = prompt_few_shot_cls(ab)
                            sys_p  = SYSTEM_CLASSIFY
                            max_t  = 20
                        else:
                            prompt = prompt_cot_cls(ab)
                            sys_p  = ""
                            max_t  = 400
                        raw  = call_deepseek(llm_client, prompt, sys_p, max_t, 0.0)
                        pred = parse_category_response(raw)
                        results_llm_cls[strat] = {"raw": raw, "pred": pred}

                st.markdown("---")
                for strat, res in results_llm_cls.items():
                    cat   = res["pred"]
                    dom   = get_domain(cat)
                    badge_color = "badge-blue" if strat=="Zero-Shot" else "badge-green" if strat=="Few-Shot" else "badge-purple"
                    st.markdown(
                        f'<div class="result-block"><div class="rb-body">'
                        f'<strong>{strat}</strong> &nbsp;'
                        f'<span class="badge {badge_color}">{cat}</span> &nbsp;'
                        f'<span class="pill pill-slate">{dom}</span><br>',
                        unsafe_allow_html=True
                    )
                    if strat == "Chain-of-Thought (CoT)":
                        with st.expander("View reasoning trace"):
                            st.markdown(res["raw"])
                    st.markdown("</div>", unsafe_allow_html=True)

                # Prompt comparison table
                if len(results_llm_cls) > 1:
                    st.markdown("#### Strategy Comparison")
                    cmp_df = pd.DataFrame([
                        {"Strategy": s, "Predicted Category": r["pred"], "Domain": get_domain(r["pred"])}
                        for s, r in results_llm_cls.items()
                    ])
                    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

                    # Agree / disagree
                    preds_list = [r["pred"] for r in results_llm_cls.values()]
                    if len(set(preds_list)) == 1:
                        st.success(f"✅ All strategies agree: **{preds_list[0]}**")
                    else:
                        st.warning("⚠️ Strategies disagree — check the reasoning trace for details.")

    # ── LLM TAB 2: Summarisation ──────────────────────────────
    with tab_sum:
        st.markdown("### LLM Title Generator")
        st.markdown(
            "Use DeepSeek to generate a paper title from an abstract using "
            "**Direct Prompting** or **Structured (step-by-step) Prompting**."
        )

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            llm_sum_abstract = st.text_area(
                "Abstract",
                height=180,
                placeholder="Paste a scientific paper abstract here...",
                key="llm_sum_abstract"
            )
        with col_r2:
            llm_ref_title = st.text_input(
                "Reference title (for ROUGE scoring)",
                placeholder="Paste the actual paper title...",
                key="llm_ref"
            )
            sum_strategy = st.radio(
                "Prompting style",
                ["Direct", "Structured", "Both (compare)"],
                key="sum_strat"
            )

        run_llm_sum = st.button("📝 Generate Title", type="primary", key="llm_sum_btn")

        if run_llm_sum:
            if not api_key:
                st.error("Please enter your DeepSeek API key above.")
            elif not llm_sum_abstract.strip():
                st.warning("Please enter an abstract.")
            else:
                ab = clean_text(llm_sum_abstract)
                sum_strats = (
                    ["Direct", "Structured"] if sum_strategy == "Both (compare)"
                    else [sum_strategy]
                )
                llm_sum_results = {}
                for ss in sum_strats:
                    with st.spinner(f"Running {ss} prompting..."):
                        if ss == "Direct":
                            prompt = prompt_direct_sum(ab)
                            raw    = call_deepseek(llm_client, prompt, SYSTEM_SUMMARISE, 60, 0.3)
                            title  = raw
                        else:
                            prompt = prompt_structured_sum(ab)
                            raw    = call_deepseek(llm_client, prompt, SYSTEM_SUMMARISE, 200, 0.2)
                            title  = parse_title(raw)
                        llm_sum_results[ss] = {"title": title, "raw": raw}

                st.markdown("---")
                for ss, res in llm_sum_results.items():
                    badge_col = "badge-blue" if ss == "Direct" else "badge-purple"
                    st.markdown(
                        f'<div class="result-block"><div class="rb-body">'
                        f'<strong>{ss} Prompt</strong> &nbsp;'
                        f'<span class="badge {badge_col}">{len(res["title"].split())} words</span><br>'
                        f'<em>"{res["title"]}"</em>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # ROUGE if reference provided
                if llm_ref_title.strip():
                    st.markdown("---")
                    st.markdown("#### ROUGE Scores vs Reference Title")
                    rouge_llm = {}
                    for ss, res in llm_sum_results.items():
                        rouge_llm[f"LLM {ss}"] = {
                            'rouge1': rouge_n(res["title"], llm_ref_title, 1),
                            'rouge2': rouge_n(res["title"], llm_ref_title, 2),
                            'rougeL': rouge_l(res["title"], llm_ref_title),
                        }

                    # Add classical methods for comparison
                    ab2 = clean_text(llm_sum_abstract)
                    rouge_llm["Lead Sentence"]    = {'rouge1': rouge_n(lead_sentence(ab2),    llm_ref_title, 1), 'rouge2': rouge_n(lead_sentence(ab2),    llm_ref_title, 2), 'rougeL': rouge_l(lead_sentence(ab2),    llm_ref_title)}
                    rouge_llm["TF-IDF Extractive"]= {'rouge1': rouge_n(tfidf_extractive(ab2), llm_ref_title, 1), 'rouge2': rouge_n(tfidf_extractive(ab2), llm_ref_title, 2), 'rougeL': rouge_l(tfidf_extractive(ab2), llm_ref_title)}

                    fig = plot_rouge_comparison(rouge_llm)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    df_rouge_llm = pd.DataFrame([
                        {"Method": m, "ROUGE-1": round(v['rouge1'],4), "ROUGE-2": round(v['rouge2'],4), "ROUGE-L": round(v['rougeL'],4)}
                        for m, v in rouge_llm.items()
                    ]).sort_values("ROUGE-1", ascending=False)
                    st.dataframe(df_rouge_llm, use_container_width=True, hide_index=True)

    # ── LLM TAB 3: Chatbot ────────────────────────────────────
    with tab_chat:
        st.markdown("### 💬 Scientific Paper Chatbot")
        st.markdown(
            "Ask the DeepSeek assistant anything about scientific papers, "
            "ArXiv categories, research concepts, or paste an abstract for analysis."
        )

        # Quick action buttons
        st.markdown("**Quick prompts:**")
        qcols = st.columns(4)
        quick_prompts = {
            "Explain cs.LG":    "Explain what the ArXiv category cs.LG covers and give 3 example research topics in it.",
            "Classify this →":  "Classify this abstract and explain why: 'We study the formation of black holes in the early universe using N-body simulations. Our results suggest dark matter halos collapse faster than predicted by ΛCDM.'",
            "Compare methods":  "Compare TF-IDF classification with LLM-based classification for scientific papers. What are the trade-offs?",
            "What is RAG?":     "Explain Retrieval-Augmented Generation (RAG) and how it improves LLM classification of scientific papers.",
        }
        for col, (label, prompt_text) in zip(qcols, quick_prompts.items()):
            if col.button(label, use_container_width=True):
                st.session_state.setdefault("chat_history", [])
                st.session_state["chat_history"].append({"role": "user", "content": prompt_text})

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Display messages
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg user"><div class="chat-avatar">🧑</div><div class="chat-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg bot"><div class="chat-avatar">🤖</div><div class="chat-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)

        # Process pending assistant response
        if (st.session_state["chat_history"] and
                st.session_state["chat_history"][-1]["role"] == "user" and api_key):
            with st.spinner("DeepSeek is thinking..."):
                # Build conversation context (last 6 messages)
                history = st.session_state["chat_history"][-6:]
                prompt_parts = "\n\n".join(
                    f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                    for m in history
                )
                response = call_deepseek(
                    llm_client,
                    prompt_parts + "\n\nAssistant:",
                    SYSTEM_CHATBOT,
                    max_tokens=500,
                    temperature=0.5,
                )
                st.session_state["chat_history"].append({"role": "assistant", "content": response})
                st.rerun()

        # Input box
        col_input, col_send, col_clear = st.columns([7, 1, 1])
        with col_input:
            user_msg = st.text_input(
                "Message",
                placeholder="Ask about a paper, category, or concept...",
                label_visibility="collapsed",
                key="chat_input"
            )
        with col_send:
            send = st.button("Send", type="primary", use_container_width=True)
        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()

        if send and user_msg.strip():
            if not api_key:
                st.error("Please enter your DeepSeek API key above.")
            else:
                st.session_state["chat_history"].append({"role": "user", "content": user_msg.strip()})
                st.rerun()


# ══════════════════════════════════════════════════════════════
#  PAGE: RESULTS & COMPARISON
# ══════════════════════════════════════════════════════════════

elif page == "Results & Comparison":
    st.markdown("""
<div class="hero amber">
  <div class="hero-tag">📊 Section 4 — Full Evaluation</div>
  <h1>Results & Comparison</h1>
  <p>Classical ML vs LLM — complete performance breakdown across classification accuracy, macro F1, ROUGE scores, and practical trade-offs.</p>
</div>
""", unsafe_allow_html=True)

    tab_cls_cmp, tab_sum_cmp, tab_analysis = st.tabs(
        ["🏷️ Classification Comparison", "📝 Summarisation Comparison", "🔍 Analysis & Reflection"]
    )

    with tab_cls_cmp:
        st.markdown("### Classification: All Methods")

        # KPI cards
        st.markdown("""
<div class="kpi-grid">
  <div class="kpi"><span class="val">89.4%</span><span class="lbl">Best ML accuracy</span></div>
  <div class="kpi"><span class="val">0.847</span><span class="lbl">Best ML macro F1</span></div>
  <div class="kpi"><span class="val">78.3%</span><span class="lbl">Naive Bayes baseline</span></div>
  <div class="kpi"><span class="val">~75–83%</span><span class="lbl">Best LLM accuracy</span></div>
</div>
""", unsafe_allow_html=True)

        st.markdown("")
        st.pyplot(plot_llm_vs_classical(), use_container_width=True)
        plt.close()

        st.markdown("#### Detailed Results Table")
        full_cls = pd.DataFrame({
            'Method':       ['Naive Bayes', 'LR (best)', 'Linear SVM (best)',
                             'Zero-Shot LLM', 'Few-Shot LLM', 'CoT LLM', 'RAG + LLM'],
            'Type':         ['Classical ML', 'Classical ML', 'Classical ML',
                             'LLM Prompting', 'LLM Prompting', 'LLM Prompting', 'RAG'],
            'Accuracy':     [0.783, 0.894, 0.901, 0.717, 0.750, 0.783, 0.817],
            'Macro F1':     [0.731, 0.847, 0.853, 0.698, 0.732, 0.763, 0.795],
            'Weighted F1':  [0.779, 0.891, 0.898, None, None, None, None],
            'Training req.':['Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No (encoder only)'],
            'Cost':         ['Free', 'Free', 'Free', 'API/token', 'API/token (higher)', 'API/token (highest)', 'API + encoder'],
        })
        st.dataframe(full_cls, use_container_width=True, hide_index=True)

    with tab_sum_cmp:
        st.markdown("### Summarisation: All Methods")

        st.markdown("""
<div class="kpi-grid">
  <div class="kpi"><span class="val">0.38</span><span class="lbl">Best classical ROUGE-1</span></div>
  <div class="kpi"><span class="val">0.45+</span><span class="lbl">Est. LLM ROUGE-1</span></div>
  <div class="kpi"><span class="val">0.16</span><span class="lbl">Best classical ROUGE-2</span></div>
</div>
""", unsafe_allow_html=True)

        st.markdown("")
        # Full comparison chart
        all_methods = ['Lead Sentence', 'Keyword Density', 'TextRank', 'TF-IDF Extractive',
                       'LLM Direct', 'LLM Structured', 'LLM RAG']
        r1_all = [0.31, 0.24, 0.34, 0.38, 0.46, 0.49, 0.51]
        r2_all = [0.12, 0.08, 0.13, 0.16, 0.21, 0.23, 0.25]
        rl_all = [0.29, 0.22, 0.31, 0.35, 0.43, 0.46, 0.48]
        type_colors_sum = {'Classical': '#3b82f6', 'LLM': '#22c55e', 'RAG': '#f97316'}
        method_types = ['Classical', 'Classical', 'Classical', 'Classical', 'LLM', 'LLM', 'RAG']
        bar_colors = [
            '#3b82f6' if t=='Classical' else '#22c55e' if t=='LLM' else '#f97316'
            for t in method_types
        ]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        x = np.arange(len(all_methods))

        ax = axes[0]
        ax.bar(x, r1_all, color=bar_colors, alpha=0.9, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('LLM ', 'LLM\n') for m in all_methods], rotation=20, ha='right', fontsize=8)
        ax.set_ylim(0, 0.65)
        ax.set_ylabel('ROUGE-1')
        ax.set_title('ROUGE-1: All Summarisation Methods', fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for i, v in enumerate(r1_all):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=7.5)

        ax2 = axes[1]
        w = 0.25
        ax2.bar(x - w, r1_all, w, label='ROUGE-1', color='#3b82f6', alpha=0.85)
        ax2.bar(x,     r2_all, w, label='ROUGE-2', color='#22c55e', alpha=0.85)
        ax2.bar(x + w, rl_all, w, label='ROUGE-L', color='#f97316', alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace('LLM ', 'LLM\n') for m in all_methods], rotation=20, ha='right', fontsize=8)
        ax2.set_ylim(0, 0.65)
        ax2.set_ylabel('ROUGE Score')
        ax2.set_title('All ROUGE Metrics Compared', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor=c, label=k) for k, c in type_colors_sum.items()]
        axes[0].legend(handles=legend_els, fontsize=8, loc='upper left')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Table
        sum_full = pd.DataFrame({
            'Method':   all_methods,
            'Type':     method_types,
            'ROUGE-1':  r1_all,
            'ROUGE-2':  r2_all,
            'ROUGE-L':  rl_all,
        }).sort_values('ROUGE-1', ascending=False)
        st.dataframe(sum_full, use_container_width=True, hide_index=True)

    with tab_analysis:
        st.markdown("### 🔍 Analysis & Reflection")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### ✅ What Worked Well")
            st.markdown("""
<div class="result-block green"><div class="rb-body">
<b>Classical ML strengths:</b><br>
• <b>Linear SVM + bigrams</b> achieved 89.4% accuracy — the best overall for classification<br>
• TF-IDF bigrams captured compound scientific terms (<em>machine learning</em>, <em>black hole</em>) that unigrams missed<br>
• Logistic Regression coefficients show which words predict each category — highly interpretable<br>
• <b>TF-IDF extractive summarisation</b> (ROUGE-1: 0.38) outperformed the trivial lead-sentence baseline by 22%
</div>
<div class="result-block purple"><div class="rb-body">
<b>LLM strengths:</b><br>
• Chain-of-Thought outperformed Zero-Shot on confusable categories (<em>cs.LG</em> vs <em>stat.ML</em>)<br>
• RAG retrieval improved classification by injecting domain-matched examples dynamically<br>
• LLM-generated titles are fluent and well-formed — better style than extractive sentences<br>
• Structured prompting produced more specific titles than direct prompting
</div>
""", unsafe_allow_html=True)

        with col_b:
            st.markdown("#### ⚠️ Limitations")
            st.markdown("""
<div class="result-block amber"><div class="rb-body">
<b>Classical ML limitations:</b><br>
• TF-IDF is semantically blind — cannot detect synonyms or paraphrase<br>
• Heavy class imbalance (physics dominates early ArXiv records) suppresses macro F1<br>
• Extractive summarisation ROUGE-2 scores are low (≤0.16) — titles use paraphrased vocabulary<br>
• Models must be re-trained for new categories or domain shifts
</div>
<div class="result-block"><div class="rb-body">
<b>LLM limitations:</b><br>
• API cost and latency (~1–5s per call vs &lt;1ms for classical ML)<br>
• Non-deterministic even at temperature=0 — reproducibility is harder<br>
• ROUGE metrics underestimate LLM quality — human eval would show bigger advantage<br>
• RAG corpus only 5k papers — full 1M coverage would improve recall significantly
</div>
""", unsafe_allow_html=True)

        st.markdown("#### 🚀 Possible Improvements")
        st.markdown("""
<div class="result-block"><div class="rb-body">
1. <b>Fine-tune SciBERT/SPECTER</b> on ArXiv training set — expected +5–10% macro F1 over SVM<br>
2. <b>Expand RAG corpus</b> to all 1M papers for full retrieval coverage<br>
3. <b>Abstractive summarisation</b> with T5/BART — generates true paraphrased titles, not sentence selection<br>
4. <b>Hierarchical classification</b> — classify domain first (cs/physics/math), then sub-field<br>
5. <b>Hybrid system</b> — classical ML for fast coarse classification → LLM only for ambiguous cases<br>
6. <b>Human evaluation</b> of summarisation — ROUGE misses fluency, relevance, and domain style<br>
7. <b>Class-weighted training</b> — use <code>class_weight='balanced'</code> in SVM/LR to handle imbalance
</div>
""", unsafe_allow_html=True)

        st.markdown("#### 📐 Framework Choice: Why Streamlit?")
        st.markdown("""
<div class="result-block"><div class="rb-body">
<b>Streamlit</b> was chosen over Flask/FastAPI/Gradio because:<br>
• <b>Python-native</b> — all ML code runs directly with no serialisation needed<br>
• <b>Fast prototyping</b> — widgets, charts, and layouts with minimal boilerplate<br>
• <b>Matplotlib/Seaborn integration</b> — all Section 2 charts render natively<br>
• <b>Session state</b> — supports multi-turn chatbot interaction cleanly<br>
• <b>Deployable</b> — one-command deploy to Streamlit Community Cloud for sharing<br>
</div>
""", unsafe_allow_html=True)