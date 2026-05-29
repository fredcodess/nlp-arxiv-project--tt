import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import time
import os
from collections import Counter
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="DG4NLP · ArXiv NLP",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── API KEY FROM ENVIRONMENT / STREAMLIT SECRETS ────────────
load_dotenv()

headers = {
    "authorization": st.secrets["DEEPSEEK_API_KEY"],
    "content-type": "application/json"
}

def get_api_key():
    try:
        return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        return os.environ.get("DEEPSEEK_API_KEY", "")

DEEPSEEK_API_KEY = get_api_key()

# ─── GLOBAL CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"], [data-testid="stAppViewContainer"],
[data-testid="stMain"] { font-family: 'Inter', sans-serif !important; }

/* ── full dark background ── */
[data-testid="stAppViewContainer"] { background: #030712 !important; }
[data-testid="stMain"]             { background: transparent !important; }
.main .block-container { padding: 0 2rem 4rem !important; max-width: 1400px; }

/* ── hide sidebar toggle & default header ── */
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer, header        { visibility: hidden; }
[data-testid="stDecoration"]     { display: none; }
[data-testid="stHeader"]         { background: transparent !important; }

/* ══════════════════════════════════════════
   TOP NAVIGATION BAR
══════════════════════════════════════════ */
.topnav {
    position: sticky;
    top: 0;
    z-index: 9999;
    background: rgba(3, 7, 18, 0.92);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(99,102,241,0.15);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2rem;
    height: 60px;
    margin-bottom: 32px;
}
.topnav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    text-decoration: none;
}
.topnav-brand .brand-icon { font-size: 1.4rem; }
.topnav-brand .brand-text {
    font-size: 1rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.01em;
}
.topnav-brand .brand-sub {
    font-size: 0.7rem;
    color: #475569;
    font-weight: 400;
    margin-left: 2px;
}
.topnav-links {
    display: flex;
    align-items: center;
    gap: 4px;
}
.nav-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 16px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.18s;
    border: 1px solid transparent;
    text-decoration: none;
    white-space: nowrap;
}
.nav-link:hover { color: #e2e8f0; background: rgba(99,102,241,0.1); }
.nav-link.active {
    color: #c4b5fd;
    background: rgba(99,102,241,0.18);
    border-color: rgba(99,102,241,0.3);
    font-weight: 600;
}
.topnav-badge {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}

/* ══════════════════════════════════════════
   PAGE CONTENT
══════════════════════════════════════════ */
h1,h2,h3,h4,h5,h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color:#f1f5f9 !important; }
p, li, span, td, th,
[data-testid="stMarkdownContainer"] p { color:#94a3b8 !important; }
label, .stTextInput label, .stTextArea label,
.stSelectbox label, .stRadio label {
    color:#94a3b8 !important; font-size:0.78rem !important;
    font-weight:600 !important; text-transform:uppercase !important;
    letter-spacing:0.06em !important;
}

/* ── inputs ── */
.stTextInput input, .stTextArea textarea,
.stSelectbox > div > div {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 9px !important; font-weight: 600 !important;
    font-size: 0.86rem !important; padding: 10px 22px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(79,70,229,0.3) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 22px rgba(79,70,229,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,23,42,0.7) !important;
    border-radius: 12px !important; padding: 4px !important;
    gap: 2px !important; border: 1px solid rgba(99,102,241,0.12) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 9px !important;
    color: #64748b !important; font-weight: 500 !important;
    font-size: 0.87rem !important; padding: 8px 18px !important;
    border: none !important; transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #a5b4fc !important; }
.stTabs [aria-selected="true"] {
    background: rgba(79,70,229,0.25) !important;
    color: #c4b5fd !important; font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important; overflow: hidden !important;
}
[data-testid="stDataFrame"] thead tr th {
    background: rgba(79,70,229,0.12) !important;
    color: #a5b4fc !important; font-size: 0.78rem !important;
    text-transform: uppercase !important; letter-spacing: 0.05em !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) { background: rgba(15,23,42,0.4) !important; }
[data-testid="stDataFrame"] tbody tr:hover { background: rgba(79,70,229,0.06) !important; }
[data-testid="stDataFrame"] td { color: #cbd5e1 !important; font-size: 0.85rem !important; }

/* ── expander ── */
.streamlit-expanderHeader {
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important; color: #a5b4fc !important;
}
.streamlit-expanderContent {
    background: rgba(8,12,24,0.9) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important;
}

/* ── alerts ── */
[data-testid="stAlert"][data-type="success"] {
    background: rgba(16,185,129,0.08) !important;
    border: 1px solid rgba(16,185,129,0.3) !important; color: #6ee7b7 !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"][data-type="warning"] {
    background: rgba(245,158,11,0.08) !important;
    border: 1px solid rgba(245,158,11,0.3) !important; color: #fcd34d !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"][data-type="error"] {
    background: rgba(239,68,68,0.08) !important;
    border: 1px solid rgba(239,68,68,0.3) !important; color: #fca5a5 !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"][data-type="info"] {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.3) !important; color: #a5b4fc !important;
    border-radius: 10px !important;
}

/* ══════════════════════════════════════════
   REUSABLE COMPONENTS
══════════════════════════════════════════ */

/* hero */
.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #0c0f1e 0%, #151030 50%, #0c0f1e 100%);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 18px; padding: 40px 44px; margin-bottom: 28px;
}
.hero::before {
    content:''; position:absolute; top:-100px; right:-100px;
    width:320px; height:320px;
    background: radial-gradient(circle, rgba(124,58,237,0.16) 0%, transparent 70%);
    border-radius:50%; pointer-events:none;
}
.hero::after {
    content:''; position:absolute; bottom:-80px; left:-80px;
    width:240px; height:240px;
    background: radial-gradient(circle, rgba(79,70,229,0.1) 0%, transparent 70%);
    border-radius:50%; pointer-events:none;
}
.hero-tag {
    display:inline-block;
    background: rgba(99,102,241,0.14); border: 1px solid rgba(99,102,241,0.32);
    border-radius:20px; padding:4px 14px; font-size:0.71rem;
    font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
    color:#818cf8; margin-bottom:14px;
}
.hero h1 {
    font-size:2.1rem !important; font-weight:800 !important;
    color:#f8fafc !important; margin:0 0 10px !important;
    line-height:1.15 !important; letter-spacing:-0.025em !important;
}
.hero p { font-size:0.96rem !important; color:#94a3b8 !important; margin:0 !important; line-height:1.65 !important; }
.hero.teal { background: linear-gradient(135deg,#041914,#0a2d22,#041914); border-color:rgba(20,184,166,0.22); }
.hero.teal::before { background:radial-gradient(circle,rgba(20,184,166,0.14) 0%,transparent 70%); }
.hero.teal .hero-tag { background:rgba(20,184,166,0.1); border-color:rgba(20,184,166,0.32); color:#5eead4; }
.hero.amber { background: linear-gradient(135deg,#16100a,#261a06,#16100a); border-color:rgba(245,158,11,0.2); }
.hero.amber .hero-tag { background:rgba(245,158,11,0.1); border-color:rgba(245,158,11,0.3); color:#fbbf24; }
.hero.rose { background: linear-gradient(135deg,#160a12,#2d0f1f,#160a12); border-color:rgba(244,63,94,0.2); }
.hero.rose::before { background:radial-gradient(circle,rgba(244,63,94,0.12) 0%,transparent 70%); }
.hero.rose .hero-tag { background:rgba(244,63,94,0.1); border-color:rgba(244,63,94,0.3); color:#fb7185; }

/* kpi grid */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:14px; margin:20px 0 28px; }
.kpi {
    background:rgba(15,23,42,0.7); border:1px solid rgba(99,102,241,0.16);
    border-radius:14px; padding:20px 18px; text-align:center;
    transition:border-color 0.2s,transform 0.2s; backdrop-filter:blur(8px);
}
.kpi:hover { border-color:rgba(99,102,241,0.4); transform:translateY(-2px); }
.kpi .val {
    font-size:1.9rem; font-weight:800;
    background:linear-gradient(135deg,#818cf8,#c4b5fd);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; display:block; line-height:1.1;
}
.kpi .lbl { font-size:0.7rem; color:#475569; font-weight:600; text-transform:uppercase; letter-spacing:0.07em; margin-top:7px; display:block; }

/* glass card */
.glass-card {
    background:rgba(15,23,42,0.65); border:1px solid rgba(99,102,241,0.16);
    border-radius:14px; padding:20px 22px; margin:10px 0;
    backdrop-filter:blur(10px); transition:border-color 0.2s;
}
.glass-card:hover { border-color:rgba(99,102,241,0.32); }
.glass-card .card-title {
    font-size:0.76rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.07em; color:#818cf8; margin-bottom:10px;
}
.glass-card .card-body { font-size:0.9rem; color:#94a3b8; line-height:1.75; }
.glass-card.green  { border-color:rgba(16,185,129,0.2); }
.glass-card.green .card-title  { color:#34d399; }
.glass-card.teal   { border-color:rgba(20,184,166,0.2); }
.glass-card.teal .card-title   { color:#5eead4; }
.glass-card.amber  { border-color:rgba(245,158,11,0.2); }
.glass-card.amber .card-title  { color:#fbbf24; }
.glass-card.red    { border-color:rgba(239,68,68,0.2); }
.glass-card.red .card-title    { color:#f87171; }

/* result block */
.result-block {
    background:rgba(8,12,24,0.9); border:1px solid rgba(99,102,241,0.2);
    border-radius:12px; padding:16px 20px; margin:10px 0;
    position:relative; overflow:hidden;
}
.result-block::before {
    content:''; position:absolute; left:0; top:0; bottom:0;
    width:3px; background:linear-gradient(180deg,#818cf8,#c4b5fd); border-radius:3px 0 0 3px;
}
.result-block.green::before  { background:linear-gradient(180deg,#34d399,#6ee7b7); }
.result-block.amber::before  { background:linear-gradient(180deg,#fbbf24,#fde68a); }
.result-block.purple::before { background:linear-gradient(180deg,#c084fc,#e879f9); }
.result-block .rb-title { font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:#818cf8; margin-bottom:8px; }
.result-block.green .rb-title  { color:#34d399; }
.result-block.amber .rb-title  { color:#fbbf24; }
.result-block.purple .rb-title { color:#c084fc; }
.result-block .rb-body { font-size:0.92rem; color:#cbd5e1; line-height:1.7; }

/* pills */
.pill { display:inline-flex; align-items:center; gap:4px; padding:3px 11px; border-radius:20px; font-size:0.72rem; font-weight:600; letter-spacing:0.03em; margin:2px 3px; }
.pill-indigo { background:rgba(99,102,241,0.14); color:#818cf8; border:1px solid rgba(99,102,241,0.28); }
.pill-green  { background:rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.28); }
.pill-amber  { background:rgba(245,158,11,0.12); color:#fbbf24; border:1px solid rgba(245,158,11,0.28); }
.pill-purple { background:rgba(168,85,247,0.12); color:#c084fc; border:1px solid rgba(168,85,247,0.28); }
.pill-slate  { background:rgba(71,85,105,0.22);  color:#94a3b8; border:1px solid rgba(71,85,105,0.38); }
.pill-teal   { background:rgba(20,184,166,0.12); color:#5eead4; border:1px solid rgba(20,184,166,0.28); }

/* divider */
.sec-divider { height:1px; background:linear-gradient(90deg,transparent,rgba(99,102,241,0.25),transparent); margin:28px 0; }

/* chat */
.chat-msg { display:flex; align-items:flex-end; gap:10px; margin-bottom:12px; }
.chat-msg.user { flex-direction:row-reverse; }
.chat-avatar { width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0;background:rgba(15,23,42,0.8);border:1px solid rgba(99,102,241,0.28); }
.chat-bubble { max-width:72%; padding:11px 16px; border-radius:16px; font-size:0.88rem; line-height:1.65; }
.chat-msg.user .chat-bubble { background:linear-gradient(135deg,#4f46e5,#7c3aed); color:#fff; border-radius:16px 4px 16px 16px; }
.chat-msg.bot  .chat-bubble { background:rgba(15,23,42,0.85); color:#cbd5e1; border:1px solid rgba(99,102,241,0.16); border-radius:4px 16px 16px 16px; }

/* about page cards */
.about-stat { text-align:center; padding:28px 20px; background:rgba(15,23,42,0.65); border:1px solid rgba(99,102,241,0.16); border-radius:16px; }
.about-stat .big { font-size:2.8rem; font-weight:800; background:linear-gradient(135deg,#818cf8,#c4b5fd); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.about-stat .sm  { font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.07em; font-weight:600; margin-top:6px; }

/* timeline */
.timeline { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.tl-item { background:rgba(15,23,42,0.65); border:1px solid rgba(99,102,241,0.14); border-radius:12px; padding:16px 18px; }
.tl-week { font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#818cf8; margin-bottom:6px; }
.tl-title { font-size:0.92rem; font-weight:600; color:#e2e8f0; margin-bottom:6px; }
.tl-body  { font-size:0.83rem; color:#64748b; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  MATPLOTLIB DARK THEME
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#1e293b", "axes.labelcolor": "#94a3b8",
    "axes.titlecolor": "#e2e8f0", "xtick.color": "#64748b",
    "ytick.color": "#64748b", "text.color": "#94a3b8",
    "grid.color": "#1e293b", "grid.alpha": 0.6, "figure.dpi": 110,
    "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
})

# ══════════════════════════════════════════════════════════════
#  SESSION STATE — current page
# ══════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "classical"

# ── top navbar ───────────────────────────────────────────────
def nav(label, key, icon):
    active = "active" if st.session_state.page == key else ""
    return f'<a class="nav-link {active}" onclick="void(0)">{icon} {label}</a>'

# render navbar as HTML + use st.columns for clickable buttons
st.markdown("""
<div class="topnav">
  <div class="topnav-brand">
    <div>
      <span class="brand-text">ArXiv NLP System</span>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px">
    <span class="topnav-badge">Fredcodess</span>
  </div>
</div>
""", unsafe_allow_html=True)

# nav buttons via Streamlit columns (the only reliable click handler)
n1, n2, n3, n4, n5, n6 = st.columns([1,1,1,1,1,3])
with n1:
    if st.button("Classical ML",  use_container_width=True,
                 type="primary" if st.session_state.page=="classical" else "secondary"):
        st.session_state.page = "classical"; st.rerun()
with n2:
    if st.button("LLM",           use_container_width=True,
                 type="primary" if st.session_state.page=="llm" else "secondary"):
        st.session_state.page = "llm"; st.rerun()
with n3:
    if st.button("Results",        use_container_width=True,
                 type="primary" if st.session_state.page=="results" else "secondary"):
        st.session_state.page = "results"; st.rerun()
with n4:
    if st.button("About",          use_container_width=True,
                 type="primary" if st.session_state.page=="about" else "secondary"):
        st.session_state.page = "about"; st.rerun()

st.markdown('<div style="margin-top:8px"></div>', unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════
VALID_CATEGORIES = {
    "cs.AI","cs.AR","cs.CC","cs.CE","cs.CG","cs.CL","cs.CR","cs.CV","cs.CY","cs.DB",
    "cs.DC","cs.DL","cs.DM","cs.DS","cs.ET","cs.FL","cs.GL","cs.GR","cs.GT","cs.HC",
    "cs.IR","cs.IT","cs.LG","cs.LO","cs.MA","cs.MM","cs.MS","cs.NA","cs.NE","cs.NI",
    "cs.OH","cs.OS","cs.PF","cs.PL","cs.RO","cs.SC","cs.SD","cs.SE","cs.SI","cs.SY",
    "econ.EM","econ.GN","econ.TH","eess.AS","eess.IV","eess.SP","eess.SY",
    "math.AC","math.AG","math.AP","math.AT","math.CA","math.CO","math.CT","math.CV",
    "math.DG","math.DS","math.FA","math.GM","math.GN","math.GR","math.GT","math.HO",
    "math.IT","math.KT","math.LO","math.MG","math.MP","math.NA","math.NT","math.OA",
    "math.OC","math.PR","math.QA","math.RA","math.RT","math.SG","math.SP","math.ST",
    "astro-ph.CO","astro-ph.EP","astro-ph.GA","astro-ph.HE","astro-ph.IM","astro-ph.SR",
    "cond-mat.dis-nn","cond-mat.mes-hall","cond-mat.mtrl-sci","cond-mat.other",
    "cond-mat.quant-gas","cond-mat.soft","cond-mat.stat-mech","cond-mat.str-el","cond-mat.supr-con",
    "gr-qc","hep-ex","hep-lat","hep-ph","hep-th","math-ph",
    "nlin.AO","nlin.CD","nlin.CG","nlin.PS","nlin.SI","nucl-ex","nucl-th",
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
    "cs":"Computer Science","math":"Mathematics","hep":"High Energy Physics",
    "cond-mat":"Condensed Matter","astro-ph":"Astrophysics","physics":"Physics",
    "quant-ph":"Quantum Physics","math-ph":"Mathematical Physics",
    "gr-qc":"General Relativity","q-bio":"Quantitative Biology",
    "q-fin":"Quantitative Finance","stat":"Statistics",
    "econ":"Economics","eess":"Elec. Eng. & Systems","nucl":"Nuclear Physics",
    "nlin":"Nonlinear Sciences",
}
def get_domain(cat):
    for prefix, label in DOMAIN_LABELS.items():
        if cat.startswith(prefix): return label
    return "Other"

def clean_text(text):
    text = str(text)
    text = re.sub(r'[\n\t\r]+', ' ', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()

def tokenise(text): return re.findall(r'\b[a-z]+\b', text.lower())

def rouge_n(hyp, ref, n):
    def ng(t,n): return Counter(tuple(t[i:i+n]) for i in range(len(t)-n+1))
    h,r = ng(tokenise(hyp),n), ng(tokenise(ref),n)
    ov = sum((h&r).values()); th,tr = sum(h.values()),sum(r.values())
    if not th or not tr: return 0.0
    p,rc = ov/th, ov/tr
    return 2*p*rc/(p+rc) if (p+rc) else 0.0

def rouge_l(hyp, ref):
    h,r = tokenise(hyp), tokenise(ref)
    if not h or not r: return 0.0
    m,n = len(r),len(h)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = dp[i-1][j-1]+1 if r[i-1]==h[j-1] else max(dp[i-1][j],dp[i][j-1])
    lcs=dp[m][n]; p,rc=lcs/n,lcs/m
    return 2*p*rc/(p+rc) if (p+rc) else 0.0

STOPWORDS = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
             'from','is','are','was','were','be','been','this','that','it','its','we',
             'our','they','their','which','also','such','both','some','any','all','more'}

CATEGORY_KEYWORDS = {
    "cs.LG":  ["learning","neural","network","training","model","gradient","deep","layer",
               "classification","prediction","supervised","reinforcement","optimization",
               "convolutional","attention","transformer","loss","epoch","batch"],
    "cs.CV":  ["image","vision","detection","segmentation","pixel","object","recognition",
               "visual","camera","feature","depth","pose","tracking","rendering","convolutional"],
    "cs.CL":  ["language","text","nlp","parsing","translation","corpus","word","sentence",
               "semantic","syntactic","discourse","dialogue","embedding","bert","gpt"],
    "cs.AI":  ["artificial","intelligence","agent","planning","reasoning","knowledge",
               "search","constraint","logic","inference","ontology","belief","expert"],
    "cs.RO":  ["robot","manipulation","motion","grasping","kinematics","trajectory","sensor",
               "autonomous","navigation","control","actuator","arm","drone"],
    "hep-th": ["string","gauge","field","bosonic","supersymmetry","holographic","conformal",
               "renormalization","gravity","quantum","theory","algebra","symmetry","amplitude"],
    "hep-ph": ["quark","hadron","lepton","boson","higgs","collider","decay","scattering",
               "parton","meson","baryon","electroweak","qcd","perturbative"],
    "quant-ph":["quantum","qubit","entanglement","coherence","decoherence","superposition",
                "measurement","hamiltonian","fidelity","channel","circuit","gate"],
    "math.CO": ["graph","combinatorial","polynomial","enumeration","coloring","chromatic",
                "tree","permutation","bijection","lattice","poset","matroid"],
    "math.AP": ["partial","differential","equation","boundary","solution","existence",
                "regularity","parabolic","elliptic","sobolev","operator","weak"],
    "stat.ML": ["bayesian","posterior","prior","inference","regression","lasso","ridge",
                "gaussian","process","distribution","estimator","likelihood","variational"],
    "astro-ph.CO":["cosmological","dark","matter","energy","galaxy","universe","expansion",
                   "redshift","cmb","inflation","baryon","halo"],
    "cond-mat.str-el":["electron","correlated","magnetic","superconducting","mott","hubbard",
                       "spin","orbital","transition","charge","phonon","band"],
    "q-bio.NC":["neuron","cortex","spike","hippocampus","synaptic","cognitive","brain",
                "firing","connectivity","oscillation","memory","perception"],
    "math.AG": ["algebraic","variety","scheme","cohomology","sheaf","morphism","moduli",
                "curve","surface","divisor","genus","abelian","projective"],
    "gr-qc":   ["gravitational","black","hole","spacetime","relativity","geodesic",
                "curvature","metric","einstein","singularity","horizon"],
}

def tfidf_classify(abstract):
    words = [w for w in tokenise(abstract) if w not in STOPWORDS and len(w)>2]
    wc = Counter(words); total = len(words) or 1
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        kw_set = set(keywords)
        overlap = sum(wc.get(w,0) for w in kw_set)
        tf = overlap / total
        idf = math.log(len(CATEGORY_KEYWORDS)/(1+sum(1 for k in CATEGORY_KEYWORDS.values() if any(w in k for w in kw_set)))+1)
        scores[cat] = tf*(1+idf)*100
    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    top = ranked[:8]; tot = sum(v for _,v in top) or 1
    return [(cat, round(v/tot*100,1)) for cat,v in top]

def lead_sentence(ab):
    sents = re.split(r'(?<=[.!?])\s+', ab.strip())
    return sents[0] if sents else ab

def tfidf_extractive(ab):
    sents = [s for s in re.split(r'(?<=[.!?])\s+', ab.strip()) if len(s.split())>=5]
    if not sents: return ab
    if len(sents)==1: return sents[0]
    all_w = [w for w in tokenise(ab) if w not in STOPWORDS and len(w)>2]
    idf = {}
    for w in set(all_w):
        df = sum(1 for s in sents if w in tokenise(s))
        idf[w] = math.log((len(sents)+1)/(df+1))
    def score(s):
        ws = [w for w in tokenise(s) if w not in STOPWORDS and len(w)>2]
        if not ws: return 0.0
        return sum((ws.count(w)/len(ws))*idf.get(w,0) for w in set(ws))/len(ws)
    return max(sents, key=score)

def textrank_summarise(ab):
    sents = [s for s in re.split(r'(?<=[.!?])\s+', ab.strip()) if len(s.split())>=4]
    if len(sents)<=2: return sents[0] if sents else ab
    def sim(a,b):
        wa,wb = set(tokenise(a)),set(tokenise(b))
        if not wa or not wb: return 0.0
        return len(wa&wb)/(math.log(len(wa)+1)+math.log(len(wb)+1))
    scores = [1.0]*len(sents)
    for _ in range(10):
        new=[0.15+0.85*sum(sim(sents[i],sents[j])*scores[j] for j in range(len(sents)) if j!=i) for i in range(len(sents))]
        scores=new
    return sents[scores.index(max(scores))]

def keyword_density(ab):
    sents = [s for s in re.split(r'(?<=[.!?])\s+', ab.strip()) if len(s.split())>=4]
    if not sents: return ab
    freq = Counter(w for w in tokenise(ab) if w not in STOPWORDS and len(w)>2)
    def score(s):
        ws=[w for w in tokenise(s) if w not in STOPWORDS and len(w)>2]
        return sum(freq[w] for w in ws)/len(ws) if ws else 0.0
    return max(sents, key=score)

# ── LLM client ────────────────────────────────────────────────
@st.cache_resource
def get_llm_client():
    try:
        from openai import OpenAI
        if not DEEPSEEK_API_KEY:
            return None
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    except ImportError:
        return None

def call_deepseek(prompt, system="", max_tokens=300, temperature=0.0):
    client = get_llm_client()
    if client is None:
        return "⚠️ DeepSeek client unavailable. Check the DEEPSEEK_API_KEY environment variable."
    messages = []
    if system: messages.append({"role":"system","content":system})
    messages.append({"role":"user","content":prompt})
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat", max_tokens=max_tokens,
                temperature=temperature, messages=messages)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                time.sleep(2**attempt)
            else:
                return f"⚠️ API error: {e}"
    return "⚠️ Rate limit exceeded — please try again shortly."

SYSTEM_CLASSIFY  = ("You are a scientific paper classifier. Output ONLY one ArXiv category code (e.g. cs.LG). No explanation.")
SYSTEM_SUMMARISE = ("You are a scientific editor. Write a concise paper title (10-15 words). Output ONLY the title.")
SYSTEM_CHATBOT   = ("You are a helpful scientific research assistant specialising in academic papers and the ArXiv repository. Be concise, accurate, and friendly.")

def prompt_zero_shot(ab): return f"Classify this abstract into one ArXiv category.\n\nAbstract:\n{ab}\n\nCategory:"
def prompt_few_shot(ab):
    ex=("Abstract: Novel deep learning architecture for image classification using attention.\nCategory: cs.CV\n\n"
        "Abstract: Quantum field theory partition function using supersymmetric methods.\nCategory: hep-th\n\n"
        "Abstract: Chromatic polynomial lower bound for planar graphs via algebraic topology.\nCategory: math.CO\n\n")
    return f"Examples:\n\n{ex}Now classify:\nAbstract: {ab}\nCategory:"
def prompt_cot(ab):
    return (f"Classify this abstract step by step.\n\nAbstract:\n{ab}\n\n"
            f"1. Key domain-specific terms?\n2. Scientific field/methodology?\n3. Best ArXiv category?\n\nEnd with:\nCATEGORY: <code>")
def parse_cat(r):
    r=r.strip()
    if r in VALID_CATEGORIES: return r
    m=re.search(r'CATEGORY:\s*([\w.\-]+)',r,re.IGNORECASE)
    if m and m.group(1) in VALID_CATEGORIES: return m.group(1)
    for c in sorted(VALID_CATEGORIES,key=len,reverse=True):
        if c in r: return c
    return r.split()[0] if r else "unknown"
def prompt_direct(ab): return f"Generate a concise paper title for:\n\nAbstract:\n{ab}\n\nTitle:"
def prompt_structured(ab):
    return (f"Abstract:\n{ab}\n\n1. Main contribution.\n2. Methodology.\n3. Write a 10-15 word title.\n\nEnd with:\nTitle: <title>")
def parse_title(r):
    m=re.search(r'Title:\s*(.+)',r,re.IGNORECASE)
    if m: return m.group(1).strip().strip('"').strip("'")
    lines=[l.strip() for l in r.split('\n') if l.strip()]
    return lines[-1] if lines else r

# ── chart helpers ─────────────────────────────────────────────
def plot_cls_bar(preds, title="Classification Confidence"):
    cats=[p[0] for p in preds[:8]]; scores=[p[1] for p in preds[:8]]
    colors=["#818cf8" if i==0 else "#312e81" for i in range(len(cats))]
    fig,ax=plt.subplots(figsize=(7,3.5),facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    bars=ax.barh(cats[::-1],scores[::-1],color=colors[::-1],edgecolor="#0d1117",height=0.62)
    for bar,val in zip(bars,scores[::-1]):
        ax.text(bar.get_width()+0.5,bar.get_y()+bar.get_height()/2,f"{val:.1f}%",va='center',fontsize=8,color='#94a3b8')
    ax.set_xlabel("Confidence (%)",fontsize=9); ax.set_xlim(0,max(scores)*1.28)
    ax.set_title(title,fontsize=10,fontweight='bold',pad=10); fig.tight_layout(); return fig

def plot_rouge_cmp(rouge_scores):
    methods=list(rouge_scores.keys())
    r1=[rouge_scores[m]['rouge1'] for m in methods]
    r2=[rouge_scores[m]['rouge2'] for m in methods]
    rl=[rouge_scores[m]['rougeL'] for m in methods]
    x=np.arange(len(methods)); w=0.25
    fig,ax=plt.subplots(figsize=(8,4),facecolor='#0d1117'); ax.set_facecolor('#0d1117')
    ax.bar(x-w,r1,w,label='ROUGE-1',color='#818cf8',alpha=0.9,edgecolor='#0d1117')
    ax.bar(x,  r2,w,label='ROUGE-2',color='#34d399',alpha=0.9,edgecolor='#0d1117')
    ax.bar(x+w,rl,w,label='ROUGE-L',color='#fbbf24',alpha=0.9,edgecolor='#0d1117')
    ax.set_xticks(x); ax.set_xticklabels([m.replace(' ','\n') for m in methods],fontsize=8)
    ax.set_ylabel('ROUGE Score',fontsize=9); ax.set_ylim(0,0.7)
    ax.set_title('Summarisation Quality — ROUGE Scores',fontsize=10,fontweight='bold',color='#e2e8f0')
    ax.legend(fontsize=8,facecolor='#0d1117',labelcolor='#94a3b8',edgecolor='#1e293b'); fig.tight_layout(); return fig

def plot_model_cmp():
    data={'Model':['Naive Bayes','LR (uni)','LR (bi)','SVM (uni)','SVM (bi)','SVM (title+ab)'],
          'Accuracy':[0.783,0.839,0.851,0.872,0.894,0.901],
          'Macro F1':[0.731,0.787,0.801,0.823,0.847,0.853],
          'Weighted F1':[0.779,0.835,0.849,0.869,0.891,0.898]}
    df=pd.DataFrame(data); x=np.arange(len(df)); w=0.27
    fig,ax=plt.subplots(figsize=(9,4.5),facecolor='#0d1117'); ax.set_facecolor('#0d1117')
    ax.bar(x-w,df['Accuracy'],  w,label='Accuracy',   color='#818cf8',alpha=0.9,edgecolor='#0d1117')
    ax.bar(x,  df['Macro F1'], w,label='Macro F1',   color='#34d399',alpha=0.9,edgecolor='#0d1117')
    ax.bar(x+w,df['Weighted F1'],w,label='Weighted F1',color='#fbbf24',alpha=0.9,edgecolor='#0d1117')
    ax.set_xticks(x); ax.set_xticklabels(df['Model'],rotation=20,ha='right',fontsize=8)
    ax.set_ylim(0.6,1.02); ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title('Classification Models — All Experiments',fontsize=10,fontweight='bold',color='#e2e8f0')
    ax.legend(fontsize=8,facecolor='#0d1117',labelcolor='#94a3b8',edgecolor='#1e293b')
    for bar in ax.patches:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,
                f'{bar.get_height()*100:.1f}',ha='center',fontsize=6.5,color='#94a3b8')
    fig.tight_layout(); return fig

def plot_sum_cmp():
    methods=['Lead Sentence','TextRank','TF-IDF Extractive','Keyword Density']
    r1=[0.31,0.34,0.38,0.24]; r2=[0.12,0.13,0.16,0.08]; rl=[0.29,0.31,0.35,0.22]
    x=np.arange(len(methods)); w=0.25
    fig,ax=plt.subplots(figsize=(8,4),facecolor='#0d1117'); ax.set_facecolor('#0d1117')
    ax.bar(x-w,r1,w,label='ROUGE-1',color='#818cf8',alpha=0.9,edgecolor='#0d1117')
    ax.bar(x,  r2,w,label='ROUGE-2',color='#34d399',alpha=0.9,edgecolor='#0d1117')
    ax.bar(x+w,rl,w,label='ROUGE-L',color='#fbbf24',alpha=0.9,edgecolor='#0d1117')
    ax.set_xticks(x); ax.set_xticklabels(methods,rotation=15,ha='right',fontsize=8)
    ax.set_ylim(0,0.5); ax.set_title('Summarisation Methods — ROUGE',fontsize=10,fontweight='bold',color='#e2e8f0')
    ax.legend(fontsize=8,facecolor='#0d1117',labelcolor='#94a3b8',edgecolor='#1e293b'); fig.tight_layout(); return fig

def plot_full_cmp():
    data={'Method':['Naive Bayes','LR (best)','SVM (best)','Zero-Shot','Few-Shot','CoT','RAG+LLM'],
          'Accuracy':[0.783,0.894,0.901,0.717,0.750,0.783,0.817],
          'Macro F1':[0.731,0.847,0.853,0.698,0.732,0.763,0.795],
          'Type':['Classical','Classical','Classical','LLM','LLM','LLM','RAG']}
    df=pd.DataFrame(data)
    color_map={'Classical':'#818cf8','LLM':'#34d399','RAG':'#fbbf24'}
    colors=[color_map[t] for t in df['Type']]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),facecolor='#0d1117')
    for ax,col,title in zip(axes,['Accuracy','Macro F1'],['Accuracy','Macro F1']):
        ax.set_facecolor('#0d1117')
        bars=ax.bar(df['Method'],df[col],color=colors,alpha=0.9,edgecolor='#0d1117')
        ax.set_xticklabels(df['Method'],rotation=22,ha='right',fontsize=8)
        ax.set_ylim(0.6,1.0); ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(f'{title} — Classical vs LLM',fontsize=10,fontweight='bold',color='#e2e8f0')
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,
                    f'{bar.get_height()*100:.1f}',ha='center',fontsize=7.5,color='#94a3b8')
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(facecolor=c,label=k) for k,c in color_map.items()],
                   fontsize=8,loc='lower right',facecolor='#0d1117',labelcolor='#94a3b8',edgecolor='#1e293b')
    fig.tight_layout(); return fig


# ══════════════════════════════════════════════════════════════
#  PAGE: CLASSICAL ML
# ══════════════════════════════════════════════════════════════
if page == "classical":
    st.markdown("""
<div class="hero">
  <div class="hero-tag">⚙️ Section 2 — Classical Machine Learning</div>
  <h1>Classical ML Pipeline</h1>
  <p>TF-IDF feature extraction · Logistic Regression · Naive Bayes · Linear SVM · Extractive Summarisation · ROUGE Evaluation</p>
</div>""", unsafe_allow_html=True)

    tab_cls, tab_sum, tab_perf = st.tabs(["🏷️  Classifier", "📝  Summariser", "📊  Performance"])

    # ── Classifier tab ────────────────────────────────────────
    with tab_cls:
        st.markdown("### Paper Category Classifier")
        st.markdown("Paste an abstract and the TF-IDF heuristic classifier scores it against all ArXiv categories.")

        samples = {
            "— type your own —": "",
            "Deep Learning (cs.LG)": "We propose a novel attention-based transformer architecture for few-shot image classification. Our model achieves state-of-the-art on miniImageNet by combining self-supervised pre-training with meta-learning, improving accuracy by 3.2% over the previous best.",
            "Quantum Physics (quant-ph)": "We investigate the entanglement entropy of a bipartite quantum system undergoing unitary evolution. Using random matrix theory we derive exact analytical expressions for the average entanglement entropy as a function of time and system size.",
            "Graph Theory (math.CO)": "We prove a new lower bound for the chromatic number of Kneser graphs using algebraic topology. Our proof generalises the Lovász method and applies to a broader family of vertex-critical graphs.",
            "NLP (cs.CL)": "We introduce a cross-lingual pre-trained language model trained on 100 languages simultaneously using a shared vocabulary. Fine-tuning achieves competitive performance with monolingual models while requiring no target language data.",
        }
        choice = st.selectbox("Load a sample", list(samples.keys()))
        user_ab = st.text_area("Abstract", value=samples[choice], height=150, placeholder="Paste a scientific paper abstract here...", label_visibility="visible")

        c1,c2 = st.columns([1,5])
        with c1: run_cls = st.button("🔍 Classify", type="primary", use_container_width=True)
        with c2:
            if user_ab:
                wc=len(user_ab.split())
                st.markdown(f'<span class="pill pill-slate">{wc} words</span><span class="pill pill-indigo">{len(set(tokenise(user_ab)))} unique terms</span>', unsafe_allow_html=True)

        if run_cls and user_ab.strip():
            with st.spinner("Running TF-IDF classification..."):
                time.sleep(0.2)
                preds = tfidf_classify(clean_text(user_ab))
            top_cat, top_conf = preds[0]
            domain = get_domain(top_cat)
            st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
            m1,m2,m3 = st.columns(3)
            m1.markdown(f'<div class="kpi"><span class="val" style="font-size:1.5rem">{top_cat}</span><span class="lbl">Predicted Category</span></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="kpi"><span class="val">{top_conf:.1f}%</span><span class="lbl">Confidence</span></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="kpi"><span class="val" style="font-size:1.1rem">{domain}</span><span class="lbl">Domain</span></div>', unsafe_allow_html=True)
            st.markdown("")
            fig = plot_cls_bar(preds)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            df_p = pd.DataFrame(preds, columns=["Category","Confidence (%)"])
            df_p["Domain"] = df_p["Category"].apply(get_domain)
            st.dataframe(df_p, use_container_width=True, hide_index=True)
        elif run_cls:
            st.warning("Please enter an abstract first.")

    # ── Summariser tab ────────────────────────────────────────
    with tab_sum:
        st.markdown("### Extractive Summariser")
        st.markdown("Compare all four classical extractive methods side-by-side with optional live ROUGE scoring.")
        c_a, c_b = st.columns(2)
        with c_a:
            sum_ab = st.text_area("Abstract", height=180, placeholder="Paste a scientific paper abstract here...", key="sum_ab")
        with c_b:
            ref_title = st.text_input("Reference title (optional — for ROUGE)", placeholder="Paste the actual paper title...", key="ref_t")
        run_sum = st.button("📝 Summarise with All Methods", type="primary")

        if run_sum and sum_ab.strip():
            with st.spinner("Running all 4 methods..."):
                time.sleep(0.2)
                ab = clean_text(sum_ab)
                results = {
                    "Lead Sentence":     lead_sentence(ab),
                    "TF-IDF Extractive": tfidf_extractive(ab),
                    "TextRank":          textrank_summarise(ab),
                    "Keyword Density":   keyword_density(ab),
                }
            st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
            for i,(method,summary) in enumerate(results.items()):
                style = ["","green","amber","purple"][i]
                wl = len(summary.split())
                st.markdown(f'<div class="result-block {style}"><div class="rb-title">{method} <span class="pill pill-slate">{wl} words</span></div><div class="rb-body">{summary}</div></div>', unsafe_allow_html=True)

            if ref_title.strip():
                st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
                st.markdown("#### 📊 ROUGE Comparison vs Reference Title")
                rouge_scores = {m: {'rouge1':rouge_n(s,ref_title,1),'rouge2':rouge_n(s,ref_title,2),'rougeL':rouge_l(s,ref_title)} for m,s in results.items()}
                fig = plot_rouge_cmp(rouge_scores); st.pyplot(fig, use_container_width=True); plt.close(fig)
                df_r = pd.DataFrame({'Method':list(rouge_scores.keys()), 'ROUGE-1':[round(v['rouge1'],4) for v in rouge_scores.values()], 'ROUGE-2':[round(v['rouge2'],4) for v in rouge_scores.values()], 'ROUGE-L':[round(v['rougeL'],4) for v in rouge_scores.values()]}).sort_values('ROUGE-1',ascending=False)
                st.dataframe(df_r, use_container_width=True, hide_index=True)
                best = df_r.iloc[0]['Method']
                st.success(f"✅ Best method: **{best}** (ROUGE-1: {df_r.iloc[0]['ROUGE-1']:.4f})")
        elif run_sum:
            st.warning("Please enter an abstract first.")

    # ── Performance tab ───────────────────────────────────────
    with tab_perf:
        st.markdown("### Model Performance — Section 2 Results")
        st.markdown("""
<div class="kpi-grid">
  <div class="kpi"><span class="val">89.4%</span><span class="lbl">Best Accuracy (SVM+bigrams)</span></div>
  <div class="kpi"><span class="val">0.847</span><span class="lbl">Best Macro F1</span></div>
  <div class="kpi"><span class="val">92.1%</span><span class="lbl">Scaled 2M Accuracy</span></div>
  <div class="kpi"><span class="val">0.38</span><span class="lbl">Best ROUGE-1 (TF-IDF)</span></div>
</div>""", unsafe_allow_html=True)
        st.pyplot(plot_model_cmp(), use_container_width=True); plt.close()
        st.markdown("")
        st.pyplot(plot_sum_cmp(), use_container_width=True); plt.close()
        st.markdown("#### Full Results Table")
        st.dataframe(pd.DataFrame({
            'Model':['Naive Bayes (α=1.0)','Naive Bayes (α=0.1)','LR C=1 unigrams','LR C=5 unigrams','LR C=5 bigrams','SVM C=1 unigrams','SVM C=0.5 unigrams','SVM C=1 bigrams','SVM title×2+abstract'],
            'Features':['Unigrams','Unigrams','Unigrams','Unigrams','Bigrams','Unigrams','Unigrams','Bigrams','Title+Abstract'],
            'Accuracy':[0.783,0.791,0.839,0.851,0.851,0.872,0.863,0.894,0.901],
            'Macro F1':[0.731,0.741,0.787,0.801,0.801,0.823,0.814,0.847,0.853],
            'Weighted F1':[0.779,0.788,0.835,0.849,0.849,0.869,0.860,0.891,0.898],
        }), use_container_width=True, hide_index=True)
        st.markdown("#### Data Scaling Experiment")
        st.dataframe(pd.DataFrame({
            'Papers':['1,000,000','2,000,000'],'MIN per category':[100,200],'MAX per category':[3000,5000],
            'Accuracy':['89.4%','92.1%'],'Macro F1':['0.847','0.879'],'Weighted F1':['0.891','0.919'],
        }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: LLM
# ══════════════════════════════════════════════════════════════
elif page == "llm":
    st.markdown("""
<div class="hero teal">
  <div class="hero-tag">🤖 Section 3 — Large Language Models</div>
  <h1>LLM Pipeline — DeepSeek-V3</h1>
  <p>Zero-Shot · Few-Shot · Chain-of-Thought · RAG · Scientific Chatbot — powered by <strong style="color:#5eead4">deepseek-chat</strong></p>
</div>""", unsafe_allow_html=True)

    api_ok = bool(DEEPSEEK_API_KEY)
    if not api_ok:
        st.info("ℹ️ **DeepSeek API key not configured.** Set `DEEPSEEK_API_KEY` in your Streamlit secrets or environment variables to enable LLM features.")

    tab_cls_l, tab_sum_l, tab_chat = st.tabs(["🏷️  LLM Classifier", "📝  Title Generator", "💬  Chatbot"])

    # ── LLM Classifier ────────────────────────────────────────
    with tab_cls_l:
        st.markdown("### LLM Paper Classifier")
        st.markdown("Compare **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought** prompting strategies using the same DeepSeek-V3 model.")
        llm_ab = st.text_area("Abstract", height=155, placeholder="Paste a scientific paper abstract here...", key="llm_ab")
        strategy = st.selectbox("Prompting Strategy", ["Zero-Shot","Few-Shot","Chain-of-Thought (CoT)","All three (compare)"])
        run_llm = st.button("🤖 Classify with DeepSeek", type="primary", key="llm_cls_btn")

        if run_llm:
            if not api_ok:
                st.error("DeepSeek API key not configured. Set DEEPSEEK_API_KEY in Streamlit secrets.")
            elif not llm_ab.strip():
                st.warning("Please enter an abstract.")
            else:
                ab = clean_text(llm_ab)
                to_run = ["Zero-Shot","Few-Shot","Chain-of-Thought (CoT)"] if strategy=="All three (compare)" else [strategy]
                results_cls = {}
                for strat in to_run:
                    with st.spinner(f"Running {strat}..."):
                        if strat=="Zero-Shot":      raw=call_deepseek(prompt_zero_shot(ab),SYSTEM_CLASSIFY,20,0.0)
                        elif strat=="Few-Shot":     raw=call_deepseek(prompt_few_shot(ab), SYSTEM_CLASSIFY,20,0.0)
                        else:                       raw=call_deepseek(prompt_cot(ab),      "",400,0.0)
                        results_cls[strat]={"raw":raw,"pred":parse_cat(raw)}

                st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
                badge_cols = {"Zero-Shot":"pill-indigo","Few-Shot":"pill-green","Chain-of-Thought (CoT)":"pill-purple"}
                for strat,res in results_cls.items():
                    cat=res["pred"]; dom=get_domain(cat)
                    st.markdown(f'<div class="result-block"><div class="rb-title">{strat}</div><div class="rb-body"><span class="pill {badge_cols.get(strat,"pill-slate")}">{cat}</span> <span class="pill pill-slate">{dom}</span></div></div>', unsafe_allow_html=True)
                    if strat=="Chain-of-Thought (CoT)":
                        with st.expander("View reasoning trace"): st.markdown(res["raw"])

                if len(results_cls)>1:
                    st.markdown("#### Strategy Comparison")
                    st.dataframe(pd.DataFrame([{"Strategy":s,"Predicted":r["pred"],"Domain":get_domain(r["pred"])} for s,r in results_cls.items()]), use_container_width=True, hide_index=True)
                    preds_l=[r["pred"] for r in results_cls.values()]
                    if len(set(preds_l))==1: st.success(f"✅ All strategies agree: **{preds_l[0]}**")
                    else: st.warning("⚠️ Strategies disagree — check the CoT reasoning trace for details.")

    # ── Title Generator ───────────────────────────────────────
    with tab_sum_l:
        st.markdown("### LLM Title Generator")
        st.markdown("Generate paper titles from abstracts using **Direct** or **Structured** prompting, compared against classical methods.")
        cl,cr = st.columns(2)
        with cl: llm_sum_ab = st.text_area("Abstract", height=175, placeholder="Paste abstract...", key="llm_sum_ab")
        with cr:
            llm_ref = st.text_input("Reference title (for ROUGE scoring)", placeholder="Actual paper title...", key="llm_ref")
            sum_strat = st.radio("Prompting style", ["Direct","Structured","Both (compare)"], key="sum_s")
        run_sum_l = st.button("📝 Generate Title", type="primary", key="llm_sum_btn")

        if run_sum_l:
            if not api_ok:
                st.error("DeepSeek API key not configured.")
            elif not llm_sum_ab.strip():
                st.warning("Please enter an abstract.")
            else:
                ab=clean_text(llm_sum_ab)
                to_run2=["Direct","Structured"] if sum_strat=="Both (compare)" else [sum_strat]
                sum_res={}
                for ss in to_run2:
                    with st.spinner(f"Running {ss} prompting..."):
                        if ss=="Direct":    raw=call_deepseek(prompt_direct(ab),    SYSTEM_SUMMARISE,60,0.3); title=raw
                        else:               raw=call_deepseek(prompt_structured(ab),SYSTEM_SUMMARISE,200,0.2); title=parse_title(raw)
                        sum_res[ss]={"title":title,"raw":raw}
                st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
                for ss,res in sum_res.items():
                    bc="pill-indigo" if ss=="Direct" else "pill-purple"
                    st.markdown(f'<div class="result-block"><div class="rb-title">{ss} Prompt <span class="pill {bc}">{len(res["title"].split())} words</span></div><div class="rb-body"><em>"{res["title"]}"</em></div></div>', unsafe_allow_html=True)

                if llm_ref.strip():
                    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
                    st.markdown("#### ROUGE Scores vs Reference Title")
                    rscores={}
                    for ss,res in sum_res.items():
                        rscores[f"LLM {ss}"]={'rouge1':rouge_n(res["title"],llm_ref,1),'rouge2':rouge_n(res["title"],llm_ref,2),'rougeL':rouge_l(res["title"],llm_ref)}
                    ab2=clean_text(llm_sum_ab)
                    rscores["Lead Sentence"]    ={'rouge1':rouge_n(lead_sentence(ab2),    llm_ref,1),'rouge2':rouge_n(lead_sentence(ab2),    llm_ref,2),'rougeL':rouge_l(lead_sentence(ab2),    llm_ref)}
                    rscores["TF-IDF Extractive"]={'rouge1':rouge_n(tfidf_extractive(ab2), llm_ref,1),'rouge2':rouge_n(tfidf_extractive(ab2), llm_ref,2),'rougeL':rouge_l(tfidf_extractive(ab2), llm_ref)}
                    fig=plot_rouge_cmp(rscores); st.pyplot(fig,use_container_width=True); plt.close(fig)
                    df_rl=pd.DataFrame([{"Method":m,"ROUGE-1":round(v['rouge1'],4),"ROUGE-2":round(v['rouge2'],4),"ROUGE-L":round(v['rougeL'],4)} for m,v in rscores.items()]).sort_values("ROUGE-1",ascending=False)
                    st.dataframe(df_rl,use_container_width=True,hide_index=True)

    # ── Chatbot ───────────────────────────────────────────────
    with tab_chat:
        st.markdown("### 💬 Scientific Research Chatbot")
        st.markdown("Ask DeepSeek anything about ArXiv categories, paper concepts, or paste an abstract for analysis.")

        qcols = st.columns(4)
        qmap = {
            "Explain cs.LG":   "Explain what the ArXiv category cs.LG covers and give 3 example research topics.",
            "What is RAG?":    "Explain Retrieval-Augmented Generation (RAG) and how it helps LLM classification of scientific papers.",
            "TF-IDF vs LLM":   "Compare TF-IDF classification with LLM-based classification for scientific papers. Key trade-offs?",
            "Classify abstract":"Classify this abstract and explain why: 'We study black hole formation in the early universe using N-body simulations. Dark matter halos collapse faster than predicted by ΛCDM.'",
        }
        for col,(label,qtext) in zip(qcols,qmap.items()):
            if col.button(label, use_container_width=True):
                st.session_state.setdefault("chat_history",[])
                st.session_state["chat_history"].append({"role":"user","content":qtext})

        st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
        if "chat_history" not in st.session_state: st.session_state["chat_history"]=[]

        for msg in st.session_state["chat_history"]:
            if msg["role"]=="user":
                st.markdown(f'<div class="chat-msg user"><div class="chat-avatar">🧑</div><div class="chat-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg bot"><div class="chat-avatar">🤖</div><div class="chat-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)

        if (st.session_state["chat_history"] and
                st.session_state["chat_history"][-1]["role"]=="user" and api_ok):
            with st.spinner("DeepSeek is thinking..."):
                history=st.session_state["chat_history"][-6:]
                prompt_ctx="\n\n".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in history)
                resp=call_deepseek(prompt_ctx+"\n\nAssistant:",SYSTEM_CHATBOT,500,0.5)
                st.session_state["chat_history"].append({"role":"assistant","content":resp})
                st.rerun()

        c_inp,c_send,c_clr = st.columns([7,1,1])
        with c_inp:
            user_msg=st.text_input("Message",placeholder="Ask about a paper, category, or research concept...",label_visibility="collapsed",key="chat_input")
        with c_send:
            if st.button("Send",type="primary",use_container_width=True) and user_msg.strip():
                if not api_ok: st.error("DeepSeek API key not configured.")
                else:
                    st.session_state["chat_history"].append({"role":"user","content":user_msg.strip()})
                    st.rerun()
        with c_clr:
            if st.button("Clear",use_container_width=True):
                st.session_state["chat_history"]=[]; st.rerun()


# ══════════════════════════════════════════════════════════════
#  PAGE: RESULTS
# ══════════════════════════════════════════════════════════════
elif page == "results":
    st.markdown("""
<div class="hero amber">
  <div class="hero-tag">📊 Full Evaluation — Classical ML vs LLM</div>
  <h1>Results & Comparison</h1>
  <p>Complete performance breakdown — classification accuracy, macro F1, ROUGE scores, and practical trade-offs across all methods.</p>
</div>""", unsafe_allow_html=True)

    tab_c, tab_s, tab_a = st.tabs(["🏷️  Classification","📝  Summarisation","🔍  Analysis & Reflection"])

    with tab_c:
        st.markdown("""
<div class="kpi-grid">
  <div class="kpi"><span class="val">89.4%</span><span class="lbl">Best ML Accuracy</span></div>
  <div class="kpi"><span class="val">0.847</span><span class="lbl">Best ML Macro F1</span></div>
  <div class="kpi"><span class="val">92.1%</span><span class="lbl">Scaled 2M Accuracy</span></div>
  <div class="kpi"><span class="val">~75–83%</span><span class="lbl">Best LLM Accuracy</span></div>
</div>""", unsafe_allow_html=True)
        st.pyplot(plot_full_cmp(), use_container_width=True); plt.close()
        st.markdown("#### Full Table")
        st.dataframe(pd.DataFrame({
            'Method':['Naive Bayes','LR (best)','SVM (best)','SVM (scaled 2M)','Zero-Shot LLM','Few-Shot LLM','CoT LLM','RAG+LLM'],
            'Type':['Classical','Classical','Classical','Classical','LLM','LLM','LLM','RAG'],
            'Accuracy':[0.783,0.894,0.901,0.921,0.717,0.750,0.783,0.817],
            'Macro F1':[0.731,0.847,0.853,0.879,0.698,0.732,0.763,0.795],
            'Training?':['Yes','Yes','Yes','Yes','No','No','No','No (encoder)'],
            'Cost':['Free','Free','Free','Free','API','API','API','API+encoder'],
        }), use_container_width=True, hide_index=True)

    with tab_s:
        st.markdown("""
<div class="kpi-grid">
  <div class="kpi"><span class="val">0.38</span><span class="lbl">Best Classical ROUGE-1</span></div>
  <div class="kpi"><span class="val">0.51</span><span class="lbl">Best LLM ROUGE-1 (RAG)</span></div>
  <div class="kpi"><span class="val">0.25</span><span class="lbl">Best LLM ROUGE-2</span></div>
</div>""", unsafe_allow_html=True)
        all_m=['Lead Sentence','Keyword Density','TextRank','TF-IDF Extractive','LLM Direct','LLM Structured','LLM RAG']
        r1a=[0.31,0.24,0.34,0.38,0.46,0.49,0.51]; r2a=[0.12,0.08,0.13,0.16,0.21,0.23,0.25]; rla=[0.29,0.22,0.31,0.35,0.43,0.46,0.48]
        types=['Classical','Classical','Classical','Classical','LLM','LLM','RAG']
        bar_c=['#818cf8' if t=='Classical' else '#34d399' if t=='LLM' else '#fbbf24' for t in types]
        fig,axes=plt.subplots(1,2,figsize=(13,5),facecolor='#0d1117')
        x=np.arange(len(all_m))
        ax=axes[0]; ax.set_facecolor('#0d1117')
        ax.bar(x,r1a,color=bar_c,alpha=0.9,edgecolor='#0d1117')
        ax.set_xticks(x); ax.set_xticklabels([m.replace('LLM ','LLM\n') for m in all_m],rotation=20,ha='right',fontsize=8)
        ax.set_ylim(0,0.65); ax.set_ylabel('ROUGE-1'); ax.set_title('ROUGE-1: All Methods',fontsize=10,fontweight='bold',color='#e2e8f0')
        for i,v in enumerate(r1a): ax.text(i,v+0.01,f'{v:.2f}',ha='center',fontsize=7.5)
        w=0.25; ax2=axes[1]; ax2.set_facecolor('#0d1117')
        ax2.bar(x-w,r1a,w,label='ROUGE-1',color='#818cf8',alpha=0.85,edgecolor='#0d1117')
        ax2.bar(x,  r2a,w,label='ROUGE-2',color='#34d399',alpha=0.85,edgecolor='#0d1117')
        ax2.bar(x+w,rla,w,label='ROUGE-L',color='#fbbf24',alpha=0.85,edgecolor='#0d1117')
        ax2.set_xticks(x); ax2.set_xticklabels([m.replace('LLM ','LLM\n') for m in all_m],rotation=20,ha='right',fontsize=8)
        ax2.set_ylim(0,0.65); ax2.set_title('All ROUGE Metrics',fontsize=10,fontweight='bold',color='#e2e8f0')
        ax2.legend(fontsize=8,facecolor='#0d1117',labelcolor='#94a3b8',edgecolor='#1e293b')
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(facecolor=c,label=k) for k,c in {'Classical':'#818cf8','LLM':'#34d399','RAG':'#fbbf24'}.items()],fontsize=8,loc='upper left',facecolor='#0d1117',labelcolor='#94a3b8',edgecolor='#1e293b')
        fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.dataframe(pd.DataFrame({'Method':all_m,'Type':types,'ROUGE-1':r1a,'ROUGE-2':r2a,'ROUGE-L':rla}).sort_values('ROUGE-1',ascending=False), use_container_width=True, hide_index=True)

    with tab_a:
        c_l,c_r=st.columns(2)
        with c_l:
            st.markdown("#### ✅ What Worked Well")
            st.markdown("""
<div class="result-block green"><div class="rb-body">
<b>Classical ML:</b><br>
• SVM + bigrams achieved <b>89.4% accuracy</b> — best classification overall<br>
• Scaling to 2M papers pushed accuracy to <b>92.1%</b>, confirming more data = better results<br>
• Higher MIN/MAX per category thresholds also improved all metrics<br>
• TF-IDF extractive (ROUGE-1: 0.38) beat the lead-sentence baseline by 22%
</div>
<div class="result-block" style="margin-top:10px"><div class="rb-body">
<b>LLM Pipeline:</b><br>
• Chain-of-Thought outperformed Zero-Shot on confusable categories (cs.LG vs stat.ML)<br>
• RAG improved LLM classification by +6.7pp — domain-matched retrieval made the difference<br>
• LLM abstractive titles are fluent and well-formed — ROUGE-1 of 0.51 vs 0.38 classical<br>
• Structured prompting outperformed direct prompting for title generation
</div>""", unsafe_allow_html=True)
        with c_r:
            st.markdown("#### ⚠️ Limitations & Improvements")
            st.markdown("""
<div class="result-block amber"><div class="rb-body">
<b>Limitations:</b><br>
• TF-IDF is semantically blind — cannot detect synonyms or paraphrase<br>
• Class imbalance suppresses macro F1 for rare categories<br>
• ROUGE-2 scores low (≤0.25) — titles use paraphrased vocab not in abstracts<br>
• API latency (~1–5s) vs sub-millisecond for classical ML<br>
• RAG corpus limited to 5k papers; full 1M coverage would improve further
</div>
<div class="result-block purple" style="margin-top:10px"><div class="rb-body">
<b>Possible Improvements:</b><br>
• Fine-tune SciBERT on ArXiv — expected +5–10% macro F1 over SVM<br>
• Abstractive summarisation (T5/BART) for genuine title generation<br>
• Hybrid system: classical ML for coarse → LLM for ambiguous cases<br>
• Class-weighted training to handle imbalance<br>
• Expand RAG index to full dataset
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "about":
    st.markdown("""
<div class="hero rose">
  <div class="hero-tag">About This Project</div>
  <h1>DG4NLP — ArXiv NLP System</h1>
  <p>An end-to-end NLP research platform for classifying and summarising scientific papers from the ArXiv repository, combining classical machine learning, large language models, and an interactive web interface.</p>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)

    cl,cr = st.columns(2)
    with cl:
        st.markdown("### 🎓 Project Overview")
        st.markdown("""
<div class="glass-card">
  <div class="card-body">
    This project was developed for the <strong style="color:#e2e8f0">DG4NLP Natural Language Processing</strong>
    module at Aston University, by Fredick Boakye<br><br>
    The system applies NLP techniques to the ArXiv metadata dataset — over 1 million scientific preprints
    from Cornell University's open-access repository. The project was built in weekly stages, progressively
    adding capabilities from data preprocessing through to this deployed web application.<br><br>
    <strong style="color:#e2e8f0">Development stack:</strong><br>
    Python 3.11 · VSCode · Git/GitHub · Streamlit Cloud<br>
    scikit-learn · FAISS · Sentence-Transformers · DeepSeek API
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
<div class="glass-card teal">
  <div class="card-body" style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
    <div>
      <div style="color:#5eead4;font-weight:700;font-size:.78rem;text-transform:uppercase;margin-bottom:6px">Data & ML</div>
      <div style="color:#94a3b8;font-size:.85rem;line-height:1.8">
        pandas · numpy<br>scikit-learn · FAISS<br>sentence-transformers<br>networkx · matplotlib
      </div>
    </div>
    <div>
      <div style="color:#5eead4;font-weight:700;font-size:.78rem;text-transform:uppercase;margin-bottom:6px">LLM & Interface</div>
      <div style="color:#94a3b8;font-size:.85rem;line-height:1.8">
        DeepSeek-V3 API<br>openai SDK<br>Streamlit<br>VSCode · GitHub
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    with cr:
        st.markdown("### Sections")
        st.markdown("""
<div class="timeline">
  <div class="tl-item">
    <div class="tl-week">Section 1</div>
    <div class="tl-title">Data Exploration & Preprocessing</div>
    <div class="tl-body">Loaded 1M ArXiv papers · filtered 41k→80 categories · TF-IDF ready</div>
  </div>
  <div class="tl-item">
    <div class="tl-week">Section 2</div>
    <div class="tl-title">Classical ML Pipeline</div>
    <div class="tl-body">LR · Naive Bayes · SVM · 4 summarisers · ROUGE evaluation · 89.4% accuracy</div>
  </div>
  <div class="tl-item">
    <div class="tl-week">Section 3</div>
    <div class="tl-title">LLM Integration & RAG</div>
    <div class="tl-body">DeepSeek-V3 · Zero/Few/CoT prompting · FAISS RAG · 0.51 ROUGE-1</div>
  </div>
  <div class="tl-item">
    <div class="tl-week">Section 4</div>
    <div class="tl-title">Streamlit Web App</div>
    <div class="tl-body">VSCode dev → GitHub → Streamlit Cloud deployment · this interface</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("### 📦 Dataset")
        st.markdown("""
<div class="glass-card amber">
  <div class="card-body">
    <strong style="color:#e2e8f0">ArXiv Metadata (Cornell University / Kaggle)</strong><br><br>
    A JSONL snapshot of 1.7M+ scientific preprints with fields including id, title, abstract, authors, categories, and version history.<br><br>
    <a href="https://www.kaggle.com/datasets/Cornell-University/arxiv" target="_blank" style="color:#fbbf24">
      🔗 kaggle.com/datasets/Cornell-University/arxiv
    </a>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center;padding:20px 0;color:#334155;font-size:.78rem">
  Natural Language Processing · Aston University · Fredick Boakye<br>
  Built with Streamlit · Deployed on Streamlit Community Cloud
</div>""", unsafe_allow_html=True)
