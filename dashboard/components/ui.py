"""Shared visual primitives for the TestWise Streamlit dashboard."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from html import escape

import streamlit as st


COLORS = {
    "bg": "#0E1117", "sidebar": "#1A1F2E", "card": "#161B2B", "border": "#263047",
    "orange": "#FF6B00", "pass": "#2ECC71", "fail": "#FF4B4B", "warn": "#F5B041",
    "info": "#4DA3FF", "text": "#E8EAF0", "muted": "#6B7494", "sub": "#A0A8C0",
}


def inject_styles() -> None:
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Source+Sans+3:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap');
    :root { --tw-bg:#0E1117; --tw-sidebar:#1A1F2E; --tw-card:#161B2B; --tw-border:#263047; --tw-orange:#FF6B00; --tw-text:#E8EAF0; --tw-muted:#6B7494; --tw-sub:#A0A8C0; }
    html, body, [class*="st-"], button, input, textarea, select { font-family:'Source Sans 3',sans-serif; }
    .material-symbols-rounded, .material-symbols-outlined, [data-testid="stIconMaterial"] { font-family:'Material Symbols Rounded' !important; font-weight:normal !important; font-style:normal; font-size:20px; line-height:1; letter-spacing:normal; text-transform:none; white-space:nowrap; word-wrap:normal; direction:ltr; font-feature-settings:'liga'; -webkit-font-feature-settings:'liga'; -webkit-font-smoothing:antialiased; }
    [data-testid="stAppViewContainer"], .stApp { background:var(--tw-bg); color:var(--tw-text); }
    [data-testid="stSidebar"] { background:var(--tw-sidebar); border-right:1px solid var(--tw-border); min-width:260px !important; max-width:260px !important; }
    [data-testid="stSidebar"] > div:first-child { padding:0.4rem 0.9rem 1rem; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.35rem; }
    [data-testid="stAppViewContainer"] .main .block-container { max-width:1088px; padding:2rem 2rem 3rem; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stToolbar"] { right:1rem; }
    h1,h2,h3,h4 { font-family:'Outfit',sans-serif !important; color:var(--tw-text); }
    p, label, [data-testid="stMarkdownContainer"] { color:var(--tw-sub); }
    [data-testid="stVerticalBlockBorderWrapper"] { background:var(--tw-card); border:1px solid var(--tw-border); border-radius:12px; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { padding:1rem; }
    [data-testid="stButton"] button { border-radius:8px; font-family:'Outfit',sans-serif; font-weight:600; min-height:2.25rem; transition:all .16s ease; }
    [data-testid="stButton"] button:hover { border-color:var(--tw-orange); color:var(--tw-orange); }
    [data-testid="stButton"] button[kind="primary"] { background:var(--tw-orange); border-color:var(--tw-orange); color:#fff; }
    [data-testid="stButton"] button[kind="primary"]:hover { background:#e85f00; color:#fff; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input { background:#1A1F2E; border-color:var(--tw-border); color:var(--tw-text); border-radius:8px; }
    [data-testid="stSlider"] [data-testid="stThumbValue"] { color:var(--tw-orange); }
    [data-testid="stDataFrame"], [data-testid="stDataEditor"] { border:1px solid var(--tw-border); border-radius:9px; overflow:hidden; }
    [data-testid="stMetric"] { background:#161B2B; border:1px solid var(--tw-border); border-radius:12px; padding:14px 16px; }
    [data-testid="stMetricLabel"] { font:500 11px 'JetBrains Mono',monospace; letter-spacing:.11em; text-transform:uppercase; color:var(--tw-muted); }
    [data-testid="stMetricValue"] { font:700 26px 'Outfit',sans-serif; color:var(--tw-text); }
    [data-testid="stRadio"] label { color:var(--tw-sub); }
    [data-testid="stCheckbox"] label { color:var(--tw-sub); }
    [data-testid="stExpander"] { background:#161B2B; border:1px solid var(--tw-border); border-radius:10px; }
    [data-testid="stCode"] { background:#0A0D15; border:1px solid var(--tw-border); border-radius:10px; }
    [data-testid="stAlert"] { border-radius:10px; }
    .tw-brand { padding:1.05rem .35rem .9rem; border-bottom:1px solid var(--tw-border); margin-bottom:.5rem; display:flex; gap:10px; align-items:center; }
    .tw-brand-mark { width:32px;height:32px;border-radius:9px;background:var(--tw-orange);display:grid;place-items:center;color:white;font:700 13px Outfit,sans-serif; }
    .tw-brand-name { color:var(--tw-text);font:700 14px Outfit,sans-serif;line-height:1.2; }
    .tw-brand-sub { color:var(--tw-muted);font:12px 'Source Sans 3',sans-serif;margin-top:3px; }
    .tw-section { display:flex;align-items:center;min-height:28px;box-sizing:border-box;color:#8791b2;font:600 11px/1.4 'JetBrains Mono',monospace;letter-spacing:.13em;text-transform:uppercase;padding:8px .55rem 5px;margin-top:.55rem;margin-bottom:14px; }
    .tw-status { border-top:1px solid var(--tw-border);padding:.8rem .4rem .1rem;color:var(--tw-muted);font-size:12px;display:flex;align-items:center;gap:8px;margin-top:1rem; }
    .tw-dot { height:8px;width:8px;background:#2ECC71;border-radius:50%;box-shadow:0 0 8px #2ECC71; }
    .tw-header { display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 8px; }
    .tw-title { color:var(--tw-orange);font:700 26px Outfit,sans-serif;display:flex;align-items:center;gap:10px;margin:0; }
    .tw-rule { height:1px;background:linear-gradient(90deg,var(--tw-orange),transparent);margin:0 0 24px; }
    .tw-sample { white-space:nowrap;color:var(--tw-orange);background:#ff6b0026;border:1px solid #ff6b004d;border-radius:30px;padding:4px 10px;font:500 11px 'JetBrains Mono',monospace; }
    .tw-card-title { color:var(--tw-text);font:700 15px Outfit,sans-serif;margin:0 0 12px; }
    .tw-eyebrow { color:var(--tw-muted);font:600 10px 'JetBrains Mono',monospace;letter-spacing:.11em;text-transform:uppercase; }
    .tw-callout { border-radius:10px;padding:13px 15px;font-size:13px;line-height:1.5;background:#4da3ff14;border:1px solid #4da3ff38;color:var(--tw-sub); }
    .tw-pill { display:inline-block;border-radius:6px;padding:2px 8px;font:500 11px 'JetBrains Mono',monospace;white-space:nowrap; }
    .tw-mono { font-family:'JetBrains Mono',monospace; }
    @media(max-width:900px) { [data-testid="stSidebar"] { min-width:225px !important; max-width:225px !important; } [data-testid="stAppViewContainer"] .main .block-container { padding:1.25rem; } }
    </style>""", unsafe_allow_html=True)


def page_header(emoji: str, title: str) -> None:
    st.markdown(f'<div class="tw-header"><h1 class="tw-title"><span>{escape(emoji)}</span>{escape(title)}</h1><span class="tw-sample">Sample data</span></div><div class="tw-rule"></div>', unsafe_allow_html=True)


def section_title(title: str) -> None:
    st.markdown(f'<div class="tw-card-title">{escape(title)}</div>', unsafe_allow_html=True)


@contextmanager
def card():
    with st.container(border=True):
        yield


def pill(label: str, color: str = "muted") -> None:
    styles = {
        "pass": ("#2ecc7126", "#2ECC71", "#2ecc714d"), "fail": ("#ff4b4b26", "#FF4B4B", "#ff4b4b4d"),
        "warn": ("#f5b04126", "#F5B041", "#f5b0414d"), "info": ("#4da3ff26", "#4DA3FF", "#4da3ff4d"),
        "orange": ("#ff6b0026", "#FF6B00", "#ff6b004d"), "muted": ("#6b749426", "#A0A8C0", "#6b74944d"),
    }
    bg, fg, border = styles.get(color, styles["muted"])
    st.markdown(f'<span class="tw-pill" style="background:{bg};color:{fg};border:1px solid {border}">{escape(str(label))}</span>', unsafe_allow_html=True)


def callout(text: str, tone: str = "info") -> None:
    colors = {"info": "#4DA3FF", "warn": "#F5B041", "error": "#FF4B4B"}
    accent = colors[tone]
    st.markdown(f'<div class="tw-callout" style="background:{accent}12;border-color:{accent}45"><span style="color:{accent};font-weight:700;margin-right:8px">●</span>{text}</div>', unsafe_allow_html=True)


def metric(label: str, value: str, sub: str = "", color: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f'<div class="tw-eyebrow">{escape(label)}</div><div style="font:700 27px Outfit,sans-serif;color:{color or COLORS["text"]};margin:.2rem 0">{escape(value)}</div><div style="color:{COLORS["muted"]};font-size:12px">{escape(sub)}</div>', unsafe_allow_html=True)


def chart_style() -> dict:
    return {
        "background": COLORS["card"], "view": {"stroke": "transparent"},
        "axis": {"labelColor": COLORS["muted"], "titleColor": COLORS["muted"], "gridColor": COLORS["border"], "domainColor": COLORS["border"], "tickColor": COLORS["border"], "labelFont": "JetBrains Mono", "titleFont": "JetBrains Mono"},
        "legend": {"labelColor": COLORS["sub"], "titleColor": COLORS["sub"], "labelFont": "Source Sans 3"},
        "title": {"color": COLORS["text"], "font": "Outfit"},
    }

