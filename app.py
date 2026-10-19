"""TestWise Streamlit UI entry point."""
import streamlit as st

from dashboard.components.ui import inject_styles
from dashboard.pages.ui import PAGES


st.set_page_config(
    page_title="TestWise · ML Test Selection",
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

if "tw_page" not in st.session_state:
    st.session_state["tw_page"] = "Home"

with st.sidebar:
    st.markdown("""<div class="tw-brand"><div class="tw-brand-mark">TW</div><div><div class="tw-brand-name">TestWise</div><div class="tw-brand-sub">ML Test Selection</div></div></div>""", unsafe_allow_html=True)
    groups = {
        "Main": ["Home"],
        "Live Pipeline": ["Repository", "Code Changes", "Test Prediction", "Test Execution"],
        "Learning": ["Bandit Learning", "Analytics"],
        "Research": ["Offline Evaluation", "Settings & Safety"],
    }
    for heading, pages in groups.items():
        st.markdown(f'<div class="tw-section">{heading}</div>', unsafe_allow_html=True)
        for page in pages:
            icon, _ = PAGES[page]
            if st.button(
                page,
                key=f"nav_{page}",
                icon=icon,
                type="primary" if st.session_state["tw_page"] == page else "secondary",
                width="stretch",
            ):
                st.session_state["tw_page"] = page
                st.rerun()
    st.markdown('<div class="tw-status"><span class="tw-dot"></span><span>Demo mode · Sample data</span><span style="margin-left:auto">♧</span></div>', unsafe_allow_html=True)

PAGES[st.session_state["tw_page"]][1]()
