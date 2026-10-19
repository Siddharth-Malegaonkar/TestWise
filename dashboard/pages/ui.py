"""UI-only page renderers ported from the supplied TestWise React design."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.components.ui import (
    COLORS as C, card, callout, chart_style, metric, page_header, pill, section_title,
)


def _title(text: str) -> None:
    section_title(text)


def home() -> None:
    page_header("🏠", "Home")
    st.markdown("""<div style="border:1px solid #2A4A8A;border-radius:16px;padding:36px 28px;margin-bottom:24px;text-align:center;background:linear-gradient(135deg,#1E3A6E 0%,#6D3AE0 100%)">
    <div style="font:800 48px Outfit,sans-serif;color:white;margin-bottom:8px">TestWise</div>
    <div style="font:400 19px 'Source Sans 3',sans-serif;color:#ffffffbf">Run fewer tests. Catch the failures that matter.</div>
    <div style="font-size:14px;color:#ffffff80;margin-top:8px">ML-powered test selection for CI pipelines — rank tests by failure risk, run only what counts, learn from results.</div></div>""", unsafe_allow_html=True)
    a, b = st.columns(2, gap="medium")
    with a, card():
        st.markdown("<div style='font-size:30px'>🚀</div><h3>Live Pipeline</h3><p>Connect a GitHub repo, get ML-ranked predictions per push, and watch TestWise learn from every run.</p>", unsafe_allow_html=True)
        st.button("Connect Repository →", type="primary", width="stretch", on_click=_go, args=("Repository",), key="home_repo")
    with b, card():
        st.markdown("<div style='font-size:30px'>🔬</div><h3>Research Lab</h3><p>Evaluate TestWise offline on the LRTS dataset — compare models, view APFD scores, and explore budget trade-offs.</p>", unsafe_allow_html=True)
        st.button("View Benchmark Results →", width="stretch", on_click=_go, args=("Offline Evaluation",), key="home_offline")
    st.markdown('<div class="tw-eyebrow" style="margin:6px 0 14px">How It Works</div>', unsafe_allow_html=True)
    steps = [("📤", "Push", "Developer pushes code to GitHub"), ("🎯", "Select", "ML model ranks tests by failure risk"), ("▶️", "Run", "Riskiest subset runs in GitHub Actions"), ("🧠", "Learn", "Results feed back to improve the model")]
    cols = st.columns(4, gap="small")
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i], card():
            st.markdown(f"<div style='text-align:center;padding:5px 0'><span style='font-size:25px'>{icon}</span><div style='font:700 14px Outfit,sans-serif;color:{C['text']};margin-top:8px'>{title}</div><div style='font-size:12px;margin-top:4px'>{desc}</div></div>", unsafe_allow_html=True)


def _go(page: str) -> None:
    st.session_state["tw_page"] = page


def repository() -> None:
    page_header("🔗", "Repository")
    left, right = st.columns([1, 1], gap="large")
    with left:
        with card():
            _title("Connect a Repository")
            repo = st.selectbox("GitHub Repository", ["user/healthInsuranceEvaluation", "user/ml-pipeline", "org/backend-api"], label_visibility="visible")
            branch = st.selectbox("Branch", ["main", "develop", "feature/preprocessing-fix"])
            st.text_input("API Token", value="ghp_xxxxxxxxxxxxxxxxxxxx", type="password")
            if st.button("Generate Workflow File", type="primary", width="stretch"):
                st.session_state["tw_connected"] = True
                st.session_state["tw_repo"] = repo
                st.session_state["tw_branch"] = branch
        with card():
            _title("GitHub Actions Workflow")
            st.code("""name: TestWise CI
on: [push, pull_request]
jobs:
  testwise:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run TestWise
        uses: testwise/action@v1
        with:
          api-token: ${{ secrets.TESTWISE_TOKEN }}
          repository: ${{ github.repository }}""", language="yaml")
    with right:
        if st.session_state.get("tw_connected", True):
            with card():
                repo_label = st.session_state.get("tw_repo", "user/healthInsuranceEvaluation")
                st.markdown(f"<div style='font:600 13px JetBrains Mono,monospace;color:{C['text']}'>⑂ &nbsp;{repo_label}</div>", unsafe_allow_html=True)
                st.caption("🟢 Connected　 ·　Last run 10 min ago")
                cols = st.columns(2)
                for col, (value, label, color) in zip(cols * 2, [("25", "Total tests", C["text"]), ("4", "Test files", C["text"]), ("98.8%", "Detection rate", C["pass"]), ("47", "Runs tracked", C["info"])]):
                    with col:
                        with st.container(border=True):
                            st.markdown(f"<div style='font:700 22px Outfit,sans-serif;color:{color}'>{value}</div><div style='font-size:12px;color:{C['muted']}'>{label}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='border-top:1px solid {C['border']};padding-top:12px;font-size:12px;color:{C['muted']}'>Branch: <span class='tw-mono' style='color:{C['text']}'>{st.session_state.get('tw_branch','main')}</span><br>Workflow: <span class='tw-mono' style='color:{C['pass']}'>.github/workflows/testwise.yml</span></div>", unsafe_allow_html=True)
                if st.button("Disconnect repository", key="disconnect_repo"):
                    st.session_state["tw_connected"] = False
                    st.rerun()
        else:
            with card():
                st.markdown("<div style='padding:60px 15px;text-align:center'><div style='font-size:38px'>📭</div><b>No repository connected yet</b><p>Fill in the form and click “Generate Workflow File” to get started.</p></div>", unsafe_allow_html=True)


def code_changes() -> None:
    page_header("📝", "Code Changes")
    with card():
        st.markdown(f"<span style='color:{C['info']};background:#4da3ff26;padding:3px 8px;border-radius:6px;font:11px JetBrains Mono,monospace'>PR #3</span> &nbsp; <b class='tw-mono' style='color:{C['text']}'>feature/preprocessing-fix</b><div style='margin-top:9px;font-size:12px'>Commit <span class='tw-mono' style='color:{C['text']}'>a1b2c3d</span>　 by <span style='color:{C['info']}'>dev@example.com</span>　 2 hours ago</div>", unsafe_allow_html=True)
    cols = st.columns(4, gap="small")
    for col, (label, value, color) in zip(cols, [("Files Changed", "2", None), ("Lines Added", "+35", C["pass"]), ("Lines Removed", "−8", C["fail"]), ("Commits", "1", None)]):
        with col: metric(label, value, color=color)
    with card():
        _title("Changed Files")
        data = pd.DataFrame([
            {"File": "backend/preprocessing.py", "Status": "modified", "Added": 30, "Removed": 5, "Diff": "View diff"},
            {"File": "requirements.txt", "Status": "modified", "Added": 5, "Removed": 3, "Diff": "View diff"},
        ])
        st.dataframe(data, hide_index=True, width="stretch", column_config={"File": st.column_config.TextColumn(width="large"), "Added": st.column_config.NumberColumn(format="+%d"), "Removed": st.column_config.NumberColumn(format="-%d")})
        for f in data["File"]:
            with st.expander(f"View diff · {f}"):
                st.code("@@ -18,4 +18,7 @@\n+ # Updated preprocessing logic\n+ # Dependency configuration changes", language="diff")
    callout("Config or dependency files changed — <code>requirements.txt</code> was modified. TestWise recommends running the <b>full test suite</b> to ensure no hidden regressions from updated dependencies.", "warn")


def prediction() -> None:
    page_header("🎯", "Test Prediction")
    if "tw_predictions" not in st.session_state:
        st.session_state["tw_predictions"] = pd.DataFrame([
            {"Rank": 1, "Test File": "tests/test_preprocessing.py", "Risk Score": .82, "Level": "High", "Why": "name matches changed file · failed in last 5 runs", "Duration": .4, "Run": True},
            {"Rank": 2, "Test File": "tests/test_validation.py", "Risk Score": .41, "Level": "Medium", "Why": "slow test · name overlap", "Duration": .3, "Run": True},
            {"Rank": 3, "Test File": "tests/test_prediction.py", "Risk Score": .18, "Level": "Low", "Why": "no recent failures", "Duration": .2, "Run": False},
            {"Rank": 4, "Test File": "tests/test_api.py", "Risk Score": .07, "Level": "Low", "Why": "stable history", "Duration": .2, "Run": False},
        ])
    left, right = st.columns([1, .33], gap="large")
    with left:
        with card():
            edited = st.data_editor(st.session_state["tw_predictions"], hide_index=True, width="stretch", disabled=["Rank", "Test File", "Risk Score", "Level", "Why", "Duration"], column_config={"Rank": st.column_config.NumberColumn(width="small"), "Test File": st.column_config.TextColumn(width="large"), "Risk Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f", width="small"), "Duration": st.column_config.NumberColumn(format="%.1f s", width="small"), "Run": st.column_config.CheckboxColumn(width="small")}, key="prediction_editor")
            st.session_state["tw_predictions"] = edited
        callout("Risk scores rank tests by predicted failure likelihood. They are <b>not probabilities</b> — a score of 0.82 does not mean 82% chance of failure.")
    with right:
        with card():
            _title("Selection Policy")
            policy = st.radio("Policy", ["Top-N", "Time Budget"], horizontal=True, label_visibility="collapsed", key="prediction_policy")
            if policy == "Top-N": st.slider("Run top tests", 1, 4, 2, key="prediction_topn")
            else: st.slider("Budget (seconds)", .5, 2.0, 1.0, .1, key="prediction_budget")
        with card():
            selected = edited.loc[edited["Run"]]
            _title("Selection Summary")
            st.markdown(f"<div style='font:700 27px Outfit,sans-serif;color:{C['text']}'>{len(selected)} <span style='font:400 16px Outfit;color:{C['muted']}'>of {len(edited)}</span></div><div style='font-size:12px'>tests selected to run</div><div style='margin-top:12px;font-size:12px'>Est. run time: <span class='tw-mono' style='color:{C['text']}'>{selected['Duration'].sum():.1f} s</span></div>", unsafe_allow_html=True)


def execution() -> None:
    page_header("▶️", "Test Execution")
    cols = st.columns(4, gap="small")
    for col, args in zip(cols, [("Tests Selected", "5 / 25", "20% of suite", C["info"]), ("Run Time", "1.1 s", "vs 2.0 s full", C["pass"]), ("Failures Found", "1", "", C["fail"]), ("Fallback Used", "No", "", C["pass"])]):
        with col: metric(*args[:3], color=args[3])
    results = [
        ("tests/test_preprocessing.py::test_normalize_input", "fail", "0.41 s"),
        ("tests/test_preprocessing.py::test_handle_missing", "pass", "0.08 s"),
        ("tests/test_preprocessing.py::test_scale_features", "pass", "0.10 s"),
        ("tests/test_validation.py::test_schema_check", "pass", "0.12 s"),
        ("tests/test_validation.py::test_type_coercion", "pass", "0.09 s"),
    ]
    with card():
        _title("Results")
        for i, (name, status, duration) in enumerate(results):
            icon, color = ("✕", C["fail"]) if status == "fail" else ("✓", C["pass"])
            with st.expander(f"{icon}　{name}　 ·　{duration}", expanded=(i == 0)):
                if status == "fail": st.code('AssertionError: expected 0.75, got 0.72\n  File "tests/test_preprocessing.py", line 34\n    assert result == 0.75', language="text")
                else: st.caption("Test passed")
    with card():
        if st.button(f"⌄  Skipped tests ({5})", key="toggle_skipped"):
            st.session_state["show_skipped"] = not st.session_state.get("show_skipped", False)
        if st.session_state.get("show_skipped", False):
            for test in ["tests/test_prediction.py::test_model_output", "tests/test_prediction.py::test_feature_importance", "tests/test_api.py::test_health_endpoint", "tests/test_api.py::test_predict_route", "tests/test_api.py::test_rate_limiting"]:
                st.markdown(f"<div class='tw-mono' style='font-size:12px;padding:6px 0;border-bottom:1px solid {C['border']}'>{test}</div>", unsafe_allow_html=True)
    callout("API Unavailable variant: If the TestWise prediction API is unreachable, the action automatically falls back to running the <b>full test suite</b>. No tests are skipped when the API cannot be reached.", "warn")


def _chart(data: pd.DataFrame, mark: str, x: str, y: str, color: str, *, height: int = 185, stack: bool = False, y_domain=None) -> None:
    enc = {
        "x": alt.X(f"{x}:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=90)) if not x.endswith(":Q") else alt.X(x, title=None),
        "y": alt.Y(y, title=None, scale=alt.Scale(domain=y_domain) if y_domain else alt.Undefined),
        "color": alt.Color(color, scale=alt.Scale(range=[C["pass"], C["fail"], C["info"], C["warn"]]), legend=alt.Legend(orient="bottom", direction="horizontal")) if color else alt.value(C["pass"]),
    }
    chart = alt.Chart(data).mark_line(point=True, strokeWidth=2) if mark == "line" else alt.Chart(data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    if stack: enc["y"] = alt.Y(y, title=None, stack="zero")
    built = chart.encode(**enc).properties(height=height, background=C["card"]).configure(**chart_style())
    st.altair_chart(built, width="stretch")


def bandit() -> None:
    page_header("🧠", "Bandit Learning")
    detection = pd.DataFrame({"run": list(range(1, 11)), "rate": [.61,.68,.72,.75,.79,.82,.85,.87,.90,.89]})
    a, b = st.columns(2, gap="medium")
    with a, card():
        st.markdown("<div style='display:flex;justify-content:space-between'><div class='tw-card-title'>Failure Detection Rate</div></div>", unsafe_allow_html=True); pill("Simulated", "info")
        chart = alt.Chart(detection).mark_line(color=C["pass"], point=alt.OverlayMarkDef(color=C["pass"], size=35), strokeWidth=2).encode(alt.X("run:Q", title="Run #", axis=alt.Axis(tickMinStep=1)), alt.Y("rate:Q", title=None, scale=alt.Scale(domain=[.5,1]), axis=alt.Axis(format=".0%")), tooltip=[alt.Tooltip("run:Q", title="Run"), alt.Tooltip("rate:Q", format=".1%")]).properties(height=180, background=C["card"]).configure(**chart_style())
        st.altair_chart(chart, width="stretch")
    exploit = pd.DataFrame({"run": list(range(1,11)), "Explore": [45,40,35,30,28,25,22,20,18,20], "Exploit": [55,60,65,70,72,75,78,80,82,80]}).melt("run", var_name="strategy", value_name="share")
    with b, card():
        _title("Explore vs Exploit Share")
        chart = alt.Chart(exploit).mark_bar().encode(alt.X("run:O", title=None), alt.Y("share:Q", title=None, stack="zero", axis=alt.Axis(labelExpr="datum.value + '%'")), alt.Color("strategy:N", scale=alt.Scale(domain=["Exploit","Explore"], range=[C["info"],C["warn"]]), legend=alt.Legend(orient="bottom")), tooltip=["run", "strategy", "share"]).properties(height=180, background=C["card"]).configure(**chart_style())
        st.altair_chart(chart, width="stretch")
    left, right = st.columns([1, .37], gap="medium")
    decisions = pd.DataFrame([
        (10,"test_preprocessing.py",.82,False,1),(10,"test_validation.py",.41,False,0),(10,"test_api.py",.07,True,0),(9,"test_preprocessing.py",.79,False,1),(9,"test_prediction.py",.22,True,0),(9,"test_validation.py",.38,False,0),(8,"test_preprocessing.py",.76,False,1),(8,"test_api.py",.09,True,1),(7,"test_validation.py",.35,False,0),(7,"test_prediction.py",.19,False,0)], columns=["Run","Test","Score","Explored?","Reward"])
    with left, card(): _title("Last 10 Decisions"); st.dataframe(decisions, hide_index=True, width="stretch", column_config={"Score":st.column_config.NumberColumn(format="%.2f"),"Explored?":st.column_config.CheckboxColumn()})
    with right, card():
        st.markdown("<div style='font-size:25px'>🎰</div>", unsafe_allow_html=True); _title("Exploration Strategy")
        st.markdown("TestWise uses an **epsilon-greedy bandit** to balance running high-ranked tests with occasionally exploring lower-ranked ones. Occasionally running a low-ranked test lets TestWise find failures its model might have missed — keeping rankings honest over time.")


def analytics() -> None:
    page_header("📊", "Analytics")
    a,b = st.columns([1,1]);
    with a: selected=st.radio("Range", ["7d","14d","30d"], horizontal=True, label_visibility="collapsed", key="analytics_range")
    with b: st.markdown("<div style='text-align:right;padding-top:8px;font-size:12px'>⌕　8 runs in range</div>", unsafe_allow_html=True)
    cols=st.columns(4,gap="small")
    for col,args in zip(cols,[("Tests Reduced","78%","avg per run",C["pass"]),("Time Saved","45%","vs full suite",C["info"]),("Failures Caught","87.5%","of all failures",C["warn"]),("Missed Failures","1","found in nightly",C["fail"])]):
        with col: metric(*args[:3],color=args[3])
    saved=pd.DataFrame({"Run":[f"Run {i}" for i in range(1,9)],"Saved":[1.2,.8,1.5,1.1,.9,1.4,1.3,1.6]})
    caught=pd.DataFrame({"Run":[f"Run {i}" for i in range(1,9)],"Caught":[1,0,1,0,1,1,2,1],"Missed":[0,0,0,0,0,1,0,0]}).melt("Run",var_name="Result",value_name="Count")
    left,right=st.columns(2,gap="medium")
    with left,card():
        _title("Time Saved per Run (s)")
        chart=alt.Chart(saved).mark_bar(color=C["pass"],cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(alt.X("Run:N",title=None),alt.Y("Saved:Q",title=None),tooltip=["Run",alt.Tooltip("Saved:Q",format=".1f")]).properties(height=180,background=C["card"]).configure(**chart_style()); st.altair_chart(chart,width="stretch")
    with right,card():
        _title("Failures Caught vs Missed")
        chart=alt.Chart(caught).mark_bar().encode(alt.X("Run:N",title=None),alt.Y("Count:Q",title=None,stack="zero"),alt.Color("Result:N",scale=alt.Scale(domain=["Caught","Missed"],range=[C["pass"],C["fail"]]),legend=alt.Legend(orient="bottom")),tooltip=["Run","Result","Count"]).properties(height=180,background=C["card"]).configure(**chart_style()); st.altair_chart(chart,width="stretch")
    history=pd.DataFrame([("run-8","2025-01-14 11:22","5/25","1.1 s",1,"No"),("run-7","2025-01-13 18:45","7/25","1.4 s",2,"No"),("run-6","2025-01-12 14:10","6/25","1.2 s",1,"No"),("run-5","2025-01-11 09:30","8/25","1.6 s",1,"No"),("run-4","2025-01-10 16:55","25/25","2.0 s",0,"Yes")],columns=["Run","Date","Tests","Time","Failures","Fallback"])
    with card(): _title("Run History"); st.dataframe(history,hide_index=True,width="stretch")


def offline() -> None:
    page_header("🔬", "Offline Evaluation (LRTS)")
    with card():
        a,b=st.columns([4,1]);
        with a: st.markdown("<h3 style='margin:0;color:#E8EAF0'>Benchmark on the LRTS Dataset</h3><div style='margin-top:4px'>tvm-cpu · time-based split · <span class='tw-mono' style='color:#E8EAF0'>82</span> failing test runs</div>",unsafe_allow_html=True)
        with b: pill("Research Lab","info")
    cols=st.columns(4,gap="small")
    for col,args in zip(cols,[("APFD","0.936","Area under ranked curve",C["pass"]),("Failing Runs @ 20%","98.8%","failing runs caught",C["info"]),("Failures @ 20%","76.9%","individual failures",C["warn"]),("Time Saved @ 20%","33.5%","of total run time",C["orange"])]):
        with col: metric(*args[:3],color=args[3])
    budgets=pd.DataFrame({"Budget":["5%","10%","20%","30%"],"Failures caught %":[57,68.7,76.9,84.7],"Failing runs hit %":[86.6,89,98.8,100],"Time saved %":[65.8,51.1,33.5,24.4]}).melt("Budget",var_name="Metric",value_name="Percent")
    apfd=pd.DataFrame({"Model":["Random","History + Overlap","History only","Code only","All features"],"APFD":[.472,.896,.879,.686,.936],"Color":[C["muted"],C["info"],"#9B59B6",C["warn"],C["pass"]]})
    left,right=st.columns(2,gap="medium")
    with left,card():
        _title("Run Top k% of Tests")
        chart=alt.Chart(budgets).mark_line(point=True,strokeWidth=2).encode(alt.X("Budget:N",title=None),alt.Y("Percent:Q",title=None,scale=alt.Scale(domain=[0,110]),axis=alt.Axis(labelExpr="datum.value + '%'")),alt.Color("Metric:N",scale=alt.Scale(domain=["Failures caught %","Failing runs hit %","Time saved %"],range=[C["warn"],C["info"],C["pass"]]),legend=alt.Legend(orient="bottom")),tooltip=["Budget","Metric",alt.Tooltip("Percent:Q",format=".1f")]).properties(height=220,background=C["card"]).configure(**chart_style()); st.altair_chart(chart,width="stretch")
    with right,card():
        _title("APFD by Model")
        chart=alt.Chart(apfd).mark_bar(cornerRadiusTopRight=4,cornerRadiusBottomRight=4).encode(alt.X("APFD:Q",title=None,scale=alt.Scale(domain=[0,1])),alt.Y("Model:N",title=None,sort="-x"),alt.Color("Color:N",scale=None,legend=None),tooltip=["Model",alt.Tooltip("APFD:Q",format=".3f")]).properties(height=220,background=C["card"]).configure(**chart_style()); st.altair_chart(chart,width="stretch")
        with st.expander("ⓘ What is APFD?"):
            st.write("Average Percentage of Faults Detected measures how quickly a test ordering detects failures. Range: 0–1, higher is better. Random baseline ≈ 0.5.")
    callout("The <b>All features</b> model (APFD 0.936) combines code change overlap, test name matching, historical failure rates, and test duration — outperforming all ablations and the random baseline by a significant margin.")


def settings() -> None:
    page_header("⚙️", "Settings & Safety")
    left,right=st.columns(2,gap="large")
    with left:
        with card():
            _title("Selection Policy")
            policy=st.radio("Policy",["Top-N Tests","Time Budget"],horizontal=True,label_visibility="collapsed",key="settings_policy")
            if policy=="Top-N Tests": st.slider("Run top tests",1,25,13,key="settings_topn")
            else: st.slider("Budget (seconds)",.5,5.,1.5,.5,key="settings_budget")
        with card():
            _title("Exploration Rate"); st.caption("How often to run a randomly-selected test outside the predicted top-N."); st.slider("Runs explore",0,30,10,format="%d%%",key="settings_explore")
    with right:
        with card():
            _title("Safety Toggles")
            for label,desc,key,default in [("Always run full suite on main","Prevents false confidence on production branch","safe_main",True),("Full suite when requirements/config change","requirements.txt, setup.cfg, pyproject.toml, etc.","safe_config",True),("Nightly full-suite run","Scheduled 02:00 UTC to catch missed failures","safe_nightly",True),("Fall back to full suite if API fails","On by default — never silently skip tests","safe_fallback",True)]:
                st.toggle(label, value=default, key=key, help=desc)
                st.caption(desc)
        st.markdown(f"<div style='border:1px solid #ff4b4b4d;background:#ff4b4b10;border-radius:12px;padding:16px'><div style='font:700 14px Outfit,sans-serif;color:{C['fail']}'>🛡️ &nbsp;Danger Zone</div><p style='font-size:12px'>Resetting history will erase all run data, scores, and bandit state. The model will restart from scratch. This action cannot be undone.</p></div>",unsafe_allow_html=True)
        if not st.session_state.get("reset_confirm"):
            if st.button("Reset History", key="reset_history"): st.session_state["reset_confirm"]=True; st.rerun()
        else:
            a,b=st.columns(2)
            with a:
                if st.button("Cancel",key="reset_cancel"): st.session_state["reset_confirm"]=False; st.rerun()
            with b:
                if st.button("Yes, reset all history",type="primary",key="reset_yes"): st.session_state["reset_confirm"]=False; st.rerun()


PAGES = {
    "Home": (":material/home:", home), "Repository": (":material/account_tree:", repository),
    "Code Changes": (":material/code:", code_changes), "Test Prediction": (":material/science:", prediction),
    "Test Execution": (":material/play_arrow:", execution), "Bandit Learning": (":material/psychology:", bandit),
    "Analytics": (":material/analytics:", analytics), "Offline Evaluation": (":material/storage:", offline),
    "Settings & Safety": (":material/settings:", settings),
}

