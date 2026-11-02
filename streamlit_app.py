"""Streamlit UI for autonomous data analysis pipeline."""

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix

from config import settings
from services.run_store import RunStore
from tools.data_loader import DataLoaderTool

# Page configuration
st.set_page_config(
    page_title="Data Analysis Pipeline",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = settings.DATA_PIPELINE_API_URL
BACKEND_MODE = os.getenv("DATA_PIPELINE_MODE", "auto").lower()


@st.cache_resource
def get_run_store() -> RunStore:
    """Create a shared run store for embedded mode."""
    return RunStore()


@st.cache_resource
def get_copilot_agent() -> Any:
    """Create the analytics copilot once per app process."""
    try:
        from agents.copilot import AnalyticsCopilotAgent
    except ModuleNotFoundError as exc:
        missing_pkg = exc.name or "required package"
        raise RuntimeError(
            f"Copilot dependencies are missing: {missing_pkg}. Install requirements.txt before using the copilot."
        ) from exc

    return AnalyticsCopilotAgent()


def _api_health_check() -> tuple[bool, str]:
    """Validate that the configured API URL points to this project backend."""
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    if health_response.status_code != 200:
        return False, f"Health check failed with status {health_response.status_code}"

    root_response = requests.get(f"{API_URL}/", timeout=2)
    if root_response.status_code != 200:
        return False, "Root endpoint is unavailable"

    payload = root_response.json()
    service_name = payload.get("service")
    if service_name != "Data Analysis Pipeline API":
        return False, f"Connected to the wrong service at {API_URL}"

    return True, "API Connected"


def resolve_backend_status() -> tuple[bool, str, str]:
    """Resolve whether the UI should use a hosted API or embedded execution."""
    if BACKEND_MODE == "embedded":
        return True, "Embedded backend active", "embedded"

    try:
        api_ok, api_message = _api_health_check()
        if api_ok:
            return True, api_message, "api"
    except Exception as exc:
        api_message = str(exc)

    if BACKEND_MODE == "auto":
        return True, "Embedded backend active", "embedded"

    return False, api_message, "api"


def save_uploaded_file(uploaded_file) -> Path:
    """Persist an uploaded file so the workflow can read it."""
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    file_path = uploads_dir / f"{uuid4().hex[:8]}_{safe_name}"
    file_path.write_bytes(uploaded_file.getvalue())
    return file_path


def run_embedded_analysis(uploaded_file, options: dict) -> dict:
    """Run the workflow directly inside Streamlit and persist the saved run."""
    try:
        from graph.workflow import run_analysis_sync
    except ModuleNotFoundError as exc:
        missing_pkg = exc.name or "required package"
        raise RuntimeError(
            f"Embedded analysis dependencies are missing: {missing_pkg}. Install requirements.txt before running analysis."
        ) from exc

    file_path = save_uploaded_file(uploaded_file)
    result = run_analysis_sync(str(file_path), uploaded_file.name, options=options)
    saved_run = get_run_store().save_run(
        file_name=uploaded_file.name,
        file_path=str(file_path),
        options=options,
        result=result,
    )
    result["run_id"] = saved_run["run_id"]
    result["workspace_summary"] = saved_run["summary"]
    return result


def inject_custom_css():
    """Apply a more intentional visual system to the Streamlit app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #0f1417;
            --panel: rgba(22, 30, 35, 0.94);
            --panel-strong: #182127;
            --ink: #edf3ef;
            --muted: #9cadb0;
            --line: rgba(237, 243, 239, 0.10);
            --accent: #ff9347;
            --accent-deep: #ffc08f;
            --accent-soft: rgba(255, 147, 71, 0.12);
            --sage: rgba(93, 191, 140, 0.16);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 147, 71, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(93, 191, 140, 0.12), transparent 24%),
                linear-gradient(180deg, #11181c 0%, var(--bg) 100%);
            color: var(--ink);
        }

        .stApp, .stApp * {
            color: var(--ink);
        }

        h1, h2, h3, h4 {
            font-family: "Space Grotesk", sans-serif;
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        p, li, div, label, span {
            font-family: "Space Grotesk", sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(17, 24, 28, 0.98), rgba(14, 20, 24, 0.98));
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink) !important;
        }

        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stCheckbox label,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p {
            color: var(--ink) !important;
        }

        .stRadio label, .stCheckbox label, .stSelectbox label, .stFileUploader label,
        .stTextInput label, .stTextArea label, .stMultiSelect label {
            color: var(--ink) !important;
            font-weight: 600;
        }

        .stRadio div[role="radiogroup"] label,
        .stCheckbox div[role="checkbox"] + label {
            color: var(--ink) !important;
        }

        .stRadio > div {
            background: rgba(28, 38, 44, 0.96);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.45rem 0.55rem;
        }

        .stRadio [role="radiogroup"] > label {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(237, 243, 239, 0.08);
            border-radius: 12px;
            padding: 0.55rem 0.7rem;
            margin-bottom: 0.45rem;
        }

        .stRadio [role="radiogroup"] > label:hover {
            background: rgba(255, 147, 71, 0.08);
            border-color: rgba(255, 147, 71, 0.22);
        }

        .stRadio [role="radiogroup"] p {
            color: var(--ink) !important;
            font-weight: 600;
        }

        .stRadio input:checked + div p,
        .stRadio input:checked + div span {
            color: var(--accent) !important;
            font-weight: 700;
        }

        .stSelectbox > div[data-baseweb="select"] > div {
            background: rgba(28, 38, 44, 0.96) !important;
            border: 1px solid var(--line) !important;
            border-radius: 14px !important;
            color: var(--ink) !important;
        }

        .stSelectbox div[data-baseweb="select"] * {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }

        .stSelectbox input,
        .stSelectbox div[role="combobox"],
        .stSelectbox span,
        .stSelectbox p {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }

        .stFileUploader > div {
            background: var(--panel) !important;
            border: 1px solid var(--line) !important;
            border-radius: 18px !important;
        }

        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            background: var(--panel-strong) !important;
            border: 2px dashed rgba(255, 147, 71, 0.30) !important;
            color: var(--ink) !important;
        }

        .stFileUploader [data-testid="stFileUploaderDropzone"] * {
            color: var(--ink) !important;
        }

        .stFileUploader button,
        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(180deg, var(--accent), var(--accent-deep)) !important;
            color: #101518 !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 18px rgba(0, 0, 0, 0.18);
        }

        .stFileUploader button:hover,
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: linear-gradient(180deg, #ff9f61, #ffc08f) !important;
            color: #0f1417 !important;
        }

        .stButton > button:disabled {
            background: #b8ada4 !important;
            color: #f8f1ea !important;
        }

        [data-testid="stTabs"] button {
            color: var(--muted) !important;
            background: rgba(255, 255, 255, 0.03) !important;
            border-radius: 12px 12px 0 0 !important;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent) !important;
            background: rgba(255, 147, 71, 0.10) !important;
            font-weight: 700 !important;
        }

        .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
            color: var(--ink) !important;
        }

        .stDataFrame, .stTable {
            color: var(--ink) !important;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: var(--ink) !important;
        }

        code {
            font-family: "IBM Plex Mono", monospace !important;
            color: var(--accent-deep) !important;
            background: rgba(255, 147, 71, 0.12) !important;
            padding: 0.14rem 0.38rem !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 147, 71, 0.14) !important;
        }

        .hero-panel {
            background: linear-gradient(135deg, rgba(24, 33, 38, 0.97), rgba(17, 24, 28, 0.95));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
            margin-bottom: 1rem;
        }

        .hero-kicker {
            color: var(--accent-deep);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .hero-title {
            font-size: 2.55rem;
            line-height: 1.02;
            margin: 0.35rem 0 0.6rem 0;
            font-weight: 700;
        }

        .hero-copy {
            color: var(--muted);
            max-width: 56rem;
            font-size: 1rem;
        }

        .metric-card, .section-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--ink);
            line-height: 1;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 0.45rem;
        }

        .chip-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin: 0.6rem 0 1rem 0;
        }

        .chip {
            background: rgba(255, 147, 71, 0.12);
            color: var(--ink);
            border: 1px solid rgba(255, 147, 71, 0.16);
            border-radius: 999px;
            padding: 0.35rem 0.72rem;
            font-size: 0.84rem;
            font-weight: 600;
        }

        .chip-success {
            background: rgba(93, 191, 140, 0.14);
            border-color: rgba(93, 191, 140, 0.16);
        }

        .sample-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.8rem;
        }

        .sample-pill {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(237, 243, 239, 0.09);
            color: var(--ink);
            border-radius: 999px;
            padding: 0.38rem 0.72rem;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            color: var(--ink);
        }

        .section-copy {
            color: var(--muted);
            margin-bottom: 0.9rem;
        }

        .section-card code,
        .section-card span,
        .section-card p,
        .section-card div {
            color: var(--ink) !important;
        }

        .section-card code {
            color: var(--accent-deep) !important;
            background: rgba(255, 147, 71, 0.12) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fetch_runs(backend_mode: str) -> list[dict]:
    """Fetch saved workspace runs from the active backend."""
    if backend_mode == "embedded":
        return [
            {
                "run_id": run["run_id"],
                "created_at": run["created_at"],
                "file_name": run["file_name"],
                "summary": run.get("summary", {}),
                "options": run.get("options", {}),
            }
            for run in get_run_store().list_runs()
        ]

    try:
        response = requests.get(f"{API_URL}/runs", timeout=5)
        if response.status_code == 200:
            return response.json().get("runs", [])
    except Exception:
        return []
    return []


def fetch_run_details(run_id: str, backend_mode: str) -> dict | None:
    """Fetch one saved run."""
    if backend_mode == "embedded":
        return get_run_store().get_run(run_id)

    try:
        response = requests.get(f"{API_URL}/runs/{run_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


def ask_copilot(question: str, run_id: str | None, backend_mode: str) -> dict:
    """Ask the analytics copilot about a selected run."""
    if backend_mode == "embedded":
        run_store = get_run_store()
        run_record = run_store.get_run(run_id) if run_id else run_store.latest_run()
        if not run_record:
            return {"status": "error", "answer": "⚠️ No saved run context was available for this dataset."}
        answer = get_copilot_agent().answer(question, run_record)
        return {
            "run_id": run_record["run_id"],
            "file_name": run_record["file_name"],
            **answer,
        }

    try:
        response = requests.post(
            f"{API_URL}/copilot/ask",
            json={"question": question, "run_id": run_id},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return {"status": "error", "answer": "⚠️ No saved run was found on the server. Please run or select an analysis first."}
        else:
            return {"status": "error", "answer": f"⚠️ Server error ({response.status_code}): {response.text}"}
    except requests.exceptions.Timeout:
        return {"status": "error", "answer": "⚠️ Copilot request timed out after 60s. The AI model is taking too long to respond."}
    except Exception as exc:
        return {"status": "error", "answer": f"⚠️ Could not reach backend server at {API_URL}: {exc}"}


def load_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    """Load an uploaded file into a dataframe for frontend chart rendering."""
    return DataLoaderTool.load_uploaded_file(uploaded_file)


def render_correlation_heatmap(df: pd.DataFrame, key: str | None = None):
    """Render an interactive correlation heatmap using Plotly."""
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        st.info("At least 2 numeric columns are required to compute a correlation heatmap.")
        return
    corr = numeric_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pairwise Pearson Correlation Heatmap",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf3ef"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_box_plot(df: pd.DataFrame, column: str, group_by: str | None = None, is_violin: bool = False, key: str | None = None):
    """Render an interactive box or violin plot for distribution and outlier inspection."""
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        st.info(f"No numeric values available for `{column}`.")
        return
    plot_df = df.copy()
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[column])
    
    if is_violin:
        fig = px.violin(
            plot_df,
            y=column,
            x=group_by,
            color=group_by,
            box=True,
            points="outliers",
            title=f"Violin & Kernel Density Plot: {column}",
        )
    else:
        fig = px.box(
            plot_df,
            y=column,
            x=group_by,
            color=group_by,
            points="outliers",
            title=f"Box Plot & Outlier Fences: {column}",
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf3ef"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_pie_chart(df: pd.DataFrame, column: str, key: str | None = None):
    """Render an interactive Donut / Pie chart for qualitative feature breakdown."""
    counts = df[column].astype(str).fillna("Missing").value_counts().head(10)
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.45,
        title=f"Category Distribution: {column}",
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf3ef"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_histogram_like_chart(df: pd.DataFrame, column: str, key: str | None = None):
    """Render an interactive distribution histogram with Plotly."""
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        st.info(f"No numeric data available for `{column}`.")
        return
    fig = px.histogram(
        df,
        x=column,
        marginal="box",
        nbins=min(30, max(10, int(series.nunique() ** 0.5))),
        title=f"Distribution Histogram: {column}",
        color_discrete_sequence=["#ff9347"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf3ef"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_scatter_chart(df: pd.DataFrame, x_col: str, y_col: str, color_col: str | None = None, key: str | None = None):
    """Render an interactive scatter chart with OLS trendline."""
    chart_df = df[[x_col, y_col] + ([color_col] if color_col else [])].copy()
    chart_df[x_col] = pd.to_numeric(chart_df[x_col], errors="coerce")
    chart_df[y_col] = pd.to_numeric(chart_df[y_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[x_col, y_col]).head(2000)

    if chart_df.empty:
        st.info(f"No usable numeric pairs found for `{x_col}` vs `{y_col}`.")
        return

    trendline = "ols" if len(chart_df) >= 4 else None
    try:
        fig = px.scatter(
            chart_df,
            x=x_col,
            y=y_col,
            color=color_col,
            trendline=trendline,
            title=f"Bivariate Scatter Plot: {x_col} vs {y_col}",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
    except Exception:
        fig = px.scatter(
            chart_df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"Bivariate Scatter Plot: {x_col} vs {y_col}",
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf3ef"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_categorical_bar_chart(df: pd.DataFrame, column: str, key: str | None = None):
    """Render category frequencies with Plotly."""
    counts = df[column].astype(str).fillna("Missing").value_counts().head(15).reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(
        counts,
        x=column,
        y="count",
        text="count",
        title=f"Category Frequencies: {column}",
        color="count",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf3ef"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_missingness_chart(profile: dict, key: str | None = None):
    """Render missing-value percentages from profile output."""
    missing = profile.get("missing_percentage", {})
    if not missing:
        st.info("No missing-value summary available.")
        return

    chart_df = (
        pd.DataFrame(
            [{"column": column, "missing_pct": round(value, 2)} for column, value in missing.items()]
        )
        .sort_values("missing_pct", ascending=False)
    )
    fig = px.bar(
        chart_df,
        x="column",
        y="missing_pct",
        text="missing_pct",
        title="Missing Value Percentage by Column (%)",
        color="missing_pct",
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf3ef"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)



def render_metric_card(label: str, value: str, note: str = ""):
    """Render a styled metric summary card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chip_row(items: list[str], success: bool = False):
    """Render rounded workflow chips."""
    chip_class = "chip chip-success" if success else "chip"
    chips = "".join([f'<span class="{chip_class}">{item}</span>' for item in items])
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def render_hero(api_connected: bool, api_message: str, backend_mode: str):
    """Render the app hero section."""
    status = "Connected" if api_connected else "Backend Needed"
    backend_chip = (
        f"Backend: Embedded"
        if backend_mode == "embedded"
        else f"Backend: API ({API_URL})"
    )
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-kicker">LangGraph Analysis Studio</div>
            <div class="hero-title">From raw file to routed multi-agent report.</div>
            <div class="hero-copy">
                Upload structured data, route it through a LangGraph workflow, inspect anomalies,
                generate LLM-backed insights, and review a saved markdown report. This frontend is
                designed to help you learn the pipeline while using it.
            </div>
            <div class="chip-row">
                <span class="chip">{status}</span>
                <span class="chip">{backend_chip}</span>
                <span class="chip">{api_message}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_custom_css()
    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = []
    api_connected, api_message, backend_mode = resolve_backend_status()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        if backend_mode == "embedded":
            st.caption("Backend Mode: Embedded")
        else:
            st.caption(f"API URL: {API_URL}")
        
        analysis_type = st.radio(
            "Analysis Type",
            ["Quick Profile", "Deep Analysis", "Comparative Analysis"],
        )
        
        include_anomalies = st.checkbox("Detect Anomalies", value=True)
        include_insights = st.checkbox("Generate Insights", value=True)
        include_visualizations = st.checkbox("Create Visualizations", value=True)
        include_report = st.checkbox("Generate Report", value=True)
        
        st.divider()
        
        # Backend status
        if api_connected:
            success_label = "✅ Embedded backend ready" if backend_mode == "embedded" else "✅ API Connected"
            st.success(success_label)
        else:
            st.error(f"❌ {api_message}")

        st.divider()
        st.subheader("Workspace Runs")
        runs = fetch_runs(backend_mode) if api_connected else []
        if runs:
            run_options = {
                f"{run['file_name']} | {run['created_at'][:19]}": run["run_id"]
                for run in runs[:15]
            }
            selected_run_label = st.selectbox(
                "Saved analyses",
                options=list(run_options.keys()),
                key="run_selector",
            )
            selected_run_id = run_options[selected_run_label]
            if st.button("Load Selected Run", use_container_width=True):
                run_record = fetch_run_details(selected_run_id, backend_mode)
                if run_record:
                    st.session_state.analysis_result = {
                        **run_record.get("result", {}),
                        "run_id": run_record.get("run_id"),
                        "workspace_summary": run_record.get("summary", {}),
                    }
                    st.session_state.last_uploaded_name = run_record.get("file_name")
                    st.session_state.selected_run_id = run_record.get("run_id")
                    file_path = run_record.get("file_path")
                    if file_path and Path(file_path).exists():
                        try:
                            st.session_state.analysis_dataframe = DataLoaderTool.load_file(file_path)
                        except Exception:
                            pass
                    st.session_state.copilot_messages = []
        else:
            st.caption("No saved runs yet. Upload data to create your first workspace.")

    render_hero(api_connected, api_message, backend_mode)

    top_metrics = st.columns(4)
    with top_metrics[0]:
        render_metric_card("Mode", analysis_type, "Choose the routing depth for this run")
    with top_metrics[1]:
        enabled_count = sum(
            [include_anomalies, include_insights, include_visualizations, include_report]
        )
        render_metric_card("Modules", str(enabled_count), "Enabled workflow modules")
    with top_metrics[2]:
        format_note = "Frontend preview + embedded workflow" if backend_mode == "embedded" else "Frontend preview + API upload"
        render_metric_card("Formats", "CSV / JSON / XLSX / TXT", format_note)
    with top_metrics[3]:
        render_metric_card(
            "Learning Focus",
            "LangGraph",
            "Routing, retries, state transitions, and outputs",
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Upload Workspace</div>
                <div class="section-copy">
                    Bring in a dataset, preview its shape, and push it through the analysis graph.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Choose a CSV, JSON, XLSX, or TXT file",
            type=["csv", "json", "xlsx", "txt"],
            help="Max 1024MB"
        )
        
        if uploaded_file:
            st.success(f"File selected: {uploaded_file.name}")
            
            # File preview
            with st.expander("Preview Data"):
                try:
                    preview_df = load_uploaded_dataframe(uploaded_file).head(10)
                    st.dataframe(preview_df, use_container_width=True)
                    st.caption(f"Showing {len(preview_df)} preview rows")
                except Exception as e:
                    st.error(f"Cannot preview: {e}")
            
            # Analyze button
            if st.button(
                "Start Analysis",
                type="primary",
                use_container_width=True,
                disabled=not api_connected,
            ):
                with st.spinner("Analyzing data... This may take a moment."):
                    try:
                        options = {
                            "analysis_type": analysis_type,
                            "include_anomalies": include_anomalies,
                            "include_insights": include_insights,
                            "include_visualizations": include_visualizations,
                            "include_report": include_report,
                        }
                        if backend_mode == "embedded":
                            result = run_embedded_analysis(uploaded_file, options)
                        else:
                            response = requests.post(
                                f"{API_URL}/analyze",
                                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                                data={
                                    "analysis_type": analysis_type,
                                    "include_anomalies": str(include_anomalies).lower(),
                                    "include_insights": str(include_insights).lower(),
                                    "include_visualizations": str(include_visualizations).lower(),
                                    "include_report": str(include_report).lower(),
                                },
                                timeout=30,
                            )
                            if response.status_code != 200:
                                try:
                                    error_payload = response.json()
                                except Exception:
                                    error_payload = {"detail": response.text}
                                st.error(f"Analysis failed: {error_payload.get('detail', response.text)}")
                                result = None
                            else:
                                result = response.json()

                        if result:
                            st.session_state.analysis_dataframe = load_uploaded_dataframe(uploaded_file)
                            st.session_state.analysis_result = result
                            st.session_state.last_uploaded_name = uploaded_file.name
                            st.session_state.selected_run_id = result.get("run_id")
                            st.session_state.copilot_messages = []
                            st.success("Analysis completed!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            if not api_connected and backend_mode == "api":
                st.warning("Start the FastAPI backend first, then retry.")
    
    with col2:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Control Panel</div>
                <div class="section-copy">
                    Active analysis mode and enabled workflow modules.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_chip_row([analysis_type])
        selected_modules = []
        if include_anomalies:
            selected_modules.append("Anomaly Detection")
        if include_insights:
            selected_modules.append("Insight Generation")
        if include_visualizations:
            selected_modules.append("Visualizations")
        if include_report:
            selected_modules.append("Report Generation")
        render_chip_row(selected_modules or ["No modules enabled"])
    
    st.divider()
    
    # Results tabs
    if "analysis_result" in st.session_state:
        result = st.session_state.analysis_result
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Analysis Results</div>
                <div class="section-copy">
                    Review the routed graph output, the generated report, and the visual layer in one place.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        overview_cols = st.columns(4)
        profile = result.get("profile", {})
        anomalies = result.get("anomalies", {})
        report = result.get("report", {})
        with overview_cols[0]:
            render_metric_card(
                "Dataset",
                st.session_state.get("last_uploaded_name", "Current run"),
                "Source analyzed in the latest request",
            )
        with overview_cols[1]:
            render_metric_card(
                "Rows",
                f"{profile.get('shape', {}).get('rows', 0):,}",
                "Loaded into the workflow",
            )
        with overview_cols[2]:
            render_metric_card(
                "Flagged Rows",
                f"{anomalies.get('total_anomalous_rows', 0):,}",
                "Potential anomalies found",
            )
        with overview_cols[3]:
            render_metric_card(
                "Report Mode",
                report.get("generation_mode", "n/a"),
                "LLM-backed or fallback output",
            )
        workspace_summary = result.get("workspace_summary", {})
        if result.get("run_id"):
            st.markdown("**Workspace Summary**")
            render_chip_row(
                [
                    f"Run ID: {result['run_id']}",
                    f"Rows: {workspace_summary.get('rows', profile.get('shape', {}).get('rows', 0)):,}",
                    f"Flagged: {workspace_summary.get('flagged_rows', anomalies.get('total_anomalous_rows', 0)):,}",
                ]
            )
        if result.get("completed_steps"):
            st.markdown("**Executed Workflow Steps**")
            render_chip_row(result.get("completed_steps", []), success=True)
            with st.expander("Workflow Execution Trace", expanded=False):
                st.write("Messages:")
                for message in result.get("messages", []):
                    st.write(f"- {message}")

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Analytics Copilot</div>
                <div class="section-copy">
                    Ask follow-up questions about the selected run, such as what to clean first,
                    which anomalies matter, or what actions should come next.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        copilot_cols = st.columns([3, 1])
        with copilot_cols[0]:
            question = st.text_input(
                "Ask the copilot about this run",
                placeholder="What should I investigate first in this dataset?",
                key="copilot_question",
            )
        with copilot_cols[1]:
            if st.button("Ask Copilot", use_container_width=True):
                if question.strip():
                    reply = ask_copilot(
                        question,
                        result.get("run_id") or st.session_state.get("selected_run_id"),
                        backend_mode,
                    )
                    if reply and reply.get("answer"):
                        st.session_state.copilot_messages.append(
                            {"role": "user", "content": question}
                        )
                        st.session_state.copilot_messages.append(
                            {"role": "assistant", "content": reply.get("answer", ""), "mode": reply.get("mode", "")}
                        )
                    else:
                        st.error("Copilot request failed. No response received from server.")

        for message in st.session_state.get("copilot_messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("mode"):
                    st.caption(f"Mode: {message['mode']}")
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            [
                "Profile",
                "Anomalies",
                "Visualizations & Heatmaps",
                "Insights",
                "Diagnostic Tests",
                "Predictive ML",
                "Report",
            ]
        )
        
        with tab1:
            st.subheader("Data Profile")
            profile = result.get("profile", {})
            if profile:
                shape = profile.get("shape", {})
                metric_cols = st.columns(3)
                metric_cols[0].metric("Rows", f"{shape.get('rows', 0):,}")
                metric_cols[1].metric("Columns", f"{shape.get('columns', 0)}")
                metric_cols[2].metric("Duplicates", f"{profile.get('duplicate_rows', 0):,}")

                missing_pct = profile.get("missing_percentage", {})
                missing_df = pd.DataFrame(
                    [{"column": col, "missing_pct": round(pct, 2)} for col, pct in missing_pct.items()]
                ).sort_values("missing_pct", ascending=False)
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.info("No profiling results returned.")
        
        with tab2:
            st.subheader("Anomaly Detection")
            anomalies = result.get("anomalies", {})
            if anomalies:
                metric_cols = st.columns(3)
                metric_cols[0].metric("Flagged Rows", f"{anomalies.get('total_anomalous_rows', 0):,}")
                metric_cols[1].metric("IQR Flags", f"{anomalies.get('methods', {}).get('iqr', {}).get('total_anomalies', 0):,}")
                metric_cols[2].metric("Z-score Flags", f"{anomalies.get('methods', {}).get('zscore', {}).get('total_anomalies', 0):,}")

                top_columns = anomalies.get("top_columns", [])
                if top_columns:
                    st.dataframe(pd.DataFrame(top_columns), use_container_width=True)
                st.text(anomalies.get("summary", ""))
            else:
                st.info("No anomaly results returned.")
        
        with tab3:
            st.subheader("Visualizations & Interactive Heatmaps")
            visualizations = result.get("visualizations", {})
            completed_steps = result.get("completed_steps", [])
            analysis_df = st.session_state.get("analysis_dataframe")

            if analysis_df is not None:
                numeric_cols = list(analysis_df.select_dtypes(include=["number"]).columns)
                categorical_cols = list(analysis_df.select_dtypes(include=["object", "category"]).columns)

                # 1. Interactive Studio Selector
                with st.expander("🎨 Interactive Chart Studio (Custom Visualizations)", expanded=True):
                    studio_cols = st.columns([2, 2, 2])
                    with studio_cols[0]:
                        chart_choice = st.selectbox(
                            "Choose Chart Type",
                            [
                                "Correlation Heatmap",
                                "Box Plot (Outlier Analysis)",
                                "Violin Plot (KDE Density)",
                                "Distribution Histogram",
                                "Bivariate Scatter Plot (with OLS Trendline)",
                                "Donut / Pie Chart (Proportions)",
                                "Category Bar Chart",
                            ],
                            key="studio_chart_type",
                        )
                    
                    if chart_choice == "Correlation Heatmap":
                        render_correlation_heatmap(analysis_df, key="studio_heatmap")
                    elif chart_choice in ["Box Plot (Outlier Analysis)", "Violin Plot (KDE Density)"]:
                        if numeric_cols:
                            with studio_cols[1]:
                                num_col = st.selectbox("Numeric Column", numeric_cols, key="box_num_col")
                            with studio_cols[2]:
                                grp_col = st.selectbox("Optional Grouping (Category)", ["None"] + categorical_cols, key="box_grp_col")
                            render_box_plot(
                                analysis_df,
                                num_col,
                                group_by=None if grp_col == "None" else grp_col,
                                is_violin=(chart_choice == "Violin Plot (KDE Density)"),
                                key="studio_box_plot",
                            )
                        else:
                            st.info("No numeric columns found for box/violin plot.")
                    elif chart_choice == "Distribution Histogram":
                        if numeric_cols:
                            with studio_cols[1]:
                                num_col = st.selectbox("Select Column", numeric_cols, key="hist_num_col")
                            render_histogram_like_chart(analysis_df, num_col, key="studio_hist_plot")
                        else:
                            st.info("No numeric columns found for histogram.")
                    elif chart_choice == "Bivariate Scatter Plot (with OLS Trendline)":
                        if len(numeric_cols) >= 2:
                            with studio_cols[1]:
                                x_c = st.selectbox("X-Axis (Independent)", numeric_cols, index=0, key="scatter_x")
                            with studio_cols[2]:
                                y_c = st.selectbox("Y-Axis (Dependent)", numeric_cols, index=1, key="scatter_y")
                            color_c = st.selectbox("Color Grouping", ["None"] + categorical_cols, key="scatter_color")
                            render_scatter_chart(
                                analysis_df,
                                x_c,
                                y_c,
                                color_col=None if color_c == "None" else color_c,
                                key="studio_scatter_plot",
                            )
                        else:
                            st.info("At least 2 numeric columns are required for a scatter plot.")
                    elif chart_choice == "Donut / Pie Chart (Proportions)":
                        if categorical_cols:
                            with studio_cols[1]:
                                cat_col = st.selectbox("Select Categorical Column", categorical_cols, key="pie_cat_col")
                            render_pie_chart(analysis_df, cat_col, key="studio_pie_chart")
                        else:
                            st.info("No categorical columns found for pie chart.")
                    elif chart_choice == "Category Bar Chart":
                        if categorical_cols:
                            with studio_cols[1]:
                                cat_col = st.selectbox("Select Column", categorical_cols, key="bar_cat_col")
                            render_categorical_bar_chart(analysis_df, cat_col, key="studio_bar_chart")
                        else:
                            st.info("No categorical columns found for bar chart.")

                st.divider()
                st.markdown("### Automated Pipeline Recommendations")
                charts = visualizations.get("recommended_charts", [])
                if charts:
                    for idx, chart in enumerate(charts):
                        chart_type = chart.get("chart_type")
                        columns = chart.get("columns", [])
                        st.markdown(f"**{chart_type.replace('_', ' ').title()}**")
                        st.caption(chart.get("reason", ""))

                        if chart_type == "correlation_heatmap":
                            render_correlation_heatmap(analysis_df, key=f"rec_heatmap_{idx}")
                        elif chart_type == "box_plot" and columns:
                            render_box_plot(analysis_df, columns[0], key=f"rec_box_{idx}")
                        elif chart_type == "pie_chart" and columns:
                            render_pie_chart(analysis_df, columns[0], key=f"rec_pie_{idx}")
                        elif chart_type == "summary_card":
                            metrics = st.columns(3)
                            metrics[0].metric("Rows", f"{profile.get('shape', {}).get('rows', 0):,}")
                            metrics[1].metric("Columns", f"{profile.get('shape', {}).get('columns', 0)}")
                            metrics[2].metric(
                                "Flagged Rows",
                                f"{result.get('anomalies', {}).get('total_anomalous_rows', 0):,}",
                            )
                        elif chart_type == "missingness_bar":
                            render_missingness_chart(profile, key=f"rec_missing_{idx}")
                        elif chart_type == "histogram" and columns:
                            render_histogram_like_chart(analysis_df, columns[0], key=f"rec_hist_{idx}")
                        elif chart_type == "scatter" and len(columns) >= 2:
                            render_scatter_chart(analysis_df, columns[0], columns[1], key=f"rec_scatter_{idx}")
                        elif chart_type == "bar" and columns:
                            render_categorical_bar_chart(analysis_df, columns[0], key=f"rec_bar_{idx}")
                else:
                    st.info("No automated chart recommendations recorded.")
            else:
                st.info("Upload a dataset or load a run to render interactive charts and heatmaps.")
        
        with tab4:
            st.subheader("Generated Insights")
            insights = result.get("insights", {})
            if insights:
                st.markdown(f"**Summary:** {insights.get('executive_summary', 'N/A')}")
                for finding in insights.get("key_findings", []):
                    st.write(f"- {finding}")

                if insights.get("risks"):
                    st.markdown("**Risks**")
                    for risk in insights.get("risks", []):
                        st.write(f"- {risk}")

                if insights.get("recommended_actions"):
                    st.markdown("**Recommended Actions**")
                    for action in insights.get("recommended_actions", []):
                        st.write(f"- {action}")
                st.caption(f"Generation mode: {insights.get('generation_mode', 'unknown')}")
            else:
                profiler_insights = result.get("profile", {}).get("insights", [])
                if profiler_insights:
                    for insight in profiler_insights:
                        st.write(f"- {insight}")
                else:
                    st.info("Insight generation is not implemented yet.")
        
        with tab5:
            st.subheader("Diagnostic Analytics & Hypothesis Testing")
            st.caption("Apply inferential statistics to test root causes, group differences, and categorical associations.")
            analysis_df = st.session_state.get("analysis_dataframe")

            if analysis_df is not None:
                numeric_cols = list(analysis_df.select_dtypes(include=["number"]).columns)
                categorical_cols = list(analysis_df.select_dtypes(include=["object", "category"]).columns)

                test_type = st.selectbox(
                    "Choose Statistical Test",
                    [
                        "Two-Sample Independent T-Test (Compare 2 Groups)",
                        "One-Way ANOVA (Compare 3+ Groups)",
                        "Chi-Square Test of Independence (Categorical Association)",
                    ],
                )

                if test_type == "Two-Sample Independent T-Test (Compare 2 Groups)":
                    if numeric_cols and categorical_cols:
                        tc1, tc2 = st.columns(2)
                        with tc1:
                            metric_col = st.selectbox("Continuous Metric (Numeric)", numeric_cols, key="ttest_metric")
                        with tc2:
                            group_col = st.selectbox("Grouping Variable (Categorical)", categorical_cols, key="ttest_group")
                        
                        unique_groups = list(analysis_df[group_col].dropna().unique())
                        if len(unique_groups) >= 2:
                            gc1, gc2 = st.columns(2)
                            with gc1:
                                grp1 = st.selectbox("Group A", unique_groups, index=0, key="ttest_g1")
                            with gc2:
                                grp2 = st.selectbox("Group B", unique_groups, index=1, key="ttest_g2")
                            
                            if grp1 != grp2:
                                s1 = pd.to_numeric(analysis_df[analysis_df[group_col] == grp1][metric_col], errors="coerce").dropna()
                                s2 = pd.to_numeric(analysis_df[analysis_df[group_col] == grp2][metric_col], errors="coerce").dropna()

                                if len(s1) > 2 and len(s2) > 2:
                                    t_stat, p_val = stats.ttest_ind(s1, s2, equal_var=False)
                                    res_cols = st.columns(4)
                                    res_cols[0].metric(f"Mean ({grp1})", f"{s1.mean():,.2f}")
                                    res_cols[1].metric(f"Mean ({grp2})", f"{s2.mean():,.2f}")
                                    res_cols[2].metric("t-Statistic", f"{t_stat:.3f}")
                                    res_cols[3].metric("p-Value", f"{p_val:.4e}")

                                    st.markdown(f"**$H_0$ (Null Hypothesis):** Mean `{metric_col}` for `{grp1}` = Mean `{metric_col}` for `{grp2}`.")
                                    st.markdown(f"**$H_1$ (Alternative):** There is a significant difference between `{grp1}` and `{grp2}`.")

                                    if p_val < 0.05:
                                        st.success(f"🟢 **Statistically Significant (p = {p_val:.4f} < 0.05)**: Reject $H_0$. There is strong evidence that `{grp1}` and `{grp2}` have significantly different mean `{metric_col}`.")
                                    else:
                                        st.info(f"⚪ **Not Statistically Significant (p = {p_val:.4f} ≥ 0.05)**: Fail to reject $H_0$. No significant difference in `{metric_col}` detected between `{grp1}` and `{grp2}`.")
                                else:
                                    st.warning("Insufficient samples in one or both selected groups.")
                            else:
                                st.warning("Please choose two different groups.")
                        else:
                            st.warning(f"`{group_col}` must have at least 2 distinct categories.")
                    else:
                        st.info("Requires at least 1 numeric column and 1 categorical column.")

                elif test_type == "One-Way ANOVA (Compare 3+ Groups)":
                    if numeric_cols and categorical_cols:
                        ac1, ac2 = st.columns(2)
                        with ac1:
                            metric_col = st.selectbox("Continuous Metric (Numeric)", numeric_cols, key="anova_metric")
                        with ac2:
                            group_col = st.selectbox("Grouping Variable (3+ Groups)", categorical_cols, key="anova_group")
                        
                        groups_data = [
                            pd.to_numeric(group[metric_col], errors="coerce").dropna()
                            for _, group in analysis_df.groupby(group_col)
                        ]
                        groups_data = [g for g in groups_data if len(g) > 2]

                        if len(groups_data) >= 3:
                            f_stat, p_val = stats.f_oneway(*groups_data)
                            res_cols = st.columns(3)
                            res_cols[0].metric("Number of Groups", len(groups_data))
                            res_cols[1].metric("F-Statistic", f"{f_stat:.3f}")
                            res_cols[2].metric("p-Value", f"{p_val:.4e}")

                            st.markdown(f"**$H_0$ (Null):** All group means of `{metric_col}` are equal across `{group_col}`.")
                            st.markdown(f"**$H_1$ (Alternative):** At least one group mean is significantly different.")

                            if p_val < 0.05:
                                st.success(f"🟢 **Statistically Significant (p = {p_val:.4f} < 0.05)**: Reject $H_0$. At least one category in `{group_col}` exhibits a significantly different `{metric_col}`.")
                            else:
                                st.info(f"⚪ **Not Significant (p = {p_val:.4f} ≥ 0.05)**: Fail to reject $H_0$. Group means do not differ significantly.")
                        else:
                            st.warning(f"`{group_col}` needs at least 3 distinct groups with sufficient rows for ANOVA.")
                    else:
                        st.info("Requires at least 1 numeric column and 1 categorical column.")

                elif test_type == "Chi-Square Test of Independence (Categorical Association)":
                    if len(categorical_cols) >= 2:
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            cat1 = st.selectbox("First Categorical Feature", categorical_cols, index=0, key="chi_cat1")
                        with cc2:
                            cat2 = st.selectbox("Second Categorical Feature", categorical_cols, index=1, key="chi_cat2")
                        
                        if cat1 != cat2:
                            contingency = pd.crosstab(analysis_df[cat1].fillna("Missing"), analysis_df[cat2].fillna("Missing"))
                            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)

                            c_cols = st.columns(3)
                            c_cols[0].metric("Chi-Square (χ²)", f"{chi2:.3f}")
                            c_cols[1].metric("Degrees of Freedom", dof)
                            c_cols[2].metric("p-Value", f"{p_val:.4e}")

                            st.markdown("**Contingency Table:**")
                            st.dataframe(contingency, use_container_width=True)

                            st.markdown(f"**$H_0$:** `{cat1}` and `{cat2}` are independent.")
                            st.markdown(f"**$H_1$:** `{cat1}` and `{cat2}` are significantly associated.")

                            if p_val < 0.05:
                                st.success(f"🟢 **Significant Association (p = {p_val:.4f} < 0.05)**: Reject $H_0$. There is a statistically significant relationship between `{cat1}` and `{cat2}`.")
                            else:
                                st.info(f"⚪ **Independent (p = {p_val:.4f} ≥ 0.05)**: Fail to reject $H_0$. No significant association detected between `{cat1}` and `{cat2}`.")
                        else:
                            st.warning("Please choose two different categorical variables.")
                    else:
                        st.info("Requires at least 2 categorical columns for Chi-Square test.")
            else:
                st.info("Upload a dataset to run diagnostic hypothesis tests.")

        with tab6:
            st.subheader("Predictive Analytics & Machine Learning")
            st.caption("Define target labels and predictor features to train and evaluate supervised learning models.")
            analysis_df = st.session_state.get("analysis_dataframe")

            if analysis_df is not None:
                all_cols = list(analysis_df.columns)
                target_col = st.selectbox("Select Target Label (y)", all_cols, key="ml_target")
                candidate_features = [c for c in all_cols if c != target_col]
                selected_features = st.multiselect(
                    "Select Feature Columns (X)",
                    candidate_features,
                    default=candidate_features[:min(4, len(candidate_features))],
                    key="ml_features",
                )

                is_numeric_target = pd.api.types.is_numeric_dtype(analysis_df[target_col])
                model_task = "Regression (Continuous Target)" if is_numeric_target else "Classification (Discrete Classes)"
                st.caption(f"Task Auto-Detected: **{model_task}**")

                if st.button("🚀 Train & Evaluate Model", type="primary", use_container_width=True):
                    if not selected_features:
                        st.error("Please select at least one feature column.")
                    else:
                        # Prepare data
                        df_clean = analysis_df[[target_col] + selected_features].copy()
                        # Impute or drop
                        df_clean = df_clean.dropna()

                        if len(df_clean) < 10:
                            st.error("Insufficient complete rows after dropping missing values.")
                        else:
                            # Encode categorical features
                            X_raw = df_clean[selected_features]
                            X = pd.get_dummies(X_raw, drop_first=True)
                            y = df_clean[target_col]

                            if is_numeric_target:
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                model = LinearRegression()
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)

                                r2 = r2_score(y_test, y_pred)
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = mse ** 0.5

                                mc1, mc2, mc3 = st.columns(3)
                                mc1.metric("R² Score (Variance Explained)", f"{r2:.3f}")
                                mc2.metric("RMSE (Root Mean Sq Error)", f"{rmse:,.2f}")
                                mc3.metric("MSE", f"{mse:,.2f}")

                                # Feature coefficients
                                coef_df = pd.DataFrame(
                                    {"Feature": X.columns, "Coefficient": model.coef_}
                                ).sort_values("Coefficient", ascending=False)
                                fig_coef = px.bar(
                                    coef_df,
                                    x="Feature",
                                    y="Coefficient",
                                    title="Feature Importance (Regression Coefficients)",
                                    color="Coefficient",
                                    color_continuous_scale="Tealgrn",
                                )
                                fig_coef.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#edf3ef"))
                                st.plotly_chart(fig_coef, use_container_width=True, key="ml_coef_plot")

                                # Predicted vs Actual Plot
                                pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                                fig_pred = px.scatter(
                                    pred_df,
                                    x="Actual",
                                    y="Predicted",
                                    title="Actual vs Predicted Values",
                                    trendline="ols",
                                )
                                fig_pred.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#edf3ef"))
                                st.plotly_chart(fig_pred, use_container_width=True, key="ml_pred_plot")
                            else:
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                model = LogisticRegression(max_iter=1000)
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                acc = accuracy_score(y_test, y_pred)

                                ac1, ac2 = st.columns(2)
                                ac1.metric("Model Accuracy", f"{acc * 100:.1f}%")
                                ac2.metric("Classes Evaluated", len(y.unique()))

                                # Confusion matrix
                                cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
                                fig_cm = px.imshow(
                                    cm,
                                    x=[str(c) for c in model.classes_],
                                    y=[str(c) for c in model.classes_],
                                    text_auto=True,
                                    color_continuous_scale="Blues",
                                    title="Classification Confusion Matrix",
                                    labels=dict(x="Predicted Class", y="Actual Class"),
                                )
                                fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#edf3ef"))
                                st.plotly_chart(fig_cm, use_container_width=True, key="ml_cm_plot")
            else:
                st.info("Upload a dataset to train and evaluate predictive models.")

        with tab7:
            st.subheader("Full Report")
            report = result.get("report", {})
            if report.get("content"):
                st.markdown(report["content"])
                st.caption(f"Saved at: {report.get('report_path', 'N/A')}")
                st.caption(f"Generation mode: {report.get('generation_mode', 'unknown')}")
            else:
                st.json(
                    {
                        "status": result.get("status"),
                        "messages": result.get("messages", []),
                        "error": result.get("error"),
                    }
                )



if __name__ == "__main__":
    main()
