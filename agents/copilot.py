"""Agentic analytics copilot powered by GPT-4o with tool-calling via langchain-openai."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from agents.llm_utils import build_chat_model
from logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level run-record store so tools can access data without arg passing
# ---------------------------------------------------------------------------
_current_run: Dict[str, Any] = {}


def _profile() -> Dict[str, Any]:
    return _current_run.get("result", {}).get("profile", {})


def _anomalies() -> Dict[str, Any]:
    return _current_run.get("result", {}).get("anomalies", {})


def _insights() -> Dict[str, Any]:
    return _current_run.get("result", {}).get("insights", {})


def _summary() -> Dict[str, Any]:
    return _current_run.get("summary", {})


def _file_name() -> str:
    return _current_run.get("file_name", "dataset")


# ---------------------------------------------------------------------------
# Tool definitions — GPT-4o decides which one(s) to call per user message
# ---------------------------------------------------------------------------

@tool
def get_dataset_overview() -> str:
    """Return a full overview of the dataset: shape, data types, missing values,
    duplicates, and a top-level executive finding. Use this when the user asks
    what the dataset is about, how many rows/columns, or wants a summary."""
    profile = _profile()
    anomalies = _anomalies()
    insights = _insights()
    summary = _summary()

    shape = profile.get("shape", {})
    rows = shape.get("rows", summary.get("rows", 0))
    cols = shape.get("columns", summary.get("columns", 0))
    numeric_stats = profile.get("numeric_statistics", {})
    categorical_stats = profile.get("categorical_statistics", {})
    missing_vals = profile.get("missing_values", {})
    duplicates = profile.get("duplicate_rows", 0)
    flagged = anomalies.get("total_anomalous_rows", summary.get("flagged_rows", 0))
    exec_summary = (
        insights.get("executive_summary")
        or summary.get("executive_summary")
        or "Dataset profiled successfully."
    )

    return json.dumps({
        "file": _file_name(),
        "rows": rows,
        "columns": cols,
        "numeric_features": len(numeric_stats),
        "categorical_features": len(categorical_stats),
        "total_missing_cells": sum(missing_vals.values()),
        "duplicate_rows": duplicates,
        "flagged_outlier_rows": flagged,
        "executive_summary": exec_summary,
        "key_findings": insights.get("key_findings", []),
        "risks": insights.get("risks", []),
        "recommended_actions": insights.get("recommended_actions", []),
    }, indent=2)


@tool
def get_column_statistics(column_name: str) -> str:
    """Return detailed statistics for a specific column by name.
    Use this when the user asks about a particular column, feature, or variable.
    Works for both numeric (mean, std, quartiles) and categorical columns (top values, unique count)."""
    profile = _profile()
    numeric_stats = profile.get("numeric_statistics", {})
    categorical_stats = profile.get("categorical_statistics", {})
    missing_vals = profile.get("missing_values", {})
    missing_pct = profile.get("missing_percentage", {})
    dtypes = profile.get("dtypes", {})

    col_match = None
    for col in profile.get("columns", []):
        if col.lower() == column_name.lower():
            col_match = col
            break

    if col_match is None:
        available = profile.get("columns", [])[:20]
        return json.dumps({"error": f"Column '{column_name}' not found.", "available_columns": available})

    result: Dict[str, Any] = {
        "column": col_match,
        "dtype": dtypes.get(col_match, "unknown"),
        "missing_values": missing_vals.get(col_match, 0),
        "missing_percentage": round(missing_pct.get(col_match, 0.0), 2),
    }

    if col_match in numeric_stats:
        st = numeric_stats[col_match]
        result.update({
            "type": "numeric",
            "mean": round(st.get("mean") or 0, 4),
            "median": round(st.get("median") or 0, 4),
            "std_dev": round(st.get("std") or 0, 4),
            "min": st.get("min"),
            "max": st.get("max"),
            "q25": st.get("q25"),
            "q75": st.get("q75"),
            "iqr": round((st.get("q75") or 0) - (st.get("q25") or 0), 4),
        })
    elif col_match in categorical_stats:
        cs = categorical_stats[col_match]
        result.update({
            "type": "categorical",
            "unique_categories": cs.get("unique_count", 0),
            "top_values": cs.get("top_values", {}),
        })

    return json.dumps(result, indent=2)


@tool
def get_outlier_analysis() -> str:
    """Return a detailed outlier and anomaly report for the dataset.
    Use this when the user asks about outliers, anomalies, extreme values, or data quality flags."""
    profile = _profile()
    anomalies = _anomalies()

    shape = profile.get("shape", {})
    rows = shape.get("rows", 1)
    total_flagged = anomalies.get("total_anomalous_rows", 0)
    methods = anomalies.get("methods", {})
    iqr_data = methods.get("iqr", {})
    z_data = methods.get("zscore", {})
    top_cols = anomalies.get("top_columns", [])

    return json.dumps({
        "total_rows": rows,
        "flagged_rows": total_flagged,
        "flagged_percentage": round(total_flagged / max(rows, 1) * 100, 2),
        "iqr_method_flags": iqr_data.get("total_anomalies", 0),
        "zscore_method_flags": z_data.get("total_anomalies", 0),
        "top_outlier_columns": top_cols[:10],
        "treatment_recommendations": [
            "Trimming/Removal: Drop only if confirmed as data entry errors.",
            "Winsorization: Cap extreme values at Q1-1.5*IQR and Q3+1.5*IQR fences.",
            "Log Transform: For right-skewed positive features to compress variance.",
            "Robust Scalers: Use RobustScaler before ML training to reduce outlier impact.",
        ],
    }, indent=2)


@tool
def get_correlation_analysis(min_correlation: float = 0.3) -> str:
    """Return all feature pairs with notable correlations above the given threshold.
    Use this when the user asks about correlations, relationships, associations, multicollinearity,
    or which features are related to each other. Default threshold is 0.3."""
    profile = _profile()
    correlations = profile.get("correlations", {})

    pairs: List[Dict[str, Any]] = []
    seen = set()
    for c1, row_vals in correlations.items():
        if not isinstance(row_vals, dict):
            continue
        for c2, val in row_vals.items():
            key = tuple(sorted([c1, c2]))
            if c1 == c2 or key in seen:
                continue
            seen.add(key)
            if abs(val) >= min_correlation:
                strength = "Strong" if abs(val) >= 0.7 else "Moderate"
                direction = "Positive" if val > 0 else "Negative"
                pairs.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "pearson_r": round(val, 4),
                    "strength": strength,
                    "direction": direction,
                })

    pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

    return json.dumps({
        "threshold_used": min_correlation,
        "notable_pairs_found": len(pairs),
        "correlations": pairs[:20],
        "interpretation_notes": (
            "High positive r means features increase together. "
            "High negative r means inverse relationship. "
            "|r| > 0.7 may cause multicollinearity in linear models."
        ),
    }, indent=2)


@tool
def get_prediction_advice(target_column: str) -> str:
    """Provide a machine learning prediction plan for a specified target column.
    Use this when the user asks about predictions, forecasting, ML models, regression,
    classification, or wants to predict a specific column from the others."""
    profile = _profile()
    numeric_stats = profile.get("numeric_statistics", {})
    categorical_stats = profile.get("categorical_statistics", {})
    all_cols: List[str] = profile.get("columns", [])

    target_match = None
    for col in all_cols:
        if col.lower() == target_column.lower():
            target_match = col
            break

    if target_match is None:
        return json.dumps({
            "error": f"Column '{target_column}' not found. Please specify a valid column name.",
            "available_columns": all_cols[:20],
        })

    feature_cols = [c for c in all_cols if c != target_match]
    is_numeric_target = target_match in numeric_stats

    if is_numeric_target:
        task_type = "Regression"
        st = numeric_stats[target_match]
        target_info = {
            "mean": round(st.get("mean") or 0, 4),
            "std": round(st.get("std") or 0, 4),
            "min": st.get("min"),
            "max": st.get("max"),
        }
        recommended_models = [
            "LinearRegression — fast baseline, interpretable coefficients",
            "Ridge / Lasso — L2/L1 regularisation to reduce overfitting",
            "RandomForestRegressor — handles non-linearity and interactions",
            "GradientBoostingRegressor (XGBoost / LightGBM) — usually best accuracy",
        ]
        eval_metrics = ["R² Score", "RMSE (Root Mean Squared Error)", "MAE (Mean Absolute Error)"]
    else:
        cs = categorical_stats.get(target_match, {})
        unique_count = cs.get("unique_count", 2)
        task_type = "Binary Classification" if unique_count <= 2 else "Multi-class Classification"
        target_info = {
            "unique_classes": unique_count,
            "top_values": cs.get("top_values", {}),
        }
        recommended_models = [
            "LogisticRegression — fast, interpretable, great baseline",
            "RandomForestClassifier — robust, handles missing values well",
            "GradientBoostingClassifier (XGBoost / LightGBM) — top performance",
            "SVM (SVC) — effective in high-dimensional spaces",
        ]
        eval_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    return json.dumps({
        "target_column": target_match,
        "task_type": task_type,
        "target_statistics": target_info,
        "suggested_features": feature_cols[:30],
        "total_feature_candidates": len(feature_cols),
        "recommended_models": recommended_models,
        "evaluation_metrics": eval_metrics,
        "preprocessing_steps": [
            "Handle missing values (impute or drop)",
            "Encode categorical features (OneHotEncoder or LabelEncoder)",
            "Scale numeric features (StandardScaler or RobustScaler)",
            "Split data: 80% train / 20% test with stratification if classification",
            "Use cross-validation (5-fold) for robust performance estimates",
        ],
    }, indent=2)


@tool
def get_visualization_recommendations() -> str:
    """Suggest the most appropriate chart types for the current dataset.
    Use this when the user asks what charts to use, how to visualise the data,
    or asks about plots, graphs, heatmaps, or visual exploration."""
    profile = _profile()
    numeric_stats = profile.get("numeric_statistics", {})
    categorical_stats = profile.get("categorical_statistics", {})
    correlations = profile.get("correlations", {})

    num_cols = list(numeric_stats.keys())
    cat_cols = list(categorical_stats.keys())

    recommendations: List[Dict[str, str]] = []

    if num_cols:
        recommendations.append({
            "chart": "Histogram / KDE Plot",
            "purpose": "Understand distribution shape, skewness, and modality of numeric features",
            "columns": ", ".join(num_cols[:5]),
        })
        recommendations.append({
            "chart": "Box Plot / Violin Plot",
            "purpose": "Compare distributions across groups and spot outliers visually",
            "columns": ", ".join(num_cols[:5]),
        })

    if cat_cols:
        recommendations.append({
            "chart": "Bar Chart / Count Plot",
            "purpose": "Show frequency of categorical values",
            "columns": ", ".join(cat_cols[:3]),
        })

    if len(num_cols) >= 2:
        recommendations.append({
            "chart": "Scatter Plot",
            "purpose": "Explore linear/non-linear relationships between two numeric features",
            "columns": f"{num_cols[0]} vs {num_cols[1]}",
        })

    if correlations and len(num_cols) > 1:
        recommendations.append({
            "chart": "Correlation Heatmap",
            "purpose": "Visualise all pairwise Pearson correlations in a colour-encoded matrix",
            "columns": "All numeric features",
        })

    if len(num_cols) >= 3:
        recommendations.append({
            "chart": "Pair Plot",
            "purpose": "Grid of scatter plots showing all pairwise combinations at once",
            "columns": f"First {min(6, len(num_cols))} numeric features",
        })

    return json.dumps({
        "dataset": _file_name(),
        "numeric_columns_available": len(num_cols),
        "categorical_columns_available": len(cat_cols),
        "recommended_visualizations": recommendations,
        "tip": (
            "The Visualizations & Heatmaps tab in the app already includes an "
            "Interactive Chart Studio where you can build these charts without code."
        ),
    }, indent=2)


@tool
def get_missing_values_report() -> str:
    """Return a complete missing values and data quality report with imputation strategies.
    Use this when the user asks about nulls, missing data, NaN values,
    incomplete data, or data cleaning / imputation strategies."""
    profile = _profile()

    missing_vals = profile.get("missing_values", {})
    missing_pct = profile.get("missing_percentage", {})
    duplicates = profile.get("duplicate_rows", 0)
    shape = profile.get("shape", {})
    rows = shape.get("rows", 1)

    affected_columns = [
        {
            "column": col,
            "missing_count": cnt,
            "missing_percentage": round(missing_pct.get(col, 0.0), 2),
            "severity": (
                "Critical (>50%)" if missing_pct.get(col, 0) > 50
                else "High (20-50%)" if missing_pct.get(col, 0) > 20
                else "Moderate (5-20%)" if missing_pct.get(col, 0) > 5
                else "Low (<5%)"
            ),
        }
        for col, cnt in missing_vals.items()
        if cnt > 0
    ]
    affected_columns.sort(key=lambda x: x["missing_percentage"], reverse=True)

    return json.dumps({
        "total_rows": rows,
        "total_columns_with_missing": len(affected_columns),
        "duplicate_rows": duplicates,
        "columns_detail": affected_columns,
        "imputation_strategies": {
            "numeric_normal_distribution": "Mean Imputation (sklearn SimpleImputer strategy='mean')",
            "numeric_skewed": "Median Imputation (strategy='median')",
            "categorical": "Mode Imputation (strategy='most_frequent') or add 'Missing' category",
            "correlated_multivariate": "KNN Imputer or IterativeImputer (MICE algorithm)",
            "time_series": "Forward-fill or backward-fill (pandas ffill / bfill)",
        },
    }, indent=2)


# ---------------------------------------------------------------------------
# All tools list
# ---------------------------------------------------------------------------
TOOLS = [
    get_dataset_overview,
    get_column_statistics,
    get_outlier_analysis,
    get_correlation_analysis,
    get_prediction_advice,
    get_visualization_recommendations,
    get_missing_values_report,
]

SYSTEM_PROMPT = (
    "You are an expert Senior Data Scientist and Analytics Copilot. "
    "You have access to a set of tools that retrieve live statistics from the user's dataset. "
    "ALWAYS call the most relevant tool before answering so your response is grounded in real data.\n\n"
    "Rules:\n"
    "1. Use tools to get real numbers — never guess or make up statistics.\n"
    "2. After calling a tool, interpret the JSON results in clear, conversational language.\n"
    "3. Format your final answer in clean GitHub markdown with bold key terms and bullet points.\n"
    "4. Be specific: mention actual column names, real numbers, and concrete next steps.\n"
    "5. For prediction/ML questions, call get_prediction_advice with the target column name.\n"
    "6. For chart/visualisation questions, call get_visualization_recommendations.\n"
    "7. Keep answers focused and practical — avoid filler text."
)


# ---------------------------------------------------------------------------
# Agentic Copilot — uses GPT-4o bind_tools (works with any langchain version)
# ---------------------------------------------------------------------------

class AnalyticsCopilotAgent:
    """Agentic analytics copilot using GPT-4o tool-calling to answer dataset questions."""

    def __init__(self):
        self.name = "AnalyticsCopilot"
        self._llm_with_tools = None

    def _get_llm(self):
        """Lazily build the GPT-4o model with tools bound."""
        if self._llm_with_tools is None:
            llm = build_chat_model(temperature=0.1)
            self._llm_with_tools = llm.bind_tools(TOOLS)
        return self._llm_with_tools

    def _run_tool(self, tool_call: dict) -> str:
        """Execute a single tool call and return its string result."""
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        tool_map = {t.name: t for t in TOOLS}
        if name not in tool_map:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return tool_map[name].invoke(args)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def answer(self, question: str, run_record: Dict[str, Any]) -> Dict[str, Any]:
        """Answer a user question using GPT-4o with tool-calling (agentic loop)."""
        global _current_run
        _current_run = run_record

        try:
            from langchain_core.messages import AIMessage, ToolMessage

            llm = self._get_llm()
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]

            # Agentic loop: allow up to 5 tool call rounds
            for _ in range(5):
                response: AIMessage = llm.invoke(messages)
                messages.append(response)

                tool_calls = getattr(response, "tool_calls", []) or []
                if not tool_calls:
                    # No more tools to call — final answer ready
                    break

                # Execute all tool calls in this round
                for tc in tool_calls:
                    tool_result = self._run_tool(tc)
                    messages.append(
                        ToolMessage(
                            content=tool_result,
                            tool_call_id=tc.get("id", tc.get("name", "")),
                        )
                    )

            final_content = response.content
            if not final_content:
                raise ValueError("GPT-4o returned an empty response.")

            return {"status": "success", "answer": final_content, "mode": "gpt4o_agent"}

        except Exception as exc:
            logger.warning(f"Agentic copilot failed, falling back to direct LLM: {exc}")
            return self._direct_llm_fallback(question, run_record)

    def _direct_llm_fallback(self, question: str, run_record: Dict[str, Any]) -> Dict[str, Any]:
        """Single-shot LLM call with compacted context injected in the prompt."""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        result = run_record.get("result", {})
        raw_profile = result.get("profile", {})
        compact_profile = {
            "shape": raw_profile.get("shape", {}),
            "columns": raw_profile.get("columns", [])[:35],
            "numeric_columns": list(raw_profile.get("numeric_statistics", {}).keys())[:25],
            "categorical_columns": list(raw_profile.get("categorical_statistics", {}).keys())[:25],
            "missing_values": {k: v for k, v in list(raw_profile.get("missing_values", {}).items())[:20] if v > 0},
            "duplicate_rows": raw_profile.get("duplicate_rows", 0),
        }
        compact_anomalies = {
            "total_anomalous_rows": result.get("anomalies", {}).get("total_anomalous_rows", 0),
            "columns_with_outliers": list(result.get("anomalies", {}).get("outliers_by_column", {}).keys())[:15],
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            (
                "human",
                (
                    "Dataset: {file_name}\n"
                    "Summary: {summary}\n"
                    "Profile: {profile}\n"
                    "Anomalies: {anomalies}\n"
                    "Insights: {insights}\n\n"
                    "Question: {question}\n\n"
                    "Provide a direct, data-grounded, markdown-formatted answer."
                ),
            ),
        ])
        try:
            llm = build_chat_model(temperature=0.1)
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "question": question,
                "file_name": run_record.get("file_name", "dataset"),
                "summary": run_record.get("summary", {}),
                "profile": compact_profile,
                "anomalies": compact_anomalies,
                "insights": result.get("insights", {}),
            })
            return {"status": "success", "answer": answer, "mode": "llm_direct"}
        except Exception as exc2:
            logger.error(f"Direct LLM fallback also failed: {exc2}")
            err_msg = str(exc2)
            if "Connection error" in err_msg or "11001" in err_msg or "getaddrinfo" in err_msg:
                user_msg = (
                    "⚠️ **Network Connection Error**: The copilot cannot connect to the AI model server (`api.groq.com`).\n\n"
                    "- Please verify that your computer is connected to the internet.\n"
                    "- If you are using a college, office, or restricted Wi-Fi network, switch to a mobile hotspot or home network that allows external AI endpoints.\n"
                    "- Check that your `GROQ_API_KEY` in `.env` is valid."
                )
            else:
                user_msg = f"⚠️ The copilot encountered an error communicating with the AI model: {err_msg}"
            return {
                "status": "error",
                "answer": user_msg,
                "mode": "error",
            }
