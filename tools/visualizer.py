"""Visualization generation tools."""

from typing import Any, Dict, List
import pandas as pd
from logger import setup_logger

logger = setup_logger(__name__)


class VisualizerTool:
    """Create charts and visualizations."""

    @staticmethod
    def recommend_visualizations(
        profile: Dict[str, Any],
        insights: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Recommend charts based on profile structure and generated insights."""
        recommendations: List[Dict[str, Any]] = []

        numeric_columns = list(profile.get("numeric_statistics", {}).keys())
        categorical_columns = list(profile.get("categorical_statistics", {}).keys())
        missing = profile.get("missing_percentage", {})

        # 1. Correlation Heatmap
        if len(numeric_columns) >= 2:
            recommendations.append(
                {
                    "chart_type": "correlation_heatmap",
                    "columns": numeric_columns,
                    "reason": "Multivariate analysis: inspect all pairwise linear associations simultaneously.",
                }
            )

        # 2. Box Plots / Outliers
        if numeric_columns:
            recommendations.append(
                {
                    "chart_type": "box_plot",
                    "columns": numeric_columns[:3],
                    "reason": "Outlier detection & dispersion: visualize medians, IQR fences, and anomaly points.",
                }
            )

        # 3. Distribution Histograms
        if numeric_columns:
            recommendations.append(
                {
                    "chart_type": "histogram",
                    "columns": numeric_columns[:2],
                    "reason": "Univariate analysis: show the distribution shape and skew of main numeric variables.",
                }
            )

        # 4. Scatter Plot
        if len(numeric_columns) >= 2:
            recommendations.append(
                {
                    "chart_type": "scatter",
                    "columns": numeric_columns[:2],
                    "reason": "Bivariate analysis: inspect relationship and pattern between two numeric variables.",
                }
            )

        # 5. Categorical Bar / Donut
        if categorical_columns:
            recommendations.append(
                {
                    "chart_type": "bar",
                    "columns": [categorical_columns[0]],
                    "reason": "Qualitative analysis: compare category frequencies for dominant categorical features.",
                }
            )
            recommendations.append(
                {
                    "chart_type": "pie_chart",
                    "columns": [categorical_columns[0]],
                    "reason": "Part-to-whole analysis: display proportion breakdown across categories.",
                }
            )

        # 6. Missingness Bar
        missing_columns = [column for column, pct in missing.items() if pct > 0]
        if missing_columns:
            recommendations.append(
                {
                    "chart_type": "missingness_bar",
                    "columns": missing_columns[:5],
                    "reason": "Data cleaning: highlight columns with missing data that require imputation.",
                }
            )

        # 7. Summary Card
        if insights.get("key_findings"):
            recommendations.append(
                {
                    "chart_type": "summary_card",
                    "columns": [],
                    "reason": "Surface executive findings in a compact dashboard-style summary.",
                }
            )

        return recommendations
