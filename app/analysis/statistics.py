"""
Analytics and statistics generation from database data.

Generates chart data and summary statistics for the analytics
dashboard using Pandas and Plotly. All data comes from the
database — nothing is hard-coded.
"""

import json

import plotly
import plotly.express as px
import plotly.graph_objects as go

from app.utils.logger import get_logger

logger = get_logger(__name__)


def generate_damage_type_chart(distribution: list[dict]) -> str | None:
    """
    Generate a bar chart for damage type distribution.

    Args:
        distribution: List of {"type": str, "name": str, "count": int}.

    Returns:
        Plotly chart JSON string, or None if no data.
    """
    if not distribution:
        return None

    labels = [f"{d['type']}\n{d['name']}" for d in distribution]
    values = [d["count"] for d in distribution]
    colors = ["#FF6B35", "#FFD700", "#FF4444", "#CC0000"]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors[: len(labels)],
            text=values,
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Damage Type Distribution",
        xaxis_title="Damage Type",
        yaxis_title="Count",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=50, b=80),
    )
    return json.loads(plotly.io.to_json(fig))


def generate_severity_chart(distribution: list[dict]) -> str | None:
    """
    Generate a pie chart for severity distribution.

    Args:
        distribution: List of {"severity": str, "count": int}.

    Returns:
        Plotly chart JSON string, or None if no data.
    """
    if not distribution:
        return None

    labels = [d["severity"] for d in distribution]
    values = [d["count"] for d in distribution]
    colors = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}
    chart_colors = [colors.get(l, "#999999") for l in labels]

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker_colors=chart_colors,
            hole=0.4,
            textinfo="label+percent+value",
        )
    ])
    fig.update_layout(
        title="Severity Distribution",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return json.loads(plotly.io.to_json(fig))


def generate_confidence_chart(confidence_values: list[float]) -> str | None:
    """
    Generate a histogram for confidence score distribution.

    Args:
        confidence_values: List of confidence values (0.0 - 1.0).

    Returns:
        Plotly chart JSON string, or None if no data.
    """
    if not confidence_values:
        return None

    fig = go.Figure(data=[
        go.Histogram(
            x=confidence_values,
            nbinsx=20,
            marker_color="#2196F3",
            opacity=0.8,
        )
    ])
    fig.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="Confidence",
        yaxis_title="Count",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(range=[0, 1]),
    )
    return json.loads(plotly.io.to_json(fig))


def generate_detections_per_session_chart(
    sessions: list[dict],
) -> str | None:
    """
    Generate a bar chart showing detections per analysis session.

    Args:
        sessions: List of {"session_id": int, "filename": str,
                  "detections": int, "processing_time": float}.

    Returns:
        Plotly chart JSON string, or None if no data.
    """
    if not sessions:
        return None

    labels = [
        f"#{s['session_id']}: {s['filename'][:20]}" for s in sessions
    ]
    values = [s["detections"] for s in sessions]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color="#9C27B0",
            text=values,
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Detections Per Analysis",
        xaxis_title="Session",
        yaxis_title="Detections",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=50, b=100),
        xaxis_tickangle=-45,
    )
    return json.loads(plotly.io.to_json(fig))


def generate_performance_chart(sessions: list[dict]) -> str | None:
    """
    Generate a chart showing processing performance over sessions.

    Args:
        sessions: List of session dicts with processing_time.

    Returns:
        Plotly chart JSON string, or None if no data.
    """
    if not sessions:
        return None

    labels = [f"#{s['session_id']}" for s in sessions]
    times = [s["processing_time"] for s in sessions]

    fig = go.Figure(data=[
        go.Scatter(
            x=labels,
            y=times,
            mode="lines+markers",
            marker_color="#00BCD4",
            line=dict(color="#00BCD4"),
        )
    ])
    fig.update_layout(
        title="Processing Time per Session",
        xaxis_title="Session",
        yaxis_title="Time (seconds)",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return json.loads(plotly.io.to_json(fig))
