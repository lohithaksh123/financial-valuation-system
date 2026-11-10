"""Plotly chart builders for the dashboard, all drawn from one small visual system."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA = PALETTE[0], PALETTE[1], PALETTE[2]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e1", "#fcfcfb"
GOOD_ZONE, WATCH_ZONE, RISK_ZONE = "#cdeedd", "#fbe7bf", "#f8d0ce"
FONT = dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=13, color=MUTED)


def _layout(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        height=height,
        margin=dict(l=8, r=16, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=FONT,
        barcornerradius=4,
        bargap=0.55,
        hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="")
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=GRID, linecolor="rgba(0,0,0,0)", ticks="")
    return fig


def _price_line(fig: go.Figure, price: float, symbol: str) -> None:
    fig.add_vline(x=price, line=dict(color=INK, width=2, dash="dash"))
    fig.add_annotation(
        x=price, y=1, yref="paper", yanchor="bottom", showarrow=False,
        text=f"Price {symbol}{price:,.0f}", font=dict(color=INK, size=12),
    )


def explorer(table: pd.DataFrame, colors: Dict[str, str], kind: str, value_format: str, axis_format: str, title: str) -> go.Figure:
    """Any set of line items over time. Each item keeps its color as the selection changes."""
    years = [str(index.year) for index in table.index]
    fig = go.Figure()
    for column in table.columns:
        color = colors[column]
        hover = f"{column}: %{{y:{value_format}}}<extra></extra>"
        if kind == "Bars":
            fig.add_trace(go.Bar(x=years, y=table[column], name=column, marker_color=color, hovertemplate=hover))
        else:
            fig.add_trace(go.Scatter(
                x=years, y=table[column], name=column, mode="lines+markers", connectgaps=True,
                line=dict(color=color, width=2), marker=dict(size=8, color=color, line=dict(color=SURFACE, width=2)),
                hovertemplate=hover,
            ))
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickformat=axis_format)
    fig.update_layout(hovermode="x unified", barmode="group", showlegend=True)
    _layout(fig, title, height=380)
    fig.update_layout(bargap=0.35, bargroupgap=0.08)
    return fig


def single_line(series: pd.Series, title: str, value_format: str, axis_format: str) -> go.Figure:
    years = [str(index.year) for index in series.index]
    fig = go.Figure(go.Scatter(
        x=years, y=series, mode="lines+markers", connectgaps=True, line=dict(color=BLUE, width=2),
        marker=dict(size=8, color=BLUE, line=dict(color=SURFACE, width=2)),
        hovertemplate=f"%{{x}}: %{{y:{value_format}}}<extra></extra>",
    ))
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickformat=axis_format)
    return _layout(fig, title, height=280)


def football_field(ranges: List, price: Optional[float], symbol: str) -> go.Figure:
    rows = list(reversed(ranges))
    labels = [row.label for row in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[row.high - row.low for row in rows], base=[row.low for row in rows],
        orientation="h", marker_color=BLUE, opacity=0.35, width=0.3, showlegend=False,
        customdata=[[row.low, row.high] for row in rows],
        hovertemplate=f"%{{y}}<br>{symbol}%{{customdata[0]:,.0f}} to {symbol}%{{customdata[1]:,.0f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        y=labels, x=[row.mid for row in rows], mode="markers", showlegend=False,
        marker=dict(size=11, color=BLUE, line=dict(color=SURFACE, width=2)),
        hovertemplate=f"%{{y}} central value: {symbol}%{{x:,.0f}}<extra></extra>",
    ))
    if price is not None:
        _price_line(fig, price, symbol)
    _layout(fig, "What each method says one share is worth", height=110 + 58 * len(rows))
    fig.update_layout(margin=dict(t=64))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, tickprefix=symbol, tickformat=",.0f")
    fig.update_yaxes(showgrid=False)
    return fig


def value_bridge(steps: List, symbol: str, unit: str) -> go.Figure:
    """Waterfall from forecast cash flows to the value left for shareholders."""
    labels = [label for label, _, _ in steps]
    values = [value for _, value, _ in steps]
    measures = [measure for _, _, measure in steps]
    text, running, peak, trough = [], 0.0, 0.0, 0.0
    for value, measure in zip(values, measures):
        if measure == "relative":
            running += value
            text.append(f"{value:+,.0f}")
        else:
            text.append(f"{running:,.0f}")
        peak, trough = max(peak, running), min(trough, running)
    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures, text=text, textposition="outside",
        increasing=dict(marker=dict(color=BLUE)), decreasing=dict(marker=dict(color=ORANGE)),
        totals=dict(marker=dict(color="#184f95")), connector=dict(line=dict(color=GRID, width=1)),
        hovertemplate=f"%{{x}}: {symbol}%{{y:,.1f}} {unit}<extra></extra>",
    ))
    _layout(fig, f"Where the company's value comes from ({symbol} {unit})", height=360)
    fig.update_layout(bargap=0.45, margin=dict(t=60))
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickformat=",.0f", range=[trough * 1.15, peak * 1.15])  # room for labels
    return fig


def distribution(values: np.ndarray, price: Optional[float], percentiles: Dict[str, float], symbol: str) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=50, marker_color=BLUE, marker_line=dict(color=SURFACE, width=1),
        hovertemplate=f"{symbol}%{{x}}: %{{y:,}} scenarios<extra></extra>",
    ))
    if price is not None:
        _price_line(fig, price, symbol)
    for label, value in percentiles.items():
        fig.add_vline(x=value, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_annotation(
            x=value, y=1, yref="paper", yanchor="bottom", showarrow=False,
            text=f"{label} {symbol}{value:,.0f}", font=dict(size=11),
        )
    _layout(fig, "Range of possible values per share", height=360)
    fig.update_layout(bargap=0.04, margin=dict(t=72))
    fig.update_xaxes(tickprefix=symbol, tickformat=",.0f")
    fig.update_yaxes(title_text="Scenarios")
    return fig


def sensitivity_heatmap(
    table: pd.DataFrame, reference: float, reference_label: str, symbol: str, row_label: str, column_label: str
) -> go.Figure:
    gap = table.to_numpy() / reference - 1
    text = [[("n/a" if np.isnan(v) else f"{symbol}{v:,.0f}") for v in row] for row in table.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=gap, x=[f"{c:.1%}" for c in table.columns], y=[f"{i:.1%}" for i in table.index],
        text=text, texttemplate="%{text}", zmid=0, zmin=-0.6, zmax=0.6, xgap=2, ygap=2,
        colorscale=[[0, ORANGE], [0.5, "#efeeea"], [1, BLUE]],
        colorbar=dict(title=f"vs {reference_label}", tickformat="+.0%", thickness=10, outlinewidth=0),
        hovertemplate=f"{row_label} %{{y}}, {column_label} %{{x}}<br>%{{text}} (%{{z:+.0%}} vs {reference_label})<extra></extra>",
    ))
    _layout(fig, f"Value per share: {row_label} (rows) by {column_label} (columns)", height=340)
    fig.update_xaxes(type="category", showgrid=False)
    fig.update_yaxes(type="category", autorange="reversed", showgrid=False, zeroline=False)
    return fig


def altman_scale(z: float) -> go.Figure:
    """Where the score sits across the distress, grey and safe zones."""
    low = min(z - 0.5, 0.0)
    high = max(z + 1.0, 5.0)
    fig = go.Figure()
    for start, end, color, label in [(low, 1.1, RISK_ZONE, "Distress"), (1.1, 2.6, WATCH_ZONE, "Grey"), (2.6, high, GOOD_ZONE, "Safe")]:
        fig.add_vrect(x0=start, x1=end, fillcolor=color, line_width=0, layer="below")
        fig.add_annotation(x=(start + end) / 2, y=0.82, yref="paper", text=label, showarrow=False, font=dict(color=INK, size=12))
    fig.add_trace(go.Scatter(
        x=[z], y=[0], mode="markers", marker=dict(size=16, color=INK, line=dict(color=SURFACE, width=2)),
        hovertemplate="Z'' score: %{x:.2f}<extra></extra>", showlegend=False,
    ))
    _layout(fig, " ", height=120)
    fig.update_layout(margin=dict(l=8, r=8, t=4, b=8))
    fig.update_xaxes(range=[low, high], showgrid=False)
    fig.update_yaxes(visible=False, range=[-1, 1.6])
    return fig


def ranking(table: pd.DataFrame) -> go.Figure:
    """Model value against price for each company, best first."""
    ordered = table.sort_values("Upside")
    colors = [BLUE if value >= 0 else ORANGE for value in ordered["Upside"]]
    fig = go.Figure(go.Bar(
        y=ordered["Ticker"], x=ordered["Upside"], orientation="h", marker_color=colors,
        text=[f"{value:+.0%}" for value in ordered["Upside"]], textposition="outside",
        customdata=ordered[["Company", "Verdict"]],
        hovertemplate="%{customdata[0]}<br>%{x:+.1%}, %{customdata[1]}<extra></extra>",
    ))
    _layout(fig, "Model value compared with share price", height=90 + 40 * len(ordered))
    fig.update_layout(bargap=0.45)
    fig.update_xaxes(tickformat="+.0%", showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=MUTED)
    fig.update_yaxes(showgrid=False, type="category")
    return fig
