"""
Beautiful UI module for Komputasi Numerik Streamlit App.
Import and use in app.py for styled components and charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────
# Color Palette
# ──────────────────────────────────────────────
COLORS = {
    "primary": "#6C63FF",
    "primary_light": "#8B83FF",
    "secondary": "#00D2FF",
    "accent": "#FF6584",
    "success": "#00C9A7",
    "warning": "#FFB347",
    "bg_dark": "#0E1117",
    "bg_card": "#1A1D29",
    "bg_card_hover": "#22263A",
    "text": "#FAFAFA",
    "text_muted": "#8892B0",
    "border": "#2D3148",
    "gradient_start": "#6C63FF",
    "gradient_end": "#00D2FF",
}

METHOD_INFO = {
    "Bi Section": {
        "icon": "✂️",
        "desc": "Membagi interval menjadi dua bagian secara berulang untuk menemukan akar.",
        "color": "#6C63FF",
        "tag": "Bracketing",
    },
    "False Position": {
        "icon": "📐",
        "desc": "Menggunakan garis lurus antara dua titik untuk estimasi akar yang lebih baik.",
        "color": "#00D2FF",
        "tag": "Bracketing",
    },
    "Fixed Point": {
        "icon": "🔄",
        "desc": "Mengiterasi fungsi g(x) sampai konvergen ke titik tetap.",
        "color": "#FF6584",
        "tag": "Open Method",
    },
    "Newton Raphson": {
        "icon": "🚀",
        "desc": "Menggunakan turunan untuk menemukan akar dengan konvergensi cepat.",
        "color": "#00C9A7",
        "tag": "Open Method",
    },
    "Secant": {
        "icon": "📏",
        "desc": "Seperti Newton-Raphson tapi tanpa perlu menghitung turunan.",
        "color": "#FFB347",
        "tag": "Open Method",
    },
    "Modified Newton Raphson": {
        "icon": "⚡",
        "desc": "Versi modifikasi NR untuk menangani akar berganda.",
        "color": "#E040FB",
        "tag": "Open Method",
    },
    "Polynomial Factorization": {
        "icon": "🧩",
        "desc": "Memfaktorkan polinomial untuk mencari semua akar sekaligus.",
        "color": "#FF7043",
        "tag": "Polynomial",
    },
}


# ──────────────────────────────────────────────
# CSS Injection
# ──────────────────────────────────────────────
def inject_css():
    """Inject custom CSS for the entire app."""
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global — exclude LaTeX (MathJax/KaTeX) */
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #6C63FF 0%, #00D2FF 50%, #00C9A7 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1B2A4A;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 0 1px 4px rgba(255,255,255,0.3);
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #1B2A4A;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
        font-weight: 400;
    }

    /* Method card */
    .method-card {
        background: linear-gradient(145deg, #1A1D29, #22263A);
        border: 1px solid #2D3148;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .method-card:hover {
        border-color: #6C63FF;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.15);
    }
    .method-icon {
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
    }
    .method-name {
        font-size: 1rem;
        font-weight: 600;
        color: #FAFAFA;
        margin: 0.3rem 0;
    }
    .method-desc {
        font-size: 0.8rem;
        color: #8892B0;
        line-height: 1.4;
    }
    .method-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(145deg, #1A1D29, #22263A);
        border: 1px solid #2D3148;
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #1A1D29, #22263A);
        border: 1px solid #2D3148;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #00D2FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8892B0;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section header */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2D3148;
    }
    .section-header-text {
        font-size: 1.2rem;
        font-weight: 700;
        color: #FAFAFA;
    }

    /* Divider */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, #6C63FF, #00D2FF, transparent);
        border: none;
        margin: 1.5rem 0;
        border-radius: 1px;
    }

    /* Success box */
    .success-box {
        background: linear-gradient(135deg, rgba(0,201,167,0.15), rgba(0,201,167,0.05));
        border: 1px solid rgba(0,201,167,0.3);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .success-text {
        color: #00C9A7;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .success-value {
        color: #FAFAFA;
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, rgba(255,101,132,0.15), rgba(255,101,132,0.05));
        border: 1px solid rgba(255,101,132,0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #FF6584;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #151929 100%);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #8892B0 !important;
        font-weight: 500;
    }

    /* Better dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #00D2FF 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4) !important;
    }

    /* Expander style */
    .streamlit-expanderHeader {
        background: #1A1D29 !important;
        border-radius: 10px !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Layout Components
# ──────────────────────────────────────────────
def render_header():
    """Render the hero header."""
    st.markdown(
        """
    <div class="hero-header">
        <div class="hero-title">Komputasi Numerik</div>
        <div class="hero-subtitle">Informatika ITS &mdash; Kalkulator Metode Numerik Interaktif</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_method_info(method_name: str):
    """Render method info card in sidebar."""
    info = METHOD_INFO.get(method_name, {})
    if not info:
        return
    st.markdown(
        f"""
    <div class="method-card">
        <div class="method-icon">{info['icon']}</div>
        <div class="method-name">{method_name}</div>
        <div class="method-desc">{info['desc']}</div>
        <div class="method-tag" style="background: {info['color']}22; color: {info['color']};">
            {info['tag']}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_section_header(icon: str, text: str):
    """Render a styled section header."""
    st.markdown(
        f"""
    <div class="section-header">
        <span style="font-size:1.3rem;">{icon}</span>
        <span class="section-header-text">{text}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_gradient_divider():
    """Render a gradient divider."""
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


def render_success_result(label: str, value):
    """Render a success result box."""
    st.markdown(
        f"""
    <div class="success-box">
        <div class="success-text">{label}</div>
        <div class="success-value">{value}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_warning(message: str):
    """Render a warning box."""
    st.markdown(
        f'<div class="warning-box">{message}</div>',
        unsafe_allow_html=True,
    )


def render_metrics(metrics: list[dict]):
    """Render metric cards. Each dict has 'label' and 'value'."""
    cards_html = ""
    for m in metrics:
        cards_html += f"""
        <div class="metric-card">
            <div class="metric-value">{m['value']}</div>
            <div class="metric-label">{m['label']}</div>
        </div>
        """
    st.markdown(f'<div class="metric-row">{cards_html}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Charts / Graphs
# ──────────────────────────────────────────────
def _chart_layout(fig, title=""):
    """Apply consistent dark theme to a plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,41,0.8)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#FAFAFA"),
        title=dict(text=title, font=dict(size=16, color="#FAFAFA"), x=0.5),
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(
            bgcolor="rgba(26,29,41,0.6)",
            bordercolor="#2D3148",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor="#2D3148", zerolinecolor="#2D3148"),
        yaxis=dict(gridcolor="#2D3148", zerolinecolor="#2D3148"),
    )
    return fig


def plot_convergence(df: pd.DataFrame, method_name: str):
    """Plot convergence chart showing how x values approach the root over iterations."""
    if df is None or df.empty:
        return

    fig = go.Figure()

    # Determine x-value column
    x_col = None
    for col in ["XR", "x_(i+1)"]:
        if col in df.columns:
            x_col = col
            break

    if x_col is None:
        return

    iterations = df["Iterasi"].tolist()
    x_vals = df[x_col].tolist()

    # X value convergence line
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=x_vals,
            mode="lines+markers",
            name=f"Nilai {x_col}",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=8, color=COLORS["primary"], line=dict(width=2, color="white")),
            fill="tozeroy",
            fillcolor="rgba(108,99,255,0.1)",
        )
    )

    # Final root line
    final_root = x_vals[-1]
    fig.add_hline(
        y=final_root,
        line_dash="dash",
        line_color=COLORS["success"],
        annotation_text=f"Akar ≈ {final_root}",
        annotation_font_color=COLORS["success"],
    )

    _chart_layout(fig, f"Konvergensi Nilai Akar — {method_name}")
    fig.update_xaxes(title_text="Iterasi", dtick=1)
    fig.update_yaxes(title_text="Nilai x")

    st.plotly_chart(fig, use_container_width=True)


def plot_error(df: pd.DataFrame, method_name: str):
    """Plot error reduction chart (Et and Ea over iterations)."""
    if df is None or df.empty:
        return

    has_et = "Et (%)" in df.columns
    has_ea = "Ea (%)" in df.columns
    if not has_et and not has_ea:
        return

    fig = go.Figure()
    iterations = df["Iterasi"].tolist()

    if has_et:
        et_vals = df["Et (%)"].tolist()
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=et_vals,
                mode="lines+markers",
                name="Et (%)",
                line=dict(color=COLORS["accent"], width=2.5),
                marker=dict(size=7, symbol="circle"),
            )
        )

    if has_ea:
        ea_vals = [v if v is not None and not (isinstance(v, float) and np.isnan(v)) else None for v in df["Ea (%)"].tolist()]
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=ea_vals,
                mode="lines+markers",
                name="Ea (%)",
                line=dict(color=COLORS["secondary"], width=2.5),
                marker=dict(size=7, symbol="diamond"),
                connectgaps=True,
            )
        )

    _chart_layout(fig, f"Penurunan Error — {method_name}")
    fig.update_xaxes(title_text="Iterasi", dtick=1)
    fig.update_yaxes(title_text="Error (%)")

    st.plotly_chart(fig, use_container_width=True)


def plot_function_with_root(fungsi_str: str, root: float, method_name: str, xl=None, xu=None):
    """Plot the function f(x) with the root marked."""
    import sympy as sp

    x = sp.symbols("x")
    try:
        f_expr = sp.sympify(fungsi_str)
        f_lambda = sp.lambdify(x, f_expr, modules=["numpy"])
    except Exception:
        return

    # Determine plot range
    if xl is not None and xu is not None:
        margin = (xu - xl) * 0.3
        x_min, x_max = xl - margin, xu + margin
    else:
        x_min, x_max = root - 10, root + 10

    x_range = np.linspace(x_min, x_max, 500)
    try:
        y_range = f_lambda(x_range)
        if isinstance(y_range, (int, float)):
            y_range = np.full_like(x_range, y_range)
    except Exception:
        return

    # Filter out extreme values for better visualization
    y_range = np.where(np.abs(y_range) > 1e6, np.nan, y_range)

    fig = go.Figure()

    # Function curve
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode="lines",
            name="f(x)",
            line=dict(color=COLORS["primary_light"], width=3),
        )
    )

    # Zero line
    fig.add_hline(y=0, line_color="#8892B0", line_width=1, line_dash="dot")

    # Root marker
    try:
        root_y = float(f_lambda(root))
    except Exception:
        root_y = 0
    fig.add_trace(
        go.Scatter(
            x=[root],
            y=[root_y],
            mode="markers",
            name=f"Akar ≈ {root}",
            marker=dict(
                size=14,
                color=COLORS["success"],
                line=dict(width=3, color="white"),
                symbol="star",
            ),
        )
    )

    # Interval markers for bracketing methods
    if xl is not None and xu is not None:
        fig.add_vrect(
            x0=xl,
            x1=xu,
            fillcolor="rgba(108,99,255,0.08)",
            line_width=0,
            annotation_text="Interval Awal",
            annotation_position="top left",
            annotation_font_color=COLORS["text_muted"],
        )

    _chart_layout(fig, f"Grafik f(x) — {method_name}")
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="f(x)")

    st.plotly_chart(fig, use_container_width=True)


def plot_polynomial_roots(roots, fungsi_str: str):
    """Plot polynomial with all roots marked."""
    import sympy as sp

    x = sp.symbols("x")
    try:
        f_expr = sp.sympify(fungsi_str)
        f_lambda = sp.lambdify(x, f_expr, modules=["numpy"])
    except Exception:
        return

    valid_roots = [r for r in roots if not (isinstance(r, float) and np.isnan(r))]
    if not valid_roots:
        return

    r_min = min(valid_roots)
    r_max = max(valid_roots)
    margin = max((r_max - r_min) * 0.5, 5)
    x_range = np.linspace(r_min - margin, r_max + margin, 500)

    try:
        y_range = f_lambda(x_range)
        if isinstance(y_range, (int, float)):
            y_range = np.full_like(x_range, y_range)
    except Exception:
        return

    y_range = np.where(np.abs(y_range) > 1e8, np.nan, y_range)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode="lines",
            name="f(x)",
            line=dict(color=COLORS["primary_light"], width=3),
        )
    )
    fig.add_hline(y=0, line_color="#8892B0", line_width=1, line_dash="dot")

    colors_roots = [COLORS["success"], COLORS["accent"], COLORS["secondary"], COLORS["warning"], "#E040FB"]
    for i, r in enumerate(valid_roots):
        try:
            ry = float(f_lambda(r))
        except Exception:
            ry = 0
        fig.add_trace(
            go.Scatter(
                x=[r],
                y=[ry],
                mode="markers",
                name=f"x{i+1} = {r}",
                marker=dict(
                    size=14,
                    color=colors_roots[i % len(colors_roots)],
                    line=dict(width=3, color="white"),
                    symbol="star",
                ),
            )
        )

    _chart_layout(fig, "Grafik Polinomial & Akar-Akar")
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="f(x)")
    st.plotly_chart(fig, use_container_width=True)


def plot_iteration_comparison(df: pd.DataFrame):
    """Plot combined convergence + error in a two-row subplot."""
    if df is None or df.empty:
        return

    x_col = None
    for col in ["XR", "x_(i+1)"]:
        if col in df.columns:
            x_col = col
            break
    if x_col is None:
        return

    has_et = "Et (%)" in df.columns
    has_ea = "Ea (%)" in df.columns
    if not has_et and not has_ea:
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Konvergensi Nilai x", "Penurunan Error"),
    )

    iterations = df["Iterasi"].tolist()

    # Top: convergence
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=df[x_col].tolist(),
            mode="lines+markers",
            name=x_col,
            line=dict(color=COLORS["primary"], width=2.5),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor="rgba(108,99,255,0.08)",
        ),
        row=1,
        col=1,
    )

    # Bottom: errors
    if has_et:
        fig.add_trace(
            go.Bar(
                x=iterations,
                y=df["Et (%)"].tolist(),
                name="Et (%)",
                marker_color=COLORS["accent"],
                opacity=0.7,
            ),
            row=2,
            col=1,
        )
    if has_ea:
        ea_vals = [v if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0 for v in df["Ea (%)"].tolist()]
        fig.add_trace(
            go.Bar(
                x=iterations,
                y=ea_vals,
                name="Ea (%)",
                marker_color=COLORS["secondary"],
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,41,0.8)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#FAFAFA"),
        height=550,
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(bgcolor="rgba(26,29,41,0.6)", bordercolor="#2D3148", borderwidth=1),
        barmode="group",
    )
    for i in range(1, 3):
        fig.update_xaxes(gridcolor="#2D3148", row=i, col=1)
        fig.update_yaxes(gridcolor="#2D3148", row=i, col=1)
    fig.update_xaxes(title_text="Iterasi", row=2, col=1, dtick=1)

    st.plotly_chart(fig, use_container_width=True)
