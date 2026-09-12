"""
Streamlit dashboard for the Intelligent Dead Reckoning prototype.

This dashboard demonstrates:

    1. IO-VNBD route selection
    2. GNSS-denied scenario simulation
    3. Dead-reckoning trajectory
    4. Reference trajectory
    5. Position-error growth
    6. SIH drift benchmark
    7. Position update-rate benchmark

This is a prototype demonstration dashboard, not a production
navigation application.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Make the project root importable when Streamlit starts from dashboard/
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from data.loader import load_vehicle_dataset
from data.preprocessing import preprocess_vehicle
from navigation.dead_reckoning import (
    calculate_position_error,
    run_dead_reckoning_prototype,
)
from evaluation.metrics import (
    DEFAULT_DRIFT_LIMIT_PERCENT,
    DEFAULT_TARGET_UPDATE_RATE_HZ,
    calculate_drift_metrics,
    calculate_update_rate,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Intelligent Dead Reckoning",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }

    .status-pass {
        padding: 1rem;
        border-radius: 0.7rem;
        border: 1px solid rgba(40, 180, 90, 0.4);
        background: rgba(40, 180, 90, 0.08);
        text-align: center;
    }

    .status-fail {
        padding: 1rem;
        border-radius: 0.7rem;
        border: 1px solid rgba(220, 70, 70, 0.4);
        background: rgba(220, 70, 70, 0.08);
        text-align: center;
    }

    .status-title {
        font-size: 1.35rem;
        font-weight: 700;
    }

    .status-detail {
        margin-top: 0.3rem;
        opacity: 0.8;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_route(dataset_id: str) -> pd.DataFrame:
    """Load and preprocess one vehicle route."""
    raw = load_vehicle_dataset(dataset_id)
    return preprocess_vehicle(raw)


@st.cache_data
def calculate_route(dataset_id: str):
    """Calculate DR and reference trajectories for a route."""
    vehicle = load_route(dataset_id)

    dead_reckoning, reference = run_dead_reckoning_prototype(
        vehicle
    )

    comparison = calculate_position_error(
        dead_reckoning,
        reference,
    )

    return vehicle, dead_reckoning, reference, comparison


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def create_trajectory_plot(
    dead_reckoning: pd.DataFrame,
    reference: pd.DataFrame,
    start_time_s: float,
    end_time_s: float,
) -> go.Figure:
    """Create the main trajectory comparison plot."""

    fig = go.Figure()

    # Reference trajectory
    fig.add_trace(
        go.Scatter(
            x=reference["x_m"],
            y=reference["y_m"],
            mode="lines",
            name="GNSS Reference",
            line=dict(width=3),
        )
    )

    # Dead reckoning
    fig.add_trace(
        go.Scatter(
            x=dead_reckoning["x_m"],
            y=dead_reckoning["y_m"],
            mode="lines",
            name="Dead Reckoning",
            line=dict(width=2),
        )
    )

    # Determine blackout samples.
    blackout = dead_reckoning[
        (dead_reckoning["time_s"] >= start_time_s)
        & (dead_reckoning["time_s"] <= end_time_s)
    ]

    if not blackout.empty:
        fig.add_trace(
            go.Scatter(
                x=blackout["x_m"],
                y=blackout["y_m"],
                mode="lines",
                name="DR during GNSS blackout",
                line=dict(
                    width=4,
                    dash="dot",
                ),
            )
        )

    fig.update_layout(
        title="Vehicle Trajectory",
        xaxis_title="East displacement (m)",
        yaxis_title="North displacement (m)",
        hovermode="closest",
        height=600,
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=30,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def create_error_plot(
    comparison: pd.DataFrame,
    start_time_s: float,
    end_time_s: float,
) -> go.Figure:
    """Create the position-error graph."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=comparison["time_s"],
            y=comparison["position_error_m"],
            mode="lines",
            name="Position Error",
            line=dict(width=2),
        )
    )

    fig.add_vrect(
        x0=start_time_s,
        x1=end_time_s,
        fillcolor="gray",
        opacity=0.15,
        line_width=0,
        annotation_text="GNSS BLACKOUT",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Position Error",
        xaxis_title="Time (s)",
        yaxis_title="Position error (m)",
        height=400,
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=30,
        ),
        hovermode="x unified",
    )

    return fig


def create_speed_plot(
    vehicle: pd.DataFrame,
    start_time_s: float,
    end_time_s: float,
) -> go.Figure:
    """Create the vehicle speed graph."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=vehicle["relative_time_s"],
            y=vehicle["velocity_mps"] * 3.6,
            mode="lines",
            name="Vehicle Speed",
            line=dict(width=2),
        )
    )

    fig.add_vrect(
        x0=start_time_s,
        x1=end_time_s,
        fillcolor="gray",
        opacity=0.15,
        line_width=0,
    )

    fig.update_layout(
        title="Vehicle Speed",
        xaxis_title="Time (s)",
        yaxis_title="Speed (km/h)",
        height=400,
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=30,
        ),
        hovermode="x unified",
    )

    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="main-title">'
    "🛰️ Intelligent Dead Reckoning"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "GNSS-denied positioning demonstrator • IO-VNBD benchmark"
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Experiment")

dataset_id = st.sidebar.selectbox(
    "Dataset",
    ["S1"],
    help="IO-VNBD synchronized vehicle route.",
)

st.sidebar.markdown("---")

st.sidebar.subheader("GNSS Blackout")

# Load route so that the sliders can use the actual duration.
vehicle = load_route(dataset_id)

route_duration = float(
    vehicle["relative_time_s"].iloc[-1]
)

max_blackout_duration = min(
    300.0,
    route_duration,
)

default_start = min(30.0, max(0.0, route_duration - 60.0))
default_duration = min(60.0, max_blackout_duration)

blackout_start = st.sidebar.slider(
    "Blackout start (s)",
    min_value=0.0,
    max_value=float(max(0.0, route_duration - 1.0)),
    value=float(default_start),
    step=1.0,
)

blackout_duration = st.sidebar.slider(
    "Blackout duration (s)",
    min_value=1.0,
    max_value=float(
        max(1.0, min(max_blackout_duration, route_duration - blackout_start))
    ),
    value=float(
        min(
            default_duration,
            max(1.0, route_duration - blackout_start),
        )
    ),
    step=1.0,
)

blackout_end = blackout_start + blackout_duration

st.sidebar.markdown("---")

st.sidebar.subheader("Benchmark")

st.sidebar.write(
    f"Drift limit: **{DEFAULT_DRIFT_LIMIT_PERCENT:.0f}%**"
)

st.sidebar.write(
    f"Target update rate: **{DEFAULT_TARGET_UPDATE_RATE_HZ:.0f} Hz**"
)


# ---------------------------------------------------------------------------
# Run calculation
# ---------------------------------------------------------------------------

with st.spinner("Running positioning simulation..."):
    (
        vehicle,
        dead_reckoning,
        reference,
        comparison,
    ) = calculate_route(dataset_id)


# ---------------------------------------------------------------------------
# Route information
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Route Overview</div>',
    unsafe_allow_html=True,
)

overview_1, overview_2, overview_3, overview_4 = st.columns(4)

overview_1.metric(
    "Dataset",
    dataset_id,
)

overview_2.metric(
    "Samples",
    f"{len(vehicle):,}",
)

overview_3.metric(
    "Duration",
    f"{route_duration / 60:.1f} min",
)

overview_4.metric(
    "Nominal update rate",
    "10 Hz",
)


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------

blackout_dr = comparison[
    (comparison["time_s"] >= blackout_start)
    & (comparison["time_s"] <= blackout_end)
].copy()

blackout_reference = reference[
    (reference["time_s"] >= blackout_start)
    & (reference["time_s"] <= blackout_end)
].copy()

if blackout_dr.empty or blackout_reference.empty:
    st.error(
        "The selected blackout interval does not contain enough data."
    )
    st.stop()

drift_metrics = calculate_drift_metrics(
    dead_reckoning_df=blackout_dr,
    reference_df=blackout_reference,
)

update_metrics = calculate_update_rate(
    blackout_dr["time_s"],
)


# ---------------------------------------------------------------------------
# Benchmark cards
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="section-title">SIH Performance Benchmark</div>',
    unsafe_allow_html=True,
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Distance travelled",
    f"{drift_metrics.distance_travelled_m:.1f} m",
)

metric_2.metric(
    "Final drift",
    f"{drift_metrics.final_error_m:.2f} m",
)

metric_3.metric(
    "Drift ratio",
    f"{drift_metrics.drift_percent:.2f}%",
    delta=(
        f"{drift_metrics.drift_percent - DEFAULT_DRIFT_LIMIT_PERCENT:.2f} pp"
    ),
    delta_color="inverse",
)

metric_4.metric(
    "Update rate",
    f"{update_metrics.mean_update_rate_hz:.2f} Hz",
)


# ---------------------------------------------------------------------------
# Overall status
# ---------------------------------------------------------------------------

overall_passed = (
    drift_metrics.passed
    and update_metrics.passed
)

if overall_passed:
    st.markdown(
        f"""
        <div class="status-pass">
            <div class="status-title">
                ✓ SIH BENCHMARK PASSED
            </div>
            <div class="status-detail">
                Drift {drift_metrics.drift_percent:.2f}%
                &lt; {DEFAULT_DRIFT_LIMIT_PERCENT:.0f}% limit
                &nbsp; • &nbsp;
                {update_metrics.mean_update_rate_hz:.1f} Hz update rate
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    reasons = []

    if not drift_metrics.passed:
        reasons.append(
            f"drift exceeds {DEFAULT_DRIFT_LIMIT_PERCENT:.0f}%"
        )

    if not update_metrics.passed:
        reasons.append(
            f"update rate below {DEFAULT_TARGET_UPDATE_RATE_HZ:.0f} Hz"
        )

    st.markdown(
        f"""
        <div class="status-fail">
            <div class="status-title">
                ✗ SIH BENCHMARK NOT MET
            </div>
            <div class="status-detail">
                {", ".join(reasons)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main trajectory
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="section-title">Trajectory</div>',
    unsafe_allow_html=True,
)

trajectory_fig = create_trajectory_plot(
    dead_reckoning=dead_reckoning,
    reference=reference,
    start_time_s=blackout_start,
    end_time_s=blackout_end,
)

st.plotly_chart(
    trajectory_fig,
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Error + speed
# ---------------------------------------------------------------------------

left, right = st.columns(2)

with left:
    error_fig = create_error_plot(
        comparison=comparison,
        start_time_s=blackout_start,
        end_time_s=blackout_end,
    )

    st.plotly_chart(
        error_fig,
        use_container_width=True,
    )

with right:
    speed_fig = create_speed_plot(
        vehicle=vehicle,
        start_time_s=blackout_start,
        end_time_s=blackout_end,
    )

    st.plotly_chart(
        speed_fig,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Blackout details
# ---------------------------------------------------------------------------

with st.expander("GNSS Blackout Details", expanded=True):
    detail_1, detail_2, detail_3 = st.columns(3)

    detail_1.metric(
        "Blackout start",
        f"{blackout_start:.0f} s",
    )

    detail_2.metric(
        "Blackout end",
        f"{blackout_end:.0f} s",
    )

    detail_3.metric(
        "Blackout duration",
        f"{blackout_duration:.0f} s",
    )

    st.write(
        "During the selected interval, the dashboard evaluates "
        "dead-reckoning performance against the reference trajectory."
    )


# ---------------------------------------------------------------------------
# Technical information
# ---------------------------------------------------------------------------

with st.expander("Technical Information"):
    st.markdown(
        """
        **Current prototype**

        The baseline trajectory uses vehicle velocity and heading
        measurements to propagate a 2D position estimate.

        **Benchmark**

        Positional drift is calculated as:

        `final position error / distance travelled × 100`

        The prototype target is less than 10%.

        **Update rate**

        Position update rate is calculated from consecutive trajectory
        timestamps.

        The current IO-VNBD smartphone sampling configuration provides
        approximately 10 Hz data.

        **Important**

        This dashboard is a prototype demonstrator. The current
        dead-reckoning baseline is not yet the final smartphone-IMU
        fusion algorithm.
        """
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")

st.caption(
    "Intelligent Dead Reckoning • SIH Prototype • IO-VNBD"
)