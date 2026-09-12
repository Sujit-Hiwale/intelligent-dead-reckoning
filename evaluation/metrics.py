"""
Evaluation and SIH benchmark metrics.

This module evaluates positioning performance during a simulated
GNSS-denied interval.

The main benchmark used by the prototype is:

    positional drift < 10% of distance travelled

The module also calculates the position update rate.

This is intentionally a lightweight evaluation layer for the
SIH prototype and dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

DEFAULT_DRIFT_LIMIT_PERCENT = 10.0
DEFAULT_TARGET_UPDATE_RATE_HZ = 10.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class DriftMetrics:
    """Results describing positional drift."""

    distance_travelled_m: float
    final_error_m: float
    maximum_error_m: float
    mean_error_m: float
    drift_percent: float
    passed: bool


@dataclass
class UpdateRateMetrics:
    """Results describing the trajectory update rate."""

    mean_update_rate_hz: float
    minimum_update_rate_hz: float
    maximum_update_rate_hz: float
    target_update_rate_hz: float
    passed: bool


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def calculate_distance_travelled(
    trajectory_df: pd.DataFrame,
) -> float:
    """
    Calculate total 2D distance travelled.

    The dataframe must contain:

        x_m
        y_m

    Consecutive points are joined to calculate the travelled distance.
    """
    required = {"x_m", "y_m"}

    missing = required - set(trajectory_df.columns)

    if missing:
        raise ValueError(
            "Trajectory dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )

    x = trajectory_df["x_m"].to_numpy(dtype=float)
    y = trajectory_df["y_m"].to_numpy(dtype=float)

    if len(x) < 2:
        return 0.0

    dx = np.diff(x)
    dy = np.diff(y)

    segment_distance = np.sqrt(
        dx**2 + dy**2
    )

    return float(np.sum(segment_distance))


# ---------------------------------------------------------------------------
# Drift evaluation
# ---------------------------------------------------------------------------

def calculate_drift_metrics(
    dead_reckoning_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    drift_limit_percent: float = DEFAULT_DRIFT_LIMIT_PERCENT,
) -> DriftMetrics:
    """
    Calculate positional drift against a reference trajectory.

    Both dataframes must contain:

        x_m
        y_m

    The two trajectories must contain the same number of samples.
    """
    if len(dead_reckoning_df) != len(reference_df):
        raise ValueError(
            "Dead-reckoning and reference trajectories must have "
            "the same number of samples."
        )

    required = {"x_m", "y_m"}

    for name, df in (
        ("dead-reckoning", dead_reckoning_df),
        ("reference", reference_df),
    ):
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name.title()} trajectory is missing required "
                f"columns: {', '.join(sorted(missing))}"
            )

    dr_x = dead_reckoning_df["x_m"].to_numpy(dtype=float)
    dr_y = dead_reckoning_df["y_m"].to_numpy(dtype=float)

    ref_x = reference_df["x_m"].to_numpy(dtype=float)
    ref_y = reference_df["y_m"].to_numpy(dtype=float)

    error = np.sqrt(
        (dr_x - ref_x) ** 2
        + (dr_y - ref_y) ** 2
    )

    distance = calculate_distance_travelled(reference_df)

    final_error = float(error[-1])
    maximum_error = float(np.max(error))
    mean_error = float(np.mean(error))

    if distance > 0.0:
        drift_percent = (
            final_error / distance
        ) * 100.0
    else:
        drift_percent = 0.0

    passed = drift_percent < drift_limit_percent

    return DriftMetrics(
        distance_travelled_m=distance,
        final_error_m=final_error,
        maximum_error_m=maximum_error,
        mean_error_m=mean_error,
        drift_percent=drift_percent,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# GNSS blackout-specific evaluation
# ---------------------------------------------------------------------------

def evaluate_gnss_blackout(
    dead_reckoning_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    start_time_s: float,
    duration_s: float,
    drift_limit_percent: float = DEFAULT_DRIFT_LIMIT_PERCENT,
) -> DriftMetrics:
    """
    Evaluate dead-reckoning performance during a GNSS blackout.

    Only the samples inside the requested blackout interval are used.

    Parameters
    ----------
    start_time_s:
        Start of the GNSS blackout.

    duration_s:
        Duration of the blackout in seconds.
    """
    if "time_s" not in dead_reckoning_df.columns:
        raise ValueError(
            "Dead-reckoning dataframe must contain 'time_s'."
        )

    if "time_s" not in reference_df.columns:
        raise ValueError(
            "Reference dataframe must contain 'time_s'."
        )

    end_time_s = start_time_s + duration_s

    dr_mask = (
        (dead_reckoning_df["time_s"] >= start_time_s)
        & (dead_reckoning_df["time_s"] <= end_time_s)
    )

    ref_mask = (
        (reference_df["time_s"] >= start_time_s)
        & (reference_df["time_s"] <= end_time_s)
    )

    dr_segment = dead_reckoning_df.loc[dr_mask].reset_index(
        drop=True
    )

    ref_segment = reference_df.loc[ref_mask].reset_index(
        drop=True
    )

    if dr_segment.empty or ref_segment.empty:
        raise ValueError(
            "The requested GNSS blackout interval contains no data."
        )

    if len(dr_segment) != len(ref_segment):
        raise ValueError(
            "Dead-reckoning and reference blackout segments "
            "contain different numbers of samples."
        )

    return calculate_drift_metrics(
        dead_reckoning_df=dr_segment,
        reference_df=ref_segment,
        drift_limit_percent=drift_limit_percent,
    )


# ---------------------------------------------------------------------------
# Update-rate evaluation
# ---------------------------------------------------------------------------

def calculate_update_rate(
    time_s: pd.Series | np.ndarray,
    target_update_rate_hz: float = DEFAULT_TARGET_UPDATE_RATE_HZ,
) -> UpdateRateMetrics:
    """
    Calculate the effective position update rate.

    The update rate is calculated from consecutive timestamps.

    For example:

        Δt = 0.1 s
        update rate = 1 / 0.1 = 10 Hz
    """
    time = np.asarray(time_s, dtype=float)

    if len(time) < 2:
        raise ValueError(
            "At least two timestamps are required."
        )

    dt = np.diff(time)

    # Ignore invalid or zero-length intervals.
    valid_dt = dt[dt > 0.0]

    if len(valid_dt) == 0:
        raise ValueError(
            "No valid positive time intervals were found."
        )

    update_rates = 1.0 / valid_dt

    mean_rate = float(np.mean(update_rates))
    minimum_rate = float(np.min(update_rates))
    maximum_rate = float(np.max(update_rates))

    # The prototype considers the target achieved when the average
    # update rate is at least the requested rate.
    passed = mean_rate >= target_update_rate_hz

    return UpdateRateMetrics(
        mean_update_rate_hz=mean_rate,
        minimum_update_rate_hz=minimum_rate,
        maximum_update_rate_hz=maximum_rate,
        target_update_rate_hz=target_update_rate_hz,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Dashboard-friendly summary
# ---------------------------------------------------------------------------

def create_benchmark_summary(
    drift_metrics: DriftMetrics,
    update_metrics: UpdateRateMetrics,
) -> dict[str, object]:
    """
    Convert benchmark results into a dictionary convenient for
    Streamlit metrics/cards.
    """
    return {
        "distance_travelled_m": drift_metrics.distance_travelled_m,
        "final_error_m": drift_metrics.final_error_m,
        "maximum_error_m": drift_metrics.maximum_error_m,
        "mean_error_m": drift_metrics.mean_error_m,
        "drift_percent": drift_metrics.drift_percent,
        "drift_limit_percent": DEFAULT_DRIFT_LIMIT_PERCENT,
        "drift_passed": drift_metrics.passed,
        "update_rate_hz": update_metrics.mean_update_rate_hz,
        "target_update_rate_hz": update_metrics.target_update_rate_hz,
        "update_rate_passed": update_metrics.passed,
        "overall_passed": (
            drift_metrics.passed
            and update_metrics.passed
        ),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def print_benchmark_report(
    drift_metrics: DriftMetrics,
    update_metrics: UpdateRateMetrics,
) -> None:
    """Print a human-readable benchmark report."""

    print("=" * 70)
    print("SIH PERFORMANCE BENCHMARK")
    print("=" * 70)

    print()
    print("POSITIONAL DRIFT")
    print("-" * 70)
    print(
        f"Distance travelled: "
        f"{drift_metrics.distance_travelled_m:.2f} m"
    )
    print(
        f"Final position error: "
        f"{drift_metrics.final_error_m:.2f} m"
    )
    print(
        f"Maximum position error: "
        f"{drift_metrics.maximum_error_m:.2f} m"
    )
    print(
        f"Mean position error: "
        f"{drift_metrics.mean_error_m:.2f} m"
    )
    print(
        f"Drift: "
        f"{drift_metrics.drift_percent:.2f}%"
    )
    print(
        f"Limit: "
        f"{DEFAULT_DRIFT_LIMIT_PERCENT:.2f}%"
    )
    print(
        f"Result: "
        f"{'PASS ✓' if drift_metrics.passed else 'FAIL ✗'}"
    )

    print()
    print("POSITION UPDATE RATE")
    print("-" * 70)
    print(
        f"Mean update rate: "
        f"{update_metrics.mean_update_rate_hz:.2f} Hz"
    )
    print(
        f"Minimum update rate: "
        f"{update_metrics.minimum_update_rate_hz:.2f} Hz"
    )
    print(
        f"Maximum update rate: "
        f"{update_metrics.maximum_update_rate_hz:.2f} Hz"
    )
    print(
        f"Target: "
        f"{update_metrics.target_update_rate_hz:.2f} Hz"
    )
    print(
        f"Result: "
        f"{'PASS ✓' if update_metrics.passed else 'FAIL ✗'}"
    )

    print()
    print("=" * 70)
    print(
        "OVERALL: "
        + (
            "PASS ✓"
            if drift_metrics.passed and update_metrics.passed
            else "FAIL ✗"
        )
    )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.loader import load_vehicle_dataset
    from data.preprocessing import preprocess_vehicle
    from navigation.dead_reckoning import (
        calculate_position_error,
        run_dead_reckoning_prototype,
    )

    print("IO-VNBD evaluation test")
    print()

    print("Loading S1 vehicle dataset...")
    vehicle_raw = load_vehicle_dataset("S1")

    print("Preprocessing...")
    vehicle = preprocess_vehicle(vehicle_raw)

    print("Running dead reckoning...")
    dead_reckoning, reference = run_dead_reckoning_prototype(
        vehicle
    )

    print("Calculating position error...")
    comparison = calculate_position_error(
        dead_reckoning,
        reference,
    )

    # Use a 60-second blackout beginning 30 seconds into the route.
    blackout_start = 30.0
    blackout_duration = 60.0

    print()
    print(
        f"Evaluating GNSS blackout: "
        f"{blackout_start:.1f}s → "
        f"{blackout_start + blackout_duration:.1f}s"
    )

    # Evaluate the blackout using the DR/reference trajectories.
    blackout_dr = comparison[
        (
            comparison["time_s"] >= blackout_start
        )
        & (
            comparison["time_s"]
            <= blackout_start + blackout_duration
        )
    ].copy()

    blackout_reference = reference[
        (
            reference["time_s"] >= blackout_start
        )
        & (
            reference["time_s"]
            <= blackout_start + blackout_duration
        )
    ].copy()

    drift_metrics = calculate_drift_metrics(
        dead_reckoning_df=blackout_dr,
        reference_df=blackout_reference,
    )

    update_metrics = calculate_update_rate(
        blackout_dr["time_s"],
    )

    print()
    print_benchmark_report(
        drift_metrics,
        update_metrics,
    )

    print()
    print("Dashboard summary:")
    summary = create_benchmark_summary(
        drift_metrics,
        update_metrics,
    )

    for key, value in summary.items():
        print(f"  {key}: {value}")

    print()
    print("Evaluation test completed successfully.")