"""
Simple 2D dead-reckoning prototype for IO-VNBD.

This module provides a lightweight trajectory propagator for the
SIH prototype.

The current prototype uses:

    velocity + heading + elapsed time

to estimate the vehicle's movement.

It intentionally does NOT attempt to be a production-grade INS.
The purpose is to provide a stable baseline that can be visualized
and compared with the reference trajectory in the dashboard.

Coordinate convention:

    x -> East
    y -> North

Heading convention:

    0°   -> North
    90°  -> East
    180° -> South
    270° -> West
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def latlon_to_local_xy(
    latitude_deg: pd.Series | np.ndarray,
    longitude_deg: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert latitude/longitude into a local East/North coordinate system.

    The first point becomes approximately (0, 0).

    This local approximation is sufficient for the short vehicle
    trajectories used by the prototype.
    """
    latitude = np.asarray(latitude_deg, dtype=float)
    longitude = np.asarray(longitude_deg, dtype=float)

    latitude_0 = latitude[0]
    longitude_0 = longitude[0]

    earth_radius_m = 6_371_000.0

    latitude_rad = np.radians(latitude_0)

    x_east = (
        np.radians(longitude - longitude_0)
        * earth_radius_m
        * np.cos(latitude_rad)
    )

    y_north = (
        np.radians(latitude - latitude_0)
        * earth_radius_m
    )

    return x_east, y_north


def local_xy_to_latlon(
    x_east_m: np.ndarray,
    y_north_m: np.ndarray,
    reference_latitude_deg: float,
    reference_longitude_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert local East/North coordinates back to latitude/longitude.
    """
    earth_radius_m = 6_371_000.0

    latitude_0_rad = np.radians(reference_latitude_deg)

    latitude = (
        reference_latitude_deg
        + np.degrees(y_north_m / earth_radius_m)
    )

    longitude = (
        reference_longitude_deg
        + np.degrees(
            x_east_m
            / (earth_radius_m * np.cos(latitude_0_rad))
        )
    )

    return latitude, longitude


# ---------------------------------------------------------------------------
# Dead reckoning
# ---------------------------------------------------------------------------

def dead_reckon(
    velocity_mps: pd.Series | np.ndarray,
    heading_deg: pd.Series | np.ndarray,
    time_s: pd.Series | np.ndarray,
    initial_x_m: float = 0.0,
    initial_y_m: float = 0.0,
) -> pd.DataFrame:
    """
    Propagate a 2D vehicle trajectory using velocity and heading.

    Parameters
    ----------
    velocity_mps:
        Vehicle forward velocity in metres/second.

    heading_deg:
        Vehicle heading where:
            0° = North
            90° = East.

    time_s:
        Time in seconds.

    initial_x_m:
        Initial East position.

    initial_y_m:
        Initial North position.

    Returns
    -------
    pandas.DataFrame
        Columns:

            time_s
            velocity_mps
            heading_deg
            x_m
            y_m
    """
    velocity = np.asarray(velocity_mps, dtype=float)
    heading = np.asarray(heading_deg, dtype=float)
    time = np.asarray(time_s, dtype=float)

    if not (
        len(velocity)
        == len(heading)
        == len(time)
    ):
        raise ValueError(
            "velocity, heading, and time must have the same length."
        )

    if len(time) == 0:
        raise ValueError("Cannot dead-reckon an empty dataset.")

    dt = np.diff(time, prepend=time[0])

    # Protect against accidental negative/invalid time differences.
    dt = np.where(dt >= 0.0, dt, 0.0)

    heading_rad = np.radians(heading)

    # Heading is measured clockwise from North.
    east_velocity = velocity * np.sin(heading_rad)
    north_velocity = velocity * np.cos(heading_rad)

    x_m = (
        initial_x_m
        + np.cumsum(east_velocity * dt)
    )

    y_m = (
        initial_y_m
        + np.cumsum(north_velocity * dt)
    )

    return pd.DataFrame(
        {
            "time_s": time,
            "velocity_mps": velocity,
            "heading_deg": heading,
            "x_m": x_m,
            "y_m": y_m,
        }
    )

def dead_reckon_blackout(
    vehicle_df: pd.DataFrame,
    start_time_s: float,
    duration_s: float,
    initial_x_m: float,
    initial_y_m: float,
) -> pd.DataFrame:
    """
    Run dead reckoning only during a simulated GNSS blackout.

    The initial position is explicitly supplied at the beginning
    of the blackout. This represents the last reliable GNSS position
    before the signal is lost.

    Parameters
    ----------
    vehicle_df:
        Preprocessed vehicle dataframe.

    start_time_s:
        GNSS blackout start time.

    duration_s:
        GNSS blackout duration.

    initial_x_m:
        East position at blackout start.

    initial_y_m:
        North position at blackout start.

    Returns
    -------
    pandas.DataFrame
        Dead-reckoned trajectory for the blackout interval.
    """
    required_columns = {
        "velocity_mps",
        "heading_deg",
        "relative_time_s",
    }

    missing = required_columns - set(vehicle_df.columns)

    if missing:
        raise ValueError(
            "Vehicle dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )

    end_time_s = start_time_s + duration_s

    mask = (
        (vehicle_df["relative_time_s"] >= start_time_s)
        & (vehicle_df["relative_time_s"] <= end_time_s)
    )

    segment = vehicle_df.loc[mask].copy()

    if segment.empty:
        raise ValueError(
            "No vehicle samples were found inside the "
            "requested GNSS blackout interval."
        )

    # Time is made relative to the beginning of the blackout.
    time_s = (
        segment["relative_time_s"].to_numpy(dtype=float)
        - start_time_s
    )

    result = dead_reckon(
        velocity_mps=segment["velocity_mps"].to_numpy(dtype=float),
        heading_deg=segment["heading_deg"].to_numpy(dtype=float),
        time_s=time_s,
        initial_x_m=initial_x_m,
        initial_y_m=initial_y_m,
    )

    return result

def create_blackout_reference(
    vehicle_df: pd.DataFrame,
    start_time_s: float,
    duration_s: float,
) -> pd.DataFrame:
    """
    Create the reference trajectory for a simulated GNSS blackout.

    The reference coordinates are converted into a local coordinate
    system for the entire route and then retained for the selected
    blackout interval.

    The first reference point therefore represents the true position
    at the beginning of the blackout.
    """
    reference = create_reference_trajectory(vehicle_df)

    end_time_s = start_time_s + duration_s

    mask = (
        (reference["time_s"] >= start_time_s)
        & (reference["time_s"] <= end_time_s)
    )

    result = reference.loc[mask].copy()

    if result.empty:
        raise ValueError(
            "No reference samples were found inside the "
            "requested GNSS blackout interval."
        )

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reference trajectory
# ---------------------------------------------------------------------------

def create_reference_trajectory(
    vehicle_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the vehicle GPS trajectory into local East/North coordinates.

    The first GPS coordinate becomes the origin.
    """
    required_columns = {
        "latitude_deg",
        "longitude_deg",
        "relative_time_s",
    }

    missing = required_columns - set(vehicle_df.columns)

    if missing:
        raise ValueError(
            "Vehicle dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )

    x_m, y_m = latlon_to_local_xy(
        vehicle_df["latitude_deg"],
        vehicle_df["longitude_deg"],
    )

    return pd.DataFrame(
        {
            "time_s": vehicle_df["relative_time_s"].to_numpy(
                dtype=float
            ),
            "latitude_deg": vehicle_df["latitude_deg"].to_numpy(
                dtype=float
            ),
            "longitude_deg": vehicle_df["longitude_deg"].to_numpy(
                dtype=float
            ),
            "x_m": x_m,
            "y_m": y_m,
        }
    )


# ---------------------------------------------------------------------------
# GPS-denied simulation
# ---------------------------------------------------------------------------

def simulate_gps_denial(
    reference_df: pd.DataFrame,
    start_time_s: float,
    duration_s: float,
) -> pd.DataFrame:
    """
    Mark a section of the reference trajectory as GPS-denied.

    This does not remove data.

    Instead, it adds a boolean column:

        gps_available

    This makes it easy for the dashboard to visualize where GPS
    is available and where dead reckoning is being used.
    """
    result = reference_df.copy()

    end_time_s = start_time_s + duration_s

    result["gps_available"] = (
        (result["time_s"] < start_time_s)
        | (result["time_s"] > end_time_s)
    )

    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def calculate_position_error(
    dead_reckoning_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate horizontal position error between two trajectories.

    Both dataframes must contain:

        time_s
        x_m
        y_m

    For the current synchronized prototype, rows are compared
    directly.
    """
    if len(dead_reckoning_df) != len(reference_df):
        raise ValueError(
            "Dead-reckoning and reference trajectories must have "
            "the same number of rows for direct comparison."
        )

    result = dead_reckoning_df.copy()

    dx = (
        result["x_m"].to_numpy()
        - reference_df["x_m"].to_numpy()
    )

    dy = (
        result["y_m"].to_numpy()
        - reference_df["y_m"].to_numpy()
    )

    result["error_x_m"] = dx
    result["error_y_m"] = dy

    result["position_error_m"] = np.sqrt(
        dx**2 + dy**2
    )

    return result


# ---------------------------------------------------------------------------
# Complete prototype pipeline
# ---------------------------------------------------------------------------

def run_dead_reckoning_prototype(
    vehicle_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete baseline trajectory calculation.

    Returns:

        dead_reckoning
        reference
    """
    required_columns = {
        "velocity_mps",
        "heading_deg",
        "relative_time_s",
    }

    missing = required_columns - set(vehicle_df.columns)

    if missing:
        raise ValueError(
            "Vehicle dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )

    dead_reckoning = dead_reckon(
        velocity_mps=vehicle_df["velocity_mps"],
        heading_deg=vehicle_df["heading_deg"],
        time_s=vehicle_df["relative_time_s"],
    )

    reference = create_reference_trajectory(vehicle_df)

    return dead_reckoning, reference


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.loader import load_vehicle_dataset
    from data.preprocessing import preprocess_vehicle

    print("IO-VNBD dead-reckoning prototype test")
    print()

    print("Loading S1 vehicle dataset...")
    vehicle_raw = load_vehicle_dataset("S1")

    print("Preprocessing vehicle dataset...")
    vehicle = preprocess_vehicle(vehicle_raw)

    print("Running dead reckoning...")
    dead_reckoning, reference = run_dead_reckoning_prototype(
        vehicle
    )

    comparison = calculate_position_error(
        dead_reckoning,
        reference,
    )

    print()
    print("=" * 70)
    print("Dead-Reckoning Result")
    print("=" * 70)

    print(f"Samples: {len(comparison):,}")
    print(
        f"Duration: "
        f"{comparison['time_s'].iloc[-1]:.3f} seconds"
    )

    print()
    print("Final dead-reckoning position:")
    print(
        f"  East:  "
        f"{dead_reckoning['x_m'].iloc[-1]:.2f} m"
    )
    print(
        f"  North: "
        f"{dead_reckoning['y_m'].iloc[-1]:.2f} m"
    )

    print()
    print("Reference final position:")
    print(
        f"  East:  "
        f"{reference['x_m'].iloc[-1]:.2f} m"
    )
    print(
        f"  North: "
        f"{reference['y_m'].iloc[-1]:.2f} m"
    )

    print()
    print("Position error:")
    print(
        f"  Final: "
        f"{comparison['position_error_m'].iloc[-1]:.2f} m"
    )
    print(
        f"  Maximum: "
        f"{comparison['position_error_m'].max():.2f} m"
    )
    print(
        f"  Mean: "
        f"{comparison['position_error_m'].mean():.2f} m"
    )

    print()
    print("First five trajectory points:")
    print(
        comparison[
            [
                "time_s",
                "x_m",
                "y_m",
                "position_error_m",
            ]
        ]
        .head()
        .to_string(index=False)
    )

    print()
    print("Dead-reckoning prototype test completed successfully.")

if __name__ == "__main__":
    from data.loader import load_vehicle_dataset
    from data.preprocessing import preprocess_vehicle

    print("IO-VNBD GNSS blackout dead-reckoning test")
    print()

    print("Loading S1 vehicle dataset...")
    vehicle_raw = load_vehicle_dataset("S1")

    print("Preprocessing vehicle dataset...")
    vehicle = preprocess_vehicle(vehicle_raw)

    blackout_start = 30.0
    blackout_duration = 60.0

    print()
    print(
        f"GNSS blackout: "
        f"{blackout_start:.1f}s → "
        f"{blackout_start + blackout_duration:.1f}s"
    )

    # Build the complete reference trajectory.
    reference = create_reference_trajectory(vehicle)

    # Find the true position exactly at the beginning of the
    # blackout. The closest available sample is used.
    start_index = (
        (reference["time_s"] - blackout_start)
        .abs()
        .idxmin()
    )

    initial_x = float(
        reference.loc[start_index, "x_m"]
    )

    initial_y = float(
        reference.loc[start_index, "y_m"]
    )

    print()
    print("GNSS position used to initialize DR:")
    print(f"  East:  {initial_x:.2f} m")
    print(f"  North: {initial_y:.2f} m")

    # Run DR only after the GNSS signal is lost.
    dead_reckoning = dead_reckon_blackout(
        vehicle_df=vehicle,
        start_time_s=blackout_start,
        duration_s=blackout_duration,
        initial_x_m=initial_x,
        initial_y_m=initial_y,
    )

    blackout_reference = create_blackout_reference(
        vehicle_df=vehicle,
        start_time_s=blackout_start,
        duration_s=blackout_duration,
    )

    print()
    print("Blackout samples:")
    print(f"  DR:        {len(dead_reckoning):,}")
    print(f"  Reference: {len(blackout_reference):,}")

    if len(dead_reckoning) != len(blackout_reference):
        raise RuntimeError(
            "DR and reference blackout trajectories have "
            "different sample counts."
        )

    print()
    print("Blackout trajectory completed successfully.")

    print()
    print("Final positions:")

    print(
        f"  DR:        "
        f"({dead_reckoning['x_m'].iloc[-1]:.2f}, "
        f"{dead_reckoning['y_m'].iloc[-1]:.2f})"
    )

    print(
        f"  Reference: "
        f"({blackout_reference['x_m'].iloc[-1]:.2f}, "
        f"{blackout_reference['y_m'].iloc[-1]:.2f})"
    )

    dx = (
        dead_reckoning["x_m"].iloc[-1]
        - blackout_reference["x_m"].iloc[-1]
    )

    dy = (
        dead_reckoning["y_m"].iloc[-1]
        - blackout_reference["y_m"].iloc[-1]
    )

    final_error = float(
        np.sqrt(dx**2 + dy**2)
    )

    print()
    print(
        f"Final blackout drift: "
        f"{final_error:.2f} m"
    )

    print()
    print(
        "GNSS blackout dead-reckoning test completed successfully."
    )