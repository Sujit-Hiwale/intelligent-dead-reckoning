"""
Prototype smartphone-IMU dead reckoning for IO-VNBD.

This module implements a lightweight smartphone-IMU dead-reckoning
demonstrator for the SIH prototype.

The prototype uses:

    - smartphone accelerometer
    - smartphone gravity vector
    - smartphone gyroscope yaw rate
    - GNSS-derived initial position
    - GNSS-derived initial speed
    - GNSS-derived initial heading

A simple calibration step estimates which horizontal smartphone
acceleration direction best corresponds to vehicle longitudinal
acceleration.

This is a prototype, not a production INS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_columns(
    df: pd.DataFrame,
    columns: set[str],
) -> None:
    """Ensure that all required columns exist."""

    missing = columns - set(df.columns)

    if missing:
        raise ValueError(
            "Dataframe is missing required columns: "
            + ", ".join(sorted(missing))
        )


def _wrap_angle_rad(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Linear acceleration
# ---------------------------------------------------------------------------

def calculate_linear_acceleration(
    smartphone_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove the gravity vector from smartphone accelerometer readings.
    """

    _require_columns(
        smartphone_df,
        {
            "accelerometer_x_mps2",
            "accelerometer_y_mps2",
            "accelerometer_z_mps2",
            "gravity_x_mps2",
            "gravity_y_mps2",
            "gravity_z_mps2",
        },
    )

    result = smartphone_df.copy()

    result["linear_accel_x_mps2"] = (
        result["accelerometer_x_mps2"]
        - result["gravity_x_mps2"]
    )

    result["linear_accel_y_mps2"] = (
        result["accelerometer_y_mps2"]
        - result["gravity_y_mps2"]
    )

    result["linear_accel_z_mps2"] = (
        result["accelerometer_z_mps2"]
        - result["gravity_z_mps2"]
    )

    return result


# ---------------------------------------------------------------------------
# Blackout extraction
# ---------------------------------------------------------------------------

def extract_blackout(
    smartphone_df: pd.DataFrame,
    start_time_s: float,
    duration_s: float,
) -> pd.DataFrame:
    """Extract and rebase one smartphone blackout interval."""

    _require_columns(
        smartphone_df,
        {"relative_time_s"},
    )

    end_time_s = start_time_s + duration_s

    result = smartphone_df[
        (smartphone_df["relative_time_s"] >= start_time_s)
        & (smartphone_df["relative_time_s"] <= end_time_s)
    ].copy()

    if result.empty:
        raise ValueError(
            "No smartphone samples were found in the "
            "requested blackout interval."
        )

    result.reset_index(drop=True, inplace=True)

    result["blackout_time_s"] = (
        result["relative_time_s"]
        - result["relative_time_s"].iloc[0]
    )

    return result


# ---------------------------------------------------------------------------
# Smartphone acceleration calibration
# ---------------------------------------------------------------------------

def calibrate_longitudinal_acceleration(
    smartphone_df: pd.DataFrame,
    vehicle_df: pd.DataFrame,
    calibration_start_s: float,
    calibration_end_s: float,
) -> tuple[float, float, float]:
    """
    Learn a 2D mapping from smartphone horizontal acceleration
    to vehicle longitudinal acceleration.

    The model is:

        vehicle_accel =
            beta_x * phone_accel_x
            + beta_y * phone_accel_y
            + bias

    The calibration is performed immediately before GNSS blackout,
    while GNSS/reference data are still available.
    """

    _require_columns(
        smartphone_df,
        {
            "relative_time_s",
            "linear_accel_x_mps2",
            "linear_accel_y_mps2",
        },
    )

    _require_columns(
        vehicle_df,
        {
            "relative_time_s",
            "indicated_longitudinal_acceleration_mps2",
        },
    )

    phone = smartphone_df[
        (smartphone_df["relative_time_s"] >= calibration_start_s)
        & (smartphone_df["relative_time_s"] <= calibration_end_s)
    ].copy()

    vehicle = vehicle_df[
        (vehicle_df["relative_time_s"] >= calibration_start_s)
        & (vehicle_df["relative_time_s"] <= calibration_end_s)
    ].copy()

    if len(phone) < 10 or len(vehicle) < 10:
        raise ValueError(
            "Not enough data for longitudinal acceleration calibration."
        )

    phone_time = phone["relative_time_s"].to_numpy(dtype=float)

    vehicle_time = vehicle["relative_time_s"].to_numpy(dtype=float)

    vehicle_accel = vehicle[
        "indicated_longitudinal_acceleration_mps2"
    ].to_numpy(dtype=float)

    reference_accel = np.interp(
        phone_time,
        vehicle_time,
        vehicle_accel,
    )

    phone_x = phone[
        "linear_accel_x_mps2"
    ].to_numpy(dtype=float)

    phone_y = phone[
        "linear_accel_y_mps2"
    ].to_numpy(dtype=float)

    # Fit:
    #
    # reference = beta_x * phone_x
    #           + beta_y * phone_y
    #           + bias
    #
    # This allows an arbitrary horizontal direction instead of
    # forcing the phone X or Y axis to be the vehicle's longitudinal axis.

    design = np.column_stack(
        [
            phone_x,
            phone_y,
            np.ones(len(phone_x)),
        ]
    )

    beta, *_ = np.linalg.lstsq(
        design,
        reference_accel,
        rcond=None,
    )

    beta_x = float(beta[0])
    beta_y = float(beta[1])
    bias = float(beta[2])

    return beta_x, beta_y, bias

# ---------------------------------------------------------------------------
# IMU dead reckoning
# ---------------------------------------------------------------------------

def run_imu_dead_reckoning(
    smartphone_df: pd.DataFrame,
    initial_x_m: float,
    initial_y_m: float,
    initial_speed_mps: float,
    initial_heading_deg: float,
    accel_beta_x: float,
    accel_beta_y: float,
    accel_bias: float,
) -> pd.DataFrame:
    """
    Propagate position using smartphone IMU measurements.

    The initial GNSS position, speed and heading represent the last
    reliable state before GNSS loss.
    """

    _require_columns(
        smartphone_df,
        {
            "blackout_time_s",
            "linear_accel_x_mps2",
            "linear_accel_y_mps2",
            "gyroscope_yaw_radps",
        },
    )

    time = smartphone_df[
        "blackout_time_s"
    ].to_numpy(dtype=float)

    phone_x = smartphone_df[
        "linear_accel_x_mps2"
    ].to_numpy(dtype=float)

    phone_y = smartphone_df[
        "linear_accel_y_mps2"
    ].to_numpy(dtype=float)

    longitudinal_accel = (
        accel_beta_x * phone_x
        + accel_beta_y * phone_y
        + accel_bias
    )

    gyro = smartphone_df[
        "gyroscope_yaw_radps"
    ].to_numpy(dtype=float)

    dt = np.diff(
        time,
        prepend=time[0],
    )

    dt = np.where(
        (dt > 0.0) & (dt < 1.0),
        dt,
        0.0,
    )

    # ---------------------------------------------------------------
    # Heading
    # ---------------------------------------------------------------

    heading_rad = np.zeros(len(time))

    heading_rad[0] = np.radians(
        initial_heading_deg
    )

    for i in range(1, len(time)):
        heading_rad[i] = (
            heading_rad[i - 1]
            + gyro[i - 1] * dt[i]
        )

        heading_rad[i] = _wrap_angle_rad(
            heading_rad[i]
        )

    # ---------------------------------------------------------------
    # Speed
    # ---------------------------------------------------------------

    speed = np.zeros(len(time))

    speed[0] = max(
        0.0,
        float(initial_speed_mps),
    )

    for i in range(1, len(time)):
        speed[i] = (
            speed[i - 1]
            + longitudinal_accel[i - 1] * dt[i]
        )

        # Prevent numerical noise from producing negative speed.
        speed[i] = max(
            0.0,
            speed[i],
        )

    # ---------------------------------------------------------------
    # Position
    # ---------------------------------------------------------------

    east_velocity = (
        speed
        * np.sin(heading_rad)
    )

    north_velocity = (
        speed
        * np.cos(heading_rad)
    )

    x_m = np.zeros(len(time))
    y_m = np.zeros(len(time))

    x_m[0] = initial_x_m
    y_m[0] = initial_y_m

    for i in range(1, len(time)):
        x_m[i] = (
            x_m[i - 1]
            + east_velocity[i - 1] * dt[i]
        )

        y_m[i] = (
            y_m[i - 1]
            + north_velocity[i - 1] * dt[i]
        )

    return pd.DataFrame(
        {
            "time_s": time,
            "speed_mps": speed,
            "heading_deg": np.degrees(
                heading_rad
            ),
            "longitudinal_acceleration_mps2": (
                longitudinal_accel
            ),
            "x_m": x_m,
            "y_m": y_m,
        }
    )


# ---------------------------------------------------------------------------
# Complete blackout experiment
# ---------------------------------------------------------------------------

def run_imu_blackout_experiment(
    smartphone_df: pd.DataFrame,
    vehicle_df: pd.DataFrame,
    start_time_s: float,
    duration_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run a smartphone-IMU GNSS blackout experiment.

    Returns:

        IMU trajectory
        reference trajectory
        calibration information
    """

    _require_columns(
        vehicle_df,
        {
            "relative_time_s",
            "latitude_deg",
            "longitude_deg",
            "velocity_mps",
            "heading_deg",
        },
    )

    from navigation.dead_reckoning import (
        create_reference_trajectory,
    )

    reference = create_reference_trajectory(
        vehicle_df
    )

    # ---------------------------------------------------------------
    # Calibrate smartphone acceleration direction.
    # ---------------------------------------------------------------

    smartphone_calibration = calculate_linear_acceleration(
        smartphone_df
    )

    calibration_duration_s = 20.0

    calibration_start_s = max(
        0.0,
        start_time_s - calibration_duration_s,
    )

    calibration_end_s = start_time_s

    beta_x, beta_y, bias = calibrate_longitudinal_acceleration(
        smartphone_df=smartphone_calibration,
        vehicle_df=vehicle_df,
        calibration_start_s=calibration_start_s,
        calibration_end_s=calibration_end_s,
    )

    # ---------------------------------------------------------------
    # Determine GNSS state at blackout start.
    # ---------------------------------------------------------------

    start_index = (
        reference["time_s"] - start_time_s
    ).abs().idxmin()

    initial_x = float(
        reference.loc[start_index, "x_m"]
    )

    initial_y = float(
        reference.loc[start_index, "y_m"]
    )

    # Find the vehicle state nearest to the requested blackout
    # start. The vehicle and smartphone streams may have slightly
    # different timestamp origins, so use the actual vehicle stream
    # independently rather than assuming identical row positions.

    vehicle_time = vehicle_df["relative_time_s"].to_numpy(
        dtype=float
    )

    vehicle_start_index = int(
        np.argmin(
            np.abs(vehicle_time - start_time_s)
        )
    )

    vehicle_start = vehicle_df.iloc[
        vehicle_start_index
    ]

    initial_speed = float(
        vehicle_start["velocity_mps"]
    )

    initial_heading = float(
        vehicle_start["heading_deg"]
    )

    # ---------------------------------------------------------------
    # Smartphone blackout.
    # ---------------------------------------------------------------

    smartphone_blackout = extract_blackout(
        smartphone_calibration,
        start_time_s=start_time_s,
        duration_s=duration_s,
    )

    imu_trajectory = run_imu_dead_reckoning(
        smartphone_df=smartphone_blackout,
        initial_x_m=initial_x,
        initial_y_m=initial_y,
        initial_speed_mps=initial_speed,
        initial_heading_deg=initial_heading,
        accel_beta_x=beta_x,
        accel_beta_y=beta_y,
        accel_bias=bias,
    )

    # ---------------------------------------------------------------
    # Reference blackout.
    # ---------------------------------------------------------------

    end_time_s = start_time_s + duration_s

    reference_blackout = reference[
        (reference["time_s"] >= start_time_s)
        & (reference["time_s"] <= end_time_s)
    ].copy()

    reference_blackout.reset_index(
        drop=True,
        inplace=True,
    )

    reference_blackout["time_s"] -= (
        reference_blackout["time_s"].iloc[0]
    )

    calibration = {
        "accel_beta_x": beta_x,
        "accel_beta_y": beta_y,
        "accel_bias": bias,
        "calibration_start_s": calibration_start_s,
        "calibration_end_s": calibration_end_s,
        "initial_speed_mps": initial_speed,
        "initial_heading_deg": initial_heading,
    }

    return (
        imu_trajectory,
        reference_blackout,
        calibration,
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.loader import (
        load_smartphone_dataset,
        load_vehicle_dataset,
    )

    from data.preprocessing import (
        preprocess_smartphone,
        preprocess_vehicle,
    )

    dataset_id = "S1"
    blackout_start = 510.0
    blackout_duration = 60.0

    print("IO-VNBD smartphone IMU dead-reckoning test")
    print()

    print("Loading datasets...")

    smartphone_raw = load_smartphone_dataset(
        dataset_id
    )

    vehicle_raw = load_vehicle_dataset(
        dataset_id
    )

    print("Preprocessing...")

    smartphone = preprocess_smartphone(
        smartphone_raw
    )

    vehicle = preprocess_vehicle(
        vehicle_raw
    )

    print()
    print(
        f"GNSS blackout: "
        f"{blackout_start:.1f}s → "
        f"{blackout_start + blackout_duration:.1f}s"
    )

    print()
    print("Running smartphone IMU dead reckoning...")

    (
        imu,
        reference,
        calibration,
    ) = run_imu_blackout_experiment(
        smartphone_df=smartphone,
        vehicle_df=vehicle,
        start_time_s=blackout_start,
        duration_s=blackout_duration,
    )

    dx = (
        imu["x_m"].iloc[-1]
        - reference["x_m"].iloc[-1]
    )

    dy = (
        imu["y_m"].iloc[-1]
        - reference["y_m"].iloc[-1]
    )

    final_error = float(
        np.sqrt(dx**2 + dy**2)
    )

    print()
    print("=" * 70)
    print("SMARTPHONE IMU RESULT")
    print("=" * 70)

    print()
    print("Calibration:")
    print(
        f"  Calibration window: "
        f"{calibration['calibration_start_s']:.1f}s → "
        f"{calibration['calibration_end_s']:.1f}s"
    )
    print(
        f"  Accel beta X: "
        f"{calibration['accel_beta_x']:.4f}"
    )
    print(
        f"  Accel beta Y: "
        f"{calibration['accel_beta_y']:.4f}"
    )
    print(
        f"  Accel bias: "
        f"{calibration['accel_bias']:.4f} m/s²"
    )

    print()
    print("Initial state:")
    print(
        f"  Speed: "
        f"{calibration['initial_speed_mps']:.3f} m/s"
    )
    print(
        f"  Heading: "
        f"{calibration['initial_heading_deg']:.2f}°"
    )

    print()
    print("Final IMU position:")
    print(
        f"  East:  {imu['x_m'].iloc[-1]:.2f} m"
    )
    print(
        f"  North: {imu['y_m'].iloc[-1]:.2f} m"
    )

    print()
    print("Final reference position:")
    print(
        f"  East:  {reference['x_m'].iloc[-1]:.2f} m"
    )
    print(
        f"  North: {reference['y_m'].iloc[-1]:.2f} m"
    )

    print()
    print(
        f"Final IMU position error: "
        f"{final_error:.2f} m"
    )

    print()
    print("First five IMU points:")

    print(
        imu[
            [
                "time_s",
                "speed_mps",
                "heading_deg",
                "x_m",
                "y_m",
            ]
        ]
        .head()
        .to_string(index=False)
    )

    print()
    print(
        "Smartphone IMU test completed successfully."
    )