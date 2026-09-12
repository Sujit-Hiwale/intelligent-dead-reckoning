"""
IO-VNBD preprocessing utilities.

This module converts raw IO-VNBD data into a clean and consistent
representation for the positioning pipeline.

Responsibilities:
    - Standardize column names.
    - Parse timestamps.
    - Convert relevant measurements to SI units.
    - Convert GPS coordinates to numeric values.
    - Convert satellite-count fields to numeric counts.
    - Add a relative time column.
    - Validate basic data quality.

This module does NOT:
    - Modify the original CSV files.
    - Perform interpolation.
    - Synchronize independently collected datasets.
    - Perform dead reckoning.
    - Perform machine-learning correction.

The raw measurements are retained wherever possible.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config import (
    DEG_TO_RAD,
    GRAVITY_MPS2,
    KMH_TO_MPS,
    RAD_TO_DEG,
)


# ---------------------------------------------------------------------------
# Column-name normalization
# ---------------------------------------------------------------------------

def _normalize_column_name(column: str) -> str:
    """
    Convert a raw IO-VNBD column name into a predictable format.

    Examples:
        "GPS LATITUDE (degrees)"
            -> "gps_latitude_deg"

        "GPS SPEED (Kmh)"
            -> "gps_speed_mps"

    The actual unit conversion is handled separately.
    """
    name = str(column).strip()

    # Repair common mojibake produced by decoding the original headers.
    replacements = {
        "Â°": "°",
        "Î¼": "μ",
        "Âµ": "μ",
    }

    for bad, good in replacements.items():
        name = name.replace(bad, good)

    # Explicit mappings for columns used by the pipeline.
    explicit = {
        "GPS LATITUDE (degrees)": "gps_latitude_deg",
        "GPS LONGITUDE (degrees)": "gps_longitude_deg",
        "GPS ALTITUDE (m)": "gps_altitude_m",
        "GPS SPEED (Kmh)": "gps_speed_mps",
        "GPS ACCURACY (m)": "gps_accuracy_m",
        "GPS ORIENTATION (°)": "gps_orientation_deg",
        "GPS SATELLITES IN RANGE": "gps_satellites",
        "TIME SINCE START (ms)": "time_since_start_s",
        "DATE (YYYY-MO-DD HH-MI-SS_SSS)": "timestamp",
        "ACCELEROMETER X (m/s²)": "accelerometer_x_mps2",
        "ACCELEROMETER Y (m/s²)": "accelerometer_y_mps2",
        "ACCELEROMETER Z (m/s²)": "accelerometer_z_mps2",
        "GRAVITY X (m/s²)": "gravity_x_mps2",
        "GRAVITY Y (m/s²)": "gravity_y_mps2",
        "GRAVITY Z (m/s²)": "gravity_z_mps2",
        "GYROSCOPE Yaw (rad/s)": "gyroscope_yaw_radps",
        "GYROSCOPE Pitch (rad/s)": "gyroscope_pitch_radps",
        "GYROSCOPE Roll (rad/s)": "gyroscope_roll_radps",
        "MAGNETIC FIELD X (μT)": "magnetic_x_ut",
        "MAGNETIC FIELD Y (μT)": "magnetic_y_ut",
        "MAGNETIC FIELD Z (μT)": "magnetic_z_ut",
        "ORIENTATION (Yaw) (°)": "orientation_yaw_deg",
        "ORIENTATION (Pitch) (°)": "orientation_pitch_deg",
        "ORIENTATION (Roll ) (°)": "orientation_roll_deg",
    }

    if name in explicit:
        return explicit[name]

    # Vehicle dataset mappings.
    explicit_vehicle = {
        "No of GPS Satellites Available": "gps_satellites",
        "Time Since Start of Day (seconds)": "time_since_start_day_s",
        "Latitude (degrees)": "latitude_deg",
        "Longitude (degrees)": "longitude_deg",
        "Velocity (km/hr)": "velocity_mps",
        "Heading (degrees)": "heading_deg",
        "Height (km)": "height_m",
        "Vertical velocity (km/hr)": "vertical_velocity_mps",
        "Sample period (seconds)": "sample_period_s",
        "Steering Angle (degrees)": "steering_angle_deg",
        "Wheel Speed Front Left (rad/sec)": "wheel_speed_front_left_radps",
        "Wheel Speed Front Right (rad/sec)": "wheel_speed_front_right_radps",
        "Wheel Speed Rear Left (rad/sec)": "wheel_speed_rear_left_radps",
        "Wheel Speed Rear Right (rad/sec)": "wheel_speed_rear_right_radps",
        "Yaw Rate (deg/sec)": "yaw_rate_radps",
        "Indicated Vehicle Speed (km/hr)": "indicated_vehicle_speed_mps",
        "Indicated Longitudinal Acceleration (g)": (
            "indicated_longitudinal_acceleration_mps2"
        ),
        "Indicated Lateral Acceleration (g)": (
            "indicated_lateral_acceleration_mps2"
        ),
        "Handbrake (0 or 1)": "handbrake",
        "Gear Requested (Number fof gear employed 1-5)": "gear_requested",
        "Gear (Number fof gear employed 1-5)": "gear",
        "Engine Speed (rev/min)": "engine_speed_rpm",
        "Coolant Temperature (degrees)": "coolant_temperature_deg",
        "Clutch Position (0 or 1)": "clutch_position",
        "Brake Pressure (psi)": "brake_pressure_psi",
        "Brake Position (0 or 1)": "brake_position",
        "Battery Voltage (volts)": "battery_voltage_v",
        "Air Temperature (degrees)": "air_temperature_deg",
        "Accelerator Pedal Position (0 or 1)": "accelerator_pedal_position",
    }

    if name in explicit_vehicle:
        return explicit_vehicle[name]

    # Generic fallback for any column not explicitly mapped.
    name = name.lower()
    name = name.replace("²", "2")
    name = name.replace("μ", "u")
    name = name.replace("°", "deg")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")

    return name


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with standardized column names.
    """
    result = df.copy()
    result.columns = [
        _normalize_column_name(column)
        for column in result.columns
    ]
    return result


# ---------------------------------------------------------------------------
# Numeric conversion helpers
# ---------------------------------------------------------------------------

def _to_numeric(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Convert selected columns to numeric values.

    Invalid values become NaN instead of causing the complete
    preprocessing operation to fail.
    """
    result = df.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def _parse_satellite_count(value: object) -> float:
    """
    Convert smartphone satellite strings such as:

        "27 / 28"

    into the number currently in range:

        27.0

    Vehicle satellite counts are already numeric and pass through
    this function unchanged.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    match = re.match(r"^\s*(\d+(?:\.\d+)?)", text)

    if match:
        return float(match.group(1))

    return np.nan


# ---------------------------------------------------------------------------
# Timestamp processing
# ---------------------------------------------------------------------------

def _parse_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the smartphone timestamp column.

    IO-VNBD timestamps use the format:

        YYYY-MM-DD HH:MM:SS:ms

    For example:

        2019-09-08 10:07:49:546

    The final three digits represent milliseconds.
    """
    result = df.copy()

    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            format="%Y-%m-%d %H:%M:%S:%f",
            errors="coerce",
        )

    return result

def _add_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``relative_time_s`` measured from the first sample.

    The source timing columns use different units:

        Smartphone:
            TIME SINCE START (ms)

        Vehicle:
            Time Since Start of Day (seconds)

    The resulting ``relative_time_s`` is always in seconds.
    """
    result = df.copy()

    if "time_since_start_s" in result.columns:
        time_ms = pd.to_numeric(
            result["time_since_start_s"],
            errors="coerce",
        )

        # Smartphone source data are in milliseconds.
        time_s = time_ms / 1000.0

        result["relative_time_s"] = (
            time_s - time_s.iloc[0]
        )

    elif "time_since_start_day_s" in result.columns:
        time_s = pd.to_numeric(
            result["time_since_start_day_s"],
            errors="coerce",
        )

        # Vehicle source data are already in seconds.
        result["relative_time_s"] = (
            time_s - time_s.iloc[0]
        )

    elif "timestamp" in result.columns:
        timestamp = result["timestamp"]

        if timestamp.notna().any():
            first_valid_timestamp = timestamp.dropna().iloc[0]

            result["relative_time_s"] = (
                timestamp - first_valid_timestamp
            ).dt.total_seconds()

    return result
# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def _convert_smartphone_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert smartphone measurements into standardized SI units.

    GPS speed:
        km/h -> m/s

    All accelerometer and gyroscope columns are already represented
    in SI-compatible units by the source dataset.
    """
    result = df.copy()

    if "gps_speed_mps" in result.columns:
        result["gps_speed_mps"] = (
            pd.to_numeric(
                result["gps_speed_mps"],
                errors="coerce",
            )
        )

        # The source column originally contains km/h.
        result["gps_speed_mps"] *= KMH_TO_MPS

    return result


def _convert_vehicle_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert vehicle measurements into standardized SI units.

    Conversions:

        km/h   -> m/s
        km     -> m
        deg/s  -> rad/s
        g      -> m/s²
    """
    result = df.copy()

    speed_columns = [
        "velocity_mps",
        "vertical_velocity_mps",
        "indicated_vehicle_speed_mps",
    ]

    for column in speed_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            ) * KMH_TO_MPS

    if "height_m" in result.columns:
        result["height_m"] = pd.to_numeric(
            result["height_m"],
            errors="coerce",
        ) * 1000.0

    if "yaw_rate_radps" in result.columns:
        result["yaw_rate_radps"] = pd.to_numeric(
            result["yaw_rate_radps"],
            errors="coerce",
        ) * DEG_TO_RAD

    acceleration_columns = [
        "indicated_longitudinal_acceleration_mps2",
        "indicated_lateral_acceleration_mps2",
    ]

    for column in acceleration_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            ) * GRAVITY_MPS2

    return result


# ---------------------------------------------------------------------------
# Public preprocessing functions
# ---------------------------------------------------------------------------

def preprocess_smartphone(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preprocess a raw IO-VNBD smartphone dataframe.
    """
    result = standardize_column_names(df)

    result = _parse_timestamp_column(result)

    numeric_columns = [
        "gps_latitude_deg",
        "gps_longitude_deg",
        "gps_altitude_m",
        "gps_speed_mps",
        "gps_accuracy_m",
        "gps_orientation_deg",
        "time_since_start_s",
        "accelerometer_x_mps2",
        "accelerometer_y_mps2",
        "accelerometer_z_mps2",
        "gravity_x_mps2",
        "gravity_y_mps2",
        "gravity_z_mps2",
        "gyroscope_yaw_radps",
        "gyroscope_pitch_radps",
        "gyroscope_roll_radps",
        "magnetic_x_ut",
        "magnetic_y_ut",
        "magnetic_z_ut",
        "orientation_yaw_deg",
        "orientation_pitch_deg",
        "orientation_roll_deg",
    ]

    result = _to_numeric(result, numeric_columns)

    if "gps_satellites" in result.columns:
        result["gps_satellites"] = result["gps_satellites"].apply(
            _parse_satellite_count
        )

    result = _convert_smartphone_units(result)
    result = _add_relative_time(result)

    return result


def preprocess_vehicle(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preprocess a raw IO-VNBD vehicle dataframe.
    """
    result = standardize_column_names(df)

    numeric_columns = [
        "gps_satellites",
        "time_since_start_day_s",
        "latitude_deg",
        "longitude_deg",
        "velocity_mps",
        "heading_deg",
        "height_m",
        "vertical_velocity_mps",
        "sample_period_s",
        "steering_angle_deg",
        "wheel_speed_front_left_radps",
        "wheel_speed_front_right_radps",
        "wheel_speed_rear_left_radps",
        "wheel_speed_rear_right_radps",
        "yaw_rate_radps",
        "indicated_vehicle_speed_mps",
        "indicated_longitudinal_acceleration_mps2",
        "indicated_lateral_acceleration_mps2",
        "handbrake",
        "gear_requested",
        "gear",
        "engine_speed_rpm",
        "coolant_temperature_deg",
        "clutch_position",
        "brake_pressure_psi",
        "brake_position",
        "battery_voltage_v",
        "air_temperature_deg",
        "accelerator_pedal_position",
    ]

    result = _to_numeric(result, numeric_columns)

    result = _convert_vehicle_units(result)
    result = _add_relative_time(result)

    return result


def preprocess_pair(
    smartphone_df: pd.DataFrame,
    vehicle_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess a synchronized smartphone/vehicle pair.

    No row alignment or interpolation is performed here.
    """
    smartphone = preprocess_smartphone(smartphone_df)
    vehicle = preprocess_vehicle(vehicle_df)

    return smartphone, vehicle


# ---------------------------------------------------------------------------
# Validation / inspection
# ---------------------------------------------------------------------------

def validate_preprocessed_dataset(
    df: pd.DataFrame,
    name: str = "Dataset",
) -> None:
    """
    Perform basic sanity checks and print a concise report.
    """
    print("=" * 70)
    print(f"Preprocessed {name}")
    print("=" * 70)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if "relative_time_s" in df.columns:
        time = df["relative_time_s"].dropna()

        if not time.empty:
            print(f"Duration: {time.iloc[-1]:.3f} seconds")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing.to_string())

    print()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.loader import load_synchronized_pair

    print("IO-VNBD preprocessing test")
    print()
    print("Loading synchronized S1 pair...")
    print()

    smartphone_raw, vehicle_raw = load_synchronized_pair("S1")

    smartphone, vehicle = preprocess_pair(
        smartphone_raw,
        vehicle_raw,
    )

    validate_preprocessed_dataset(
        smartphone,
        "Smartphone S-S1",
    )

    validate_preprocessed_dataset(
        vehicle,
        "Vehicle V-S1",
    )

    print("First smartphone rows:")
    print(smartphone.head().to_string())

    print()
    print("First vehicle rows:")
    print(vehicle.head().to_string())

    print()
    print("Preprocessing test completed successfully.")