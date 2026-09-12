"""
IO-VNBD dataset loader.

This module is responsible for locating and loading the raw
IO-VNBD CSV files without modifying the original dataset.

The loader supports:
    - Smartphone (S-) datasets
    - Vehicle (V-) datasets
    - Synchronized V/S dataset pairs

Important:
    The raw IO-VNBD files are intentionally kept unchanged.
    Cleaning, unit conversion, interpolation, and sensor fusion
    belong to later stages of the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from config import (
    SYNCHRONIZED_CATEGORIZED_DIR,
    SYNCHRONIZED_UNCATEGORIZED_DIR,
    UNSYNCHRONIZED_CATEGORIZED_DIR,
    UNSYNCHRONIZED_UNCATEGORIZED_DIR,
)


DatasetType = Literal["smartphone", "vehicle"]


# ============================================================
# Internal helpers
# ============================================================

def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove accidental whitespace from CSV column names.

    This does NOT rename columns or alter their meaning.
    """

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def _read_csv(path: Path) -> pd.DataFrame:
    """
    Read an IO-VNBD CSV file.

    The original files contain non-UTF-8 characters such as:
        - ²
        - μ
        - degree symbols

    latin-1 is deliberately used here because it can decode every
    possible byte value and therefore prevents the parser from
    failing on these dataset headers.

    We keep the raw values unchanged at this stage.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file does not exist:\n{path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, received:\n{path}"
        )

    try:
        df = pd.read_csv(
            path,
            encoding="latin-1",
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"CSV file is empty:\n{path}"
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Could not parse CSV file:\n{path}"
        ) from exc

    return _clean_column_names(df)


def _dataset_search_roots(
    include_unsynchronized: bool = False,
) -> list[Path]:
    """
    Return directories in which IO-VNBD datasets may be found.

    Synchronized data are searched first because they are the
    primary paired data source for our project.
    """

    roots = [
        SYNCHRONIZED_CATEGORIZED_DIR,
        SYNCHRONIZED_UNCATEGORIZED_DIR,
    ]

    if include_unsynchronized:
        roots.extend(
            [
                UNSYNCHRONIZED_CATEGORIZED_DIR,
                UNSYNCHRONIZED_UNCATEGORIZED_DIR,
            ]
        )

    return [
        root
        for root in roots
        if root.exists()
    ]


def _find_dataset_file(
    dataset_id: str,
    dataset_type: DatasetType,
    include_unsynchronized: bool = False,
) -> Path:
    """
    Locate one IO-VNBD CSV file by dataset identifier.

    Examples:
        dataset_id="S1", dataset_type="smartphone"
            -> S-S1.csv

        dataset_id="S1", dataset_type="vehicle"
            -> V-S1.csv

        dataset_id="Vta29", dataset_type="smartphone"
            -> S-Vta29.csv

    The search is case-insensitive because the repository contains
    some inconsistent filename capitalization such as V-vta2.csv.
    """

    dataset_id = dataset_id.strip()

    if not dataset_id:
        raise ValueError("dataset_id cannot be empty.")

    if dataset_type == "smartphone":
        prefix = "s-"
    elif dataset_type == "vehicle":
        prefix = "v-"
    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type!r}"
        )

    expected_name = f"{prefix}{dataset_id}.csv".lower()

    matches: list[Path] = []

    for root in _dataset_search_roots(
        include_unsynchronized=include_unsynchronized
    ):
        for path in root.rglob("*.csv"):
            if path.name.lower() == expected_name:
                matches.append(path)

    if not matches:
        searched_roots = "\n".join(
            f"  - {root}"
            for root in _dataset_search_roots(
                include_unsynchronized=include_unsynchronized
            )
        )

        raise FileNotFoundError(
            f"Could not find {dataset_type} dataset "
            f"for ID {dataset_id!r}.\n\n"
            f"Searched in:\n{searched_roots}"
        )

    # Prefer the most useful dataset location when duplicates exist.
    #
    # Priority:
    #   1. Synchronized categorized
    #   2. Synchronized uncategorized
    #   3. Unsynchronized categorized
    #   4. Unsynchronized uncategorized
    priority_roots = [
        SYNCHRONIZED_CATEGORIZED_DIR,
        SYNCHRONIZED_UNCATEGORIZED_DIR,
        UNSYNCHRONIZED_CATEGORIZED_DIR,
        UNSYNCHRONIZED_UNCATEGORIZED_DIR,
    ]

    for root in priority_roots:
        preferred_matches = [
            path
            for path in matches
            if root in path.parents
        ]

        if preferred_matches:
            if len(preferred_matches) > 1:
                locations = "\n".join(
                    f"  - {path}" for path in preferred_matches
                )
                raise RuntimeError(
                    f"Multiple matching {dataset_type} datasets were found "
                    f"for ID {dataset_id!r} inside:\n"
                    f"{root}\n\n"
                    f"{locations}"
                )

            matches = preferred_matches
            break

    return matches[0]


# ============================================================
# Public loading functions
# ============================================================

def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load any IO-VNBD CSV file directly.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Raw dataset contents with whitespace removed from column
        names.
    """

    return _read_csv(Path(path))


def load_smartphone_dataset(
    dataset_id: str,
    include_unsynchronized: bool = False,
) -> pd.DataFrame:
    """
    Load an IO-VNBD smartphone dataset.

    Examples
    --------
    load_smartphone_dataset("S1")
    load_smartphone_dataset("Vta29")
    load_smartphone_dataset("Vw4")
    """

    path = _find_dataset_file(
        dataset_id=dataset_id,
        dataset_type="smartphone",
        include_unsynchronized=include_unsynchronized,
    )

    return _read_csv(path)


def load_vehicle_dataset(
    dataset_id: str,
    include_unsynchronized: bool = False,
) -> pd.DataFrame:
    """
    Load an IO-VNBD vehicle dataset.

    Examples
    --------
    load_vehicle_dataset("S1")
    load_vehicle_dataset("Vta29")
    load_vehicle_dataset("Vw4")
    """

    path = _find_dataset_file(
        dataset_id=dataset_id,
        dataset_type="vehicle",
        include_unsynchronized=include_unsynchronized,
    )

    return _read_csv(path)


def load_synchronized_pair(
    dataset_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the synchronized smartphone + vehicle pair.

    Returns
    -------
    smartphone_df, vehicle_df

    Both DataFrames correspond to the same IO-VNBD dataset ID.

    Example
    -------
    smartphone_df, vehicle_df = load_synchronized_pair("S1")
    """

    smartphone_path = _find_dataset_file(
        dataset_id=dataset_id,
        dataset_type="smartphone",
        include_unsynchronized=False,
    )

    vehicle_path = _find_dataset_file(
        dataset_id=dataset_id,
        dataset_type="vehicle",
        include_unsynchronized=False,
    )

    smartphone_df = _read_csv(smartphone_path)
    vehicle_df = _read_csv(vehicle_path)

    return smartphone_df, vehicle_df


# ============================================================
# Dataset information helpers
# ============================================================

def get_dataset_path(
    dataset_id: str,
    dataset_type: DatasetType,
    include_unsynchronized: bool = False,
) -> Path:
    """
    Return the actual path of an IO-VNBD CSV without loading it.
    """

    return _find_dataset_file(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        include_unsynchronized=include_unsynchronized,
    )


def describe_dataset(
    df: pd.DataFrame,
    name: str = "Dataset",
) -> None:
    """
    Print a compact description of a loaded dataset.

    This is intended for inspection/debugging rather than the
    actual processing pipeline.
    """

    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")

    for index, column in enumerate(df.columns):
        print(f"{index:2d}: {column}")

    print("\nMissing values:")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing.to_string())

    print()


# ============================================================
# Basic command-line test
# ============================================================

if __name__ == "__main__":
    print("IO-VNBD loader test")
    print()

    print("Loading synchronized S1 pair...")
    print()

    smartphone, vehicle = load_synchronized_pair("S1")

    describe_dataset(
        smartphone,
        "Smartphone S-S1",
    )

    describe_dataset(
        vehicle,
        "Vehicle V-S1",
    )

    print("Dataset locations:")
    print(
        "Smartphone:",
        get_dataset_path(
            "S1",
            "smartphone",
        ),
    )
    print(
        "Vehicle:",
        get_dataset_path(
            "S1",
            "vehicle",
        ),
    )

    print("\nLoader test completed successfully.")