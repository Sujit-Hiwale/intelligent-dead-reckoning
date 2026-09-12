from pathlib import Path


# ============================================================
# Project directories
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# IO-VNBD is kept separately from the implementation.
DATASET_ROOT = PROJECT_ROOT.parent / "IO-VNBD"


# ============================================================
# Dataset directories
# ============================================================

# The IO-VNBD repository contains two main dataset collections:
#
# 1. Synchronised V and S datasets
# 2. Unsynchronised V and S Dataset
#
# The spelling below intentionally matches the actual directory
# name in the downloaded dataset repository.

SYNCHRONIZED_DIR = DATASET_ROOT / "Synchronised V abd S datasets"
UNSYNCHRONIZED_DIR = DATASET_ROOT / "Unsynchronised V and S Dataset"


# Inside the synchronized dataset
SYNCHRONIZED_CATEGORIZED_DIR = (
    SYNCHRONIZED_DIR / "Categorised IOVNB Dataset"
)

SYNCHRONIZED_UNCATEGORIZED_DIR = (
    SYNCHRONIZED_DIR / "Uncategorised IOVNB Dataset"
)


# Inside the unsynchronized dataset
UNSYNCHRONIZED_CATEGORIZED_DIR = (
    UNSYNCHRONIZED_DIR / "Categorised IOVNB (V) Dataset"
)

UNSYNCHRONIZED_UNCATEGORIZED_DIR = (
    UNSYNCHRONIZED_DIR / "Uncategorised IOVNB (V and S) Dataset"
)


# ============================================================
# Project output directories
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = PROJECT_ROOT / "figures"
LOG_DIR = PROJECT_ROOT / "logs"


for directory in (
    OUTPUT_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    FIGURE_DIR,
    LOG_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Dataset / sensor configuration
# ============================================================

# IO-VNBD smartphone sensors were sampled at 10 Hz.
SMARTPHONE_SAMPLE_RATE_HZ = 10.0

# Smartphone GPS was updated at 1 Hz.
SMARTPHONE_GPS_RATE_HZ = 1.0


# ============================================================
# Physical constants
# ============================================================

EARTH_RADIUS_METERS = 6_371_000.0

GRAVITY_MPS2 = 9.80665


# ============================================================
# Unit conversion constants
# ============================================================

KMH_TO_MPS = 1000.0 / 3600.0

DEG_TO_RAD = 3.141592653589793 / 180.0

RAD_TO_DEG = 180.0 / 3.141592653589793


# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = 42


# ============================================================
# Validation
# ============================================================

def validate_paths() -> None:
    """
    Validate the project and dataset paths.

    The project can still be imported if the dataset is absent,
    but this function reports which important directories are
    actually available.
    """

    if not PROJECT_ROOT.exists():
        raise RuntimeError(
            f"Project directory does not exist: {PROJECT_ROOT}"
        )


def print_dataset_status() -> None:
    """Print the availability of the main IO-VNBD directories."""

    paths = {
        "Dataset root": DATASET_ROOT,
        "Synchronized dataset": SYNCHRONIZED_DIR,
        "Unsynchronized dataset": UNSYNCHRONIZED_DIR,
        "Synchronized categorized": SYNCHRONIZED_CATEGORIZED_DIR,
        "Synchronized uncategorized": SYNCHRONIZED_UNCATEGORIZED_DIR,
        "Unsynchronized categorized": UNSYNCHRONIZED_CATEGORIZED_DIR,
        "Unsynchronized uncategorized": UNSYNCHRONIZED_UNCATEGORIZED_DIR,
    }

    for name, path in paths.items():
        status = "FOUND" if path.exists() else "NOT FOUND"
        print(f"{name}: {status}")
        print(f"    {path}")


# ============================================================
# Direct execution
# ============================================================

if __name__ == "__main__":
    print("Project root:")
    print(PROJECT_ROOT)

    print("\nDataset root:")
    print(DATASET_ROOT)

    print("\nDataset structure:")
    print_dataset_status()

    print("\nSensor configuration:")
    print(f"  Smartphone sample rate: {SMARTPHONE_SAMPLE_RATE_HZ} Hz")
    print(f"  Smartphone GPS rate:    {SMARTPHONE_GPS_RATE_HZ} Hz")

    validate_paths()

    print("\nConfiguration OK.")