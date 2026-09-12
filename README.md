# Intelligent Dead Reckoning

An intelligent dead-reckoning and positioning system for maintaining vehicle position estimates during temporary GNSS/GPS outages.

## Overview

Global Navigation Satellite System (GNSS) signals can become unreliable or unavailable in environments such as tunnels, underground roads, urban areas, and other signal-degraded environments.

This project focuses on estimating a vehicle's position during such GNSS-denied periods using vehicle motion and inertial information, with the long-term goal of developing an intelligent sensor-fusion system that can reduce the positional drift of conventional dead reckoning.

The project uses the **IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark Dataset)** as the primary dataset for development and evaluation.

## Objectives

The project aims to:

* Estimate vehicle position during temporary GNSS outages.
* Implement and evaluate dead-reckoning techniques using vehicle motion information.
* Process inertial and navigation data from both vehicle and smartphone sensors.
* Quantify positional drift during GNSS-denied intervals.
* Develop an intelligent sensor-fusion approach for improved positioning.
* Evaluate positioning accuracy against a reference trajectory.
* Investigate the effect of synchronization and sensor uncertainty on positioning performance.

## Dataset — IO-VNBD

This project uses the **IO-VNBD: Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning**.

The dataset was developed as a benchmark for vehicle positioning and inertial-navigation research. It contains vehicle-side measurements collected from a research vehicle as well as measurements recorded using Android smartphones. The vehicle data include GPS, inertial/navigation information, wheel-speed measurements and other vehicle parameters, while the smartphone data include inertial sensors and GPS measurements sampled at 10 Hz, with smartphone GPS updates at approximately 1 Hz.

The dataset contains diverse driving conditions and vehicle dynamics, including traffic, roundabouts, hard braking, different road types, and different driving patterns.

### Dataset Repository

The original IO-VNBD dataset is available from the authors' repository:

[IO-VNBD — GitHub repository](https://github.com/onyekpeu/IO-VNBD?utm_source=chatgpt.com)

### Dataset Publication

> U. Onyekpe, V. Palade, S. Kanarachos and A. Szkolnik, "IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning", *Data in Brief*, Volume 35, 2021, Article 106885.

[IO-VNBD publication — ScienceDirect](https://doi.org/10.1016/j.dib.2021.106885?utm_source=chatgpt.com)

The dataset is provided in CSV format and contains GPS coordinates for the recorded samples. The authors provide both vehicle (`V-`) and smartphone (`S-`) datasets, with synchronized V/S datasets available for recordings that were collected simultaneously and subsequently synchronized.

## How IO-VNBD Is Used

The dataset serves two main purposes in this project:

### 1. Motion and Sensor Input

The recorded vehicle and smartphone measurements provide the sensor inputs required to develop positioning algorithms.

For the current dead-reckoning implementation, the vehicle-side dataset provides:

* Vehicle velocity
* Vehicle heading
* Sampling/time information

These measurements are used to propagate the vehicle's position forward in time without continuously relying on GNSS position updates.

The smartphone portion of IO-VNBD provides additional inertial measurements such as:

* Accelerometer
* Gyroscope
* Magnetometer
* Smartphone orientation
* Smartphone GPS

These measurements are used for the development and evaluation of the inertial positioning component.

### 2. Reference Trajectory

The recorded GPS coordinates are used to construct a reference vehicle trajectory for evaluation.

Latitude and longitude coordinates are converted into a local coordinate system representing East/North displacement. The dead-reckoned trajectory can then be compared against this reference trajectory to determine how much positional error accumulates during a simulated GNSS outage.

Conceptually:

```text
IO-VNBD Dataset
       │
       ├───────────────┐
       │               │
       ▼               ▼
 Vehicle Data      Smartphone Data
       │               │
       ▼               ▼
 Velocity/Heading    IMU/GPS
       │               │
       ▼               ▼
 Vehicle DR        IMU-based DR
       │               │
       └───────┬───────┘
               │
               ▼
        Position Estimate
               │
               ▼
     Compare with Reference
               │
               ▼
       Position Error
       & Drift Metrics
```

## Dead-Reckoning Approach

The current baseline dead-reckoning implementation estimates displacement from vehicle velocity and heading.

For each time interval:

* Vehicle velocity determines the magnitude of displacement.
* Vehicle heading determines the direction of movement.
* The time difference between samples determines the integration interval.
* Eastward and northward displacement are accumulated to obtain the estimated trajectory.

The geographic GPS trajectory is transformed into a local East/North coordinate frame so that the estimated and reference trajectories can be compared directly in metres.

## GNSS-Outage Evaluation

GNSS-denied operation is simulated by selecting a time interval from the recorded trajectory and evaluating the dead-reckoning estimate over that interval.

The reference GPS trajectory remains available **only for evaluation**. It is not used as a position correction during the simulated outage.

The evaluation measures include:

* Total distance travelled
* Final position error
* Maximum position error
* Mean position error
* Positional drift percentage
* Position update rate

The drift percentage is calculated relative to the distance travelled during the evaluated interval.

## Initial Evaluation

An initial evaluation using the synchronized IO-VNBD dataset produced approximately **1.2% positional drift** for the vehicle-data dead-reckoning baseline over the evaluated GNSS-denied interval.

This result provides an initial indication that the underlying dead-reckoning approach can maintain a relatively accurate trajectory over a temporary GNSS outage when reliable vehicle motion measurements are available.

The result is specifically associated with the **vehicle-data dead-reckoning baseline** and should not be interpreted as the final performance of the smartphone inertial or intelligent sensor-fusion system.

## Project Structure

```text
intelligent-dead-reckoning/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── config.py
│
├── data/
│   ├── __init__.py
│   ├── loader.py
│   └── preprocessing.py
│
├── navigation/
│   ├── __init__.py
│   └── dead_reckoning.py
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
│
└── dashboard/
    ├── __init__.py
    └── app.py
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd intelligent-dead-reckoning
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

The IO-VNBD dataset is not included in this repository.

Download the dataset from the original repository:

[Download IO-VNBD](https://github.com/onyekpeu/IO-VNBD?utm_source=chatgpt.com)

The project expects the dataset to be available separately from the source code.

The dataset directory should contain the synchronized and/or unsynchronized IO-VNBD data directories supplied by the dataset authors.

## Running the Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

The dashboard provides visualization of the positioning experiment, trajectory, vehicle motion and evaluation metrics.

## Running the Dead-Reckoning Evaluation

The dead-reckoning implementation can be executed from the project root using:

```bash
python -m navigation.dead_reckoning
```

The evaluation module can be run using:

```bash
python -m evaluation.metrics
```

## Development Direction

The current implementation establishes the data-processing, trajectory-generation and evaluation pipeline required for GNSS-denied positioning.

Further development is focused on:

* Smartphone IMU-based position estimation.
* Robust smartphone-to-vehicle coordinate transformation.
* Inertial sensor calibration.
* GNSS/IMU sensor fusion.
* Machine-learning-based drift estimation and correction.
* Evaluation using synchronized and unsynchronized data.
* Improved robustness to sensor noise, timing differences and changing vehicle dynamics.
* Validation under longer GNSS-denied intervals and more diverse driving conditions.

## References

**IO-VNBD Dataset**

Onyekpe, U., Palade, V., Kanarachos, S., & Szkolnik, A. (2021).
*IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning.*
Data in Brief, 35, 106885.

[Dataset repository](https://github.com/onyekpeu/IO-VNBD?utm_source=chatgpt.com)
[Research paper](https://doi.org/10.1016/j.dib.2021.106885?utm_source=chatgpt.com)

## License

This project is intended for research and development purposes.

The IO-VNBD dataset is maintained and distributed by its original authors. Please refer to the original dataset repository and publication for dataset licensing, attribution and usage information.