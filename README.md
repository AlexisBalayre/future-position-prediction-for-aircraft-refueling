# Future Position Prediction for Aircraft Refuelling Port

## Overview

This repository contains the source code for predicting the future position and size of the pressure refuelling port on commercial aircraft using advanced deep learning techniques. The project integrates object detection models with sequence models to achieve accurate predictions in various scenarios.

## Repository Structure

- **code/**

  - **data_utils/**: Contains scripts for data preprocessing, dataset analysis, and synthetic data generation.

    - `data_preprocessing/`: Includes filters and preprocessing scripts (e.g., EMA, Gaussian, Rolling Mean, Savitzky-Golay).
    - `dataset_analyse.ipynb`: Jupyter notebook for dataset analysis.
    - `dataset_preparation.ipynb`: Jupyter notebook for dataset preparation.
    - `synthetic_data_generation.ipynb`: Jupyter notebook for generating synthetic data.

  - **framework/**: Hosts the core framework components such as evaluation metrics, filter implementations, and model definitions.

    - `evaluation.py`: Code for evaluating model performance.
    - `filters.py`: Contains filtering techniques like EMA, Gaussian, and Savitzky-Golay designed for data post-processing.
    - `object_detection_model.py`: Contains the logic to detect the refuelling port using YOLOv10.
    - `sequence_model.py`: Contains the logic to predict future positions using GRU and LSTM models.

  - **future_position_prediction/**: Contains specific implementations and models for predicting future positions.

    - `Baseline/`: Includes baseline methods like Constant Velocity (CV) and Linear Kalman Filter (LKF).
    - `GRU/`: GRU-based models for position and size prediction, with subdirectories for various configurations.
    - `LSTM/`: LSTM-based models for position and size prediction, similar to the GRU structure.

  - **ml_studio/**: Contains Label Studio backend configurations for automating the annotation process.

    - `FasterRCNN_backend/`: Faster R-CNN configurations.
    - `YOLOv10_backend/`: YOLOv10 configurations.

  - **object_detection/**: Includes the YOLOv10 object detection model configurations and training scripts.

- **data/**: Contains datasets, including annotated frames, YOLO datasets, and synthetic data.

- **report/**: LaTeX files for the thesis report.

- **test/**: Framework output files for testing purposes.

## Installation

1. Create a new virtual environment using Python 3.12.3:

```bash
python3 -m venv venv
```

2. Activate the virtual environment:

```bash
source venv/bin/activate
```

3. Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```
