# Future Position Prediction for Aircraft Refuelling Port

## Overview

This repository contains the source code for predicting the future position and size of the pressure refuelling port on commercial aircraft using advanced deep learning techniques. The project integrates object detection models with sequence models to achieve accurate predictions in various scenarios.

## Repository Structure

```plaintext
.
├── code/
│   ├── data_utils/
│   │   ├── data_preprocessing/
│   │   ├── dataset_analyse.ipynb
│   │   ├── dataset_preparation.ipynb
│   │   ├── synthetic_data_generation.ipynb
│   ├── framework/
│   │   ├── evaluation.py
│   │   ├── filters.py
│   │   ├── object_detection_model.py
│   │   ├── sequence_model.py
│   ├── future_position_prediction/
│   │   ├── GRU/
│   │   ├── LSTM/
│   ├── ml_studio/
│   ├── object_detection/
├── data/
│   ├── AARP/
│   ├── synthetic/
├── report/
├── test/
├── requirements.txt
```

### Summary of Main Directories

- **code/**

  - **data_utils/**: Scripts for data preprocessing, analysis, and synthetic data generation.
    - `data_preprocessing/`: Filtering and preprocessing scripts (e.g., EMA, Gaussian, Rolling Mean, Savitzky-Golay).
    - `dataset_analyse.ipynb`: Dataset analysis notebook.
    - `dataset_preparation.ipynb`: Dataset preparation notebook.
    - `synthetic_data_generation.ipynb`: Synthetic data generation notebook.
  - **framework/**: Core components such as evaluation metrics, filter implementations, and model definitions.
    - `evaluation.py`: Model performance evaluation.
    - `filters.py`: Data post-processing filters.
    - `object_detection_model.py`: Refuelling port detection logic using YOLOv10.
    - `sequence_model.py`: Future position prediction logic using GRU and LSTM models.
  - **future_position_prediction/**: Implementations for predicting future positions.
    - `GRU/`: GRU-based prediction models.
    - `LSTM/`: LSTM-based prediction models.
  - **ml_studio/**: Backend configurations for automating annotation processes.
  - **object_detection/**: YOLOv10 model configurations and training scripts.

- **report/**: LaTeX files for the thesis report.

- **test/**: Output files from framework testing.

## Getting Started

### Prerequisites

- Python 3.12.3
- PyTorch Lightning
- OpenCV
- Label Studio
- Docker
- YOLOv10

### Installation

1. Clone the repository:

```bash
git clone https://github.com/AlexisBalayre/future-position-prediction-for-aircraft-refueling
cd future-position-prediction-for-aircraft-refueling
```

2. Create a new virtual environment using Python 3.12.3:

```bash
python3 -m venv venv
```

3. Activate the virtual environment:

```bash
source venv/bin/activate
```

4. Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

## License

This project is licensed under the terms of the Apache V2 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to my supervisors, Dr. Boyu Kuang and Dr. Stuart Barnes, and sponsors Airbus, UKRI, and ATI for their support in this research through the ONEheart project.
