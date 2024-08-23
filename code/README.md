# Code Folder

## Overview

The `code/` folder contains all the essential scripts, models, and utilities for data preprocessing, model training, evaluation, and object detection for the project focused on predicting the future position and size of the pressure refuelling port on commercial aircraft. The structure is organised into various subdirectories, each addressing a specific aspect of the overall workflow.

## Folder Structure

```plaintext
code/
│
├── data_utils/
│   ├── data_preprocessing/
│   │   ├── preprocess_data_with_ema_filter.py
│   │   ├── preprocess_data_with_gaussian_filter.py
│   │   ├── preprocess_data_with_rolling_mean_filter.py
│   │   ├── preprocess_data_with_sg_filter.py
│   ├── dataset_analyse.ipynb
│   ├── dataset_preparation.ipynb
│   ├── synthetic_data_generation.ipynb
│
├── framework/
│   ├── evaluation.py
│   ├── filters.py
│   ├── object_detection_model.py
│   ├── sequence_model.py
│
├── future_position_prediction/
│   ├── Baseline/
│   │   ├── CV/
│   │   │   ├── CVLightningDataset.py
│   │   │   ├── predict.py
│   │   ├── LKF/
│   │       ├── LKFLightningDataset.py
│   │       ├── predict.py
│   ├── GRU/
│   │   ├── PosVelAcc/
│   │   │   ├── Checkpoints/
│   │   │   ├── GRUNet/
│   │   │   │   ├── DecoderGRU.py
│   │   │   │   ├── EncoderGRU.py
│   │   │   │   ├── SelfAttention.py
│   │   │   ├── hptuning.py
│   │   │   ├── MetricsMonitoring.py
│   │   │   ├── PosVelAccGRULightningDataModule.py
│   │   │   ├── PosVelAccGRULightningDataset.py
│   │   │   ├── PosVelAccGRULightningModelAverage.py
│   │   │   ├── PosVelAccGRULightningModelClassic.py
│   │   │   ├── PosVelAccGRULightningModelConcat.py
│   │   │   ├── PosVelAccGRULightningModelSum.py
│   │   │   ├── testModel.py
│   │   │   ├── trainModel.py
│   │   │   ├── utils.py
│   │   ├── SizPos/
│   │       ├── Checkpoints/
│   │       ├── GRUNet/
│   │       │   ├── DecoderGRU.py
│   │       │   ├── EncoderGRU.py
│   │       │   ├── SelfAttention.py
│   │       ├── hptuning.py
│   │       ├── MetricsMonitoring.py
│   │       ├── SizPosGRULightningDataModule.py
│   │       ├── SizPosGRULightningDataset.py
│   │       ├── SizPosGRULightningModelAverage.py
│   │       ├── SizPosGRULightningModelClassic.py
│   │       ├── SizPosGRULightningModelConcat.py
│   │       ├── SizPosGRULightningModelSum.py
│   │       ├── testModel.py
│   │       ├── trainModel.py
│   │       ├── utils.py
│   ├── LSTM/
│   │   ├── PosVelAcc/
│   │   │   ├── Checkpoints/
│   │   │   ├── LSTM/
│   │   │   │   ├── LSTMDecoder.py
│   │   │   │   ├── LSTMEncoder.py
│   │   │   │   ├── SelfAttention.py
│   │   │   ├── hptuning.py
│   │   │   ├── MetricsMonitoring.py
│   │   │   ├── LSTMPosVelAccLightningDataModule.py
│   │   │   ├── LSTMPosVelAccLightningDataset.py
│   │   │   ├── LSTMPosVelAccLightningModelAverage.py
│   │   │   ├── LSTMPosVelAccLightningModelClassic.py
│   │   │   ├── LSTMPosVelAccLightningModelConcat.py
│   │   │   ├── LSTMPosVelAccLightningModelSum.py
│   │   │   ├── testModel.py
│   │   │   ├── trainModel.py
│   │   │   ├── utils.py
│   │   ├── SizPos/
│   │       ├── Checkpoints/
│   │       ├── LSTM/
│   │       │   ├── LSTMDecoder.py
│   │       │   ├── LSTMEncoder.py
│   │       │   ├── SelfAttention.py
│   │       ├── hptuning.py
│   │       ├── MetricsMonitoring.py
│   │       ├── LSTMSizPosLightningDataModule.py
│   │       ├── LSTMSizPosLightningDataset.py
│   │       ├── LSTMSizPosLightningModelAverage.py
│   │       ├── LSTMSizPosLightningModelClassic.py
│   │       ├── LSTMSizPosLightningModelConcat.py
│   │       ├── LSTMSizPosLightningModelSum.py
│   │       ├── testModel.py
│   │       ├── trainModel.py
│   │       ├── utils.py
│
├── ml_studio/
│   ├── FasterRCNN_backend/
│   ├── YOLOv10_backend/
│
├── object_detection/
│   ├── YOLOv10/
│       ├── pretrained_base_models/
│       ├── dataset.yaml
│       ├── val.yaml
│       ├── trainer_yolo.py
│       ├── gpu.sub
│       ├── runs/
│           ├── train22/
│           ├── train23/
│           ├── train24/
```

### Subdirectories Overview

- **data_utils/**: Utilities and scripts for data preprocessing and analysis.

  - **data_preprocessing/**: Includes filtering and preprocessing scripts for various data processing techniques.
    - `preprocess_data_with_ema_filter.py`: Applies Exponential Moving Average (EMA) filter.
    - `preprocess_data_with_gaussian_filter.py`: Applies Gaussian filter.
    - `preprocess_data_with_rolling_mean_filter.py`: Applies Rolling Mean filter.
    - `preprocess_data_with_sg_filter.py`: Applies Savitzky-Golay filter.
  - `dataset_analyse.ipynb`: Notebook for analysing datasets.
  - `dataset_preparation.ipynb`: Notebook for preparing datasets.
  - `synthetic_data_generation.ipynb`: Notebook for generating synthetic data.

- **framework/**: Contains core components for model evaluation and implementation.

  - `evaluation.py`: Script for evaluating model performance.
  - `filters.py`: Implements data post-processing filters.
  - `object_detection_model.py`: Defines object detection models using YOLOv10.
  - `sequence_model.py`: Defines sequence models for predicting future positions.

- **future_position_prediction/**: Contains models and scripts for predicting future positions.

  - **Baseline/**: Baseline models using Constant Velocity (CV) and Linear Kalman Filter (LKF).
    - **CV/**: CV-based model scripts.
      - `CVLightningDataset.py`: Dataset class for CV model.
      - `predict.py`: Prediction script for CV model.
    - **LKF/**: LKF-based model scripts.
      - `LKFLightningDataset.py`: Dataset class for LKF model.
      - `predict.py`: Prediction script for LKF model.
  - **GRU/**: GRU-based models with various configurations for predicting position, velocity, and size.
    - **PosVelAcc/**: Models predicting position, velocity, and acceleration.
      - `Checkpoints/`: Directory for saving model checkpoints.
      - `GRUNet/`: GRU network architecture.
        - `DecoderGRU.py`: GRU decoder implementation.
        - `EncoderGRU.py`: GRU encoder implementation.
        - `SelfAttention.py`: Self-attention mechanism.
      - `hptuning.py`: Hyperparameter tuning script.
      - `MetricsMonitoring.py`: Script for monitoring training metrics.
      - `PosVelAccGRULightningDataModule.py`: Data module for GRU models.
      - `PosVelAccGRULightningDataset.py`: Dataset class for GRU models.
      - GRU Model Variants:
        - `PosVelAccGRULightningModelAverage.py`
        - `PosVelAccGRULightningModelClassic.py`
        - `PosVelAccGRULightningModelConcat.py`
        - `PosVelAccGRULightningModelSum.py`
      - `testModel.py`: Model testing script.
      - `trainModel.py`: Model training script.
      - `utils.py`: Utility functions for GRU models.
    - **SizPos/**: Models predicting size and position.
      - Same structure as PosVelAcc.
  - **LSTM/**: LSTM-based models with similar structure to GRU models.
    - **PosVelAcc/**: Models predicting position, velocity, and acceleration.
      - Similar structure to GRU/PosVelAcc.
    - **SizPos/**: Models predicting size and position.
      - Similar structure to GRU/SizPos.

- **ml_studio/**: Configurations for Label Studio backend, supporting Faster R-CNN and YOLOv10.

  - `FasterRCNN_backend/`: Backend configurations for Faster R-CNN.
  - `YOLOv10_backend/`: Backend configurations for YOLOv10.

- **object_detection/**: YOLOv10 model configurations and training scripts.
  - **YOLOv10/**: YOLOv10-specific implementations.
    - `pretrained_base_models/`: Directory for storing pretrained YOLOv10 models.
    - `dataset.yaml`: Configuration for the YOLOv10 dataset.
    - `val.yaml`: Validation configuration file.
    - `trainer_yolo.py`: YOLOv10 training script.
    - `gpu.sub`: Script for running YOLOv10 on GPUs.
    - **runs/**: Directory for training outputs.
      - `train22/`, `train23/`, `train24/`: Output directories for different YOLOv10 training runs.

## Usage

1. **Preprocessing**: Use scripts in the `data_utils/data_preprocessing/` directory to preprocess raw datasets.
2. **Training**: Train GRU or LSTM models using scripts in `future_position_prediction/GRU/` or `future_position_prediction/LSTM/`.
3. **Evaluation**: Evaluate models using `evaluation.py` in the `framework/` directory.
4. **Object Detection**: Train and configure YOLOv10 models using the `object_detection/YOLOv10/` directory.
