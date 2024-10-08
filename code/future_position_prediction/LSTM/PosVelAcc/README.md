# LSTM/PosVelAcc Directory

## Overview

This directory contains scripts and models for training, tuning, testing, and evaluating the PosVelAcc-LSTM model. The model predicts the future position, velocity, and acceleration of the pressure refuelling port on commercial aircraft. The provided scripts facilitate hyperparameter tuning, training, and evaluation of different model variants, along with utility functions for performance metrics.

## Folder Structure

```plaintext
LSTM/PosVelAcc/
├── Checkpoints/
│   └── ... (model checkpoints saved during training)
├── LSTM/
│   ├── LSTMDecoder.py
│   ├── LSTMEncoder.py
│   ├── SelfAttention.py
├── hptuning.py
├── MetricsMonitoring.py
├── LSTMPosVelAccLightningDataModule.py
├── LSTMPosVelAccLightningDataset.py
├── LSTMPosVelAccLightningModelAverage.py
├── LSTMPosVelAccLightningModelClassic.py
├── LSTMPosVelAccLightningModelConcat.py
├── LSTMPosVelAccLightningModelSum.py
├── testModel.py
├── trainModel.py
└── utils.py
```

### Contents

#### Scripts

- **hptuning.py**

  - This script is used to perform hyperparameter tuning on the PosVelAcc-LSTM model. It explores different configurations by varying parameters such as hidden sizes, learning rates, and dropout rates.
  - **Usage**:

    ```bash
    python -m code.future_position_prediction.LSTM.PosVelAcc.hptuning
    ```

- **trainModel.py**

  - This script is used to train the PosVelAcc-LSTM model using a predefined configuration. It specifies model parameters like input frames, output frames, hidden layers, and more.
  - **Usage**:

    ```bash
    python -m code.future_position_prediction.LSTM.PosVelAcc.trainModel
    ```

- **testModel.py**

  - This script tests the PosVelAcc-LSTM model using trained weights. It evaluates the model's performance on the test dataset and outputs relevant metrics.
  - **Usage**:

    ```bash
    python -m code.future_position_prediction.LSTM.PosVelAcc.testModel
    ```

#### Model Variants

- **LSTMPosVelAccLightningModelSum.py**
  - An LSTM model variant where the hidden states are summed.
- **LSTMPosVelAccLightningModelAverage.py**
  - An LSTM model variant where the hidden states are averaged.
- **LSTMPosVelAccLightningModelConcat.py**
  - An LSTM model variant where the hidden states are concatenated.
- **LSTMPosVelAccLightningModelClassic.py**
  - The classic LSTM model without combining hidden states.

#### Data Handling

- **LSTMPosVelAccLightningDataModule.py**
  - This module manages data loading and preparation for training, validation, and testing phases. It handles batch processing, shuffling, and dataset splitting.

#### Utility Functions

- **utils.py**
  - Contains utility functions for converting velocities to positions, computing Average Displacement Error (ADE), Final Displacement Error (FDE), Intersection over Union (IoU), Average IoU (AIoU), and Final IoU (FIoU).

#### Metrics Monitoring

- **MetricsMonitoring.py**
  - A class for monitoring and computing various metrics during the training and evaluation of trajectory prediction models. It tracks metrics such as ADE, FDE, AIoU, and FIoU.

#### Checkpoints

- **Checkpoints/**
  - Directory where model checkpoints are saved during training. These checkpoints can be used to resume training or for testing purposes.

## Data Requirements

- **Train, Validation, and Test Datasets**:
  - These datasets are required for training, validating, and testing the models. The datasets should be formatted and located as specified in the scripts.

## Getting Started

1. **Hyperparameter Tuning**:
   - Run the `hptuning.py` script to explore different model configurations.
2. **Training the Model**:
   - Use `trainModel.py` to train the LSTM model with your chosen configuration.
3. **Testing the Model**:
   - After training, evaluate the model performance using `testModel.py`.

## Notes

- Ensure that the datasets are properly preprocessed and available at the paths specified in the scripts.
- The results from the hyperparameter tuning are stored in `results_LSTM_PosVelAcc.csv`.
- Adjust the paths within the scripts as necessary to match your local setup.
