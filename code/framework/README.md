# Framework Directory

## Overview

The `framework/` directory contains core components and utilities used across the project to support the development, evaluation, and application of machine learning models for predicting the future position and size of the pressure refuelling port on commercial aircraft. This directory includes scripts for model evaluation, filtering techniques, object detection, and sequence modeling.

## Folder Structure

```plaintext
framework/
├── evaluation.py
├── filters.py
├── object_detection_model.py
└── sequence_model.py
```

### Contents

#### Scripts

- **evaluation.py**

  - This script is used to evaluate the performance of a trained GRU model for predicting the future positions and sizes of the refuelling port in aircraft video frames. It performs object detection, future position prediction, and computes various performance metrics.

  - **Key Features**:

    - **Object Detection**: Detects objects in video frames using a pre-trained YOLOv10 model.
    - **Future Position Prediction**: Predicts future positions of detected objects using a trained GRU model.
    - **Smoothing Filters**: Applies various smoothing filters to the predictions.
    - **Linear Kalman Filter (LKF)**: Optionally applies a Linear Kalman Filter to smooth the predictions.
    - **Metrics Calculation**: Computes metrics such as Final Displacement Error (FDE), Average Displacement Error (ADE), Final Intersection over Union (FIoU), and Average Intersection over Union (AIoU).
    - **Results Saving**: Saves the predictions and ground truth to a JSON file.

  - **Usage**:

    - Run the script using the command:

      ```bash
      python -m code.framework.evaluation
      ```

    - **Example Workflow**:
      1. **Object Detection**: The script first detects objects in the input video using YOLOv10.
      2. **Future Position Prediction**: It then predicts the future positions of these objects using a trained GRU model.
      3. **Smoothing and Kalman Filters**: Various smoothing filters and the option of applying a Linear Kalman Filter (LKF) are used to refine the predictions.
      4. **Metrics Calculation**: The script calculates key metrics to evaluate the model's performance.
      5. **Saving Results**: The predictions, along with the ground truth, are saved in a JSON file for further analysis.

  - **Configuration**:
    - Paths to the YOLO model weights, GRU model checkpoint, and hyperparameters file must be set correctly in the script.
    - The script allows configuration of input/output frames, smoothing filters, and the use of LKF.

- **filters.py**

  - Implements various filtering techniques used for preprocessing and smoothing the predicted trajectories.

  - **Main Filters**:

    - `Exponential Moving Average (EMA)`
    - `Gaussian Filter`
    - `Rolling Mean Filter`
    - `Savitzky-Golay Filter`

  - **Usage**:
    - Apply filters to smooth the predictions or data during preprocessing.

- **object_detection_model.py**

  - Defines the object detection model logic, primarily utilizing YOLOv10 for detecting the refuelling port in video frames.

  - **Usage**:
    - Load a pre-trained YOLO model and detect objects in video frames.

- **sequence_model.py**

  - Contains the sequence model implementations for predicting future positions based on past trajectories, using models like GRU or LSTM.

  - **Usage**:
    - Initialize and run sequence models to predict future positions of detected objects.

## Getting Started

### Evaluation

To evaluate the performance of a trained GRU model using the `evaluation.py` script, follow these steps:

1. **Prepare the Environment**:

   - Ensure all dependencies are installed.
   - Set up the necessary model files and data paths in the script.

2. **Run the Script**:

   - Use the following command to run the evaluation:

     ```bash
     python -m code.framework.evaluation
     ```

3. **Review Results**:
   - The script will output various metrics and save the predictions to JSON files.

## Notes

- The `evaluation.py` script is designed to be flexible, allowing the user to experiment with different smoothing filters and the use of a Linear Kalman Filter (LKF).
- Ensure the paths to the models and datasets are correctly configured within the script.
- The script outputs useful metrics that can be used to compare different model configurations or processing techniques.
