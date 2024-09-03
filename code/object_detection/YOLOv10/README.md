# YOLOv10 Object Detection for Aircraft Refuelling Port

## Overview

This directory contains the implementation and configuration for using YOLOv10 to detect the refuelling port on aircraft in video frames. The setup includes pre-trained model weights, dataset configurations, training scripts, and a submission script for running on a HPC cluster.

## Folder Structure

```plaintext
YOLOv10/
├── pretrained_base_models/  # Pre-trained YOLOv10 model weights.
│   └── yolov10s.pt          # YOLOv10 small model pre-trained weights.
├── dataset.yaml             # Dataset configuration file for training.
├── val.yaml                 # Validation dataset configuration file.
├── trainer_yolo.py          # Script to train the YOLOv10 model.
├── gpu.sub                  # SLURM submission script for running on a GPU cluster.
└── runs/                    # Directory to store training weights, results and logs.
    ├── train22/             # Training run directory for YOLOv10 N model.
    ├── train23/             # Training run directory for YOLOv10 M model.
    └── train24/             # Training run directory for YOLOv10 S model.
```

## Setup and Installation

Before running any scripts in this directory, ensure that the required Python packages are installed. This typically includes the `ultralytics` package, which contains the YOLOv10 implementation.

1. **Install Dependencies**:
   Run the following command to install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **Pre-trained Models**:
   The `pretrained_base_models/` directory should contain the pre-trained YOLOv10 weights (`yolov10s.pt`), which can be used as a starting point for fine-tuning on your specific dataset.

## Usage

### 1. Training the Model

To train the YOLOv10 model on your dataset, use the `trainer_yolo.py` script. This script loads the dataset specified in `dataset.yaml`, initializes the YOLOv10 model, and begins training.

- **trainer_yolo.py**:

  - **Model Name**: Specifies the pre-trained YOLOv10 model to be used for training.
  - **Dataset**: Specifies the dataset configuration file `dataset.yaml`.
  - **Validation**: Specifies the validation dataset configuration file `val.yaml`.
  - **Training Configuration**: Sets batch size, image size, number of epochs, and verbosity.

  **Example Command**:

  ```bash
  python trainer_yolo.py
  ```

### 2. Validation

After training, the model can be validated using the validation dataset configuration specified in `val.yaml`.

- **val.yaml**:

  - Specifies the dataset paths and class labels for validation.
  - **Example Configuration**:

    ```yaml
    names:
      0: Fuel Port [CLOSED]
      1: Fuel Port [SEMI-OPEN]
      2: Fuel Port [OPEN]
    nc: 3
    train: data/AARP/frames/full_dataset_annotated_YOLO/balanced_dataset/train
    val: data/AARP/frames/full_dataset_annotated_YOLO/balanced_dataset/test
    ```

### 3. Running on a GPU Cluster

The `gpu.sub` script is provided for running training or inference on HPC clusters.

- **gpu.sub**:

  - **Job Name**: Set the job name for identification in the queue.
  - **Resource Allocation**: Specifies the number of CPUs, GPUs, and memory to be used.
  - **Environment Setup**: Loads the necessary modules and activates the Conda environment.
  - **Execution**: Runs the `trainer_yolo.py` script.

  **Example Submission**:

  ```bash
  qsub gpu.sub
  ```

### 4. Results and Outputs

All results, including trained models, logs, and performance metrics, will be stored in the `runs/` directory. Each subdirectory within `runs/` corresponds to a different training run, organized by date and model configuration.

## Notes

- **Customizing Training**: The `trainer_yolo.py` script and the dataset configuration files (`dataset.yaml`, `val.yaml`) should be customized based on the specifics of your dataset and desired model settings.
- **Pre-trained Models**: If you want to fine-tune a pre-trained model, ensure the correct weights are located in the `pretrained_base_models/` directory.
