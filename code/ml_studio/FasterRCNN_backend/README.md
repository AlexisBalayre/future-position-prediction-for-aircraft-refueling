# FasterRCNN Backend for Aircraft Refuelling Port Detection

## Overview

This directory contains the implementation of a custom machine learning backend for detecting the aircraft refuelling port using a Faster R-CNN model. This backend is designed to be integrated with Label Studio to automate the labeling process and can be trained, deployed, and used to make predictions on images directly within the project.

## Folder Structure

```plaintext
FasterRCNN_backend/
├── model.py           # Main model implementation file.
├── README.md          # This README file.
├── Dockerfile         # Dockerfile for containerizing the ML backend.
├── docker-compose.yml # Docker Compose configuration.
└── requirements.txt   # Python dependencies.
```

## Setup and Installation

To use this backend within the project, follow these steps:

1. **Install Dependencies**:
   Ensure that Python is installed on your system. Then, install the required Python packages by running:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment Variables**:
   Create a `.env` file in the `FasterRCNN_backend/` directory with the following content:

   ```plaintext
   LABEL_STUDIO_URL=http://localhost:8080
   LABEL_STUDIO_API_KEY=your_label_studio_api_key
   ```

   Replace `http://localhost:8080` with your Label Studio instance URL and `your_label_studio_api_key` with your API key.

3. **Run the Backend**:
   You can run the backend using Docker or directly via Python.

   - **With Docker**:

     ```bash
     docker-compose up
     ```

   - **Without Docker**:
     For debugging or development purposes, run the backend directly:

     ```bash
     label-studio-ml start FasterRCNN_backend -p 9090
     ```

## Model Implementation

The main model is implemented in `model.py` and extends the `LabelStudioMLBase` class. Key components include:

- **CustomDataset Class**: Handles loading images and annotations for training the model.
- **FasterRCNNRefuellingPortDetector Class**: Implements the Faster R-CNN model, including methods for training (`fit`) and making predictions (`predict`).

### Prediction Workflow

1. **Loading Images**: The `predict` method loads images from a given path, converts them to tensors, and feeds them into the Faster R-CNN model.
2. **Making Predictions**: The model predicts bounding boxes and labels for detected objects.
3. **Formatting Predictions**: The results are formatted to be compatible with Label Studio’s expected output format.

### Training Workflow

1. **Fetching Annotations**: The `fit` method retrieves annotated data from Label Studio.
2. **Data Processing**: Annotations are processed into a format suitable for training the Faster R-CNN model.
3. **Model Training**: The model is trained using the annotated data, with configurable parameters such as batch size and learning rate.

## Example Usage

### Predicting on New Images

Once the backend is running, it will listen for incoming prediction requests from Label Studio. It will process the images, detect the refuelling port, and return the results.

### Training the Model

The model can be retrained using labeled data from Label Studio by triggering the `fit` method, which will process the annotations and update the model.
