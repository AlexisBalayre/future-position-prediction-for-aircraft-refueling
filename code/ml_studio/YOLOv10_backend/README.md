# YOLOv10 Refuelling Port Detector Backend

## Overview

This directory contains the implementation of a custom machine learning backend using YOLOv10 for detecting the refueling port on aircraft in images. This backend is designed to integrate with Label Studio to automate the labeling process, providing predictions directly in Label Studio's format.

## Folder Structure

```plaintext
YOLOv10_backend/
├── model.py           # Main model implementation file for YOLOv10.
├── README.md          # This README file.
├── Dockerfile         # Dockerfile for containerizing the ML backend.
├── docker-compose.yml # Docker Compose configuration.
└── requirements.txt   # Python dependencies.
```

## Setup and Installation

To use this backend within your project, follow these steps:

1. **Install Dependencies**:
   Ensure that Python is installed on your system. Then, install the required Python packages by running:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment Variables**:
   Create a `.env` file in the `YOLOv10_backend/` directory with the following content:

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
     label-studio-ml start YOLOv10_backend -p 9090
     ```

## Model Implementation

The main model is implemented in `model.py` and extends the `LabelStudioMLBase` class. Key components include:

- **Yolov10RefuellingPortDetector Class**: Implements the YOLOv10 model, including methods for setup, prediction, and formatting results for Label Studio.

### Prediction Workflow

1. **Setup**: The `setup` method initializes the YOLOv10 model when the backend starts.
2. **Performing Predictions**: The `predict` method processes tasks received from Label Studio, loading images and running them through the YOLOv10 model.
3. **Formatting Results**: The `format_results` method formats the model's predictions into a format compatible with Label Studio, including bounding boxes and confidence scores.
