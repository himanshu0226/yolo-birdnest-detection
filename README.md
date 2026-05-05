# YOLOv8 Birdnest Detection Project

This project contains a complete pipeline to detect birdnests in images using the YOLOv8 model. It includes data validation, model training, prediction, coordinate normalization, and a set of FastAPI endpoints to serve the model predictions.

## Project Structure

- `yolov8_birdnest_detection.py`: Main script for training the YOLOv8 model, running inference, and evaluating accuracy.
- `corrupt_data.py`: Utility script to find and delete corrupted images and their corresponding annotation files.
- `normalize_txt.py`: Utility script to normalize YOLO `.txt` bounding boxes to the `[0, 1]` format.
- `api.py`: FastAPI application serving the Detection API.
- `orchestrator.py`: Orchestrator API that coordinates requests between a Frontend, Backend, and the Detection APIs.
- `requirements.txt`: List of Python dependencies.

## Prerequisites

1. Install Python 3.9+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Utilities

### Checking for Corrupt Data
Use this script to remove corrupted images and labels before training:
```bash
python corrupt_data.py --input_dir path/to/raw_data --output_dir path/to/clean_data
```

### Normalizing Bounding Boxes
Ensure your bounding boxes are normalized:
```bash
python normalize_txt.py --input_dir path/to/labels --output_dir path/to/normalized_labels --image_width 640 --image_height 640
```

## Training and Inference

To train the YOLOv8 model and run predictions on the test dataset:

```bash
python yolov8_birdnest_detection.py
```
This script will automatically load the dataset from `dataset.yaml`, validate files, train the model, evaluate metrics (mAP, Precision, Recall), run predictions, and save the results. The outputs are saved in timestamped directories (e.g. `Final_Train_Predict_Output_YYYYMMDD_HHMMSS`).

## Running the APIs

### 1. Detection API
Starts the core prediction service on port `8001`.
```bash
uvicorn api:app --host 0.0.0.0 --port 8001
```

### 2. Orchestrator API
Starts the orchestrator on port `8002`. Ensure that the frontend and backend APIs are running as defined in `orchestrator.py`.
```bash
uvicorn orchestrator:app --host 0.0.0.0 --port 8002
```

## Accessing the APIs
After starting the servers, you can access the interactive Swagger documentation:
- Detection API: `http://localhost:8001/docs`
- Orchestrator API: `http://localhost:8002/docs`
