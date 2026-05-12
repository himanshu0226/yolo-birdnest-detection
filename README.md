<!-- # YOLOv8 Birdnest Detection Project

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
- Orchestrator API: `http://localhost:8002/docs` -->


# YOLOv8 Bird Nest Detection Project

This project contains a complete pipeline to detect bird nests in images using the YOLOv8 model. It includes:

- Dataset validation
- Corrupted file removal
- Bounding box normalization
- YOLOv8 model training
- Model evaluation
- Prediction and inference
- Detection APIs using FastAPI
- Orchestrator API for workflow integration
- GPS-based unique bird nest detection

---

# Project Structure

```text
.
├── api.py
├── orchestrator.py
├── yolov8_birdnest_detection.py
├── corrupt_data.py
├── normalize_txt.py
├── create_sample_dataset.py
├── requirements.txt
├── README.md
├── dataset.yaml
├── sample_data/
├── saved_models/
├── predictions/
├── prediction_results/
└── Final_Train_Predict_Output_*/
```

---

# Files Description

| File | Description |
|------|-------------|
| `yolov8_birdnest_detection.py` | Main training, evaluation, and prediction pipeline |
| `api.py` | FastAPI Detection API |
| `orchestrator.py` | Orchestrator API connecting frontend, backend, and detection services |
| `corrupt_data.py` | Removes corrupted images and annotation files |
| `normalize_txt.py` | Converts Pascal VOC-style TXT bounding boxes into normalized YOLO format |
| `create_sample_dataset.py` | Creates a smaller sample dataset for testing/debugging |
| `requirements.txt` | Python dependencies |
| `dataset.yaml` | YOLOv8 dataset configuration |

---

# Features

- YOLOv8 bird nest detection
- Automatic dataset validation
- Corrupted image/annotation cleanup
- Bounding box normalization
- Automatic model checkpointing
- Prediction result saving
- FastAPI-based inference service
- Orchestrator microservice
- GPS-based unique bird nest detection
- Hyperparameter logging
- Timestamped output organization

---

# Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional but recommended)

---

# Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
cd <repository-folder>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

Main dependencies include:

- FastAPI
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas
- PyTorch
- Pillow
- Geopy

All dependencies are listed in:

```text
requirements.txt
```

---

# Dataset Structure

Expected YOLO dataset structure:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
```

---

# dataset.yaml Example

```yaml
path: ./dataset

train: images/train
val: images/val
test: images/test

names:
  0: birdnest
```

---

# Utilities

## 1. Remove Corrupted Files

This script removes corrupted images and corresponding annotation files.

### Run

```bash
python corrupt_data.py --input_dir path/to/raw_data --output_dir path/to/clean_data
```

---

# 2. Normalize Bounding Boxes

Converts Pascal VOC-style bounding boxes:

```text
class_id x_min y_min x_max y_max
```

into normalized YOLO format.

### Run

```bash
python normalize_txt.py \
    --input_dir path/to/labels \
    --output_dir path/to/normalized_labels \
    --image_width 640 \
    --image_height 640
```

---

# 3. Create Sample Dataset

Creates a smaller dataset for quick debugging/testing.

### Run

```bash
python create_sample_dataset.py
```

---

# Training the Model

Run the training pipeline:

```bash
python yolov8_birdnest_detection.py
```

The script automatically:

- Validates datasets
- Removes invalid data
- Loads previous weights if available
- Trains YOLOv8
- Evaluates metrics
- Saves predictions
- Saves trained models
- Generates logs and reports

---

# Outputs

Outputs are automatically organized into timestamped folders:

```text
Final_Train_Predict_Output_YYYYMMDD_HHMMSS/
```

Generated outputs include:

- Trained models
- Prediction images
- Prediction TXT results
- Accuracy reports
- Hyperparameter logs
- Timing reports

---

# Model Evaluation Metrics

The training script reports:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

---

# Running the APIs

## 1. Detection API

Runs the YOLOv8 inference API.

### Start API

```bash
uvicorn api:app --host 0.0.0.0 --port 8001
```

### Swagger Documentation

```text
http://localhost:8001/docs
```

### Health Check

```text
http://localhost:8001/health
```

---

# 2. Orchestrator API

Coordinates:

- Frontend API
- Backend API
- Detection API

### Default Service URLs

| Service | Default URL |
|----------|-------------|
| Frontend API | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Detection API | http://localhost:8001 |

### Start Orchestrator

```bash
uvicorn orchestrator:app --host 0.0.0.0 --port 8002
```

### Swagger Documentation

```text
http://localhost:8002/docs
```

---

# API Endpoints

## Detection API

### POST `/predict`

Upload an image and receive bird nest detections.

### Example Request

```bash
curl -X POST "http://localhost:8001/predict" \
     -F "file=@image.jpg"
```

---

# GPU Support

The training script automatically uses:

- CUDA GPU if available
- CPU otherwise

---

# Saved Models

Trained models are saved in:

```text
saved_models/
```

with timestamped filenames.

---

# Logging

The project logs:

- Training metrics
- Prediction results
- Hyperparameters
- Execution time

---

# Notes

- Ensure all labels follow YOLO format.
- Ensure images and labels have matching filenames.
- Use GPU for significantly faster training.
- Use smaller sample datasets for debugging.

---

# Future Improvements

- Docker support
- Kubernetes deployment
- Multi-class detection
- Model quantization
- ONNX/TensorRT optimization
- CI/CD pipeline integration

---

# License

This project is intended for research and development purposes.

---

# Author

Himanshu Prakash

Deep Learning | Computer Vision | AI/ML Engineer