import os

# Enable CUDA debugging BEFORE importing torch
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import cv2
import yaml
import time
import logging
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import shutil
import torch

from geopy.distance import geodesic
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)


# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def validate_path(path, path_type="directory", description="Path", tracked_paths=None):

    if path_type == "directory":

        if not os.path.isdir(path):
            print(f"{description} not found. Creating: {path}")
            ensure_directory_exists(path)

        if tracked_paths is not None:
            tracked_paths.add(path)

    elif path_type == "file":

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{description} is invalid. File not found: {path}"
            )

    print(f"Validated {description}: {path}")


def validate_labels(label_dir, num_classes):

    for label_file in os.listdir(label_dir):

        path = os.path.join(label_dir, label_file)

        with open(path, 'r') as f:

            for line_num, line in enumerate(f.readlines(), 1):

                parts = line.strip().split()

                if len(parts) != 5:
                    print(
                        f"Malformed label in {label_file} at line {line_num}: {line}"
                    )
                    continue

                class_id = int(parts[0])

                if class_id < 0 or class_id >= num_classes:
                    print(
                        f"Invalid class ID in {label_file} at line {line_num}: {class_id}"
                    )

                bbox = list(map(float, parts[1:]))

                if not all(0 <= coord <= 1 for coord in bbox):
                    print(
                        f"Invalid bounding box in {label_file} at line {line_num}: {bbox}"
                    )


def validate_images(image_dir):

    for img_file in os.listdir(image_dir):

        try:
            img_path = os.path.join(image_dir, img_file)

            img = Image.open(img_path)

            img.verify()

        except Exception as e:
            print(f"Invalid image file: {img_file}, Error: {e}")


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = "."

IMAGE_DIR = os.path.join(BASE_DIR, "sample_data/data/images/train")
LABEL_DIR = os.path.join(BASE_DIR, "sample_data/data/labels/train")

CHUNK_OUTPUT_DIR = os.path.join(BASE_DIR, "output_chunks")

data_yaml = os.path.join(BASE_DIR, "dataset.yaml")

test_images_dir = os.path.join(BASE_DIR, "test_images")

output_dir = os.path.join(BASE_DIR, "predictions")

predictions_txt_dir = os.path.join(BASE_DIR, "prediction_results")

output_file = os.path.join(
    BASE_DIR,
    "prediction_results/unique_nests.txt"
)

hyperparameter_output_dir = os.path.join(
    BASE_DIR,
    "hyperparameter_logs"
)

DEFAULT_WEIGHTS_DIR = os.path.join(
    BASE_DIR,
    "default_weights"
)

SAVE_DIR = os.path.join(BASE_DIR, "saved_models")


# ==========================================================
# VALIDATE CRITICAL PATHS
# ==========================================================

try:

    validate_path(
        data_yaml,
        "file",
        description="Dataset YAML file"
    )

    validate_path(
        test_images_dir,
        "directory",
        description="Test images directory"
    )

    validate_path(
        DEFAULT_WEIGHTS_DIR,
        "directory",
        description="Default weights directory"
    )

    validate_path(
        SAVE_DIR,
        "directory",
        description="Model save directory"
    )

except Exception as e:

    print(
        f"Error during validation: {e}\n{traceback.format_exc()}"
    )

    exit()


# ==========================================================
# DEFAULT WEIGHTS
# ==========================================================

def fetch_default_weights(default_weights_dir):

    model_files = [
        f for f in os.listdir(default_weights_dir)
        if f.endswith(('.pt', '.onnx'))
    ]

    if not model_files:
        raise FileNotFoundError(
            f"No model files (.pt or .onnx) found in: {default_weights_dir}"
        )

    default_weights = os.path.join(
        default_weights_dir,
        model_files[0]
    )

    print(f"Default weights found: {default_weights}")

    return default_weights


# ==========================================================
# YAML
# ==========================================================

def read_yaml(yaml_path):

    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


# ==========================================================
# FILE VALIDATION
# ==========================================================

def get_file_paths(directory, extensions):

    extensions = tuple(ext.lower() for ext in extensions)

    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extensions)
    ]


def validate_image(image_path):

    try:

        img = cv2.imread(image_path)

        if img is None:
            raise ValueError("Image is unreadable")

        return True

    except Exception:
        return False


def validate_label(label_path):

    try:

        with open(label_path, 'r') as file:

            for line in file:

                parts = line.strip().split()

                if len(parts) != 5:
                    raise ValueError("Invalid label format")

                class_id = int(parts[0])

                if class_id < 0:
                    raise ValueError("Invalid class id")

                coords = list(map(float, parts[1:]))

                if not all(0 <= c <= 1 for c in coords):
                    raise ValueError("Invalid bbox coordinates")

        return True

    except Exception:
        return False


def validate_files(base_path, image_dir, label_dir, label_ext=".txt"):

    image_dir = os.path.join(
        base_path,
        image_dir.lstrip('./')
    )

    label_dir = os.path.join(
        base_path,
        label_dir.lstrip('./')
    )

    images = get_file_paths(
        image_dir,
        ('.jpg', '.jpeg', '.png')
    )

    labels = get_file_paths(
        label_dir,
        (label_ext,)
    )

    image_basenames = {
        os.path.splitext(os.path.basename(img))[0]: img
        for img in images
    }

    label_basenames = {
        os.path.splitext(os.path.basename(lbl))[0]: lbl
        for lbl in labels
    }

    unmatched_images = set(image_basenames.keys()) - \
        set(label_basenames.keys())

    unmatched_labels = set(label_basenames.keys()) - \
        set(image_basenames.keys())

    for img_key in unmatched_images:

        os.remove(image_basenames[img_key])

        print(
            f"Deleted unmatched image: {image_basenames[img_key]}"
        )

    for lbl_key in unmatched_labels:

        os.remove(label_basenames[lbl_key])

        print(
            f"Deleted unmatched label: {label_basenames[lbl_key]}"
        )

    for img_key, img_path in image_basenames.items():

        if img_key not in label_basenames:
            continue

        label_path = label_basenames[img_key]

        if not validate_image(img_path):

            os.remove(img_path)
            os.remove(label_path)

            print(
                f"Deleted corrupted image and label: "
                f"{img_path}, {label_path}"
            )

            continue

        if not validate_label(label_path):

            os.remove(label_path)
            os.remove(img_path)

            print(
                f"Deleted corrupted label and image: "
                f"{label_path}, {img_path}"
            )


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(weights_path):

    try:

        model = YOLO(weights_path)

        print(f"Model loaded with weights: {weights_path}")

        return model

    except Exception as e:

        raise RuntimeError(
            f"Error loading YOLO model: {e}"
        )


# ==========================================================
# SPLIT DATASET
# ==========================================================

def split_dataset(image_dir, label_dir, output_dir, chunk_size=200):

    image_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    }

    label_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(label_dir)
        if f.lower().endswith('.txt')
    }

    common_keys = sorted(
        set(image_files.keys()) &
        set(label_files.keys())
    )

    for i in range(0, len(common_keys), chunk_size):

        chunk_keys = common_keys[i:i + chunk_size]

        chunk_dir = os.path.join(
            output_dir,
            f"chunk_{i // chunk_size}"
        )

        os.makedirs(chunk_dir, exist_ok=True)

        os.makedirs(
            os.path.join(chunk_dir, "images"),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(chunk_dir, "labels"),
            exist_ok=True
        )

        for key in chunk_keys:

            img = image_files[key]
            lbl = label_files[key]

            shutil.copy(
                os.path.join(image_dir, img),
                os.path.join(chunk_dir, "images", img)
            )

            shutil.copy(
                os.path.join(label_dir, lbl),
                os.path.join(chunk_dir, "labels", lbl)
            )

    logging.info(
        f"Dataset split into chunks saved at: {output_dir}"
    )


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_yolov8_model(
    data_yaml,
    weights_path,
    model_name='yolov8n.pt',
    epochs=1,
    batch_size=8,
    use_cpu=False
):

    device = (
        "cpu"
        if use_cpu
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    if weights_path and os.path.exists(weights_path):

        model = YOLO(
            weights_path,
            task='detect'
        ).to(device)

        print(
            f"Loaded pre-trained weights from: {weights_path}"
        )

    else:

        model = YOLO(
            model_name,
            task='detect'
        ).to(device)

        print(f"Using default model: {model_name}")

    try:

        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=1024,
            device=device,
            patience=20,
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3,
            optimizer='AdamW',
            weight_decay=0.01,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.5,
            iou=0.4,
            conf=0.5,
            amp=True,
            verbose=True,
        )

    except RuntimeError as e:

        print(
            f"Training failed with error: "
            f"{e}\n{traceback.format_exc()}"
        )

        raise

    except AssertionError as e:

        print(
            f"Assertion error: {e}. "
            f"Check dataset and labels."
        )

        raise

    return model


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(model):

    try:

        results = model.val()

        precision = results.box.mp
        recall = results.box.mr
        map50 = results.box.map50
        map50_95 = results.box.map

        confusion_matrix = results.confusion_matrix.matrix

        tp = np.diag(confusion_matrix).sum()

        fn = confusion_matrix.sum(axis=1) - \
            np.diag(confusion_matrix)

        recall_from_confusion = (
            tp / (tp + fn.sum())
            if (tp + fn.sum()) > 0 else 0.0
        )

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"mAP@0.5: {map50:.4f}")
        print(f"mAP@0.5:0.95: {map50_95:.4f}")

        print(
            f"Recall from Confusion Matrix: "
            f"{recall_from_confusion:.4f}"
        )

    except Exception as e:

        raise RuntimeError(
            f"Evaluation failed: {e}"
        )


# ==========================================================
# SAVE ACCURACY
# ==========================================================

def save_and_display_accuracy(model, results_dir):

    ensure_directory_exists(results_dir)

    try:

        results = model.val()

        precision = results.box.mp
        recall = results.box.mr
        map50 = results.box.map50
        map50_95 = results.box.map

        accuracy_report = (
            f"Model Accuracy Report\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"mAP@0.5: {map50:.4f}\n"
            f"mAP@0.5:0.95: {map50_95:.4f}\n"
        )

        print(accuracy_report)

        accuracy_file = os.path.join(
            results_dir,
            f'model_accuracy_'
            f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        )

        with open(accuracy_file, 'w') as f:
            f.write(accuracy_report)

        print(
            f"Model accuracy saved to: "
            f"{accuracy_file}"
        )

    except Exception as e:

        print(
            f"Failed to evaluate and save accuracy: {e}"
        )


# ==========================================================
# PREDICT AND SAVE IMAGES
# ==========================================================

def predict_and_save_images(
    model,
    test_images_dir,
    output_dir,
    confidence_threshold=0.6
):

    image_files = [
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith(
            ('.png', '.jpg', '.jpeg')
        )
    ]

    print(
        f"Found {len(image_files)} images "
        f"in {test_images_dir}."
    )

    predicted_count = 0

    predictions = []

    for image_file in image_files:

        image_path = os.path.join(
            test_images_dir,
            image_file
        )

        try:

            results = model.predict(
                source=image_path,
                conf=confidence_threshold
            )

            if (
                results and
                len(results) > 0 and
                hasattr(results[0], 'boxes') and
                results[0].boxes is not None and
                len(results[0].boxes.xyxy) > 0
            ):

                predictions.append(
                    (results[0], image_file)
                )

                img = cv2.imread(image_path)

                boxes = results[0].boxes.xyxy.cpu().numpy()

                confidences = results[0].boxes.conf.cpu().numpy()

                for i, box in enumerate(boxes):

                    x1, y1, x2, y2 = [
                        int(coord) for coord in box
                    ]

                    confidence = confidences[i]

                    cls_id = int(
                        results[0].boxes.cls[i]
                    )

                    class_name = model.names[cls_id]

                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    label = (
                        f"{class_name}: "
                        f"{confidence:.2f}"
                    )

                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                save_path = os.path.join(
                    output_dir,
                    image_file
                )

                cv2.imwrite(save_path, img)

                predicted_count += 1

                print(
                    f"Saved predicted image: "
                    f"{save_path}"
                )

            else:

                print(
                    f"No predictions for image: "
                    f"{image_file}"
                )

        except Exception as e:

            print(
                f"Error processing image "
                f"{image_file}: {e}"
            )

    print(
        f"Total predicted images saved: "
        f"{predicted_count}/{len(image_files)}"
    )

    return predictions


# ==========================================================
# SAVE MODEL
# ==========================================================

def save_model(model, save_dir=SAVE_DIR):

    ensure_directory_exists(save_dir)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    model_save_path = os.path.join(
        save_dir,
        f'model_{timestamp}.pt'
    )

    model.save(model_save_path)

    print(f"Model saved at: {model_save_path}")


# ==========================================================
# LOAD LAST SAVED WEIGHTS
# ==========================================================

def get_last_saved_weights(save_dir):

    validate_path(save_dir, "directory")

    weight_files = [
        f for f in os.listdir(save_dir)
        if f.endswith('.pt')
    ]

    if not weight_files:

        print(
            "No .pt weights found. "
            "Continuing without saved weights."
        )

        return None

    weight_files = sorted(
        weight_files,
        key=lambda f: os.path.getmtime(
            os.path.join(save_dir, f)
        ),
        reverse=True
    )

    last_saved_weights = os.path.join(
        save_dir,
        weight_files[0]
    )

    print(
        f"Last saved weights found: "
        f"{last_saved_weights}"
    )

    return last_saved_weights


# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

def save_prediction_results(
    predictions,
    results_dir
):

    ensure_directory_exists(results_dir)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    result_file = os.path.join(
        results_dir,
        f'prediction_results_{timestamp}.txt'
    )

    with open(result_file, 'w') as f:

        f.write("Image Name\tConfidence\n")

        for prediction, image_file in predictions:

            confidences = prediction.boxes.conf.cpu().numpy()

            for confidence in confidences:

                f.write(
                    f"{image_file}\t{confidence:.4f}\n"
                )

    print(
        f"Prediction results saved to: "
        f"{result_file}"
    )


# ==========================================================
# EXIF
# ==========================================================

exif_file = os.environ.get(
    "EXIF_INFO_PATH",
    "site1.xlsx"
)


def load_exif_data(exif_file):

    exif_df = pd.read_excel(exif_file)

    exif_df['filePath'] = (
        exif_df['filePath']
        .str.lower()
        .str.replace(".jpg", "")
        .str.replace(".jpeg", "")
    )

    return exif_df


def normalize_image_name(image_name):

    base_name = os.path.splitext(image_name)[0]

    if "_crop_" in base_name:
        base_name = base_name.split("_crop_")[0]

    return base_name.lower()


def find_unique_bird_nests(
    predictions,
    exif_data,
    distance_threshold=5.0
):

    unique_nests = []

    for image_name, _ in predictions:

        normalized_name = normalize_image_name(
            image_name
        )

        exif_row = exif_data[
            exif_data['filePath'] ==
            normalized_name
        ]

        if not exif_row.empty:

            latitude, longitude, altitude = (
                exif_row.iloc[0][
                    ['latitude', 'longitude', 'altitude']
                ]
            )

            nest_coords = (latitude, longitude)

            is_unique = True

            for unique_nest in unique_nests:

                distance = geodesic(
                    nest_coords,
                    (
                        unique_nest['latitude'],
                        unique_nest['longitude']
                    )
                ).meters

                if distance < distance_threshold:
                    is_unique = False
                    break

            if is_unique:

                unique_nests.append({
                    'latitude': latitude,
                    'longitude': longitude,
                    'altitude': altitude
                })

    return unique_nests


def process_unique_nests(
    predictions,
    exif_file,
    output_file
):

    exif_data = load_exif_data(exif_file)

    unique_nests = find_unique_bird_nests(
        predictions,
        exif_data
    )

    with open(output_file, 'w') as f:

        f.write("Unique Bird Nests:\n")

        for idx, nest in enumerate(unique_nests, 1):

            f.write(
                f"{idx}. Latitude: "
                f"{nest['latitude']}, "
                f"Longitude: {nest['longitude']}, "
                f"Altitude: {nest['altitude']}\n"
            )

    print(
        f"Unique bird nests saved to "
        f"{output_file}"
    )


def predict_and_process_unique_nests(
    model,
    test_images_dir,
    exif_file,
    output_file,
    confidence_threshold=0.6
):

    image_files = [
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith(
            ('.png', '.jpg', '.jpeg')
        )
    ]

    predictions = []

    for image_file in image_files:

        image_path = os.path.join(
            test_images_dir,
            image_file
        )

        results = model.predict(
            source=image_path,
            conf=confidence_threshold
        )

        if results and hasattr(results[0], 'path'):

            predictions.append(
                (
                    os.path.basename(results[0].path),
                    results[0].boxes
                )
            )

    process_unique_nests(
        predictions,
        exif_file,
        output_file
    )


# ==========================================================
# TIME LOGGING
# ==========================================================

def log_training_and_prediction_time(
    training_start_time,
    prediction_start_time,
    output_dir
):

    training_end_time = time.time()

    total_training_time = (
        training_end_time -
        training_start_time
    )

    prediction_end_time = time.time()

    total_prediction_time = (
        prediction_end_time -
        prediction_start_time
    )

    def format_time(seconds):

        hours = int(seconds // 3600)

        minutes = int(
            (seconds % 3600) // 60
        )

        seconds = seconds % 60

        return (
            f"{hours}h "
            f"{minutes}m "
            f"{seconds:.2f}s"
        )

    training_time_str = format_time(
        total_training_time
    )

    prediction_time_str = format_time(
        total_prediction_time
    )

    time_log = (
        f"Time Report\n"
        f"Training Time: "
        f"{training_time_str}\n"
        f"Prediction Time: "
        f"{prediction_time_str}\n"
    )

    ensure_directory_exists(output_dir)

    time_log_path = os.path.join(
        output_dir,
        f"time_log_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    with open(time_log_path, 'w') as log_file:
        log_file.write(time_log)

    print(time_log)

    print(
        f"Time log saved at: "
        f"{time_log_path}"
    )


# ==========================================================
# SAVE HYPERPARAMETERS
# ==========================================================

def save_hyperparameters_from_script(
    hyperparameter_output_dir
):

    os.makedirs(
        hyperparameter_output_dir,
        exist_ok=True
    )

    hyperparameters = {
        "epochs": 1,
        "batch_size": 8,
        "imgsz": 1024,
        "patience": 20,
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 3,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mosaic": 0.5,
        "iou": 0.4,
        "conf": 0.5,
        "amp": True,
        "verbose": True,
        "chunk_size": 150
    }

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    file_name = (
        f"hyperParameter_info_"
        f"{timestamp}.txt"
    )

    file_path = os.path.join(
        hyperparameter_output_dir,
        file_name
    )

    with open(file_path, "w") as file:

        file.write(
            "Hyperparameter Information\n"
        )

        file.write(
            f"Saved on: {datetime.now()}\n"
        )

        file.write("=" * 50 + "\n")

        for key, value in hyperparameters.items():

            file.write(
                f"{key}: {value}\n"
            )

    print(
        f"Hyperparameters saved to: "
        f"{file_path}"
    )


# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

def create_parent_output_folder(
    base_dir,
    base_name="Final_Train_Predict_Output"
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    folder_name = (
        f"{base_name}_{timestamp}"
    )

    folder_path = os.path.join(
        base_dir,
        folder_name
    )

    os.makedirs(
        folder_path,
        exist_ok=True
    )

    print(
        f"Created output folder: "
        f"{folder_path}"
    )

    return folder_path


def cleanup_duplicate_directories(
    base_dir,
    parent_folder,
    tracked_paths
):

    for path in tracked_paths:

        expected_path = path

        duplicate_path = os.path.join(
            base_dir,
            os.path.basename(path)
        )

        if (
            os.path.exists(duplicate_path) and
            duplicate_path != expected_path
        ):

            try:

                shutil.rmtree(duplicate_path)

                print(
                    f"Removed duplicate directory: "
                    f"{duplicate_path}"
                )

            except Exception as e:

                print(
                    f"Failed to remove duplicate directory: "
                    f"{duplicate_path}. Error: {e}"
                )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    training_start_time = time.time()

    base_directory = "."

    parent_folder = create_parent_output_folder(
        base_dir=base_directory
    )

    validated_directories = set()

    output_dir = os.path.join(
        parent_folder,
        "predictions"
    )

    predictions_txt_dir = os.path.join(
        parent_folder,
        "prediction_results"
    )

    save_dir = os.path.join(
        parent_folder,
        "saved_models"
    )

    chunk_output_dir = os.path.join(
        parent_folder,
        "output_chunks"
    )

    hyperparameter_output_dir = os.path.join(
        parent_folder,
        "hyperparameter_logs"
    )

    ensure_directory_exists(output_dir)
    ensure_directory_exists(predictions_txt_dir)
    ensure_directory_exists(save_dir)
    ensure_directory_exists(chunk_output_dir)
    ensure_directory_exists(
        hyperparameter_output_dir
    )

    try:

        validate_path(
            output_dir,
            "directory",
            description="Output directory",
            tracked_paths=validated_directories
        )

        validate_path(
            predictions_txt_dir,
            "directory",
            description="Predictions TXT directory",
            tracked_paths=validated_directories
        )

        validate_path(
            save_dir,
            "directory",
            description="Model save directory",
            tracked_paths=validated_directories
        )

        validate_path(
            chunk_output_dir,
            "directory",
            description="Chunk output directory",
            tracked_paths=validated_directories
        )

        validate_path(
            hyperparameter_output_dir,
            "directory",
            description="Hyperparameter output directory",
            tracked_paths=validated_directories
        )

        print(
            "All directories are validated and tracked."
        )

    except Exception as e:

        print(
            f"Error during directory validation: {e}"
        )

    cleanup_duplicate_directories(
        base_directory,
        parent_folder,
        validated_directories
    )

    try:

        print("Script execution starts here...")

        data = read_yaml(data_yaml)

        base_path = data['path']

        train_images = data['train']

        val_images = data['val']

        print("Validating training data...")

        validate_files(
            base_path,
            train_images,
            train_images.replace(
                'images',
                'labels'
            )
        )

        print("Validating validation data...")

        validate_files(
            base_path,
            val_images,
            val_images.replace(
                'images',
                'labels'
            )
        )

        print("Validation completed.")

        validate_labels(
            LABEL_DIR,
            num_classes=1
        )

        validate_images(IMAGE_DIR)

        save_hyperparameters_from_script(
            hyperparameter_output_dir
        )

        split_dataset(
            IMAGE_DIR,
            LABEL_DIR,
            CHUNK_OUTPUT_DIR,
            chunk_size=150
        )

        selected_chunk = os.path.join(
            CHUNK_OUTPUT_DIR,
            "chunk_0"
        )

        image_dir = os.path.join(
            selected_chunk,
            "images"
        )

        label_dir = os.path.join(
            selected_chunk,
            "labels"
        )

        weights_to_use = get_last_saved_weights(
            SAVE_DIR
        )

        if weights_to_use:

            print(
                f"Loading last saved weights: "
                f"{weights_to_use}"
            )

        else:

            print(
                "No saved weights found. "
                "Using default YOLOv8 model."
            )

            weights_to_use = fetch_default_weights(
                DEFAULT_WEIGHTS_DIR
            )

        model = train_yolov8_model(
            data_yaml,
            weights_to_use
        )

        training_end_time = time.time()

        prediction_start_time = time.time()

        evaluate_model(model)

        save_and_display_accuracy(
            model,
            predictions_txt_dir
        )

        predictions = predict_and_save_images(
            model,
            test_images_dir,
            output_dir
        )

        save_prediction_results(
            predictions,
            predictions_txt_dir
        )

        predict_and_process_unique_nests(
            model,
            test_images_dir,
            exif_file,
            output_file,
            confidence_threshold=0.6
        )

        save_model(model, save_dir)

        log_training_and_prediction_time(
            training_start_time,
            prediction_start_time,
            output_dir
        )

    except Exception as e:

        print(
            f"Error during training or execution: "
            f"{e}\n{traceback.format_exc()}"
        )