import os
import cv2
import yaml
import time
import inspect
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


# Utility function to create directories if they don't exist
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Validate a file or directory path


def validate_path(path, path_type="directory", description="Path", tracked_paths=None):
    """
    Validate whether the given path is a valid file or directory.
    If the directory does not exist, create it automatically.
    Track the validated paths for further operations like cleanup.
    :param path: Path to validate.
    :param path_type: "file" to check for a file, "directory" to check for a directory.
    :param description: Description of the path for error messages.
    :param tracked_paths: A set to track validated directory paths (for cleanup).
    """
    if path_type == "directory":
        if not os.path.isdir(path):
            print(f"{description} not found. Creating: {path}")
            ensure_directory_exists(path)
        # Track the validated directory
        if tracked_paths is not None:
            tracked_paths.add(path)
    elif path_type == "file":
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{description} is invalid. File not found: {path}")
    print(f"Validated {description}: {path}")


def validate_labels(label_dir, num_classes):
    for label_file in os.listdir(label_dir):
        path = os.path.join(label_dir, label_file)
        with open(path, 'r') as f:
            for line_num, line in enumerate(f.readlines(), 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(
                        f"Malformed label in {label_file} at line {line_num}: {line}")
                    continue
                class_id = int(parts[0])
                if class_id < 0 or class_id >= num_classes:
                    print(
                        f"Invalid class ID in {label_file} at line {line_num}: {class_id}")
                bbox = list(map(float, parts[1:]))
                if not all(0 <= coord <= 1 for coord in bbox):
                    print(
                        f"Invalid bounding box in {label_file} at line {line_num}: {bbox}")


def validate_images(image_dir):
    for img_file in os.listdir(image_dir):
        try:
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path)
            img.verify()  # Check if image is valid
        except Exception as e:
            print(f"Invalid image file: {img_file}, Error: {e}")


# Define paths - Refactored to relative paths
BASE_DIR = "."

IMAGE_DIR = os.path.join(BASE_DIR, "sample_data/data/images/train")
LABEL_DIR = os.path.join(BASE_DIR, "sample_data/data/labels/train")
CHUNK_OUTPUT_DIR = os.path.join(BASE_DIR, "output_chunks")
# Path to your dataset.yaml file
data_yaml = os.path.join(BASE_DIR, "dataset.yaml")
test_images_dir = os.path.join(BASE_DIR, "test_images")
output_dir = os.path.join(BASE_DIR, "predictions")
predictions_txt_dir = os.path.join(BASE_DIR, "prediction_results")
output_file = os.path.join(BASE_DIR, "prediction_results/unique_nests.txt")

# Output directory for saving hyperparameter files
hyperparameter_output_dir = os.path.join(BASE_DIR, "hyperparameter_logs")
# Set default weight path in the code
DEFAULT_WEIGHTS_DIR = os.path.join(BASE_DIR, "default_weights")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")  # Directory to save models


# Validate critical paths
try:
    validate_path(data_yaml, "file", description="Dataset YAML file")
    validate_path(test_images_dir, "directory",
                  description="Test images directory")
    validate_path(DEFAULT_WEIGHTS_DIR, "directory",
                  description="Default weights directory")
    validate_path(SAVE_DIR, "directory", description="Model save directory")
except Exception as e:
    print(f"Error during validation: {e}\n{traceback.format_exc()}")
    exit()


# Step 2: Fetch default weights if no saved weights are found
def fetch_default_weights(default_weights_dir):
    """
    Fetch a model file from the default weights directory if no saved weights are found.
    """
    model_files = [f for f in os.listdir(
        default_weights_dir) if f.endswith(('.pt', '.onnx'))]
    if not model_files:
        raise FileNotFoundError(
            f"No model files (.pt or .onnx) found in the default weights directory: {default_weights_dir}")

    # Return the first available model file
    default_weights = os.path.join(default_weights_dir, model_files[0])
    print(f"Default weights found: {default_weights}")
    return default_weights


# Global variable for the YAML file path
# yaml_path = "path_to_your_yaml_file.yaml"

def read_yaml(yaml_path):
    """Read and parse the YAML file."""
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


def get_file_paths(directory, extensions):
    """Get file paths with specific extensions in a directory, case-insensitively."""
    extensions = tuple(ext.lower() for ext in extensions)
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(extensions)]


def validate_files(base_path, image_dir, label_dir, label_ext=".txt"):
    """Check and delete unmatched or corrupted files."""
    image_dir = os.path.join(base_path, image_dir.lstrip('./'))
    label_dir = os.path.join(base_path, label_dir.lstrip('./'))

    images = get_file_paths(image_dir, ('.jpg', '.jpeg', '.png'))
    labels = get_file_paths(label_dir, (label_ext,))

    # Map filenames without extensions
    image_basenames = {os.path.splitext(os.path.basename(img))[
        0]: img for img in images}
    label_basenames = {os.path.splitext(os.path.basename(lbl))[
        0]: lbl for lbl in labels}

    # Identify unmatched files
    unmatched_images = set(image_basenames.keys()) - \
        set(label_basenames.keys())
    unmatched_labels = set(label_basenames.keys()) - \
        set(image_basenames.keys())

    # Remove unmatched files
    for img_key in unmatched_images:
        os.remove(image_basenames[img_key])
        print(f"Deleted unmatched image: {image_basenames[img_key]}")

    for lbl_key in unmatched_labels:
        os.remove(label_basenames[lbl_key])
        print(f"Deleted unmatched label: {label_basenames[lbl_key]}")

    # Validate and remove corrupted files
    for img_key, img_path in image_basenames.items():
        if img_key not in label_basenames:
            continue

        label_path = label_basenames[img_key]

        # Check image validity
        if not validate_image(img_path):
            os.remove(img_path)
            os.remove(label_path)
            print(
                f"Deleted corrupted image and its label: {img_path}, {label_path}")
            continue

        # Check label validity
        if not validate_label(label_path):
            os.remove(label_path)
            os.remove(img_path)
            print(
                f"Deleted corrupted label and its image: {label_path}, {img_path}")


def validate_image(image_path):
    """Check if an image file is corrupted."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image is unreadable")
        return True
    except Exception:
        return False


def validate_label(label_path):
    """Check if a label file is in the correct format."""
    try:
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 5:
                    raise ValueError("Invalid label format")
                _ = [float(p) for p in parts]  # Ensure all parts are floats
        return True
    except Exception:
        return False


# Step 3: Load YOLO model
def load_model(weights_path):
    """
    Load the YOLO model with the specified weights.
    """
    try:
        model = YOLO(weights_path)
        print(f"Model loaded with weights: {weights_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading YOLO model: {e}")


logging.basicConfig(level=logging.INFO)

# Splitting the dataset into smaller chunks


def split_dataset(image_dir, label_dir, output_dir, chunk_size=200):
    """
    Splits a dataset into smaller chunks for easier debugging and memory management.
    """
    images = sorted(os.listdir(image_dir))
    labels = sorted(os.listdir(label_dir))
    assert len(images) == len(labels), "Mismatch between images and labels"

    for i in range(0, len(images), chunk_size):
        chunk_images = images[i:i + chunk_size]
        chunk_labels = labels[i:i + chunk_size]
        chunk_dir = os.path.join(output_dir, f"chunk_{i // chunk_size}")
        os.makedirs(chunk_dir, exist_ok=True)
        os.makedirs(os.path.join(chunk_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(chunk_dir, "labels"), exist_ok=True)

        for img, lbl in zip(chunk_images, chunk_labels):
            shutil.copy(os.path.join(image_dir, img),
                        os.path.join(chunk_dir, "images", img))
            shutil.copy(os.path.join(label_dir, lbl),
                        os.path.join(chunk_dir, "labels", lbl))
    logging.info(f"Dataset split into chunks saved at: {output_dir}")


# Step 1: Train YOLOv8 model
def train_yolov8_model(data_yaml, weights_path, model_name='yolov8n.pt', epochs=1, batch_size=8, use_cpu=False):
    """
    Train YOLOv8 model using the provided images and labels.
    :param data_yaml: Path to the dataset YAML file.
    :param model_name: YOLOv8 model variant.
    :param epochs: Number of training epochs.
    :param weights_path: Path to pre-trained weights (optional).
    :return: Trained model object.
    """
    device = "cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model with pre-trained weights or default YOLOv8 model
    if weights_path and os.path.exists(weights_path):
        model = YOLO(weights_path, task='detect').to(device)
        print(f"Loaded pre-trained weights from: {weights_path}")
    else:
        model = YOLO(model_name, task='detect').to(device)
        print(f"Using default model: {model_name}")

    # model.train(data=data_yaml, epochs=epochs, batch=batch_size, device=device)
    try:
        model.train(
            data=data_yaml,
            epochs=epochs,	     # Total number of epochs
            batch=batch_size,	     # Batch size
            imgsz=1024,  	     # Image size
            device=device,
            patience=20,             # Early stopping patience
            lr0=0.001,               # Initial learning rate
            lrf=0.01,                # Final learning rate fraction
            warmup_epochs=3,         # Warmup epochs
            optimizer='AdamW',       # Optimizer
            weight_decay=0.01,       # Weight decay
            hsv_h=0.015,             # Hue augmentation
            hsv_s=0.7,               # Saturation augmentation
            hsv_v=0.4,               # Value augmentation
            fliplr=0.5,              # Horizontal flip probability
            flipud=0.0,              # Vertical flip probability
            mosaic=0.5,              # Mosaic augmentation probability
            iou=0.4,                 # IoU threshold
            # Confidence threshold - Increase confidence threshold to reduce false positives
            conf=0.5,
            amp=True,                # Automatic mixed precision
            verbose=True,

        )
    except RuntimeError as e:
        print(f"Training failed with error: {e}\n{traceback.format_exc()}")
        raise
    except AssertionError as e:
        print(f"Assertion error: {e}. Check dataset and labels.")
        raise
    return model

# Step 2: Evaluate model and print metrics


def evaluate_model(model):
    """
    Evaluate the YOLOv8 model and print relevant metrics.
    :param model: Trained YOLOv8 model.
    """

    try:
        results = model.val()  # Validation results
        precision = results.box.map50
        map50_95 = results.box.map
        confusion_matrix = results.confusion_matrix.matrix
        tp = np.diag(confusion_matrix).sum()
        fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        recall_from_confusion = tp / \
            (tp + fn.sum()) if (tp + fn.sum()) > 0 else 0.0
        print(f"Precision: {precision:.4f}")
        print(f"mAP@0.5: {precision:.4f}")
        print(f"mAP@0.5:0.95: {map50_95:.4f}")
        print(f"Recall from Confusion Matrix: {recall_from_confusion:.4f}")
    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {e}")


# Step 8: Save model accuracy to a file and display it on the terminal
def save_and_display_accuracy(model, results_dir):
    """
    Evaluate the YOLO model, save its accuracy to a file, and display it on the terminal.
    """
    ensure_directory_exists(results_dir)

    try:
        # Evaluate the model and fetch metrics
        results = model.val()  # Validation results
        precision = results.box.map50
        map50_95 = results.box.map
        accuracy_report = (
            f"Model Accuracy Report\n"
            f"Precision (mAP@0.5): {precision:.4f}\n"
            f"mAP@0.5:0.95: {map50_95:.4f}\n"
        )
        print(accuracy_report)  # Display on the terminal

        # Save accuracy to a text file
        accuracy_file = os.path.join(
            results_dir, f'model_accuracy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(accuracy_file, 'w') as f:
            f.write(accuracy_report)

        print(f"Model accuracy saved to: {accuracy_file}")

    except Exception as e:
        print(f"Failed to evaluate and save accuracy: {e}")


# Step 2: Predict and save only images with predictions
def predict_and_save_images(model, test_images_dir, output_dir, confidence_threshold=0.6):
    """
    Predict on images in the test directory and save only those with predictions.
    """
    image_files = [f for f in os.listdir(
        test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in {test_images_dir}.")

    predicted_count = 0
    predictions = []

    for image_file in image_files:
        image_path = os.path.join(test_images_dir, image_file)
        try:
            # Perform prediction
            results = model.predict(
                source=image_path, conf=confidence_threshold)

            # Validate results
            if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
                # Store valid predictions for results file
                predictions.append((results[0], image_file))

                # Load the image
                img = cv2.imread(image_path)

                # Draw bounding boxes
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    confidence = confidences[i]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Conf: {confidence:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save the image with predictions
                save_path = os.path.join(output_dir, image_file)
                cv2.imwrite(save_path, img)
                predicted_count += 1
                print(f"Saved predicted image: {save_path}")
            else:
                print(f"No predictions for image: {image_file}")

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")

    print(
        f"Total predicted images saved: {predicted_count}/{len(image_files)}")
    return predictions


# Step 5: Auto-save trained model with a unique name
def save_model(model, save_dir=SAVE_DIR):
    """
    Save the trained YOLO model with a unique timestamped name.
    """
    ensure_directory_exists(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(save_dir, f'model_{timestamp}.pt')
    model.save(model_save_path)
    print(f"Model saved at: {model_save_path}")

# Step 6: Load last saved weights


def get_last_saved_weights(save_dir):
    """
    Get the path of the most recently saved weights.
    """
    validate_path(save_dir, "directory")
    weight_files = [f for f in os.listdir(
        save_dir) if f.endswith('.pt')]  # Only .pt files
    if not weight_files:
        print("No .pt weights found in the specified directory. Continuing without saved weights.")
        return None

    # Sort files by modification time
    weight_files = sorted(weight_files, key=lambda f: os.path.getmtime(
        os.path.join(save_dir, f)), reverse=True)
    last_saved_weights = os.path.join(save_dir, weight_files[0])
    print(f"Last saved weights found: {last_saved_weights}")
    return last_saved_weights

# Step 7: Save prediction results


def save_prediction_results(predictions, results_dir):
    """
    Save prediction results to a text file.
    """
    ensure_directory_exists(results_dir)

    # Generate a unique filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Type of datetime: {type(datetime)}")

    result_file = os.path.join(
        results_dir, f'prediction_results_{timestamp}.txt')

    with open(result_file, 'w') as f:
        f.write("Image Name\tConfidence\n")
        for prediction, image_file in predictions:
            confidences = prediction.boxes.conf.cpu().numpy()
            for confidence in confidences:
                f.write(f"{image_file}\t{confidence:.4f}\n")

    print(f"Prediction results saved to: {result_file}")


# Set default EXIF file or read from environment variable
exif_file = os.environ.get("EXIF_INFO_PATH", "site1.xlsx")

# Load EXIF data


def load_exif_data(exif_file):
    exif_df = pd.read_excel(exif_file)
    exif_df['filePath'] = exif_df['filePath'].str.lower(
    ).str.replace(".jpg", "").str.replace(".jpeg", "")
    return exif_df

# Normalize image names


def normalize_image_name(image_name):
    base_name = os.path.splitext(image_name)[0]
    if "_crop_" in base_name:
        base_name = base_name.split("_crop_")[0]
    return base_name.lower()

# Find unique bird nests based on GPS coordinates


def find_unique_bird_nests(predictions, exif_data, distance_threshold=5.0):
    unique_nests = []
    for image_name, _ in predictions:
        normalized_name = normalize_image_name(image_name)
        exif_row = exif_data[exif_data['filePath'] == normalized_name]
        if not exif_row.empty:
            latitude, longitude, altitude = exif_row.iloc[0][[
                'latitude', 'longitude', 'altitude']]
            nest_coords = (latitude, longitude)
            is_unique = True
            for unique_nest in unique_nests:
                distance = geodesic(
                    nest_coords, (unique_nest['latitude'], unique_nest['longitude'])).meters
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

# Process and save unique nests


def process_unique_nests(predictions, exif_file, output_file):
    exif_data = load_exif_data(exif_file)
    unique_nests = find_unique_bird_nests(predictions, exif_data)
    with open(output_file, 'w') as f:
        f.write("Unique Bird Nests:\n")
        for idx, nest in enumerate(unique_nests, 1):
            f.write(
                f"{idx}. Latitude: {nest['latitude']}, Longitude: {nest['longitude']}, Altitude: {nest['altitude']}\n")
    print(f"Unique bird nests saved to {output_file}")

# Main function to integrate predictions


def predict_and_process_unique_nests(model, test_images_dir, exif_file, output_file, confidence_threshold=0.6):
    image_files = [f for f in os.listdir(
        test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    predictions = []

    for image_file in image_files:
        image_path = os.path.join(test_images_dir, image_file)
        results = model.predict(source=image_path, conf=confidence_threshold)
        if results and hasattr(results[0], 'path'):
            # Store image name and boxes
            predictions.append(
                (os.path.basename(results[0].path), results[0].boxes))

    # Process unique nests
    process_unique_nests(predictions, exif_file, output_file)


def log_training_and_prediction_time(training_start_time, prediction_start_time, output_dir):
    """
    Calculate and save the training and prediction times in hours, minutes, and seconds.
    :param training_start_time: Timestamp when training started.
    :param prediction_start_time: Timestamp when prediction started.
    :param output_dir: Directory to save the time logs.
    """
    # Calculate times
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    prediction_end_time = time.time()
    total_prediction_time = prediction_end_time - prediction_start_time

    # Convert seconds to hours, minutes, seconds
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.2f}s"

    # Prepare log content
    training_time_str = format_time(total_training_time)
    prediction_time_str = format_time(total_prediction_time)

    time_log = (
        f"Time Report\n"
        f"Training Time: {training_time_str}\n"
        f"Prediction Time: {prediction_time_str}\n"
    )

    # Save to file
    ensure_directory_exists(output_dir)
    time_log_path = os.path.join(
        output_dir, f"time_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(time_log_path, 'w') as log_file:
        log_file.write(time_log)

    # Display on terminal
    print(time_log)
    print(f"Time log saved at: {time_log_path}")


def save_hyperparameters_from_script(hyperparameter_output_dir):
    """
    Dynamically extract and save hyperparameters from the script's global variables.

    :param global_vars: The global variables of the script (use `globals()`).
    :param output_dir: Directory to save the hyperparameter file.
    """
    os.makedirs(hyperparameter_output_dir,
                exist_ok=True)  # Ensure the output directory exists

    # Define the list of hyperparameters to extract
    hyperparameter_keys = [
        "epochs", "batch_size", "imgsz", "patience",
        "lr0", "lrf", "warmup_epochs", "optimizer",
        "weight_decay", "hsv_h", "hsv_s", "hsv_v",
        "fliplr", "flipud", "mosaic", "iou", "conf",
        "amp", "verbose", "chunk_size"
    ]

    # Filter global variables to find the hyperparameters

    # Dictionary to store the hyperparameters
    hyperparameters = {key: "NOT FOUND" for key in hyperparameter_keys}

    # Inspect all functions in the current script
    for name, obj in inspect.getmembers(__import__("__main__")):
        if inspect.isfunction(obj):  # Check if the object is a function
            try:
                # Get the function's source code
                source_code = inspect.getsource(obj)
                for key in hyperparameter_keys:
                    if f"{key} =" in source_code or f"{key}=" in source_code:
                        try:
                            # Execute the function's code to extract variables
                            exec_globals = {}
                            exec_locals = {}
                            exec(source_code, exec_globals, exec_locals)
                            if key in exec_locals:
                                hyperparameters[key] = exec_locals[key]
                        except Exception as e:
                            # Catch errors from try-except blocks or invalid code
                            if key not in hyperparameters or hyperparameters[key] == "NOT FOUND":
                                print(
                                    f"Warning: Could not extract {key} from {obj.__name__}: {e}")
            except Exception as e:
                print(f"Error while processing function {name}: {e}")

    # hyperparameters = {key: global_vars.get(key, "NOT DEFINED") for key in hyperparameter_keys}

    # Generate a unique filename using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"hyperParameter_info_{timestamp}.txt"
    file_path = os.path.join(hyperparameter_output_dir, file_name)

    # Write hyperparameters to the file
    with open(file_path, "w") as file:
        file.write("Hyperparameter Information\n")
        file.write(f"Saved on: {datetime.now()}\n")
        file.write("=" * 50 + "\n")
        for key, value in hyperparameters.items():
            file.write(f"{key}: {value}\n")

    print(f"Hyperparameters saved to: {file_path}")


# print(globals().keys())


def create_parent_output_folder(base_dir, base_name="Final_Train_Predict_Output"):
    """
    Create a parent output folder with a unique name in the specified base directory.
    :param base_dir: The directory where the parent folder will be created.
    :param base_name: The base name of the parent folder.
    :return: The path to the created folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{base_name}_{timestamp}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created output folder: {folder_path}")
    return folder_path


def cleanup_duplicate_directories(base_dir, parent_folder, tracked_paths):
    """
    Remove any duplicate directories outside the parent folder.
    :param base_dir: The base directory where duplicates might exist.
    :param parent_folder: The parent folder where subdirectories are created.
    :param tracked_paths: A set of tracked directory paths to check for duplicates.
    """
    for path in tracked_paths:
        expected_path = path
        duplicate_path = os.path.join(base_dir, os.path.basename(path))

        if os.path.exists(duplicate_path) and duplicate_path != expected_path:
            try:
                # Use shutil.rmtree to handle non-empty directories
                shutil.rmtree(duplicate_path)
                print(f"Removed duplicate directory: {duplicate_path}")
            except Exception as e:
                print(
                    f"Failed to remove duplicate directory: {duplicate_path}. Error: {e}")


def get_directory_snapshot(base_dir):
    """
    Take a snapshot of all files and directories in the base directory.
    :param base_dir: The directory to monitor.
    :return: A set of paths for all files and directories.
    """
    snapshot = set()
    for root, dirs, files in os.walk(base_dir):
        for name in dirs + files:
            snapshot.add(os.path.join(root, name))
    return snapshot


if __name__ == "__main__":

    # Enable CUDA device-side assertions
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Enable CUDA debugging

    # Start the timer for training
    training_start_time = time.time()

    # Define base directory for monitoring
    base_directory = "."

    # Create the parent output folder
    parent_folder = create_parent_output_folder(base_dir=base_directory)

    # Track validated directories
    validated_directories = set()

    # Paths to directories
    # Define paths for new directories inside the parent folder
    output_dir = os.path.join(parent_folder, "predictions")
    predictions_txt_dir = os.path.join(parent_folder, "prediction_results")
    save_dir = os.path.join(parent_folder, "saved_models")
    chunk_output_dir = os.path.join(parent_folder, "output_chunks")
    hyperparameter_output_dir = os.path.join(
        parent_folder, "hyperparameter_logs")

    # Ensure directories exist before validation
    ensure_directory_exists(output_dir)
    ensure_directory_exists(predictions_txt_dir)
    ensure_directory_exists(save_dir)
    ensure_directory_exists(chunk_output_dir)
    ensure_directory_exists(hyperparameter_output_dir)

    # Create the generator for organizing items
    # generator = organize_generated_items(base_dir=base_directory, main_dir_name="Final_Train_Predict_Output")
    # next(generator)  # Initial snapshot before the script logic

    try:
        # yaml_path = "/home/himanshu/Innovis_Work/DL_WorkFlow/yolov8/yolov8_birdnest_detection/dataset_NoVal.yaml"

        # Validate paths (confirm they exist or create them)

        validate_path(output_dir, "directory", description="Output directory",
                      tracked_paths=validated_directories)
        validate_path(predictions_txt_dir, "directory",
                      description="Predictions TXT directory", tracked_paths=validated_directories)
        validate_path(save_dir, "directory", description="Model save directory",
                      tracked_paths=validated_directories)
        validate_path(chunk_output_dir, "directory",
                      description="Chunk output directory", tracked_paths=validated_directories)
        validate_path(hyperparameter_output_dir, "directory",
                      description="Hyperparameter output directory", tracked_paths=validated_directories)
        print("All directories are validated and tracked.")
    except Exception as e:
        print(f"Error during directory validation: {e}")

    # Cleanup: Remove duplicates for all tracked directories
    cleanup_duplicate_directories(
        base_directory, parent_folder, validated_directories)

    try:
        # Your additional script logic goes here
        print("Script execution starts here...")

        # Load YAML content
        data = read_yaml(data_yaml)
        base_path = data['path']
        train_images = data['train']
        val_images = data['val']

        # Validate training and validation datasets
        print("Validating training data...")
        validate_files(base_path, train_images,
                       train_images.replace('images', 'labels'))

        print("Validating validation data...")
        validate_files(base_path, val_images,
                       val_images.replace('images', 'labels'))

        print("Validation completed.")
        # Get the latest .pt file from the directory
        # latest_pt_file = get_latest_pt_file(SAVE_DIR)

        # Validate labels
        validate_labels(LABEL_DIR, num_classes=1)
        # Validate images
        validate_images(IMAGE_DIR)

        # Organize items generated by the main script logic
        # organize_generated_items(main_script_logic, base_dir=".")

        # Save hyperparameters dynamically from the script
        save_hyperparameters_from_script(hyperparameter_output_dir)

        # Split the dataset into smaller chunks (optional)
        split_dataset(IMAGE_DIR, LABEL_DIR, CHUNK_OUTPUT_DIR, chunk_size=150)

        # Select one chunk for training (e.g., chunk_0)
        selected_chunk = os.path.join(CHUNK_OUTPUT_DIR, "chunk_0")
        image_dir = os.path.join(selected_chunk, "images")
        label_dir = os.path.join(selected_chunk, "labels")

        # Load the last saved weights or use default YOLOv8 model
        weights_to_use = get_last_saved_weights(SAVE_DIR)

        if weights_to_use:
            print(f"Loading last saved weights: {weights_to_use}")
        else:
            print("No saved weights found. Using default YOLOv8 model.")
            weights_to_use = fetch_default_weights(DEFAULT_WEIGHTS_DIR)

        # Train the model (either from saved weights or default)
        model = train_yolov8_model(data_yaml, weights_to_use)

        # Save training time and start prediction timer
        training_end_time = time.time()  # End training timer
        prediction_start_time = time.time()  # Start prediction timer

        # Evaluate the model
        evaluate_model(model)

        # Save and display the model accuracy
        save_and_display_accuracy(model, predictions_txt_dir)

        # Predict and save only images with predictions
        predictions = predict_and_save_images(
            model, test_images_dir, output_dir)

        # Save prediction results in a text file
        save_prediction_results(predictions, predictions_txt_dir)

        # Assuming `model` is already loaded
        predict_and_process_unique_nests(
            model, test_images_dir, exif_file, output_file, confidence_threshold=0.6)

        # Save the model
        save_model(model, SAVE_DIR)

        # Log the time
        log_training_and_prediction_time(
            training_start_time, prediction_start_time, output_dir)

    except Exception as e:
        print(f"Error during training or execution: {e}\n{traceback.format_exc()}")
