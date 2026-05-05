import os
import csv
import json
import xml.etree.ElementTree as ET
import tensorflow as tf
from PIL import Image

def is_image_corrupted(image_path):
    """Check if the image is corrupted."""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify the image integrity
        return False  # Image is not corrupted
    except Exception:
        return True  # Image is corrupted

def is_json_corrupted(json_path):
    """Check if the JSON annotation file is corrupted."""
    try:
        with open(json_path, 'r') as f:
            json.load(f)  # Try to parse JSON file
        return False  # JSON is not corrupted
    except Exception:
        return True  # JSON is corrupted

def is_xml_corrupted(xml_path):
    """Check if the XML annotation file is corrupted."""
    try:
        tree = ET.parse(xml_path)  # Try to parse XML file
        return False  # XML is not corrupted
    except ET.ParseError:
        return True  # XML is corrupted

def is_txt_corrupted(txt_path):
    """Check if the TXT annotation file is corrupted."""
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()  # Try to read the file
        return False  # TXT is not corrupted
    except Exception:
        return True  # TXT is corrupted

def is_csv_corrupted(csv_path):
    """Check if the CSV annotation file is corrupted."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                pass  # Just iterate through the rows
        return False  # CSV is not corrupted
    except Exception:
        return True  # CSV is corrupted

def is_tfrecord_corrupted(tfrecord_path):
    """Check if the TFRecord annotation file is corrupted."""
    try:
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        for _ in raw_dataset:  # Try to iterate over the records
            pass
        return False  # TFRecord is not corrupted
    except Exception:
        return True  # TFRecord is corrupted

def delete_related_files(base_name, directory):
    """Delete the image and all its related annotation files."""
    for file in os.listdir(directory):
        if file.startswith(base_name):
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

def process_directory(input_dir, output_dir):
    """Process the images and annotations, and remove corrupted files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        base_name, ext = os.path.splitext(file)

        # Check if the file is an image
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            # Check if the image is corrupted
            if is_image_corrupted(file_path):
                print(f"Corrupted image detected: {file}")
                delete_related_files(base_name, input_dir)
                continue

            # Check if related annotation files are corrupted
            corrupted = False
            for annotation_ext in ['.json', '.xml', '.txt', '.csv', '.tfrecord']:
                annotation_file = os.path.join(input_dir, base_name + annotation_ext)
                if os.path.exists(annotation_file):
                    if annotation_ext == '.json' and is_json_corrupted(annotation_file):
                        corrupted = True
                    elif annotation_ext == '.xml' and is_xml_corrupted(annotation_file):
                        corrupted = True
                    elif annotation_ext == '.txt' and is_txt_corrupted(annotation_file):
                        corrupted = True
                    elif annotation_ext == '.csv' and is_csv_corrupted(annotation_file):
                        corrupted = True
                    elif annotation_ext == '.tfrecord' and is_tfrecord_corrupted(annotation_file):
                        corrupted = True
                    if corrupted:
                        print(f"Corrupted annotation file detected: {annotation_file}")
                        delete_related_files(base_name, input_dir)
                        break

            # If no corruption was detected, copy the valid image to the output directory
            if not corrupted:
                new_image_path = os.path.join(output_dir, file)
                os.rename(file_path, new_image_path)  # Move the file
                print(f"Saved valid image: {new_image_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process the images and annotations, and remove corrupted files.")
    parser.add_argument('--input_dir', required=True, help="Path to input directory")
    parser.add_argument('--output_dir', required=True, help="Path to output directory")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
