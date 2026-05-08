import os
import csv
import json
import shutil
import xml.etree.ElementTree as ET

import tensorflow as tf

from PIL import Image


def is_image_corrupted(image_path):
    """Check if the image is corrupted."""

    try:

        with Image.open(image_path) as img:
            img.verify()

        return False

    except Exception:
        return True


def is_json_corrupted(json_path):
    """Check if the JSON annotation file is corrupted."""

    try:

        with open(json_path, 'r') as f:
            json.load(f)

        return False

    except Exception:
        return True


def is_xml_corrupted(xml_path):
    """Check if the XML annotation file is corrupted."""

    try:

        ET.parse(xml_path)

        return False

    except Exception:
        return True


def is_txt_corrupted(txt_path):
    """Check if the TXT annotation file is corrupted."""

    try:

        with open(txt_path, 'r') as f:

            lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                # YOLO format:
                # class_id x_center y_center width height

                if len(parts) != 5:
                    return True

                class_id = int(parts[0])

                if class_id < 0:
                    return True

                coords = list(map(float, parts[1:]))

                if not all(0 <= x <= 1 for x in coords):
                    return True

        return False

    except Exception:
        return True


def is_csv_corrupted(csv_path):
    """Check if the CSV annotation file is corrupted."""

    try:

        with open(csv_path, 'r') as f:

            reader = csv.reader(f)

            for row in reader:
                pass

        return False

    except Exception:
        return True


def is_tfrecord_corrupted(tfrecord_path):
    """Check if the TFRecord annotation file is corrupted."""

    try:

        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

        for raw_record in raw_dataset.take(1):

            tf.train.Example.FromString(
                raw_record.numpy()
            )

        return False

    except Exception:
        return True


def delete_related_files(base_name, directory):
    """Delete the image and all its related annotation files."""

    for file in os.listdir(directory):

        name, _ = os.path.splitext(file)

        if name == base_name:

            file_path = os.path.join(directory, file)

            try:

                os.remove(file_path)

                print(f"Deleted: {file_path}")

            except Exception as e:

                print(
                    f"Failed to delete "
                    f"{file_path}: {e}"
                )


def process_directory(input_dir, output_dir):
    """
    Process the images and annotations,
    and remove corrupted files.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):

        file_path = os.path.join(input_dir, file)

        base_name, ext = os.path.splitext(file)

        # Process only image files

        if ext.lower() in [
            '.jpg',
            '.jpeg',
            '.png',
            '.bmp',
            '.gif'
        ]:

            # Check image corruption

            if is_image_corrupted(file_path):

                print(
                    f"Corrupted image detected: {file}"
                )

                delete_related_files(
                    base_name,
                    input_dir
                )

                continue

            corrupted = False

            # Check annotation corruption

            for annotation_ext in [
                '.json',
                '.xml',
                '.txt',
                '.csv',
                '.tfrecord'
            ]:

                annotation_file = os.path.join(
                    input_dir,
                    base_name + annotation_ext
                )

                if os.path.exists(annotation_file):

                    if (
                        annotation_ext == '.json'
                        and is_json_corrupted(annotation_file)
                    ):
                        corrupted = True

                    elif (
                        annotation_ext == '.xml'
                        and is_xml_corrupted(annotation_file)
                    ):
                        corrupted = True

                    elif (
                        annotation_ext == '.txt'
                        and is_txt_corrupted(annotation_file)
                    ):
                        corrupted = True

                    elif (
                        annotation_ext == '.csv'
                        and is_csv_corrupted(annotation_file)
                    ):
                        corrupted = True

                    elif (
                        annotation_ext == '.tfrecord'
                        and is_tfrecord_corrupted(annotation_file)
                    ):
                        corrupted = True

                    if corrupted:

                        print(
                            f"Corrupted annotation "
                            f"file detected: "
                            f"{annotation_file}"
                        )

                        delete_related_files(
                            base_name,
                            input_dir
                        )

                        break

            # Copy valid image + annotations

            if not corrupted:

                try:

                    # Copy image

                    new_image_path = os.path.join(
                        output_dir,
                        file
                    )

                    shutil.copy2(
                        file_path,
                        new_image_path
                    )

                    print(
                        f"Saved valid image: "
                        f"{new_image_path}"
                    )

                    # Copy annotation files

                    for annotation_ext in [
                        '.json',
                        '.xml',
                        '.txt',
                        '.csv',
                        '.tfrecord'
                    ]:

                        annotation_file = os.path.join(
                            input_dir,
                            base_name + annotation_ext
                        )

                        if os.path.exists(annotation_file):

                            destination_annotation = os.path.join(
                                output_dir,
                                os.path.basename(annotation_file)
                            )

                            shutil.copy2(
                                annotation_file,
                                destination_annotation
                            )

                            print(
                                f"Copied annotation: "
                                f"{destination_annotation}"
                            )

                except Exception as e:

                    print(
                        f"Error while copying "
                        f"{file}: {e}"
                    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Process images and annotations, "
            "remove corrupted files, "
            "and copy valid files."
        )
    )

    parser.add_argument(
        '--input_dir',
        required=True,
        help="Path to input directory"
    )

    parser.add_argument(
        '--output_dir',
        required=True,
        help="Path to output directory"
    )

    args = parser.parse_args()

    process_directory(
        args.input_dir,
        args.output_dir
    )