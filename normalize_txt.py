# import os

# def normalize_bbox(x_min, y_min, x_max, y_max, img_width, img_height):
#     """
#     Normalize bounding box coordinates to the range [0, 1].
#     This function handles negative values by preserving them.
#     """
#     x_min_normalized = x_min / img_width
#     y_min_normalized = y_min / img_height
#     x_max_normalized = x_max / img_width
#     y_max_normalized = y_max / img_height
    
#     return x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized

# def process_directory(input_dir, output_dir, image_width, image_height):
#     # Ensure output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Process each file in the input directory
#     for filename in os.listdir(input_dir):
#         if filename.endswith(".txt"):
#             input_file_path = os.path.join(input_dir, filename)
#             output_file_path = os.path.join(output_dir, filename)
            
#             with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
#                 for line in infile:
#                     # Assuming the format of the annotation is: class_id x_min y_min x_max y_max
#                     # Modify this parsing if the annotation format is different
#                     parts = line.strip().split()
#                     class_id = parts[0]  # Class ID, assuming the first entry is a class label
#                     x_min, y_min, x_max, y_max = map(float, parts[1:5])
                    
#                     # Normalize bounding box coordinates
#                     x_min_norm, y_min_norm, x_max_norm, y_max_norm = normalize_bbox(
#                         x_min, y_min, x_max, y_max, image_width, image_height
#                     )
                    
#                     # Write the normalized data to the output file
#                     outfile.write(f"{class_id} {x_min_norm} {y_min_norm} {x_max_norm} {y_max_norm}\n")

#     print("Annotation processing complete. Normalized files saved to:", output_dir)

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Normalize bounding box coordinates in TXT files.")
#     parser.add_argument('--input_dir', required=True, help="Path to input directory containing .txt files")
#     parser.add_argument('--output_dir', required=True, help="Path to output directory")
#     parser.add_argument('--image_width', type=int, default=640, help="Image width (default: 640)")
#     parser.add_argument('--image_height', type=int, default=640, help="Image height (default: 640)")
#     args = parser.parse_args()

#     process_directory(args.input_dir, args.output_dir, args.image_width, args.image_height)

import os


def normalize_bbox(
    x_min,
    y_min,
    x_max,
    y_max,
    img_width,
    img_height
):
    """
    Convert Pascal VOC format:
    x_min y_min x_max y_max

    into normalized YOLO format:
    x_center y_center width height
    """

    # Calculate center coordinates

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    # Calculate width and height

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # Normalize

    x_center_normalized = x_center / img_width
    y_center_normalized = y_center / img_height

    bbox_width_normalized = bbox_width / img_width
    bbox_height_normalized = bbox_height / img_height

    return (
        x_center_normalized,
        y_center_normalized,
        bbox_width_normalized,
        bbox_height_normalized
    )


def process_directory(
    input_dir,
    output_dir,
    image_width,
    image_height
):

    # Ensure output directory exists

    if not os.path.exists(output_dir):

        os.makedirs(output_dir)

    # Process each TXT file

    for filename in os.listdir(input_dir):

        if filename.endswith(".txt"):

            input_file_path = os.path.join(
                input_dir,
                filename
            )

            output_file_path = os.path.join(
                output_dir,
                filename
            )

            with open(input_file_path, 'r') as infile, \
                    open(output_file_path, 'w') as outfile:

                for line in infile:

                    # Expected input format:
                    # class_id x_min y_min x_max y_max

                    parts = line.strip().split()

                    # Skip malformed lines

                    if len(parts) < 5:

                        print(
                            f"Skipping malformed line "
                            f"in {filename}: {line.strip()}"
                        )

                        continue

                    class_id = parts[0]

                    try:

                        x_min, y_min, x_max, y_max = map(
                            float,
                            parts[1:5]
                        )

                    except ValueError:

                        print(
                            f"Skipping invalid numeric "
                            f"values in {filename}: "
                            f"{line.strip()}"
                        )

                        continue

                    # Convert + normalize

                    (
                        x_center_norm,
                        y_center_norm,
                        width_norm,
                        height_norm
                    ) = normalize_bbox(
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        image_width,
                        image_height
                    )

                    # Write YOLO formatted annotation

                    outfile.write(
                        f"{class_id} "
                        f"{x_center_norm:.6f} "
                        f"{y_center_norm:.6f} "
                        f"{width_norm:.6f} "
                        f"{height_norm:.6f}\n"
                    )

    print(
        "Annotation processing complete. "
        "Normalized YOLO files saved to:",
        output_dir
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convert Pascal VOC TXT annotations "
            "to normalized YOLO TXT format."
        )
    )

    parser.add_argument(
        '--input_dir',
        required=True,
        help=(
            "Path to input directory "
            "containing TXT files"
        )
    )

    parser.add_argument(
        '--output_dir',
        required=True,
        help="Path to output directory"
    )

    parser.add_argument(
        '--image_width',
        type=int,
        default=640,
        help="Image width (default: 640)"
    )

    parser.add_argument(
        '--image_height',
        type=int,
        default=640,
        help="Image height (default: 640)"
    )

    args = parser.parse_args()

    process_directory(
        args.input_dir,
        args.output_dir,
        args.image_width,
        args.image_height
    )