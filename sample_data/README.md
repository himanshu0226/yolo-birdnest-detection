# Sample Dataset

This directory contains a random **10% sample** of the original full dataset (roughly a few hundred files) so that the repository size remains within GitHub's limits and you can easily clone and test the pipeline.

## Where is the Full Dataset?

Due to GitHub's size limitations, the complete, full-scale dataset (which contains thousands of high-resolution images) is intentionally ignored via `.gitignore` and is not tracked in this repository.

To run full-scale training:
1. Obtain the full dataset from your internal secure storage or cloud provider (e.g., AWS S3, Google Drive, or local storage).
2. Replace the contents of the `data/` and `birdnest_data/` folders on your machine with the full dataset.
3. The scripts (`yolov8_birdnest_detection.py` and `dataset.yaml`) are perfectly configured to use the data from these folders regardless of whether it's the 10% sample or the 100% full dataset.
