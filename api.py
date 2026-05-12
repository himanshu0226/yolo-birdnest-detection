# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from typing import List
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import io

# app = FastAPI(
#     title="Detection API",
#     description="API for detecting bird nests using a trained YOLOv8 model.",
#     version="1.0.0"
# )

# # Load the model at startup
# MODEL_PATH = "yolov8n.pt"  # As requested, using the default YOLOv8n model for now
# try:
#     model = YOLO(MODEL_PATH)
#     print(f"Successfully loaded model from {MODEL_PATH}")
# except Exception as e:
#     print(f"Failed to load model: {e}")
#     model = None

# class BoundingBox(BaseModel):
#     x_min: float
#     y_min: float
#     x_max: float
#     y_max: float
#     confidence: float
#     class_id: int
#     class_name: str

# class DetectionResponse(BaseModel):
#     filename: str
#     detections: List[BoundingBox]
#     message: str

# @app.get("/health")
# def health_check():
#     """Check API and Model health status."""
#     if model is None:
#         return {"status": "unhealthy", "reason": "Model failed to load."}
#     return {"status": "healthy"}

# @app.post("/predict", response_model=DetectionResponse)
# async def predict_image(file: UploadFile = File(...)):
#     """
#     Predict bird nests in an uploaded image.
#     Accepts JPG/PNG image files.
#     """
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model is not loaded.")

#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

#     try:
#         # Read the file bytes
#         contents = await file.read()
        
#         # Convert bytes to numpy array then to OpenCV format
#         nparr = np.frombuffer(contents, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         if img is None:
#             raise HTTPException(status_code=400, detail="Could not decode the image.")

#         # Run model inference
#         # conf=0.25 is default, but we can set it to a desired threshold
#         results = model.predict(source=img, conf=0.50)

#         # Parse results
#         detections = []
#         if results and len(results) > 0:
#             result = results[0]
#             if result.boxes:
#                 boxes = result.boxes.xyxy.cpu().numpy()
#                 confidences = result.boxes.conf.cpu().numpy()
#                 classes = result.boxes.cls.cpu().numpy()

#                 for i, box in enumerate(boxes):
#                     x1, y1, x2, y2 = box
#                     class_id = int(classes[i])
#                     class_name = result.names[class_id] if result.names else str(class_id)
#                     detections.append(BoundingBox(
#                         x_min=float(x1),
#                         y_min=float(y1),
#                         x_max=float(x2),
#                         y_max=float(y2),
#                         confidence=float(confidences[i]),
#                         class_id=class_id,
#                         class_name=class_name
#                     ))

#         return DetectionResponse(
#             filename=file.filename,
#             detections=detections,
#             message=f"Successfully processed image. Found {len(detections)} detection(s)."
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from fastapi import HTTPException

from pydantic import BaseModel
from typing import List

import cv2
import numpy as np

from ultralytics import YOLO


app = FastAPI(
    title="Detection API",
    description=(
        "API for detecting bird nests "
        "using a trained YOLOv8 model."
    ),
    version="1.0.0"
)

# =========================================================
# Load model at startup
# =========================================================

MODEL_PATH = "yolov8n.pt"

try:

    model = YOLO(MODEL_PATH)

    print(
        f"Successfully loaded model "
        f"from {MODEL_PATH}"
    )

except Exception as e:

    print(f"Failed to load model: {e}")

    model = None


# =========================================================
# Response Models
# =========================================================

class BoundingBox(BaseModel):

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    confidence: float

    class_id: int
    class_name: str


class DetectionResponse(BaseModel):

    filename: str
    detections: List[BoundingBox]
    message: str


# =========================================================
# Health Endpoint
# =========================================================

@app.get("/health")
def health_check():

    """
    Check API and model health status.
    """

    if model is None:

        return {
            "status": "unhealthy",
            "reason": "Model failed to load."
        }

    return {
        "status": "healthy"
    }


# =========================================================
# Prediction Endpoint
# =========================================================

@app.post(
    "/predict",
    response_model=DetectionResponse
)
async def predict_image(
    file: UploadFile = File(...)
):

    """
    Predict bird nests in an uploaded image.

    Accepts JPG/PNG image files.
    """

    # -----------------------------------------------------
    # Validate model
    # -----------------------------------------------------

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Model is not loaded."
        )

    # -----------------------------------------------------
    # Validate content type safely
    # -----------------------------------------------------

    if (
        file.content_type is None or
        not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. "
                "Please upload an image."
            )
        )

    try:

        # -------------------------------------------------
        # Read uploaded file
        # -------------------------------------------------

        contents = await file.read()

        # Validate empty upload

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # -------------------------------------------------
        # Decode image
        # -------------------------------------------------

        nparr = np.frombuffer(
            contents,
            np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img is None:

            raise HTTPException(
                status_code=400,
                detail="Could not decode the image."
            )

        # -------------------------------------------------
        # Run inference
        # -------------------------------------------------

        results = model.predict(
            source=img,
            conf=0.50
        )

        # -------------------------------------------------
        # Parse detections
        # -------------------------------------------------

        detections = []

        if results and len(results) > 0:

            result = results[0]

            if result.boxes:

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                classes = (
                    result.boxes.cls
                    .cpu()
                    .numpy()
                )

                for i, box in enumerate(boxes):

                    x1, y1, x2, y2 = box

                    class_id = int(classes[i])

                    class_name = (
                        result.names[class_id]
                        if result.names
                        else str(class_id)
                    )

                    detections.append(
                        BoundingBox(
                            x_min=float(x1),
                            y_min=float(y1),
                            x_max=float(x2),
                            y_max=float(y2),
                            confidence=float(
                                confidences[i]
                            ),
                            class_id=class_id,
                            class_name=class_name
                        )
                    )

        # -------------------------------------------------
        # Return response
        # -------------------------------------------------

        return DetectionResponse(
            filename=file.filename,
            detections=detections,
            message=(
                "Successfully processed image. "
                f"Found {len(detections)} "
                f"detection(s)."
            )
        )

    # -----------------------------------------------------
    # Preserve FastAPI HTTPExceptions
    # -----------------------------------------------------

    except HTTPException:

        raise

    # -----------------------------------------------------
    # Handle unexpected exceptions
    # -----------------------------------------------------

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )