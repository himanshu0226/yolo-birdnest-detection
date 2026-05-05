import os
import requests
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Orchestrator API",
    description="Single API that orchestrates the workflow between Frontend, Backend, and Detection APIs.",
    version="1.0.0"
)

# Base URLs for the other APIs (can be overridden with environment variables)
FRONTEND_API_URL = os.getenv("FRONTEND_API_URL", "http://localhost:3000")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
DETECTION_API_URL = os.getenv("DETECTION_API_URL", "http://localhost:8001")

@app.post("/tasks/detect/{image_id}")
def perform_detection_task(image_id: str):
    """
    Coordinates the workflow:
    1. Loads the image via the Frontend API.
    2. Looks up the image details in the database via the Backend API.
    3. Performs detection using the Detection API.
    """
    # Step 1: Frontend API loads the image
    frontend_url = f"{FRONTEND_API_URL}/api/images/{image_id}"
    try:
        frontend_response = requests.get(frontend_url)
        if frontend_response.status_code != 200:
            raise HTTPException(
                status_code=frontend_response.status_code, 
                detail=f"Frontend API failed to load image: {frontend_response.text}"
            )
        
        # We assume the Frontend API returns the raw image bytes
        image_bytes = frontend_response.content
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Frontend API: {e}")

    # Step 2: Backend API looks up in the database
    backend_url = f"{BACKEND_API_URL}/api/metadata/{image_id}"
    try:
        backend_response = requests.get(backend_url)
        if backend_response.status_code != 200:
            raise HTTPException(
                status_code=backend_response.status_code, 
                detail=f"Backend API failed to lookup data: {backend_response.text}"
            )
        
        # Assume the Backend API returns JSON metadata about the image
        db_metadata = backend_response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Backend API: {e}")

    # Step 3: Detection API performs detection
    detection_url = f"{DETECTION_API_URL}/predict"
    try:
        # Prepare the file payload matching what Detection API requires (UploadFile = File(...))
        files = {
            'file': (f"image_{image_id}.jpg", image_bytes, 'image/jpeg')
        }
        
        detection_response = requests.post(detection_url, files=files)
        if detection_response.status_code != 200:
            raise HTTPException(
                status_code=detection_response.status_code, 
                detail=f"Detection API failed: {detection_response.text}"
            )
        
        detection_results = detection_response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Detection API: {e}")

    # Final Step: Combine the results from Backend and Detection and return
    return {
        "status": "success",
        "image_id": image_id,
        "database_info": db_metadata,
        "detection_results": detection_results
    }
