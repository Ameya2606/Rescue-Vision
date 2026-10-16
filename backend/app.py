import os
import sys
import io
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# Add parent directory to path so rescuevision can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rescuevision import RescueVisionModel, RescuePrediction, PRIORITY_DEFINITIONS

app = FastAPI(
    title="RescueVision AI Disaster Assessment API",
    description="High-performance REST API for Hurricane damage assessment and rescue operational prioritization.",
    version="1.0.0"
)

# Enable CORS for all frontends (Streamlit, React, mobile apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model_instance: Optional[RescueVisionModel] = None


def get_model() -> RescueVisionModel:
    global model_instance
    if model_instance is None:
        has_weights = os.path.exists("rescuevision_model.keras") or os.path.exists("../rescuevision_model.keras")
        # Initialize real model if weights exist, otherwise initialize mock mode gracefully
        model_instance = RescueVisionModel(mock_mode=not has_weights)
    return model_instance


@app.on_event("startup")
async def startup_event():
    get_model()


@app.get("/health", tags=["System"])
async def health_check():
    """Service health and model status."""
    model = get_model()
    return {
        "status": "healthy",
        "service": "RescueVision AI Backend",
        "mock_mode": model.mock_mode,
        "model_file_present": os.path.exists("rescuevision_model.keras"),
        "version": "1.0.0"
    }


@app.get("/api/v1/priorities", tags=["Metadata"])
async def get_priorities():
    """Retrieve disaster rescue priority level definitions (P1-P4)."""
    return {k: v.__dict__ for k, v in PRIORITY_DEFINITIONS.items()}


@app.post("/api/v1/predict", tags=["Inference"])
async def predict_single(file: UploadFile = File(...)):
    """
    Classify a satellite crop image as damaged or undamaged, and assign rescue priority.
    """
    model = get_model()
    try:
        content = await file.read()
        prediction = model.predict(content, source_id=file.filename)
        return prediction.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


@app.post("/api/v1/predict-batch", tags=["Inference"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch classification for multiple satellite crops.
    """
    model = get_model()
    results = []
    priority_counts = {"P1 - CRITICAL": 0, "P2 - HIGH": 0, "P3 - MEDIUM": 0, "P4 - LOW": 0}
    damage_count = 0

    try:
        for f in files:
            content = await f.read()
            pred = model.predict(content, source_id=f.filename)
            results.append(pred.to_dict())
            if pred.is_damaged:
                damage_count += 1
            if pred.priority in priority_counts:
                priority_counts[pred.priority] += 1

        total = len(results)
        return {
            "total_analyzed": total,
            "damaged_count": damage_count,
            "intact_count": total - damage_count,
            "damage_rate": f"{(damage_count / total * 100):.1f}%" if total > 0 else "0%",
            "priority_summary": priority_counts,
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch error: {str(e)}")


@app.post("/api/v1/assess-card", tags=["Visualization"])
async def generate_assessment_card(file: UploadFile = File(...)):
    """
    Generate an annotated visual assessment card image (PNG) for rescue operations.
    """
    model = get_model()
    try:
        content = await file.read()
        prediction = model.predict(content, source_id=file.filename)

        # Generate card buffer
        buf = io.BytesIO()
        model.visualize(content, prediction=prediction, save_path=buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Visualization error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
