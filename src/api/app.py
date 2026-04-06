"""
FastAPI Backend — Rice Leaf Disease Detection Multi-Agent System
Usage: uvicorn src.api.app:app --reload
"""
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from src.core.workflow import run_diagnosis

app = FastAPI(
    title="Rice Leaf Disease Detection API",
    description="Multi-Agent AI system for diagnosing rice leaf diseases using Vision Transformer + Gemini.",
    version="1.0.0"
)

# CORS settings so that a frontend (Streamlit, React, etc.) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiagnosisResponse(BaseModel):
    disease_label: str
    confidence: float
    morphology_analysis: str
    knowledge_info: str
    final_report: str

@app.get("/")
def root():
    return {"message": "Rice Leaf Disease Detection API is running. POST to /diagnose to analyze a leaf image."}

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(file: UploadFile = File(...)):
    """
    Upload a rice leaf image and receive a full multi-agent diagnostic report.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, and WebP images are supported.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")
    
    try:
        result = run_diagnosis(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {str(e)}")
    
    return DiagnosisResponse(
        disease_label=result["classification"]["disease_label"],
        confidence=result["classification"]["confidence"],
        morphology_analysis=result["morphology"],
        knowledge_info=result["knowledge"],
        final_report=result["final_report"]
    )
