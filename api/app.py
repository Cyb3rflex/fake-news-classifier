from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.predict import Predictor
import uvicorn
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Classifier API", description="Production-ready API for detecting fake news.")

# Simplified CORS for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS Middleware configured with allow_origins=['*']")

# Global predictor instance
predictor = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = Predictor()
        logger.info("API initialized and model loaded.")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.get("/")
def home():
    return {"message": "Fake News Classifier API is running. Go to /docs for Swagger UI."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    try:
        result = predictor.predict(request.text)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"])
