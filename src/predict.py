import joblib
import yaml
import logging
from src.preprocessing import TextPreprocessor
from typing import Dict, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessor = TextPreprocessor(config_path)
        self.load_artifacts()
        
    def load_artifacts(self):
        try:
            self.model = joblib.load(self.config["model"]["path"])
            self.preprocessor.load_vectorizer(self.config["model"]["vectorizer_path"])
            logger.info("Model and Vectorizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise e

    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        cleaned_text = self.preprocessor.clean_text(text)
        vec_text = self.preprocessor.transform([cleaned_text])
        
        prediction = self.model.predict(vec_text)[0]
        # probability if model supports it
        try:
            prob = self.model.predict_proba(vec_text)[0]
            confidence = float(max(prob))
        except (AttributeError, Exception):
            confidence = 1.0 # Fallback for models like LinearSVC without proba
            
        label = "Fake" if prediction == 1 else "Real"
        
        return {
            "prediction": label,
            "confidence": confidence
        }

if __name__ == "__main__":
    predictor = Predictor()
    sample_text = "Breaking News: Scientists discover that chocolate is actually a vegetable."
    result = predictor.predict(sample_text)
    print(f"Text: {sample_text}")
    print(f"Result: {result}")
