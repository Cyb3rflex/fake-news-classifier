import pandas as pd
import yaml
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.preprocessor = TextPreprocessor(config_path)
        
    def train(self):
        # 1. Load and Validate Data
        loader = DataLoader()
        try:
            df = loader.load_data()
        except FileNotFoundError:
            logger.error("Please place Fake.csv and True.csv in the data/ directory before training.")
            return
            
        df = loader.validate_data(df)
        
        # 2. Preprocess Text
        logger.info("Cleaning text (this may take a while)...")
        df["cleaned_text"] = df["text"].apply(self.preprocessor.clean_text)
        
        # 3. Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            df["cleaned_text"], 
            df["label"], 
            test_size=self.config["data"]["test_size"], 
            random_state=self.config["data"]["random_state"]
        )
        
        # 4. Vectorization
        X_train_vec = self.preprocessor.fit_transform(X_train)
        X_test_vec = self.preprocessor.transform(X_test)
        
        # 5. Initialize Model
        model_type = self.config["model"]["type"]
        if model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "naive_bayes":
            model = MultinomialNB()
        elif model_type == "svm":
            model = LinearSVC()
        else:
            logger.warning(f"Unknown model type {model_type}, defaulting to Logistic Regression.")
            model = LogisticRegression()
            
        # 6. Fit Model
        logger.info(f"Training {model_type} model...")
        model.fit(X_train_vec, y_train)
        
        # 7. Save Artifacts
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(model, self.config["model"]["path"])
        self.preprocessor.save_vectorizer(self.config["model"]["vectorizer_path"])
        
        logger.info("Training complete.")
        return model, X_test_vec, y_test

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
