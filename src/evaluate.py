import joblib
import yaml
import logging
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
    def evaluate(self, X_test_vec, y_test):
        logger.info("Loading model for evaluation...")
        model = joblib.load(self.config["model"]["path"])
        
        y_pred = model.predict(X_test_vec)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Real", "Fake"])
        cm = confusion_matrix(y_test, y_pred)
        
        print("\n--- Model Evaluation Results ---")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(report)
        
        self.plot_confusion_matrix(cm)
        
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["Predicted Real", "Predicted Fake"],
                    yticklabels=["Actual Real", "Actual Fake"])
        plt.title("Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "confusion_matrix.png")
        logger.info(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
        plt.close()

if __name__ == "__main__":
    # This requires X_test_vec and y_test which are usually available after training
    # For a standalone evaluation, one would need to load the test set separately.
    logger.info("Evaluator module initialized. Use in conjunction with training or testing sets.")
