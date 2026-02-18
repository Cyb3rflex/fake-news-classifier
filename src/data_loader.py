import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
    def load_data(self) -> pd.DataFrame:
        """
        Loads raw Fake and True datasets, merges them, and adds labels.
        0 for Real, 1 for Fake.
        """
        fake_path = Path(self.config["data"]["raw_fake_path"])
        true_path = Path(self.config["data"]["raw_true_path"])
        
        if not fake_path.exists() or not true_path.exists():
            logger.error("Dataset files not found. Please ensure Fake.csv and True.csv are in the data/ directory.")
            raise FileNotFoundError("Dataset files missing.")
            
        logger.info("Loading datasets...")
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)
        
        df_fake["label"] = 1
        df_true["label"] = 0
        
        df = pd.concat([df_fake, df_true]).reset_index(drop=True)
        logger.info(f"Loaded {len(df)} rows.")
        return df

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes nulls, duplicates and shuffles the dataset.
        """
        initial_len = len(df)
        df = df.dropna(subset=["text"])
        df = df.drop_duplicates(subset=["text"])
        
        # We focus on the 'text' column for this classifier
        df = df[["text", "label"]]
        
        df = df.sample(frac=1, random_state=self.config["data"]["random_state"]).reset_index(drop=True)
        
        logger.info(f"Cleaned data: dropped {initial_len - len(df)} rows (nulls/duplicates).")
        return df

    def save_processed_data(self, df: pd.DataFrame):
        output_path = self.config["data"]["processed_data_path"]
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    # For testing data loader locally
    try:
        loader = DataLoader()
        data = loader.load_data()
        cleaned_data = loader.validate_data(data)
        loader.save_processed_data(cleaned_data)
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
